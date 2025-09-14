from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Annotated, List, Any
try:
    # Python 3.8+ / 3.11+ compatibility
    from typing import TypedDict
except Exception:
    from typing_extensions import TypedDict

from langchain_core.messages import AnyMessage, AIMessage, HumanMessage, SystemMessage
from langgraph.graph.message import add_messages
from langgraph.graph import START, END, StateGraph
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition

# Note: keep the exact LLM / client imports you use in your environment
from langchain_google_genai import ChatGoogleGenerativeAI
from tavily import TavilyClient
from dotenv import load_dotenv
import os
import asyncio
import time
import logging
from uuid import uuid4
from sqlalchemy.orm import Session
from database import SessionLocal, engine, Base
import models

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# App-level placeholders for services — initialize on startup
llm = None
llm_with_tools = None
tavily_client = None

# System prompt (kept compact here; can be moved to env var)
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", """
You are a professional news agent with access to three tools:
1. 'runtime_searches' - for web searches to get latest news
2. 'Summarize_text' - to analyze and categorize search results
3. 'write_text' - to create professional news articles

IMPORTANT: When users ask for news (containing words like 'latest', 'today', 'current', 'breaking', 'news', 'Pakistan', etc.), you MUST:
1. ALWAYS call 'runtime_searches' first with a relevant query
2. Then call 'Summarize_text' with the search results
3. Finally call 'write_text' with the summary to create the final article

Do NOT respond without calling these tools. Do NOT say you cannot create articles. Always use the tools to get real information.

For non-news queries (greetings, coding help, general questions), you may respond directly without tools.
""")

# --- Typed state definition ---
class MessagesState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]

# --- Tools ---
@tool
def runtime_searches(query: str, min_results: int = 1, max_results: int = 5) -> str:
    """Search using Tavily and return a concatenated summary of results.
    This function is intentionally synchronous to match the way tools are used in your graph.
    The content maybe relavent to the user query.
    """
    logger.info(f"runtime_searches called with: {query}")
    global tavily_client
    if tavily_client is None:
        logger.error("Tavily client not initialized.")
        return "Error: Tavily API key not configured. Please set TAVILY_API_KEY environment variable."

    try:
        search_queries = [query]
        if "latest news" in query.lower():
            search_queries += [query.replace("latest news", "breaking news"), query.replace("latest news", "current events")]

        for search_query in search_queries:
            try:
                result = tavily_client.search(
                    query=search_query,
                    max_results=max_results,
                    search_depth="basic",
                    include_domains=["reuters.com", "bbc.com", "cnn.com", "ap.org", "dawn.com", "tribune.com.pk"],
                )
                items = result.get("results", []) if isinstance(result, dict) else []

                if items:
                    formatted = []
                    for item in items:
                        title = item.get("title", "Untitled")
                        url = item.get("url", "")
                        snippet = item.get("content") or item.get("snippet", "")
                        formatted.append(f"Title: {title}\nContent: {snippet}\nSource: {url}")
                    return "\n\n".join(formatted)
            except Exception as e:
                logger.exception(f"Tavily search failed for query '{search_query}': {e}")
                continue

        return "No relevant news results found. Please try a different query or check if the topic is currently in the news."
    except Exception as e:
        logger.exception("runtime_searches unexpected error")
        return f"Search error: {e}"


@tool
def Summarize_text(result: str) -> str:
    """Summarize and categorize search results using the LLM."""
    global llm
    if llm is None:
        return "Error: Google API key not configured. Please set GOOGLE_API_KEY environment variable."

    if not result or "No relevant news results found" in result:
        return f"Limited search results available: {result}"

    summary_prompt = (
        "Analyze the following search results and organize them into clear news categories. "
        "Extract the most important stories and group them by topic. "
        f"Search results: {result}"
    )
    try:
        # Fall back safely if the llm object doesn't support 'invoke' directly
        if hasattr(llm, "invoke"):
            response = llm.invoke(summary_prompt)
            return getattr(response, "content", str(response))
        else:
            response = llm(summary_prompt)
            return str(response)
    except Exception as e:
        logger.exception("Summarize_text error")
        return f"Summarization error: {e}"


@tool
def write_text(result: str) -> str:
    """Rewrite analysis into a professional article."""
    global llm
    if llm is None:
        return "Error: Google API key not configured. Please set GOOGLE_API_KEY environment variable."

    if not result:
        return "No analysis provided to write from."

    if "Limited search results available" in result:
        write_prompt = (
            "The following analysis indicates limited search results were available. "
            "Create a professional news article based on the available information, "
            "and acknowledge limitations. Analysis: " + result
        )
    else:
        write_prompt = (
            "Transform the following analyzed news into a professional news article format. "
            "Structure it with clear section headings. "
            "Analyzed news: " + result
        )
    try:
        if hasattr(llm, "invoke"):
            response = llm.invoke(write_prompt)
            return getattr(response, "content", str(response))
        else:
            response = llm(write_prompt)
            return str(response)
    except Exception as e:
        logger.exception("write_text error")
        return f"Writing error: {e}"


# A small helper to safely bind tools if llm supports it
def bind_tools_safely(llm_obj):
    try:
        return llm_obj.bind_tools([runtime_searches, Summarize_text, write_text])
    except Exception:
        logger.warning("LLM does not support bind_tools; continuing without tool binding.")
        return None

# Build graph (unchanged structure but created after startup to ensure services exist)
builder = StateGraph(MessagesState)

# We'll add nodes after startup once llm/llm_with_tools exist

# FastAPI app
app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# DB helpers
def load_history(db: Session, session_id: str):
    records = db.query(models.Message).filter(models.Message.session_id == session_id).order_by(models.Message.id.asc()).all()
    return [{"role": r.role, "content": r.content} for r in records]


def append_message(db: Session, session_id: str, role: str, content: str):
    msg = models.Message(session_id=session_id, role=role, content=content)
    db.add(msg)
    db.commit()
    db.refresh(msg)
    return msg

# Pydantic model for query input (used by POST endpoint)
class QueryInput(BaseModel):
    query: str
    session_id: str | None = None


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.on_event("startup")
def on_startup():
    global llm, tavily_client, llm_with_tools, builder, graph

    # Create DB tables
    try:
        import models  # noqa: F401
    except Exception:
        pass
    Base.metadata.create_all(bind=engine)

    # Initialize API keys and clients at startup so load_dotenv has taken effect
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

    if not GOOGLE_API_KEY:
        logger.warning("GOOGLE_API_KEY/GEMINI_API_KEY not found in environment variables")
    if not TAVILY_API_KEY:
        logger.warning("TAVILY_API_KEY not found in environment variables")

    if GOOGLE_API_KEY:
        try:
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)
            llm_with_tools = bind_tools_safely(llm)
            logger.info("LLM initialized")
        except Exception:
            logger.exception("Failed to initialize LLM")
            llm = None

    if TAVILY_API_KEY:
        try:
            tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
            logger.info("Tavily client initialized")
        except Exception:
            logger.exception("Failed to initialize Tavily client")
            tavily_client = None

    # Build the LangGraph now that llm_with_tools may exist
    try:
        builder.add_node("tool_calling_llm", tool_calling_llm)
        builder.add_node("tools", ToolNode([runtime_searches, Summarize_text, write_text]))
        builder.add_edge(START, "tool_calling_llm")
        builder.add_conditional_edges("tool_calling_llm", tools_condition)
        builder.add_edge("tools", "tool_calling_llm")
        graph = builder.compile()
        logger.info("LangGraph built")
    except Exception:
        logger.exception("Failed to build LangGraph")


# tool_calling_llm kept outside builder to keep logic testable
def tool_calling_llm(state: MessagesState):
    global llm_with_tools, llm
    if llm_with_tools is None and llm is None:
        return {"messages": [AIMessage(content="Error: Google API key not configured. Please set GOOGLE_API_KEY environment variable.")]} 

    guidance = SystemMessage(content=SYSTEM_PROMPT)
    msgs = [guidance] + state["messages"]

    try:
        if llm_with_tools:
            return {"messages": [llm_with_tools.invoke(msgs)]}
        # If tool binding not available, call llm directly (best-effort)
        if hasattr(llm, "invoke"):
            return {"messages": [llm.invoke(msgs)]}
        else:
            # attempt a plain call
            return {"messages": [AIMessage(content=str(llm(msgs)))]}
    except Exception as e:
        logger.exception("tool_calling_llm failed")
        return {"messages": [AIMessage(content=f"LLM invocation error: {e}")]} 


@app.get("/")
async def root():
    return {"welcome": "news-agent platform"}


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "services": {
            "llm": llm is not None,
            "tavily": tavily_client is not None,
            "database": True,
        },
    }


@app.get("/test")
async def test_endpoint():
    return {"message": "Backend is working!", "timestamp": time.time()}


# Keep GET /query for quick testing but prefer POST /query for real usage
@app.get("/query")
async def handle_query_get(query: str, session_id: str | None = None, db: Session = Depends(get_db)):
    # delegate to the POST implementation
    body = QueryInput(query=query, session_id=session_id)
    return await handle_query_post(body, db)


@app.post("/query")
async def handle_query_post(payload: QueryInput, db: Session = Depends(get_db)):
    # Generate a session ID if none provided
    session_id = payload.session_id or str(uuid4())
    user_text = payload.query.strip()

    # Persist incoming user message
    append_message(db, session_id, "user", user_text)

    # Load full history from DB
    history = load_history(db, session_id)

    # Convert history to LangGraph messages
    langgraph_messages = []
    for msg in history:
        if msg["role"] == "user":
            langgraph_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            langgraph_messages.append(AIMessage(content=msg["content"]))

    # Check for missing services
    missing_services = []
    if llm is None:
        missing_services.append("Google API key / LLM")
    if tavily_client is None:
        missing_services.append("Tavily API key / client")

    if missing_services:
        error_msg = f"Service unavailable: Missing {', '.join(missing_services)}. Please configure the required API keys."
        append_message(db, session_id, "assistant", error_msg)
        raise HTTPException(status_code=503, detail=error_msg)

    if not user_text:
        error_msg = "Query cannot be empty."
        append_message(db, session_id, "assistant", error_msg)
        raise HTTPException(status_code=400, detail=error_msg)

    if len(user_text) > 500:
        error_msg = "Query too long. Maximum length is 500 characters."
        append_message(db, session_id, "assistant", error_msg)
        raise HTTPException(status_code=400, detail=error_msg)

    # Persist query as a transaction (best-effort)
    try:
        transaction = models.Transaction(query=user_text)
        db.add(transaction)
        db.commit()
        db.refresh(transaction)
    except Exception:
        logger.exception("Failed to persist transaction; continuing")

    # Execute LangGraph workflow with a timeout budget
    try:
        result = await asyncio.wait_for(graph.ainvoke({"messages": langgraph_messages}), timeout=55)
    except asyncio.TimeoutError:
        error_msg = "Request timeout. Please try again with a simpler query."
        append_message(db, session_id, "assistant", error_msg)
        raise HTTPException(status_code=408, detail=error_msg)
    except Exception as e:
        logger.exception("Graph invocation failed")
        error_msg = f"Error processing request: {e}"
        append_message(db, session_id, "assistant", error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

    final_msgs = result.get("messages", [])
    ai_msgs = [m for m in final_msgs if isinstance(m, AIMessage)]

    if ai_msgs and getattr(ai_msgs[-1], "content", None):
        final_response = ai_msgs[-1].content
        append_message(db, session_id, "assistant", final_response)
        return {"response": final_response, "session_id": session_id}

    error_msg = "No valid response generated by the news agent."
    append_message(db, session_id, "assistant", error_msg)
    raise HTTPException(status_code=500, detail=error_msg)
