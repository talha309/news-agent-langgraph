from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage, SystemMessage
from langgraph.graph.message import add_messages
from langgraph.graph import START, END, StateGraph
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
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

# Initialize API keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Check for missing API keys
if not GOOGLE_API_KEY:
    logger.error("GOOGLE_API_KEY/GEMINI_API_KEY not found in environment variables")
if not TAVILY_API_KEY:
    logger.error("TAVILY_API_KEY not found in environment variables")

# Initialize LLM
llm = None
if GOOGLE_API_KEY:
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=GOOGLE_API_KEY
    )

# Initialize Tavily client
tavily_client = None
if TAVILY_API_KEY:
    tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

# Define tools
@tool
def runtime_searches(query: str, min_results:int = 1, max_results: int = 5) -> str:
    """
    This tool searches the internet using Tavily and gives a simple summary.
    Arguments:
        query: What do you want to search? (e.g., "latest news")
        max_results: How many results to show? Default is max_results.
    Returns:
        A string with titles, links, and summaries of top results.
    """
    logger.info(f"Searching for: {query}")
    if not tavily_client:
        logger.error("Tavily client not initialized. Please set TAVILY_API_KEY environment variable.")
        return "Error: Tavily API key not configured. Please set TAVILY_API_KEY environment variable."
    
    try:
        search_queries = [query]
        if "latest news" in query.lower():
            search_queries.append(query.replace("latest news", "breaking news"))
            search_queries.append(query.replace("latest news", "current events"))
        
        for search_query in search_queries:
            try:
                result = tavily_client.search(
                    query=search_query, 
                    max_results=max_results, 
                    search_depth="basic",
                    include_domains=["reuters.com", "bbc.com", "cnn.com", "ap.org", "dawn.com", "tribune.com.pk"]
                )
                items = result.get("results", []) if isinstance(result, dict) else []
                
                if items:
                    logger.info(f"Found {len(items)} results for query: {search_query}")
                    formatted_results = []
                    for item in items:
                        title = item.get("title", "Untitled")
                        url = item.get("url", "")
                        snippet = item.get("content") or item.get("snippet", "")
                        if snippet:
                            formatted_results.append(f"Title: {title}\nContent: {snippet}\nSource: {url}")
                    
                    if formatted_results:
                        return "\n\n".join(formatted_results)
                        
            except Exception as e:
                logger.error(f"Search failed for '{search_query}': {e}")
                continue
        
        return "No relevant news results found. Please try a different query or check if the topic is currently in the news."
        
    except Exception as e:
        logger.error(f"Search error: {str(e)}. Please try a different query or check API configuration.")
        raise e

@tool
def Summarize_text(result: str) -> str:
    """Summarize the given search results in a concise, structured format."""
    if not llm:
        return "Error: Google API key not configured. Please set GOOGLE_API_KEY environment variable."
    
    if "No relevant news results found" in result or "Search error" in result:
        return f"Limited search results available: {result}"
    
    summary_prompt = (
        f"Analyze the following search results and organize them into clear news categories. "
        f"Extract the most important stories and group them by topic (e.g., Politics, Economy, Security, Natural Disasters, etc.). "
        f"For each story, identify key facts, numbers, dates, and sources. "
        f"Focus on the most significant developments and their impact. "
        f"If the results are limited, work with what's available and note any limitations. "
        f"Search results: {result}"
    )
    try:
        summary = llm.invoke(summary_prompt).content
        return summary
    except Exception as e:
        return f"Summarization error: {e}"

@tool
def write_text(result: str) -> str:
    """Rewrite the summary in a vivid, human-like style with detailed news descriptions."""
    if not llm:
        return "Error: Google API key not configured. Please set GOOGLE_API_KEY environment variable."
    
    if "Limited search results available" in result:
        write_prompt = (
            f"The following analysis indicates limited search results were available. "
            f"Create a professional news article based on the available information, "
            f"but acknowledge the limitations in data availability. "
            f"Structure it with clear headings and provide the best possible analysis with available data. "
            f"End with a note about the limited information available. "
            f"Analysis: {result}"
        )
    else:
        write_prompt = (
            f"Transform the following analyzed news into a professional news article format. "
            f"Structure it with clear section headings (like 'Flood Crisis Deepens in Punjab', 'China Withdraws from Major CPEC Project', etc.). "
            f"Each section should be 2-4 paragraphs with engaging, detailed descriptions. "
            f"Include specific numbers, dates, and impact details. "
            f"Add source citations in brackets like [AP News], [Reuters], etc. "
            f"End with a 'Summary Snapshot' table with Category and Summary columns. "
            f"Make it read like a professional news report with vivid details and clear organization. "
            f"Analyzed news: {result}"
        )
    
    try:
        written_text = llm.invoke(write_prompt).content
        return written_text
    except Exception as e:
        return f"Writing error: {e}"

# Bind tools with LLM
llm_with_tools = llm.bind_tools([runtime_searches, Summarize_text, write_text]) if llm else None

# State definition
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# System prompt
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

# Node function
def tool_calling_llm(state: MessagesState):
    if not llm_with_tools:
        return {"messages": [AIMessage(content="Error: Google API key not configured. Please set GOOGLE_API_KEY environment variable.")]}
    
    guidance = SystemMessage(content=SYSTEM_PROMPT)
    msgs = [guidance] + state["messages"]
    return {"messages": [llm_with_tools.invoke(msgs)]}

# Build graph
builder = StateGraph(MessagesState)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode([runtime_searches, Summarize_text, write_text]))
builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges("tool_calling_llm", tools_condition)
builder.add_edge("tools", "tool_calling_llm")
graph = builder.compile()

# FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# DB-backed history helpers
def load_history(db: Session, session_id: str):
    records = db.query(models.Message).filter(models.Message.session_id == session_id).order_by(models.Message.id.asc()).all()
    return [{"role": r.role, "content": r.content} for r in records]


def append_message(db: Session, session_id: str, role: str, content: str):
    msg = models.Message(session_id=session_id, role=role, content=content)
    db.add(msg)
    db.commit()
    db.refresh(msg)
    return msg

# Pydantic model for query input
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
    # Ensure all tables are created
    try:
        import models  # noqa: F401  # ensure models are imported for metadata
    except Exception:
        pass
    Base.metadata.create_all(bind=engine)


@app.get("/")
async def root():
    return {"message": "News Agent API is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "services": {
        "llm": llm is not None,
        "tavily": tavily_client is not None,
        "database": True
    }}

@app.get("/test")
async def test_endpoint():
    return {"message": "Backend is working!", "timestamp": time.time()}

@app.get("/query")
async def handle_query(query: str, session_id: str | None = None, db: Session = Depends(get_db)):
    # Generate a session ID if none provided
    if not session_id:
        session_id = str(uuid4())
    
    # Persist incoming user message
    append_message(db, session_id, "user", query)
    
    # Load full history from DB
    history = load_history(db, session_id)
    
    # Convert history to LangGraph messages
    langgraph_messages = []
    for msg in history:
        if msg["role"] == "user":
            langgraph_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            langgraph_messages.append(AIMessage(content=msg["content"]))

    try:
        # Check for missing services
        missing_services = []
        if not llm:
            missing_services.append("Google API key")
        if not tavily_client:
            missing_services.append("Tavily API key")
        
        if missing_services:
            error_msg = f"Service unavailable: Missing {', '.join(missing_services)}. Please configure the required API keys."
            append_message(db, session_id, "assistant", error_msg)
            raise HTTPException(status_code=503, detail=error_msg)

        # Validate query
        user_text = query.strip()
        if not user_text:
            error_msg = "Query cannot be empty."
            append_message(db, session_id, "assistant", error_msg)
            raise HTTPException(status_code=400, detail=error_msg)
        
        if len(user_text) > 500:
            error_msg = "Query too long. Maximum length is 500 characters."
            append_message(db, session_id, "assistant", error_msg)
            raise HTTPException(status_code=400, detail=error_msg)

        # Persist incoming query
        try:
            transaction = models.Transaction(query=user_text)
            db.add(transaction)
            db.commit()
            db.refresh(transaction)
        except Exception as db_err:
            # Do not fail the whole request due to logging failure; log and continue
            logger.error(f"Failed to persist query: {db_err}")

        # Execute LangGraph workflow with server-side timeout budget
        try:
            result = await asyncio.wait_for(
                graph.ainvoke({"messages": langgraph_messages}), timeout=55
            )
        except asyncio.TimeoutError:
            error_msg = "Request timeout. Please try again with a simpler query."
            append_message(db, session_id, "assistant", error_msg)
            raise HTTPException(status_code=408, detail=error_msg)
        final_msgs = result.get("messages", [])
        ai_msgs = [m for m in final_msgs if isinstance(m, AIMessage)]
        
        if ai_msgs and ai_msgs[-1].content:
            final_response = ai_msgs[-1].content
            append_message(db, session_id, "assistant", final_response)
        else:
            error_msg = "No valid response generated by the news agent."
            append_message(db, session_id, "assistant", error_msg)
            raise HTTPException(status_code=500, detail=error_msg)

        # Return response with session ID
        return {"response": final_response, "session_id": session_id}

    except Exception as e:
        error_msg = str(e)
        if "timeout" in error_msg.lower():
            error_msg = "Request timeout. Please try again with a simpler query."
        elif "rate limit" in error_msg.lower():
            error_msg = "Rate limit exceeded. Please wait a moment before trying again."
        else:
            error_msg = f"Error processing request: {error_msg}"
        append_message(db, session_id, "assistant", error_msg)
        # Preserve 408 if already mapped above
        if "timeout" in error_msg.lower():
            raise HTTPException(status_code=408, detail=error_msg)
        raise HTTPException(status_code=500, detail=error_msg)