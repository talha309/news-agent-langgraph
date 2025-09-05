from fastapi import FastAPI, HTTPException, Request
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
import time

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="News Agent API",
    description="A FastAPI-based news agent using LangGraph and Tavily for real-time news summaries.",
    version="1.0.0"
)

# CORS middleware
allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    print(f"{request.method} {request.url.path} - {response.status_code} - {process_time:.3f}s")
    return response

# Initialize LLM with error handling
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    print("Warning: GOOGLE_API_KEY not found in environment variables")
    llm = None
else:
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=google_api_key
    )

# Initialize Tavily client with error handling
tavily_api_key = os.getenv("TAVILY_API_KEY")
if not tavily_api_key:
    print("Warning: TAVILY_API_KEY not found in environment variables")
    tavily_client = None
else:
    tavily_client = TavilyClient(api_key=tavily_api_key)

# Define tools
@tool
def runtime_searches(query: str, max_results: int = 10) -> str:
    """Run a real-time web/news search using Tavily for the latest news and information."""
    if not tavily_client:
        return "Error: Tavily API key not configured. Please set TAVILY_API_KEY environment variable."
    try:
        result = tavily_client.search(query=query, max_results=max_results, search_depth="advanced")
        items = result.get("results", []) if isinstance(result, dict) else []
        if not items:
            return "No results found for the query."
        formatted_results = []
        for item in items:
            title = item.get("title", "Untitled")
            url = item.get("url", "")
            snippet = item.get("content") or item.get("snippet", "")
            formatted_results.append(f"Title: {title}\nContent: {snippet}\nSource: {url}")
        return "\n\n".join(formatted_results)
    except Exception as e:
        return f"Search error: {str(e)}. Please try a different query or check API configuration."

@tool
def Summarize_text(result: str) -> str:
    """Summarize the given search results in a concise, structured format."""
    if not llm:
        return "Error: Google API key not configured. Please set GOOGLE_API_KEY environment variable."
    summary_prompt = (
        f"Analyze the following search results and organize them into clear news categories. "
        f"Extract the most important stories and group them by topic (e.g., Politics, Economy, Security, Natural Disasters, etc.). "
        f"For each story, identify key facts, numbers, dates, and sources. "
        f"Focus on the most significant developments and their impact. "
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

# Bind tools with LLM (only if LLM is available)
llm_with_tools = llm.bind_tools([runtime_searches, Summarize_text, write_text]) if llm else None
# Note: tool_calling_llm checks for llm_with_tools before invoking

# State definition
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# System prompt
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", """
    You are a professional news agent with access to tools: 'runtime_searches' for web searches,
    'Summarize_text' to analyze and categorize news, and 'write_text' to create professional news articles.
    When the user asks for the latest news (e.g., 'latest news in Pakistan' or similar queries with 'latest', 'today',
    'current', 'breaking', 'live', 'trending', 'updates', 'now'), follow this process:
    1. Call 'runtime_searches' with a broad query (e.g., 'latest news Pakistan')
    2. Call 'Summarize_text' to analyze and categorize the search results
    3. Call 'write_text' to transform the analysis into a professional news article format
    The final article should have clear section headings, detailed paragraphs with specific facts and numbers,
    source citations in brackets, and end with a 'Summary Snapshot' table.
    Do not ask for clarifications; fetch relevant news directly.
    Return the 'write_text' output as the final response without calling additional tools.
    For non-news queries (e.g., greetings, coding help, timeless facts), do not call tools.
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

# FastAPI query route
@app.get("/query")
async def query(q: str):
    """
    Handle user queries via GET request and return the news agent's response.
    
    Args:
        q (str): The user's query string.
    
    Returns:
        dict: Contains the news agent's response or an error message.
    """
    # Check if required services are available
    missing_services = []
    if not llm:
        missing_services.append("Google API key")
    if not tavily_client:
        missing_services.append("Tavily API key")
    
    if missing_services:
        raise HTTPException(
            status_code=503, 
            detail=f"Service unavailable: Missing {', '.join(missing_services)}. Please configure the required API keys."
        )
    
    user_text = q.strip()
    
    if not user_text:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    
    if len(user_text) > 500:
        raise HTTPException(status_code=400, detail="Query too long. Maximum length is 500 characters.")
    
    try:
        # Invoke graph with user query
        result = graph.invoke({"messages": [HumanMessage(content=user_text)]})
        
        # Extract AI messages
        msgs = result.get("messages", [])
        ai_msgs = [m for m in msgs if isinstance(m, AIMessage)]
        
        if ai_msgs:
            return {"response": ai_msgs[-1].content}
        elif msgs:
            return {"response": msgs[-1].content}
        else:
            raise HTTPException(status_code=500, detail="No valid response generated by the news agent.")
    
    except Exception as e:
        error_msg = str(e)
        if "timeout" in error_msg.lower():
            raise HTTPException(status_code=408, detail="Request timeout. Please try again with a simpler query.")
        elif "rate limit" in error_msg.lower():
            raise HTTPException(status_code=429, detail="Rate limit exceeded. Please wait a moment before trying again.")
        else:
            raise HTTPException(status_code=500, detail=f"Error processing request: {error_msg}")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "API is running"}