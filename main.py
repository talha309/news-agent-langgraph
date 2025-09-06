import chainlit as cl
from fastapi import HTTPException
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
import logging

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
def runtime_searches(query: str, max_results: int = 10) -> str:
    """
    This tool searches the internet using Tavily and gives a simple summary.
    Arguments:
        query: What do you want to search? (e.g., "latest news")
        max_results: How many results to show? Default is 10.
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
                    search_depth="advanced",
                    include_domains=["reuters.com", "bbc.com", "cnn.com", "ap.org", "aljazeera.com", "dawn.com", "tribune.com.pk"]
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

# Chainlit message handler
@cl.on_message
async def on_message(message: cl.Message):
    # Initialize or retrieve chat history
    history = cl.user_session.get("history") or []
    history.append({"role": "user", "content": message.content})
    
    # Convert Chainlit history to LangGraph messages
    langgraph_messages = []
    for msg in history:
        if msg["role"] == "user":
            langgraph_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            langgraph_messages.append(AIMessage(content=msg["content"]))

    # Create Chainlit message for streaming
    msg = cl.Message(content="")
    await msg.send()

    try:
        # Check for missing services
        missing_services = []
        if not llm:
            missing_services.append("Google API key")
        if not tavily_client:
            missing_services.append("Tavily API key")
        
        if missing_services:
            error_msg = f"Service unavailable: Missing {', '.join(missing_services)}. Please configure the required API keys."
            await msg.stream_token(error_msg)
            history.append({"role": "assistant", "content": error_msg})
            cl.user_session.set("history", history)
            await msg.update()
            return

        # Validate query
        user_text = message.content.strip()
        if not user_text:
            error_msg = "Query cannot be empty."
            await msg.stream_token(error_msg)
            history.append({"role": "assistant", "content": error_msg})
            cl.user_session.set("history", history)
            await msg.update()
            return
        
        if len(user_text) > 500:
            error_msg = "Query too long. Maximum length is 500 characters."
            await msg.stream_token(error_msg)
            history.append({"role": "assistant", "content": error_msg})
            cl.user_session.set("history", history)
            await msg.update()
            return

        # Stream LangGraph execution
        async for event in graph.astream({"messages": langgraph_messages}):
            for msg in event.get("messages", []):
                if isinstance(msg, AIMessage) and msg.content:
                    await msg.stream_token(msg.content)
                elif isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
                    # Handle tool call updates (optional, for debugging or intermediate feedback)
                    for tool_call in msg.tool_calls:
                        await msg.stream_token(f"Calling tool: {tool_call['name']}\n")
        
        # Get final result
        result = await graph.ainvoke({"messages": langgraph_messages})
        final_msgs = result.get("messages", [])
        ai_msgs = [m for m in final_msgs if isinstance(m, AIMessage)]
        
        if ai_msgs and ai_msgs[-1].content:
            final_response = ai_msgs[-1].content
            await msg.stream_token(final_response)
            history.append({"role": "assistant", "content": final_response})
        else:
            error_msg = "No valid response generated by the news agent."
            await msg.stream_token(error_msg)
            history.append({"role": "assistant", "content": error_msg})

    except Exception as e:
        error_msg = str(e)
        if "timeout" in error_msg.lower():
            error_msg = "Request timeout. Please try again with a simpler query."
        elif "rate limit" in error_msg.lower():
            error_msg = "Rate limit exceeded. Please wait a moment before trying again."
        else:
            error_msg = f"Error processing request: {error_msg}"
        await msg.stream_token(error_msg)
        history.append({"role": "assistant", "content": error_msg})

    # Update session history
    cl.user_session.set("history", history)
    await msg.update()

# Chainlit startup handler (optional, for initialization)
@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("history", [])
    await cl.Message(content="Welcome to the News Agent! Ask for the latest news or any other question.").send()