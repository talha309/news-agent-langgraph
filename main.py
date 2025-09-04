from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pprint import pprint
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage, SystemMessage
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from tavily import TavilyClient
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Chatbot API",
    description="A FastAPI-based chatbot using LangGraph and Tavily for real-time web searches.",
    version="1.0.0"
)

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Initialize Tavily client
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# Define tool
@tool
def runtime_searches(query: str, max_results: int = 10) -> str:
    """Run a real-time web/news search using Tavily when the user asks to search.
    This tool is used to search the web for the latest news and information.
    """
    try:
        result = tavily_client.search(query=query, max_results=max_results)
    except Exception as e:
        return f"Search error: {e}"

    items = result.get("results", []) if isinstance(result, dict) else []
    if not items:
        return "No results found."

    lines = []
    for i, item in enumerate(items, start=1):
        title = item.get("title") or "Untitled"
        url = item.get("url") or ""
        snippet = item.get("content") or item.get("snippet") or ""
        snippet = (snippet[:240] + "…") if len(snippet) > 240 else snippet
        lines.append(f"{i}. {title}\n{url}\n{snippet}")

    return "\n\n".join(lines)

@tool 
def Summarize_text(result: str) -> str:
    """Summarize the given text.
    This tool is used to summarize the text of the runtime_searches tool.
    """
    summary_prompt = f"Summarize the following search results concisely, focusing on key facts, dates, and details: {result}"
    try:
        summary = llm.invoke(summary_prompt).content
        return summary
    except Exception as e:
        return f"Summarization error: {e}"

@tool
def write_text(result: str) -> str:
    """Write the given text.
    This tool is used to write the text of the Summarize_text tool.
    """
    write_prompt = f"Rewrite the following summary in a human-like, engaging style. Add beautiful details, elaborate on news aspects, make it vivid and informative, like a well-written article excerpt: {result}"
    try:
        written_text = llm.invoke(write_prompt).content
        return written_text
    except Exception as e:
        return f"Writing error: {e}"

# Bind tool with LLM
llm_with_tools = llm.bind_tools([runtime_searches, Summarize_text, write_text])

# State definition
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# Node function
def tool_calling_llm(state: MessagesState):
    guidance = SystemMessage(content=(
        "You are an assistant with access to tools: 'runtime_searches' for web searches, 'Summarize_text' to summarize search results, and 'write_text' to rewrite summaries beautifully. "
        "Always call 'runtime_searches' whenever the user asks about anything that requires up-to-date, real-time, "
        "or factual online information. This includes queries containing words like 'latest', 'today', "
        "'current', 'breaking', 'live', 'trending', 'updates', 'now', or similar. "
        "If you are unsure whether the answer requires real-time accuracy, default to calling the tool. "
        "For general conversation (e.g., greetings, opinions, explanations of concepts, coding help, "
        "or timeless facts), do not call the tool. "
        "Never guess about recent events or ongoing changes. Instead, use 'runtime_searches' to provide "
        "accurate, current information. "
        "For queries like 'what is about current matches of ICC?', do not ask for clarifications like time or teams; "
        "directly call 'runtime_searches' with a broad query like 'current ICC matches' to fetch relevant websites and info. "
        "After getting search results from 'runtime_searches', always call 'Summarize_text' on the output to get a concise summary. "
        "Then, call 'write_text' on the summary output to rewrite it in a beautiful, human-like style with more vivid news details. "
        "Finally, once you have the output from 'write_text', respond to the user with that as your final answer without calling more tools."
    ))
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
    Handle user queries via GET request and return the assistant's response.
    
    Args:
        q (str): The user's query string.
    
    Returns:
        dict: Contains the assistant's response or an error message.
    """
    user_text = q.strip()
    
    if not user_text:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    
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
            return {"response": "No response generated."}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

# Optional: Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "API is running"}