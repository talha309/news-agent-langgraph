#!/usr/bin/env python3
"""
Startup script for the News Agent FastAPI backend
"""
import uvicorn
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

if __name__ == "__main__":
    # Check for required environment variables
    google_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    tavily_key = os.getenv("TAVILY_API_KEY")
    
    print("🚀 Starting News Agent Backend...")
    print(f"📍 Database: {os.getenv('DATABASE_URL', 'sqlite:///./app.db')}")
    print(f"🤖 Google API: {'✅ Configured' if google_key else '❌ Missing'}")
    print(f"🔍 Tavily API: {'✅ Configured' if tavily_key else '❌ Missing'}")
    
    if not google_key:
        print("⚠️  Warning: GOOGLE_API_KEY not found. News generation will be limited.")
    if not tavily_key:
        print("⚠️  Warning: TAVILY_API_KEY not found. Web search will be disabled.")
    
    print("\n🌐 Server starting at: http://localhost:8000")
    print("📖 API docs at: http://localhost:8000/docs")
    print("❤️  Health check: http://localhost:8000/health")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
