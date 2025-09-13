# News Agent - Full Stack Application

A modern full-stack news agent application built with FastAPI backend and React frontend, featuring real-time news search, AI-powered summarization, and persistent chat history.

## 🏗️ Architecture

- **Backend**: FastAPI with LangGraph, SQLAlchemy, and SQLite
- **Frontend**: React with modern UI components
- **AI**: Google Gemini for text generation and Tavily for web search
- **Database**: SQLite with persistent message storage

## 🚀 Quick Start

### Prerequisites

- Python 3.8+ with `uv` or `pip`
- Node.js 16+ with npm
- API Keys: Google API Key and Tavily API Key

### 1. Backend Setup

```bash
cd backend_fastapi

# Install dependencies
uv pip install -e .

# Or with pip
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys:
# GOOGLE_API_KEY=your_google_api_key
# TAVILY_API_KEY=your_tavily_api_key
# DATABASE_URL=sqlite:///./app.db  # Optional, defaults to SQLite

# Start the backend
python start.py
```

The backend will be available at:
- API: http://localhost:8000
- Docs: http://localhost:8000/docs
- Health: http://localhost:8000/health

### 2. Frontend Setup

```bash
cd frontend_react/news-agent

# Install dependencies
npm install

# Start the development server
npm start
```

The frontend will be available at: http://localhost:3000

## 🔧 Configuration

### Environment Variables

Create a `.env` file in `backend_fastapi/`:

```env
# Required API Keys
GOOGLE_API_KEY=your_google_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here

# Optional
DATABASE_URL=sqlite:///./app.db
SYSTEM_PROMPT=Your custom system prompt
```

### API Keys Setup

1. **Google API Key**: Get from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. **Tavily API Key**: Get from [Tavily](https://tavily.com/)

## 📱 Features

### Backend Features
- ✅ Real-time web search with Tavily
- ✅ AI-powered news summarization with Google Gemini
- ✅ Persistent chat history in SQLite
- ✅ CORS-enabled for frontend integration
- ✅ Health check endpoints
- ✅ Error handling and logging

### Frontend Features
- ✅ Modern, responsive chat interface
- ✅ Real-time message display
- ✅ Formatted news content with tables and sections
- ✅ Loading states and error handling
- ✅ Mobile-friendly design
- ✅ Session-based chat history

## 🗄️ Database Schema

The application uses SQLite with two main tables:

### `transactions`
- `id`: Primary key
- `query`: User query text

### `messages`
- `id`: Primary key
- `session_id`: Chat session identifier
- `role`: 'user' or 'assistant'
- `content`: Message content
- `created_at`: Timestamp

## 🔄 API Endpoints

### GET `/`
- Root endpoint with basic info

### GET `/health`
- Health check with service status

### GET `/query?query={text}&session_id={id}`
- Main chat endpoint
- Parameters:
  - `query`: User message (required)
  - `session_id`: Chat session ID (optional, auto-generated)

## 🎨 Frontend Components

### `App.js`
- Main application component
- Handles chat state and API communication
- Includes `NewsContent` component for formatted display

### `NewsContent`
- Formats news articles with:
  - Section headers
  - Tables
  - Paragraphs
  - Proper styling

## 🚀 Deployment

### Backend Deployment

```bash
# Production build
uv pip install -e . --no-dev

# Run with production server
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Frontend Deployment

```bash
# Build for production
npm run build

# Serve static files
npm install -g serve
serve -s build -l 3000
```

## 🛠️ Development

### Backend Development

```bash
# Install development dependencies
uv pip install -e .[dev]

# Run with auto-reload
python start.py
```

### Frontend Development

```bash
# Start development server with hot reload
npm start
```

## 📊 Monitoring

- Backend logs are available in the console
- Health check endpoint: `/health`
- Database file: `backend_fastapi/app.db` (SQLite)

## 🔍 Troubleshooting

### Common Issues

1. **CORS Errors**: Ensure backend is running on port 8000
2. **API Key Errors**: Check `.env` file configuration
3. **Database Errors**: Ensure write permissions in backend directory
4. **Frontend Connection**: Verify proxy setting in `package.json`

### Logs

- Backend: Console output with structured logging
- Frontend: Browser developer tools console

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the API documentation at `/docs`
3. Check the health endpoint at `/health`
