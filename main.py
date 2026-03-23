"""
AgriLink RAG Chatbot Server

FastAPI application that provides AI-powered chat support
using Retrieval Augmented Generation (RAG) with Google Gemini.
"""

import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.chat import router as chat_router
from rag.vector_store import initialize_vector_store


# Load environment variables
load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the RAG pipeline on startup."""
    print("🚀 Starting AgriLink Chatbot Server...")
    print("⏳ Initializing RAG pipeline...")

    try:
        initialize_vector_store()
        print("✅ RAG pipeline ready!")
    except Exception as e:
        print(f"❌ Failed to initialize RAG pipeline: {e}")
        print("⚠️ Server will start but chat may not work properly.")

    yield

    print("👋 Shutting down AgriLink Chatbot Server...")


app = FastAPI(
    title="AgriLink Chatbot API",
    description="RAG-powered chatbot for the AgriLink platform",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS
origins = [
    "http://localhost:3000",
    "https://www.aravinthkumar.me",
    "https://aravinthkumar.me",
    # Add any other production domains here
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(chat_router)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "agrilink-chatbot",
        "version": "1.0.0",
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "AgriLink Chatbot API",
        "docs": "/docs",
        "health": "/health",
    }


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,
    )
