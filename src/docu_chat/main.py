"""
Document Chat Application - Main Entry Point

Modular RAG-based document Q&A system with:
- FastAPI REST API
- MongoDB conversation history
- AWS S3 document storage
- Perplexity AI for LLM
- HuggingFace for embeddings

Usage:
    poetry run uvicorn src.docu_chat.main:app --reload
"""
import uvicorn
from .api import app
from .config import APP_HOST, APP_PORT

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=APP_HOST,
        port=APP_PORT,
        reload=True
    )
