"""
Document Chat Module - Initialization

Modular RAG-based document Q&A system components:
- config: Environment configuration
- database: MongoDB conversation history
- storage: AWS S3 document storage
- models: AI models (embeddings, LLM, RAG pipeline)
- api: FastAPI endpoints
- main: Application entry point
"""
from .main import app

__all__ = ['app']
