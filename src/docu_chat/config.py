"""
Configuration module for docu_chat application
Loads environment variables and provides configuration constants

Currently implemented providers:
- Storage: S3 (AWS)
- Database: MongoDB
- Embeddings: HuggingFace (local)
- LLM: Perplexity AI
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Provider Selection
STORAGE_PROVIDER = os.getenv("STORAGE_PROVIDER", "s3")
DATABASE_PROVIDER = os.getenv("DATABASE_PROVIDER", "mongodb")
EMBEDDINGS_PROVIDER = os.getenv("EMBEDDINGS_PROVIDER", "huggingface")

# OpenAI/Perplexity API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")  # e.g., https://api.perplexity.ai
LLM_MODEL = os.getenv("LLM_MODEL", "sonar")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))

# AWS S3 Configuration
S3_KEY = os.getenv("AWS_ACCESS_KEY_ID")
S3_SECRET = os.getenv("AWS_SECRET_ACCESS_KEY")
S3_BUCKET = os.getenv("AWS_S3_BUCKET", "docchat")
S3_REGION = os.getenv("AWS_S3_REGION", "us-east-1")
S3_PATH = os.getenv("AWS_S3_PATH", "documents/")

# MongoDB Configuration
MONGO_URL = os.getenv("MONGODB_URL", "mongodb+srv://admin:admin@cluster0.jyupp.mongodb.net/?retryWrites=true&w=majority&ssl=true")
MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "chat_with_doc")
MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION", "chat_history")

# HuggingFace Embeddings Configuration
HUGGINGFACE_EMBEDDING_MODEL = os.getenv("HUGGINGFACE_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")

# Application Configuration
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", "8000"))
