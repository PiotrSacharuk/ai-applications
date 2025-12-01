"""
Factory for creating provider instances based on configuration

Currently implemented providers:
- Storage: S3 (AWS)
- Database: MongoDB
- Embeddings: HuggingFace (local)
- LLM: Perplexity AI

Usage:
    storage = ProviderFactory.create_storage()
    database = ProviderFactory.create_database()
    embeddings = ProviderFactory.create_embeddings()
"""
import os
from .interfaces import StorageProvider, DatabaseProvider, EmbeddingsProvider
from .adapters.storage import S3Storage
from .adapters.database import MongoDatabase
from .adapters.embeddings import HuggingFaceEmbeddings


class ProviderFactory:
    """Factory for creating provider instances"""

    @staticmethod
    def create_storage() -> StorageProvider:
        """Create storage provider (currently only S3 implemented)"""
        provider = os.getenv("STORAGE_PROVIDER", "s3").lower()

        if provider == "s3":
            return S3Storage()
        else:
            raise ValueError(f"Storage provider '{provider}' not implemented. Available: s3")

    @staticmethod
    def create_database() -> DatabaseProvider:
        """Create database provider (currently only MongoDB implemented)"""
        provider = os.getenv("DATABASE_PROVIDER", "mongodb").lower()

        if provider == "mongodb":
            return MongoDatabase()
        else:
            raise ValueError(f"Database provider '{provider}' not implemented. Available: mongodb")

    @staticmethod
    def create_embeddings() -> EmbeddingsProvider:
        """Create embeddings provider (currently only HuggingFace implemented)"""
        provider = os.getenv("EMBEDDINGS_PROVIDER", "huggingface").lower()

        if provider == "huggingface":
            model_name = os.getenv("HUGGINGFACE_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
            device = os.getenv("EMBEDDING_DEVICE", "cpu")
            return HuggingFaceEmbeddings(model_name=model_name, device=device)
        else:
            raise ValueError(f"Embeddings provider '{provider}' not implemented. Available: huggingface")
