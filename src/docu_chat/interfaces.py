"""
Abstract interfaces for pluggable components

Currently implemented:
- StorageProvider: S3 (AWS)
- DatabaseProvider: MongoDB
- EmbeddingsProvider: HuggingFace (local)

Note: LLM uses LangChain's ChatOpenAI directly (Perplexity API compatible)
"""
from abc import ABC, abstractmethod
from typing import List, Tuple


class StorageProvider(ABC):
    """Abstract interface for document storage"""

    @abstractmethod
    def download_document(self, filename: str) -> str:
        """
        Download document from storage

        Args:
            filename: Name of the file to download

        Returns:
            Local file path where document was saved
        """
        pass

    @abstractmethod
    def upload_document(self, local_path: str) -> str:
        """
        Upload document to storage

        Args:
            local_path: Path to local file

        Returns:
            Storage path/URL of uploaded file
        """
        pass


class DatabaseProvider(ABC):
    """Abstract interface for conversation storage"""

    @abstractmethod
    def load_history(self, session_id: str) -> List[Tuple[str, str]]:
        """
        Load conversation history for session

        Args:
            session_id: Unique session identifier

        Returns:
            List of (user_message, bot_response) tuples
        """
        pass

    @abstractmethod
    def save_history(self, session_id: str, messages: List[str]) -> None:
        """
        Save conversation messages to session

        Args:
            session_id: Unique session identifier
            messages: List of messages to append (alternating user/bot)
        """
        pass


class EmbeddingsProvider(ABC):
    """Abstract interface for embeddings"""

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for documents

        Args:
            texts: List of text strings

        Returns:
            List of embedding vectors
        """
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for search query

        Args:
            text: Query string

        Returns:
            Embedding vector
        """
        pass