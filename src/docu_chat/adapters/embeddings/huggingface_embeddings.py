"""
HuggingFace Embeddings Implementation (local, free)
"""
from typing import List
from langchain_huggingface import HuggingFaceEmbeddings as HFEmbeddings
from ...interfaces import EmbeddingsProvider


class HuggingFaceEmbeddings(EmbeddingsProvider):
    """HuggingFace sentence-transformers (local, CPU/GPU)"""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu"):
        """
        Initialize HuggingFace embeddings

        Args:
            model_name: HuggingFace model name
            device: 'cpu' or 'cuda'
        """
        self.embeddings = HFEmbeddings(
            model_name=model_name,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True}
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for documents"""
        return self.embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for query"""
        return self.embeddings.embed_query(text)
