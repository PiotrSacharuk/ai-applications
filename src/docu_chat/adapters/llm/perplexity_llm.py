"""
Perplexity LLM Implementation
"""
from typing import List, Tuple, Dict, Any
from langchain_openai import ChatOpenAI
from ...config import OPENAI_API_BASE


class PerplexityLLM:
    """Perplexity AI LLM (uses OpenAI-compatible API)"""

    def __init__(self, model: str = "sonar-reasoning", api_key: str = None, temperature: float = 0.0):
        """
        Initialize Perplexity LLM

        Args:
            model: Perplexity model name (sonar, sonar-pro, sonar-reasoning)
            api_key: Perplexity API key
            temperature: Response randomness (0.0 = deterministic)
        """
        self.llm = ChatOpenAI(
            model_name=model,
            temperature=temperature,
            openai_api_key=api_key,
            openai_api_base=OPENAI_API_BASE
        )

    def generate_response(
        self,
        query: str,
        context: str,
        chat_history: List[Tuple[str, str]]
    ) -> Dict[str, Any]:
        """Generate response using Perplexity"""
        # Format chat history for prompt
        history_text = "\n".join([
            f"User: {user}\nAssistant: {assistant}"
            for user, assistant in chat_history
        ])

        # Create prompt with context and history
        prompt = f"""Context from document:
{context}

Previous conversation:
{history_text}

Current question: {query}

Please answer based on the context provided."""

        response = self.llm.invoke(prompt)

        return {
            "answer": response.content,
            "model": self.llm.model_name
        }
