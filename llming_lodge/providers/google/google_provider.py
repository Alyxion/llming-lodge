"""Google provider implementation."""
import os
from typing import List, Optional

from llming_lodge.providers import BaseProvider, register_provider
from llming_lodge.llm_base_client import LlmClient
from .google_models import GOOGLE_MODELS, LLMInfo
from .google_client import GoogleClient


@register_provider("google")
class GoogleProvider(BaseProvider):
    """Google provider implementation."""

    def __init__(self):
        """Initialize Google provider."""
        super().__init__("google", "Google")
        self._api_key = os.environ.get('GEMINI_KEY')

    @property
    def is_available(self) -> bool:
        """Check if provider is available (has valid API key)."""
        return self._api_key is not None

    def get_models(self) -> List[LLMInfo]:
        """Get list of available Google models."""
        return GOOGLE_MODELS

    def create_client(
        self,
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        streaming: bool = False,
        base_url: Optional[str] = None,
        **kwargs
    ) -> LlmClient:
        """Create a Google Gemini client.

        Args:
            model: Model name to use
            temperature: Temperature for responses
            max_tokens: Maximum tokens to generate
            streaming: Whether to stream responses
            base_url: Optional base URL for the API (not used by Google)
            **kwargs: Additional arguments

        Returns:
            Configured GoogleClient instance

        Raises:
            ValueError: If GEMINI_KEY environment variable is not set
        """
        if not self.is_available:
            raise ValueError("GEMINI_KEY environment variable is not set")

        return GoogleClient(
            api_key=self._api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=streaming
        )
