"""Mistral provider implementation."""
import os
from typing import List, Optional

from llming_lodge.providers import BaseProvider, register_provider
from llming_lodge.llm_base_client import LlmClient
from .mistral_models import MISTRAL_MODELS, LLMInfo
from llming_lodge.providers.openai_compat_client import OpenAICompatibleClient


@register_provider("mistral")
class MistralProvider(BaseProvider):
    """Mistral provider implementation."""

    DEFAULT_BASE_URL = "https://api.mistral.ai/v1"

    def __init__(self):
        """Initialize Mistral provider."""
        super().__init__("mistral", "Mistral")
        self._api_key = os.environ.get('MISTRAL_API_KEY')

    @property
    def is_available(self) -> bool:
        """Check if provider is available (has valid API key)."""
        return self._api_key is not None

    def get_models(self) -> List[LLMInfo]:
        """Get list of available Mistral models."""
        return MISTRAL_MODELS

    def create_client(
        self,
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        streaming: bool = False,
        base_url: Optional[str] = None,
        **kwargs
    ) -> LlmClient:
        """Create a Mistral chat model client.

        Args:
            model: Model name to use
            temperature: Temperature for responses
            max_tokens: Maximum tokens to generate
            streaming: Whether to stream responses
            base_url: Optional base URL for the API
            **kwargs: Additional arguments

        Returns:
            Configured OpenAICompatibleClient instance

        Raises:
            ValueError: If MISTRAL_API_KEY environment variable is not set
        """
        if not self.is_available:
            raise ValueError("MISTRAL_API_KEY environment variable is not set")

        return OpenAICompatibleClient(
            api_key=self._api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=streaming,
            base_url=base_url or self.DEFAULT_BASE_URL
        )
