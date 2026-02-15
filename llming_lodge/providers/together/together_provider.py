"""Together provider implementation."""
import os
from typing import List, Optional

from llming_lodge.providers import BaseProvider, register_provider
from llming_lodge.llm_base_client import LlmClient
from .deepseek import TOGETHER_DEEPSEEK_MODELS
from ..llm_provider_models import LLMInfo
from llming_lodge.providers.openai_compat_client import OpenAICompatibleClient


@register_provider("together")
class TogetherProvider(BaseProvider):
    """Together provider implementation."""

    DEFAULT_BASE_URL = "https://api.together.xyz/v1"

    def __init__(self):
        """Initialize Together provider."""
        super().__init__("together", "Together")
        self._api_key = os.environ.get('TOGETHER_API_KEY')

    @property
    def is_available(self) -> bool:
        """Check if provider is available (has valid API key)."""
        return self._api_key is not None

    def get_models(self) -> List[LLMInfo]:
        """Get list of available Together-hosted models."""
        return [
            *TOGETHER_DEEPSEEK_MODELS,
        ]

    def create_client(
        self,
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        streaming: bool = False,
        base_url: Optional[str] = None,
        **kwargs
    ) -> LlmClient:
        """Create a Together chat model client.

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
            ValueError: If TOGETHER_API_KEY environment variable is not set
        """
        if not self.is_available:
            raise ValueError("TOGETHER_API_KEY environment variable is not set")

        return OpenAICompatibleClient(
            api_key=self._api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=streaming,
            base_url=base_url or self.DEFAULT_BASE_URL
        )
