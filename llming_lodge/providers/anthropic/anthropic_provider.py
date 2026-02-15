"""Anthropic provider implementation."""
import os
import logging
from typing import List, Optional

from llming_lodge.providers import BaseProvider, register_provider
from llming_lodge.llm_base_client import LlmClient
from .anthropic_models import ANTHROPIC_MODELS, LLMInfo
from .anthropic_client import AnthropicClient

logger = logging.getLogger(__name__)


@register_provider("anthropic")
class AnthropicProvider(BaseProvider):
    """Anthropic provider implementation."""

    def __init__(self):
        """Initialize Anthropic provider."""
        super().__init__("anthropic", "Anthropic")
        self._api_key = os.environ.get('ANTHROPIC_API_KEY')

    @property
    def is_available(self) -> bool:
        """Check if provider is available (has valid API key)."""
        return self._api_key is not None

    def get_models(self) -> List[LLMInfo]:
        """Get list of available Anthropic models."""
        return ANTHROPIC_MODELS

    def create_client(
        self,
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        streaming: bool = False,
        base_url: Optional[str] = None,
        toolboxes: Optional[List] = None,
        **kwargs
    ) -> LlmClient:
        """Create an Anthropic chat model client.

        Args:
            model: Model name to use
            temperature: Temperature for responses
            max_tokens: Maximum tokens to generate
            streaming: Whether to stream responses
            base_url: Optional base URL for the API (not used by Anthropic)
            toolboxes: Optional list of toolboxes for tool support
            **kwargs: Additional arguments passed to client

        Returns:
            Configured Anthropic client instance

        Raises:
            ValueError: If ANTHROPIC_API_KEY environment variable is not set
        """
        if not self.is_available:
            raise ValueError("ANTHROPIC_API_KEY environment variable is not set")

        return AnthropicClient(
            api_key=self._api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=streaming,
            toolboxes=toolboxes or []
        )
