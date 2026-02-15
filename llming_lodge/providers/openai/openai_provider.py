"""OpenAI provider implementation."""
import os
from typing import List, Optional

from llming_lodge.providers import BaseProvider, register_provider
from llming_lodge.llm_base_client import LlmClient
from .openai_models import OPENAI_MODELS, LLMInfo
from .openai_client import OpenAILlmClient
from llming_lodge.tools.llm_toolbox import LlmToolbox
from llming_lodge.providers.llm_provider_models import ReasoningEffort


@register_provider("openai")
class OpenAIProvider(BaseProvider):
    """OpenAI provider implementation."""

    def __init__(self):
        """Initialize OpenAI provider."""
        super().__init__("openai", "OpenAI")
        self._api_key = os.environ.get('OPENAI_API_KEY')

    @property
    def is_available(self) -> bool:
        """Check if provider is available (has valid API key)."""
        return self._api_key is not None

    def get_models(self) -> List[LLMInfo]:
        """Get list of available OpenAI models."""
        return OPENAI_MODELS

    def create_client(
        self,
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        streaming: bool = False,
        base_url: Optional[str] = None,
        toolboxes: Optional[List[LlmToolbox]] = None,
        reasoning_effort: Optional[ReasoningEffort] = None,
        **kwargs
    ) -> LlmClient:
        """Create an OpenAI chat model client.

        Args:
            model: Model name to use
            temperature: Temperature for responses
            max_tokens: Maximum tokens to generate
            streaming: Whether to stream responses
            base_url: Optional base URL for the API
            toolboxes: Optional list of LlmToolbox objects to provide tool support
            reasoning_effort: Optional reasoning effort level for GPT-5 models
            **kwargs: Additional arguments

        Returns:
            Configured LlmClient instance

        Raises:
            ValueError: If OPENAI_API_KEY environment variable is not set
        """
        if not self.is_available:
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        return OpenAILlmClient(
            api_key=self._api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=streaming,
            base_url=base_url,
            toolboxes=toolboxes,
            reasoning_effort=reasoning_effort
        )
