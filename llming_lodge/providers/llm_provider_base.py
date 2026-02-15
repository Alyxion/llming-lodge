"""Base provider interface for LLM providers."""
from abc import ABC, abstractmethod
from typing import List, Optional

from .llm_provider_models import LLMInfo
from ..llm_base_client import LlmClient
from ..tools.llm_toolbox import LlmToolbox


class BaseProvider(ABC):
    """Base class for LLM providers."""

    def __init__(self, name: str, label: str):
        """Initialize provider.

        :param name: The provider name, "openai", "anthropic", etc.
        :param label: The provider label, "OpenAI", "Anthropic", etc.
        """
        self.name = name
        self.label = label

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available (has valid API key)."""
        pass

    @abstractmethod
    def get_models(self) -> List[LLMInfo]:
        """Get list of available models for this provider."""
        pass

    @abstractmethod
    def create_client(
        self,
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        streaming: bool = False,
        base_url: Optional[str] = None,
        toolboxes: Optional[List[LlmToolbox]] = None,
        **kwargs
    ) -> LlmClient:
        """Create an LLM client.

        Args:
            model: Model name to use
            temperature: Temperature for responses
            max_tokens: Maximum tokens to generate
            streaming: Whether to stream responses
            base_url: Optional base URL for the API
            toolboxes: Optional list of LlmToolbox objects to provide tool support
            **kwargs: Additional provider-specific arguments

        Returns:
            Configured LlmClient instance
        """
        pass
