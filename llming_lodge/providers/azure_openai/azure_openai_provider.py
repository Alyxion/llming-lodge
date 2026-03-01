"""Azure OpenAI provider implementation."""
import os
from typing import List, Optional

from llming_lodge.providers import BaseProvider, register_provider
from llming_lodge.llm_base_client import LlmClient
from .azure_openai_models import AZURE_OPENAI_MODELS, LLMInfo
from llming_lodge.providers.openai.openai_client import OpenAILlmClient
from llming_lodge.tools.llm_toolbox import LlmToolbox
from llming_lodge.providers.llm_provider_models import ReasoningEffort


@register_provider("azure_openai")
class AzureOpenAIProvider(BaseProvider):
    """Azure OpenAI provider implementation.

    Uses the same OpenAILlmClient as the standard OpenAI provider but
    configured with Azure endpoints and API type.
    """

    def __init__(self):
        """Initialize Azure OpenAI provider."""
        super().__init__("azure_openai", "Azure OpenAI")
        self._api_key = os.environ.get('AZURE_OPENAI_API_KEY')
        self._endpoint = os.environ.get('AZURE_OPENAI_ENDPOINT')
        self._api_version = os.environ.get(
            'AZURE_OPENAI_API_VERSION', '2025-04-01-preview'
        )

    @property
    def is_available(self) -> bool:
        """Check if provider is available (has valid API key and endpoint)."""
        return self._api_key is not None and self._endpoint is not None

    def get_models(self) -> List[LLMInfo]:
        """Get list of available Azure OpenAI models."""
        return AZURE_OPENAI_MODELS

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
        """Create an Azure OpenAI client.

        Args:
            model: Deployment name to use (matches model names)
            temperature: Temperature for responses
            max_tokens: Maximum tokens to generate
            streaming: Whether to stream responses
            base_url: Optional base URL override (defaults to AZURE_OPENAI_ENDPOINT)
            toolboxes: Optional list of LlmToolbox objects
            reasoning_effort: Optional reasoning effort level
            **kwargs: Additional arguments

        Returns:
            Configured LlmClient instance
        """
        if not self.is_available:
            raise ValueError(
                "AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT "
                "environment variables must be set"
            )

        return OpenAILlmClient(
            api_key=self._api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=streaming,
            base_url=base_url or self._endpoint,
            toolboxes=toolboxes,
            api_type="azure",
            api_version=self._api_version,
            reasoning_effort=reasoning_effort,
        )
