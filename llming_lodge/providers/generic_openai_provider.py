"""Generic provider for any OpenAI-compatible API endpoint.

Register custom providers at runtime via LLMManager::

    from llming_lodge.llm_provider_manager import LLMManager
    from llming_lodge.providers.llm_provider_models import LLMInfo

    manager = LLMManager()
    manager.register_openai_compatible(
        name="deepseek",
        label="DeepSeek",
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
        models=[
            LLMInfo(
                provider="deepseek", name="deepseek_chat",
                label="DeepSeek V3", model="deepseek-chat",
                description="Optimized for conversational interactions.",
                input_token_price=0.5, output_token_price=1.5,
            ),
        ],
    )
"""
from typing import List, Optional

from llming_lodge.providers.llm_provider_base import BaseProvider
from llming_lodge.llm_base_client import LlmClient
from llming_lodge.providers.llm_provider_models import LLMInfo
from llming_lodge.providers.openai_compat_client import OpenAICompatibleClient


class GenericOpenAIProvider(BaseProvider):
    """Provider for any OpenAI-compatible API endpoint.

    Unlike the built-in providers which read API keys from environment
    variables, this provider accepts all configuration explicitly.
    """

    def __init__(
        self,
        name: str,
        label: str,
        api_key: str,
        base_url: str,
        models: List[LLMInfo],
    ):
        super().__init__(name, label)
        self._api_key = api_key
        self._base_url = base_url
        self._models = models

    @property
    def is_available(self) -> bool:
        return bool(self._api_key)

    def get_models(self) -> List[LLMInfo]:
        return self._models

    def create_client(
        self,
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        streaming: bool = False,
        base_url: Optional[str] = None,
        **kwargs,
    ) -> LlmClient:
        return OpenAICompatibleClient(
            api_key=self._api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=streaming,
            base_url=base_url or self._base_url,
        )
