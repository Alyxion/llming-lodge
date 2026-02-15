"""LLM Manager for managing available LLM providers and sessions."""
from typing import Dict, List, Optional, Type, Set, TYPE_CHECKING

from .session import LLMConfig, ChatSession
from .config import LLMUserConfig, LLMGlobalConfig
from .providers import (
    LLMInfo,
    get_provider,
    BaseProvider,
)

if TYPE_CHECKING:
    from llming_lodge.budget import LLMBudgetManager


class LLMManager:
    """Manages multiple LLM providers and sessions."""

    # Currently supported providers
    SUPPORTED_PROVIDERS = {
        "openai",
        "anthropic",
        "mistral",
        "google",
        "together"
    }

    def __init__(self, *, user_config: LLMUserConfig | None = None, budget_manager: Optional["LLMBudgetManager"] = None):
        """Initialize LLM manager."""
        # Initialize provider instances
        self.user_config = user_config or LLMUserConfig()
        self.budget_manager: Optional["LLMBudgetManager"] = budget_manager
        if self.budget_manager is None and (len(self.user_config.budgets) > 0 or len(self.user_config.global_config.budgets) > 0):
            from llming_lodge.budget import LLMBudgetManager
            combined_list = self.user_config.budgets + self.user_config.global_config.budgets
            self.budget_manager = LLMBudgetManager(combined_list)
        self.providers: Dict[str, BaseProvider] = {}
        for provider_name in self.SUPPORTED_PROVIDERS:
            try:
                provider_class = get_provider(provider_name)
                provider = provider_class()
                # Only add provider if it's available (has valid API key)
                if provider.is_available:
                    # check if any model is available for the user
                    models = provider.get_models()
                    any_model = False
                    for model in models:
                        full_model_name = f"{provider_name}:{model.name}"
                        if self.user_config.is_model_supported(full_model_name):
                            any_model = True
                            break
                    if any_model:
                        self.providers[provider_name] = provider
            except (ValueError, KeyError):
                # Skip providers that can't be initialized
                continue

    def register_provider(self, provider: BaseProvider) -> None:
        """Register a custom provider instance.

        Use this to add OpenAI-compatible or other custom providers at runtime.

        :param provider: A BaseProvider instance to register
        """
        from .providers import PROVIDERS
        PROVIDERS[provider.name] = type(provider)
        if provider.is_available:
            self.providers[provider.name] = provider

    def register_openai_compatible(
        self,
        name: str,
        label: str,
        api_key: str,
        base_url: str,
        models: List[LLMInfo],
    ) -> None:
        """Register a generic OpenAI-compatible provider.

        Convenience method for adding any OpenAI-compatible API endpoint.

        Example::

            manager.register_openai_compatible(
                name="deepseek",
                label="DeepSeek",
                api_key=os.environ["DEEPSEEK_API_KEY"],
                base_url="https://api.deepseek.com",
                models=[
                    LLMInfo(provider="deepseek", name="deepseek_chat", label="DeepSeek V3",
                            model="deepseek-chat", description="...",
                            input_token_price=0.5, output_token_price=1.5),
                ],
            )

        :param name: Provider name (used as key in registry)
        :param label: Human-readable label
        :param api_key: API key for the endpoint
        :param base_url: Base URL for the OpenAI-compatible API
        :param models: List of LLMInfo model definitions
        """
        from .providers.generic_openai_provider import GenericOpenAIProvider
        provider = GenericOpenAIProvider(
            name=name,
            label=label,
            api_key=api_key,
            base_url=base_url,
            models=models,
        )
        self.register_provider(provider)

    def get_available_llms(self) -> List[LLMInfo]:
        """Get list of available LLMs (only for providers with valid API keys)."""
        models = []
        for provider in self.providers.values():
            models.extend(provider.get_models())
        return models

    def get_providers_for_model(self, model: str) -> Set[str]:
        """Get all providers that can serve a given model name.
        
        :param model: The model name or high-level name to look up. Alternatively, the model name can be specified as "provider/model".
        :return: Set of provider names that can serve this model
        :raises ValueError: If model is not found
        """
        if ":" in model:  # explicit provider name
            provider_name, model_name = model.split(":")
            if provider_name not in self.providers:
                return set()
            provider = self.providers[provider_name]
            for model_info in provider.get_models():
                if model_info.name == model_name:
                    return {provider_name}
            return set()
        providers = set()
        for provider_name, provider in self.providers.items():
            for model_info in provider.get_models():
                if model_info.name == model or model_info.model == model:
                    full_model_name = f"{provider_name}:{model_info.name}"
                    if not self.user_config.is_model_supported(full_model_name):
                        continue
                    providers.add(provider_name)
                    break
        
        if not providers:
            raise ValueError(f"Model {model} not found")
        return providers

    def get_provider_for_model(self, model: str) -> str:
        """Get a provider for a given model name.
        
        :param model: The model name to look up, or "provider/model" to look up by provider name
        :return: First available provider name for the specified model
            
        :raises ValueError: If model is not found
        """
        providers = self.get_providers_for_model(model)
        if not providers:
            raise ValueError(f"Model {model} not found")
        return next(iter(providers))

    def get_default_model(self, category: str) -> str | None:
        """Get the default model name for a given category.
        
        :param category: The category to look up. See ModelCategories for valid categories. Example: "small", "medium", "large", "reasoning_small", "reasoning_medium", "reasoning_large"
        :return: The default model for the specified category or None if not found
        """   
        default_model = self.user_config.get_default_model(category)
        return default_model

    def get_model_info(self, model: str) -> LLMInfo:
        """Get model info for a specific model.
        
        :param model: The model to get info for, or "provider/model" to look up by provider name
        :return: LLMInfo for the specified model
            
        :raises ValueError: If model is not found
        """
        provider_name = self.get_provider_for_model(model)
        provider = self.providers[provider_name]
        if ":" in model:
            model = model.split(":")[1]
        
        for info in provider.get_models():
            if info.name == model or info.model == model:
                return info
                
        raise ValueError(f"Model {model} not found")

    def get_config_for_model(self, model: str) -> LLMConfig:
        """Get configuration for a specific model.
        
        :param model: The model to get configuration for
        :return: LLMConfig for the specified model
            
        :raises ValueError: If model is not found or provider is not available
        """
        model_info = self.get_model_info(model)
        return LLMConfig(
            provider=model_info.provider,
            model=model_info.model,
            base_url=model_info.api_base,
            temperature=0.7,
            max_input_tokens=model_info.max_input_tokens,
            max_tokens=model_info.max_output_tokens
        )

    def create_session(
        self,
        config: Optional[LLMConfig] = None,
        model: Optional[str] = None,
        category: Optional[str] = None,
        system_prompt: Optional[str] = None,
        budget_manager: Optional["LLMBudgetManager"] = None,
        user_id: Optional[str] = None
    ) -> ChatSession:
        """Create a new chat session.
        
        :param config: Optional LLMConfig to use. If not provided, model must be specified.
        :param model: Optional model name to use. Ignored if config is provided.
        :param category: Optional category to use. Ignored if config is provided.
        :param system_prompt: Optional system prompt to set context
        :param budget_manager: Optional budget manager to use. Defaults to predefined budget manager if provided
        :param user_id: Optional user ID to use
        :return: New ChatSession instance
        
        :raises ValueError: If neither config nor model is provided, or if model is not found
        """
        if category is not None:
            model = self.get_default_model(category)
            if model is None:
                raise ValueError(f"No default model found for category {category}")
        if budget_manager is None:
            budget_manager = self.budget_manager
        if config is None:
            if model is None:
                raise ValueError("Either config or model must be provided")
            config = self.get_config_for_model(model)
        
        return ChatSession(config=config, system_prompt=system_prompt, budget_manager=budget_manager, user_id=user_id)
