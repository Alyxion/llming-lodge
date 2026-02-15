from dataclasses import dataclass, field
from .llm_global_config import LLMGlobalConfig, LLMBaseConfig

@dataclass
class LLMUserConfig(LLMBaseConfig):
    """User specific configuration for LLMs."""

    global_config: LLMGlobalConfig = field(default_factory=LLMGlobalConfig)
    """Pointer to the global configuration."""
    default_models: dict[str, str] = field(default_factory=dict)    
    """Default models for providers."""
    budgets: list["LLMBudget"] = field(default_factory=list)
    """List of budgets."""
    prompt_parameters: dict[str, str] = field(default_factory=dict)
    """Prompt parameters."""
    
    def is_model_supported(self, model: str) -> bool:
        """Check if a model is supported by the user config."""
        return self.global_config.is_model_supported(model) and super().is_model_supported(model)

    def get_default_model(self, category: str) -> str:
        """Get the default model for a given category. The user setting overrules the global setting.
        
        :param category: The category to look up. See ModelCategories for valid categories.
        :return: The default model for the specified category
        
        :raises ValueError: If the category is not found
        """
        if category in self.default_models:
            return self.default_models[category]
        else:
            return self.global_config.get_default_model(category)
            