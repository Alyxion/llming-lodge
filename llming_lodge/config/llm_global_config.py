import fnmatch
from dataclasses import dataclass, field
from typing import ClassVar

@dataclass
class LLMBaseConfig:

    """Base configuration for LLMs."""
    included_models: list[str] = field(default_factory=lambda: ["*"])
    """Filter for included models. The filter is applied to the fully qualified model name (e.g., "anthropic:claude_3_opus"). "*" matches all models."""
    excluded_models: list[str] = field(default_factory=list)
    """Filter for excluded models. The filter is applied to the fully qualified model name (e.g., "anthropic:claude_3_opus"). "*" matches all models."""

    def is_model_supported(self, model: str) -> bool:
        """Check if a model is supported by the global config.
        
        :param model: The fully qualified model name (e.g., "anthropic:claude_3_opus")
        :return: True if the model is supported, False otherwise
        """
        included: bool = False
        for included_model in self.included_models:
            if fnmatch.fnmatch(model, included_model):
                included = True
                break
        if not included:
            return False
        for excluded_model in self.excluded_models:
            if fnmatch.fnmatch(model, excluded_model):
                return False
        return True
            
@dataclass
class LLMGlobalConfig(LLMBaseConfig):
    """Defines the global configuration for LLMs."""
    default_models: dict[str, str] = field(default_factory=lambda: {"small": "openai:gpt-5-nano", "medium": "openai:gpt-5-mini", "large": "openai:gpt-5.2",
                                                                    "reasoning_small": "openai:gpt-5-mini", "reasoning_medium": "openai:gpt-5.2", "reasoning_large": "openai:gpt-5.2"})
    """Default models for providers."""
    budgets: list["LLMBudget"] = field(default_factory=list)
    """List of budgets."""

    def get_default_model(self, category: str) -> str | None:
        """Get the default model for a given category.
        
        :param category: The category to look up. See ModelCategories for valid categories.
        :return: The default model for the specified category
        """
        return self.default_models.get(category, None)


    