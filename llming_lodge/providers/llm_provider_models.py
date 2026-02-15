"""Common models for LLM providers."""
from dataclasses import dataclass, field
from enum import IntEnum, Enum
from typing import Dict, List, Optional, Union, Literal


class ModelSize(IntEnum):
    """Size categories for LLM models."""
    VERY_SMALL = 1
    SMALL = 2
    MEDIUM = 3
    LARGE = 4
    VERY_LARGE = 5


class ReasoningEffort(str, Enum):
    """Reasoning effort levels for models that support it.

    - NONE: Disable reasoning completely (fastest, most deterministic)
    - MINIMAL: Very fast, almost no thinking traces
    - LOW: Some reasoning, faster responses
    - MEDIUM: Moderate reasoning (default for most tasks)
    - HIGH: Full reasoning capabilities
    """
    NONE = "none"      # Explicitly disable reasoning
    MINIMAL = "minimal"  # Very fast, no thinking traces
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class LLMInfo:
    """Information about an LLM model."""
    provider: str  # Provider name
    name: str  # High-level name for identification
    label: str  # Human-readable label
    model: str  # Actual model name for the API
    description: str
    input_token_price: float  # Price per 1M input tokens
    cached_input_token_price: float = 0.0  # Price per 1M cached input tokens (default 0.0 for backward compatibility)
    output_token_price: float = 0.0  # Price per 1M output tokens
    size: ModelSize = ModelSize.MEDIUM  # Model size category
    max_input_tokens: int = 64000  # Maximum number of input tokens
    max_output_tokens: int = 4096  # Maximum number of output tokens
    api_base: Optional[str] = None  # Base URL for the API, None to use default
    supports_system_prompt: bool = True  # Whether the model supports system prompts
    tokenizer_name: Optional[str] = None  # Name of the tokenizer to use for token counting, None to use provider default
    model_icon: Optional[str] = None  # Path to model-specific icon
    company_icon: Optional[str] = None  # Path to company/inventor icon
    hosting_icon: Optional[str] = None  # Path to optional hosting company icon
    popularity: int = 0  # Higher value = more popular (OpenAI should be highest)
    reasoning: bool = False  # True if this is a reasoning model
    reasoning_effort: Optional[ReasoningEffort] = None  # Reasoning effort level (None = use model default)
    default_reasoning_effort: Optional[ReasoningEffort] = None  # Default reasoning effort for this model size category
    enforced_temperature: Optional[float] = None  # If set, this temperature value will be enforced for this model
    supports_image_input: bool = False  # Whether the model supports image inputs

    # UI metadata for model selector
    speed: int = 5  # 1–10, higher = faster response
    quality: int = 5  # 1–10, higher = better reasoning/output quality
    best_use: str = "General"  # Short label for the model's strength
    highlights: List[str] = field(default_factory=list)  # Key capabilities shown in detail view, e.g. ["Reasoning", "Images", "Web search"]

    # Tool configuration
    default_tools: List[str] = field(default_factory=list)  # Default tools to enable for this model
    native_tools: Dict[str, Dict] = field(default_factory=dict)  # Provider-native tool configs (e.g., {"web_search": {"type": "web_search"}})
