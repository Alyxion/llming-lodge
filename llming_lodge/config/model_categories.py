from typing import ClassVar
from dataclasses import dataclass

@dataclass
class ModelCategories:
    SMALL: ClassVar[str] = "small"
    MEDIUM: ClassVar[str] = "medium"
    LARGE: ClassVar[str] = "large"
    REASONING_SMALL: ClassVar[str] = "reasoning_small"
    REASONING_MEDIUM: ClassVar[str] = "reasoning_medium"
    REASONING_LARGE: ClassVar[str] = "reasoning_large"
    MODEL_CATEGORIES: ClassVar[set[str]] = {
        SMALL,
        MEDIUM,
        LARGE,
        REASONING_SMALL,
        REASONING_MEDIUM,
        REASONING_LARGE
    }