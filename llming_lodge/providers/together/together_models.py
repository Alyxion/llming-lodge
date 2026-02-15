"""Together model configurations."""
from typing import List

from ..llm_provider_models import LLMInfo, ModelSize
from llming_lodge.providers.together.deepseek.deepseek_models import TOGETHER_DEEPSEEK_MODELS


# Together only hosts other providers' models
TOGETHER_MODELS = TOGETHER_DEEPSEEK_MODELS

# Basic model for quick tests (smallest available)
BASIC_MODEL = next(model for model in TOGETHER_MODELS if model.size == ModelSize.MEDIUM)
