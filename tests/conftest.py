"""Test configuration and fixtures."""
from llming_lodge.llm_provider_manager import LLMManager
from llming_lodge.providers.openai.openai_models import BASIC_MODEL as OPENAI_BASIC
from llming_lodge.providers.anthropic.anthropic_models import BASIC_MODEL as ANTHROPIC_BASIC
from llming_lodge.providers.mistral.mistral_models import BASIC_MODEL as MISTRAL_BASIC
from llming_lodge.providers.google.google_models import BASIC_MODEL as GOOGLE_BASIC
from llming_lodge.providers.together.together_models import BASIC_MODEL as TOGETHER_BASIC

# Get all models at import time to ensure consistent test collection
_manager = LLMManager()
ALL_MODELS = sorted([(info.provider, info.model) for info in _manager.get_available_llms()],
                   key=lambda x: f"{x[0]}/{x[1]}")

# Basic models for quick tests (smallest available from each provider)
BASIC_MODELS = [
    (OPENAI_BASIC.provider, OPENAI_BASIC.model),
    (ANTHROPIC_BASIC.provider, ANTHROPIC_BASIC.model),
    (MISTRAL_BASIC.provider, MISTRAL_BASIC.model),
    (GOOGLE_BASIC.provider, GOOGLE_BASIC.model),
    (TOGETHER_BASIC.provider, TOGETHER_BASIC.model),
]
