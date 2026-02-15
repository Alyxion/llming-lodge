"""Google model configurations."""
from typing import List

from ..llm_provider_models import LLMInfo, ModelSize


GOOGLE_MODELS = [
    LLMInfo(
        provider="google",
        name="gemini_pro",
        label="Gemini 2.5 Pro",
        model="gemini-2.5-pro",
        description="Google's most powerful Gemini model with adaptive thinking.",
        input_token_price=1.25,
        output_token_price=5.00,
        supports_system_prompt=True,
        model_icon="models/google-gemini-icon.svg",
        company_icon="companies/Google_logo.svg",
        hosting_icon=None,
        size=ModelSize.LARGE,
        max_input_tokens=1000000,
        max_output_tokens=8192,
        popularity=80,
        speed=6,
        quality=9,
        best_use="Complex reasoning",
        highlights=["1M context", "Reasoning", "Multimodal", "Code"],
    ),
    LLMInfo(
        provider="google",
        name="gemini_flash",
        label="Gemini 2.5 Flash",
        model="gemini-2.5-flash",
        description="Fast and efficient version optimized for quick responses.",
        input_token_price=0.50,
        output_token_price=2.00,
        supports_system_prompt=True,
        model_icon="models/google-gemini-icon.svg",
        company_icon="companies/Google_logo.svg",
        hosting_icon=None,
        size=ModelSize.SMALL,
        max_input_tokens=1000000,
        max_output_tokens=8192,
        popularity=65,
        speed=9,
        quality=7,
        best_use="Quick tasks",
        highlights=["1M context", "Fast", "Low cost"],
    ),
]

# Basic model for quick tests (smallest available)
BASIC_MODEL = next(model for model in GOOGLE_MODELS if model.size == ModelSize.SMALL)
