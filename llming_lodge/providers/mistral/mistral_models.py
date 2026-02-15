"""Mistral model configurations."""
from typing import List

from ..llm_provider_models import LLMInfo, ModelSize


MISTRAL_MODELS = [
    LLMInfo(
        provider="mistral",
        name="mistral_large",
        label="Mistral Large",
        model="mistral-large-latest",
        description="Flagship large model delivering state-of-the-art performance.",
        input_token_price=2.00,
        output_token_price=6.00,
        model_icon="models/mistral-ai-icon.svg",
        company_icon="companies/Mistral_AI_logo.svg",
        hosting_icon=None,
        size=ModelSize.LARGE,
        max_input_tokens=128000,
        max_output_tokens=4096,
        popularity=40,
        speed=6,
        quality=8,
        best_use="Multilingual",
        highlights=["Multilingual", "Reasoning", "EU hosted"],
    ),
    LLMInfo(
        provider="mistral",
        name="mistral_medium",
        label="Mistral Medium",
        model="mistral-medium-latest",
        description="Balanced model for most tasks.",
        input_token_price=0.50,
        output_token_price=1.50,
        model_icon="models/mistral-ai-icon.svg",
        company_icon="companies/Mistral_AI_logo.svg",
        hosting_icon=None,
        size=ModelSize.MEDIUM,
        max_input_tokens=32000,
        max_output_tokens=4096,
        popularity=30,
        speed=8,
        quality=6,
        best_use="Multilingual",
        highlights=["Multilingual", "Balanced", "EU hosted"],
    ),
    LLMInfo(
        provider="mistral",
        name="mistral_small",
        label="Mistral Small",
        model="mistral-small-latest",
        description="Fast and efficient model for simpler tasks.",
        input_token_price=0.20,
        output_token_price=0.60,
        model_icon="models/mistral-ai-icon.svg",
        company_icon="companies/Mistral_AI_logo.svg",
        hosting_icon=None,
        size=ModelSize.SMALL,
        max_input_tokens=16384,
        max_output_tokens=4096,
        popularity=20,
        speed=9,
        quality=4,
        best_use="Quick tasks",
        highlights=["Fast", "Low cost", "EU hosted"],
    ),
]

# Basic model for quick tests (smallest available)
BASIC_MODEL = next(model for model in MISTRAL_MODELS if model.size == ModelSize.SMALL)
