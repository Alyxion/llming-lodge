"""Anthropic model configurations."""
from typing import List

from ..llm_provider_models import LLMInfo, ModelSize


ANTHROPIC_MODELS = [
    LLMInfo(
        provider="anthropic",
        name="claude_opus",
        label="Claude Opus 4.5",
        model="claude-opus-4-5-20251101",
        description="Most capable Claude model for complex tasks.",
        input_token_price=15.00,
        output_token_price=75.00,
        model_icon="models/claude-240.svg",
        company_icon="companies/Anthropic_logo.svg",
        hosting_icon=None,
        size=ModelSize.LARGE,
        max_input_tokens=200000,
        max_output_tokens=8192,
        popularity=70,
        speed=5,
        quality=10,
        best_use="Deep analysis",
        highlights=["Best reasoning", "Code", "Analysis", "Web search"],
        default_tools=["web_search"],
    ),
    LLMInfo(
        provider="anthropic",
        name="claude_sonnet",
        label="Claude Sonnet 4.5",
        model="claude-sonnet-4-5-20250929",
        description="Balanced model with best-in-class reasoning and coding.",
        input_token_price=3.00,
        output_token_price=15.00,
        model_icon="models/claude-240.svg",
        company_icon="companies/Anthropic_logo.svg",
        hosting_icon=None,
        size=ModelSize.MEDIUM,
        max_input_tokens=200000,
        max_output_tokens=8192,
        popularity=90,
        speed=7,
        quality=9,
        best_use="Analysis & code",
        highlights=["Reasoning", "Code", "Analysis", "Web search"],
        default_tools=["web_search"],
    ),
    LLMInfo(
        provider="anthropic",
        name="claude_haiku",
        label="Claude Haiku 4.5",
        model="claude-haiku-4-5-20251001",
        description="Fastest model with near-frontier intelligence.",
        input_token_price=1.00,
        output_token_price=5.00,
        model_icon="models/claude-240.svg",
        company_icon="companies/Anthropic_logo.svg",
        hosting_icon=None,
        size=ModelSize.SMALL,
        max_input_tokens=200000,
        max_output_tokens=64000,
        popularity=75,
        speed=10,
        quality=7,
        best_use="Quick tasks",
        highlights=["Fast", "Code", "Web search"],
        default_tools=["web_search"],
    ),
]

# Basic model for quick tests (smallest/cheapest available)
BASIC_MODEL = next(model for model in ANTHROPIC_MODELS if model.size == ModelSize.SMALL)
