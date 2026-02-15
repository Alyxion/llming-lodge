"""Together-hosted DeepSeek model configurations."""
from typing import List

from ...llm_provider_models import LLMInfo, ModelSize


TOGETHER_DEEPSEEK_MODELS = [
    LLMInfo(
        provider="together",
        name="deepseek_reasoner",
        label="DeepSeek R1 (Together)",
        model="deepseek-ai/DeepSeek-R1",
        description="Advanced model for complex reasoning and generation tasks.",
        input_token_price=8.00,
        output_token_price=14.00,
        api_base="https://api.together.xyz/v1",
        tokenizer_name="cl100k_base",
        model_icon="models/deepseek-logo-icon.svg",
        company_icon="companies/DeepSeek_logo.svg",
        hosting_icon="companies/together-ai-branding-darkOnLight.png",
        size=ModelSize.LARGE,
        max_input_tokens=131072,
        max_output_tokens=4096,
        popularity=50,
        speed=4,
        quality=9,
        best_use="Deep reasoning",
        highlights=["Deep reasoning", "Math", "Code"],
    ),
    LLMInfo(
        provider="together",
        name="deepseek_chat",
        label="DeepSeek V3 (Together)",
        model="deepseek-ai/DeepSeek-V3",
        description="Optimized for conversational interactions.",
        input_token_price=1.75,
        output_token_price=2.75,
        api_base="https://api.together.xyz/v1",
        tokenizer_name="cl100k_base",
        model_icon="models/deepseek-logo-icon.svg",
        company_icon="companies/DeepSeek_logo.svg",
        hosting_icon="companies/together-ai-branding-darkOnLight.png",
        size=ModelSize.MEDIUM,
        max_input_tokens=131072,
        max_output_tokens=4096,
        popularity=45,
        speed=7,
        quality=7,
        best_use="Code & math",
        highlights=["Code", "Math", "Low cost"],
    ),
]

# Basic model for quick tests (smallest available)
BASIC_MODEL = next(model for model in TOGETHER_DEEPSEEK_MODELS if model.size == ModelSize.MEDIUM)
