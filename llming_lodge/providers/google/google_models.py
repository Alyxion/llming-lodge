"""Google model configurations."""
from ..llm_provider_models import LLMInfo, ModelSize


GOOGLE_MODELS = [
    LLMInfo(
        provider="google",
        name="gemini_pro",
        label="Gemini 3 Pro",
        model="gemini-3-pro-preview",
        description="Google's most powerful model with deep reasoning capabilities.",
        input_token_price=2.00,
        cached_input_token_price=0.20,
        output_token_price=12.00,
        supports_system_prompt=True,
        model_icon="models/google-gemini-icon.svg",
        company_icon="companies/Google_logo.svg",
        hosting_icon=None,
        size=ModelSize.LARGE,
        max_input_tokens=1000000,
        max_output_tokens=64000,
        popularity=80,
        speed=6,
        quality=9,
        best_use="Complex reasoning",
        highlights=["1M context", "Reasoning", "Multimodal", "Code"],
        supports_image_input=True,
        reasoning=True,
    ),
    LLMInfo(
        provider="google",
        name="gemini_flash",
        label="Gemini 3 Flash",
        model="gemini-3-flash-preview",
        description="Fast and efficient model optimized for quick responses.",
        input_token_price=0.50,
        cached_input_token_price=0.05,
        output_token_price=3.00,
        supports_system_prompt=True,
        model_icon="models/google-gemini-icon.svg",
        company_icon="companies/Google_logo.svg",
        hosting_icon=None,
        size=ModelSize.SMALL,
        max_input_tokens=1000000,
        max_output_tokens=64000,
        popularity=65,
        speed=9,
        quality=7,
        best_use="Quick tasks",
        highlights=["1M context", "Fast", "Low cost"],
        supports_image_input=True,
        reasoning=True,
    ),
]

# Basic model for quick tests (smallest available)
BASIC_MODEL = next(model for model in GOOGLE_MODELS if model.size == ModelSize.SMALL)
