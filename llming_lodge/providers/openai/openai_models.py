"""OpenAI model configurations."""
from ..llm_provider_models import LLMInfo, ModelSize, ReasoningEffort


OPENAI_MODELS = [
    # --- GPT-5.2 (Latest Flagship - December 2025) ---
    # All GPT-5 models: 272K input, 128K output, 400K total context
    # Source: https://openai.com/index/introducing-gpt-5-for-developers/
    LLMInfo(
        provider="openai",
        name="gpt-5.2",
        label="GPT-5.2",
        model="gpt-5.2",
        description="Most capable model for professional knowledge work with thinking capabilities.",
        input_token_price=1.75,    # 1.75 USD per 1M tokens
        cached_input_token_price=0.175,  # 90% discount on cached input
        output_token_price=14.00,  # 14.00 USD per 1M tokens
        model_icon="models/chatgpt-240.svg",
        company_icon="companies/OpenAI_logo_2025.svg",
        hosting_icon=None,
        size=ModelSize.LARGE,
        max_input_tokens=272000,
        max_output_tokens=128000,
        popularity=100,
        speed=7,
        quality=9,
        best_use="General purpose",
        highlights=["Reasoning", "Images", "Web search", "Tools"],
        supports_image_input=True,
        reasoning=True,
        default_reasoning_effort=ReasoningEffort.LOW,
        enforced_temperature=1.0,
        default_tools=["web_search", "generate_image"],
        native_tools={"web_search": {"type": "web_search"}},
    ),
    # --- GPT-5 Mini/Nano (August 2025) ---
    LLMInfo(
        provider="openai",
        name="gpt-5-mini",
        label="GPT-5 Mini",
        model="gpt-5-mini",
        description="Fast and capable model for general chat and tool calling.",
        input_token_price=0.40,
        cached_input_token_price=0.10,
        output_token_price=1.60,
        model_icon="models/chatgpt-240.svg",
        company_icon="companies/OpenAI_logo_2025.svg",
        hosting_icon=None,
        size=ModelSize.MEDIUM,
        max_input_tokens=272000,
        max_output_tokens=128000,
        popularity=85,
        speed=9,
        quality=7,
        best_use="Quick tasks",
        highlights=["Fast", "Images", "Web search", "Tools"],
        supports_image_input=True,
        reasoning=True,
        default_reasoning_effort=ReasoningEffort.MINIMAL,
        enforced_temperature=1.0,
        default_tools=["web_search", "generate_image"],
        native_tools={"web_search": {"type": "web_search"}},
    ),
    LLMInfo(
        provider="openai",
        name="gpt-5-nano",
        label="GPT-5 Nano",
        model="gpt-5-nano",
        description="Fastest, most cost-effective model for low-latency tasks.",
        input_token_price=0.10,
        cached_input_token_price=0.025,
        output_token_price=0.40,
        model_icon="models/chatgpt-240.svg",
        company_icon="companies/OpenAI_logo_2025.svg",
        hosting_icon=None,
        size=ModelSize.SMALL,
        max_input_tokens=272000,
        max_output_tokens=128000,
        popularity=60,
        speed=10,
        quality=5,
        best_use="Low-latency tasks",
        highlights=["Fastest", "Low cost", "Web search"],
        reasoning=True,
        default_reasoning_effort=ReasoningEffort.NONE,
        enforced_temperature=1.0,
        default_tools=[],
        native_tools={"web_search": {"type": "web_search"}},
    ),
]

# Basic model for quick tests (smallest available)
BASIC_MODEL = next(model for model in OPENAI_MODELS if model.size == ModelSize.SMALL)
