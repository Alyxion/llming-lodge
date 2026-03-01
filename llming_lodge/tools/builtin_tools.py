"""Built-in tools for LLM interactions."""
import logging
from typing import Optional, Callable
from .llm_tool import LlmTool
from .llm_toolbox import LlmToolbox

logger = logging.getLogger(__name__)

# gpt-image-1 pricing (USD per image)
IMAGE_GEN_PRICING = {
    ("1024x1024", "low"): 0.011,
    ("1024x1024", "medium"): 0.042,
    ("1024x1024", "high"): 0.167,
    ("1536x1024", "low"): 0.016,
    ("1024x1536", "low"): 0.016,
    ("1536x1024", "medium"): 0.063,
    ("1024x1536", "medium"): 0.063,
    ("1536x1024", "high"): 0.250,
    ("1024x1536", "high"): 0.250,
}

# Backward compat alias
DALLE3_PRICING = IMAGE_GEN_PRICING


def create_image_generation_tool(openai_client, cost_callback: Optional[Callable[[str, float], None]] = None) -> LlmTool:
    """Create a GPT Image generation tool.

    Args:
        openai_client: OpenAI client with generate_image_sync method
        cost_callback: Optional callback(tool_name, cost_usd) to log costs

    Returns:
        LlmTool configured for image generation
    """
    import os
    image_model = os.environ.get("DALLE_DEPLOYMENT_NAME", "gpt-image-1")

    def generate_image(prompt: str, size: str = "1024x1024", quality: str = "medium") -> str:
        """Generate an image using GPT Image.

        Args:
            prompt: Text description of the image to generate
            size: Image size - "1024x1024", "1536x1024", or "1024x1536"
            quality: "low", "medium", or "high"

        Returns:
            Base64-encoded image data
        """
        # Calculate cost
        cost_key = (size, quality)
        cost_usd = IMAGE_GEN_PRICING.get(cost_key, 0.042)
        logger.debug(f"[IMAGE_GEN] Generating image: model={image_model}, size={size}, quality={quality}, cost=${cost_usd:.3f}")

        result = openai_client.generate_image_sync(
            prompt=prompt,
            size=size,
            quality=quality,
            model=image_model,
        )

        # Log the cost
        if cost_callback:
            cost_callback("generate_image", cost_usd)
            logger.debug(f"[IMAGE_GEN] Cost logged: ${cost_usd:.3f}")

        return result

    return LlmTool(
        name="generate_image",
        description="Generate an image from a text description using GPT Image. Use this when the user asks you to create, draw, or generate an image. Returns base64-encoded image data.",
        func=generate_image,
        parameters={
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Detailed text description of the image to generate. Be specific about style, colors, composition, etc."
                },
                "size": {
                    "type": "string",
                    "enum": ["1024x1024", "1536x1024", "1024x1536"],
                    "description": "Image size. Use 1024x1024 for square, 1536x1024 for landscape, 1024x1536 for portrait.",
                    "default": "1024x1024"
                },
                "quality": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                    "description": "Image quality. 'high' produces more detail but costs more.",
                    "default": "medium"
                }
            },
            "required": ["prompt", "size", "quality"]
        }
    )


def create_web_search_toolbox() -> LlmToolbox:
    """Create a web search toolbox.

    Note: OpenAI's web_search is a built-in tool type, not a function call.
    This toolbox just contains the string "web_search" which the OpenAI client
    handles specially.

    Returns:
        LlmToolbox with web search capability
    """
    return LlmToolbox(
        name="web_search",
        description="Search the web for current information",
        tools=["web_search"]
    )


def create_image_generation_toolbox(openai_client, cost_callback: Optional[Callable[[str, float], None]] = None) -> LlmToolbox:
    """Create an image generation toolbox.

    Args:
        openai_client: OpenAI client with generate_image_sync method
        cost_callback: Optional callback(tool_name, cost_usd) to log costs

    Returns:
        LlmToolbox with DALL-E image generation capability
    """
    return LlmToolbox(
        name="image_generation",
        description="Generate images using GPT Image",
        tools=[create_image_generation_tool(openai_client, cost_callback)]
    )
