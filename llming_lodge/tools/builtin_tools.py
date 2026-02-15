"""Built-in tools for LLM interactions."""
import logging
from typing import Optional, Callable
from .llm_tool import LlmTool
from .llm_toolbox import LlmToolbox

logger = logging.getLogger(__name__)

# DALL-E 3 pricing (USD per image)
DALLE3_PRICING = {
    ("1024x1024", "standard"): 0.040,
    ("1024x1024", "hd"): 0.080,
    ("1792x1024", "standard"): 0.080,
    ("1024x1792", "standard"): 0.080,
    ("1792x1024", "hd"): 0.120,
    ("1024x1792", "hd"): 0.120,
}


def create_image_generation_tool(openai_client, cost_callback: Optional[Callable[[str, float], None]] = None) -> LlmTool:
    """Create a DALL-E image generation tool.

    Args:
        openai_client: OpenAI client with generate_image_sync method
        cost_callback: Optional callback(tool_name, cost_usd) to log costs

    Returns:
        LlmTool configured for image generation
    """
    def generate_image(prompt: str, size: str = "1024x1024", quality: str = "standard") -> str:
        """Generate an image using DALL-E 3.

        Args:
            prompt: Text description of the image to generate
            size: Image size - "1024x1024", "1792x1024", or "1024x1792"
            quality: "standard" or "hd"

        Returns:
            Base64-encoded image data
        """
        # Calculate cost
        cost_key = (size, quality)
        cost_usd = DALLE3_PRICING.get(cost_key, 0.080)  # Default to $0.08 if unknown
        logger.debug(f"[DALLE] Generating image: size={size}, quality={quality}, cost=${cost_usd:.3f}")

        result = openai_client.generate_image_sync(
            prompt=prompt,
            size=size,
            quality=quality,
            model="dall-e-3"
        )

        # Log the cost
        if cost_callback:
            cost_callback("generate_image", cost_usd)
            logger.debug(f"[DALLE] Cost logged: ${cost_usd:.3f}")

        return result

    return LlmTool(
        name="generate_image",
        description="Generate an image from a text description using DALL-E 3. Use this when the user asks you to create, draw, or generate an image. Returns base64-encoded image data.",
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
                    "enum": ["1024x1024", "1792x1024", "1024x1792"],
                    "description": "Image size. Use 1024x1024 for square, 1792x1024 for landscape, 1024x1792 for portrait.",
                    "default": "1024x1024"
                },
                "quality": {
                    "type": "string",
                    "enum": ["standard", "hd"],
                    "description": "Image quality. 'hd' produces higher detail but takes longer.",
                    "default": "standard"
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
        description="Generate images using DALL-E",
        tools=[create_image_generation_tool(openai_client, cost_callback)]
    )
