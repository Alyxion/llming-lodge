"""Gemini image generation using Google's Imagen 3 model.

Uses the google-genai SDK to generate images via Gemini 2.5 Flash Image model.
"""
import base64
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Gemini image generation pricing (approximate)
# Based on Gemini API pricing for image generation
GEMINI_IMAGE_PRICING = {
    "1:1": 0.04,    # Square
    "16:9": 0.04,   # Landscape
    "9:16": 0.04,   # Portrait
    "4:3": 0.04,    # Standard
    "3:4": 0.04,    # Portrait standard
}


async def generate_image_with_gemini(
    prompt: str,
    api_key: str,
    aspect_ratio: str = "1:1",
    model: str = "gemini-2.0-flash-exp-image-generation",
) -> str:
    """Generate an image using Gemini's native image generation.

    Args:
        prompt: Text description of the image to generate
        api_key: Google API key
        aspect_ratio: Aspect ratio (1:1, 16:9, 9:16, 4:3, 3:4)
        model: Gemini model to use for image generation

    Returns:
        Base64-encoded image data

    Raises:
        Exception: If image generation fails
    """
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise ImportError(
            "google-genai package is required for Gemini image generation. "
            "Install it with: pip install google-genai"
        )

    logger.info(f"[GEMINI_IMAGE] Generating image: prompt={prompt[:50]}..., aspect_ratio={aspect_ratio}, model={model}")

    client = genai.Client(api_key=api_key)

    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                image_config=types.ImageConfig(
                    aspect_ratio=aspect_ratio,
                )
            ),
        )

        # Extract image from response
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'inline_data') and part.inline_data:
                    # Get the image data
                    image_data = part.inline_data.data
                    # If it's already bytes, encode to base64
                    if isinstance(image_data, bytes):
                        return base64.b64encode(image_data).decode('utf-8')
                    # If it's already base64 string, return as-is
                    return image_data

        raise ValueError("No image data in response")

    except Exception as e:
        logger.error(f"[GEMINI_IMAGE] Error generating image: {e}")
        raise


def generate_image_with_gemini_sync(
    prompt: str,
    api_key: str,
    aspect_ratio: str = "1:1",
    model: str = "gemini-2.0-flash-exp-image-generation",
) -> str:
    """Synchronous wrapper for Gemini image generation.

    Args:
        prompt: Text description of the image to generate
        api_key: Google API key
        aspect_ratio: Aspect ratio (1:1, 16:9, 9:16, 4:3, 3:4)
        model: Gemini model to use for image generation

    Returns:
        Base64-encoded image data
    """
    import asyncio

    try:
        loop = asyncio.get_running_loop()
        # If there's a running loop, use run_coroutine_threadsafe
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(
                asyncio.run,
                generate_image_with_gemini(prompt, api_key, aspect_ratio, model)
            )
            return future.result()
    except RuntimeError:
        # No running loop, create one
        return asyncio.run(generate_image_with_gemini(prompt, api_key, aspect_ratio, model))


def _get_google_api_key(google_client=None, api_key: Optional[str] = None) -> str:
    """Extract Google API key from various sources.

    Args:
        google_client: Optional Google client
        api_key: Explicit API key

    Returns:
        Google API key

    Raises:
        ValueError: If no API key can be found
    """
    import os

    if api_key:
        return api_key
    elif google_client and hasattr(google_client, '_client'):
        # Try to get from client
        key = getattr(google_client._client, 'google_api_key', None)
        if key:
            return key
    # Fall back to environment variables (try both names)
    key = os.environ.get('GEMINI_KEY') or os.environ.get('GOOGLE_API_KEY')
    if key:
        return key

    raise ValueError("Google API key is required for image generation")


async def generate_image_gemini(
    prompt: str,
    google_client=None,
    api_key: Optional[str] = None,
    aspect_ratio: str = "1:1",
) -> str:
    """Generate an image using Gemini - async entry point for tool execution.

    Args:
        prompt: Text description of the image to generate
        google_client: Optional Google client (for extracting API key)
        api_key: Google API key (if not using client)
        aspect_ratio: Aspect ratio (1:1, 16:9, 9:16, 4:3, 3:4)

    Returns:
        Base64-encoded image data
    """
    key = _get_google_api_key(google_client, api_key)

    return await generate_image_with_gemini(
        prompt=prompt,
        api_key=key,
        aspect_ratio=aspect_ratio,
    )


def generate_image_gemini_sync(
    prompt: str,
    google_client=None,
    api_key: Optional[str] = None,
    aspect_ratio: str = "1:1",
) -> str:
    """Generate an image using Gemini - sync entry point for tool execution.

    Args:
        prompt: Text description of the image to generate
        google_client: Optional Google client (for extracting API key)
        api_key: Google API key (if not using client)
        aspect_ratio: Aspect ratio (1:1, 16:9, 9:16, 4:3, 3:4)

    Returns:
        Base64-encoded image data
    """
    key = _get_google_api_key(google_client, api_key)

    return generate_image_with_gemini_sync(
        prompt=prompt,
        api_key=key,
        aspect_ratio=aspect_ratio,
    )
