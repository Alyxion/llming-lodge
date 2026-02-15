"""
Image utilities for LLM providers.

This module provides shared image handling utilities used by LLM provider clients.

Key functions:
- is_likely_image_data: Detect if a string contains base64 image data
"""

import logging

logger = logging.getLogger(__name__)


def is_likely_image_data(value: str) -> bool:
    """
    Check if a string looks like base64 image data.

    Args:
        value: String to check

    Returns:
        True if the string appears to be base64-encoded image data
    """
    if not isinstance(value, str):
        return False
    if len(value) < 1000:  # Too small to be a meaningful image
        return False

    # Check for data URI
    if value.startswith("data:image/"):
        return True

    # Check for raw base64 that looks like PNG or JPEG
    # PNG magic bytes (89 50 4E 47) in base64 start with "iVBOR"
    if value.startswith("iVBOR"):
        return True
    # JPEG magic bytes (FF D8 FF) in base64 start with "/9j/"
    if value.startswith("/9j/"):
        return True

    return False


def sniff_image_mime(value: str) -> str:
    """Sniff MIME type from raw base64 or data URI string.

    Returns a full data URI if the input is raw base64,
    or the input unchanged if it already has a data: prefix.
    """
    if value.startswith("data:"):
        return value
    # Magic byte prefixes in base64
    if value.startswith("iVBOR"):
        return f"data:image/png;base64,{value}"
    if value.startswith("/9j/"):
        return f"data:image/jpeg;base64,{value}"
    if value.startswith("R0lG"):
        return f"data:image/gif;base64,{value}"
    if value.startswith("UklG"):
        return f"data:image/webp;base64,{value}"
    # Fallback — let the browser figure it out
    return f"data:application/octet-stream;base64,{value}"
