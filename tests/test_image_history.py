"""
Test that generated images are properly added to chat history
and can be referenced by the model in subsequent turns.
"""
import asyncio
import pytest
import os
import json

from llming_lodge.llm_provider_manager import LLMManager
from llming_lodge.session import ChatSession, LLMConfig


@pytest.fixture
def llm_manager():
    """Create an LLM manager with image generation enabled."""
    manager = LLMManager()
    return manager


@pytest.fixture
def openai_api_key():
    """Get OpenAI API key from environment."""
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        pytest.skip("OPENAI_API_KEY not set")
    return key


@pytest.mark.regression
@pytest.mark.asyncio
async def test_generated_image_in_history(llm_manager, openai_api_key):
    """
    Test that:
    1. An image can be generated via async chat
    2. The generated image is stored in chat history
    3. The model can reference the image in subsequent turns

    This matches the exact flow used in the UI.
    """
    # Create config with image generation tool enabled
    config = LLMConfig(
        provider="openai",
        model="gpt-5.2",  # Use a capable model with vision
        temperature=0.7,
        max_tokens=1024,
        tools=["generate_image"],  # Enable DALL-E image generation
    )

    # Create session - toolboxes are built internally based on config flags
    session = ChatSession(config=config)

    # Step 1: Generate an image of a skating cow
    print("\n=== Step 1: Generate image of skating cow ===")
    full_response = ""

    gen = await session.chat_async(
        "Generate an image of a skating cow",
        streaming=True
    )

    async for chunk in gen:
        full_response += chunk.content
        # Print progress
        if chunk.content and not chunk.content.startswith('{'):
            print(chunk.content, end='', flush=True)

    print(f"\n\nFull response length: {len(full_response)}")

    # Verify image was generated (response contains function_call_result with base64)
    assert "function_call_result" in full_response or "generate_image" in full_response, \
        "Expected image generation function call in response"

    # Check that history contains the image
    history = session.get_history()
    print(f"\n=== History has {len(history.messages)} messages ===")

    # Find assistant message with images
    assistant_msg_with_images = None
    for msg in history.messages:
        print(f"  - Role: {msg.role}, Has images: {bool(msg.images)}, Content preview: {msg.content[:100]}...")
        if msg.role.value == "assistant" and msg.images:
            assistant_msg_with_images = msg

    assert assistant_msg_with_images is not None, \
        "Expected assistant message with generated image in history"
    assert len(assistant_msg_with_images.images) > 0, \
        "Expected at least one image in assistant message"

    # Verify the image is valid base64 (should be long)
    image_data = assistant_msg_with_images.images[0]
    assert len(image_data) > 1000, \
        f"Expected substantial base64 image data, got {len(image_data)} chars"

    print(f"\n=== Image stored successfully ({len(image_data)} chars) ===")

    # Step 2: Ask to modify the image
    print("\n=== Step 2: Ask to make image black & white ===")
    full_response_2 = ""

    gen2 = await session.chat_async(
        "Make the image black & white",
        streaming=True
    )

    async for chunk in gen2:
        full_response_2 += chunk.content
        if chunk.content and not chunk.content.startswith('{'):
            print(chunk.content, end='', flush=True)

    print(f"\n\nSecond response length: {len(full_response_2)}")

    # The model should either:
    # 1. Generate a new black & white image (function_call_result)
    # 2. Acknowledge it can see the image and explain it can't modify (text response)
    # Either way proves the model can see the image in context

    # Check that the model didn't say "I don't see any image" or similar
    no_image_phrases = [
        "don't see any image",
        "no image",
        "haven't provided",
        "please upload",
        "please share",
        "can't see",
        "cannot see",
    ]

    response_lower = full_response_2.lower()
    for phrase in no_image_phrases:
        assert phrase not in response_lower, \
            f"Model doesn't seem to see the image - found '{phrase}' in response"

    print("\n=== SUCCESS: Model can reference previously generated image ===")

    # Final history check
    final_history = session.get_history()
    print(f"\n=== Final history has {len(final_history.messages)} messages ===")
    for msg in final_history.messages:
        has_img = "📷" if msg.images else ""
        print(f"  {has_img} {msg.role.value}: {msg.content[:80]}...")


@pytest.mark.asyncio
async def test_image_extraction_from_response():
    """Test the _extract_generated_images helper method."""
    from llming_lodge.session import ChatSession, LLMConfig

    config = LLMConfig(provider="openai", model="gpt-5-nano")
    session = ChatSession(config=config)

    # Test with a mock response containing image data (minimum 100+ chars)
    # Use a longer base64 string to pass the validation threshold
    long_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk" + "A" * 100
    mock_response = f'{{"function_pending": true, "function": "generate_image", "call_id": "call_123"}}{{"function_call_result": "{long_base64}", "call_id": "call_123", "function": "generate_image"}}'

    images = session._extract_generated_images(mock_response)

    assert len(images) == 1, f"Expected 1 image, got {len(images)}"
    assert images[0].startswith("iVBOR"), "Expected PNG base64 data"

    print(f"Extracted {len(images)} image(s) from mock response")


@pytest.mark.asyncio
async def test_clean_response_for_history():
    """Test the _clean_response_for_history helper method."""
    from llming_lodge.session import ChatSession, LLMConfig

    config = LLMConfig(provider="openai", model="gpt-5-nano")
    session = ChatSession(config=config)

    # Test with a response containing large base64 data
    large_base64 = "A" * 2000  # Simulate large image data
    mock_response = f'{{"function_call_result": "{large_base64}", "call_id": "call_123", "function": "generate_image"}}'

    cleaned = session._clean_response_for_history(mock_response)

    assert "[IMAGE_GENERATED]" in cleaned, "Expected placeholder in cleaned response"
    assert large_base64 not in cleaned, "Expected base64 data to be removed"
    assert len(cleaned) < len(mock_response), "Cleaned response should be shorter"

    print(f"Cleaned response: {len(mock_response)} -> {len(cleaned)} chars")


if __name__ == "__main__":
    # Run with: poetry run pytest tests/test_image_history.py -v -s
    asyncio.run(test_generated_image_in_history(LLMManager(), os.environ.get("OPENAI_API_KEY")))
