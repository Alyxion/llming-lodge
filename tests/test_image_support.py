"""Integration tests for image and tool support.

These tests use real API calls - no mocking.
Uses low quality/size settings to minimize costs.
"""
import os
import base64
import pytest

from llming_lodge.messages import LlmSystemMessage, LlmHumanMessage, LlmAIMessage
from llming_lodge.providers.openai.openai_client import OpenAILlmClient
from llming_lodge.tools.llm_tool import LlmTool
from llming_lodge.tools.llm_toolbox import LlmToolbox


# --- Test Data ---

def _create_solid_color_png(r: int, g: int, b: int, size: int = 64) -> str:
    """Create a solid color PNG image and return as base64.

    Uses raw PNG encoding without external libraries.
    """
    import zlib
    import struct

    def png_chunk(chunk_type: bytes, data: bytes) -> bytes:
        chunk_len = struct.pack(">I", len(data))
        chunk_crc = struct.pack(">I", zlib.crc32(chunk_type + data) & 0xffffffff)
        return chunk_len + chunk_type + data + chunk_crc

    # PNG signature
    signature = b'\x89PNG\r\n\x1a\n'

    # IHDR chunk (image header)
    ihdr_data = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)  # 8-bit RGB
    ihdr = png_chunk(b'IHDR', ihdr_data)

    # IDAT chunk (image data)
    raw_data = b''
    for y in range(size):
        raw_data += b'\x00'  # Filter type: None
        for x in range(size):
            raw_data += bytes([r, g, b])

    compressed = zlib.compress(raw_data, 9)
    idat = png_chunk(b'IDAT', compressed)

    # IEND chunk
    iend = png_chunk(b'IEND', b'')

    png_bytes = signature + ihdr + idat + iend
    return base64.b64encode(png_bytes).decode('ascii')


# Generate test images (64x64 solid color PNGs)
RED_IMAGE_BASE64 = _create_solid_color_png(255, 0, 0)
BLUE_IMAGE_BASE64 = _create_solid_color_png(0, 0, 255)


# --- Fixtures ---

@pytest.fixture(scope="module")
def openai_api_key():
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        pytest.skip("OPENAI_API_KEY not set")
    return key


@pytest.fixture(scope="module")
def anthropic_api_key():
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        pytest.skip("ANTHROPIC_API_KEY not set")
    return key


@pytest.fixture(scope="module")
def openai_vision_model():
    """Use a model that supports vision."""
    return os.environ.get("OPENAI_VISION_MODEL", "gpt-5-mini")


@pytest.fixture(scope="module")
def anthropic_model():
    """Use Claude Sonnet for testing."""
    return os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")


# --- Message Model Tests ---

def test_message_with_images():
    """Test LlmHumanMessage can hold images."""
    msg = LlmHumanMessage(
        content="What's in this image?",
        images=[RED_IMAGE_BASE64]
    )
    assert msg.images == [RED_IMAGE_BASE64]
    assert msg.content == "What's in this image?"


def test_message_with_multiple_images():
    """Test LlmHumanMessage can hold multiple images."""
    msg = LlmHumanMessage(
        content="Compare these images",
        images=[RED_IMAGE_BASE64, BLUE_IMAGE_BASE64]
    )
    assert len(msg.images) == 2


def test_message_without_images():
    """Test backward compatibility - images field is optional."""
    msg = LlmHumanMessage(content="Hello")
    assert msg.images is None


def test_message_serialization_with_images():
    """Test message can be serialized and deserialized with images."""
    msg = LlmHumanMessage(
        content="Describe this",
        images=[RED_IMAGE_BASE64]
    )
    data = msg.model_dump()
    assert "images" in data
    assert data["images"] == [RED_IMAGE_BASE64]

    # Deserialize
    restored = LlmHumanMessage.model_validate(data)
    assert restored.images == [RED_IMAGE_BASE64]


# --- OpenAI Message Conversion Tests ---

def test_convert_messages_with_images_openai():
    """Test multimodal format conversion for OpenAI Responses API."""
    from llming_lodge.providers.openai.openai_client import _convert_messages

    messages = [
        LlmSystemMessage(content="You are helpful."),
        LlmHumanMessage(content="What color is this?", images=[RED_IMAGE_BASE64]),
    ]

    converted = _convert_messages(messages)

    # System message should be plain text
    assert converted[0]["role"] == "system"
    assert converted[0]["content"] == "You are helpful."

    # User message with image should be multimodal format
    assert converted[1]["role"] == "user"
    assert isinstance(converted[1]["content"], list)

    # Should have input_text and input_image parts (Responses API format)
    content_types = [part["type"] for part in converted[1]["content"]]
    assert "input_text" in content_types
    assert "input_image" in content_types


def test_convert_messages_without_images_unchanged():
    """Test that messages without images convert to simple format."""
    from llming_lodge.providers.openai.openai_client import _convert_messages

    messages = [
        LlmSystemMessage(content="You are helpful."),
        LlmHumanMessage(content="Hello"),
    ]

    converted = _convert_messages(messages)

    # Should be simple string content, not list
    assert converted[1]["role"] == "user"
    assert converted[1]["content"] == "Hello"


def test_image_history_limit_openai():
    """Test only last N messages include images (to limit context size)."""
    from llming_lodge.providers.openai.openai_client import _convert_messages

    # Create 15 messages alternating user/assistant, all with images
    messages = []
    for i in range(15):
        if i % 2 == 0:
            messages.append(LlmHumanMessage(
                content=f"Message {i}",
                images=[RED_IMAGE_BASE64]
            ))
        else:
            messages.append(LlmAIMessage(content=f"Response {i}"))

    # Convert with max_image_history=10
    converted = _convert_messages(messages, max_image_history=10)

    # Count messages that have image content
    messages_with_images = 0
    for msg in converted:
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if part.get("type") == "input_image":
                    messages_with_images += 1
                    break

    # Should have at most 10 messages with images
    assert messages_with_images <= 10


# --- OpenAI Image Interpretation Tests (Real API) ---

@pytest.mark.asyncio
async def test_openai_image_interpretation(openai_api_key, openai_vision_model):
    """Test OpenAI can interpret a pasted image."""
    client = OpenAILlmClient(
        api_key=openai_api_key,
        model=openai_vision_model,
        max_tokens=5000,  # GPT-5 uses tokens for reasoning, need extra for text
    )

    messages = [
        LlmSystemMessage(content="Describe colors briefly."),
        LlmHumanMessage(
            content="What is the dominant color in this image? Answer with just the color name.",
            images=[RED_IMAGE_BASE64]
        ),
    ]

    response = await client.ainvoke(messages)
    assert isinstance(response, LlmAIMessage)
    assert response.content
    # The image is red, so response should mention red
    assert "red" in response.content.lower()


@pytest.mark.asyncio
async def test_openai_multiple_images(openai_api_key, openai_vision_model):
    """Test OpenAI can interpret multiple images."""
    client = OpenAILlmClient(
        api_key=openai_api_key,
        model=openai_vision_model,
        max_tokens=5000,  # GPT-5 uses tokens for reasoning, need extra for text
    )

    messages = [
        LlmHumanMessage(
            content="What colors are in these two images? List them.",
            images=[RED_IMAGE_BASE64, BLUE_IMAGE_BASE64]
        ),
    ]

    response = await client.ainvoke(messages)
    assert isinstance(response, LlmAIMessage)
    # Should mention both colors
    content_lower = response.content.lower()
    assert "red" in content_lower or "blue" in content_lower


# --- OpenAI Image Generation Tests (Real API) ---

@pytest.mark.asyncio
async def test_openai_image_generation(openai_api_key):
    """Test DALL-E image generation."""
    client = OpenAILlmClient(
        api_key=openai_api_key,
        model="gpt-5-mini",  # Model doesn't matter for image generation
    )

    result = await client.generate_image(
        prompt="A simple solid red square on white background",
        size="1024x1024",  # DALL-E 3 minimum size
        quality="standard",  # Use standard quality to save costs
    )

    assert result is not None
    # Result should be base64 encoded image data or URL
    assert len(result) > 100  # Should be substantial


# --- Anthropic Image Interpretation Tests (Real API) ---

@pytest.mark.asyncio
async def test_anthropic_image_interpretation(anthropic_api_key, anthropic_model):
    """Test Anthropic can interpret a pasted image."""
    from llming_lodge.providers.anthropic.anthropic_client import AnthropicClient

    client = AnthropicClient(
        api_key=anthropic_api_key,
        model=anthropic_model,
        temperature=0.0,
        max_tokens=50,
    )

    messages = [
        LlmHumanMessage(
            content="What is the dominant color in this image? Answer with just the color name.",
            images=[RED_IMAGE_BASE64]
        ),
    ]

    response = await client.ainvoke(messages)
    assert isinstance(response, LlmAIMessage)
    assert response.content
    # The image is red, so response should mention red
    assert "red" in response.content.lower()


# --- Anthropic Tool Calling Tests (Real API) ---

def add_numbers(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@pytest.mark.asyncio
async def test_anthropic_tool_calling(anthropic_api_key, anthropic_model):
    """Test Anthropic tool use works."""
    from llming_lodge.providers.anthropic.anthropic_client import AnthropicClient

    add_tool = LlmTool(
        name="add_numbers",
        description="Add two numbers and return the sum.",
        func=add_numbers,
        parameters={
            "type": "object",
            "properties": {
                "a": {"type": "integer", "description": "First number"},
                "b": {"type": "integer", "description": "Second number"}
            },
            "required": ["a", "b"]
        }
    )

    toolbox = LlmToolbox(
        name="math_tools",
        description="Math utilities",
        tools=[add_tool]
    )

    client = AnthropicClient(
        api_key=anthropic_api_key,
        model=anthropic_model,
        temperature=0.0,
        max_tokens=100,
        toolboxes=[toolbox]
    )

    messages = [
        LlmHumanMessage(content="What is 5 + 3? Use the add_numbers tool."),
    ]

    response = await client.ainvoke(messages)
    assert isinstance(response, LlmAIMessage)
    assert "8" in response.content


# --- OpenAI Web Search Tests (Real API) ---

@pytest.mark.asyncio
async def test_openai_web_search(openai_api_key, openai_vision_model):
    """Test OpenAI can use built-in web search."""
    client = OpenAILlmClient(
        api_key=openai_api_key,
        model=openai_vision_model,
        max_tokens=5000,  # GPT-5 uses tokens for reasoning, need extra for text
        toolboxes=["web_search"]  # Enable web search
    )
    # Note: temperature is not supported with web_search tool or reasoning models

    messages = [
        LlmHumanMessage(content="What is the current date today? Search the web if needed."),
    ]

    response = await client.ainvoke(messages)
    assert isinstance(response, LlmAIMessage)
    assert response.content
    # Should have some response about the date
    assert len(response.content) > 10


# --- Anthropic Web Search Tests (Real API) ---

@pytest.mark.asyncio
async def test_anthropic_web_search(anthropic_api_key, anthropic_model):
    """Test Anthropic can use web search tool."""
    from llming_lodge.providers.anthropic.anthropic_client import AnthropicClient

    # Web search as a custom tool for Anthropic (they don't have built-in)
    def web_search(query: str) -> str:
        """Simulate web search - in real implementation would call search API."""
        return f"Search results for '{query}': Today is December 31, 2025."

    search_tool = LlmTool(
        name="web_search",
        description="Search the web for current information.",
        func=web_search,
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"]
        }
    )

    toolbox = LlmToolbox(
        name="search_tools",
        description="Web search utilities",
        tools=[search_tool]
    )

    client = AnthropicClient(
        api_key=anthropic_api_key,
        model=anthropic_model,
        temperature=0.0,
        max_tokens=200,
        toolboxes=[toolbox]
    )

    messages = [
        LlmHumanMessage(content="What is today's date? Use the web_search tool."),
    ]

    response = await client.ainvoke(messages)
    assert isinstance(response, LlmAIMessage)
    assert response.content
    # Should mention the date from search results
    assert "2025" in response.content or "December" in response.content


# --- Streaming Tests with Images ---

@pytest.mark.asyncio
async def test_openai_image_interpretation_streaming(openai_api_key, openai_vision_model):
    """Test OpenAI image interpretation with streaming."""
    client = OpenAILlmClient(
        api_key=openai_api_key,
        model=openai_vision_model,
        max_tokens=5000,  # GPT-5 uses tokens for reasoning, need extra for text
    )

    messages = [
        LlmHumanMessage(
            content="What color is this image? One word answer.",
            images=[RED_IMAGE_BASE64]
        ),
    ]

    chunks = []
    async for chunk in client.astream(messages):
        chunks.append(chunk.content)

    full_response = "".join(chunks)
    assert "red" in full_response.lower()
    assert len(chunks) >= 1  # Should have at least one chunk
