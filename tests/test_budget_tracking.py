"""Tests for budget/usage tracking with tool auto-continue.

These tests verify that token usage is properly tracked across multiple
API iterations when tools are executed.
"""

import os
import pytest

from llming_lodge.messages import LlmSystemMessage, LlmHumanMessage
from llming_lodge.tools.tool_call import ToolCallStatus
from llming_lodge.tools.llm_tool import LlmTool
from llming_lodge.tools.llm_toolbox import LlmToolbox


# --- Mock inventory tool ---

MOCK_PRODUCTS = {
    "P001": {"name": "Wireless Headphones", "price": 79.99, "stock": 150},
}


def search_products(query: str) -> dict:
    """Search products by name."""
    results = []
    for pid, product in MOCK_PRODUCTS.items():
        if query.lower() in product["name"].lower():
            results.append({"id": pid, **product})
    return {"products": results, "count": len(results)}


@pytest.fixture
def inventory_toolbox():
    """Create toolbox with inventory tool."""
    search_tool = LlmTool(
        name="search_products",
        description="Search products by name.",
        func=search_products,
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"]
        }
    )
    return LlmToolbox(
        name="inventory_tools",
        description="Product search tools",
        tools=[search_tool]
    )


@pytest.fixture
def test_messages():
    """Messages that will trigger tool usage."""
    return [
        LlmSystemMessage(content="You are an inventory assistant. Use tools to answer."),
        LlmHumanMessage(content="Search for headphones and tell me the stock."),
    ]


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


class TestOpenAIUsageTracking:
    """Test OpenAI usage tracking with tool auto-continue."""

    @pytest.mark.asyncio
    async def test_astream_tracks_total_usage(self, openai_api_key, inventory_toolbox, test_messages):
        """Test that total usage is tracked across tool iterations."""
        from llming_lodge.providers.openai.openai_client import OpenAILlmClient

        client = OpenAILlmClient(
            api_key=openai_api_key,
            model="gpt-5-nano",
            temperature=0.0,
            max_tokens=1024,
            toolboxes=[inventory_toolbox]
        )

        final_chunk = None
        async for chunk in client.astream(test_messages):
            if chunk.is_final:
                final_chunk = chunk

        assert final_chunk is not None, "Expected final chunk"
        assert 'total_input_tokens' in final_chunk.response_metadata, "Expected total_input_tokens"
        assert 'total_output_tokens' in final_chunk.response_metadata, "Expected total_output_tokens"

        total_input = final_chunk.response_metadata['total_input_tokens']
        total_output = final_chunk.response_metadata['total_output_tokens']

        # Should have consumed some tokens
        assert total_input > 0, f"Expected input tokens > 0, got {total_input}"
        assert total_output > 0, f"Expected output tokens > 0, got {total_output}"

        print(f"OpenAI total usage: {total_input} input, {total_output} output")

    @pytest.mark.asyncio
    async def test_astream_usage_callback(self, openai_api_key, inventory_toolbox, test_messages):
        """Test that usage callback is called for each API iteration."""
        from llming_lodge.providers.openai.openai_client import OpenAILlmClient

        client = OpenAILlmClient(
            api_key=openai_api_key,
            model="gpt-5-nano",
            temperature=0.0,
            max_tokens=1024,
            toolboxes=[inventory_toolbox]
        )

        callback_calls = []
        def usage_callback(input_tokens: int, output_tokens: int):
            callback_calls.append((input_tokens, output_tokens))

        final_chunk = None
        async for chunk in client.astream(test_messages, usage_callback=usage_callback):
            if chunk.is_final:
                final_chunk = chunk

        # Should have at least one callback call (could be 2+ if tools were used)
        assert len(callback_calls) >= 1, f"Expected at least 1 callback call, got {len(callback_calls)}"

        # Sum of callbacks should match total
        total_input = sum(c[0] for c in callback_calls)
        total_output = sum(c[1] for c in callback_calls)

        assert final_chunk is not None
        assert total_input == final_chunk.response_metadata.get('total_input_tokens', 0), \
            "Callback sum should match total"
        assert total_output == final_chunk.response_metadata.get('total_output_tokens', 0), \
            "Callback sum should match total"

        print(f"OpenAI callback calls: {callback_calls}")

    @pytest.mark.asyncio
    async def test_astream_multi_iteration_usage(self, openai_api_key, inventory_toolbox, test_messages):
        """Test usage tracking when tool triggers multiple API iterations."""
        from llming_lodge.providers.openai.openai_client import OpenAILlmClient

        client = OpenAILlmClient(
            api_key=openai_api_key,
            model="gpt-5-nano",
            temperature=0.0,
            max_tokens=1024,
            toolboxes=[inventory_toolbox]
        )

        callback_calls = []
        tool_calls_seen = []

        def usage_callback(input_tokens: int, output_tokens: int):
            callback_calls.append((input_tokens, output_tokens))

        async for chunk in client.astream(test_messages, usage_callback=usage_callback):
            if chunk.tool_call and chunk.tool_call.status == ToolCallStatus.COMPLETED:
                tool_calls_seen.append(chunk.tool_call.name)

        # If tools were called, we should have multiple iterations
        if tool_calls_seen:
            # With tool execution, we expect at least 2 iterations:
            # 1. First call where model decides to use tool
            # 2. Second call with tool result to get final answer
            assert len(callback_calls) >= 2, \
                f"Expected >= 2 iterations with tools, got {len(callback_calls)}"
            print(f"OpenAI multi-iteration: {len(callback_calls)} iterations, tools: {tool_calls_seen}")


class TestAnthropicUsageTracking:
    """Test Anthropic usage tracking with tool auto-continue."""

    @pytest.mark.asyncio
    async def test_astream_tracks_total_usage(self, anthropic_api_key, inventory_toolbox, test_messages):
        """Test that total usage is tracked across tool iterations."""
        from llming_lodge.providers.anthropic.anthropic_client import AnthropicClient

        client = AnthropicClient(
            api_key=anthropic_api_key,
            model="claude-haiku-4-5-20251001",
            temperature=0.0,
            max_tokens=1024,
            toolboxes=[inventory_toolbox]
        )

        final_chunk = None
        async for chunk in client.astream(test_messages):
            if chunk.is_final:
                final_chunk = chunk

        assert final_chunk is not None, "Expected final chunk"
        assert 'total_input_tokens' in final_chunk.response_metadata, "Expected total_input_tokens"
        assert 'total_output_tokens' in final_chunk.response_metadata, "Expected total_output_tokens"

        total_input = final_chunk.response_metadata['total_input_tokens']
        total_output = final_chunk.response_metadata['total_output_tokens']

        # Should have consumed some tokens
        assert total_input > 0, f"Expected input tokens > 0, got {total_input}"
        assert total_output > 0, f"Expected output tokens > 0, got {total_output}"

        print(f"Anthropic total usage: {total_input} input, {total_output} output")

    @pytest.mark.asyncio
    async def test_astream_usage_callback(self, anthropic_api_key, inventory_toolbox, test_messages):
        """Test that usage callback is called for each API iteration."""
        from llming_lodge.providers.anthropic.anthropic_client import AnthropicClient

        client = AnthropicClient(
            api_key=anthropic_api_key,
            model="claude-haiku-4-5-20251001",
            temperature=0.0,
            max_tokens=1024,
            toolboxes=[inventory_toolbox]
        )

        callback_calls = []
        def usage_callback(input_tokens: int, output_tokens: int):
            callback_calls.append((input_tokens, output_tokens))

        final_chunk = None
        async for chunk in client.astream(test_messages, usage_callback=usage_callback):
            if chunk.is_final:
                final_chunk = chunk

        # Should have at least one callback call
        assert len(callback_calls) >= 1, f"Expected at least 1 callback call, got {len(callback_calls)}"

        # Sum of callbacks should match total
        total_input = sum(c[0] for c in callback_calls)
        total_output = sum(c[1] for c in callback_calls)

        assert final_chunk is not None
        assert total_input == final_chunk.response_metadata.get('total_input_tokens', 0), \
            "Callback sum should match total"
        assert total_output == final_chunk.response_metadata.get('total_output_tokens', 0), \
            "Callback sum should match total"

        print(f"Anthropic callback calls: {callback_calls}")

    @pytest.mark.asyncio
    async def test_astream_multi_iteration_usage(self, anthropic_api_key, inventory_toolbox, test_messages):
        """Test usage tracking when tool triggers multiple API iterations."""
        from llming_lodge.providers.anthropic.anthropic_client import AnthropicClient

        client = AnthropicClient(
            api_key=anthropic_api_key,
            model="claude-haiku-4-5-20251001",
            temperature=0.0,
            max_tokens=1024,
            toolboxes=[inventory_toolbox]
        )

        callback_calls = []
        tool_calls_seen = []

        def usage_callback(input_tokens: int, output_tokens: int):
            callback_calls.append((input_tokens, output_tokens))

        async for chunk in client.astream(test_messages, usage_callback=usage_callback):
            if chunk.tool_call and chunk.tool_call.status == ToolCallStatus.COMPLETED:
                tool_calls_seen.append(chunk.tool_call.name)

        # If tools were called, we should have multiple iterations
        if tool_calls_seen:
            # With tool execution, we expect at least 2 iterations
            assert len(callback_calls) >= 2, \
                f"Expected >= 2 iterations with tools, got {len(callback_calls)}"
            print(f"Anthropic multi-iteration: {len(callback_calls)} iterations, tools: {tool_calls_seen}")


class TestUsageWithoutTools:
    """Test usage tracking without tool execution."""

    @pytest.mark.asyncio
    async def test_openai_simple_chat_usage(self, openai_api_key):
        """Test OpenAI usage tracking for simple chat (no tools)."""
        from llming_lodge.providers.openai.openai_client import OpenAILlmClient

        client = OpenAILlmClient(
            api_key=openai_api_key,
            model="gpt-5-nano",
            temperature=0.0,
            max_tokens=100,
        )

        messages = [
            LlmSystemMessage(content="You are helpful."),
            LlmHumanMessage(content="Say hello in one word."),
        ]

        callback_calls = []
        def usage_callback(input_tokens: int, output_tokens: int):
            callback_calls.append((input_tokens, output_tokens))

        final_chunk = None
        async for chunk in client.astream(messages, usage_callback=usage_callback):
            if chunk.is_final:
                final_chunk = chunk

        # Should have exactly 1 iteration (no tools)
        assert len(callback_calls) == 1, f"Expected 1 callback, got {len(callback_calls)}"
        assert final_chunk is not None
        assert final_chunk.response_metadata.get('total_input_tokens', 0) > 0

        print(f"OpenAI simple chat: {callback_calls[0]}")

    @pytest.mark.asyncio
    async def test_anthropic_simple_chat_usage(self, anthropic_api_key):
        """Test Anthropic usage tracking for simple chat (no tools)."""
        from llming_lodge.providers.anthropic.anthropic_client import AnthropicClient

        client = AnthropicClient(
            api_key=anthropic_api_key,
            model="claude-haiku-4-5-20251001",
            temperature=0.0,
            max_tokens=100,
        )

        messages = [
            LlmSystemMessage(content="You are helpful."),
            LlmHumanMessage(content="Say hello in one word."),
        ]

        callback_calls = []
        def usage_callback(input_tokens: int, output_tokens: int):
            callback_calls.append((input_tokens, output_tokens))

        final_chunk = None
        async for chunk in client.astream(messages, usage_callback=usage_callback):
            if chunk.is_final:
                final_chunk = chunk

        # Should have exactly 1 iteration (no tools)
        assert len(callback_calls) == 1, f"Expected 1 callback, got {len(callback_calls)}"
        assert final_chunk is not None
        assert final_chunk.response_metadata.get('total_input_tokens', 0) > 0

        print(f"Anthropic simple chat: {callback_calls[0]}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
