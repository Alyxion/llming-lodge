"""Tests for tool execution with auto-continue to text response.

These tests verify that after executing a tool, the model automatically
continues to formulate a text response using the tool results.
"""

import os
import pytest

from llming_lodge.messages import LlmSystemMessage, LlmHumanMessage, LlmAIMessage
from llming_lodge.tools.tool_call import ToolCallStatus
from llming_lodge.tools.llm_tool import LlmTool
from llming_lodge.tools.llm_toolbox import LlmToolbox


# --- Mock inventory tool (similar to MCP sample_server) ---

MOCK_PRODUCTS = {
    "P001": {"name": "Wireless Headphones", "price": 79.99, "stock": 150, "category": "Electronics"},
    "P002": {"name": "USB-C Cable", "price": 12.99, "stock": 500, "category": "Electronics"},
    "P003": {"name": "Coffee Mug", "price": 9.99, "stock": 75, "category": "Kitchen"},
}


def search_products(query: str) -> dict:
    """Search products by name or category."""
    results = []
    for pid, product in MOCK_PRODUCTS.items():
        if query.lower() in product["name"].lower() or query.lower() in product["category"].lower():
            results.append({"id": pid, **product})
    return {"products": results, "count": len(results)}


def get_product_details(product_id: str) -> dict:
    """Get details for a specific product."""
    if product_id in MOCK_PRODUCTS:
        return {"id": product_id, **MOCK_PRODUCTS[product_id]}
    return {"error": f"Product {product_id} not found"}


def check_inventory(product_id: str) -> dict:
    """Check inventory levels for a specific product."""
    if product_id in MOCK_PRODUCTS:
        p = MOCK_PRODUCTS[product_id]
        return {"product_id": product_id, "name": p["name"], "stock": p["stock"], "price": p["price"]}
    return {"error": f"Product {product_id} not found"}


# --- Fixtures ---

@pytest.fixture
def inventory_toolbox():
    """Create toolbox with inventory tools."""
    search_tool = LlmTool(
        name="search_products",
        description="Search products by name or category. Use this to find products.",
        func=search_products,
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query for product name or category"}
            },
            "required": ["query"]
        }
    )
    details_tool = LlmTool(
        name="get_product_details",
        description="Get detailed information about a specific product by ID.",
        func=get_product_details,
        parameters={
            "type": "object",
            "properties": {
                "product_id": {"type": "string", "description": "Product ID (e.g., P001)"}
            },
            "required": ["product_id"]
        }
    )
    inventory_tool = LlmTool(
        name="check_inventory",
        description="Check inventory stock levels for a specific product by ID.",
        func=check_inventory,
        parameters={
            "type": "object",
            "properties": {
                "product_id": {"type": "string", "description": "Product ID to check (e.g., P001)"}
            },
            "required": ["product_id"]
        }
    )
    return LlmToolbox(
        name="inventory_tools",
        description="Tools for searching products and checking inventory",
        tools=[search_tool, details_tool, inventory_tool]
    )


@pytest.fixture
def inventory_messages():
    """Messages that will trigger tool usage."""
    return [
        LlmSystemMessage(content="You are a helpful inventory assistant. Use the available tools to answer questions about products and stock."),
        LlmHumanMessage(content="How many headphones do we have in stock?"),
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


# --- OpenAI Tests ---

class TestOpenAIToolAutoContinue:
    """Test OpenAI tool execution with auto-continue to text response."""

    @pytest.mark.asyncio
    async def test_astream_tool_to_text_response(self, openai_api_key, inventory_toolbox, inventory_messages):
        """Test async streaming: tool call -> result -> text response."""
        from llming_lodge.providers.openai.openai_client import OpenAILlmClient

        client = OpenAILlmClient(
            api_key=openai_api_key,
            model=os.environ.get("OPENAI_MODEL", "gpt-5-nano"),
            temperature=0.0,
            max_tokens=1024,
            toolboxes=[inventory_toolbox]
        )

        text_content = ""
        tool_calls = []

        async for chunk in client.astream(inventory_messages):
            if chunk.tool_call:
                tool_calls.append(chunk.tool_call)
            if chunk.content:
                text_content += chunk.content

        # Should have at least one tool call
        assert len(tool_calls) > 0, "Expected at least one tool call"

        # Should have both PENDING and COMPLETED status
        statuses = [tc.status for tc in tool_calls]
        assert ToolCallStatus.PENDING in statuses, "Expected PENDING tool call status"
        assert ToolCallStatus.COMPLETED in statuses, "Expected COMPLETED tool call status"

        # Find the completed tool call
        completed = [tc for tc in tool_calls if tc.status == ToolCallStatus.COMPLETED]
        assert len(completed) > 0, "Expected at least one completed tool call"
        assert completed[0].result is not None, "Expected tool result"

        # Model should have formulated a text response
        assert len(text_content) > 0, "Expected text response after tool execution"

        # Text should contain relevant info from tool results (headphones, stock number)
        text_lower = text_content.lower()
        assert "headphone" in text_lower or "150" in text_content or "stock" in text_lower, \
            f"Expected response to mention headphones or stock. Got: {text_content}"

    @pytest.mark.asyncio
    async def test_ainvoke_tool_to_text_response(self, openai_api_key, inventory_toolbox, inventory_messages):
        """Test async invoke: tool call -> result -> text response."""
        from llming_lodge.providers.openai.openai_client import OpenAILlmClient

        client = OpenAILlmClient(
            api_key=openai_api_key,
            model=os.environ.get("OPENAI_MODEL", "gpt-5-nano"),
            temperature=0.0,
            max_tokens=1024,
            toolboxes=[inventory_toolbox]
        )

        result = await client.ainvoke(inventory_messages)

        assert isinstance(result, LlmAIMessage)
        assert result.content, "Expected non-empty response"

        # Response should mention the tool result data
        text_lower = result.content.lower()
        assert "headphone" in text_lower or "150" in result.content or "stock" in text_lower, \
            f"Expected response about headphones/stock. Got: {result.content}"

    def test_invoke_tool_to_text_response(self, openai_api_key, inventory_toolbox, inventory_messages):
        """Test sync invoke: tool call -> result -> text response."""
        from llming_lodge.providers.openai.openai_client import OpenAILlmClient

        client = OpenAILlmClient(
            api_key=openai_api_key,
            model=os.environ.get("OPENAI_MODEL", "gpt-5-nano"),
            temperature=0.0,
            max_tokens=1024,
            toolboxes=[inventory_toolbox]
        )

        result = client.invoke(inventory_messages)

        assert isinstance(result, LlmAIMessage)
        assert result.content, "Expected non-empty response"



# --- Anthropic Tests ---

class TestAnthropicToolAutoContinue:
    """Test Anthropic tool execution with auto-continue to text response."""

    @pytest.mark.asyncio
    async def test_astream_tool_to_text_response(self, anthropic_api_key, inventory_toolbox, inventory_messages):
        """Test async streaming: tool call -> result -> text response."""
        from llming_lodge.providers.anthropic.anthropic_client import AnthropicClient

        client = AnthropicClient(
            api_key=anthropic_api_key,
            model="claude-sonnet-4-20250514",
            temperature=0.0,
            max_tokens=256,
            toolboxes=[inventory_toolbox]
        )

        text_content = ""
        tool_calls = []

        async for chunk in client.astream(inventory_messages):
            if chunk.tool_call:
                tool_calls.append(chunk.tool_call)
            if chunk.content:
                text_content += chunk.content

        # Should have at least one tool call
        assert len(tool_calls) > 0, "Expected at least one tool call"

        # Should have both PENDING and COMPLETED status
        statuses = [tc.status for tc in tool_calls]
        assert ToolCallStatus.PENDING in statuses, "Expected PENDING tool call status"
        assert ToolCallStatus.COMPLETED in statuses, "Expected COMPLETED tool call status"

        # Find the completed tool call
        completed = [tc for tc in tool_calls if tc.status == ToolCallStatus.COMPLETED]
        assert len(completed) > 0, "Expected at least one completed tool call"
        assert completed[0].result is not None, "Expected tool result"

        # Model should have formulated a text response
        assert len(text_content) > 0, "Expected text response after tool execution"

        # Text should contain relevant info from tool results
        text_lower = text_content.lower()
        assert "headphone" in text_lower or "150" in text_content or "stock" in text_lower, \
            f"Expected response to mention headphones or stock. Got: {text_content}"

    @pytest.mark.asyncio
    async def test_ainvoke_tool_to_text_response(self, anthropic_api_key, inventory_toolbox, inventory_messages):
        """Test async invoke: tool call -> result -> text response."""
        from llming_lodge.providers.anthropic.anthropic_client import AnthropicClient

        client = AnthropicClient(
            api_key=anthropic_api_key,
            model="claude-sonnet-4-20250514",
            temperature=0.0,
            max_tokens=256,
            toolboxes=[inventory_toolbox]
        )

        result = await client.ainvoke(inventory_messages)

        assert isinstance(result, LlmAIMessage)
        assert result.content, "Expected non-empty response"

        # Response should mention the tool result data
        text_lower = result.content.lower()
        assert "headphone" in text_lower or "150" in result.content or "stock" in text_lower, \
            f"Expected response about headphones/stock. Got: {result.content}"

    def test_invoke_tool_to_text_response(self, anthropic_api_key, inventory_toolbox, inventory_messages):
        """Test sync invoke: tool call -> result -> text response."""
        from llming_lodge.providers.anthropic.anthropic_client import AnthropicClient

        client = AnthropicClient(
            api_key=anthropic_api_key,
            model="claude-sonnet-4-20250514",
            temperature=0.0,
            max_tokens=256,
            toolboxes=[inventory_toolbox]
        )

        result = client.invoke(inventory_messages)

        assert isinstance(result, LlmAIMessage)
        assert result.content, "Expected non-empty response"


# --- Multi-tool Tests ---

class TestMultiToolExecution:
    """Test scenarios requiring multiple tool calls."""

    @pytest.mark.asyncio
    async def test_openai_multiple_tools(self, openai_api_key, inventory_toolbox):
        """Test OpenAI calling multiple tools in sequence."""
        from llming_lodge.providers.openai.openai_client import OpenAILlmClient

        client = OpenAILlmClient(
            api_key=openai_api_key,
            model=os.environ.get("OPENAI_MODEL", "gpt-5-nano"),
            temperature=0.0,
            max_tokens=1024,
            toolboxes=[inventory_toolbox]
        )

        messages = [
            LlmSystemMessage(content="You are an inventory assistant. Use tools to answer questions."),
            LlmHumanMessage(content="Search for headphones and tell me the stock level and price."),
        ]

        text_content = ""
        tool_names_used = set()

        async for chunk in client.astream(messages):
            if chunk.tool_call and chunk.tool_call.status == ToolCallStatus.COMPLETED:
                tool_names_used.add(chunk.tool_call.name)
            if chunk.content:
                text_content += chunk.content

        # Should have used at least one tool
        assert len(tool_names_used) >= 1, f"Expected tool usage. Tools used: {tool_names_used}"

        # Should have text response
        assert len(text_content) > 0, "Expected text response"

    @pytest.mark.asyncio
    async def test_anthropic_multiple_tools(self, anthropic_api_key, inventory_toolbox):
        """Test Anthropic calling multiple tools in sequence."""
        from llming_lodge.providers.anthropic.anthropic_client import AnthropicClient

        client = AnthropicClient(
            api_key=anthropic_api_key,
            model="claude-sonnet-4-20250514",
            temperature=0.0,
            max_tokens=512,
            toolboxes=[inventory_toolbox]
        )

        messages = [
            LlmSystemMessage(content="You are an inventory assistant. Use tools to answer questions."),
            LlmHumanMessage(content="Search for headphones and tell me the stock level and price."),
        ]

        text_content = ""
        tool_names_used = set()

        async for chunk in client.astream(messages):
            if chunk.tool_call and chunk.tool_call.status == ToolCallStatus.COMPLETED:
                tool_names_used.add(chunk.tool_call.name)
            if chunk.content:
                text_content += chunk.content

        # Should have used at least one tool
        assert len(tool_names_used) >= 1, f"Expected tool usage. Tools used: {tool_names_used}"

        # Should have text response
        assert len(text_content) > 0, "Expected text response"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
