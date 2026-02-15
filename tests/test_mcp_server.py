"""Tests for sample MCP server and MCP integration."""

import pytest
import pytest_asyncio

from llming_lodge.tools.mcp.test_client import MCPTestClient


@pytest_asyncio.fixture
async def mcp_client() -> MCPTestClient:
    """Create and start MCP test client."""
    client = MCPTestClient(server_module="llming_lodge.tools.mcp.sample_server")
    await client.start()
    try:
        yield client
    finally:
        await client.stop()


def _has_tool(tools: list[dict], name: str) -> bool:
    """Check if a tool exists in the list."""
    return any(t.get("name") == name for t in tools)


class TestMCPServerTools:
    """Tests for MCP server tool listing."""

    @pytest.mark.asyncio
    async def test_list_tools_returns_expected_tools(self, mcp_client: MCPTestClient):
        """Test that server lists all expected tools."""
        tools = await mcp_client.list_tools()
        assert tools, "Expected at least one MCP tool"

        expected = {
            "search_products",
            "get_product_details",
            "list_categories",
            "get_customer",
            "check_inventory",
        }

        missing = sorted([name for name in expected if not _has_tool(tools, name)])
        assert not missing, f"Missing tools: {missing}"

    @pytest.mark.asyncio
    async def test_tool_schemas_are_valid(self, mcp_client: MCPTestClient):
        """Test that all tools have valid JSON schemas."""
        tools = await mcp_client.list_tools()

        for tool in tools:
            assert "name" in tool, "Tool missing name"
            assert "description" in tool, f"Tool {tool['name']} missing description"
            assert "inputSchema" in tool, f"Tool {tool['name']} missing inputSchema"

            schema = tool["inputSchema"]
            assert schema.get("type") == "object", f"Tool {tool['name']} schema type should be 'object'"
            assert "properties" in schema, f"Tool {tool['name']} schema missing properties"


class TestSearchProducts:
    """Tests for search_products tool."""

    @pytest.mark.asyncio
    async def test_search_by_name(self, mcp_client: MCPTestClient):
        """Test searching products by name."""
        result = await mcp_client.call_tool_text("search_products", {"query": "headphones"})
        assert "Wireless Headphones" in result
        assert "P001" in result

    @pytest.mark.asyncio
    async def test_search_by_category(self, mcp_client: MCPTestClient):
        """Test searching products by category."""
        result = await mcp_client.call_tool_text("search_products", {
            "query": "electronics",
            "category": "Electronics"
        })
        assert "Electronics" in result

    @pytest.mark.asyncio
    async def test_search_no_results(self, mcp_client: MCPTestClient):
        """Test search with no matching products."""
        result = await mcp_client.call_tool_text("search_products", {"query": "nonexistent"})
        assert "No products found" in result

    @pytest.mark.asyncio
    async def test_search_max_results(self, mcp_client: MCPTestClient):
        """Test search with max_results limit."""
        result = await mcp_client.call_tool_text("search_products", {
            "query": "",  # Match all
            "max_results": 2
        })
        # Should limit results
        lines = [l for l in result.split("\n") if l.startswith("- P")]
        assert len(lines) <= 2


class TestGetProductDetails:
    """Tests for get_product_details tool."""

    @pytest.mark.asyncio
    async def test_get_existing_product(self, mcp_client: MCPTestClient):
        """Test getting details of an existing product."""
        result = await mcp_client.call_tool_text("get_product_details", {"product_id": "P001"})
        assert "P001" in result
        assert "Wireless Headphones" in result
        assert "Electronics" in result
        assert "$79.99" in result

    @pytest.mark.asyncio
    async def test_get_nonexistent_product(self, mcp_client: MCPTestClient):
        """Test getting details of a nonexistent product."""
        result = await mcp_client.call_tool_text("get_product_details", {"product_id": "P999"})
        assert "not found" in result


class TestListCategories:
    """Tests for list_categories tool."""

    @pytest.mark.asyncio
    async def test_list_all_categories(self, mcp_client: MCPTestClient):
        """Test listing all categories."""
        result = await mcp_client.call_tool_text("list_categories", {})
        assert "Electronics" in result
        assert "Office" in result
        assert "Kitchen" in result
        assert "products" in result


class TestGetCustomer:
    """Tests for get_customer tool."""

    @pytest.mark.asyncio
    async def test_get_customer_by_id(self, mcp_client: MCPTestClient):
        """Test getting customer by ID."""
        result = await mcp_client.call_tool_text("get_customer", {"customer_id": "C001"})
        assert "John Smith" in result
        assert "john@example.com" in result
        assert "Gold" in result

    @pytest.mark.asyncio
    async def test_get_customer_by_email(self, mcp_client: MCPTestClient):
        """Test getting customer by email."""
        result = await mcp_client.call_tool_text("get_customer", {"email": "jane@example.com"})
        assert "Jane Doe" in result
        assert "Silver" in result

    @pytest.mark.asyncio
    async def test_get_nonexistent_customer(self, mcp_client: MCPTestClient):
        """Test getting a nonexistent customer."""
        result = await mcp_client.call_tool_text("get_customer", {"customer_id": "C999"})
        assert "not found" in result


class TestCheckInventory:
    """Tests for check_inventory tool."""

    @pytest.mark.asyncio
    async def test_check_all_inventory(self, mcp_client: MCPTestClient):
        """Test checking all inventory."""
        result = await mcp_client.call_tool_text("check_inventory", {})
        assert "Inventory Report" in result
        # Should show some products
        assert "P00" in result

    @pytest.mark.asyncio
    async def test_check_low_stock_only(self, mcp_client: MCPTestClient):
        """Test checking only low stock items."""
        result = await mcp_client.call_tool_text("check_inventory", {"low_stock_only": True})
        # Low stock items have stock < 100
        assert "LOW" in result

    @pytest.mark.asyncio
    async def test_check_inventory_by_category(self, mcp_client: MCPTestClient):
        """Test checking inventory filtered by category."""
        result = await mcp_client.call_tool_text("check_inventory", {"category": "Electronics"})
        assert "Inventory Report" in result
        # Should only show electronics
        assert "Headphones" in result or "Keyboard" in result or "Cable" in result


class TestMCPClientConnection:
    """Tests for MCP client connection handling."""

    @pytest.mark.asyncio
    async def test_client_starts_and_stops(self):
        """Test that client can start and stop cleanly."""
        client = MCPTestClient(server_module="llming_lodge.tools.mcp.sample_server")
        await client.start()

        # Should be able to list tools
        tools = await client.list_tools()
        assert len(tools) > 0

        # Should stop cleanly
        await client.stop()

    @pytest.mark.asyncio
    async def test_multiple_tool_calls(self, mcp_client: MCPTestClient):
        """Test making multiple tool calls in sequence."""
        # Call different tools
        result1 = await mcp_client.call_tool_text("list_categories", {})
        result2 = await mcp_client.call_tool_text("search_products", {"query": "cable"})
        result3 = await mcp_client.call_tool_text("get_customer", {"customer_id": "C001"})

        assert "Electronics" in result1
        assert "Cable" in result2
        assert "John Smith" in result3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
