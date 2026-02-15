"""Tests for MCP configuration integration with ChatSession."""

import pytest
import pytest_asyncio
import sys

from llming_lodge.session import LLMConfig, ChatSession
from llming_lodge.tools.tool_definition import MCPServerConfig
from llming_lodge.tools.tool_registry import get_default_registry, reset_default_registry
from llming_lodge import LLMManager


# Get provider and model ID from LLMInfo name
def get_provider_and_model(llm_name: str) -> tuple[str, str]:
    """Get provider and model ID from LLMInfo name."""
    manager = LLMManager()
    for llm_info in manager.get_available_llms():
        if llm_info.name == llm_name:
            return llm_info.provider, llm_info.model
    raise ValueError(f"Model {llm_name} not found")


# Test both providers
PROVIDER_MODELS = [
    pytest.param("gpt-5-nano", id="openai"),
    pytest.param("claude_haiku", id="anthropic"),
]


class TestMCPConfigInLLMConfig:
    """Tests for MCP server configuration in LLMConfig."""

    @pytest.mark.parametrize("llm_name", PROVIDER_MODELS)
    def test_mcp_servers_survives_model_dump(self, llm_name):
        """Test that mcp_servers works after model_dump serialization.

        This catches the bug where model_dump() converts MCPServerConfig
        to dicts, breaking code that expects objects.
        """
        provider, model = get_provider_and_model(llm_name)
        server_config = MCPServerConfig(
            command="python",
            args=["-m", "my_mcp_server"]
        )
        config = LLMConfig(
            provider=provider,
            model=model,
            mcp_servers=[server_config]
        )

        # Simulate what happens in _get_client
        dumped = config.model_dump()
        mcp_servers = dumped.get('mcp_servers')

        # This is what the fixed code does - handle both objects and dicts
        assert mcp_servers is not None
        for s in mcp_servers:
            # After model_dump, s is a dict not an object
            assert isinstance(s, dict)
            cmd = s.get('command') if isinstance(s, dict) else s.command
            assert cmd == "python"

    @pytest.mark.parametrize("llm_name", PROVIDER_MODELS)
    def test_mcp_servers_field_exists(self, llm_name):
        """Test that LLMConfig has mcp_servers field."""
        provider, model = get_provider_and_model(llm_name)
        config = LLMConfig(
            provider=provider,
            model=model
        )
        assert hasattr(config, 'mcp_servers')
        assert config.mcp_servers is None

    @pytest.mark.parametrize("llm_name", PROVIDER_MODELS)
    def test_mcp_servers_accepts_list(self, llm_name):
        """Test that mcp_servers accepts a list of MCPServerConfig."""
        provider, model = get_provider_and_model(llm_name)
        server_config = MCPServerConfig(
            command="python",
            args=["-m", "my_mcp_server"]
        )
        config = LLMConfig(
            provider=provider,
            model=model,
            mcp_servers=[server_config]
        )
        assert len(config.mcp_servers) == 1
        assert config.mcp_servers[0].command == "python"

    @pytest.mark.parametrize("llm_name", PROVIDER_MODELS)
    def test_mcp_servers_with_multiple_servers(self, llm_name):
        """Test configuration with multiple MCP servers."""
        provider, model = get_provider_and_model(llm_name)
        servers = [
            MCPServerConfig(command="python", args=["-m", "server1"]),
            MCPServerConfig(url="http://localhost:8080"),
        ]
        config = LLMConfig(
            provider=provider,
            model=model,
            mcp_servers=servers
        )
        assert len(config.mcp_servers) == 2
        assert config.mcp_servers[0].is_stdio()
        assert config.mcp_servers[1].is_http()


class TestMCPDiscovery:
    """Tests for MCP tool discovery in ChatSession."""

    def setup_method(self):
        """Reset registry before each test."""
        reset_default_registry()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("llm_name", PROVIDER_MODELS)
    async def test_discover_mcp_tools_registers_tools(self, llm_name):
        """Test that MCP tools are discovered and registered."""
        provider, model = get_provider_and_model(llm_name)
        # Configure with sample MCP server
        server_config = MCPServerConfig(
            command=sys.executable,
            args=["-m", "llming_lodge.tools.mcp.sample_server"]
        )
        config = LLMConfig(
            provider=provider,
            model=model,
            mcp_servers=[server_config]
        )

        session = ChatSession(config=config)

        # Trigger discovery
        await session._discover_mcp_tools()

        # Check that tools were registered
        registry = get_default_registry()
        assert registry.has("search_products")
        assert registry.has("get_product_details")
        assert registry.has("list_categories")
        assert registry.has("get_customer")
        assert registry.has("check_inventory")

        # Clean up
        await session.close_mcp_connections()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("llm_name", PROVIDER_MODELS)
    async def test_mcp_tools_added_to_enabled_tools(self, llm_name):
        """Test that discovered MCP tools are added to enabled tools list."""
        provider, model = get_provider_and_model(llm_name)
        server_config = MCPServerConfig(
            command=sys.executable,
            args=["-m", "llming_lodge.tools.mcp.sample_server"]
        )
        config = LLMConfig(
            provider=provider,
            model=model,
            tools=[],  # Start with empty tools
            mcp_servers=[server_config]
        )

        session = ChatSession(config=config)
        await session._discover_mcp_tools()

        # MCP tools should be added to config.tools
        assert "search_products" in session.config.tools
        assert "get_product_details" in session.config.tools

        await session.close_mcp_connections()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("llm_name", PROVIDER_MODELS)
    async def test_discovery_only_happens_once(self, llm_name):
        """Test that MCP discovery only happens on first call."""
        provider, model = get_provider_and_model(llm_name)
        server_config = MCPServerConfig(
            command=sys.executable,
            args=["-m", "llming_lodge.tools.mcp.sample_server"]
        )
        config = LLMConfig(
            provider=provider,
            model=model,
            mcp_servers=[server_config]
        )

        session = ChatSession(config=config)

        # First discovery
        await session._discover_mcp_tools()
        assert session._mcp_tools_discovered

        # Get connection count
        initial_connections = len(session._mcp_connections)

        # Second discovery should be skipped
        await session._discover_mcp_tools()
        assert len(session._mcp_connections) == initial_connections

        await session.close_mcp_connections()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("llm_name", PROVIDER_MODELS)
    async def test_close_mcp_connections(self, llm_name):
        """Test that MCP connections are properly closed."""
        provider, model = get_provider_and_model(llm_name)
        server_config = MCPServerConfig(
            command=sys.executable,
            args=["-m", "llming_lodge.tools.mcp.sample_server"]
        )
        config = LLMConfig(
            provider=provider,
            model=model,
            mcp_servers=[server_config]
        )

        session = ChatSession(config=config)
        await session._discover_mcp_tools()

        assert len(session._mcp_connections) > 0
        assert session._mcp_tools_discovered

        await session.close_mcp_connections()

        assert len(session._mcp_connections) == 0
        assert not session._mcp_tools_discovered


class TestMCPToolExecution:
    """Tests for executing MCP tools through the registry."""

    def setup_method(self):
        """Reset registry before each test."""
        reset_default_registry()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("llm_name", PROVIDER_MODELS)
    async def test_execute_mcp_tool_via_registry(self, llm_name):
        """Test that MCP tools can be executed through the registry."""
        provider, model = get_provider_and_model(llm_name)
        server_config = MCPServerConfig(
            command=sys.executable,
            args=["-m", "llming_lodge.tools.mcp.sample_server"]
        )
        config = LLMConfig(
            provider=provider,
            model=model,
            mcp_servers=[server_config]
        )

        session = ChatSession(config=config)
        await session._discover_mcp_tools()

        # Execute a tool through the registry
        registry = get_default_registry()
        result = await registry.execute("search_products", {"query": "headphones"})

        assert "Wireless Headphones" in result
        assert "P001" in result

        await session.close_mcp_connections()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("llm_name", PROVIDER_MODELS)
    async def test_execute_list_categories_tool(self, llm_name):
        """Test executing the list_categories tool."""
        provider, model = get_provider_and_model(llm_name)
        server_config = MCPServerConfig(
            command=sys.executable,
            args=["-m", "llming_lodge.tools.mcp.sample_server"]
        )
        config = LLMConfig(
            provider=provider,
            model=model,
            mcp_servers=[server_config]
        )

        session = ChatSession(config=config)
        await session._discover_mcp_tools()

        registry = get_default_registry()
        result = await registry.execute("list_categories", {})

        assert "Electronics" in result
        assert "Office" in result
        assert "Kitchen" in result

        await session.close_mcp_connections()


class TestChatStateFlow:
    """Tests that mimic the exact ChatState initialization flow."""

    def setup_method(self):
        """Reset registry before each test."""
        reset_default_registry()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("llm_name", PROVIDER_MODELS)
    async def test_chat_state_mcp_flow(self, llm_name):
        """Test the exact flow that ChatState uses.

        1. Create config with mcp_servers but empty tools
        2. Create session
        3. First chat_async call should discover MCP tools
        4. Tools should be sent to the model
        """
        provider, model = get_provider_and_model(llm_name)
        server_config = MCPServerConfig(
            command=sys.executable,
            args=["-m", "llming_lodge.tools.mcp.sample_server"]
        )

        # This mimics what ChatState does - tools are set but MCP tools not yet discovered
        config = LLMConfig(
            provider=provider,
            model=model,
            tools=[],  # Start with empty tools (avoid provider-specific defaults)
            mcp_servers=[server_config]
        )

        session = ChatSession(config=config)

        # At this point, config.tools should NOT have MCP tools
        assert "search_products" not in session.config.tools

        # Simulate what chat_async does - discover MCP tools
        await session._discover_mcp_tools()

        # Now MCP tools should be in config.tools
        assert "search_products" in session.config.tools, \
            f"MCP tools not discovered. config.tools = {session.config.tools}"

        # Build toolboxes like _get_client does
        toolboxes = session._build_toolboxes()

        # Extract all tool names from toolboxes
        all_tool_names = []
        for tb in toolboxes:
            for tool in tb.tools:
                if isinstance(tool, str):
                    all_tool_names.append(tool)
                elif hasattr(tool, 'name'):
                    all_tool_names.append(tool.name)
                elif isinstance(tool, dict):
                    all_tool_names.append(tool.get('type', str(tool)))

        print(f"Toolboxes: {[tb.name for tb in toolboxes]}")
        print(f"All tool names: {all_tool_names}")

        # MCP tools should be in toolboxes
        assert "search_products" in all_tool_names, \
            f"search_products not in toolboxes! Got: {all_tool_names}"

        await session.close_mcp_connections()


class TestMCPToolboxIntegration:
    """Tests for MCP tools being included in toolboxes sent to the model."""

    def setup_method(self):
        """Reset registry before each test."""
        reset_default_registry()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("llm_name", PROVIDER_MODELS)
    async def test_mcp_tools_included_in_toolboxes(self, llm_name):
        """Test that MCP tools are included in the toolboxes after discovery.

        This is the critical integration test that verifies:
        1. MCP tools are discovered and registered
        2. _build_toolboxes() creates toolboxes for MCP tools
        3. The toolboxes contain the actual LlmTool objects
        """
        provider, model = get_provider_and_model(llm_name)
        server_config = MCPServerConfig(
            command=sys.executable,
            args=["-m", "llming_lodge.tools.mcp.sample_server"]
        )
        config = LLMConfig(
            provider=provider,
            model=model,
            mcp_servers=[server_config]
        )

        session = ChatSession(config=config)

        # Step 1: Discover MCP tools
        await session._discover_mcp_tools()

        # Verify tools were added to config
        assert "search_products" in session.config.tools
        assert "get_product_details" in session.config.tools

        # Step 2: Build toolboxes (this is what gets sent to the model)
        toolboxes = session._build_toolboxes()

        # Step 3: Verify MCP tools are in the toolboxes
        toolbox_names = [tb.name for tb in toolboxes]
        tool_names_in_toolboxes = []
        for tb in toolboxes:
            for tool in tb.tools:
                if hasattr(tool, 'name'):
                    tool_names_in_toolboxes.append(tool.name)

        # MCP tools should be in the toolboxes
        assert "search_products" in tool_names_in_toolboxes, \
            f"search_products not in toolboxes. Got: {tool_names_in_toolboxes}"
        assert "get_product_details" in tool_names_in_toolboxes, \
            f"get_product_details not in toolboxes. Got: {tool_names_in_toolboxes}"

        await session.close_mcp_connections()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("llm_name", PROVIDER_MODELS)
    async def test_mcp_tools_have_correct_schema(self, llm_name):
        """Test that MCP tool schemas are preserved in toolboxes."""
        provider, model = get_provider_and_model(llm_name)
        server_config = MCPServerConfig(
            command=sys.executable,
            args=["-m", "llming_lodge.tools.mcp.sample_server"]
        )
        config = LLMConfig(
            provider=provider,
            model=model,
            mcp_servers=[server_config]
        )

        session = ChatSession(config=config)
        await session._discover_mcp_tools()

        toolboxes = session._build_toolboxes()

        # Find the search_products tool
        search_tool = None
        for tb in toolboxes:
            for tool in tb.tools:
                if hasattr(tool, 'name') and tool.name == "search_products":
                    search_tool = tool
                    break

        assert search_tool is not None, "search_products tool not found in toolboxes"

        # Verify it has the expected schema
        assert hasattr(search_tool, 'parameters')
        assert "properties" in search_tool.parameters
        assert "query" in search_tool.parameters["properties"]

        await session.close_mcp_connections()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
