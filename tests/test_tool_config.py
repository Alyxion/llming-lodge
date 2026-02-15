"""Tests for tool configuration system."""
import pytest
from llming_lodge.session import LLMConfig
from llming_lodge.tools import (
    ToolDefinition,
    ToolSource,
    ToolUIMetadata,
    ToolRegistry,
    get_default_registry,
    reset_default_registry,
    ToolboxAdapter,
    get_toolboxes_for_config,
)


class TestToolDefinition:
    """Tests for ToolDefinition model."""

    def test_basic_tool_definition(self):
        """Test creating a basic tool definition."""
        tool = ToolDefinition(
            name="test_tool",
            description="A test tool",
            inputSchema={"type": "object", "properties": {"arg1": {"type": "string"}}}
        )
        assert tool.name == "test_tool"
        assert tool.description == "A test tool"
        assert tool.source == ToolSource.BUILTIN
        assert tool.inputSchema["properties"]["arg1"]["type"] == "string"

    def test_tool_with_ui_metadata(self):
        """Test tool with UI metadata."""
        tool = ToolDefinition(
            name="fancy_tool",
            description="A fancy tool",
            ui=ToolUIMetadata(
                icon="star",
                display_name="Fancy Tool",
                category="utilities",
                hidden=False
            )
        )
        assert tool.get_display_name() == "Fancy Tool"
        assert tool.get_icon() == "star"
        assert tool.is_visible() is True

    def test_hidden_tool(self):
        """Test hidden tool visibility."""
        tool = ToolDefinition(
            name="hidden_tool",
            description="A hidden tool",
            ui=ToolUIMetadata(hidden=True)
        )
        assert tool.is_visible() is False

    def test_display_name_fallback(self):
        """Test display name falls back to formatted tool name."""
        tool = ToolDefinition(
            name="my_cool_tool",
            description="A cool tool"
        )
        assert tool.get_display_name() == "My Cool Tool"

    def test_to_mcp_dict(self):
        """Test MCP-compatible dict conversion."""
        tool = ToolDefinition(
            name="mcp_tool",
            description="MCP compatible",
            inputSchema={"type": "object", "properties": {}}
        )
        mcp_dict = tool.to_mcp_dict()
        assert mcp_dict == {
            "name": "mcp_tool",
            "description": "MCP compatible",
            "inputSchema": {"type": "object", "properties": {}}
        }


class TestToolRegistry:
    """Tests for ToolRegistry."""

    def setup_method(self):
        """Reset registry before each test."""
        reset_default_registry()

    def test_default_registry_has_default_tools(self):
        """Test that default registry has web_search and generate_image."""
        registry = get_default_registry()
        assert registry.has("web_search")
        assert registry.has("generate_image")

    def test_register_builtin_tool(self):
        """Test registering a builtin tool."""
        registry = ToolRegistry(auto_register_defaults=False)

        def my_callback(arg1: str) -> str:
            return f"result: {arg1}"

        registry.register_builtin(
            name="my_tool",
            description="My custom tool",
            callback=my_callback,
            parameters={"type": "object", "properties": {"arg1": {"type": "string"}}},
            fixed_cost_usd=0.01
        )

        assert registry.has("my_tool")
        retrieved = registry.get("my_tool")
        assert retrieved.name == "my_tool"
        assert retrieved.source == ToolSource.BUILTIN
        assert retrieved.fixed_cost_usd == 0.01

    def test_register_provider_native_tool(self):
        """Test registering a provider-native tool."""
        registry = ToolRegistry(auto_register_defaults=False)

        registry.register_provider_native(
            name="native_search",
            description="Native search",
            provider_config={"type": "web_search"},
            requires_provider="openai"
        )

        assert registry.has("native_search")
        retrieved = registry.get("native_search")
        assert retrieved.source == ToolSource.PROVIDER_NATIVE
        assert retrieved.requires_provider == "openai"

    def test_get_for_provider(self):
        """Test filtering tools by provider."""
        registry = get_default_registry()
        openai_tools = registry.get_for_provider("openai")
        anthropic_tools = registry.get_for_provider("anthropic")

        # Both providers now support web_search (each has their own native implementation)
        openai_names = [t.name for t in openai_tools]
        anthropic_names = [t.name for t in anthropic_tools]

        assert "web_search" in openai_names
        assert "web_search" in anthropic_names  # Anthropic has native web_search too

        # generate_image is OpenAI-only (uses DALL-E)
        assert "generate_image" in openai_names
        assert "generate_image" not in anthropic_names

    def test_unregister_tool(self):
        """Test unregistering a tool."""
        registry = ToolRegistry(auto_register_defaults=False)
        registry.register_builtin("temp_tool", "Temporary", lambda: None)

        assert registry.has("temp_tool")
        result = registry.unregister("temp_tool")
        assert result is True
        assert not registry.has("temp_tool")

    def test_get_visible_tools(self):
        """Test getting only visible tools."""
        registry = ToolRegistry(auto_register_defaults=False)
        registry.register(ToolDefinition(name="visible", description="Visible tool"))
        registry.register(ToolDefinition(
            name="hidden",
            description="Hidden tool",
            ui=ToolUIMetadata(hidden=True)
        ))

        visible = registry.get_visible()
        names = [t.name for t in visible]
        assert "visible" in names
        assert "hidden" not in names


class TestToolConfig:
    """Tests for tool_config functionality."""

    def test_llmconfig_with_tool_config(self):
        """Test LLMConfig accepts tool_config."""
        config = LLMConfig(
            provider="openai",
            model="gpt-5.2",
            tools=["generate_image"],
            tool_config={
                "generate_image": {
                    "size": "1024x1024",
                    "quality": "standard"
                }
            }
        )
        assert config.tools == ["generate_image"]
        assert config.tool_config["generate_image"]["size"] == "1024x1024"
        assert config.tool_config["generate_image"]["quality"] == "standard"

    def test_llmconfig_without_deprecated_flags(self):
        """Test that deprecated flags are removed from LLMConfig."""
        import inspect
        sig = inspect.signature(LLMConfig)
        params = list(sig.parameters.keys())

        assert "enable_web_search" not in params
        assert "enable_image_generation" not in params

    def test_tool_config_empty_by_default(self):
        """Test tool_config is None by default."""
        config = LLMConfig(
            provider="openai",
            model="gpt-5.2"
        )
        assert config.tool_config is None

    def test_tools_none_uses_model_defaults(self):
        """Test tools=None means use model defaults."""
        config = LLMConfig(
            provider="openai",
            model="gpt-5.2",
            tools=None
        )
        assert config.tools is None


class TestToolboxAdapter:
    """Tests for ToolboxAdapter."""

    def setup_method(self):
        """Reset registry before each test."""
        reset_default_registry()

    def test_get_toolboxes_for_explicit_tools(self):
        """Test getting toolboxes for explicit tool list."""
        toolboxes = get_toolboxes_for_config(
            tools=["web_search"],
            provider="openai",
            model_default_tools=["generate_image"]  # Should be ignored
        )

        # Should only have web_search, not generate_image
        names = [tb.name for tb in toolboxes]
        assert "web_search" in names
        assert "image_generation" not in names

    def test_get_toolboxes_uses_model_defaults(self):
        """Test getting toolboxes when tools=None uses model defaults."""
        toolboxes = get_toolboxes_for_config(
            tools=None,
            provider="openai",
            model_default_tools=["web_search"]
        )

        names = [tb.name for tb in toolboxes]
        assert "web_search" in names

    def test_get_toolboxes_empty_when_no_tools(self):
        """Test getting toolboxes with empty tools list."""
        toolboxes = get_toolboxes_for_config(
            tools=[],
            provider="openai",
            model_default_tools=["web_search", "generate_image"]
        )

        assert len(toolboxes) == 0

    def test_tool_config_passed_to_provider_native(self):
        """Test tool_config is passed to provider-native toolbox."""
        toolboxes = get_toolboxes_for_config(
            tools=["web_search"],
            provider="openai",
            tool_config={"web_search": {"search_context_size": "low"}}
        )

        assert len(toolboxes) == 1
        # Provider native tools with config return a dict, not a string
        tool_spec = toolboxes[0].tools[0]
        assert isinstance(tool_spec, dict)
        assert tool_spec["type"] == "web_search"
        assert tool_spec["search_context_size"] == "low"

    def test_tool_config_without_config_returns_string(self):
        """Test provider-native tool without config returns string."""
        toolboxes = get_toolboxes_for_config(
            tools=["web_search"],
            provider="openai",
            tool_config=None
        )

        assert len(toolboxes) == 1
        # Without config, returns just the string
        tool_spec = toolboxes[0].tools[0]
        assert tool_spec == "web_search"

    def test_provider_filtering(self):
        """Test tools are filtered by provider requirement."""
        # generate_image requires OpenAI (uses DALL-E), should not appear for anthropic
        # web_search is available for both providers (each has native implementation)
        toolboxes = get_toolboxes_for_config(
            tools=["web_search", "generate_image"],
            provider="anthropic"
        )

        names = [tb.name for tb in toolboxes]
        assert "web_search" in names  # Anthropic has native web_search
        assert "image_generation" not in names  # DALL-E requires OpenAI


class TestImageGenerationConfig:
    """Tests for image generation tool configuration."""

    def setup_method(self):
        """Reset registry before each test."""
        reset_default_registry()

    def test_image_tool_uses_default_config(self):
        """Test image generation uses tool_config defaults."""
        # We can't fully test without OpenAI client, but we can test the config is accepted
        config = LLMConfig(
            provider="openai",
            model="gpt-5.2",
            tools=["generate_image"],
            tool_config={
                "generate_image": {
                    "size": "1792x1024",
                    "quality": "hd"
                }
            }
        )

        assert config.tool_config["generate_image"]["size"] == "1792x1024"
        assert config.tool_config["generate_image"]["quality"] == "hd"

    def test_image_tool_allowed_sizes_config(self):
        """Test image generation with restricted sizes."""
        config = LLMConfig(
            provider="openai",
            model="gpt-5.2",
            tools=["generate_image"],
            tool_config={
                "generate_image": {
                    "allowed_sizes": ["1024x1024"],
                    "allowed_qualities": ["standard"]
                }
            }
        )

        allowed = config.tool_config["generate_image"]["allowed_sizes"]
        assert allowed == ["1024x1024"]
        assert "1792x1024" not in allowed


class TestOpenAISchemaCompliance:
    """Tests to ensure tool schemas comply with OpenAI's requirements."""

    def setup_method(self):
        """Reset registry before each test."""
        reset_default_registry()

    def test_image_generation_schema_has_all_required_fields(self):
        """Test that image generation schema has all properties in required.

        OpenAI requires all properties to be listed in the 'required' array.
        This test ensures we don't accidentally break this requirement.
        """
        from unittest.mock import MagicMock

        # Create a mock OpenAI client
        mock_client = MagicMock()
        mock_client.generate_image_sync = MagicMock(return_value="base64data")

        adapter = ToolboxAdapter()
        toolboxes = adapter.get_toolboxes(
            tool_names=["generate_image"],
            provider="openai",
            openai_client=mock_client
        )

        assert len(toolboxes) == 1
        toolbox = toolboxes[0]
        assert len(toolbox.tools) == 1

        tool = toolbox.tools[0]
        schema = tool.parameters

        # OpenAI requires all properties to be in "required"
        properties = set(schema["properties"].keys())
        required = set(schema["required"])

        assert properties == required, (
            f"OpenAI requires all properties to be in 'required'. "
            f"Properties: {properties}, Required: {required}"
        )

    def test_tool_schema_structure(self):
        """Test that tool schemas have proper structure for OpenAI."""
        from unittest.mock import MagicMock

        mock_client = MagicMock()
        mock_client.generate_image_sync = MagicMock(return_value="base64data")

        adapter = ToolboxAdapter()
        toolboxes = adapter.get_toolboxes(
            tool_names=["generate_image"],
            provider="openai",
            openai_client=mock_client
        )

        tool = toolboxes[0].tools[0]
        schema = tool.parameters

        # Check required schema structure
        assert "type" in schema
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "required" in schema
        assert isinstance(schema["required"], list)

        # Check each property has type and description
        for prop_name, prop_def in schema["properties"].items():
            assert "type" in prop_def, f"Property '{prop_name}' missing 'type'"
            assert "description" in prop_def, f"Property '{prop_name}' missing 'description'"


class TestAnthropicWebSearch:
    """Tests for Anthropic's native web search tool."""

    def setup_method(self):
        """Reset registry before each test."""
        reset_default_registry()

    def test_anthropic_web_search_tool_format(self):
        """Test that web_search is converted to correct Anthropic format.

        Anthropic uses a special tool type 'web_search_20250305' for native web search.
        """
        from llming_lodge.providers.anthropic.anthropic_client import _convert_tools
        from llming_lodge.tools.llm_toolbox import LlmToolbox

        # Create toolbox with web_search as string (provider-native)
        toolbox = LlmToolbox(
            name="web_search",
            description="Web search",
            tools=["web_search"]
        )

        tools = _convert_tools([toolbox])

        assert len(tools) == 1
        assert tools[0]["type"] == "web_search_20250305"
        assert tools[0]["name"] == "web_search"
        assert "max_uses" in tools[0]

    def test_anthropic_web_search_with_config(self):
        """Test web_search with custom configuration."""
        from llming_lodge.providers.anthropic.anthropic_client import _convert_tools
        from llming_lodge.tools.llm_toolbox import LlmToolbox

        # Create toolbox with web_search as dict with config
        toolbox = LlmToolbox(
            name="web_search",
            description="Web search",
            tools=[{
                "type": "web_search",
                "max_uses": 10,
                "allowed_domains": ["example.com"],
                "blocked_domains": ["spam.com"]
            }]
        )

        tools = _convert_tools([toolbox])

        assert len(tools) == 1
        assert tools[0]["type"] == "web_search_20250305"
        assert tools[0]["max_uses"] == 10
        assert tools[0]["allowed_domains"] == ["example.com"]
        assert tools[0]["blocked_domains"] == ["spam.com"]

    def test_anthropic_web_search_via_adapter(self):
        """Test that adapter creates correct toolbox for Anthropic web_search."""
        adapter = ToolboxAdapter()
        toolboxes = adapter.get_toolboxes(
            tool_names=["web_search"],
            provider="anthropic"
        )

        assert len(toolboxes) == 1
        assert toolboxes[0].name == "web_search"
        # The toolbox should contain a string for provider-native tool
        assert "web_search" in toolboxes[0].tools


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
