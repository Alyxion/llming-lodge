"""Central registry for LLM tools.

This module provides a ToolRegistry that manages tool registration,
discovery, and execution across different sources (builtin, MCP, provider-native).
"""
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .tool_definition import (
    ToolDefinition,
    ToolSource,
    ToolUIMetadata,
    MCPServerConfig,
    DEFAULT_WEB_SEARCH_TOOL,
    OPENAI_WEB_SEARCH_TOOL,
    ANTHROPIC_WEB_SEARCH_TOOL,
    DEFAULT_IMAGE_GENERATION_TOOL,
)

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Central registry for managing LLM tools.

    Provides methods for:
    - Registering tools from various sources
    - Loading tools from JSON configuration
    - Querying available tools
    - Executing tools (delegating to appropriate handler)
    """

    def __init__(self, auto_register_defaults: bool = True):
        """Initialize the tool registry.

        Args:
            auto_register_defaults: If True, register default tools (web_search, generate_image)
        """
        self._tools: Dict[str, ToolDefinition] = {}
        self._mcp_connections: Dict[str, Any] = {}  # MCPConnection instances
        self._event_loop = None  # Event loop for MCP async operations

        if auto_register_defaults:
            self._register_default_tools()

    def set_event_loop(self, loop) -> None:
        """Set the event loop for async MCP operations."""
        self._event_loop = loop

    def get_event_loop(self):
        """Get the event loop for async MCP operations."""
        return self._event_loop

    def _register_default_tools(self) -> None:
        """Register the default built-in tools."""
        # Web search - register provider-specific versions
        # Both registered under same name but with different requires_provider
        # The adapter will pick the right one based on provider
        self._tools["web_search:openai"] = OPENAI_WEB_SEARCH_TOOL
        self._tools["web_search:anthropic"] = ANTHROPIC_WEB_SEARCH_TOOL
        # Also register default for backward compat
        self.register(DEFAULT_WEB_SEARCH_TOOL)

        # Image generation - provider-specific versions
        # OpenAI uses DALL-E, Google uses Imagen 3
        self.register(DEFAULT_IMAGE_GENERATION_TOOL)  # DALL-E for OpenAI

    def register(self, tool: ToolDefinition) -> None:
        """Register a tool definition.

        Args:
            tool: The tool definition to register
        """
        if tool.name in self._tools:
            logger.warning(f"Tool '{tool.name}' already registered, overwriting")
        self._tools[tool.name] = tool
        logger.debug(f"Registered tool: {tool.name} (source={tool.source.value})")

    def register_builtin(
        self,
        name: str,
        description: str,
        callback: Callable[..., Any],
        parameters: Optional[Dict[str, Any]] = None,
        ui: Optional[ToolUIMetadata] = None,
        fixed_cost_usd: Optional[float] = None,
        requires_provider: Optional[str] = None,
    ) -> ToolDefinition:
        """Register a built-in Python tool.

        Args:
            name: Unique tool identifier
            description: Human-readable description
            callback: Python function to execute
            parameters: JSON Schema for tool parameters
            ui: UI display metadata
            fixed_cost_usd: Fixed cost per invocation
            requires_provider: If set, only works with this provider

        Returns:
            The registered ToolDefinition
        """
        tool = ToolDefinition(
            name=name,
            description=description,
            inputSchema=parameters or {"type": "object", "properties": {}},
            source=ToolSource.BUILTIN,
            callback=callback,
            ui=ui,
            fixed_cost_usd=fixed_cost_usd,
            requires_provider=requires_provider,
        )
        self.register(tool)
        return tool

    def register_provider_native(
        self,
        name: str,
        description: str,
        provider_config: Dict[str, Any],
        ui: Optional[ToolUIMetadata] = None,
        requires_provider: Optional[str] = None,
    ) -> ToolDefinition:
        """Register a provider-native tool (e.g., OpenAI's web_search).

        Args:
            name: Unique tool identifier
            description: Human-readable description
            provider_config: Provider-specific configuration
            ui: UI display metadata
            requires_provider: Provider this tool requires

        Returns:
            The registered ToolDefinition
        """
        tool = ToolDefinition(
            name=name,
            description=description,
            source=ToolSource.PROVIDER_NATIVE,
            provider_config=provider_config,
            ui=ui,
            requires_provider=requires_provider,
        )
        self.register(tool)
        return tool

    def register_mcp_stdio(
        self,
        name: str,
        description: str,
        command: str,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        ui: Optional[ToolUIMetadata] = None,
    ) -> ToolDefinition:
        """Register an MCP tool via stdio transport.

        Args:
            name: Unique tool identifier
            description: Human-readable description
            command: Command to execute
            args: Command arguments
            env: Environment variables
            cwd: Working directory
            parameters: JSON Schema for tool parameters
            ui: UI display metadata

        Returns:
            The registered ToolDefinition
        """
        tool = ToolDefinition(
            name=name,
            description=description,
            inputSchema=parameters or {"type": "object", "properties": {}},
            source=ToolSource.MCP_STDIO,
            mcp_server=MCPServerConfig(
                command=command,
                args=args,
                env=env,
                cwd=cwd,
            ),
            ui=ui,
        )
        self.register(tool)
        return tool

    def register_mcp_http(
        self,
        name: str,
        description: str,
        url: str,
        api_key: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        ui: Optional[ToolUIMetadata] = None,
    ) -> ToolDefinition:
        """Register an MCP tool via HTTP transport.

        Args:
            name: Unique tool identifier
            description: Human-readable description
            url: Server URL
            api_key: API key for authentication
            headers: Additional HTTP headers
            parameters: JSON Schema for tool parameters
            ui: UI display metadata

        Returns:
            The registered ToolDefinition
        """
        tool = ToolDefinition(
            name=name,
            description=description,
            inputSchema=parameters or {"type": "object", "properties": {}},
            source=ToolSource.MCP_HTTP,
            mcp_server=MCPServerConfig(
                url=url,
                api_key=api_key,
                headers=headers,
            ),
            ui=ui,
        )
        self.register(tool)
        return tool

    def load_from_json(self, path: Path) -> List[ToolDefinition]:
        """Load tool definitions from a JSON file.

        Expected format:
        {
            "tools": [
                {
                    "name": "tool_name",
                    "description": "...",
                    "source": "builtin|mcp_stdio|mcp_http|provider_native",
                    "inputSchema": {...},
                    "ui": {...},
                    ...
                }
            ]
        }

        Args:
            path: Path to JSON file

        Returns:
            List of loaded ToolDefinitions
        """
        loaded = []
        try:
            with open(path) as f:
                data = json.load(f)

            tools_data = data.get("tools", [])
            for tool_data in tools_data:
                # Convert nested objects
                if "ui" in tool_data and tool_data["ui"]:
                    tool_data["ui"] = ToolUIMetadata(**tool_data["ui"])
                if "mcp_server" in tool_data and tool_data["mcp_server"]:
                    tool_data["mcp_server"] = MCPServerConfig(**tool_data["mcp_server"])
                if "source" in tool_data:
                    tool_data["source"] = ToolSource(tool_data["source"])

                tool = ToolDefinition(**tool_data)
                self.register(tool)
                loaded.append(tool)

            logger.info(f"Loaded {len(loaded)} tools from {path}")
        except Exception as e:
            logger.error(f"Failed to load tools from {path}: {e}")

        return loaded

    def unregister(self, name: str) -> bool:
        """Unregister a tool by name.

        Args:
            name: Tool name to unregister

        Returns:
            True if tool was found and removed, False otherwise
        """
        if name in self._tools:
            del self._tools[name]
            logger.debug(f"Unregistered tool: {name}")
            return True
        return False

    def get(self, name: str, provider: Optional[str] = None) -> Optional[ToolDefinition]:
        """Get a tool by name, optionally for a specific provider.

        Args:
            name: Tool name
            provider: Optional provider name to get provider-specific version

        Returns:
            ToolDefinition if found, None otherwise
        """
        # If provider specified, try provider-specific key first
        if provider:
            provider_key = f"{name}:{provider}"
            tool = self._tools.get(provider_key)
            if tool:
                return tool

        # Fall back to generic name
        return self._tools.get(name)

    def get_all(self) -> List[ToolDefinition]:
        """Get all registered tools.

        Returns:
            List of all ToolDefinitions
        """
        return list(self._tools.values())

    def get_visible(self) -> List[ToolDefinition]:
        """Get all tools that should be shown in UI.

        Returns:
            List of visible ToolDefinitions
        """
        return [t for t in self._tools.values() if t.is_visible()]

    def get_by_category(self, category: str) -> List[ToolDefinition]:
        """Get all tools in a specific category.

        Args:
            category: Category name

        Returns:
            List of ToolDefinitions in the category
        """
        return [
            t for t in self._tools.values()
            if t.ui and t.ui.category == category
        ]

    def get_for_provider(self, provider: str) -> List[ToolDefinition]:
        """Get all tools available for a specific provider.

        Args:
            provider: Provider name (openai, anthropic, etc.)

        Returns:
            List of ToolDefinitions available for the provider
        """
        return [
            t for t in self._tools.values()
            if (t.requires_provider is None or t.requires_provider == provider)
            and not (t.exclude_providers and provider in t.exclude_providers)
        ]

    def get_names(self) -> List[str]:
        """Get all registered tool names.

        Returns:
            List of tool names
        """
        return list(self._tools.keys())

    def has(self, name: str) -> bool:
        """Check if a tool is registered.

        Args:
            name: Tool name

        Returns:
            True if registered
        """
        return name in self._tools

    def set_callback(self, name: str, callback: Callable[..., Any]) -> bool:
        """Set the callback for a builtin tool.

        This is useful for tools like generate_image where the callback
        needs to be created with runtime dependencies (like an OpenAI client).

        Args:
            name: Tool name
            callback: Callback function

        Returns:
            True if tool exists and callback was set
        """
        tool = self._tools.get(name)
        if tool and tool.source == ToolSource.BUILTIN:
            tool.callback = callback
            return True
        return False

    async def execute(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a tool by name.

        Args:
            name: Tool name
            arguments: Tool arguments

        Returns:
            Tool execution result

        Raises:
            ValueError: If tool not found or source not supported
        """
        tool = self.get(name)
        if not tool:
            raise ValueError(f"Tool not found: {name}")

        if tool.source == ToolSource.BUILTIN:
            if not tool.callback:
                raise ValueError(f"Tool '{name}' has no callback configured")
            # Call the callback (may be sync or async)
            import asyncio
            if asyncio.iscoroutinefunction(tool.callback):
                return await tool.callback(**arguments)
            else:
                return tool.callback(**arguments)

        elif tool.source == ToolSource.PROVIDER_NATIVE:
            # Provider-native tools are handled by the provider, not here
            raise ValueError(f"Provider-native tool '{name}' must be executed by the provider")

        elif tool.source in (ToolSource.MCP_STDIO, ToolSource.MCP_HTTP, ToolSource.MCP_INPROCESS):
            # MCP tools require a connection
            connection = self._mcp_connections.get(name)
            if not connection:
                raise ValueError(f"MCP connection not established for tool '{name}'")
            return await connection.call_tool(name, arguments)

        else:
            raise ValueError(f"Unsupported tool source: {tool.source}")


# Global default registry instance
_default_registry: Optional[ToolRegistry] = None


def get_default_registry() -> ToolRegistry:
    """Get the default global tool registry.

    Creates the registry on first access with default tools registered.

    Returns:
        The default ToolRegistry instance
    """
    global _default_registry
    if _default_registry is None:
        _default_registry = ToolRegistry(auto_register_defaults=True)
    return _default_registry


def reset_default_registry() -> None:
    """Reset the default registry (primarily for testing)."""
    global _default_registry
    _default_registry = None
