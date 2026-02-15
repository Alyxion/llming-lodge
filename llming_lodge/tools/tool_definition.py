"""MCP-compatible tool definitions for LLM interactions.

This module provides data models compatible with the Model Context Protocol (MCP)
and Anthropic's Agents API, while supporting multiple tool sources:
- Built-in Python callbacks
- Provider-native tools (e.g., OpenAI's web_search)
- MCP servers via stdio or HTTP
"""
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from llming_lodge.tools.mcp.config import MCPServerConfig


class ToolSource(str, Enum):
    """Source type for tool execution."""
    BUILTIN = "builtin"              # Direct Python callback
    MCP_STDIO = "mcp_stdio"          # Local MCP via stdio
    MCP_HTTP = "mcp_http"            # Remote MCP via HTTP/SSE
    MCP_INPROCESS = "mcp_inprocess"  # In-process MCP server (direct Python, no subprocess)
    PROVIDER_NATIVE = "provider_native"  # Provider handles internally (e.g., OpenAI web_search)


class ToolUIMetadata(BaseModel):
    """UI-related metadata for tool display."""
    icon: Optional[str] = Field(None, description="Icon path, emoji, or icon name")
    display_name: Optional[str] = Field(None, description="Human-readable name for UI")
    description: Optional[str] = Field(None, description="Short description for UI menu (full description goes to LLM)")
    category: Optional[str] = Field(None, description="Category for grouping (search, media, file, etc.)")
    hidden: bool = Field(False, description="If True, tool is not shown in UI")
    color: Optional[str] = Field(None, description="Optional accent color for UI")


class ToolDefinition(BaseModel):
    """MCP-compatible tool definition with execution config.

    Core fields (name, description, inputSchema) are compatible with MCP's
    Tool interface. Additional fields support execution and UI integration.
    """
    # MCP-compatible core fields
    name: str = Field(..., description="Unique tool identifier")
    description: str = Field(..., description="Human-readable description of what the tool does")
    inputSchema: Dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}},
        description="JSON Schema for tool parameters"
    )

    # Execution configuration
    source: ToolSource = Field(ToolSource.BUILTIN, description="How the tool is executed")
    callback: Optional[Callable[..., Any]] = Field(None, description="Python callback for BUILTIN tools", exclude=True)
    mcp_server: Optional[MCPServerConfig] = Field(None, description="MCP server config for MCP_* tools")
    provider_config: Optional[Dict[str, Any]] = Field(None, description="Provider-specific config for PROVIDER_NATIVE tools")

    # Extensions
    ui: Optional[ToolUIMetadata] = Field(None, description="UI display metadata")
    fixed_cost_usd: Optional[float] = Field(None, description="Fixed cost per invocation in USD")
    requires_provider: Optional[str] = Field(None, description="If set, tool only works with this provider")
    exclude_providers: Optional[List[str]] = Field(None, description="Providers this tool does NOT support. None = no exclusions.")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_mcp_dict(self) -> Dict[str, Any]:
        """Convert to MCP-compatible tool dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.inputSchema
        }

    def get_display_name(self) -> str:
        """Get the display name for UI."""
        if self.ui and self.ui.display_name:
            return self.ui.display_name
        # Convert snake_case to Title Case
        return self.name.replace("_", " ").title()

    def get_ui_description(self) -> str:
        """Get the short description for UI menu. Falls back to full description."""
        if self.ui and self.ui.description:
            return self.ui.description
        return self.description

    def get_icon(self) -> Optional[str]:
        """Get the icon for UI."""
        return self.ui.icon if self.ui else None

    def is_visible(self) -> bool:
        """Check if tool should be shown in UI."""
        return not (self.ui and self.ui.hidden)


# Default tool definitions for commonly used tools
# OpenAI web search (native tool)
OPENAI_WEB_SEARCH_TOOL = ToolDefinition(
    name="web_search",
    description="Search the web for current information",
    inputSchema={
        "type": "object",
        "properties": {},
        "additionalProperties": False
    },
    source=ToolSource.PROVIDER_NATIVE,
    provider_config={"type": "web_search"},
    ui=ToolUIMetadata(
        icon="search",
        display_name="Web Search",
        category="search"
    ),
    requires_provider="openai"
)

# Anthropic web search (native tool - uses Brave Search under the hood)
# Pricing: $10 per 1,000 searches
ANTHROPIC_WEB_SEARCH_TOOL = ToolDefinition(
    name="web_search",
    description="Search the web for current information",
    inputSchema={
        "type": "object",
        "properties": {},
        "additionalProperties": False
    },
    source=ToolSource.PROVIDER_NATIVE,
    provider_config={
        "type": "web_search_20250305",
        "name": "web_search",
        "max_uses": 5,  # Default limit per request
    },
    ui=ToolUIMetadata(
        icon="search",
        display_name="Web Search",
        category="search"
    ),
    requires_provider="anthropic",
    fixed_cost_usd=0.01,  # $10 per 1000 searches = $0.01 per search
)

# Default web search tool (OpenAI for backward compat - use get_web_search_tool_for_provider)
DEFAULT_WEB_SEARCH_TOOL = OPENAI_WEB_SEARCH_TOOL


def get_web_search_tool_for_provider(provider: str) -> ToolDefinition:
    """Get the appropriate web search tool for a provider."""
    if provider == "anthropic":
        return ANTHROPIC_WEB_SEARCH_TOOL
    return OPENAI_WEB_SEARCH_TOOL

DEFAULT_IMAGE_GENERATION_TOOL = ToolDefinition(
    name="generate_image",
    description="Generate images using DALL-E 3. Use this when the user asks you to create, draw, or generate an image.",
    inputSchema={
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "Detailed text description of the image to generate. Be specific about style, colors, composition, etc."
            },
            "size": {
                "type": "string",
                "enum": ["1024x1024", "1792x1024", "1024x1792"],
                "description": "Image size. Use 1024x1024 for square, 1792x1024 for landscape, 1024x1792 for portrait.",
                "default": "1024x1024"
            },
            "quality": {
                "type": "string",
                "enum": ["standard", "hd"],
                "description": "Image quality. 'hd' produces higher detail but takes longer.",
                "default": "standard"
            }
        },
        "required": ["prompt", "size", "quality"]
    },
    source=ToolSource.BUILTIN,
    ui=ToolUIMetadata(
        icon="image",
        display_name="Generate Image",
        category="media"
    ),
    fixed_cost_usd=0.04,  # Standard DALL-E 3 cost
    requires_provider="openai"
)

