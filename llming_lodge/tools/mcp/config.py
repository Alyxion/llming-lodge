"""Configuration for connecting to MCP servers."""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class MCPServerConfig(BaseModel):
    """Configuration for connecting to an MCP server."""
    # Stdio mode
    command: Optional[str] = Field(None, description="Command to execute for stdio transport")
    args: Optional[List[str]] = Field(None, description="Arguments for the command")
    env: Optional[Dict[str, str]] = Field(None, description="Environment variables to set")
    cwd: Optional[str] = Field(None, description="Working directory for the command")

    # HTTP mode
    url: Optional[str] = Field(None, description="URL for HTTP/SSE transport")
    api_key: Optional[str] = Field(None, description="API key for HTTP authentication")
    headers: Optional[Dict[str, str]] = Field(None, description="Additional HTTP headers")

    # In-process mode
    server_instance: Optional[Any] = Field(None, exclude=True, description="In-process MCP server instance")

    # UI / toggle metadata
    label: Optional[str] = Field(None, description="Display name (e.g. 'Stock Agent')")
    description: Optional[str] = Field(None, description="Short description for UI")
    category: Optional[str] = Field(None, description="Grouping category (e.g. 'Experimental')")
    enabled_by_default: bool = Field(False, description="If True, tools are enabled on discovery; otherwise opt-in")
    default_enabled_tools: Optional[List[str]] = Field(None, description="When set, only these tools are enabled by default (per-tool control). Requires enabled_by_default=True.")
    exclude_providers: Optional[List[str]] = Field(None, description="Providers this MCP does NOT support. None = all providers.")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def is_stdio(self) -> bool:
        """Check if this is a stdio-based connection."""
        return self.command is not None

    def is_http(self) -> bool:
        """Check if this is an HTTP-based connection."""
        return self.url is not None

    def is_inprocess(self) -> bool:
        """Check if this is an in-process server."""
        return self.server_instance is not None
