"""LLM Tools for function calling.

This module provides both the legacy LlmTool/LlmToolbox interface and the
new MCP-compatible ToolDefinition/ToolRegistry system.

New code should use:
- ToolDefinition: MCP-compatible tool definition
- ToolRegistry: Central registry for tool management
- ToolboxAdapter: Bridge to legacy LlmToolbox interface

Legacy code can continue to use:
- LlmTool: Simple tool wrapper
- LlmToolbox: Collection of tools
- create_*_toolbox: Factory functions
"""

# Tool call tracking
from .tool_call import ToolCallInfo, ToolCallStatus

# Legacy interface (for backward compatibility)
from .llm_tool import LlmTool
from .llm_toolbox import LlmToolbox
from .builtin_tools import (
    create_image_generation_tool,
    create_web_search_toolbox,
    create_image_generation_toolbox,
    DALLE3_PRICING,
)

# New MCP-compatible interface
from .tool_definition import (
    ToolDefinition,
    ToolSource,
    ToolUIMetadata,
    DEFAULT_WEB_SEARCH_TOOL,
    DEFAULT_IMAGE_GENERATION_TOOL,
)
from .mcp import MCPServerConfig
from .tool_registry import (
    ToolRegistry,
    get_default_registry,
    reset_default_registry,
)
from .toolbox_adapter import (
    ToolboxAdapter,
    get_toolboxes_for_config,
)

__all__ = [
    # Tool call tracking
    'ToolCallInfo',
    'ToolCallStatus',

    # Legacy interface
    'LlmTool',
    'LlmToolbox',
    'create_image_generation_tool',
    'create_web_search_toolbox',
    'create_image_generation_toolbox',
    'DALLE3_PRICING',

    # New MCP-compatible interface
    'ToolDefinition',
    'ToolSource',
    'ToolUIMetadata',
    'MCPServerConfig',
    'DEFAULT_WEB_SEARCH_TOOL',
    'DEFAULT_IMAGE_GENERATION_TOOL',

    # Registry
    'ToolRegistry',
    'get_default_registry',
    'reset_default_registry',

    # Adapter
    'ToolboxAdapter',
    'get_toolboxes_for_config',
]
