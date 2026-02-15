"""Adapter to bridge ToolRegistry with existing LlmToolbox interface.

This module provides backward compatibility by converting ToolDefinitions
from the new registry into LlmToolbox instances that the existing
provider clients understand.
"""
import logging
from typing import Any, Callable, Dict, List, Optional

from .llm_tool import LlmTool
from .llm_toolbox import LlmToolbox
from .tool_definition import ToolDefinition, ToolSource
from .tool_registry import ToolRegistry, get_default_registry
from .builtin_tools import DALLE3_PRICING

logger = logging.getLogger(__name__)


class ToolboxAdapter:
    """Bridges ToolRegistry to existing LlmToolbox interface.

    Converts ToolDefinition instances into LlmToolbox instances
    for backward compatibility with existing provider clients.
    """

    def __init__(self, registry: Optional[ToolRegistry] = None):
        """Initialize the adapter.

        Args:
            registry: ToolRegistry to use. If None, uses the default registry.
        """
        self.registry = registry or get_default_registry()

    def get_toolboxes(
        self,
        tool_names: List[str],
        provider: str,
        tool_config: Optional[Dict[str, Any]] = None,
        cost_callback: Optional[Callable[[str, float], None]] = None,
        openai_client: Optional[Any] = None,
        google_client: Optional[Any] = None,
    ) -> List[LlmToolbox]:
        """Convert tool definitions to LlmToolbox instances.

        Args:
            tool_names: List of tool names to include
            provider: Provider name (openai, anthropic, google, etc.)
            tool_config: Per-tool configuration dict
            cost_callback: Optional callback for cost tracking
            openai_client: Optional OpenAI client for DALL-E image generation
            google_client: Deprecated, unused

        Returns:
            List of LlmToolbox instances
        """
        logger.debug(f"[ADAPTER] get_toolboxes called with tool_names={tool_names}, provider={provider}")

        toolboxes = []
        tool_config = tool_config or {}

        for name in tool_names:
            # Look up tool with provider context for provider-specific variants
            tool = self.registry.get(name, provider=provider)
            if not tool:
                logger.warning(f"Tool '{name}' not found in registry")
                continue

            # Check provider compatibility (for tools that are provider-specific)
            if tool.requires_provider and tool.requires_provider != provider:
                logger.debug(f"Tool '{name}' requires provider '{tool.requires_provider}', skipping for '{provider}'")
                continue

            # Get per-tool config
            this_tool_config = tool_config.get(name, {})

            toolbox = self._create_toolbox(
                tool, provider, this_tool_config, cost_callback, openai_client
            )
            if toolbox:
                toolboxes.append(toolbox)

        return toolboxes

    def _create_toolbox(
        self,
        tool: ToolDefinition,
        provider: str,
        tool_config: Dict[str, Any],
        cost_callback: Optional[Callable[[str, float], None]] = None,
        openai_client: Optional[Any] = None,
    ) -> Optional[LlmToolbox]:
        """Create a LlmToolbox from a ToolDefinition.

        Args:
            tool: Tool definition
            provider: Provider name
            tool_config: Configuration for this specific tool
            cost_callback: Optional cost callback
            openai_client: Optional OpenAI client for DALL-E

        Returns:
            LlmToolbox instance or None if not applicable
        """
        if tool.source == ToolSource.PROVIDER_NATIVE:
            return self._create_provider_native_toolbox(tool, tool_config)

        elif tool.source == ToolSource.BUILTIN:
            if tool.name == "generate_image":
                return self._create_image_generation_toolbox(
                    tool, openai_client, tool_config, cost_callback
                )
            elif tool.callback:
                return self._create_callback_toolbox(tool, tool_config, cost_callback)
            else:
                logger.warning(f"Builtin tool '{tool.name}' has no callback")
                return None

        elif tool.source in (ToolSource.MCP_STDIO, ToolSource.MCP_HTTP, ToolSource.MCP_INPROCESS):
            return self._create_mcp_toolbox(tool, tool_config, cost_callback)

        else:
            logger.warning(f"Unknown tool source: {tool.source}")
            return None

    def _create_provider_native_toolbox(
        self,
        tool: ToolDefinition,
        tool_config: Dict[str, Any],
    ) -> LlmToolbox:
        """Create toolbox for provider-native tools.

        Provider-native tools are represented as strings or dicts in the tools list.
        The provider client handles them specially.
        """
        # For OpenAI's web_search, we can pass config like search_context_size
        if tool_config:
            # Return as dict with type and config merged
            tool_spec = {"type": tool.name, **tool_config}
            return LlmToolbox(
                name=tool.name,
                description=tool.description,
                tools=[tool_spec]
            )
        else:
            # Simple string for no config
            return LlmToolbox(
                name=tool.name,
                description=tool.description,
                tools=[tool.name]
            )

    def _create_image_generation_toolbox(
        self,
        tool: ToolDefinition,
        openai_client: Optional[Any],
        tool_config: Dict[str, Any],
        cost_callback: Optional[Callable[[str, float], None]] = None,
    ) -> Optional[LlmToolbox]:
        """Create toolbox for DALL-E image generation.

        Supports tool_config options:
        - size: Default size (1024x1024, 1792x1024, 1024x1792)
        - quality: Default quality (standard, hd)
        - allowed_sizes: List of allowed sizes (restricts what model can request)
        - allowed_qualities: List of allowed qualities
        """
        if not openai_client:
            logger.warning("Image generation requires OpenAI client")
            return None

        # Get config options with defaults
        default_size = tool_config.get("size", "1024x1024")
        default_quality = tool_config.get("quality", "standard")
        allowed_sizes = tool_config.get("allowed_sizes")
        allowed_qualities = tool_config.get("allowed_qualities")

        def generate_image(
            prompt: str,
            size: str = default_size,
            quality: str = default_quality
        ) -> str:
            """Generate an image using DALL-E 3."""
            # Enforce allowed sizes if configured
            if allowed_sizes and size not in allowed_sizes:
                logger.warning(f"Size '{size}' not allowed, using '{default_size}'")
                size = default_size

            # Enforce allowed qualities if configured
            if allowed_qualities and quality not in allowed_qualities:
                logger.warning(f"Quality '{quality}' not allowed, using '{default_quality}'")
                quality = default_quality

            # Calculate cost
            cost_key = (size, quality)
            cost_usd = DALLE3_PRICING.get(cost_key, 0.080)
            logger.debug(f"[DALLE] Generating image: size={size}, quality={quality}, cost=${cost_usd:.3f}")

            result = openai_client.generate_image_sync(
                prompt=prompt,
                size=size,
                quality=quality,
                model="dall-e-3"
            )

            # Log the cost
            if cost_callback:
                cost_callback("generate_image", cost_usd)
                logger.debug(f"[DALLE] Cost logged: ${cost_usd:.3f}")

            return result

        # Build input schema with defaults from config
        # Note: OpenAI requires all properties to be in "required" array
        input_schema = {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Detailed text description of the image to generate."
                },
                "size": {
                    "type": "string",
                    "enum": allowed_sizes or ["1024x1024", "1792x1024", "1024x1792"],
                    "description": f"Image size. Default: {default_size}",
                },
                "quality": {
                    "type": "string",
                    "enum": allowed_qualities or ["standard", "hd"],
                    "description": f"Image quality. Default: {default_quality}",
                }
            },
            "required": ["prompt", "size", "quality"]
        }

        llm_tool = LlmTool(
            name=tool.name,
            description=tool.description,
            func=generate_image,
            parameters=input_schema
        )

        return LlmToolbox(
            name="image_generation",
            description=tool.description,
            tools=[llm_tool]
        )

    def _create_callback_toolbox(
        self,
        tool: ToolDefinition,
        tool_config: Dict[str, Any],
        cost_callback: Optional[Callable[[str, float], None]] = None,
    ) -> LlmToolbox:
        """Create toolbox for a builtin tool with a callback."""
        # Wrap callback to handle costs and config
        def wrapped_callback(**kwargs) -> Any:
            # Merge tool_config defaults with provided kwargs
            merged_kwargs = {**tool_config, **kwargs}
            result = tool.callback(**merged_kwargs)
            if cost_callback and tool.fixed_cost_usd:
                cost_callback(tool.name, tool.fixed_cost_usd)
            return result

        llm_tool = LlmTool(
            name=tool.name,
            description=tool.description,
            func=wrapped_callback if (tool.fixed_cost_usd or tool_config) else tool.callback,
            parameters=tool.inputSchema
        )

        return LlmToolbox(
            name=tool.name,
            description=tool.description,
            tools=[llm_tool]
        )

    def _create_mcp_toolbox(
        self,
        tool: ToolDefinition,
        tool_config: Dict[str, Any],
        cost_callback: Optional[Callable[[str, float], None]] = None,
    ) -> LlmToolbox:
        """Create toolbox for an MCP tool.

        Note: MCP tools are async by nature, but LlmTool expects sync.
        We wrap them in a sync wrapper that runs the async code.
        """
        import asyncio

        # Capture the registry reference for the closure
        registry = self.registry
        tool_name = tool.name

        def sync_wrapper(**kwargs) -> Any:
            """Sync wrapper for async MCP tool execution."""
            import threading
            # Merge tool_config defaults with provided kwargs
            merged_kwargs = {**tool_config, **kwargs}
            logger.debug(f"[MCP_EXEC] Executing {tool_name}")

            async def _execute():
                return await registry.execute(tool_name, merged_kwargs)

            # Get the event loop where MCP connections were created
            mcp_loop = registry.get_event_loop()

            if mcp_loop and mcp_loop.is_running():
                # Use run_coroutine_threadsafe to run in the MCP event loop
                future = asyncio.run_coroutine_threadsafe(_execute(), mcp_loop)
                try:
                    return future.result(timeout=30)
                except Exception as e:
                    logger.error(f"[MCP_EXEC] Error executing {tool_name}: {e}")
                    raise
            else:
                # Fallback: try to use existing loop or create new one
                try:
                    loop = asyncio.get_running_loop()
                    future = asyncio.run_coroutine_threadsafe(_execute(), loop)
                    return future.result(timeout=30)
                except RuntimeError:
                    return asyncio.run(_execute())

        llm_tool = LlmTool(
            name=tool.name,
            description=tool.description,
            func=sync_wrapper,
            parameters=tool.inputSchema
        )

        return LlmToolbox(
            name=tool.name,
            description=tool.description,
            tools=[llm_tool]
        )


def get_toolboxes_for_config(
    tools: Optional[List[str]],
    provider: str,
    model_default_tools: Optional[List[str]] = None,
    tool_config: Optional[Dict[str, Any]] = None,
    cost_callback: Optional[Callable[[str, float], None]] = None,
    openai_client: Optional[Any] = None,
    google_client: Optional[Any] = None,
    registry: Optional[ToolRegistry] = None,
) -> List[LlmToolbox]:
    """Helper function to get toolboxes for a given configuration.

    This function handles the logic of determining which tools to use:
    1. If explicit tools list is provided, use it
    2. Otherwise, use model default tools

    Args:
        tools: Explicit list of tool names (None = use model defaults)
        provider: Provider name
        model_default_tools: Default tools for the model
        tool_config: Per-tool configuration dict
        cost_callback: Cost tracking callback
        openai_client: OpenAI client for DALL-E image generation
        google_client: Deprecated, unused
        registry: ToolRegistry to use (None = default)

    Returns:
        List of LlmToolbox instances
    """
    # Determine tool names to use
    if tools is not None:
        # Explicit list provided
        tool_names = list(tools)
    elif model_default_tools:
        # Use model defaults
        tool_names = list(model_default_tools)
    else:
        # No tools
        tool_names = []

    # Create adapter and get toolboxes
    adapter = ToolboxAdapter(registry)
    return adapter.get_toolboxes(
        tool_names=tool_names,
        provider=provider,
        tool_config=tool_config,
        cost_callback=cost_callback,
        openai_client=openai_client,
    )
