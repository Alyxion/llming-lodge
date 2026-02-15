#!/usr/bin/env python3
"""Base ChatController with all business logic for chat operations.

Views (NiceGUI, Terminal) subclass this and implement abstract hooks
for view-specific rendering.
"""

import asyncio
import logging
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from llming_lodge import ChatSession, ChatHistory, LLMConfig, LLMManager
from llming_lodge.budget import MemoryBudgetLimit, LimitPeriod, BudgetHandler
from llming_lodge.budget.budget_limit import BudgetLimit
from llming_lodge.budget.budget_manager import LLMBudgetManager
from llming_lodge.llm_base_models import Role, ChatMessage
from llming_lodge.tools.tool_call import ToolCallInfo, ToolCallStatus
from llming_lodge.tools.tool_definition import MCPServerConfig, ToolDefinition
from llming_lodge.tools.tool_registry import get_default_registry

logger = logging.getLogger(__name__)

# Get the shared LLM manager
llm_manager = LLMManager()


class ChatController(ABC):
    """Base chat controller with all business logic.

    Subclass and implement abstract hooks for view-specific rendering.
    All streaming, model switching, and state management happens here.
    """

    def __init__(
        self,
        *,
        user_id: str = "cli_user",
        user_mail: Optional[str] = None,
        budget_limits: Optional[List[BudgetLimit]] = None,
        system_prompt: Optional[str] = None,
        context_preamble: Optional[str] = None,
        mcp_servers: Optional[List[MCPServerConfig]] = None,
        initial_model: Optional[str] = None,
        user_avatar: Optional[str] = None,
        budget_handler: Optional[BudgetHandler] = None,
    ):
        """Initialize controller with configuration.

        Args:
            user_id: User identifier for budget tracking
            user_mail: Optional user email
            budget_limits: List of budget limits to apply
            system_prompt: Optional system prompt
            context_preamble: Optional hidden preamble silently prepended to the
                system prompt at API call time (e.g. user identity). Not visible
                in the editable prompt.
            mcp_servers: Optional MCP server configurations
            initial_model: Optional initial model (default: gpt-5.2)
            user_avatar: Optional URL to the user's avatar image
            budget_handler: Optional callback returning current BudgetInfo
        """
        self.user_id = user_id
        self.user_mail = user_mail or user_id
        self.user_avatar = user_avatar or ""
        self.budget_handler = budget_handler
        self.mcp_servers = mcp_servers or []

        # Get available models
        available_llms = list(llm_manager.get_available_llms())
        if not available_llms:
            raise ValueError("No LLM models available")

        # Set default model
        if initial_model:
            self.model = initial_model
        else:
            self.model = next(
                (info.model for info in available_llms if info.model == "gpt-5.2"),
                available_llms[0].model
            )

        # Get model info and config
        model_info = llm_manager.get_model_info(self.model)
        self.config = llm_manager.get_config_for_model(self.model)
        self.provider = llm_manager.get_provider_for_model(self.model)

        # Settings
        self.temperature = 0.7
        self.max_input_tokens = model_info.max_input_tokens
        self.max_output_tokens = model_info.max_output_tokens
        self.system_prompt = system_prompt or "You are a helpful assistant."
        self.context_preamble = context_preamble
        self.notifications = False

        # Tool settings
        self.available_tools = self._get_available_tools_for_provider(self.provider)
        self.enabled_tools = list(getattr(model_info, 'default_tools', self.available_tools))
        self.tool_config: Dict[str, Any] = {}

        # Budget manager
        if budget_limits:
            self.budget_manager = LLMBudgetManager(budget_limits)
        else:
            default_limit = MemoryBudgetLimit(
                name="default_budget",
                amount=10000.0,
                period=LimitPeriod.TOTAL
            )
            self.budget_manager = LLMBudgetManager(limits=[default_limit])

        # Configure LLM config
        self.config.temperature = self.temperature
        self.config.max_input_tokens = self.max_input_tokens
        self.config.max_tokens = self.max_output_tokens
        self.config.tools = self.enabled_tools
        self.config.tool_config = self.tool_config
        self.config.mcp_servers = self.mcp_servers if self.mcp_servers else None

        # Create session
        self.session = ChatSession(
            config=self.config,
            system_prompt=self.system_prompt,
            budget_manager=self.budget_manager,
            user_id=self.user_mail
        )
        self.session._context_preamble = self.context_preamble

        # Streaming state
        self._streaming_task: Optional[asyncio.Task] = None
        self._text_content: str = ""
        self._tool_calls: List[ToolCallInfo] = []
        self._chunk_images: List[str] = []
        self._displayed_image: bool = False
        self._generated_image_base64: Optional[str] = None

    @property
    def history(self) -> ChatHistory:
        """Get session history."""
        return self.session.history

    @history.setter
    def history(self, value: ChatHistory):
        """Set session history."""
        self.session.history = value

    @property
    def available_budget(self) -> float:
        """Get available budget amount."""
        return self.budget_manager.available_budget

    # ========== BUSINESS LOGIC (concrete methods) ==========

    async def send_message(
        self,
        message: str,
        images: Optional[List[str]] = None
    ) -> str:
        """Send message and stream response, calling hooks for rendering.

        Args:
            message: User message to send
            images: Optional list of base64 image data

        Returns:
            Complete response text
        """
        if self._streaming_task and not self._streaming_task.done():
            await self.stop_streaming()
            return ""

        # Reset streaming state
        self._text_content = ""
        self._tool_calls = []
        self._chunk_images = []
        self._displayed_image = False
        self._generated_image_base64 = None

        # Notify view that response is starting
        self._on_response_started()

        async def stream_response(gen):
            async for chunk in gen:
                if self._streaming_task and self._streaming_task.cancelled():
                    raise asyncio.CancelledError()

                # Handle tool call events
                if chunk.tool_call:
                    tc = chunk.tool_call
                    existing = next(
                        (t for t in self._tool_calls if t.call_id == tc.call_id),
                        None
                    )
                    if existing:
                        idx = self._tool_calls.index(existing)
                        self._tool_calls[idx] = tc
                    else:
                        self._tool_calls.append(tc)
                    self._on_tool_event(tc)

                # Accumulate text content (skip chunks that carry image data)
                if chunk.content and not chunk.tool_call:
                    self._text_content += chunk.content
                    self._on_text_chunk(chunk.content)
                elif (chunk.content and chunk.tool_call
                      and chunk.tool_call.is_image_generation
                      and chunk.tool_call.status == ToolCallStatus.COMPLETED):
                    # Image generation yields base64 in content alongside tool_call
                    self._chunk_images.append(chunk.content)
                    self._on_image_received(chunk.content)

                # Collect images from multimodal chunks
                if hasattr(chunk, 'images') and chunk.images:
                    for img in chunk.images:
                        if img not in self._chunk_images:
                            self._chunk_images.append(img)
                            self._on_image_received(img)

                await asyncio.sleep(0.01)

            return self._text_content

        try:
            gen = await self.session.chat_async(message, streaming=True, images=images)
            self._streaming_task = asyncio.create_task(stream_response(gen))
            result = await self._streaming_task

            # Check for images from tool execution
            if self.session.history.messages:
                last_msg = self.session.history.messages[-1]
                if last_msg.role == Role.ASSISTANT and last_msg.images:
                    for img in last_msg.images:
                        if img not in self._chunk_images:
                            self._on_image_received(img)

            self._on_response_completed(result)
            return result

        except asyncio.CancelledError:
            self._text_content += "\n\n*Stopped by user*"
            self._on_response_cancelled()
            return self._text_content

        except Exception as e:
            logger.error(f"[STREAM] Error: {type(e).__name__}: {e}")
            self._on_error(e)
            raise

        finally:
            self._streaming_task = None

    async def stop_streaming(self) -> None:
        """Cancel current streaming response."""
        if self._streaming_task and not self._streaming_task.done():
            self._streaming_task.cancel()
            try:
                await self._streaming_task
            except asyncio.CancelledError:
                pass
            self._streaming_task = None

    async def switch_model(self, model: str) -> None:
        """Switch to a different model, preserving history.

        Args:
            model: Model name to switch to
        """
        # Handle @ prefix for provider defaults
        if model.startswith('@'):
            model = llm_manager.get_default_model(model[1:])

        if model == self.model:
            return

        # Stop any ongoing streaming
        if self._streaming_task and not self._streaming_task.done():
            await self.stop_streaming()

        old_model = self.model
        model_info = llm_manager.get_model_info(model)
        new_config = llm_manager.get_config_for_model(model)

        # Update config with current settings and model's token limits
        new_config.temperature = self.temperature
        new_config.max_input_tokens = model_info.max_input_tokens
        new_config.max_tokens = model_info.max_output_tokens
        new_config.mcp_servers = self.mcp_servers if self.mcp_servers else None
        new_config.tool_config = self.tool_config

        # Update tools for new provider, preserving user-enabled tools that remain available
        new_provider = llm_manager.get_provider_for_model(model)
        new_available_tools = self._get_available_tools_for_provider(new_provider)

        # Start with model defaults
        new_enabled_tools = list(getattr(model_info, 'default_tools', new_available_tools))

        # Preserve user-enabled tools that are still available on the new provider
        for tool_name in self.enabled_tools:
            if tool_name in new_available_tools and tool_name not in new_enabled_tools:
                new_enabled_tools.append(tool_name)

        new_config.tools = new_enabled_tools

        # Create new history without system message
        old_history = self.session.get_history()
        new_history = ChatHistory()
        for msg in old_history.messages:
            if msg.role != Role.SYSTEM:
                new_history.add_message(msg)

        # Create new session with preserved history
        self.session = ChatSession.create_with_history(
            config=new_config,
            history=new_history,
            system_prompt=self.system_prompt,
            budget_manager=self.budget_manager,
            user_id=self.user_mail
        )

        # Update state
        self.config = new_config
        self.model = model
        self.provider = new_provider
        self.available_tools = new_available_tools
        self.enabled_tools = new_enabled_tools
        self.max_input_tokens = model_info.max_input_tokens
        self.max_output_tokens = model_info.max_output_tokens

        # Notify view
        self._on_model_switched(old_model, model)

    def update_settings(
        self,
        temperature: Optional[float] = None,
        max_input_tokens: Optional[int] = None,
        max_output_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[str]] = None,
        tool_config: Optional[Dict] = None
    ) -> None:
        """Update session settings.

        Args:
            temperature: Optional temperature value
            max_input_tokens: Optional max input tokens
            max_output_tokens: Optional max output tokens
            system_prompt: Optional system prompt
            tools: Optional list of enabled tools
            tool_config: Optional per-tool configuration
        """
        if temperature is not None:
            self.temperature = temperature
            self.config.temperature = temperature
        if max_input_tokens is not None:
            self.max_input_tokens = max_input_tokens
            self.config.max_input_tokens = max_input_tokens
        if max_output_tokens is not None:
            self.max_output_tokens = max_output_tokens
            self.config.max_tokens = max_output_tokens
        if system_prompt is not None:
            self.system_prompt = system_prompt
        if tools is not None:
            self.enabled_tools = tools
            self.config.tools = tools
        if tool_config is not None:
            self.tool_config = tool_config
            self.config.tool_config = tool_config

        # Apply to session
        self.session.config.temperature = self.temperature
        self.session.config.max_input_tokens = self.max_input_tokens
        self.session.config.max_tokens = self.max_output_tokens
        self.session.config.tools = self.enabled_tools
        self.session.config.tool_config = self.tool_config
        self.session.system_prompt = self.system_prompt
        self.session.invalidate_client()

    def is_tool_enabled(self, tool_name: str) -> bool:
        """Check if a specific tool is enabled."""
        return tool_name in self.enabled_tools

    def extract_inline_images(self, text: str) -> tuple[List[str], str]:
        """Extract inline images from markdown text.

        Handles markdown image syntax: ![alt text](data:image/...;base64,...)
        Also handles plain data URIs that might appear in responses.

        Args:
            text: Markdown text that may contain inline images

        Returns:
            Tuple of (list of image data URIs, text with images removed)
        """
        images = []
        text_without_images = text

        # Pattern 1: Markdown image with base64 data URI
        md_pattern = r'!\[([^\]]*)\]\((data:image/[^;]+;base64,[A-Za-z0-9+/=]+)\)'
        for match in re.finditer(md_pattern, text):
            data_uri = match.group(2)
            images.append(data_uri)
            text_without_images = text_without_images.replace(match.group(0), '')

        # Strip attachment:// placeholder URLs (LLM artifacts, not renderable)
        attachment_pattern = r'!\[([^\]]*)\]\(attachment://[^)]+\)'
        text_without_images = re.sub(attachment_pattern, '', text_without_images)

        # Pattern 2: Plain data URI on its own line
        plain_pattern = r'^(data:image/[^;]+;base64,[A-Za-z0-9+/=]+)$'
        for match in re.finditer(plain_pattern, text_without_images, re.MULTILINE):
            data_uri = match.group(1)
            if data_uri not in images:
                images.append(data_uri)
            text_without_images = text_without_images.replace(match.group(0), '')

        # Clean up extra blank lines
        text_without_images = re.sub(r'\n{3,}', '\n\n', text_without_images)

        return images, text_without_images.strip()

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.session.history = ChatHistory()

    def get_model_info(self, model: Optional[str] = None):
        """Get model info for current or specified model."""
        return llm_manager.get_model_info(model or self.model)

    def get_available_llms(self):
        """Get list of available LLM models."""
        return list(llm_manager.get_available_llms())

    async def discover_tools(self) -> None:
        """Trigger early tool discovery (before first message).

        Connects to MCP servers and discovers available tools.
        Syncs newly enabled MCP tools back to controller state.
        """
        logger.info(f"[TOOLS] Before discover: enabled_tools={self.enabled_tools}, config.tools={self.config.tools}")
        await self.session.discover_tools()
        logger.info(f"[TOOLS] After discover: session.config.tools={self.session.config.tools}")

        # Sync: session discovery may have added MCP tools to config.tools
        # that the controller doesn't know about yet.
        if self.session.config.tools:
            for tool_name in self.session.config.tools:
                if tool_name not in self.enabled_tools:
                    self.enabled_tools.append(tool_name)
            # Refresh available tools list to include MCP tools
            self.available_tools = self._get_available_tools_for_provider(self.provider)
        logger.info(f"[TOOLS] After sync: enabled_tools={self.enabled_tools}, available_tools={self.available_tools}")

    def _is_tool_available_for_provider(self, tool: ToolDefinition, provider: str) -> bool:
        """Check if a tool is available for a given provider.

        A tool is available if:
        - requires_provider is None or matches the provider, AND
        - provider is NOT in exclude_providers
        """
        if tool.requires_provider and tool.requires_provider != provider:
            return False
        if tool.exclude_providers and provider in tool.exclude_providers:
            return False
        return True

    def get_all_known_tools(self) -> List[dict]:
        """Get unified list of all known tools with availability and toggle state.

        - Built-in tools are de-duplicated (web_search:openai / web_search:anthropic -> single entry).
        - MCP tools are listed individually with is_mcp_group=True and the server's category,
          so the JS UI renders them in a category sub-menu with per-tool toggles.
        """
        registry = get_default_registry()
        all_tools = registry.get_visible()

        # Build lookup: tool_name -> server group metadata
        server_groups = getattr(self.session, '_mcp_server_groups', {})
        mcp_tool_group: Dict[str, Dict[str, Any]] = {}
        for group in server_groups.values():
            for tn in group.get("tool_names", []):
                mcp_tool_group[tn] = group

        # De-duplicate built-in / provider-native tools; emit MCP tools individually
        seen: Dict[str, dict] = {}
        for tool in all_tools:
            canonical = tool.name.split(":")[0]

            available = self._is_tool_available_for_provider(tool, self.provider)
            enabled = canonical in self.enabled_tools

            group = mcp_tool_group.get(canonical)
            if group:
                # MCP tool — use server group's category; tool's own display_name/icon
                exclude = group.get("exclude_providers")
                available = not (exclude and self.provider in exclude)

                entry = {
                    "name": canonical,
                    "display_name": tool.get_display_name(),
                    "description": tool.get_ui_description(),
                    "category": group.get("category", "General"),
                    "icon": tool.get_icon() or "smart_toy",
                    "available": available,
                    "enabled": enabled,
                    "is_mcp_group": True,
                    "server_label": group.get("label", ""),
                    "server_description": group.get("description", ""),
                }
                seen[canonical] = entry
            else:
                # Built-in / provider-native tool
                required_provider = None
                if tool.requires_provider and tool.requires_provider != self.provider:
                    required_provider = tool.requires_provider
                if tool.exclude_providers and self.provider in tool.exclude_providers:
                    required_provider = "another provider"

                category = tool.ui.category.capitalize() if tool.ui and tool.ui.category else "General"

                entry = {
                    "name": canonical,
                    "display_name": tool.get_display_name(),
                    "description": tool.get_ui_description(),
                    "category": category,
                    "group": "",
                    "icon": tool.get_icon() or "build",
                    "available": available,
                    "enabled": enabled,
                    "required_provider": required_provider,
                    "is_mcp_group": False,
                    "children": [],
                }

                if canonical not in seen or (available and not seen[canonical]["available"]):
                    seen[canonical] = entry

        return list(seen.values())

    def toggle_tool(self, tool_name: str, enabled: bool) -> None:
        """Toggle a tool or MCP server group on or off.

        For MCP groups, toggles all tools belonging to that server.
        """
        logger.info(f"[TOOLS] toggle_tool({tool_name!r}, {enabled}) — before: enabled_tools={self.enabled_tools}")
        # Check if this is an MCP server group
        server_groups = getattr(self.session, '_mcp_server_groups', {})
        group = server_groups.get(tool_name)
        if group:
            logger.info(f"[TOOLS] Found MCP group {tool_name!r}: tool_names={group.get('tool_names')}")
            for tn in group.get("tool_names", []):
                if enabled and tn not in self.enabled_tools:
                    self.enabled_tools.append(tn)
                elif not enabled and tn in self.enabled_tools:
                    self.enabled_tools.remove(tn)
        else:
            if enabled and tool_name not in self.enabled_tools:
                self.enabled_tools.append(tool_name)
            elif not enabled and tool_name in self.enabled_tools:
                self.enabled_tools.remove(tool_name)
        logger.info(f"[TOOLS] toggle_tool — after: enabled_tools={self.enabled_tools}")
        self.update_settings(tools=self.enabled_tools)

    @staticmethod
    def _get_available_tools_for_provider(provider: str) -> List[str]:
        """Get the list of available built-in tools for a provider.

        Args:
            provider: Provider name (openai, google, anthropic, etc.)

        Returns:
            List of tool names available for this provider
        """
        tools = []

        # Web search - available for OpenAI and Anthropic
        if provider in ("openai", "anthropic"):
            tools.append("web_search")

        # Image generation - DALL-E for OpenAI
        if provider == "openai":
            tools.append("generate_image")

        # Also include MCP tools that are available for this provider
        registry = get_default_registry()
        for tool in registry.get_all():
            if tool.source.value in ("mcp_stdio", "mcp_http", "mcp_inprocess"):
                canonical = tool.name.split(":")[0]
                if canonical not in tools:
                    # Check exclude_providers
                    if not tool.exclude_providers or provider not in tool.exclude_providers:
                        tools.append(canonical)

        return tools

    # ========== ABSTRACT HOOKS (subclasses implement) ==========

    @abstractmethod
    def _on_response_started(self) -> None:
        """Called when response streaming begins.

        View should prepare UI for incoming response (show spinner, etc.)
        """
        pass

    @abstractmethod
    def _on_text_chunk(self, content: str) -> None:
        """Called for each text chunk received.

        Args:
            content: Text chunk to display
        """
        pass

    @abstractmethod
    def _on_tool_event(self, tool_call: ToolCallInfo) -> None:
        """Called when tool status changes (pending/completed).

        Args:
            tool_call: Tool call information
        """
        pass

    @abstractmethod
    def _on_image_received(self, image_data: str) -> None:
        """Called when an image is received.

        Args:
            image_data: Base64 image data or data URI
        """
        pass

    @abstractmethod
    def _on_response_completed(self, full_text: str) -> None:
        """Called when streaming completes successfully.

        Args:
            full_text: Complete response text
        """
        pass

    @abstractmethod
    def _on_response_cancelled(self) -> None:
        """Called when streaming is cancelled by user."""
        pass

    @abstractmethod
    def _on_error(self, error: Exception) -> None:
        """Called when an error occurs.

        Args:
            error: The exception that occurred
        """
        pass

    @abstractmethod
    def _on_model_switched(self, old_model: str, new_model: str) -> None:
        """Called after model switch completes.

        Args:
            old_model: Previous model name
            new_model: New model name
        """
        pass
