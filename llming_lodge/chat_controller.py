#!/usr/bin/env python3
"""Base ChatController with all business logic for chat operations.

Subclasses (WebSocketChatController, etc.) implement abstract hooks
for view-specific rendering.
"""

import asyncio
import logging
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from llming_models import ChatSession, ChatHistory, LLMConfig, LLMManager
from llming_models.budget import MemoryBudgetLimit, LimitPeriod, BudgetHandler
from llming_models.budget.budget_limit import BudgetLimit
from llming_models.budget.budget_manager import LLMBudgetManager
from llming_models.llm_base_models import Role, ChatMessage
from llming_models.tools.tool_call import ToolCallInfo, ToolCallStatus
from llming_models.tools.tool_definition import MCPServerConfig, PROVIDER_COMPAT, ToolDefinition, ToolSource
from llming_models.tools.tool_registry import get_default_registry

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
            initial_model: Optional initial model (default: category "large")
            user_avatar: Optional URL to the user's avatar image
            budget_handler: Optional callback returning current BudgetInfo
        """
        self.user_id = user_id
        self.user_mail = user_mail or user_id
        self.user_avatar = user_avatar or ""
        self.budget_handler = budget_handler
        self.mcp_servers = mcp_servers or []
        self._message_tool_calls: dict[int, list[dict]] = {}

        # Get available models
        available_llms = list(llm_manager.get_available_llms())
        if not available_llms:
            raise ValueError("No LLM models available")

        # Set default model — @auto enables intelligent auto-selection
        if initial_model == "@auto" or (not initial_model):
            # Default to auto-select: start on GPT-5.4, upgrade to Opus when needed
            default_large = llm_manager.get_default_model("large")
            self.model = next(
                (info.model for info in available_llms if info.name == default_large or info.model == default_large),
                available_llms[0].model
            )
            self._auto_select = True
            self._auto_select_base_model = self.model
            logger.info("[AUTO-SELECT] Default enabled — starting on %s", self.model)
        elif initial_model:
            self.model = initial_model
            self._auto_select = False
            self._auto_select_base_model = ""
        else:
            default_large = llm_manager.get_default_model("large")
            self.model = next(
                (info.model for info in available_llms if info.name == default_large or info.model == default_large),
                available_llms[0].model
            )
            self._auto_select = False
            self._auto_select_base_model = ""

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
        if "web_search" not in self.enabled_tools:
            self.enabled_tools.append("web_search")
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

        # Auto-select mode: silently picks the best model per turn
        # NOTE: _auto_select and _auto_select_base_model are set in the
        # model selection block above (lines 76-94). Do NOT reset them here.

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

    async def available_budget_async(self) -> float:
        """Get available budget amount (async — use from event loop)."""
        return await self.budget_manager.available_budget_async()

    # ========== BUSINESS LOGIC (concrete methods) ==========

    # ── Auto-select model routing ──────────────────────────────

    async def _auto_select_model(self, message: str, images: list | None) -> None:
        """Ensure the auto-select base model is active.

        Called before ``send_message()`` when ``_auto_select`` is True.
        Always uses the configured default large model (GPT-5.4).
        """
        base_model = llm_manager.get_default_model("large") or "gpt-5.4"
        target_info = llm_manager.get_model_info(base_model)
        if not target_info:
            logger.warning("[AUTO-SELECT] Base model %s not available", base_model)
            return

        actual_model = target_info.model
        if actual_model != self.model:
            old = self.model
            await self._silent_switch_model(actual_model)
            logger.info("[AUTO-SELECT] %s → %s", old, actual_model)

    async def _silent_switch_model(self, model: str) -> None:
        """Switch model without sending model_switched to the client.

        Same as ``switch_model()`` but suppresses the UI notification.
        The user sees the auto-select label, not the underlying model.
        """
        if model == self.model:
            return

        model_info = llm_manager.get_model_info(model)
        new_config = llm_manager.get_config_for_model(model)
        new_config.temperature = self.temperature
        new_config.max_input_tokens = model_info.max_input_tokens
        new_config.max_tokens = model_info.max_output_tokens
        new_config.mcp_servers = self.mcp_servers if self.mcp_servers else None
        new_config.tool_config = self.tool_config

        new_provider = llm_manager.get_provider_for_model(model)
        new_available_tools = self._get_available_tools_for_provider(new_provider)
        # Preserve all user-enabled tools (unavailable ones are grayed out, not removed)
        new_enabled_tools = list(self.enabled_tools)
        model_defaults = list(getattr(model_info, 'default_tools', []))
        for tool_name in model_defaults:
            if tool_name not in new_enabled_tools:
                new_enabled_tools.append(tool_name)
        new_config.tools = new_enabled_tools

        old_history = self.session.get_history()
        old_condensed = getattr(self.session, '_condensed_summary', None)
        new_history = ChatHistory()
        for msg in old_history.messages:
            if msg.role != Role.SYSTEM:
                new_history.add_message(msg)

        old_mcp_groups = getattr(self.session, '_mcp_server_groups', {})
        old_mcp_connections = getattr(self.session, '_mcp_connections', {})

        self.session = ChatSession.create_with_history(
            config=new_config, history=new_history,
            system_prompt=self.system_prompt,
            budget_manager=self.budget_manager, user_id=self.user_mail,
        )
        self.session._mcp_server_groups = old_mcp_groups
        self.session._mcp_connections = old_mcp_connections
        self.session._context_preamble = self.context_preamble
        ad_catalog = getattr(self, "_auto_discover_catalog", None)
        if ad_catalog:
            self.session._system_prompt_suffix = ad_catalog
        if old_condensed:
            self.session._condensed_summary = old_condensed

        self.config = new_config
        self.model = model_info.model  # actual model/deployment name
        self.provider = new_provider
        self.available_tools = new_available_tools
        self.enabled_tools = new_enabled_tools
        self.max_input_tokens = model_info.max_input_tokens
        self.max_output_tokens = model_info.max_output_tokens
        # NOTE: no _on_model_switched() call — silent switch

    # ── Message sending ──────────────────────────────────────

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

        # Auto-select: pick optimal model before sending (skip if model is locked by a droplet)
        if self._auto_select and not getattr(self, '_model_locked', False):
            await self._auto_select_model(message, images)

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

    async def switch_model(self, model: str, _force: bool = False) -> None:
        """Switch to a different model, preserving history.

        Args:
            model: Model name to switch to. Use ``@auto`` for intelligent
                auto-selection mode (picks the best model per turn).
            _force: Internal flag to bypass model lock (used by system droplets).
        """
        # Block switching when model is locked by a system droplet
        if not _force and getattr(self, '_model_locked', False):
            logger.info("[MODEL] Switch to %s blocked — model locked by %s",
                        model, getattr(self, '_model_locked_reason', ''))
            return

        # Handle @auto — enter auto-select mode
        if model == "@auto":
            if not self._auto_select:
                self._auto_select = True
                self._auto_select_base_model = self.model
                logger.info("[AUTO-SELECT] Enabled — base model: %s", self.model)
                # Start on GPT-5.4 (fast capable default)
                default = llm_manager.get_default_model("large") or "gpt-5.4"
                default_info = llm_manager.get_model_info(default)
                if default_info and default_info.model != self.model:
                    old = self.model
                    await self._silent_switch_model(default_info.model)
                    logger.info("[AUTO-SELECT] Initial switch: %s → %s", old, default_info.model)
                # Notify client of the switch
                self._on_model_switched(self.model, "@auto")
            return

        # Leaving auto-select mode (explicit model choice)
        if self._auto_select:
            self._auto_select = False
            self._auto_select_base_model = ""
            logger.info("[AUTO-SELECT] Disabled — user chose %s", model)

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

        # Preserve ALL user-enabled tools across model switches.
        # Tools unavailable for the new provider stay in enabled_tools
        # but are grayed out in the UI (available=false). When switching
        # back to a provider that supports them, they reappear as enabled.
        new_enabled_tools = list(self.enabled_tools)
        # Add model defaults that aren't already enabled
        model_defaults = list(getattr(model_info, 'default_tools', []))
        for tool_name in model_defaults:
            if tool_name not in new_enabled_tools:
                new_enabled_tools.append(tool_name)

        new_config.tools = new_enabled_tools

        # Create new history without system message
        old_history = self.session.get_history()
        old_condensed = getattr(self.session, '_condensed_summary', None)
        new_history = ChatHistory()
        for msg in old_history.messages:
            if msg.role != Role.SYSTEM:
                new_history.add_message(msg)

        # Preserve MCP state from old session before creating new one
        old_mcp_groups = getattr(self.session, '_mcp_server_groups', {})
        old_mcp_connections = getattr(self.session, '_mcp_connections', {})

        # Create new session with preserved history
        self.session = ChatSession.create_with_history(
            config=new_config,
            history=new_history,
            system_prompt=self.system_prompt,
            budget_manager=self.budget_manager,
            user_id=self.user_mail
        )
        # Preserve MCP server groups and connections (tools are in the global registry)
        self.session._mcp_server_groups = old_mcp_groups
        self.session._mcp_connections = old_mcp_connections
        # Preserve context preamble (includes doc plugins + MCP hints)
        self.session._context_preamble = self.context_preamble
        # Preserve system prompt suffix (auto-discover catalog)
        ad_catalog = getattr(self, "_auto_discover_catalog", None)
        if ad_catalog:
            self.session._system_prompt_suffix = ad_catalog
        # Preserve condensed summary so conversation context survives model switch
        if old_condensed:
            self.session._condensed_summary = old_condensed

        # Update state
        self.config = new_config
        self.model = model_info.model  # actual model/deployment name
        self.provider = new_provider
        self.available_tools = new_available_tools
        self.enabled_tools = new_enabled_tools
        self.max_input_tokens = model_info.max_input_tokens
        self.max_output_tokens = model_info.max_output_tokens

        # Notify view
        self._on_model_switched(old_model, model_info.model)

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
        # web_search is always enabled by default
        if "web_search" not in self.enabled_tools:
            self.enabled_tools.append("web_search")
        logger.info(f"[TOOLS] After sync: enabled_tools={self.enabled_tools}, available_tools={self.available_tools}")

    def _is_tool_available_for_provider(self, tool: ToolDefinition, provider: str) -> bool:
        """Check if a tool is available for a given provider."""
        return tool.is_available_for_provider(provider)

    def get_all_known_tools(self) -> List[dict]:
        """Get unified list of all known tools with availability and toggle state.

        - Built-in tools are de-duplicated (web_search:openai / web_search:anthropic -> single entry).
        - MCP tools are listed individually with is_mcp_group=True and the server's category,
          so the JS UI renders them in a category sub-menu with per-tool toggles.
        - For categories like "Documents", tools are collapsed into one entry per MCP server.
        """
        registry = get_default_registry()
        all_tools = registry.get_visible()

        # Build lookup: tool_name -> (group_id, group_meta)
        server_groups = getattr(self.session, '_mcp_server_groups', {})
        mcp_tool_group: Dict[str, Dict[str, Any]] = {}
        mcp_tool_group_id: Dict[str, str] = {}
        hidden_tool_names: set = set()
        for group_id, group in server_groups.items():
            if group.get("hidden"):
                hidden_tool_names.update(group.get("tool_names", []))
                continue
            for tn in group.get("tool_names", []):
                mcp_tool_group[tn] = group
                mcp_tool_group_id[tn] = group_id

        # Categories where individual tools are hidden; one toggle per MCP server
        collapsed_categories = {"Documents"}

        # De-duplicate built-in / provider-native tools; emit MCP tools individually
        seen: Dict[str, dict] = {}
        # Track collapsed MCP groups so we emit one entry per group
        collapsed_groups_seen: Dict[str, str] = {}  # group_id -> entry key in seen
        for tool in all_tools:
            canonical = tool.name.split(":")[0]
            if canonical in hidden_tool_names:
                continue

            available = self._is_tool_available_for_provider(tool, self.provider)
            enabled = canonical in self.enabled_tools

            group = mcp_tool_group.get(canonical)
            if group:
                gid = mcp_tool_group_id.get(canonical, "")
                cat = group.get("category", "General")
                exclude = group.get("exclude_providers")
                requires = group.get("requires_providers")
                available = not (exclude and self.provider in exclude)
                if available and requires:
                    from llming_models.tools.tool_definition import PROVIDER_COMPAT
                    effective = PROVIDER_COMPAT.get(self.provider, self.provider)
                    available = (self.provider in requires or effective in requires)

                req_prov = requires if not available else None

                if cat in collapsed_categories or group.get("collapse_tools"):
                    # Collapsed: one entry per MCP server group
                    if gid in collapsed_groups_seen:
                        # Already emitted — just track enabled state
                        existing = seen[collapsed_groups_seen[gid]]
                        if enabled:
                            existing["_any_enabled"] = True
                        existing["_tool_names"].append(canonical)
                        continue
                    all_group_tools = group.get("tool_names", [])
                    any_enabled = any(tn in self.enabled_tools for tn in all_group_tools)
                    entry = {
                        "name": gid,
                        "display_name": group.get("label", gid),
                        "description": group.get("description", ""),
                        "category": cat,
                        "icon": "smart_toy",
                        "available": available,
                        "enabled": any_enabled,
                        "is_mcp_group": True,
                        "server_label": group.get("label", ""),
                        "server_description": group.get("description", ""),
                        "group_id": gid,
                        "collapse_tools": True,
                        "flyout": group.get("flyout", False),
                        "required_provider": req_prov,
                        "_any_enabled": any_enabled,
                        "_tool_names": list(all_group_tools),
                    }
                    seen[gid] = entry
                    collapsed_groups_seen[gid] = gid
                else:
                    entry = {
                        "name": canonical,
                        "display_name": tool.get_display_name(),
                        "description": tool.get_ui_description(),
                        "category": cat,
                        "icon": tool.get_icon() or "smart_toy",
                        "available": available,
                        "enabled": enabled,
                        "is_mcp_group": True,
                        "server_label": group.get("label", ""),
                        "server_description": group.get("description", ""),
                        "group_id": gid,
                        "flyout": group.get("flyout", False),
                        "required_provider": req_prov,
                    }
                    seen[canonical] = entry
            else:
                # Built-in / provider-native tool
                required_provider = None
                if not tool.is_available_for_provider(self.provider):
                    rp = tool.requires_providers or ([tool.requires_provider] if tool.requires_provider else None)
                    required_provider = rp or ["another provider"]

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

        # Clean up internal fields from collapsed groups
        result = []
        for entry in seen.values():
            entry.pop("_any_enabled", None)
            entry.pop("_tool_names", None)
            result.append(entry)
        return result

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

        # Web search - available for OpenAI, Azure OpenAI and Anthropic
        effective = PROVIDER_COMPAT.get(provider, provider)
        if effective in ("openai", "anthropic"):
            tools.append("web_search")

        # Image generation - DALL-E for OpenAI (including Azure OpenAI)
        if effective == "openai":
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

        # Include BUILTIN callback tools without provider restrictions
        # (e.g. consult_nudge) — these work with any provider
        for tool in registry.get_all():
            if (tool.source == ToolSource.BUILTIN
                    and tool.callback
                    and not tool.requires_provider
                    and tool.name not in tools):
                tools.append(tool.name)

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
