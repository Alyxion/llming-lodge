"""WebSocket-based chat API for static frontends.

SessionRegistry — maps session_id → active sessions
WebSocketChatController — ChatController subclass that sends JSON over WS
build_ws_router() — FastAPI APIRouter with /ws/{session_id} endpoint
"""

import asyncio
import base64
import io
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from starlette.websockets import WebSocket, WebSocketDisconnect, WebSocketState

from llming_lodge import ChatSession, ChatHistory, LLMManager
from llming_lodge.budget import BudgetHandler
from llming_lodge.budget.budget_limit import BudgetLimit
from llming_lodge.llm_base_models import Role, ChatMessage
from llming_lodge.tools.tool_call import ToolCallInfo, ToolCallStatus
from llming_lodge.chat_controller import ChatController, llm_manager
from llming_lodge.tools.tool_definition import MCPServerConfig, ToolUIMetadata
from llming_lodge.tools.tool_registry import get_default_registry
from pathlib import Path

from llming_lodge.documents import UploadManager
from llming_lodge.utils.image_utils import sniff_image_mime
from llming_lodge.utils import LlmMarkdownPostProcessor
from llming_lodge.i18n import t_chat

logger = logging.getLogger(__name__)


# ── PDF Viewer MCP — auto-register/unregister on PDF upload/remove ──

def _build_pdf_sources(um) -> dict:
    """Build a file_id → bytes dict from upload manager files."""
    return {
        f.file_id: f.raw_data
        for f in um.files
        if f.mime_type == "application/pdf" and f.raw_data
    }


async def _register_mcp_tools(controller: "WebSocketChatController", mcp_config: MCPServerConfig) -> None:
    """Register tools from a single in-process MCP server into the session.

    Does NOT re-discover all MCP servers — only adds the new one.
    """
    from llming_lodge.tools.mcp import create_connection
    from llming_lodge.tools.tool_registry import get_default_registry

    registry = get_default_registry()
    connection = create_connection(mcp_config)
    await connection.start()
    tools = await connection.list_tools()

    label = mcp_config.label or "MCP"
    category = mcp_config.category or "General"
    group_tool_names = []

    for tool in tools:
        if category:
            from llming_lodge.tools.tool_definition import ToolUIMetadata
            if tool.ui:
                tool.ui.category = category
            else:
                tool.ui = ToolUIMetadata(category=category)
        registry.register(tool)
        controller.session._mcp_connections[tool.name] = connection
        registry._mcp_connections[tool.name] = connection
        group_tool_names.append(tool.name)

        # Auto-enable
        if mcp_config.enabled_by_default:
            if controller.session.config.tools is None:
                controller.session.config.tools = []
            if tool.name not in controller.session.config.tools:
                controller.session.config.tools = list(controller.session.config.tools) + [tool.name]
            if tool.name not in controller.enabled_tools:
                controller.enabled_tools.append(tool.name)

    controller.session._mcp_server_groups[label] = {
        "label": label,
        "description": mcp_config.description,
        "category": category,
        "exclude_providers": mcp_config.exclude_providers,
        "requires_providers": mcp_config.requires_providers,
        "collapse_tools": getattr(mcp_config, 'collapse_tools', False),
        "flyout": getattr(mcp_config, 'flyout', False),
        "tool_names": group_tool_names,
    }
    controller.available_tools = controller._get_available_tools_for_provider(controller.provider)

    # Collect prompt hints
    inst = mcp_config.server_instance
    if inst:
        try:
            hints = await inst.get_prompt_hints()
            if hints:
                controller.session._mcp_prompt_hints.extend(hints)
        except Exception:
            pass

    logger.info("[PDF_MCP] Registered tools: %s", group_tool_names)


async def _unregister_mcp_tools(controller: "WebSocketChatController", label: str) -> None:
    """Unregister tools from a named MCP server group."""
    from llming_lodge.tools.tool_registry import get_default_registry

    registry = get_default_registry()
    group = controller.session._mcp_server_groups.pop(label, None)
    if not group:
        return

    for tool_name in group.get("tool_names", []):
        registry.unregister(tool_name)
        controller.session._mcp_connections.pop(tool_name, None)
        registry._mcp_connections.pop(tool_name, None)
        if tool_name in controller.enabled_tools:
            controller.enabled_tools.remove(tool_name)
        if controller.session.config.tools and tool_name in controller.session.config.tools:
            controller.session.config.tools = [t for t in controller.session.config.tools if t != tool_name]

    controller.available_tools = controller._get_available_tools_for_provider(controller.provider)
    # Remove stale prompt hints (re-collect from remaining servers)
    controller.session._mcp_prompt_hints = []
    for sc in controller.mcp_servers:
        inst = getattr(sc, 'server_instance', None)
        if inst:
            try:
                hints = await inst.get_prompt_hints()
                if hints:
                    controller.session._mcp_prompt_hints.extend(hints)
            except Exception:
                pass

    logger.info("[PDF_MCP] Unregistered tools: %s", group.get("tool_names", []))


async def _sync_pdf_viewer_mcp(controller: "WebSocketChatController", entry: Any) -> None:
    """Ensure PdfViewerMCP is registered iff PDFs are currently attached."""
    um = entry.upload_manager
    if not um:
        return

    pdf_sources = _build_pdf_sources(um)
    existing_mcp = getattr(controller, "_pdf_viewer_mcp", None)

    if pdf_sources and not existing_mcp:
        # Create and register PdfViewerMCP
        try:
            from llming_lodge.documents.pdf_viewer_mcp import PdfViewerMCP
            from llming_lodge.documents.image_registry import ImageRegistry

            registry = getattr(controller, "_image_registry", None)
            if not registry:
                registry = ImageRegistry()
                controller._image_registry = registry

            def _image_callback(images: list[str], show: bool, page_info: list[dict] | None = None) -> None:
                """Inject images into the LLM's conversation context. Optionally send to browser."""
                if not controller.session or not controller.session.history.messages:
                    return
                last_msg = controller.session.history.messages[-1]
                if not hasattr(last_msg, "images") or last_msg.images is None:
                    last_msg.images = []
                # Strip data URI prefix — history stores raw base64, the
                # provider adds its own data: prefix when building API payloads.
                raw_images = []
                for img in images:
                    if img.startswith("data:"):
                        raw_images.append(img.split(",", 1)[1])
                    else:
                        raw_images.append(img)
                last_msg.images.extend(raw_images)
                controller.session._enforce_image_limit()

                # Send previews to browser (these need the full data URI)
                if show and page_info:
                    for img_data, info in zip(images, page_info):
                        asyncio.create_task(controller._send({
                            "type": "pdf_page_preview",
                            "data": img_data if img_data.startswith("data:") else f"data:image/jpeg;base64,{img_data}",
                            "file_id": info["file_id"],
                            "page": info["page"],
                            "total_pages": info["total_pages"],
                        }))
                elif show:
                    for img_data in images:
                        controller._on_image_received(img_data)

            mcp = PdfViewerMCP(pdf_sources, registry, _image_callback)
            controller._pdf_viewer_mcp = mcp

            mcp_config = MCPServerConfig(
                server_instance=mcp,
                label="PDF Viewer",
                description="Visually inspect PDF pages (charts, diagrams, layout)",
                category="Documents",
                enabled_by_default=True,
            )
            controller._pdf_viewer_mcp_config = mcp_config
            controller.mcp_servers.append(mcp_config)

            await _register_mcp_tools(controller, mcp_config)
            if controller._ws:
                await controller._send({
                    "type": "tools_updated",
                    "tools": controller.get_all_known_tools(),
                })
            logger.info("[PDF_MCP] Registered PdfViewerMCP with %d PDF(s)", len(pdf_sources))

        except ImportError as e:
            logger.warning("[PDF_MCP] pypdfium2 not available: %s", e)
        except Exception as e:
            logger.warning("[PDF_MCP] Failed to create PdfViewerMCP: %s", e)

    elif pdf_sources and existing_mcp:
        # Update: add new PDFs, remove gone ones
        current_ids = set(existing_mcp._pdf_sources.keys())
        new_ids = set(pdf_sources.keys())
        for fid in new_ids - current_ids:
            existing_mcp.add_pdf(fid, pdf_sources[fid])
        for fid in current_ids - new_ids:
            existing_mcp.remove_pdf(fid)

    elif not pdf_sources and existing_mcp:
        # Remove PdfViewerMCP — no more PDFs
        mcp_config = getattr(controller, "_pdf_viewer_mcp_config", None)
        if mcp_config and mcp_config in controller.mcp_servers:
            controller.mcp_servers.remove(mcp_config)
        img_registry = getattr(controller, "_image_registry", None)
        if img_registry:
            img_registry.unregister_source("pdf")
        controller._pdf_viewer_mcp = None
        controller._pdf_viewer_mcp_config = None

        await _unregister_mcp_tools(controller, "PDF Viewer")
        if controller._ws:
            await controller._send({
                "type": "tools_updated",
                "tools": controller.get_all_known_tools(),
            })
        logger.info("[PDF_MCP] Removed PdfViewerMCP — no more PDFs")


# ── Auto-Discover Nudge Tool ───────────────────────────────────────
# Per-session context for lazy content fetching.
# Keys: session_id → {store, user_email, user_teams, uids: set, loop}
_discoverable_sessions: dict[str, dict] = {}


def _consult_nudge_callback(uid: str, _session_id: str = "") -> str:
    """Lazily fetch and return nudge content for the given session + uid.

    Called from a thread-pool thread (via run_in_executor).  Schedules the
    async MongoDB fetch on the main event loop with run_coroutine_threadsafe.
    """
    ctx = _discoverable_sessions.get(_session_id)
    if not ctx or uid not in ctx["uids"]:
        return "Knowledge base not found or not available."

    async def _fetch() -> str:
        from llming_lodge.nudge_store import get_file_cache

        store = ctx["store"]
        nudge = await store.get_for_user(uid, ctx["user_email"], ctx["user_teams"])
        if not nudge:
            return "Knowledge base not found or not available."

        parts: list[str] = []
        sp = nudge.get("system_prompt", "")
        if sp:
            parts.append(sp)
        cached_files = await get_file_cache().get_files(
            uid, store, ctx["user_email"],
            user_teams=ctx["user_teams"], nudge=nudge,
        )
        for cf in cached_files:
            if cf.text_content:
                parts.append(f"--- {cf.name} ---\n{cf.text_content}")
        return "\n\n".join(parts) if parts else "Knowledge base has no content."

    loop = ctx.get("loop")
    if loop and loop.is_running():
        future = asyncio.run_coroutine_threadsafe(_fetch(), loop)
        try:
            return future.result(timeout=30)
        except Exception as e:
            logger.error("[CONSULT_NUDGE] Fetch error: %s", e)
            return f"Error consulting knowledge base: {e}"
    else:
        logger.error("[CONSULT_NUDGE] No running event loop for session %s", _session_id)
        return "Knowledge base temporarily unavailable."


# Register once at import time — _session_id is injected via tool_config,
# NOT exposed in the JSON schema (the LLM only sees 'uid').
get_default_registry().register_builtin(
    name="consult_nudge",
    description=(
        "Look up verified company-specific information from a knowledge base. "
        "ALWAYS call this tool before answering questions about HR, leave, "
        "benefits, or other topics listed in the Knowledge Bases section of "
        "your instructions. Never guess — use this tool to get accurate answers."
    ),
    callback=_consult_nudge_callback,
    parameters={
        "type": "object",
        "properties": {
            "uid": {
                "type": "string",
                "description": "The unique ID of the knowledge base to consult.",
            },
        },
        "required": ["uid"],
    },
    ui=ToolUIMetadata(hidden=True),
)


# ── Browser-Hosted MCP Nudges ─────────────────────────────────────
# Per-session context for MCP nudges running as browser Web Workers.
# Keys: session_id → {store, user_email, user_teams, uids: set, loop,
#                      controller, pending_requests, active_mcp_nudges: set}
_browser_mcp_sessions: dict[str, dict] = {}


async def _activate_browser_mcp(ctx: dict, uid: str) -> str:
    """Core async logic to activate a browser-hosted MCP nudge Worker.

    Reused by both the ``activate_mcp_nudge`` tool callback (via auto-discover)
    and the direct nudge-chat flow (auto-activation on new_chat).
    """
    from llming_lodge.nudge_store import get_file_cache
    from llming_lodge.tools.mcp.browser_connection import MCPBrowserConnection
    from llming_lodge.tools.tool_definition import ToolDefinition, ToolSource, ToolUIMetadata

    store = ctx["store"]
    controller = ctx["controller"]
    nudge = await store.get_for_user(uid, ctx["user_email"], ctx["user_teams"])
    if not nudge:
        return "MCP nudge not found or not available."

    # Check if nudge has any JS files (MCP capability)
    has_js = any(
        (f.get("name", "").endswith(".js") or f.get("name", "").endswith(".mjs"))
        for f in nudge.get("files", [])
    )
    if not has_js:
        return "This nudge has no MCP tools — use consult_nudge instead."

    # Separate JS source files from data files
    import base64 as b64mod
    files_data = {}   # JS source: {name: source_code}
    data_files = []   # Non-JS attachments: [{name, mime_type, size, content, text_content}]

    # Pre-fetch extracted text for all files (cached, single call)
    text_by_name: dict[str, str] = {}
    try:
        cached_files = await get_file_cache().get_files(
            uid, store, ctx["user_email"],
            user_teams=ctx["user_teams"], nudge=nudge,
        )
        for cf in cached_files:
            if cf.text_content:
                text_by_name[cf.name] = cf.text_content
    except Exception:
        pass

    for f in nudge.get("files", []):
        name = f.get("name", "")
        mime = f.get("mime_type", "")
        is_js = mime.startswith("application/javascript") or name.endswith(".js")
        content_raw = f.get("content", "")
        if not content_raw:
            continue

        # Strip data-URL prefix for base64 decoding
        b64_part = content_raw.split(",", 1)[-1] if content_raw.startswith("data:") else content_raw

        if is_js:
            try:
                files_data[name] = b64mod.b64decode(b64_part).decode("utf-8")
            except Exception:
                files_data[name] = content_raw
        else:
            # Data file — send both raw base64 and extracted text
            entry = {
                "name": name,
                "mime_type": mime,
                "size": f.get("size", 0),
                "content_base64": b64_part,
            }
            if name in text_by_name:
                entry["text_content"] = text_by_name[name]
            data_files.append(entry)

    if not files_data:
        return "MCP nudge has no JavaScript files."

    entry_point = nudge.get("mcp_entry_point", "index.js")

    # Send start message to browser and await tool list
    request_id = str(__import__("uuid").uuid4())
    loop = ctx["loop"]
    future = loop.create_future()
    ctx["pending_requests"][request_id] = future

    await controller._send({
        "type": "start_browser_mcp",
        "request_id": request_id,
        "nudge_uid": uid,
        "entry_point": entry_point,
        "files": files_data,
        "data_files": data_files,
    })

    try:
        result_msg = await asyncio.wait_for(future, timeout=15.0)
    except asyncio.TimeoutError:
        ctx["pending_requests"].pop(request_id, None)
        return "Browser MCP activation timed out (15s). Is the browser tab open?"

    if "error" in result_msg:
        return f"Failed to activate MCP nudge: {result_msg['error']}"

    # Register each tool from the Worker's tool list
    tools = result_msg.get("tools", [])
    if not tools:
        return "MCP nudge activated but reported no tools."

    connection = MCPBrowserConnection(uid, ctx)
    registry = get_default_registry()
    tool_names = []

    for t in tools:
        tool_name = t["name"]
        tool_def = ToolDefinition(
            name=tool_name,
            description=t.get("description", ""),
            inputSchema=t.get("inputSchema", {"type": "object", "properties": {}}),
            source=ToolSource.MCP_BROWSER,
            ui=ToolUIMetadata(hidden=True),
        )
        registry.register(tool_def)
        registry._mcp_connections[tool_name] = connection
        tool_names.append(tool_name)

        # Enable for this session
        if tool_name not in controller.enabled_tools:
            controller.enabled_tools.append(tool_name)
        if tool_name not in controller.available_tools:
            controller.available_tools.append(tool_name)

    # Update session config
    controller.update_settings(tools=controller.enabled_tools)

    # Inject prompt hint so the LLM knows it MUST use these tools.
    # Build a concise block listing each tool with its description.
    tool_lines = []
    for t in tools:
        desc = t.get("description", "")
        tool_lines.append(f"- **{t['name']}**: {desc}" if desc else f"- **{t['name']}**")
    nudge_name = nudge.get("name", uid)
    hint_block = (
        f"\n\n## MCP Tools — {nudge_name}\n"
        "CRITICAL: You have the following specialised tools available.  "
        "You MUST call the appropriate tool(s) BEFORE answering.  "
        "Never guess or answer from your own knowledge when a tool can provide the answer.\n\n"
        + "\n".join(tool_lines)
    )
    # Append to context preamble (persists across messages)
    base_preamble = controller.context_preamble or ""
    controller.context_preamble = base_preamble + hint_block
    if controller.session:
        controller.session._context_preamble = controller.context_preamble

    if controller._ws:
        await controller._send({
            "type": "tools_updated",
            "tools": controller.get_all_known_tools(),
        })

    ctx.setdefault("active_mcp_nudges", set()).add(uid)
    ctx.setdefault("active_tool_names", {})[uid] = tool_names

    return f"Activated MCP server '{nudge_name}' with {len(tools)} tools: {', '.join(tool_names)}"


def _activate_mcp_nudge_callback(uid: str, _session_id: str = "") -> str:
    """Activate a browser-hosted MCP nudge: spawn Worker, register tools.

    Called from a thread-pool thread (via run_in_executor). Schedules
    async operations on the main event loop with run_coroutine_threadsafe.
    """
    ctx = _browser_mcp_sessions.get(_session_id)
    if not ctx or uid not in ctx["uids"]:
        return "MCP nudge not found or not available."

    # Already activated?
    if uid in ctx.get("active_mcp_nudges", set()):
        return f"MCP nudge {uid} is already active."

    loop = ctx.get("loop")
    if loop and loop.is_running():
        future = asyncio.run_coroutine_threadsafe(_activate_browser_mcp(ctx, uid), loop)
        try:
            return future.result(timeout=30)
        except Exception as e:
            logger.error("[ACTIVATE_MCP_NUDGE] Error: %s", e)
            return f"Error activating MCP nudge: {e}"
    else:
        logger.error("[ACTIVATE_MCP_NUDGE] No running event loop for session %s", _session_id)
        return "MCP nudge activation temporarily unavailable."


get_default_registry().register_builtin(
    name="activate_mcp_nudge",
    description=(
        "Activate a browser-hosted MCP tool server from the nudge catalog. "
        "Call this ONCE for [MCP] entries in the Knowledge Bases section. "
        "After activation, the server's tools become available for direct use. "
        "Do NOT call this for regular knowledge bases — use consult_nudge instead."
    ),
    callback=_activate_mcp_nudge_callback,
    parameters={
        "type": "object",
        "properties": {
            "uid": {
                "type": "string",
                "description": "The unique ID of the MCP nudge to activate.",
            },
        },
        "required": ["uid"],
    },
    ui=ToolUIMetadata(hidden=True),
)


# ── Session Registry ────────────────────────────────────────────────


@dataclass
class SessionEntry:
    """A registered chat session with its controller and metadata."""
    controller: "WebSocketChatController"
    user_id: str
    user_name: str = ""
    user_avatar: str = ""
    websocket: Optional[WebSocket] = None
    created_at: float = field(default_factory=time.monotonic)
    last_activity: float = field(default_factory=time.monotonic)
    upload_manager: Optional[UploadManager] = None
    mcp_servers: Optional[List[MCPServerConfig]] = None
    doc_manager: Optional[Any] = None
    _cleanup_done: bool = False


class SessionRegistry:
    """Maps session_id → SessionEntry. Thread-safe via asyncio."""

    _instance: Optional["SessionRegistry"] = None
    # Global template registry — survives session cleanup so exports work from restored chats
    _presentation_templates: Dict[str, Any] = {}

    def __init__(self):
        self._sessions: Dict[str, SessionEntry] = {}

    @classmethod
    def get(cls) -> "SessionRegistry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register(
        self,
        session_id: str,
        controller: "WebSocketChatController",
        user_id: str,
        user_name: str = "",
        user_avatar: str = "",
        mcp_servers: Optional[List[MCPServerConfig]] = None,
    ) -> SessionEntry:
        entry = SessionEntry(
            controller=controller,
            user_id=user_id,
            user_name=user_name,
            user_avatar=user_avatar,
            upload_manager=UploadManager.create(session_id, user_id),
            mcp_servers=mcp_servers,
        )
        self._sessions[session_id] = entry
        logger.info(f"[REGISTRY] Registered session {session_id} for user {user_id}")
        return entry

    def get_session(self, session_id: str) -> Optional[SessionEntry]:
        entry = self._sessions.get(session_id)
        if entry:
            entry.last_activity = time.monotonic()
        return entry

    def remove(self, session_id: str) -> Optional[SessionEntry]:
        entry = self._sessions.pop(session_id, None)
        if entry:
            logger.info(f"[REGISTRY] Removed session {session_id}")
        return entry

    def cleanup_expired(self, ttl: float = 300.0) -> int:
        """Remove sessions idle for more than ttl seconds. Returns count removed."""
        now = time.monotonic()
        expired = [
            sid for sid, entry in self._sessions.items()
            if now - entry.last_activity > ttl
        ]
        for sid in expired:
            entry = self._sessions.pop(sid, None)
            if entry and entry.upload_manager:
                entry.upload_manager.cleanup()
        if expired:
            logger.info(f"[REGISTRY] Cleaned up {len(expired)} expired sessions")
        return len(expired)

    @property
    def active_count(self) -> int:
        return len(self._sessions)

    # ── Global presentation template registry ──────────────────

    @classmethod
    def register_templates(cls, templates: list) -> None:
        """Register presentation templates globally (by name).

        Called once per session setup — templates persist even after the
        session expires so that PPTX export works from restored chats.
        """
        for tpl in templates:
            name = getattr(tpl, "name", None) or ""
            if name:
                cls._presentation_templates[name] = tpl

    @classmethod
    def get_template(cls, name: str) -> Optional[Any]:
        """Look up a presentation template by name (global, session-independent)."""
        return cls._presentation_templates.get(name)


# ── WebSocket Chat Controller ───────────────────────────────────────


class WebSocketChatController(ChatController):
    """ChatController subclass that sends events as JSON over WebSocket."""

    def __init__(
        self,
        *,
        session_id: str,
        user_id: str = "ws_user",
        user_mail: Optional[str] = None,
        budget_limits: Optional[List[BudgetLimit]] = None,
        system_prompt: Optional[str] = None,
        context_preamble: Optional[str] = None,
        mcp_servers: Optional[List[MCPServerConfig]] = None,
        initial_model: Optional[str] = None,
        user_avatar: Optional[str] = None,
        budget_handler: Optional[BudgetHandler] = None,
        quick_actions: Optional[List[dict]] = None,
        locale: str = "en-us",
        on_language_change: Optional[Any] = None,
        on_action_callback: Optional[Any] = None,
        tool_toggle_notifications: Optional[dict[str, str]] = None,
        supported_languages: Optional[List[dict]] = None,
        directory_service: Optional[Any] = None,
        email_service: Optional[Any] = None,
        on_message_intercept: Optional[Any] = None,
        on_new_chat: Optional[Any] = None,
    ):
        super().__init__(
            user_id=user_id,
            user_mail=user_mail,
            budget_limits=budget_limits,
            system_prompt=system_prompt,
            context_preamble=context_preamble,
            mcp_servers=mcp_servers,
            initial_model=initial_model,
            user_avatar=user_avatar,
            budget_handler=budget_handler,
        )
        self.session_id = session_id
        self._ws: Optional[WebSocket] = None
        self._base_system_prompt = system_prompt or self.system_prompt
        self._app_system_prompt = self._base_system_prompt  # immutable app-level default
        self._conversation_title: Optional[str] = None
        self._title_msg_count: int = 0
        self._title_task: Optional[asyncio.Task] = None
        self._custom_quick_actions: Optional[List[dict]] = quick_actions
        self._locale: str = locale
        self._on_language_change = on_language_change
        self._on_action_callback = on_action_callback
        self._tool_toggle_notifications = tool_toggle_notifications or {}
        self._directory_service = directory_service
        self._email_service = email_service
        self._on_message_intercept = on_message_intercept
        self._on_new_chat = on_new_chat
        self._project_id: Optional[str] = None
        self._nudge_id: Optional[str] = None
        self._translation_overrides: dict[str, str] = {}
        self._speech_response: bool = False
        self._speech_service: Optional[Any] = None
        self._saved_max_output_tokens: Optional[int] = None
        self._saved_system_prompt: Optional[str] = None
        self._supported_languages: List[dict] = supported_languages or []
        self._speech_max_tokens: int = 2000
        self._tts_voice: str = ""  # empty = default voice (set from config)
        self._tts_default_voice: str = ""  # app-config default voice
        self._tts_model: str = ""  # app-config TTS model override
        self._tts_text_buffer: str = ""
        self._tts_segment_idx: int = 0
        self._tts_sending: bool = False
        self._tts_pending: list[str] = []
        self._tts_flush_timer: Optional[asyncio.TimerHandle] = None

    def get_all_known_tools(self) -> List[dict]:
        """Override to translate built-in tool names and categories.

        Also filters out hidden MCP server groups (dev MCPs).
        """
        # Collect tool names AND group labels belonging to hidden MCP groups
        hidden_tools: set = set()
        hidden_groups: set = set()
        server_groups = getattr(self.session, '_mcp_server_groups', {})
        for group_label, group in server_groups.items():
            if group.get("hidden"):
                hidden_tools.update(group.get("tool_names", []))
                hidden_groups.add(group_label)

        tools = super().get_all_known_tools()
        if hidden_tools or hidden_groups:
            tools = [
                t for t in tools
                if t.get("name") not in hidden_tools
                and t.get("group_id") not in hidden_groups
            ]
        for tool in tools:
            if tool.get("is_mcp_group"):
                continue
            name_key = f"chat.tool.{tool['name']}"
            cat_key = f"chat.tool_cat.{(tool.get('category') or '').lower()}"
            translated_name = t_chat(name_key, self._locale)
            translated_cat = t_chat(cat_key, self._locale)
            if translated_name != name_key:
                tool["display_name"] = translated_name
            if translated_cat != cat_key:
                tool["category"] = translated_cat
        return tools

    def set_websocket(self, ws: WebSocket) -> None:
        self._ws = ws

    async def _send(self, msg: dict) -> None:
        """Send a JSON message over the WebSocket."""
        if self._ws and self._ws.client_state == WebSocketState.CONNECTED:
            try:
                await self._ws.send_json(msg)
            except Exception as e:
                logger.debug(f"[WS] Send failed: {e}")

    # ── Abstract hook implementations ─────────────────────────

    def _on_response_started(self) -> None:
        model_info = llm_manager.get_model_info(self.model)
        asyncio.create_task(self._send({
            "type": "response_started",
            "model": self.model,
            "model_icon": model_info.model_icon if model_info else "",
            "model_label": model_info.label if model_info else self.model,
        }))

    _SENTENCE_BREAK = re.compile(r'[.!?]\s')
    _LONG_BREAK = re.compile(r'[,;:–]\s')
    _URL_RE = re.compile(r'https?://\S+')
    _MD_LINK_RE = re.compile(r'\[([^\]]+)\]\(https?://[^)]+\)')

    _CODE_BLOCK_RE = re.compile(r'```[^\n]*\n[\s\S]*?```')

    @staticmethod
    def _has_open_fence(text: str) -> bool:
        """Check if text has an unclosed ``` fence."""
        return text.count('```') % 2 != 0

    def _on_text_chunk(self, content: str) -> None:
        asyncio.create_task(self._send({
            "type": "text_chunk",
            "content": content,
        }))
        if self._speech_response:
            self._tts_text_buffer += content
            # Strip completed code blocks (```...```) from TTS buffer
            self._tts_text_buffer = self._CODE_BLOCK_RE.sub(' ', self._tts_text_buffer)
            # If an unclosed fence remains, wait for it to close before flushing
            if self._has_open_fence(self._tts_text_buffer):
                return
            self._flush_tts_sentences()
            # Start a timer to flush even without sentence punctuation
            self._schedule_tts_flush_timer()

    def _on_tool_event(self, tool_call: ToolCallInfo) -> None:
        asyncio.create_task(self._send({
            "type": "tool_event",
            "name": tool_call.name,
            "call_id": tool_call.call_id,
            "display_name": tool_call.display_name,
            "status": tool_call.status.value if hasattr(tool_call.status, "value") else str(tool_call.status),
            "is_image_generation": tool_call.is_image_generation,
            "result": tool_call.result if tool_call.status == ToolCallStatus.COMPLETED else None,
        }))

    def _on_image_received(self, image_data: str) -> None:
        # Store for response_completed payload
        self._generated_image_base64 = image_data
        asyncio.create_task(self._send({
            "type": "image_received",
            "data": sniff_image_mime(image_data),
        }))

    def _on_response_completed(self, full_text: str) -> None:
        # Process generated image
        generated_image = None
        if self._generated_image_base64:
            generated_image = sniff_image_mime(self._generated_image_base64)
            # Store in history
            try:
                if self.session.history.messages:
                    last_msg = self.session.history.messages[-1]
                    if last_msg.role == Role.ASSISTANT:
                        if not last_msg.images:
                            last_msg.images = []
                        last_msg.images.append(self._generated_image_base64)
            except Exception as e:
                logger.warning(f"[HISTORY] Failed to update message with generated image: {e}")

        tool_calls_data = []
        for tc in self._tool_calls:
            tool_calls_data.append({
                "name": tc.name,
                "call_id": tc.call_id,
                "display_name": tc.display_name,
                "status": tc.status.value if hasattr(tc.status, "value") else str(tc.status),
                "is_image_generation": tc.is_image_generation,
            })

        asyncio.create_task(self._send({
            "type": "response_completed",
            "full_text": full_text,
            "generated_image": generated_image,
            "tool_calls": tool_calls_data,
        }))

        # Sentence-level TTS streaming: flush remaining buffer and send done
        if self._speech_response:
            asyncio.create_task(self._flush_remaining_tts())

        # Trigger title generation and save in background
        asyncio.create_task(self._post_response_tasks())

    async def _post_response_tasks(self) -> None:
        """After response: save conversation immediately, then kick off title in background."""
        # Save conversation IMMEDIATELY — no LLM calls before this
        try:
            data = self._serialize_conversation()
            if data:
                await self._send({"type": "save_conversation", "data": data})
        except Exception as e:
            logger.debug(f"[SAVE] Conversation save failed: {e}")

        try:
            await self._send_context_info()
        except Exception:
            pass

        # Send updated budget
        try:
            budget = await self.available_budget_async()
            if self.budget_handler:
                info = self.budget_handler()
                budget = info.get("available", budget)
            await self._send({"type": "budget_update", "budget": budget})
        except Exception:
            pass

    def _get_speech_service(self):
        """Lazy-init SpeechService."""
        if not self._speech_service:
            from llming_lodge.speech_service import SpeechService
            svc = SpeechService()
            if self._tts_model:
                svc.TTS_MODEL = self._tts_model
            if self._tts_default_voice:
                svc.TTS_DEFAULT_VOICE = self._tts_default_voice
            self._speech_service = svc
        return self._speech_service

    @staticmethod
    def _get_tts_voices():
        from llming_lodge.speech_service import TTS_VOICES
        return TTS_VOICES

    def _get_speech_service_default_voice(self) -> str:
        from llming_lodge.speech_service import SpeechService
        return self._tts_default_voice or SpeechService.TTS_DEFAULT_VOICE

    async def _auto_tts(self, text: str) -> None:
        """Synthesize TTS for the completed response and send to client (legacy single-shot)."""
        try:
            text = self._clean_text_for_tts(text)
            if not text:
                return
            service = self._get_speech_service()
            mp3_bytes, word_timings = await service.synthesize(text, locale=self._locale, voice=self._tts_voice)
            audio_b64 = base64.b64encode(mp3_bytes).decode("ascii")
            await self._send({
                "type": "tts_audio",
                "audio_b64": audio_b64,
                "word_timings": word_timings,
            })
        except Exception as e:
            logger.warning(f"[TTS] Auto-TTS failed: {e}")

    # ── Sentence-level TTS streaming ───────────────────

    def _schedule_tts_flush_timer(self) -> None:
        """Schedule a timer to flush TTS buffer if no sentence break arrives soon."""
        if self._tts_flush_timer:
            self._tts_flush_timer.cancel()
        loop = asyncio.get_event_loop()
        self._tts_flush_timer = loop.call_later(1.5, self._timer_flush_tts)

    def _timer_flush_tts(self) -> None:
        """Timer callback: flush whatever is in the buffer, even without punctuation."""
        self._tts_flush_timer = None
        # Don't flush while inside an unclosed code fence
        if self._has_open_fence(self._tts_text_buffer):
            return
        text = self._tts_text_buffer.strip()
        if text:
            self._tts_text_buffer = ""
            self._tts_pending.append(text)
            if not self._tts_sending:
                asyncio.create_task(self._process_tts_queue())

    def _flush_tts_sentences(self) -> None:
        """Extract complete sentences from TTS buffer and queue them."""
        flushed = False
        while True:
            match = self._SENTENCE_BREAK.search(self._tts_text_buffer)
            if not match and len(self._tts_text_buffer) > 200:
                match = self._LONG_BREAK.search(self._tts_text_buffer)
            if not match:
                break
            idx = match.start() + 1  # include punctuation
            sentence = self._tts_text_buffer[:idx].strip()
            self._tts_text_buffer = self._tts_text_buffer[idx:].lstrip()
            if sentence:
                self._tts_pending.append(sentence)
                flushed = True
        # Cancel timer when we got sentence breaks
        if flushed and self._tts_flush_timer:
            self._tts_flush_timer.cancel()
            self._tts_flush_timer = None
        if self._tts_pending and not self._tts_sending:
            asyncio.create_task(self._process_tts_queue())

    async def _process_tts_queue(self) -> None:
        """Process pending TTS sentences sequentially (ensures correct order)."""
        self._tts_sending = True
        try:
            while self._tts_pending:
                text = self._tts_pending.pop(0)
                self._tts_segment_idx += 1
                await self._send_tts_segment(text, self._tts_segment_idx)
        finally:
            self._tts_sending = False

    # Emoji pattern: covers most common emoji ranges + variation selectors + ZWJ sequences
    _EMOJI_RE = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map
        "\U0001F900-\U0001F9FF"  # supplemental symbols
        "\U0001FA00-\U0001FA6F"  # chess symbols
        "\U0001FA70-\U0001FAFF"  # symbols extended-A
        "\U00002702-\U000027B0"  # dingbats
        "\U0000FE00-\U0000FE0F"  # variation selectors
        "\U0000200D"             # zero width joiner
        "\U000024C2-\U0001F251"  # enclosed characters
        "\U00002600-\U000026FF"  # misc symbols
        "\U00002300-\U000023FF"  # misc technical
        "]+", flags=re.UNICODE
    )

    @classmethod
    def _clean_text_for_tts(cls, text: str) -> str:
        """Strip URLs, markdown links, emojis, and other non-speakable content."""
        # Replace [label](url) with just label
        text = cls._MD_LINK_RE.sub(r'\1', text)
        # Remove bare URLs
        text = cls._URL_RE.sub('', text)
        # Remove markdown bold/italic markers
        text = text.replace('**', '').replace('__', '')
        # Remove emojis
        text = cls._EMOJI_RE.sub('', text)
        # Remove mail icon (✉️) and similar text symbols
        text = text.replace('✉️', '').replace('✉', '')
        # Collapse whitespace
        return ' '.join(text.split()).strip()

    async def _send_tts_segment(self, text: str, segment: int) -> None:
        """Synthesize one sentence and send as a TTS audio segment."""
        try:
            text = self._clean_text_for_tts(text)
            if not text:
                return
            service = self._get_speech_service()
            logger.info(f"[TTS] Synthesizing segment {segment}: {text[:60]}...")
            mp3_bytes = await service.synthesize_fast(text, locale=self._locale, voice=self._tts_voice)
            audio_b64 = base64.b64encode(mp3_bytes).decode("ascii")
            await self._send({
                "type": "tts_audio",
                "audio_b64": audio_b64,
                "segment": segment,
            })
            logger.info(f"[TTS] Sent segment {segment} ({len(mp3_bytes)} bytes)")
        except Exception as e:
            logger.warning(f"[TTS] Segment {segment} failed: {e}")

    async def _flush_remaining_tts(self) -> None:
        """Flush remaining TTS buffer and send done marker."""
        # Cancel pending flush timer
        if self._tts_flush_timer:
            self._tts_flush_timer.cancel()
            self._tts_flush_timer = None

        # Strip any remaining code blocks from buffer
        self._tts_text_buffer = self._CODE_BLOCK_RE.sub(' ', self._tts_text_buffer)
        # Strip trailing unclosed fence (``` to end)
        if self._has_open_fence(self._tts_text_buffer):
            idx = self._tts_text_buffer.rfind('```')
            if idx >= 0:
                self._tts_text_buffer = self._tts_text_buffer[:idx]

        remaining = self._tts_text_buffer.strip()
        self._tts_text_buffer = ""
        if remaining:
            self._tts_pending.append(remaining)
        if self._tts_pending and not self._tts_sending:
            await self._process_tts_queue()
        elif self._tts_sending:
            # Wait for current queue processing to finish
            while self._tts_sending:
                await asyncio.sleep(0.05)
        await self._send({"type": "tts_done"})
        self._tts_segment_idx = 0

        # Title generation — completely decoupled, fire-and-forget, non-stacking
        self._schedule_title_generation()

    def _schedule_title_generation(self) -> None:
        """Fire-and-forget title generation. Cancels any in-flight title task to prevent stacking."""
        if self._title_task and not self._title_task.done():
            self._title_task.cancel()
        self._title_task = asyncio.create_task(self._generate_title_background())

    async def _generate_title_background(self) -> None:
        """Generate title in isolation — never blocks save or conversation flow."""
        try:
            await self._generate_title()
            # Re-save so the title is persisted
            data = self._serialize_conversation()
            if data:
                await self._send({"type": "save_conversation", "data": data})
        except asyncio.CancelledError:
            pass  # Superseded by a newer title task
        except Exception as e:
            logger.debug(f"[TITLE] Background title generation failed: {e}")

    def _on_response_cancelled(self) -> None:
        asyncio.create_task(self._send({"type": "response_cancelled"}))

    def _on_error(self, error: Exception) -> None:
        error_type = type(error).__name__
        if "InsufficientBudgetError" in error_type:
            msg = "Insufficient budget available"
        else:
            msg = str(error)
        asyncio.create_task(self._send({
            "type": "error",
            "error_type": error_type,
            "message": msg,
        }))

    def _on_model_switched(self, old_model: str, new_model: str) -> None:
        old_info = llm_manager.get_model_info(old_model)
        new_info = llm_manager.get_model_info(new_model)
        asyncio.create_task(self._send({
            "type": "model_switched",
            "old_model": old_model,
            "new_model": new_model,
            "old_label": old_info.label if old_info else old_model,
            "new_label": new_info.label if new_info else new_model,
            "old_icon": old_info.model_icon if old_info else "",
            "new_icon": new_info.model_icon if new_info else "",
            "available_tools": self.get_all_known_tools(),
        }))

    # ── Context info (mirrors ChatView._update_context_preview) ───

    async def _send_context_info(self) -> None:
        info = self._compute_context_info()
        if info:
            await self._send({"type": "context_info", **info})

    def _compute_context_info(self) -> Optional[dict]:
        try:
            model_info = llm_manager.get_model_info(self.model)
            if not model_info:
                return None
            max_input = self.max_input_tokens or model_info.max_input_tokens

            base_tokens = len(self._app_system_prompt or "") // 4
            master_tokens = len(getattr(self, "_master_prompt", "") or "") // 4
            nudge_tokens = 0
            project_tokens = 0
            if self._nudge_id and self._base_system_prompt:
                nudge_tokens = max(0, len(self._base_system_prompt) - len(self._app_system_prompt or "")) // 4
            elif self._project_id and self._base_system_prompt:
                project_tokens = max(0, len(self._base_system_prompt) - len(self._app_system_prompt or "")) // 4
            doc_tokens = 0
            sp = self.system_prompt or ""
            marker = "\n\n---\n[Attached documents context]"
            idx = sp.find(marker)
            if idx != -1:
                doc_tokens = len(sp[idx:]) // 4

            history_tokens = 0
            image_count = 0
            image_tokens = 0
            if self.session._condensed_summary:
                history_tokens += len(self.session._condensed_summary) // 4
            for msg in self.session.history.messages:
                if msg.content_stale:
                    continue
                history_tokens += len(msg.content) // 4 + 4
                if msg.images and not msg.images_stale:
                    img_count = len(msg.images)
                    image_count += img_count
                    image_tokens += img_count * 765

            tool_tokens = 0
            try:
                registry = get_default_registry()
                for tool_name in self.enabled_tools:
                    tool_def = registry.get(tool_name, self.provider)
                    if tool_def:
                        tool_tokens += len(json.dumps(tool_def.to_mcp_dict())) // 4
            except Exception:
                pass

            total = base_tokens + master_tokens + nudge_tokens + project_tokens + doc_tokens + history_tokens + image_tokens + tool_tokens
            pct_exact = min(100.0, total / max_input * 100) if max_input > 0 else 0.0
            est_cost = (total * model_info.input_token_price + 500 * model_info.output_token_price) / 1_000_000

            return {
                "pct": round(pct_exact),
                "pctExact": round(pct_exact, 2),
                "historyTokens": history_tokens,
                "masterTokens": master_tokens,
                "nudgeTokens": nudge_tokens,
                "projectTokens": project_tokens,
                "docTokens": doc_tokens,
                "imageTokens": image_tokens,
                "imageCount": image_count,
                "toolTokens": tool_tokens,
                "totalTokens": total,
                "maxTokens": max_input,
                "estCost": f"{est_cost:.2f}" if est_cost >= 0.01 else "< 0.01",
            }
        except Exception as e:
            logger.debug(f"[CONTEXT-INFO] Error: {e}")
            return None

    # ── Prompt Inspector (admin-only deep inspection) ───────────────

    def _compute_prompt_inspector(self) -> Optional[dict]:
        """Build detailed prompt section breakdown for admin debugging."""
        try:
            model_info = llm_manager.get_model_info(self.model)
            if not model_info:
                return None
            max_input = self.max_input_tokens or model_info.max_input_tokens

            sections = []

            # 1. Context Preamble
            preamble = getattr(self.session, "_context_preamble", "") or ""
            if preamble:
                sections.append({
                    "id": "preamble", "label": "Context Preamble",
                    "tokens": len(preamble) // 4, "chars": len(preamble),
                    "color": "#3b82f6", "content": preamble,
                })

            # 2. System Prompt (base, excluding documents section)
            sp = self.system_prompt or ""
            doc_marker = "\n\n---\n[Attached documents context]"
            doc_idx = sp.find(doc_marker)
            sp_clean = sp[:doc_idx] if doc_idx != -1 else sp
            # Strip master prompt from display if present
            master = getattr(self, "_master_prompt", "") or ""
            if master and sp_clean.startswith(master):
                sp_display = sp_clean[len(master):].lstrip("\n")
            else:
                sp_display = sp_clean
            if sp_display:
                sections.append({
                    "id": "system", "label": "System Prompt",
                    "tokens": len(sp_display) // 4, "chars": len(sp_display),
                    "color": "#8b5cf6", "content": sp_display,
                })

            # 2b. Master prompt (if present)
            if master:
                sections.append({
                    "id": "master", "label": "Master Prompt",
                    "tokens": len(master) // 4, "chars": len(master),
                    "color": "#a855f7", "content": master,
                })

            # 3. Attached Documents
            doc_text = sp[doc_idx:] if doc_idx != -1 else ""
            sections.append({
                "id": "documents", "label": "Attached Documents",
                "tokens": len(doc_text) // 4 if doc_text else 0,
                "chars": len(doc_text),
                "color": "#f59e0b", "content": doc_text,
            })

            # 4. Condensed Summary
            condensed = getattr(self.session, "_condensed_summary", "") or ""
            sections.append({
                "id": "condensed", "label": "Condensed Summary",
                "tokens": len(condensed) // 4 if condensed else 0,
                "chars": len(condensed),
                "color": "#6b7280", "content": condensed,
            })

            # 5. Auto-Discover Suffix
            suffix = getattr(self.session, "_system_prompt_suffix", "") or ""
            sections.append({
                "id": "suffix", "label": "Auto-Discover Suffix",
                "tokens": len(suffix) // 4 if suffix else 0,
                "chars": len(suffix),
                "color": "#10b981", "content": suffix,
            })

            # 6. Messages
            msg_items = []
            msg_total = 0
            for msg in self.session.history.messages:
                if msg.content_stale:
                    continue
                mtk = len(msg.content) // 4 + 4
                img_tk = 0
                if msg.images and not msg.images_stale:
                    img_tk = len(msg.images) * 765
                mtk += img_tk
                msg_total += mtk
                msg_items.append({
                    "role": msg.role.value if hasattr(msg.role, 'value') else str(msg.role),
                    "tokens": mtk,
                    "preview": msg.content[:80],
                })
            sections.append({
                "id": "messages", "label": "Messages",
                "tokens": msg_total, "color": "#ec4899",
                "items": msg_items,
            })

            # 7. Tool Definitions
            tool_items = []
            tool_total = 0
            try:
                registry = get_default_registry()
                for tool_name in self.enabled_tools:
                    tool_def = registry.get(tool_name, self.provider)
                    if tool_def:
                        ttk = len(json.dumps(tool_def.to_mcp_dict())) // 4
                        tool_total += ttk
                        tool_items.append({"name": tool_name, "tokens": ttk})
            except Exception:
                pass
            sections.append({
                "id": "tools", "label": "Tool Definitions",
                "tokens": tool_total, "color": "#f97316",
                "count": len(tool_items), "items": tool_items,
            })

            total = sum(s["tokens"] for s in sections)
            return {
                "type": "prompt_inspector",
                "model": model_info.label or self.model,
                "sections": sections,
                "totalTokens": total,
                "maxTokens": max_input,
            }
        except Exception as e:
            logger.debug(f"[PROMPT-INSPECTOR] Error: {e}")
            return None

    # ── Document context sync (mirrors ChatView._sync_document_context) ───

    def sync_document_context(self, upload_manager: Optional[UploadManager]) -> None:
        """Rebuild system prompt = base prompt + attached document texts."""
        from llming_lodge.documents import extract_text, truncate_to_token_budget

        raw_base = self._base_system_prompt or self.system_prompt or ""

        # Prepend master prompt (always present, survives nudge switches)
        master = getattr(self, "_master_prompt", "") or ""
        base = (master + "\n\n" + raw_base).strip() if master else raw_base

        # Strip any previous document section
        marker = "\n\n---\n[Attached documents context]"
        idx = base.find(marker)
        if idx != -1:
            base = base[:idx]

        # Dynamic budget: 30% of model context, capped at 150k tokens
        doc_token_budget = min(int(self.max_input_tokens * 0.30), 150_000)

        doc_section = ""
        if upload_manager:
            raw_docs = []
            for f in upload_manager.files:
                if f.mime_type.startswith("image/"):
                    continue
                if not f.text_content and f.raw_data:
                    f.text_content = extract_text(f.raw_data, f.mime_type)
                raw_docs.append((f.name, f.text_content))
            if raw_docs:
                result = truncate_to_token_budget(raw_docs, max_tokens=doc_token_budget)
                doc_texts = [f"### {name}\n{text}" for name, text in result.documents]
                doc_section = marker + "\n" + "\n\n".join(doc_texts)

                # Notify user if documents were truncated
                if result.was_truncated:
                    pages_estimate = result.total_tokens_before // 500
                    budget_k = doc_token_budget // 1000
                    import asyncio
                    asyncio.ensure_future(self._send({
                        "type": "notification",
                        "message": (
                            f"Document limit of {budget_k}k tokens exceeded "
                            f"(~{pages_estimate} pages). Some content was truncated."
                        ),
                        "level": "negative",
                    }))

        self._base_system_prompt = base
        final_prompt = base + doc_section
        logger.info("[SYNC_DOC] base=%d chars, doc_section=%d chars, final=%d chars, files=%d, budget=%dk",
                     len(base), len(doc_section), len(final_prompt),
                     len(upload_manager.files) if upload_manager else 0,
                     doc_token_budget // 1000)
        self.update_settings(system_prompt=final_prompt)

    # ── Conversation serialization (mirrors ChatSessionHandler) ───

    def _serialize_conversation(self) -> Optional[dict]:
        if not self.session:
            return None
        messages = self.session.history.messages
        if not messages:
            return None

        title = self._conversation_title
        if not title:
            first_user = next((m for m in messages if m.role == Role.USER), None)
            raw = (first_user.content or "").strip() if first_user else ""
            if raw:
                # Take first ~30 chars, truncate at word boundary
                words = raw.split()
                title = ""
                for w in words:
                    candidate = f"{title} {w}".strip() if title else w
                    if len(candidate) > 30:
                        break
                    title = candidate
                if not title:
                    title = raw[:30]
                if len(raw) > len(title):
                    title += "…"
            else:
                title = "Untitled"

        now = datetime.now().isoformat()
        created_at = messages[0].timestamp.isoformat() if messages else now

        result = {
            "id": self.session_id,
            "title": title,
            "created_at": created_at,
            "updated_at": now,
            "model": self.model,
            "provider": self.provider,
            "messages": [m.model_dump(mode="json") for m in messages],
            "condensed_summary": self.session._condensed_summary,
            "base_system_prompt": self._base_system_prompt,
            "config": self.config.model_dump(mode="json"),
            "enabled_tools": list(self.enabled_tools),
        }
        if self._project_id:
            result["project_id"] = self._project_id
        if self._nudge_id:
            result["nudge_id"] = self._nudge_id
        return result

    # ── Title generation (mirrors ChatSessionHandler) ──────────

    async def _generate_title(self) -> None:
        if not self.session:
            return
        user_messages = [
            m for m in self.session.history.messages
            if m.role == Role.USER and not m.content_stale
        ]
        if len(user_messages) < 1:
            return
        if self._title_msg_count >= len(user_messages) and self._conversation_title:
            return
        self._title_msg_count = len(user_messages)

        # Extract context
        def first_n_sentences(text: str, n: int = 2) -> str:
            parts = re.split(r"(?<=[.!?])\s+", text.strip())
            return " ".join(parts[:n])

        def last_n_sentences(text: str, n: int = 2) -> str:
            parts = re.split(r"(?<=[.!?])\s+", text.strip())
            return " ".join(parts[-n:])

        context_parts = [first_n_sentences(user_messages[0].content)]
        if len(user_messages) > 1:
            context_parts.append(last_n_sentences(user_messages[-1].content))
        context = "\n".join(context_parts)
        if not context:
            return

        try:
            from llming_lodge.providers.llm_provider_models import ReasoningEffort
            from llming_lodge.messages import LlmSystemMessage, LlmHumanMessage

            provider = self.session._provider
            all_models = provider.get_models()
            candidates = sorted(all_models, key=lambda m: m.input_token_price)
            condense_model = candidates[0].model if candidates else self.model

            client = provider.create_client(
                provider=self.config.provider,
                model=condense_model,
                base_url=self.config.base_url,
                temperature=0.3,
                max_tokens=25,
                toolboxes=[],
                reasoning_effort=ReasoningEffort.NONE,
            )

            messages = [
                LlmSystemMessage(content="Generate a short conversation title (max 5 words, max 30 characters). Output ONLY the title, nothing else."),
                LlmHumanMessage(content=context),
            ]

            response = await asyncio.wait_for(client.ainvoke(messages), timeout=10)
            title = response.content.strip().strip("\"'")
            if title and len(title) > 30:
                title = title[:28].rsplit(" ", 1)[0] + "..."

            if title and len(title) > 1:
                self._conversation_title = title
                await self._send({"type": "title_generated", "title": title})
        except Exception as e:
            logger.debug(f"[TITLE] Title generation failed (non-critical): {e}")

    # ── Condensation hooks ────────────────────────────────────

    def _wire_condensation(self) -> None:
        """Wire session condensation callbacks to send WS events."""
        self.session.on_condense_start = lambda: asyncio.create_task(
            self._send({"type": "condense_start"})
        )
        self.session.on_condense_progress = lambda pct: asyncio.create_task(
            self._send({"type": "condense_progress", "pct": pct})
        )
        self.session.on_condense_end = lambda: asyncio.create_task(
            self._on_condense_end()
        )

    async def _on_condense_end(self) -> None:
        await self._send({"type": "condense_end"})
        await self._send_context_info()
        # Save after condensation
        data = self._serialize_conversation()
        if data:
            await self._send({"type": "save_conversation", "data": data})

    # ── Session init payload ──────────────────────────────────

    @staticmethod
    def _context_label(tokens: int) -> str:
        """Human-readable context window label, e.g. '1M', '272K', '32K'."""
        if tokens >= 1_000_000:
            v = tokens / 1_000_000
            return f"{v:g}M"
        return f"{tokens // 1000}K"

    async def build_session_init(self, user_name: str = "") -> dict:
        """Build the session_init message sent when WS connects."""
        import math

        models = []
        for info in llm_manager.get_available_llms():
            # Normalize cost to 1–10 scale (log scale, blended input+output)
            # Weight input 4:1 vs output — input dominates real-world cost
            inp = info.input_token_price or 0
            out = info.output_token_price or 0
            price = 0.8 * inp + 0.2 * out
            cost = min(10, max(1, round(math.log10(max(price, 0.1)) / math.log10(75) * 10))) if price else 1

            # Normalize memory to 1–10 scale (log scale: 16K→1, 1M→10)
            mem = info.max_input_tokens or 16000
            memory = min(10, max(1, round(
                (math.log2(mem) - math.log2(16000)) / (math.log2(1_000_000) - math.log2(16000)) * 9 + 1
            )))

            models.append({
                "model": info.model,
                "label": info.label,
                "icon": info.model_icon,
                "provider": llm_manager.get_provider_for_model(info.model),
                "max_input_tokens": info.max_input_tokens,
                "max_output_tokens": info.max_output_tokens,
                "popularity": info.popularity,
                "speed": info.speed,
                "quality": info.quality,
                "cost": cost,
                "memory": memory,
                "context_label": self._context_label(info.max_input_tokens or 0),
                "best_use": info.best_use,
                "highlights": info.highlights,
            })

        quick_actions = self._custom_quick_actions or []

        # Resolve budget: prefer budget_handler if set
        budget = await self.available_budget_async()
        if self.budget_handler:
            try:
                info = self.budget_handler()
                budget = info.get("available", budget)
            except Exception as e:
                logger.debug(f"[BUDGET] budget_handler failed: {e}")

        # Supported languages for mid-chat language switching (configured by host app)
        supported_languages = self._supported_languages

        return {
            "type": "session_init",
            "session_id": self.session_id,
            "user_name": user_name,
            "user_avatar": self.user_avatar,
            "models": models,
            "current_model": self.model,
            "tools": self.get_all_known_tools(),
            "budget": budget,
            "system_prompt": self._base_system_prompt or self.system_prompt,
            "temperature": self.temperature,
            "max_input_tokens": self.max_input_tokens,
            "max_output_tokens": self.max_output_tokens,
            "quick_actions": quick_actions,
            "locale": self._locale,
            "supported_languages": supported_languages,
            "tts_voices": [{"id": v[0], "label": v[1], "gender": v[2]} for v in self._get_tts_voices()],
            "tts_default_voice": self._tts_default_voice or self._get_speech_service_default_voice(),
            "speech_max_tokens": self._speech_max_tokens,
            "client_renderers": getattr(self.session, '_mcp_client_renderers', None) or [],
        }


# ── WebSocket Router Builder ────────────────────────────────────────


def build_ws_router():
    """Build a FastAPI APIRouter with the WebSocket chat endpoint."""
    from fastapi import APIRouter

    from llming_lodge.server import API_PREFIX
    router = APIRouter(prefix=API_PREFIX)

    @router.websocket("/ws/{session_id}")
    async def websocket_chat(ws: WebSocket, session_id: str):
        registry = SessionRegistry.get()
        entry = registry.get_session(session_id)
        if not entry:
            await ws.close(code=4004, reason="Session not found")
            return

        await ws.accept()
        controller = entry.controller
        controller.set_websocket(ws)
        entry.websocket = ws

        # Wire condensation callbacks
        controller._wire_condensation()

        # Send session init
        init_msg = await controller.build_session_init(user_name=entry.user_name)
        await ws.send_json(init_msg)

        # Send initial context info
        await controller._send_context_info()

        # Send MCP client-side renderers (if discovery already completed)
        renderers = getattr(controller.session, '_mcp_client_renderers', None)
        if renderers:
            await controller._send({
                "type": "register_renderers",
                "renderers": renderers,
            })

        # Send favorite nudges if store is available
        if getattr(controller, "_nudge_store", None) and controller.user_mail:
            try:
                favs = await controller._nudge_store.get_favorites(controller.user_mail)
                await controller._send({
                    "type": "nudge_favorites_result",
                    "nudges": favs,
                })
            except Exception as e:
                logger.warning(f"[NUDGE] Failed to send favorites: {e}")

        logger.info(f"[WS] Client connected to session {session_id}")

        try:
            while True:
                raw = await ws.receive_text()
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    await ws.send_json({"type": "error", "error_type": "InvalidJSON", "message": "Invalid JSON"})
                    continue

                entry.last_activity = time.monotonic()
                await _handle_client_message(controller, entry, msg)

        except WebSocketDisconnect:
            logger.info(f"[WS] Client disconnected from session {session_id}")
        except Exception as e:
            logger.error(f"[WS] Error in session {session_id}: {type(e).__name__}: {e}")
        finally:
            controller.set_websocket(None)
            entry.websocket = None

    return router


async def _handle_rt_tool_call(controller: "WebSocketChatController", event: dict) -> None:
    """Execute a Realtime tool call server-side and send result back to Azure."""
    call_id = event.get("call_id", "")
    name = event.get("name", "")
    arguments_str = event.get("arguments", "{}")
    azure_ws = getattr(controller, "_rt_ws", None)
    if not azure_ws:
        return
    try:
        arguments = json.loads(arguments_str) if isinstance(arguments_str, str) else arguments_str
        result = await get_default_registry().execute(name, arguments)
        result_str = str(result)
    except Exception as e:
        logger.error(f"[REALTIME] Tool call '{name}' failed: {e}")
        result_str = f"Error: {e}"

    # Send function_call_output back to Azure
    await azure_ws.send(json.dumps({
        "type": "conversation.item.create",
        "item": {
            "type": "function_call_output",
            "call_id": call_id,
            "output": result_str,
        },
    }))
    # Trigger response generation
    await azure_ws.send(json.dumps({"type": "response.create"}))


def _resolve_chat_file_attachments(raw_atts: list, upload_manager) -> list:
    """Resolve ``chat_file`` attachments by reading file bytes from UploadManager.

    ``chat_file`` entries have ``{type: 'chat_file', fileId: '...', name, content_type}``.
    They are converted to ``{type: 'file', name, content_type, data: <base64>, size}``.
    Other attachment types are passed through unchanged.
    """
    if not raw_atts:
        return raw_atts
    resolved = []
    for att in raw_atts:
        if att.get("type") == "chat_file" and att.get("fileId") and upload_manager:
            file_id = att["fileId"]
            found = False
            for f in upload_manager.files:
                if f.file_id == file_id:
                    try:
                        if not f.raw_data:
                            logger.warning("[EMAIL-WS] chat_file %s has no raw_data", file_id)
                            break
                        content_b64 = base64.b64encode(f.raw_data).decode()
                        resolved.append({
                            "type": "file",
                            "name": f.name,
                            "content_type": f.mime_type,
                            "data": content_b64,
                            "size": f.size,
                        })
                        logger.info("[EMAIL-WS] Resolved chat_file %s (%d bytes)", f.name, f.size)
                        found = True
                    except Exception as e:
                        logger.warning("[EMAIL-WS] Failed to read chat_file %s: %s", file_id, e)
                    break
            if not found:
                logger.warning("[EMAIL-WS] chat_file not found: %s", file_id)
        else:
            resolved.append(att)
    return resolved


async def _handle_client_message(
    controller: WebSocketChatController,
    entry: SessionEntry,
    msg: dict,
) -> None:
    """Dispatch a client WebSocket message to the appropriate handler."""
    msg_type = msg.get("type", "")

    if msg_type == "send_message":
        text = msg.get("text", "").strip()
        images = msg.get("images")
        if text or images:
            # Dev MCP intercept: check if text matches a secret command
            if text and not images and controller._on_message_intercept:
                try:
                    intercept_result = await controller._on_message_intercept(text, controller)
                except Exception as e:
                    logger.warning("[WS] on_message_intercept error: %s", e)
                    intercept_result = None
                if intercept_result is not None:
                    # Synthetic assistant response (user bubble already rendered client-side)
                    model_info = llm_manager.get_model_info(controller.model)
                    await controller._send({
                        "type": "response_started",
                        "model": controller.model,
                        "model_icon": model_info.model_icon if model_info else "",
                        "model_label": model_info.label if model_info else controller.model,
                    })
                    await controller._send({"type": "text_chunk", "content": intercept_result})
                    await controller._send({"type": "response_completed"})
                    await controller._send({
                        "type": "tools_updated",
                        "tools": controller.get_all_known_tools(),
                    })
                    return

            async def _run_send():
                try:
                    await controller.send_message(text, images=images)
                except Exception as e:
                    logger.error(f"[WS] send_message error: {e}")
            asyncio.create_task(_run_send())

    elif msg_type == "stop_streaming":
        await controller.stop_streaming()

    elif msg_type == "switch_model":
        model = msg.get("model", "")
        if model:
            await controller.switch_model(model)

    elif msg_type == "update_settings":
        kwargs = {}
        if "temperature" in msg:
            kwargs["temperature"] = msg["temperature"]
        if "max_input_tokens" in msg:
            kwargs["max_input_tokens"] = msg["max_input_tokens"]
        if "max_output_tokens" in msg:
            kwargs["max_output_tokens"] = msg["max_output_tokens"]
        if "system_prompt" in msg:
            kwargs["system_prompt"] = msg["system_prompt"]
            controller._base_system_prompt = msg["system_prompt"]
        if "speech_response" in msg:
            enabled = bool(msg["speech_response"])
            controller._speech_response = enabled
            max_tok = controller._speech_max_tokens
            if enabled:
                controller._saved_max_output_tokens = controller.max_output_tokens
                # Save the FULL system prompt (including document context)
                controller._saved_system_prompt = controller.system_prompt
                kwargs["max_output_tokens"] = max_tok
                # Append conciseness instruction to the full system prompt
                concise_hint = (
                    f"\nKeep your response concise — aim for approximately "
                    f"{max_tok // 2} tokens (~{max_tok // 4} words). "
                    f"Be brief but complete your sentences naturally."
                )
                kwargs["system_prompt"] = (controller.system_prompt or "") + concise_hint
            else:
                if controller._saved_max_output_tokens is not None:
                    kwargs["max_output_tokens"] = controller._saved_max_output_tokens
                    controller._saved_max_output_tokens = None
                if controller._saved_system_prompt is not None:
                    kwargs["system_prompt"] = controller._saved_system_prompt
                    controller._saved_system_prompt = None
        if "tts_voice" in msg:
            controller._tts_voice = str(msg["tts_voice"])
        if kwargs:
            controller.update_settings(**kwargs)
            await controller._send_context_info()

    elif msg_type == "toggle_tool":
        name = msg.get("name", "")
        enabled = msg.get("enabled", True)
        restore = msg.get("restore", False)
        if name:
            controller.toggle_tool(name, enabled)
            if not restore:
                # Echo updated tools back to client (skip for restore —
                # client already has the correct local state, and the echo
                # would consume _toolPrefsNeedReapply before MCP discovery)
                await controller._send({
                    "type": "tools_updated",
                    "tools": controller.get_all_known_tools(),
                })
            await controller._send_context_info()
            # Show notification when enabling a tool that has one (skip on restore from prefs)
            if enabled and not restore and name in controller._tool_toggle_notifications:
                await controller._send({
                    "type": "action_callback_result",
                    "action_id": f"tool:{name}",
                    "notification": controller._tool_toggle_notifications[name],
                    "notification_type": "warning",
                })

    elif msg_type == "new_chat":
        # Save current conversation first
        data = controller._serialize_conversation()
        if data:
            await controller._send({"type": "save_conversation", "data": data})

        # Dev MCP cleanup
        if controller._on_new_chat:
            try:
                await controller._on_new_chat(controller)
            except Exception as e:
                logger.warning("[WS] on_new_chat error: %s", e)

        # Clear history and create new session
        controller.clear_history()
        old_session_id = controller.session_id
        new_session_id = str(uuid4())
        controller.session_id = new_session_id
        controller._conversation_title = None
        controller._title_msg_count = 0
        controller._project_id = None
        controller._nudge_id = None

        # Migrate auto-discover session context to new session ID
        if old_session_id in _discoverable_sessions:
            _discoverable_sessions[new_session_id] = _discoverable_sessions.pop(old_session_id)
            controller.tool_config["consult_nudge"] = {"_session_id": new_session_id}
            if "consult_nudge" not in controller.enabled_tools:
                controller.enabled_tools.append("consult_nudge")

        # Migrate browser MCP session context to new session ID
        if old_session_id in _browser_mcp_sessions:
            bctx = _browser_mcp_sessions.pop(old_session_id)
            # Stop all active Workers (new chat = fresh slate)
            for nudge_uid in list(bctx.get("active_mcp_nudges", set())):
                await controller._send({"type": "stop_browser_mcp", "nudge_uid": nudge_uid})
                # Unregister tools from registry
                for tool_name in bctx.get("active_tool_names", {}).get(nudge_uid, []):
                    get_default_registry().unregister(tool_name)
                    get_default_registry()._mcp_connections.pop(tool_name, None)
                    if tool_name in controller.enabled_tools:
                        controller.enabled_tools.remove(tool_name)
                    if tool_name in controller.available_tools:
                        controller.available_tools.remove(tool_name)
            bctx["active_mcp_nudges"] = set()
            bctx["active_tool_names"] = {}
            bctx["pending_requests"] = {}
            _browser_mcp_sessions[new_session_id] = bctx
            controller.tool_config["activate_mcp_nudge"] = {"_session_id": new_session_id}

        # Restore app-level default system prompt
        controller._base_system_prompt = controller._app_system_prompt
        if controller._base_system_prompt:
            controller.update_settings(system_prompt=controller._base_system_prompt)

        # Apply preset if provided (project or nudge)
        preset = msg.get("preset")
        nudge_meta = None
        if preset:
            preset_type = preset.get("type", "project")

            # Server-side nudge fetch: nudge_uid → fetch from MongoDB
            if preset_type == "nudge" and preset.get("nudge_uid") and getattr(controller, "_nudge_store", None):
                try:
                    store = controller._nudge_store
                    user_teams = getattr(controller, "_user_teams", None)
                    nudge = await store.get_for_user(preset["nudge_uid"], controller.user_mail or "", user_teams=user_teams)
                    if nudge:
                        controller._nudge_id = nudge["uid"]
                        nudge_prompt = nudge.get("system_prompt") or ""
                        base = getattr(controller, "_nudge_base_system_prompt", "")
                        if nudge_prompt or base:
                            effective_prompt = (base + "\n\n" + nudge_prompt).strip() if base and nudge_prompt else (base or nudge_prompt)
                            controller._base_system_prompt = effective_prompt
                            controller.update_settings(system_prompt=effective_prompt)
                        if nudge.get("model"):
                            try:
                                await controller.switch_model(nudge["model"])
                            except Exception as ex:
                                logger.warning(f"[NUDGE] Could not switch to model {nudge['model']}: {ex}")
                        if nudge.get("language") and nudge["language"] != "auto":
                            await _handle_language_change(controller, nudge["language"])
                        # Fetch cached file texts (in-memory, freshness-checked)
                        from llming_lodge.nudge_store import get_file_cache
                        preset["_cached_files"] = await get_file_cache().get_files(
                            preset["nudge_uid"], store, controller.user_mail or "",
                            nudge=nudge,
                        )
                        nudge_meta = {
                            "uid": nudge.get("uid"),
                            "name": nudge.get("name"),
                            "icon": nudge.get("icon"),
                            "description": nudge.get("description"),
                            "creator_name": nudge.get("creator_name"),
                            "suggestions": nudge.get("suggestions"),
                            "capabilities": nudge.get("capabilities"),
                            "doc_plugins": nudge.get("doc_plugins"),
                            "mode": nudge.get("mode"),
                            "category": nudge.get("category"),
                            "sub_category": nudge.get("sub_category"),
                        }

                        # Auto-activate browser MCP if nudge has JS files.
                        # NOTE: This must be deferred via create_task — NOT awaited
                        # here.  The WS receive loop is blocked inside this handler,
                        # so the browser's ``browser_mcp_result`` reply would never
                        # be read, causing a deadlock / 15-s timeout.  Scheduling
                        # the activation as a background task lets this handler
                        # finish (sends ``chat_cleared`` to the client) and the
                        # receive loop to resume, so the reply is processed normally.
                        _nudge_has_js = any(
                            (f.get("name", "").endswith(".js") or f.get("name", "").endswith(".mjs"))
                            for f in nudge.get("files", [])
                        )
                        if _nudge_has_js:
                            _uid = nudge["uid"]
                            if new_session_id not in _browser_mcp_sessions:
                                _browser_mcp_sessions[new_session_id] = {
                                    "store": store,
                                    "user_email": controller.user_mail or "",
                                    "user_teams": user_teams or [],
                                    "uids": {_uid},
                                    "loop": asyncio.get_running_loop(),
                                    "controller": controller,
                                    "pending_requests": {},
                                    "active_mcp_nudges": set(),
                                    "active_tool_names": {},
                                }
                            else:
                                _browser_mcp_sessions[new_session_id]["uids"].add(_uid)

                            async def _deferred_mcp_activate(_ctx, _u):
                                """Run after a short yield so chat_cleared arrives first."""
                                await asyncio.sleep(0.2)
                                try:
                                    res = await _activate_browser_mcp(_ctx, _u)
                                    logger.info("[NUDGE] Auto-activated MCP for %s: %s", _u, res)
                                except Exception as exc:
                                    logger.warning("[NUDGE] MCP auto-activation failed for %s: %s", _u, exc)

                            asyncio.create_task(
                                _deferred_mcp_activate(
                                    _browser_mcp_sessions[new_session_id], _uid,
                                )
                            )

                    else:
                        logger.warning(f"[NUDGE] Nudge {preset['nudge_uid']} not found or not visible")
                except Exception as e:
                    logger.error(f"[NUDGE] Failed to fetch nudge: {e}", exc_info=True)
            else:
                # Legacy client-side preset flow
                if preset_type == "project":
                    controller._project_id = preset.get("id")
                elif preset_type == "nudge":
                    controller._nudge_id = preset.get("id")

                preset_prompt = preset.get("system_prompt") or ""
                base = getattr(controller, "_nudge_base_system_prompt", "") if preset_type == "nudge" else ""
                if preset_prompt or base:
                    effective_prompt = (base + "\n\n" + preset_prompt).strip() if base and preset_prompt else (base or preset_prompt)
                    controller._base_system_prompt = effective_prompt
                    controller.update_settings(system_prompt=effective_prompt)
                if preset.get("model"):
                    try:
                        await controller.switch_model(preset["model"])
                    except Exception as ex:
                        logger.warning(f"[PRESET] Could not switch to model {preset['model']}: {ex}")
                if preset.get("language") and preset["language"] != "auto":
                    await _handle_language_change(controller, preset["language"])

        # Clear pasted images
        from llming_lodge.server import SessionDataStore
        SessionDataStore.clear_pasted_images(old_session_id)

        # Clean up old upload manager, create new one
        if entry.upload_manager:
            entry.upload_manager.cleanup()
        entry.upload_manager = UploadManager.create(new_session_id, entry.user_id)

        # Inject preset files into document context (after new upload manager)
        cached_files = preset.get("_cached_files") if preset else None
        client_files = preset.get("files") if preset else None
        if cached_files or client_files:
            await _inject_preset_files(
                cached_files or client_files, entry, controller,
            )

        # Always inject master nudge files (silently stacked on top)
        master_files = getattr(controller, "_master_cached_files", [])
        if master_files:
            await _inject_preset_files(master_files, entry, controller)

        # Apply doc plugin configuration from preset or nudge
        # (no-preset random chat leaves doc tools disabled; preamble still
        # describes all capabilities so the LLM can use fenced code blocks)
        if preset:
            preset_doc_plugins = (
                nudge_meta.get("doc_plugins") if nudge_meta and "doc_plugins" in (nudge_meta or {})
                else preset.get("doc_plugins")
            )
            if preset_doc_plugins is not None:
                await _apply_doc_plugins(controller, entry, preset_doc_plugins)

        # Apply nudge tool capabilities (only explicitly set ones; null = keep user setting)
        nudge_caps = nudge_meta.get("capabilities") if nudge_meta else (preset.get("capabilities") if preset else None)
        if nudge_caps:
            for tool_name, enabled in nudge_caps.items():
                if enabled is not None:
                    controller.toggle_tool(tool_name, enabled)

        # Re-register in registry under new ID
        registry = SessionRegistry.get()
        registry.remove(old_session_id)
        registry._sessions[new_session_id] = entry

        # Ensure auto-discover catalog suffix and tools are active
        ad_catalog = getattr(controller, "_auto_discover_catalog", "")
        if ad_catalog:
            if controller.session:
                controller.session._system_prompt_suffix = ad_catalog
            if "consult_nudge" not in controller.enabled_tools:
                controller.enabled_tools.append("consult_nudge")
            if new_session_id in _browser_mcp_sessions and "activate_mcp_nudge" not in controller.enabled_tools:
                controller.enabled_tools.append("activate_mcp_nudge")
            controller.update_settings(tools=controller.enabled_tools)

        # Re-wire condensation callbacks
        controller._wire_condensation()

        response = {
            "type": "chat_cleared",
            "new_session_id": new_session_id,
        }
        if controller._project_id:
            response["project_id"] = controller._project_id
        if controller._nudge_id:
            response["nudge_id"] = controller._nudge_id
        if nudge_meta:
            response["nudge_meta"] = nudge_meta
        await controller._send(response)

    elif msg_type == "load_conversation":
        data = msg.get("data")
        if data:
            await _load_conversation(controller, entry, data)

    elif msg_type == "condense":
        if controller.session:
            non_stale = [m for m in controller.session.history.messages if not m.content_stale]
            if len(non_stale) >= 2:
                await controller.session.check_and_condense(force=True)

    elif msg_type == "get_context_info":
        await controller._send_context_info()

    elif msg_type == "get_prompt_inspector":
        if getattr(controller, "_is_nudge_admin", False):
            data = controller._compute_prompt_inspector()
            if data:
                await controller._send(data)

    elif msg_type == "directory:search":
        q = msg.get("query", "")
        if not controller._directory_service:
            await controller._send({"type": "directory:search_result", "results": [], "error": "No directory service"})
        else:
            try:
                results = await controller._directory_service.search_people(q)
                await controller._send({"type": "directory:search_result", "results": results})
            except Exception as e:
                logger.warning("[DIR-WS] search_people failed: %s", e)
                await controller._send({"type": "directory:search_result", "results": [], "error": str(e)})

    elif msg_type in ("email:draft", "email:update_draft", "email:send"):
        # Resolve chat_file attachments (read from UploadManager on disk)
        _raw_atts = msg.get("attachments") or []
        _resolved_atts = _resolve_chat_file_attachments(_raw_atts, entry.upload_manager)
        logger.info("[EMAIL-WS] %s: attachments=%d (after resolve=%d)",
                     msg_type, len(_raw_atts), len(_resolved_atts))

        if msg_type == "email:draft":
            if not controller._email_service:
                await controller._send({"type": "email:action_result", "ok": False, "error": "No email service", "action": "draft"})
            else:
                try:
                    result = await controller._email_service.save_draft(
                        subject=msg.get("subject", ""),
                        to=msg.get("to", []),
                        cc=msg.get("cc", []),
                        bcc=msg.get("bcc", []),
                        body_html=msg.get("body_html", ""),
                        attachments=_resolved_atts or None,
                    )
                    await controller._send({"type": "email:action_result", **result, "action": "draft"})
                except Exception as e:
                    logger.warning("[EMAIL-WS] save_draft failed: %s", e)
                    await controller._send({"type": "email:action_result", "ok": False, "error": str(e), "action": "draft"})

        elif msg_type == "email:update_draft":
            mid = msg.get("message_id", "")
            if not controller._email_service:
                await controller._send({"type": "email:action_result", "ok": False, "error": "No email service", "action": "update_draft"})
            elif not mid:
                await controller._send({"type": "email:action_result", "ok": False, "error": "No message_id", "action": "update_draft"})
            else:
                try:
                    result = await controller._email_service.update_draft(
                        message_id=mid,
                        subject=msg.get("subject", ""),
                        to=msg.get("to", []),
                        cc=msg.get("cc", []),
                        bcc=msg.get("bcc", []),
                        body_html=msg.get("body_html", ""),
                        attachments=_resolved_atts or None,
                    )
                    await controller._send({"type": "email:action_result", **result, "action": "update_draft"})
                except Exception as e:
                    logger.warning("[EMAIL-WS] update_draft failed: %s", e)
                    await controller._send({"type": "email:action_result", "ok": False, "error": str(e), "action": "update_draft"})

        else:  # email:send
            if not controller._email_service:
                await controller._send({"type": "email:action_result", "ok": False, "error": "No email service", "action": "send"})
            else:
                try:
                    result = await controller._email_service.send_email(
                        subject=msg.get("subject", ""),
                        to=msg.get("to", []),
                        cc=msg.get("cc", []),
                        bcc=msg.get("bcc", []),
                        body_html=msg.get("body_html", ""),
                        draft_id=msg.get("message_id"),
                        attachments=_resolved_atts or None,
                    )
                    await controller._send({"type": "email:action_result", **result, "action": "send"})
                except Exception as e:
                    logger.warning("[EMAIL-WS] send_email failed: %s", e)
                    await controller._send({"type": "email:action_result", "ok": False, "error": str(e), "action": "send"})

    elif msg_type == "file_uploaded":
        # Frontend uploaded a file via HTTP; rebuild document context
        controller.sync_document_context(entry.upload_manager)
        await _sync_pdf_viewer_mcp(controller, entry)
        await controller._send_context_info()

    elif msg_type == "file_removed":
        file_id = msg.get("file_id", "")
        if file_id and entry.upload_manager:
            entry.upload_manager.remove_file(file_id, entry.user_id)
        controller.sync_document_context(entry.upload_manager)
        await _sync_pdf_viewer_mcp(controller, entry)
        await controller._send_context_info()

    elif msg_type == "conversation_list":
        # Response from browser with IDB conversation list
        convs = msg.get("conversations", [])
        if hasattr(controller, '_pending_conv_list') and controller._pending_conv_list:
            try:
                controller._pending_conv_list.set_result(convs)
            except Exception:
                pass

    elif msg_type == "preset_list":
        # Response from browser with IDB preset list
        presets = msg.get("presets", [])
        if hasattr(controller, '_pending_preset_list') and controller._pending_preset_list:
            try:
                controller._pending_preset_list.set_result(presets)
            except Exception:
                pass

    elif msg_type == "preset_detail":
        # Response from browser with full preset data
        if hasattr(controller, '_pending_preset_detail') and controller._pending_preset_detail:
            try:
                controller._pending_preset_detail.set_result(msg.get("preset"))
            except Exception:
                pass

    elif msg_type == "preset_saved_ack":
        if hasattr(controller, '_pending_preset_save') and controller._pending_preset_save:
            try:
                controller._pending_preset_save.set_result(msg.get("ok", False))
            except Exception:
                pass

    elif msg_type == "preset_deleted_ack":
        if hasattr(controller, '_pending_preset_delete') and controller._pending_preset_delete:
            try:
                controller._pending_preset_delete.set_result(msg.get("ok", False))
            except Exception:
                pass

    elif msg_type == "idb_file_list":
        if hasattr(controller, '_pending_idb_file_list') and controller._pending_idb_file_list:
            try:
                controller._pending_idb_file_list.set_result(msg.get("files", []))
            except Exception:
                pass

    elif msg_type == "idb_file_detail":
        if hasattr(controller, '_pending_idb_file_detail') and controller._pending_idb_file_detail:
            try:
                controller._pending_idb_file_detail.set_result(msg.get("file"))
            except Exception:
                pass

    elif msg_type == "idb_file_refs":
        if hasattr(controller, '_pending_idb_file_refs') and controller._pending_idb_file_refs:
            try:
                controller._pending_idb_file_refs.set_result(msg.get("file_refs", []))
            except Exception:
                pass

    elif msg_type == "inject_file_ack":
        if hasattr(controller, '_pending_inject_file_ack') and controller._pending_inject_file_ack:
            try:
                controller._pending_inject_file_ack.set_result(msg)
            except Exception:
                pass

    elif msg_type == "remove_file_ack":
        if hasattr(controller, '_pending_remove_file_ack') and controller._pending_remove_file_ack:
            try:
                controller._pending_remove_file_ack.set_result(msg)
            except Exception:
                pass

    elif msg_type == "scroll_ack":
        if hasattr(controller, '_pending_scroll_ack') and controller._pending_scroll_ack:
            try:
                controller._pending_scroll_ack.set_result(msg)
            except Exception:
                pass

    elif msg_type == "open_project_ack":
        if hasattr(controller, '_pending_open_project_ack') and controller._pending_open_project_ack:
            try:
                controller._pending_open_project_ack.set_result(msg)
            except Exception:
                pass

    elif msg_type == "open_droplet_ack":
        if hasattr(controller, '_pending_open_droplet_ack') and controller._pending_open_droplet_ack:
            try:
                controller._pending_open_droplet_ack.set_result(msg)
            except Exception:
                pass

    elif msg_type == "open_conversation_ack":
        if hasattr(controller, '_pending_open_conversation_ack') and controller._pending_open_conversation_ack:
            try:
                controller._pending_open_conversation_ack.set_result(msg)
            except Exception:
                pass

    elif msg_type == "doc_command_result":
        req_id = msg.get("request_id", "")
        pending = getattr(controller, '_pending_doc_cmds', {})
        future = pending.get(req_id)
        if future and not future.done():
            try:
                future.set_result(msg)
            except Exception:
                pass

    elif msg_type == "change_language":
        new_lang = msg.get("language", "en-us")
        await _handle_language_change(controller, new_lang)

    elif msg_type == "action_callback":
        action_id = msg.get("action_id", "")
        text = msg.get("text", "")
        await _handle_action_callback(controller, action_id, text)

    elif msg_type == "transcribe":
        audio_b64 = msg.get("audio_b64", "")
        filename = msg.get("filename", "voice.webm")
        content_type = msg.get("content_type", "audio/webm")
        if audio_b64:
            try:
                audio_bytes = base64.b64decode(audio_b64)
                if len(audio_bytes) < 1000:
                    logger.info(f"[STT] Skipping transcription: audio too small ({len(audio_bytes)} bytes)")
                    await controller._send({"type": "transcription_result", "text": ""})
                    return
                logger.info(f"[STT] Transcribing {len(audio_bytes)} bytes, content_type={content_type}")
                service = controller._get_speech_service()
                # Extract ISO-639-1 language code from locale (e.g. "de-de" → "de")
                lang = controller._locale[:2] if controller._locale else ""
                text = await service.transcribe(audio_bytes, filename, content_type, language=lang)
                await controller._send({"type": "transcription_result", "text": text})
            except Exception as e:
                logger.error(f"[STT] Transcription failed: {e}")
                await controller._send({
                    "type": "error",
                    "error_type": "TranscriptionError",
                    "message": str(e),
                })

    elif msg_type == "tts":
        text = msg.get("text", "").strip()
        if text:
            try:
                service = controller._get_speech_service()
                mp3_bytes, word_timings = await service.synthesize(text, locale=controller._locale, voice=controller._tts_voice)
                audio_b64 = base64.b64encode(mp3_bytes).decode("ascii")
                await controller._send({
                    "type": "tts_audio",
                    "audio_b64": audio_b64,
                    "word_timings": word_timings,
                })
            except Exception as e:
                logger.error(f"[TTS] Synthesis failed: {e}")
                await controller._send({
                    "type": "error",
                    "error_type": "TTSError",
                    "message": str(e),
                })

    elif msg_type == "realtime_start":
        # Start a server-proxied Realtime session (API key stays on server)
        try:
            import websockets
            from llming_lodge.speech_service import SpeechService
            service = controller._get_speech_service()
            voice = controller._tts_voice or controller._tts_default_voice or SpeechService.TTS_DEFAULT_VOICE
            instructions = controller._base_system_prompt or "You are a helpful assistant."
            lang_name = SpeechService._LOCALE_NAMES.get(controller._locale, "")
            if lang_name:
                instructions += f"\nSpeak in {lang_name}."

            # Gather tool definitions for Realtime function calling
            # Only include tools explicitly flagged for realtime use
            rt_tools = []
            from llming_lodge.tools.tool_definition import ToolSource
            for tool_name in controller.enabled_tools:
                tool_def = get_default_registry().get(tool_name)
                if tool_def and tool_def.realtime_enabled and tool_def.source != ToolSource.PROVIDER_NATIVE:
                    rt_tools.append({
                        "type": "function",
                        "name": tool_def.name,
                        "description": tool_def.description or "",
                        "parameters": tool_def.inputSchema,
                    })

            ws_url = service.get_realtime_ws_url()
            api_key = service.get_realtime_api_key()
            headers = {"api-key": api_key}

            azure_ws = await websockets.connect(ws_url, additional_headers=headers)
            controller._rt_ws = azure_ws

            # Configure the session
            session_cfg = {
                "type": "session.update",
                "session": {
                    "voice": voice,
                    "instructions": instructions,
                    "modalities": ["text", "audio"],
                    "input_audio_format": "pcm16",
                    "output_audio_format": "pcm16",
                    "input_audio_transcription": {"model": "whisper-1"},
                    "turn_detection": {"type": "server_vad"},
                },
            }
            if rt_tools:
                session_cfg["session"]["tools"] = rt_tools
            await azure_ws.send(json.dumps(session_cfg))

            await controller._send({"type": "realtime_ready"})
            logger.info("[REALTIME] Proxy session started")

            # Relay Azure → client in background
            async def _relay_from_azure():
                try:
                    async for raw_msg in azure_ws:
                        try:
                            event = json.loads(raw_msg)
                            event_type = event.get("type", "")
                            # Handle tool calls server-side
                            if event_type == "response.function_call_arguments.done":
                                await _handle_rt_tool_call(controller, event)
                                continue
                            # Forward all other events to client
                            await controller._send({"type": "rt_event", "event": event})
                        except json.JSONDecodeError:
                            pass
                except websockets.ConnectionClosed:
                    logger.info("[REALTIME] Azure WS closed")
                except Exception as e:
                    logger.warning(f"[REALTIME] Relay error: {e}")
                finally:
                    await controller._send({"type": "rt_event", "event": {"type": "session.closed"}})
                    controller._rt_ws = None

            controller._rt_relay_task = asyncio.create_task(_relay_from_azure())

        except Exception as e:
            logger.error(f"[REALTIME] Failed to start session: {e}")
            await controller._send({
                "type": "error",
                "error_type": "RealtimeError",
                "message": str(e),
            })

    elif msg_type == "rt_send":
        # Forward a client event to the Azure Realtime WS
        azure_ws = getattr(controller, "_rt_ws", None)
        if azure_ws:
            event = msg.get("event")
            if event:
                try:
                    await azure_ws.send(json.dumps(event))
                except Exception as e:
                    logger.warning(f"[REALTIME] Forward to Azure failed: {e}")

    elif msg_type == "realtime_stop":
        azure_ws = getattr(controller, "_rt_ws", None)
        if azure_ws:
            try:
                await azure_ws.close()
            except Exception:
                pass
            controller._rt_ws = None
        relay_task = getattr(controller, "_rt_relay_task", None)
        if relay_task:
            relay_task.cancel()
            controller._rt_relay_task = None

    elif msg_type == "heartbeat":
        await controller._send({"type": "heartbeat_ack"})

    elif msg_type == "save_preset":
        # Validate preset data through Pydantic and echo back
        preset_data = msg.get("data")
        if preset_data:
            try:
                from llming_lodge.chat_preset import Project, Nudge
                if preset_data.get("type") == "nudge":
                    validated = Nudge.model_validate(preset_data)
                else:
                    validated = Project.model_validate(preset_data)

                # Extract text for binary files that don't have text_content yet
                await _extract_preset_file_texts(validated)

                # Strip heavy base64 content from files to stay within WS 1 MB limit
                result = validated.model_dump(mode="json")
                for f in result.get("files", []):
                    f.pop("content", None)
                await controller._send({
                    "type": "preset_saved",
                    "data": result,
                })
            except Exception as e:
                logger.warning(f"[PRESET] Validation failed: {e}")
                await controller._send({
                    "type": "error",
                    "error_type": "PresetValidationError",
                    "message": str(e),
                })

    elif msg_type == "load_preset":
        # Apply preset settings to current session
        preset_data = msg.get("data")
        logger.info("[PRESET] load_preset received: model=%s, prompt_len=%d, type=%s",
                     preset_data.get("model") if preset_data else None,
                     len(preset_data.get("system_prompt", "")) if preset_data else 0,
                     preset_data.get("type") if preset_data else None)
        if preset_data:
            preset_type = preset_data.get("type", "project")
            if preset_type == "project":
                controller._project_id = preset_data.get("id")
            elif preset_type == "nudge":
                controller._nudge_id = preset_data.get("id")

            if preset_data.get("system_prompt") is not None:
                controller._base_system_prompt = preset_data["system_prompt"]
                controller.update_settings(system_prompt=preset_data["system_prompt"])
                logger.info("[PRESET] System prompt updated (%d chars)", len(preset_data["system_prompt"]))
            if preset_data.get("model"):
                try:
                    logger.info("[PRESET] Switching model to %s", preset_data["model"])
                    await controller.switch_model(preset_data["model"])
                except Exception as ex:
                    logger.warning(f"[PRESET] Could not switch to model {preset_data['model']}: {ex}")
            if preset_data.get("language") and preset_data["language"] != "auto":
                await _handle_language_change(controller, preset_data["language"])

            # Inject preset files into document context
            preset_files = preset_data.get("files") or []
            if preset_files:
                await _inject_preset_files(preset_files, entry, controller)

            # Apply doc plugin configuration from preset
            if "doc_plugins" in preset_data:
                await _apply_doc_plugins(controller, entry, preset_data["doc_plugins"])

            await controller._send({
                "type": "preset_applied",
                "preset_id": preset_data.get("id"),
                "preset_type": preset_type,
            })

    # ── Nudge (MongoDB) messages ────────────────────────────
    elif msg_type == "nudge_search":
        await _handle_nudge_search(controller, msg)

    elif msg_type == "nudge_get":
        await _handle_nudge_get(controller, msg)

    elif msg_type == "nudge_save":
        await _handle_nudge_save(controller, msg)

    elif msg_type == "nudge_delete":
        await _handle_nudge_delete(controller, msg)

    elif msg_type == "nudge_flush":
        await _handle_nudge_flush(controller, msg)

    elif msg_type == "nudge_favorite":
        await _handle_nudge_favorite(controller, msg)

    elif msg_type == "nudge_validate_favorites":
        await _handle_nudge_validate_favorites(controller, msg)

    # ── AI document editing ────────────────────────────────
    elif msg_type == "ai_edit_request":
        from llming_lodge.api.ai_edit_handler import AIEditHandler
        if not hasattr(controller, '_ai_edit_handler'):
            controller._ai_edit_handler = AIEditHandler(
                controller, user_name=entry.user_name, user_mail=entry.user_id)
        await controller._ai_edit_handler.handle_edit_request(msg)

    elif msg_type == "ai_edit_cancel":
        if hasattr(controller, '_ai_edit_handler'):
            await controller._ai_edit_handler.cancel_edit(msg)

    elif msg_type == "ai_task_request":
        from llming_lodge.api.ai_edit_handler import AIEditHandler
        if not hasattr(controller, '_ai_edit_handler'):
            controller._ai_edit_handler = AIEditHandler(
                controller, user_name=entry.user_name, user_mail=entry.user_id)
        await controller._ai_edit_handler.handle_task_request(msg)

    elif msg_type == "ai_task_cancel":
        if hasattr(controller, '_ai_edit_handler'):
            await controller._ai_edit_handler.cancel_task(msg)

    elif msg_type == "ai_typeahead_request":
        from llming_lodge.api.ai_edit_handler import AIEditHandler
        if not hasattr(controller, '_ai_edit_handler'):
            controller._ai_edit_handler = AIEditHandler(
                controller, user_name=entry.user_name, user_mail=entry.user_id)
        await controller._ai_edit_handler.handle_typeahead_request(msg)

    elif msg_type == "ai_typeahead_cancel":
        if hasattr(controller, '_ai_edit_handler'):
            controller._ai_edit_handler.cancel_typeahead(msg)

    # ── Document plugin messages ──────────────────────────
    elif msg_type == "doc_list_request":
        if entry.doc_manager:
            docs = entry.doc_manager.store.list_all()
            await controller._send({
                "type": "doc_list",
                "documents": [d.model_dump() for d in docs],
            })

    elif msg_type == "doc_restore":
        if entry.doc_manager and msg.get("documents"):
            entry.doc_manager.store.restore_from_list(msg["documents"])
            # Auto-enable per-type editing tools for restored document types
            await _auto_enable_restored_doc_tools(controller, msg["documents"])

    # ── Browser-hosted MCP messages ───────────────────────
    elif msg_type == "browser_mcp_result":
        request_id = msg.get("request_id")
        ctx = _browser_mcp_sessions.get(controller.session_id)
        if ctx and request_id and request_id in ctx["pending_requests"]:
            future = ctx["pending_requests"].pop(request_id)
            if not future.done():
                future.set_result(msg)
        else:
            logger.warning("[BROWSER_MCP] Unexpected result for request_id=%s", request_id)

    else:
        logger.warning(f"[WS] Unknown message type: {msg_type}")


async def _apply_doc_plugins(
    controller: WebSocketChatController,
    entry: SessionEntry,
    doc_plugins: list[str] | None,
) -> None:
    """Reconfigure doc plugins when a preset is applied.

    Updates the doc manager's enabled types, rebuilds the context preamble,
    and toggles per-type MCP tools on/off.
    """
    doc_manager = getattr(entry, "doc_manager", None)
    if not doc_manager:
        return

    from llming_lodge.doc_plugins.manager import ALL_DOC_PLUGIN_TYPES, TYPE_TOOL_PREFIXES

    old_types = set(doc_manager.enabled_types)
    doc_manager.set_enabled_types(doc_plugins)
    new_types = set(doc_manager.enabled_types)

    # Rebuild context preamble (base + doc plugins + MCP hints + auto-discover)
    base_preamble = getattr(controller, "_base_context_preamble", "")
    new_preamble = base_preamble + doc_manager.get_preamble()
    mcp_hints = getattr(controller, "_mcp_prompt_hints_block", "")
    if mcp_hints:
        new_preamble = new_preamble + "\n\n" + mcp_hints
    controller.context_preamble = new_preamble
    if controller.session:
        controller.session._context_preamble = new_preamble

    # Toggle per-type MCP tools
    for doc_type in ALL_DOC_PLUGIN_TYPES:
        prefix = TYPE_TOOL_PREFIXES.get(doc_type)
        if not prefix:
            continue
        should_enable = doc_type in new_types
        was_enabled = doc_type in old_types
        if should_enable == was_enabled:
            continue
        # Toggle all tools with this prefix
        for tool in controller.get_all_known_tools():
            if tool.get("name", "").startswith(prefix):
                controller.toggle_tool(tool["name"], should_enable)

    # Also toggle the creator MCP tools when ALL doc plugins are disabled
    if not new_types:
        for tool in controller.get_all_known_tools():
            if tool.get("name") in ("create_document", "list_documents", "get_document", "delete_document"):
                controller.toggle_tool(tool["name"], False)
    elif not old_types and new_types:
        for tool in controller.get_all_known_tools():
            if tool.get("name") in ("create_document", "list_documents", "get_document", "delete_document"):
                controller.toggle_tool(tool["name"], True)

    # Notify frontend of updated tools
    if controller._ws:
        await controller._send({
            "type": "tools_updated",
            "tools": controller.get_all_known_tools(),
        })
        await controller._send_context_info()
    logger.info("[DOC_PLUGINS] Enabled types: %s", list(new_types))


async def _auto_enable_restored_doc_tools(
    controller: WebSocketChatController,
    documents: list[dict],
) -> None:
    """Auto-enable per-type editing tools for document types present in a restored session.

    Called when the client sends ``doc_restore`` (conversation loaded from IndexedDB).
    For each unique document type, enable the corresponding collapsed MCP tool group
    (e.g. type "plotly" → group "Plotly Charts") so the LLM can edit existing documents.

    This is the server-side complement to the client-side auto-enable in
    ``_autoEnableDocTools`` / ``_autoEnableForRenderedBlocks`` (chat-app-core.js).
    Both sides are needed because doc_restore may arrive before or after MCP discovery.
    """
    from llming_lodge.doc_plugins.manager import _MCP_SERVERS, _TYPE_ALIASES

    # Collect unique doc types from restored documents
    # Documents have a "type" field (plotly, table, text_doc, presentation, html)
    doc_types = set()
    for doc in documents:
        doc_type = doc.get("type")
        if doc_type:
            doc_types.add(_TYPE_ALIASES.get(doc_type, doc_type))

    if not doc_types:
        return

    # Map doc type → MCP group label (which is the toggle_tool name for collapsed groups)
    changed = False
    for doc_type in doc_types:
        spec = _MCP_SERVERS.get(doc_type)
        if not spec:
            continue
        group_label = spec["label"]
        # toggle_tool with group label toggles all tools in the group
        server_groups = getattr(controller.session, '_mcp_server_groups', {})
        group = server_groups.get(group_label)
        if group:
            all_group_tools = group.get("tool_names", [])
            if not any(tn in controller.enabled_tools for tn in all_group_tools):
                controller.toggle_tool(group_label, True)
                changed = True

    if changed and controller._ws:
        await controller._send({
            "type": "tools_updated",
            "tools": controller.get_all_known_tools(),
        })
        await controller._send_context_info()
    if doc_types:
        logger.info("[DOC_RESTORE] Auto-enabled tools for types: %s", list(doc_types))


async def _extract_preset_file_texts(preset) -> None:
    """Extract text_content for binary preset files that don't have it yet."""
    from llming_lodge.documents import extract_text

    for f in preset.files:
        if f.text_content or not f.content:
            continue
        # Text-like files should already have text_content from client
        if f.mime_type.startswith("text/"):
            continue
        try:
            raw = base64.b64decode(f.content.split(",", 1)[-1])
            f.text_content = extract_text(raw, f.mime_type)
        except Exception as e:
            logger.warning(f"[PRESET] Text extraction failed for {f.name}: {e}")


async def _inject_preset_files(
    preset_files: list,
    entry: SessionEntry,
    controller: WebSocketChatController,
) -> None:
    """Inject preset files into the upload manager and sync document context.

    Accepts either CachedFile objects (from NudgeFileCache) or plain dicts
    (from IDB presets sent by the frontend).  Everything stays in memory —
    no files are written to disk.
    """
    if not entry.upload_manager:
        return
    from llming_lodge.documents.upload_manager import FileAttachment
    from llming_lodge.nudge_store import CachedFile

    injected = 0
    for pf in preset_files:
        # CachedFile (from in-memory file cache) — has raw bytes + text
        if isinstance(pf, CachedFile):
            # Preserve raw bytes for PDFs so PdfViewerMCP can render pages
            attachment = FileAttachment(
                name=pf.name,
                size=pf.size,
                mime_type=pf.mime_type,
                file_id=str(uuid4()),
                text_content=pf.text_content,
                raw_data=pf.raw,
            )
            entry.upload_manager.files.append(attachment)
            injected += 1
            logger.info("[INJECT] CachedFile %s: %d chars text", pf.name, len(pf.text_content))
        # Dict from IDB preset files
        elif isinstance(pf, dict):
            text_content = pf.get("text_content", "")
            raw_bytes = None
            mime = pf.get("mime_type", "text/plain")
            # Decode base64 content to raw bytes (no disk I/O)
            if pf.get("content"):
                try:
                    raw_bytes = base64.b64decode(pf["content"].split(",", 1)[-1])
                except Exception:
                    pass
            # If no text_content, extract from raw bytes in memory
            if not text_content and raw_bytes:
                try:
                    import asyncio as _aio
                    from llming_lodge.documents import extract_text as _extr
                    text_content = await _aio.to_thread(_extr, raw_bytes, mime)
                    logger.info("[INJECT] Extracted %d chars from %s", len(text_content), pf.get("name"))
                except Exception as e:
                    logger.warning("[INJECT] Text extraction failed for %s: %s", pf.get("name"), e)
            if text_content:
                attachment = FileAttachment(
                    name=pf.get("name", "file"),
                    size=len(text_content.encode()),
                    mime_type=mime,
                    file_id=str(uuid4()),
                    text_content=text_content,
                    raw_data=raw_bytes,
                )
                entry.upload_manager.files.append(attachment)
                injected += 1
            else:
                logger.warning("[INJECT] Skipped file (no text): %s, keys=%s",
                               pf.get("name"), list(pf.keys()))

    logger.info("[INJECT] Injected %d/%d preset files", injected, len(preset_files))
    controller.sync_document_context(entry.upload_manager)
    await controller._send_context_info()
    # Register PdfViewerMCP if any injected files are PDFs
    await _sync_pdf_viewer_mcp(controller, entry)


async def _handle_language_change(
    controller: WebSocketChatController,
    new_lang: str,
) -> None:
    """Handle mid-chat language switching."""
    controller._locale = new_lang

    # Get chat UI translations from llming-lodge's own i18n
    from llming_lodge.i18n import get_translations
    translations = get_translations(new_lang)

    response: dict = {
        "type": "language_changed",
        "locale": new_lang,
        "translations": translations,  # overrides merged below after host callback
        "tools": controller.get_all_known_tools(),
    }

    # Let the host app re-translate QuickActions & context preamble
    if controller._on_language_change:
        try:
            extra = await controller._on_language_change(new_lang)
            if isinstance(extra, dict):
                if "quick_actions" in extra:
                    controller._custom_quick_actions = extra["quick_actions"]
                    response["quick_actions"] = extra["quick_actions"]
                if "context_preamble" in extra:
                    new_preamble = extra["context_preamble"]
                    # Update base preamble and re-append MCP hints
                    controller._base_context_preamble = new_preamble
                    mcp_hints = getattr(controller, "_mcp_prompt_hints_block", "")
                    if mcp_hints:
                        new_preamble = new_preamble + "\n\n" + mcp_hints
                    controller.context_preamble = new_preamble
                    controller.session._context_preamble = new_preamble
                if "app_title" in extra:
                    response["app_title"] = extra["app_title"]
                if "tool_toggle_notifications" in extra:
                    controller._tool_toggle_notifications = extra["tool_toggle_notifications"]
                if "translation_overrides" in extra:
                    controller._translation_overrides = extra["translation_overrides"]
        except Exception as e:
            logger.warning(f"[LANG] on_language_change callback failed: {e}")

    # Merge translation overrides from the host app
    if controller._translation_overrides:
        response["translations"].update(controller._translation_overrides)

    await controller._send(response)


async def _handle_action_callback(
    controller: WebSocketChatController,
    action_id: str,
    text: str,
) -> None:
    """Handle a quick-action callback (server-side hook instead of chat message)."""
    if not controller._on_action_callback:
        logger.warning(f"[ACTION] No callback registered, ignoring {action_id}")
        return

    try:
        result = await controller._on_action_callback(action_id, text)
        if isinstance(result, dict):
            await controller._send({
                "type": "action_callback_result",
                "action_id": action_id,
                **result,
            })
    except Exception as e:
        logger.warning(f"[ACTION] Callback for {action_id} failed: {e}")
        await controller._send({
            "type": "action_callback_result",
            "action_id": action_id,
            "notification": f"Error: {e}",
            "notification_type": "negative",
        })


async def _load_conversation(
    controller: WebSocketChatController,
    entry: SessionEntry,
    data: dict,
) -> None:
    """Load a conversation from client-provided data (IndexedDB)."""
    try:
        conv_id = data.get("id")
        if not conv_id:
            return

        # Update session ID
        old_session_id = controller.session_id
        controller.session_id = conv_id
        controller._conversation_title = data.get("title")
        controller._title_msg_count = 0
        controller._project_id = data.get("project_id")
        controller._nudge_id = data.get("nudge_id")

        # Migrate auto-discover session context to new session ID
        if old_session_id in _discoverable_sessions:
            _discoverable_sessions[conv_id] = _discoverable_sessions.pop(old_session_id)
            controller.tool_config["consult_nudge"] = {"_session_id": conv_id}

        # Re-register in registry
        registry = SessionRegistry.get()
        registry.remove(old_session_id)
        registry._sessions[conv_id] = entry

        # Switch model if needed
        loaded_model = data.get("model")
        if loaded_model and loaded_model != controller.model:
            try:
                await controller.switch_model(loaded_model)
            except Exception as ex:
                logger.warning(f"[LOAD] Could not switch to model {loaded_model}: {ex}")

        # Restore base system prompt
        if data.get("base_system_prompt"):
            controller._base_system_prompt = data["base_system_prompt"]
            controller.update_settings(system_prompt=data["base_system_prompt"])

        # Rebuild history
        controller.session.history = ChatHistory()
        for msg_dict in data.get("messages", []):
            msg = ChatMessage.model_validate(msg_dict)
            controller.session.history.add_message(msg)

        # Restore condensed summary
        controller.session._condensed_summary = data.get("condensed_summary")

        # Do NOT restore enabled_tools from saved conversation.
        # Tools are session-scoped — each session starts with server defaults
        # (set by MCP discovery + model defaults).  Restoring old tool lists
        # would re-enable tools the user didn't explicitly toggle.

        # Re-enable consult_nudge if auto-discover is active
        ad_catalog = getattr(controller, "_auto_discover_catalog", None)
        if ad_catalog:
            if "consult_nudge" not in controller.enabled_tools:
                controller.enabled_tools.append("consult_nudge")
                controller.session.config.tools = list(controller.enabled_tools)
            controller.session._system_prompt_suffix = ad_catalog

        # Re-inject preset (nudge/project) files into document context
        # Try server-side file cache first, fallback to client-sent files
        preset_files = data.get("_preset_files") or []
        nudge_meta = None
        if data.get("nudge_id") and getattr(controller, "_nudge_store", None):
            try:
                store = controller._nudge_store
                user_teams = getattr(controller, "_user_teams", None)
                from llming_lodge.nudge_store import get_file_cache
                cached = await get_file_cache().get_files(
                    data["nudge_id"], store, controller.user_mail or "",
                    user_teams=user_teams,
                )
                if cached:
                    preset_files = cached
                nudge = await store.get_for_user(data["nudge_id"], controller.user_mail or "", user_teams=user_teams)
                if nudge:
                    nudge_meta = {
                        "uid": nudge.get("uid"),
                        "name": nudge.get("name"),
                        "icon": nudge.get("icon"),
                        "description": nudge.get("description"),
                        "creator_name": nudge.get("creator_name"),
                        "suggestions": nudge.get("suggestions"),
                        "capabilities": nudge.get("capabilities"),
                        "doc_plugins": nudge.get("doc_plugins"),
                        "mode": nudge.get("mode"),
                        "category": nudge.get("category"),
                        "sub_category": nudge.get("sub_category"),
                    }
                    # Apply doc plugin config from nudge
                    if nudge.get("doc_plugins") is not None:
                        await _apply_doc_plugins(controller, entry, nudge["doc_plugins"])
            except Exception as e:
                logger.warning(f"[NUDGE] Failed to fetch nudge for conversation load: {e}")
        # Always create an upload manager for the loaded conversation
        # (needed for file restore from client-side IDB)
        if entry.upload_manager:
            entry.upload_manager.cleanup()
        entry.upload_manager = UploadManager.create(conv_id, entry.user_id)
        if preset_files:
            await _inject_preset_files(preset_files, entry, controller)

        # Always inject master nudge files (silently stacked on top)
        master_files = getattr(controller, "_master_cached_files", [])
        if master_files:
            await _inject_preset_files(master_files, entry, controller)

        # Ensure auto-discover catalog suffix is set (may have been reset by switch_model)
        ad_catalog = getattr(controller, "_auto_discover_catalog", "")
        if ad_catalog and controller.session:
            controller.session._system_prompt_suffix = ad_catalog

        # Re-wire condensation callbacks
        controller._wire_condensation()

        await controller._send_context_info()

        # Send nudge metadata to client for header rendering
        if nudge_meta:
            await controller._send({
                "type": "nudge_meta",
                "nudge": nudge_meta,
            })

        # Notify client of the new session ID so uploads use the correct ID
        await controller._send({
            "type": "session_id_updated",
            "session_id": conv_id,
        })

        logger.info(f"[LOAD] Loaded conversation {conv_id} with {len(data.get('messages', []))} messages")

    except Exception as e:
        logger.error(f"[LOAD] Failed to load conversation: {e}", exc_info=True)
        await controller._send({
            "type": "error",
            "error_type": "LoadError",
            "message": f"Failed to load conversation: {e}",
        })


# ── Nudge handlers ──────────────────────────────────────────────


async def _handle_nudge_search(controller: WebSocketChatController, msg: dict):
    store = getattr(controller, "_nudge_store", None)
    if not store:
        return
    user_teams = getattr(controller, "_user_teams", None)
    is_admin = getattr(controller, "_is_nudge_admin", False)
    all_users = bool(msg.get("all_users")) and is_admin
    try:
        results, has_more = await store.search(
            controller.user_mail or "",
            query=msg.get("query", ""),
            category=msg.get("category", ""),
            mine=msg.get("mine", False),
            page=msg.get("page", 0),
            page_size=24,
            user_teams=user_teams,
            include_master=is_admin,
            all_users=all_users,
        )
        await controller._send({
            "type": "nudge_search_result",
            "nudges": results,
            "page": msg.get("page", 0),
            "has_more": has_more,
        })
    except Exception as e:
        logger.error(f"[NUDGE] Search failed: {e}", exc_info=True)


async def _handle_nudge_get(controller: WebSocketChatController, msg: dict):
    store = getattr(controller, "_nudge_store", None)
    if not store:
        return
    user_teams = getattr(controller, "_user_teams", None)
    try:
        uid = msg.get("uid", "")
        mode = msg.get("mode")
        if mode:
            nudge = await store.get(uid, mode, controller.user_mail or "", user_teams=user_teams)
        else:
            nudge = await store.get_for_user(uid, controller.user_mail or "", user_teams=user_teams)
        await controller._send({
            "type": "nudge_detail",
            "nudge": nudge,
        })
    except Exception as e:
        logger.error(f"[NUDGE] Get failed: {e}", exc_info=True)


async def _handle_nudge_save(controller: WebSocketChatController, msg: dict):
    store = getattr(controller, "_nudge_store", None)
    if not store:
        return
    user_teams = getattr(controller, "_user_teams", None)
    is_admin = getattr(controller, "_is_nudge_admin", False)
    try:
        data = msg.get("data", {})
        # Admin can transfer ownership by setting creator_email to another user
        if is_admin and data.get("creator_email"):
            pass  # keep the provided email
        else:
            data["creator_email"] = data.get("creator_email") or controller.user_mail or ""
        # If team_id is set, look up team name for creator_name
        if data.get("team_id") and user_teams:
            team = next((t for t in user_teams if t["team_id"] == data["team_id"]), None)
            if team:
                data["creator_name"] = team["name"]
        meta = await store.save(data, controller.user_mail or "", user_teams=user_teams, is_admin=is_admin)
        await controller._send({
            "type": "nudge_saved",
            "nudge": meta,
        })
    except PermissionError as e:
        await controller._send({
            "type": "error",
            "error_type": "PermissionError",
            "message": str(e),
        })
    except Exception as e:
        logger.error(f"[NUDGE] Save failed: {e}", exc_info=True)
        await controller._send({
            "type": "error",
            "error_type": "NudgeSaveError",
            "message": str(e),
        })


async def _handle_nudge_delete(controller: WebSocketChatController, msg: dict):
    store = getattr(controller, "_nudge_store", None)
    if not store:
        return
    user_teams = getattr(controller, "_user_teams", None)
    try:
        uid = msg.get("uid", "")
        await store.delete(uid, controller.user_mail or "", user_teams=user_teams)
        await controller._send({
            "type": "nudge_deleted",
            "uid": uid,
        })
    except PermissionError as e:
        await controller._send({
            "type": "error",
            "error_type": "PermissionError",
            "message": str(e),
        })
    except Exception as e:
        logger.error(f"[NUDGE] Delete failed: {e}", exc_info=True)


async def _handle_nudge_flush(controller: WebSocketChatController, msg: dict):
    store = getattr(controller, "_nudge_store", None)
    if not store:
        return
    user_teams = getattr(controller, "_user_teams", None)
    try:
        uid = msg.get("uid", "")
        ok = await store.flush_to_live(uid, controller.user_mail or "", user_teams=user_teams)
        await controller._send({
            "type": "nudge_flushed",
            "uid": uid,
            "ok": ok,
        })
    except PermissionError as e:
        await controller._send({
            "type": "error",
            "error_type": "PermissionError",
            "message": str(e),
        })
    except Exception as e:
        logger.error(f"[NUDGE] Flush failed: {e}", exc_info=True)


async def _handle_nudge_favorite(controller: WebSocketChatController, msg: dict):
    store = getattr(controller, "_nudge_store", None)
    if not store:
        return
    user_teams = getattr(controller, "_user_teams", None)
    try:
        uid = msg.get("uid", "")
        favorite = msg.get("favorite", True)
        await store.set_favorite(controller.user_mail or "", uid, favorite)
        # Return updated favorites list
        favs = await store.get_favorites(controller.user_mail or "", user_teams=user_teams)
        await controller._send({
            "type": "nudge_favorites_result",
            "nudges": favs,
        })
    except Exception as e:
        logger.error(f"[NUDGE] Favorite toggle failed: {e}", exc_info=True)


async def _handle_nudge_validate_favorites(controller: WebSocketChatController, msg: dict):
    """Validate that the user can still see their favorited nudge UIDs."""
    store = getattr(controller, "_nudge_store", None)
    if not store:
        return
    uids = msg.get("uids", [])
    if not uids:
        return
    user_teams = getattr(controller, "_user_teams", None)
    try:
        valid_uids = await store.validate_visible(uids, controller.user_mail or "", user_teams)
        await controller._send({
            "type": "nudge_favorites_validated",
            "valid_uids": valid_uids,
        })
    except Exception as e:
        logger.error(f"[NUDGE] Favorites validation failed: {e}", exc_info=True)
