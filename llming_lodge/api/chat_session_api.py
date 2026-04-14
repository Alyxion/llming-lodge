"""WebSocket-based chat API for static frontends.

SessionRegistry — maps session_id → active sessions
WebSocketChatController — ChatController subclass that sends JSON over WS
build_ws_router() — FastAPI APIRouter with /ws/{session_id} endpoint
"""

import asyncio
import base64
import hashlib
import io
import json
import logging
import re
import secrets
import time
from dataclasses import dataclass, field

from llming_com import BaseSessionEntry, BaseSessionRegistry
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo
from uuid import uuid4

from starlette.websockets import WebSocket, WebSocketDisconnect, WebSocketState

from llming_models import ChatSession, ChatHistory, LLMManager
from llming_models.budget import BudgetHandler
from llming_models.budget.budget_limit import BudgetLimit
from llming_lodge.system_nudges import is_system_nudge_uid, get_system_nudge, search_system_nudges, _meta as _sys_nudge_meta
from llming_models.llm_base_models import Role, ChatMessage
from llming_models.tools.tool_call import ToolCallInfo, ToolCallStatus
from llming_lodge.chat_controller import ChatController, llm_manager
from llming_models.tools.tool_definition import MCPServerConfig, ToolUIMetadata
from llming_models.tools.tool_registry import get_default_registry
from pathlib import Path

from llming_lodge.documents import UploadManager
from llming_models.utils.image_utils import sniff_image_mime
from llming_lodge.utils import LlmMarkdownPostProcessor
from llming_lodge.i18n import t_chat

logger = logging.getLogger(__name__)


# ── API Keys collection helper ──────────────────────────────────────

_api_keys_coll_cache = None
_api_keys_indexes_created = False


def _get_api_keys_coll(controller):
    """Get the api_keys MongoDB collection using the controller's nudge_store."""
    global _api_keys_coll_cache, _api_keys_indexes_created
    if _api_keys_coll_cache is not None:
        return _api_keys_coll_cache
    ns = getattr(controller, "_nudge_store", None)
    if not ns:
        return None
    from nice_droplets.utils.mongo_helpers import get_async_mongo_client
    client = get_async_mongo_client(ns._mongo_uri)
    db = client[ns._mongo_db]
    _api_keys_coll_cache = db["api_keys"]
    return _api_keys_coll_cache


async def _ensure_api_keys_indexes(coll):
    """Create indexes on the api_keys collection (idempotent)."""
    global _api_keys_indexes_created
    if _api_keys_indexes_created:
        return
    try:
        await coll.create_index("key_id", unique=True)
        await coll.create_index("user_email")
        await coll.create_index("key_hash")
        _api_keys_indexes_created = True
    except Exception as e:
        logger.warning("[API_KEYS] Index creation failed: %s", e)


# ── Hybrid Intercept System ───────────────────────────────────────
#
# The hybrid intercept pattern allows plugins to render visual content
# (cards, charts, etc.) BEFORE the LLM responds, then let the LLM add
# a natural follow-up comment with full context.
#
# How it works:
#   1. An ``on_message_intercept`` handler returns a string containing:
#      - A fenced code block (e.g. ```kantini_result\n{...}\n```)
#      - Followed by a text summary after the closing ```
#   2. ``setup_hybrid_intercept()`` splits these two parts.
#   3. The fenced block is stored as ``controller._intercept_prefix``
#      and sent to the client as the first ``text_chunk`` before the
#      LLM streams (renders visual cards immediately).
#   4. The text summary is injected into the context preamble so the
#      LLM has knowledge of the data for its follow-up comment.
#   5. After the LLM response completes, the preamble is restored.
#   6. The prefix is prepended to the assistant message in history
#      so it persists in IndexedDB and renders on conversation restore.
#
# To create a new hybrid intercept plugin:
#   - Write an ``async intercept_message(text, controller)`` handler
#   - Return ``"{fenced_block}\n\n{text_summary}"`` for hybrid mode
#   - Return a plain string (no fenced block) for pure intercept
#   - Return ``None`` to skip (no interception)
#   - Optionally set ``controller._intercept_preamble`` to a custom
#     LLM instruction string before returning
#   - Register via ``ChatUserConfig(on_message_intercept=handler)``
# ──────────────────────────────────────────────────────────────────

_DEFAULT_INTERCEPT_PREAMBLE = (
    "[CONTEXT — visual content cards are ALREADY displayed above your response. "
    "The user can see all details in the cards. "
    "Your ONLY job: write 1-2 SHORT friendly sentences (max 30 words). "
    "NEVER repeat content from the cards. "
    "NEVER output JSON, code blocks, or markdown tables.]"
)


def setup_hybrid_intercept(controller, intercept_result: str,
                           preamble: str | None = None) -> bool:
    """Detect and configure hybrid intercept mode on the controller.

    Checks if ``intercept_result`` contains a fenced code block followed
    by a text summary.  If so, configures the controller for hybrid mode:
    the fenced block renders as visual content immediately, and the LLM
    adds a short follow-up with the summary as context.

    The preamble (instruction to the LLM) is resolved in priority order:
      1. Explicit ``preamble`` argument
      2. ``controller._intercept_preamble`` (set by the intercept handler)
      3. ``_DEFAULT_INTERCEPT_PREAMBLE`` fallback

    Args:
        controller: The WebSocketChatController instance.
        intercept_result: Full result from the intercept handler.
            Must contain ``\\n```\\n`` to trigger hybrid mode.
        preamble: Optional custom LLM instruction. If provided, takes
            precedence over ``controller._intercept_preamble``.

    Returns:
        ``True`` if hybrid mode was set up (caller should fall through
        to the normal LLM send path).
        ``False`` if no fenced block was found (caller should handle
        as a pure intercept — no LLM follow-up).
    """
    _fence_end = intercept_result.find("\n```\n")
    _has_summary = _fence_end > 0 and len(intercept_result) > _fence_end + 5
    if _has_summary:
        fenced_block = intercept_result[:_fence_end + 4]
        summary = intercept_result[_fence_end + 4:].strip()
        controller._intercept_prefix = fenced_block + "\n\n"
        # Use preamble from: explicit arg > controller attr (set by intercept handler) > default
        _preamble = preamble or getattr(controller, '_intercept_preamble', None) or _DEFAULT_INTERCEPT_PREAMBLE
        if hasattr(controller, '_intercept_preamble'):
            del controller._intercept_preamble
        _old_preamble = controller.context_preamble or ""
        controller.context_preamble = (
            _old_preamble + f"\n\n{_preamble}\n{summary}"
        )
        controller.session._context_preamble = controller.context_preamble
        controller._restore_preamble = _old_preamble
        return True
    return False


# ── Remote tasks collection helper ──────────────────────────────────

_remote_tasks_coll_cache = None
_remote_tasks_indexes_created = False


def _get_remote_tasks_coll(controller):
    """Get the remote_tasks MongoDB collection."""
    global _remote_tasks_coll_cache, _remote_tasks_indexes_created
    if _remote_tasks_coll_cache is not None:
        return _remote_tasks_coll_cache
    ns = getattr(controller, "_nudge_store", None)
    if not ns:
        return None
    from nice_droplets.utils.mongo_helpers import get_async_mongo_client
    client = get_async_mongo_client(ns._mongo_uri)
    db = client[ns._mongo_db]
    _remote_tasks_coll_cache = db["remote_tasks"]
    return _remote_tasks_coll_cache


async def _ensure_remote_tasks_indexes(coll):
    """Create indexes on remote_tasks collection (idempotent)."""
    global _remote_tasks_indexes_created
    if _remote_tasks_indexes_created:
        return
    try:
        await coll.create_index("task_id", unique=True)
        await coll.create_index([("user_email", 1), ("status", 1)])
        await coll.create_index("created_at", expireAfterSeconds=600)
        _remote_tasks_indexes_created = True
    except Exception as e:
        logger.warning("[REMOTE_TASKS] Index creation failed: %s", e)


# ── MCP Droplet Trust collection helper ──────────────────────────────

_mcp_trust_coll_cache = None
_mcp_trust_indexes_created = False


def _get_trust_coll(controller):
    """Get the mcp_droplet_trust MongoDB collection."""
    global _mcp_trust_coll_cache
    if _mcp_trust_coll_cache is not None:
        return _mcp_trust_coll_cache
    ns = getattr(controller, "_nudge_store", None)
    if not ns:
        return None
    from nice_droplets.utils.mongo_helpers import get_async_mongo_client
    client = get_async_mongo_client(ns._mongo_uri)
    db = client[ns._mongo_db]
    _mcp_trust_coll_cache = db["mcp_droplet_trust"]
    return _mcp_trust_coll_cache


async def _ensure_trust_indexes(coll):
    """Create indexes on mcp_droplet_trust (idempotent)."""
    global _mcp_trust_indexes_created
    if _mcp_trust_indexes_created:
        return
    try:
        await coll.create_index(
            [("user_email", 1), ("nudge_uid", 1)], unique=True
        )
        _mcp_trust_indexes_created = True
    except Exception as e:
        logger.warning("[MCP_TRUST] Index creation failed: %s", e)


async def _is_droplet_trusted(controller, nudge_uid: str) -> bool:
    """Check if a droplet is trusted by the current user."""
    coll = _get_trust_coll(controller)
    if coll is None:
        return False
    await _ensure_trust_indexes(coll)
    doc = await coll.find_one({
        "user_email": (controller.user_mail or "").lower(),
        "nudge_uid": nudge_uid,
    })
    return bool(doc and doc.get("trusted"))


async def _get_trusted_nudge_uids(controller) -> list[str]:
    """Return all trusted nudge UIDs for the current user."""
    coll = _get_trust_coll(controller)
    if coll is None:
        return []
    await _ensure_trust_indexes(coll)
    docs = await coll.find(
        {"user_email": (controller.user_mail or "").lower(), "trusted": True}
    ).to_list(500)
    return [d["nudge_uid"] for d in docs if d.get("nudge_uid")]


# ── Rich MCP Render Storage ─────────────────────────────────

_rich_mcp_coll_cache = None


def _get_rich_mcp_coll(controller):
    """Get the rich_mcp_renders MongoDB collection."""
    global _rich_mcp_coll_cache
    if _rich_mcp_coll_cache is not None:
        return _rich_mcp_coll_cache
    ns = getattr(controller, "_nudge_store", None)
    if not ns:
        return None
    from nice_droplets.utils.mongo_helpers import get_async_mongo_client
    client = get_async_mongo_client(ns._mongo_uri)
    db = client[ns._mongo_db]
    _rich_mcp_coll_cache = db["rich_mcp_renders"]
    return _rich_mcp_coll_cache


async def _store_rich_mcp_render(controller, render_spec: dict, tool_name: str) -> str | None:
    """Store a rich MCP render spec in MongoDB. Returns UUID string or None on failure."""
    coll = _get_rich_mcp_coll(controller)
    if coll is None:
        return None
    uid = str(uuid4())
    try:
        await coll.insert_one({
            "_id": uid,
            "session_id": controller.session_id,
            "tool_name": tool_name,
            "render": render_spec,
            "version": render_spec.get("version", "1.0") if isinstance(render_spec, dict) else "1.0",
            "created_at": datetime.now(timezone.utc),
        })
        return uid
    except Exception as e:
        logger.warning("[RICH_MCP] Failed to store render spec: %s", e)
        return None


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
    from llming_models.tools.mcp import create_connection
    from llming_models.tools.tool_registry import get_default_registry

    registry = get_default_registry()
    connection = create_connection(mcp_config)
    await connection.start()
    tools = await connection.list_tools()

    label = mcp_config.label or "MCP"
    category = mcp_config.category or "General"
    group_tool_names = []

    for tool in tools:
        if category:
            from llming_models.tools.tool_definition import ToolUIMetadata
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
        "avatar": getattr(mcp_config, 'avatar', None),
        "auto_activate_keywords": getattr(mcp_config, 'auto_activate_keywords', None),
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
    from llming_models.tools.tool_registry import get_default_registry

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
    from llming_models.tools.mcp.browser_connection import MCPBrowserConnection
    from llming_models.tools.tool_definition import ToolDefinition, ToolSource, ToolUIMetadata

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

    # Sort files so dependencies come before dependents.
    # Files with no local imports are loaded first.
    import re as _re
    _local_import_re = _re.compile(r"""import\s+\{[^}]+\}\s+from\s+['"]\./?([^'"]+)['"]""")

    def _dep_sort_key(name_source):
        name, source = name_source
        imports = _local_import_re.findall(source)
        # Count how many local modules this file imports
        return len(imports)

    sorted_items = sorted(files_data.items(), key=_dep_sort_key)
    files_data = dict(sorted_items)

    entry_point = nudge.get("mcp_entry_point", "index.js")

    # Send start message to browser and await tool list
    request_id = str(__import__("uuid").uuid4())
    loop = ctx["loop"]
    future = loop.create_future()
    ctx["pending_requests"][request_id] = {"future": future, "nudge_uid": uid}

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


async def _activate_server_mcp(controller: "WebSocketChatController", nudge: dict) -> str:
    """Activate a server-side (in-process) MCP server for a system droplet.

    The nudge's capabilities.server_mcp specifies the Python class path, e.g.
    ``llming_lodge.tools.math_mcp.MathMCP``.  The class must subclass
    ``InProcessMCPServer``.
    """
    import importlib
    from llming_models.tools.mcp.config import MCPServerConfig

    caps = nudge.get("capabilities") or {}
    class_path = caps.get("server_mcp", "")
    if not class_path:
        return "No server_mcp capability configured."

    # Import and instantiate the server class
    module_path, _, class_name = class_path.rpartition(".")
    try:
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
    except Exception:
        # Fallback: tools moved from llming_lodge → llming_models
        fallback = module_path.replace("llming_lodge.", "llming_models.", 1)
        try:
            mod = importlib.import_module(fallback)
            cls = getattr(mod, class_name)
            logger.info("[SERVER_MCP] Resolved %s via fallback %s", class_path, fallback)
        except Exception as exc:
            logger.error("[SERVER_MCP] Failed to import %s (also tried %s): %s", class_path, fallback, exc)
            return f"Server MCP import failed: {exc}"

    # Try to pass document_store if the server supports it
    doc_store = None
    entry = SessionRegistry.get().get_session(controller.session_id)
    if entry and entry.doc_manager:
        doc_store = entry.doc_manager.store

    try:
        if doc_store:
            try:
                server_instance = cls(document_store=doc_store)
            except TypeError:
                server_instance = cls()
        else:
            server_instance = cls()
    except Exception as exc:
        logger.error("[SERVER_MCP] Failed to instantiate %s: %s", class_path, exc)
        return f"Server MCP instantiation failed: {exc}"

    nudge_name = nudge.get("name", nudge.get("uid", ""))
    mcp_config = MCPServerConfig(
        label=nudge_name,
        description=nudge.get("description", ""),
        server_instance=server_instance,
        enabled_by_default=True,
        collapse_tools=True,
        avatar=nudge.get("avatar"),
    )

    await _register_mcp_tools(controller, mcp_config)

    # Model selection for droplets — switch to the default model and optionally
    # restrict the model selector to a set of allowed models.
    enforced_model = caps.get("enforced_model", "claude_opus")
    allowed_models = caps.get("allowed_models")  # None = locked to enforced_model
    try:
        controller._saved_model_pref = "@auto" if controller._auto_select else controller.model
        controller._saved_auto_select = controller._auto_select
        controller._auto_select = False
        await controller._silent_switch_model(enforced_model)
        # Use the resolved model name (deployment name), not the bare name
        resolved_model = controller.model

        if allowed_models:
            # Resolve bare names to actual model/deployment names
            resolved_allowed = []
            for m in allowed_models:
                try:
                    info = llm_manager.get_model_info(m)
                    resolved_allowed.append(info.model)
                except ValueError:
                    resolved_allowed.append(m)
            controller._model_locked = False
            controller._model_locked_reason = ""
            await controller._send({
                "type": "model_restricted",
                "model": resolved_model,
                "allowed_models": resolved_allowed,
                "reason": nudge.get("name", "System Droplet"),
            })
            logger.info("[SERVER_MCP] Model %s (allowed: %s) for droplet '%s'",
                        resolved_model, resolved_allowed, nudge.get("name", ""))
        else:
            controller._model_locked = True
            controller._model_locked_reason = nudge.get("name", "System Droplet")
            await controller._send({
                "type": "model_locked",
                "model": resolved_model,
                "reason": controller._model_locked_reason,
            })
            logger.info("[SERVER_MCP] Enforced model %s for droplet '%s'",
                        resolved_model, nudge.get("name", ""))
    except Exception as ex:
        logger.warning("[SERVER_MCP] Could not enforce model %s: %s", enforced_model, ex)

    # Also inject a tool hint into the context preamble
    tools = await server_instance.list_tools()
    tool_lines = []
    for t in tools:
        desc = t.get("description", "")
        tool_lines.append(f"- **{t['name']}**: {desc}" if desc else f"- **{t['name']}**")
    hint_block = (
        f"\n\n## MCP Tools — {nudge_name}\n"
        "CRITICAL: You have the following specialised tools available.  "
        "You MUST call the appropriate tool(s) BEFORE answering.  "
        "Never guess or answer from your own knowledge when a tool can provide the answer.\n\n"
        + "\n".join(tool_lines)
    )
    base_preamble = controller.context_preamble or ""
    controller.context_preamble = base_preamble + hint_block
    if controller.session:
        controller.session._context_preamble = controller.context_preamble

    if controller._ws:
        await controller._send({
            "type": "tools_updated",
            "tools": controller.get_all_known_tools(),
        })

    # Register client-side renderers (e.g. flux source footer for PDF/table previews)
    try:
        client_renderers = await server_instance.get_client_renderers()
        if client_renderers and controller._ws:
            await controller._send({
                "type": "register_renderers",
                "renderers": client_renderers,
            })
            logger.info("[SERVER_MCP] Registered %d client renderer(s) for '%s'",
                        len(client_renderers), nudge_name)
    except Exception as ex:
        logger.warning("[SERVER_MCP] Failed to get client renderers for '%s': %s", nudge_name, ex)

    logger.info("[SERVER_MCP] Activated '%s' with %d tools", nudge_name, len(tools))
    return f"Activated server MCP '{nudge_name}' with {len(tools)} tools"


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
class SessionEntry(BaseSessionEntry):
    """A registered chat session with its controller and metadata.

    Extends llming-com's BaseSessionEntry with chat-specific fields.
    """
    app_type: str = "lodge"
    upload_manager: Optional[UploadManager] = None
    mcp_servers: Optional[List[MCPServerConfig]] = None
    doc_manager: Optional[Any] = None


class SessionRegistry(BaseSessionRegistry["SessionEntry"]):
    """Chat session registry — extends llming-com's BaseSessionRegistry.

    Adds a global presentation template registry that survives session
    cleanup (so PPTX export works from restored chats).
    """

    # Global template registry — class-level, session-independent
    _presentation_templates: Dict[str, Any] = {}

    def register_session(
        self,
        session_id: str,
        controller: "WebSocketChatController",
        user_id: str,
        user_name: str = "",
        user_avatar: str = "",
        mcp_servers: Optional[List[MCPServerConfig]] = None,
    ) -> SessionEntry:
        """Register a chat session (convenience method with UploadManager creation)."""
        entry = SessionEntry(
            controller=controller,
            user_id=user_id,
            user_name=user_name,
            user_avatar=user_avatar,
            upload_manager=UploadManager.create(session_id, user_id),
            mcp_servers=mcp_servers,
        )
        return self.register(session_id, entry)

    def on_session_expired(self, session_id: str, entry: SessionEntry) -> None:
        """Clean up upload manager when a session expires."""
        if entry.upload_manager:
            entry.upload_manager.cleanup()

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
        self._translation_overrides: dict[str, Any] = {}
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
        self._client_timezone: str = "UTC"
        self._pending_rich_mcp_blocks: list[dict] = []  # inline data for rich_mcp fenced blocks
        self._pending_doc_blocks: list[tuple[str, dict]] = []  # (type, data) for doc-store documents
        # Maps message object id() -> list of (type, data) tuples for IndexedDB persistence
        self._message_doc_blocks: dict[int, list[tuple[str, dict]]] = {}
        # Rich MCP blocks per message — for serialization to IndexedDB only (NOT for LLM history).
        # Maps message object id() -> list of block dicts.  Kept separate from history so the LLM
        # never sees plotly/render JSON on future turns.
        self._message_rich_mcp: dict[int, list[dict]] = {}
        # Maps message object id() -> list of tool call dicts (for IndexedDB persistence)
        self._message_tool_calls: dict[int, list[dict]] = {}
        # Maps message object id() -> avatar override dict (for MCP custom avatars)
        self._message_avatar_overrides: dict[int, dict] = {}
        # Maps message object id() -> display text (interim narration stripped)
        self._message_display_text: dict[int, str] = {}

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
                payload = json.dumps(msg, ensure_ascii=False)
                payload_len = len(payload)
                msg_type = msg.get("type", "?")
                if payload_len > 50_000:
                    logger.warning("[WS_SIZE] type=%s size=%d bytes (%.1f KB)", msg_type, payload_len, payload_len / 1024)
                else:
                    logger.debug("[WS_SIZE] type=%s size=%d bytes", msg_type, payload_len)
                await self._ws.send_text(payload)
            except Exception as e:
                logger.warning("[WS] Send failed (type=%s): %s", msg.get("type", "?"), e)

    # ── Abstract hook implementations ─────────────────────────

    def _get_mcp_avatar_for_tools(self, tool_names: set[str]) -> tuple[str, str] | None:
        """Find the MCP avatar for the given tool names. Returns (icon, label) or None."""
        if not hasattr(self.session, '_mcp_server_groups'):
            return None
        for group in self.session._mcp_server_groups.values():
            avatar = group.get("avatar")
            if not avatar:
                continue
            if tool_names.intersection(group.get("tool_names", [])):
                return avatar, group.get("label", "")
        return None

    def _on_response_started(self) -> None:
        model_info = llm_manager.get_model_info(self.model)
        msg = {
            "type": "response_started",
            "model": self.model,
            "model_icon": model_info.model_icon if model_info else "",
            "model_label": model_info.label if model_info else self.model,
        }
        if self._auto_select:
            msg["auto_icon"] = "models/auto-select.svg"
            msg["model_label"] = f"Auto ({msg['model_label']})"
        asyncio.create_task(self._send(msg))
        # ── Hybrid intercept: inject visual card before LLM streams ──
        # If setup_hybrid_intercept() set _intercept_prefix, send it as
        # the first text_chunk so the frontend renders the card immediately.
        # The prefix is also prepended to _text_content so it's included
        # in full_text when the LLM finishes (for client-side persistence).
        # _used_intercept_prefix is saved for _on_response_completed to
        # prepend the prefix to the server-side history message too.
        prefix = getattr(self, '_intercept_prefix', None)
        if prefix:
            self._intercept_prefix = None
            self._used_intercept_prefix = prefix
            self._text_content = prefix
            asyncio.create_task(self._send({"type": "text_chunk", "content": prefix}))

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
        # Insert line break when text resumes after a tool call so segments
        # are visually separated during streaming.
        if getattr(self, '_had_tool_since_text', False):
            self._had_tool_since_text = False
            # Archive previous segment as interim
            if not hasattr(self, '_interim_text_segments'):
                self._interim_text_segments = []
            prev = getattr(self, '_current_text_segment', '')
            if prev.strip():
                self._interim_text_segments.append(prev.strip())
            self._current_text_segment = ''
            content = "\n" + content
        if not hasattr(self, '_current_text_segment'):
            self._current_text_segment = ''
        self._current_text_segment += content
        asyncio.create_task(self._send({
            "type": "text_chunk",
            "content": content,
        }))
        # Push chunk to remote task document if active
        remote_task_id = getattr(self, "_remote_task_id", None)
        if remote_task_id:
            coll = _get_remote_tasks_coll(self)
            if coll is not None:
                asyncio.create_task(
                    coll.update_one(
                        {"task_id": remote_task_id},
                        {"$push": {"chunks": content}},
                    )
                )
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
        self._had_tool_since_text = True
        self._tool_text_segment_idx = getattr(self, '_tool_text_segment_idx', 0) + 1
        result = tool_call.result if tool_call.status == ToolCallStatus.COMPLETED else None

        # Detect rich MCP envelope in in-process tool results
        if result is not None:
            result_size = len(str(result))
            logger.info("[TOOL_RESULT] tool=%s status=%s result_size=%d bytes (%.1f KB)",
                        tool_call.name, tool_call.status, result_size, result_size / 1024)
            try:
                parsed = result if isinstance(result, dict) else json.loads(result)
                if isinstance(parsed, dict) and "__rich_mcp__" in parsed:
                    rich_mcp = parsed["__rich_mcp__"]
                    # Build LLM summary from structured data
                    llm_summary = rich_mcp.get("llm_summary")
                    if not llm_summary:
                        parts = []
                        if rich_mcp.get("formatted"):
                            parts.append(f"Product: {rich_mcp['formatted']}")
                        if rich_mcp.get("ordering"):
                            parts.append(f"Ordering: {rich_mcp['ordering']}")
                        for k, v in (rich_mcp.get("info") or {}).items():
                            if v and v != "N/A":
                                parts.append(f"{k}: {v}")
                        llm_summary = "\n".join(parts) if parts else f"[{rich_mcp.get('title', 'Visualization')}]"
                    # Replace tool result with summary so LLM doesn't see the full envelope
                    result = llm_summary
                    tool_call = ToolCallInfo(
                        name=tool_call.name,
                        call_id=tool_call.call_id,
                        status=tool_call.status,
                        arguments=tool_call.arguments,
                        result=llm_summary,
                        error=tool_call.error,
                        display_name=tool_call.display_name,
                        is_image_generation=tool_call.is_image_generation,
                    )
                    # Queue inline data for fenced code block injection (no MongoDB)
                    block_data = dict(rich_mcp)
                    block_data.pop("llm_summary", None)
                    # Keep render data for types that need client-side rendering
                    render_data = block_data.get("render")
                    if render_data and render_data.get("type") not in ("html_sandbox", "math_result"):
                        block_data.pop("render", None)
                    block_data["id"] = str(uuid4())
                    self._pending_rich_mcp_blocks.append(block_data)
            except (json.JSONDecodeError, TypeError, AttributeError):
                pass

        # Detect __inline_doc__ in tool results (e.g. math_plot_2d).
        # The tool already created the doc in the store and included
        # document_id in its result (visible to the LLM).  We just need to
        # queue a fenced code block for inline rendering + persistence,
        # and strip the bulky data from the WS event.
        #
        # IMPORTANT: If the tool also called doc_store.create(), a "doc_created"
        # WS event was already sent which renders the block via _injectToolDocBlock.
        # In that case we must NOT queue a fenced code block (it would duplicate).
        # We detect this by checking for "document_id" in the result — if present,
        # the doc store already created and notified the client.
        if result is not None:
            try:
                parsed_res = result if isinstance(result, dict) else json.loads(result)
                _SAFE_DOC_TYPES = {"plotly", "table", "text_doc", "presentation", "html", "html_sandbox", "latex", "email_draft"}
                if isinstance(parsed_res, dict) and "__inline_doc__" in parsed_res:
                    inline = parsed_res.pop("__inline_doc__")
                    doc_type = inline.get("type", "plotly")
                    if doc_type not in _SAFE_DOC_TYPES:
                        logger.warning("[TOOL_RESULT] Rejected unknown doc type: %s", doc_type)
                        raise KeyError("unknown doc type")  # caught by except below, skips injection
                    doc_name = inline.get("name", "Document")
                    doc_data = inline.get("data", {})
                    doc_id = inline.get("id") or parsed_res.get("document_id") or uuid4().hex[:12]

                    # Skip fenced block injection if doc_store already notified the
                    # client via doc_created (prevents duplicate rendering).
                    already_notified = bool(parsed_res.get("document_id"))

                    block = dict(doc_data)
                    block["id"] = doc_id
                    block["name"] = doc_name
                    if not already_notified:
                        # Queue for both live injection (full_text) and persistence
                        self._pending_doc_blocks.append((doc_type, block))
                        logger.info("[TOOL_RESULT] Queued inline doc type=%s id=%s for injection",
                                    doc_type, doc_id)
                    else:
                        # Already rendered live — queue ONLY for serialization
                        # (not in _pending_doc_blocks which appends to full_text)
                        if not hasattr(self, '_persist_only_doc_blocks'):
                            self._persist_only_doc_blocks = []
                        self._persist_only_doc_blocks.append((doc_type, block))
                        logger.info("[TOOL_RESULT] Queued doc type=%s id=%s for persistence only",
                                    doc_type, doc_id)

                    # Strip bulky __inline_doc__ from WS event but keep
                    # the original tool result for conversation history
                    result = json.dumps(parsed_res)
                    tool_call = ToolCallInfo(
                        name=tool_call.name,
                        call_id=tool_call.call_id,
                        status=tool_call.status,
                        arguments=tool_call.arguments,
                        result=result,
                        error=tool_call.error,
                        display_name=tool_call.display_name,
                        is_image_generation=tool_call.is_image_generation,
                    )
            except (json.JSONDecodeError, TypeError, AttributeError, KeyError):
                pass

        # Auto-enable per-type editing tools when a document is created.
        # This covers both create_document tool and __inline_doc__ tool results.
        if result is not None and tool_call.status == ToolCallStatus.COMPLETED:
            try:
                _parsed = result if isinstance(result, dict) else json.loads(result)
                _doc_type = None
                if isinstance(_parsed, dict):
                    # create_document returns {"status": "created", "type": "text_doc", ...}
                    if _parsed.get("status") == "created" and _parsed.get("type"):
                        _doc_type = _parsed["type"]
                    # __inline_doc__ results have document_id + we know the type
                    elif _parsed.get("document_id") and _parsed.get("status") == "chart_created":
                        _doc_type = "plotly"
                if _doc_type:
                    from llming_docs.manager import _MCP_SERVERS, _TYPE_ALIASES
                    _doc_type = _TYPE_ALIASES.get(_doc_type, _doc_type)
                    spec = _MCP_SERVERS.get(_doc_type)
                    if spec:
                        group_label = spec["label"]
                        server_groups = getattr(self.session, '_mcp_server_groups', {})
                        group = server_groups.get(group_label)
                        if group:
                            all_group_tools = group.get("tool_names", [])
                            if not any(tn in self.enabled_tools for tn in all_group_tools):
                                self.toggle_tool(group_label, True)
                                asyncio.create_task(self._send({
                                    "type": "tools_updated",
                                    "tools": self.get_all_known_tools(),
                                }))
                                logger.info("[AUTO_ENABLE] Enabled %s tools for doc type %s",
                                            group_label, _doc_type)
            except (json.JSONDecodeError, TypeError, AttributeError, KeyError):
                pass

        # Resolve display name from tool registry (honors MCP displayName)
        # rather than the ToolCallInfo fallback (snake_case → Title Case).
        display_name = tool_call.display_name
        tool_def = get_default_registry().get(tool_call.name)
        if tool_def:
            display_name = tool_def.get_display_name()

        tool_event = {
            "type": "tool_event",
            "name": tool_call.name,
            "call_id": tool_call.call_id,
            "display_name": display_name,
            "status": tool_call.status.value if hasattr(tool_call.status, "value") else str(tool_call.status),
            "is_image_generation": tool_call.is_image_generation,
            "result": result,
        }
        if tool_call.arguments:
            tool_event["arguments"] = tool_call.arguments
        asyncio.create_task(self._send(tool_event))

    def _on_image_received(self, image_data: str) -> None:
        # Store for response_completed payload
        self._generated_image_base64 = image_data
        asyncio.create_task(self._send({
            "type": "image_received",
            "data": sniff_image_mime(image_data),
        }))

    def _on_response_completed(self, full_text: str) -> None:
        # ── Strip interim narration from visible text ────────────
        # The server history keeps the full text (useful for follow-up context).
        # The frontend gets a clean version with only the final answer.
        interim_segments = getattr(self, '_interim_text_segments', [])
        display_text = full_text
        if interim_segments:
            for seg in interim_segments:
                display_text = display_text.replace(seg, '', 1)
            display_text = display_text.lstrip()
            self._interim_text_segments = []
            # Track display_text for serialization — the history keeps full text
            # (with interim) for LLM context, but IDB stores display_text
            try:
                if self.session.history.messages:
                    last_msg = self.session.history.messages[-1]
                    if last_msg.role == Role.ASSISTANT:
                        self._message_display_text[id(last_msg)] = display_text
            except Exception:
                pass
        self._current_text_segment = ''
        self._tool_text_segment_idx = 0
        self._had_tool_since_text = False
        # Use display_text for the frontend
        full_text = display_text

        # ── Hybrid intercept cleanup ─────────────────────────────
        # 1. Restore the context preamble to its pre-intercept state
        #    (the summary was only needed for this one LLM turn).
        _restore = getattr(self, '_restore_preamble', None)
        if _restore is not None:
            self.context_preamble = _restore
            self.session._context_preamble = _restore
            self._restore_preamble = None
        # 2. Prepend the visual card (fenced block) to the assistant
        #    message in server-side history.  The LLM only generated
        #    the follow-up text, but the stored message must contain
        #    BOTH so that _serialize_messages() → save_conversation →
        #    IndexedDB has the full content for conversation restore.
        _prefix = getattr(self, '_used_intercept_prefix', None)
        if _prefix:
            self._used_intercept_prefix = None
            try:
                if self.session.history.messages:
                    last_msg = self.session.history.messages[-1]
                    if last_msg.role == Role.ASSISTANT:
                        last_msg.content = _prefix + (last_msg.content or "")
            except Exception as e:
                logger.warning("[INTERCEPT] Failed to prepend prefix to history: %s", e)
        logger.info("[RESP_DONE] full_text=%d chars, pending_rich_mcp=%d blocks",
                     len(full_text), len(self._pending_rich_mcp_blocks))
        # Append rich MCP fenced code blocks to the WS payload for client rendering,
        # but do NOT inject them into the server-side history — the LLM should never
        # see plotly/render JSON on future conversation turns.
        if self._pending_rich_mcp_blocks:
            blocks = list(self._pending_rich_mcp_blocks)
            for i, entry in enumerate(blocks):
                block_json = json.dumps(entry, ensure_ascii=False)
                logger.info("[RESP_DONE] rich_mcp block %d: %d bytes (%.1f KB)",
                            i, len(block_json), len(block_json) / 1024)
                full_text += f"\n\n```rich_mcp\n{block_json}\n```"
            self._pending_rich_mcp_blocks.clear()
            logger.info("[RESP_DONE] full_text after rich_mcp injection: %d chars (%.1f KB)",
                        len(full_text), len(full_text) / 1024)
            # Track blocks for conversation serialization (IndexedDB persistence)
            try:
                if self.session.history.messages:
                    last_msg = self.session.history.messages[-1]
                    if last_msg.role == Role.ASSISTANT:
                        self._message_rich_mcp[id(last_msg)] = blocks
            except Exception as e:
                logger.warning("[RICH_MCP] Failed to track blocks for serialization: %s", e)

        # Append document-store blocks (e.g. plotly charts from math MCP)
        # as fenced code blocks so they survive conversation reload.
        if self._pending_doc_blocks:
            doc_blocks = list(self._pending_doc_blocks)
            for doc_type, block_data in doc_blocks:
                block_json = json.dumps(block_data, ensure_ascii=False)
                logger.info("[RESP_DONE] doc block type=%s: %d bytes (%.1f KB)",
                            doc_type, len(block_json), len(block_json) / 1024)
                full_text += f"\n\n```{doc_type}\n{block_json}\n```"
            self._pending_doc_blocks.clear()
            logger.info("[RESP_DONE] full_text after doc injection: %d chars (%.1f KB)",
                        len(full_text), len(full_text) / 1024)
            # Track for IndexedDB serialization (like rich_mcp blocks)
            try:
                if self.session.history.messages:
                    last_msg = self.session.history.messages[-1]
                    if last_msg.role == Role.ASSISTANT:
                        self._message_doc_blocks[id(last_msg)] = doc_blocks
            except Exception as e:
                logger.warning("[DOC_BLOCKS] Failed to track for serialization: %s", e)

        # Track persist-only doc blocks (already rendered live, need serialization only)
        persist_only = getattr(self, '_persist_only_doc_blocks', None)
        if persist_only:
            all_doc_blocks = list(persist_only)
            self._persist_only_doc_blocks.clear()
            try:
                if self.session.history.messages:
                    last_msg = self.session.history.messages[-1]
                    if last_msg.role == Role.ASSISTANT:
                        existing = self._message_doc_blocks.get(id(last_msg), [])
                        self._message_doc_blocks[id(last_msg)] = existing + all_doc_blocks
            except Exception as e:
                logger.warning("[DOC_BLOCKS] Failed to track persist-only blocks: %s", e)

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
        registry = get_default_registry()
        for tc in self._tool_calls:
            # Resolve display name from registry (honors MCP displayName)
            dn = tc.display_name
            td = registry.get(tc.name)
            if td:
                dn = td.get_display_name()
            tc_data = {
                "name": tc.name,
                "call_id": tc.call_id,
                "display_name": dn,
                "status": tc.status.value if hasattr(tc.status, "value") else str(tc.status),
                "is_image_generation": tc.is_image_generation,
            }
            # Include arguments and result for persistence & UI display
            if tc.arguments:
                tc_data["arguments"] = tc.arguments
            if tc.result is not None:
                # For rich_mcp results, store only the LLM summary (the full
                # envelope is already persisted as a fenced code block)
                result_str = tc.result

                # Extract __FLUX_SOURCES__ before truncation (flux source attribution)
                if isinstance(result_str, str) and "\n\n__FLUX_SOURCES__\n" in result_str:
                    parts = result_str.split("\n\n__FLUX_SOURCES__\n", 1)
                    result_str = parts[0]
                    try:
                        sources = json.loads(parts[1])
                        if isinstance(sources, list):
                            # Limit table rows to 50 for storage
                            for src in sources:
                                if isinstance(src, dict) and src.get("rows") and len(src["rows"]) > 50:
                                    src["rows"] = src["rows"][:50]
                            tc_data["sources"] = sources
                    except (json.JSONDecodeError, TypeError):
                        pass

                try:
                    parsed = result_str if isinstance(result_str, dict) else json.loads(result_str) if isinstance(result_str, str) else None
                    if isinstance(parsed, dict) and "__rich_mcp__" in parsed:
                        result_str = parsed["__rich_mcp__"].get("llm_summary", str(result_str))
                except (json.JSONDecodeError, TypeError, AttributeError):
                    pass
                # Truncate very large results for storage
                if isinstance(result_str, str) and len(result_str) > 4000:
                    result_str = result_str[:4000] + "…"
                tc_data["result"] = result_str
            if tc.error:
                tc_data["error"] = tc.error
            tool_calls_data.append(tc_data)

        # Track tool calls for this assistant message (for IndexedDB serialization)
        if tool_calls_data:
            try:
                if self.session.history.messages:
                    last_msg = self.session.history.messages[-1]
                    if last_msg.role == Role.ASSISTANT:
                        self._message_tool_calls[id(last_msg)] = tool_calls_data
            except Exception as e:
                logger.warning("[TOOL_CALLS] Failed to track for serialization: %s", e)

        # Check if tools belong to an MCP with a custom avatar
        used_tools = {tc.name for tc in self._tool_calls}
        mcp_avatar = self._get_mcp_avatar_for_tools(used_tools) if used_tools else None
        avatar_override = {"icon": mcp_avatar[0], "label": mcp_avatar[1]} if mcp_avatar else None

        # Track avatar override for serialization
        if avatar_override:
            try:
                if self.session.history.messages:
                    last_msg = self.session.history.messages[-1]
                    if last_msg.role == Role.ASSISTANT:
                        self._message_avatar_overrides[id(last_msg)] = avatar_override
            except Exception:
                pass

        msg = {
            "type": "response_completed",
            "full_text": full_text,
            "generated_image": generated_image,
            "tool_calls": tool_calls_data,
        }
        if avatar_override:
            msg["avatar_override"] = avatar_override
        asyncio.create_task(self._send(msg))

        # Mark remote task as completed
        remote_task_id = getattr(self, "_remote_task_id", None)
        if remote_task_id:
            coll = _get_remote_tasks_coll(self)
            if coll is not None:
                asyncio.create_task(
                    coll.update_one(
                        {"task_id": remote_task_id},
                        {"$set": {
                            "status": "completed",
                            "response": {"text": full_text, "model": self.model},
                        }},
                    )
                )
            self._remote_task_id = None

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
                save_json = json.dumps({"type": "save_conversation", "data": data}, ensure_ascii=False)
                logger.info("[SAVE] save_conversation payload: %d bytes (%.1f KB), messages=%d",
                            len(save_json), len(save_json) / 1024, len(data.get("messages", [])))
                await self._send({"type": "save_conversation", "data": data})
        except Exception as e:
            logger.warning("[SAVE] Conversation save failed: %s", e)

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
        """Lazy-init OpenAIMediaProvider from llming-models."""
        if not self._speech_service:
            from llming_models.media import OpenAIMediaProvider
            svc = OpenAIMediaProvider()
            if self._tts_model:
                svc.tts_model = self._tts_model
            self._speech_service = svc
        return self._speech_service

    @staticmethod
    def _get_tts_voices():
        from llming_models.media import OpenAIMediaProvider
        return [(v.id, v.name, v.gender) for v in OpenAIMediaProvider.VOICES]

    def _get_speech_service_default_voice(self) -> str:
        return self._tts_default_voice or "cedar"

    async def _auto_tts(self, text: str) -> None:
        """Synthesize TTS for the completed response and send to client (legacy single-shot)."""
        try:
            text = self._clean_text_for_tts(text)
            if not text:
                return
            service = self._get_speech_service()
            result = await service.synthesize(text, voice=self._tts_voice or "nova", language=self._locale, with_timings=True)
            audio_b64 = base64.b64encode(result.audio_bytes).decode("ascii")
            word_timings = [{"word": w.text, "start": w.start, "end": w.end} for w in result.word_timings]
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
            logger.info(f"[TTS] Synthesizing segment {segment} ({len(text)} chars)")
            result = await service.synthesize(text, voice=self._tts_voice or "nova", language=self._locale)
            mp3_bytes = result.audio_bytes
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
        # Persist any rich_mcp blocks that were received before cancellation
        if self._pending_rich_mcp_blocks:
            blocks = list(self._pending_rich_mcp_blocks)
            for entry in blocks:
                block_json = json.dumps(entry, ensure_ascii=False)
                self._text_content += f"\n\n```rich_mcp\n{block_json}\n```"
            self._pending_rich_mcp_blocks.clear()
            try:
                if self.session.history.messages:
                    last_msg = self.session.history.messages[-1]
                    if last_msg.role == Role.ASSISTANT:
                        self._message_rich_mcp[id(last_msg)] = blocks
            except Exception:
                pass
        # Same for pending doc blocks
        if self._pending_doc_blocks:
            doc_blocks = list(self._pending_doc_blocks)
            for doc_type, block_data in doc_blocks:
                block_json = json.dumps(block_data, ensure_ascii=False)
                self._text_content += f"\n\n```{doc_type}\n{block_json}\n```"
            self._pending_doc_blocks.clear()
            try:
                if self.session.history.messages:
                    last_msg = self.session.history.messages[-1]
                    if last_msg.role == Role.ASSISTANT:
                        self._message_doc_blocks[id(last_msg)] = doc_blocks
            except Exception:
                pass
        # Restore preamble if intercept was active
        _restore = getattr(self, '_restore_preamble', None)
        if _restore is not None:
            self.context_preamble = _restore
            self.session._context_preamble = _restore
            self._restore_preamble = None
        asyncio.create_task(self._send({"type": "response_cancelled"}))
        # Save the partial conversation so charts/blocks survive reload
        asyncio.create_task(self._post_response_tasks())

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
        # Guard @auto virtual model — not in the LLM registry
        old_info = None if old_model == "@auto" else llm_manager.get_model_info(old_model)
        new_info = None if new_model == "@auto" else llm_manager.get_model_info(new_model)

        if old_model == "@auto":
            old_label, old_icon = "Auto", "models/auto-select.svg"
        else:
            old_label = old_info.label if old_info else old_model
            old_icon = old_info.model_icon if old_info else ""

        if new_model == "@auto":
            new_label, new_icon = "Auto", "models/auto-select.svg"
        else:
            new_label = new_info.label if new_info else new_model
            new_icon = new_info.model_icon if new_info else ""

        asyncio.create_task(self._send({
            "type": "model_switched",
            "old_model": old_model,
            "new_model": new_model,
            "old_label": old_label,
            "new_label": new_label,
            "old_icon": old_icon,
            "new_icon": new_icon,
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

    def _serialize_messages(self, messages) -> list[dict]:
        """Serialize messages for IndexedDB storage.

        Re-injects rich_mcp blocks into assistant message content so that saved
        conversations render plots on reload, WITHOUT polluting the live
        server-side history that feeds the LLM.

        Also attaches tool_calls data to assistant messages so the UI can
        display tool usage details on conversation restore.
        """
        result = []
        for m in messages:
            data = m.model_dump(mode="json")
            if m.role == Role.ASSISTANT:
                # Use display text (interim narration stripped) if available
                display = self._message_display_text.get(id(m))
                if display is not None:
                    data["content"] = display
                blocks = self._message_rich_mcp.get(id(m))
                if blocks:
                    for entry in blocks:
                        block_json = json.dumps(entry, ensure_ascii=False)
                        data["content"] = (data.get("content") or "") + f"\n\n```rich_mcp\n{block_json}\n```"
                # Re-inject document blocks (plotly charts, etc. from tool results)
                doc_blocks = self._message_doc_blocks.get(id(m))
                if doc_blocks:
                    for doc_type, block_data in doc_blocks:
                        block_json = json.dumps(block_data, ensure_ascii=False)
                        data["content"] = (data.get("content") or "") + f"\n\n```{doc_type}\n{block_json}\n```"
                tc_data = self._message_tool_calls.get(id(m))
                if tc_data:
                    data["tool_calls"] = tc_data
                av_override = self._message_avatar_overrides.get(id(m))
                if av_override:
                    data["avatar_override"] = av_override
            result.append(data)
        return result

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
            "model": "@auto" if self._auto_select else self.model,
            "auto_underlying_model": self.model if self._auto_select else None,
            "model_defaults_version": llm_manager.user_config.global_config.model_defaults_version,
            "provider": self.provider,
            "messages": self._serialize_messages(messages),
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
            from llming_models.providers.llm_provider_models import ReasoningEffort
            from llming_models.messages import LlmSystemMessage, LlmHumanMessage

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
        # Prune rich_mcp + tool_calls tracking — condensed messages are gone
        if self.session:
            live_ids = {id(m) for m in self.session.history.messages}
            if self._message_rich_mcp:
                self._message_rich_mcp = {k: v for k, v in self._message_rich_mcp.items() if k in live_ids}
            if self._message_doc_blocks:
                self._message_doc_blocks = {k: v for k, v in self._message_doc_blocks.items() if k in live_ids}
            if self._message_tool_calls:
                self._message_tool_calls = {k: v for k, v in self._message_tool_calls.items() if k in live_ids}
            if self._message_avatar_overrides:
                self._message_avatar_overrides = {k: v for k, v in self._message_avatar_overrides.items() if k in live_ids}
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

        # Add "Auto" virtual model at the top (highest popularity)
        models.insert(0, {
            "model": "@auto",
            "label": "Auto",
            "icon": "models/auto-select.svg",
            "provider": "auto",
            "max_input_tokens": 200000,
            "max_output_tokens": 16384,
            "popularity": 100,
            "speed": 8,
            "quality": 9,
            "cost": 5,
            "memory": 9,
            "context_label": "200K",
            "best_use": "Smart routing",
            "highlights": ["Auto-selects", "Cost-efficient"],
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
            "current_model": "@auto" if self._auto_select else self.model,
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
            "model_defaults_version": llm_manager.user_config.global_config.model_defaults_version,
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

        # Send trusted droplet list for rich MCP rendering
        try:
            trusted_uids = await _get_trusted_nudge_uids(controller)
            if trusted_uids:
                await controller._send({
                    "type": "mcp_trust_list",
                    "nudge_uids": trusted_uids,
                })
        except Exception as e:
            logger.warning(f"[MCP_TRUST] Failed to send trust list: {e}")

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


def _refresh_datetime_preamble(controller: WebSocketChatController) -> None:
    """Update the context preamble with current date/time in the client's timezone."""
    tz_name = controller._client_timezone or "UTC"
    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        tz = ZoneInfo("UTC")
        tz_name = "UTC"
    now = datetime.now(tz)
    dt_line = f"Current date and time: {now.strftime('%A, %Y-%m-%d %H:%M')} ({tz_name})"

    # Update context_preamble: strip any previous datetime line, append fresh one
    preamble = controller.context_preamble or ""
    lines = preamble.split("\n")
    lines = [l for l in lines if not l.startswith("Current date and time:")]
    # Remove trailing empty lines before appending
    while lines and lines[-1] == "":
        lines.pop()
    lines.append(dt_line)
    new_preamble = "\n".join(lines)

    controller.context_preamble = new_preamble
    if controller.session:
        controller.session._context_preamble = new_preamble


_FOLLOWUP_PREAMBLE = """
## Interactive Follow-Up Questions

You can present interactive questions to the user using a fenced code block with language `followup` containing JSON. The UI renders clickable cards, multiple choice, input fields, and sorting exercises. Use this for quizzes, surveys, parameter collection, or any structured input.

Format:
```
​```followup
{
  "title": "Optional title",
  "steps": [
    {
      "title": "Optional step title",
      "elements": [
        {"type": "cards", "label": "Pick one",
         "options": [{"label": "Option A", "value": "a", "description": "optional desc"}]},
        {"type": "choice", "label": "Select", "id": "q1", "select": "single|multiple",
         "options": ["A", "B", "C"]},
        {"type": "text", "id": "name", "label": "Your name", "placeholder": "...", "required": true},
        {"type": "number", "id": "age", "label": "Age", "min": 0, "max": 150, "default": 25, "hint": "tip"},
        {"type": "sort", "label": "Categorize these",
         "categories": [{"label": "Cat A", "color": "#27ae60"}, {"label": "Cat B", "color": "#e67e22"}],
         "items": [{"label": "Item 1", "color": "#e74c3c"}, {"label": "Item 2", "color": "#3498db"}]}
      ]
    }
  ]
}
​```
```

Rules:
- Single-select cards/choices auto-submit on click. Multi-select and inputs show a Submit button.
- Multiple steps create a paginated form (prev/next navigation).
- The user's answers are auto-sent as "Q: <label> A: <value>" lines — you will receive them as a normal user message.
- **MANDATORY: ALWAYS use a `followup` block when you present options, multiple choice, or expect the user to pick from alternatives.** NEVER write "Reply with A, B, C, or D", NEVER list options as plain text expecting the user to type a letter. EVERY question with options MUST end with a `followup` block. This applies to EVERY message in a conversation, not just the first one — every round of a quiz, every follow-up question, every time you offer choices.
- For quizzes and multiple choice: put the question text and context in your markdown above, then ALWAYS add a `followup` block with `cards` or `choice` at the end. Do this for EVERY round.
- Cards get auto-colored from a palette (10 distinct colors) — you do NOT need to specify `color` unless you want a specific one.
- Columns are auto-detected: long text = 1 column, short labels = 2-3 columns. Override with `"columns": N` if needed.
- Use emojis in card labels to make them visually engaging (e.g. vocabulary quizzes, mood picks, category icons).
- Keep it concise — one followup block per message, at the end.
""".strip()


def _inject_followup_preamble(controller: "WebSocketChatController") -> None:
    """Append the followup format description to the context preamble."""
    marker = "## Interactive Follow-Up Questions"
    preamble = controller.context_preamble or ""
    if marker in preamble:
        return  # already injected
    new_preamble = preamble + "\n\n" + _FOLLOWUP_PREAMBLE
    controller.context_preamble = new_preamble
    if controller.session:
        controller.session._context_preamble = new_preamble


async def _execute_trust_gated_actions(
    controller: "WebSocketChatController",
    ctx: dict,
    nudge_uid: str,
    rich: dict,
) -> None:
    """Execute trust-gated actions from a rich MCP result.

    Handles inject_messages and trigger_llm_call for trusted/approved droplets.
    """
    # Inject messages into chat history
    inject = rich.get("inject_messages")
    if inject and isinstance(inject, list):
        for m in inject:
            role_str = m.get("role")
            content = m.get("content")
            if role_str in ("user", "assistant") and content:
                role = Role.USER if role_str == "user" else Role.ASSISTANT
                if controller.session:
                    controller.session.history.add_message(
                        ChatMessage(role=role, content=content)
                    )
                await controller._send({
                    "type": "injected_message",
                    "role": role_str,
                    "content": content,
                    "source_nudge": nudge_uid,
                })

    # Queue a follow-up LLM call
    trigger = rich.get("trigger_llm_call")
    if trigger and isinstance(trigger, dict) and trigger.get("prompt"):
        prompt = trigger["prompt"]

        async def _do_followup():
            try:
                if controller.session:
                    controller.session.history.add_message(
                        ChatMessage(role=Role.USER, content=prompt)
                    )
                    await controller._send({
                        "type": "injected_message",
                        "role": "user",
                        "content": prompt,
                        "source_nudge": nudge_uid,
                    })
                    await controller.generate_response()
            except Exception as e:
                logger.error("[RICH_MCP] Follow-up LLM call failed: %s", e)

        asyncio.create_task(_do_followup())


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
        # When the user sends only attachments (images/files) with no text,
        # provide a minimal prompt so all LLM providers have valid input.
        if not text and images:
            text = "Analyze the attached content."
        if text or images:
            # Built-in dev commands + app-specific intercept
            intercept_result = None
            if text and not images:
                from llming_lodge.dev.dev_commands import handle_dev_command
                try:
                    intercept_result = await handle_dev_command(text, controller)
                except Exception as e:
                    logger.warning("[WS] handle_dev_command error: %s", e)
                if intercept_result is None and controller._on_message_intercept:
                    try:
                        intercept_result = await controller._on_message_intercept(text, controller)
                    except Exception as e:
                        logger.warning("[WS] on_message_intercept error: %s", e)
                if intercept_result is not None:
                    if setup_hybrid_intercept(controller, intercept_result):
                        pass  # Hybrid: fall through to LLM send below
                    else:
                        # Pure intercept (dev commands etc.) — no LLM follow-up
                        # Store user + assistant messages in history for persistence
                        from llming_models.llm_base_models import ChatMessage, Role
                        controller.session.history.add_message(
                            ChatMessage(role=Role.USER, content=text))
                        controller.session.history.add_message(
                            ChatMessage(role=Role.ASSISTANT, content=intercept_result))

                        _model = controller.model if not controller._auto_select else controller._auto_select_base_model or controller.model
                        try:
                            model_info = llm_manager.get_model_info(_model)
                        except ValueError:
                            model_info = None
                        await controller._send({
                            "type": "response_started",
                            "model": controller.model,
                            "model_icon": model_info.model_icon if model_info else "",
                            "model_label": model_info.label if model_info else controller.model,
                            "intercept": True,
                        })
                        await controller._send({"type": "text_chunk", "content": intercept_result})
                        await controller._send({"type": "response_completed", "intercept": True, "full_text": intercept_result})
                        await controller._send({
                            "type": "tools_updated",
                            "tools": controller.get_all_known_tools(),
                        })
                        return

            async def _run_send():
                try:
                    _refresh_datetime_preamble(controller)
                    await controller.send_message(text, images=images)
                except Exception as e:
                    logger.error(f"[WS] send_message error: {e}")
            asyncio.create_task(_run_send())

            # Early save: persist conversation with user message immediately
            # so it appears in the sidebar before the LLM responds.
            async def _early_save():
                initial_count = len(controller.session.history.messages) if controller.session else 0
                for _ in range(20):
                    await asyncio.sleep(0.05)
                    if controller.session and len(controller.session.history.messages) > initial_count:
                        break
                data = controller._serialize_conversation()
                if data:
                    await controller._send({"type": "save_conversation", "data": data})
            asyncio.create_task(_early_save())

    elif msg_type == "client_hello":
        tz = msg.get("timezone", "UTC")
        controller._client_timezone = tz
        _refresh_datetime_preamble(controller)
        _inject_followup_preamble(controller)

        # Send bolt definitions from eligible nudges:
        # (a) system nudges, (b) master/global nudges, (c) user's favorites
        _bolt_bundles = []  # [(uid, bolts)]

        # (a) System nudges
        from llming_lodge.system_nudges import SYSTEM_NUDGE_REGISTRY
        for _sn_key, _sn in SYSTEM_NUDGE_REGISTRY.items():
            _sn_bolts = _sn.get("bolts", [])
            if _sn_bolts:
                _bolt_bundles.append((_sn.get("uid", f"sys:{_sn_key}"), _sn_bolts))

        # (b) Master nudges (already loaded on controller during page setup)
        for _mn in getattr(controller, "_master_nudges_raw", []):
            _mn_bolts = _mn.get("bolts", [])
            if _mn_bolts:
                _bolt_bundles.append((_mn.get("uid"), _mn_bolts))

        # (c) Favorited nudges + any live nudge with bolts visible to this user
        _already = {uid for uid, _ in _bolt_bundles}
        if getattr(controller, "_nudge_store", None):
            try:
                store = controller._nudge_store
                coll, _ = store._ensure_colls()
                _query = {"mode": "live", "bolts": {"$exists": True, "$ne": []}}
                async for _fn in coll.find(
                    _query,
                    {"uid": 1, "bolts": 1, "_id": 0},
                ):
                    if _fn["uid"] not in _already:
                        _bolt_bundles.append((_fn["uid"], _fn["bolts"]))
                        _already.add(_fn["uid"])
            except Exception as _e:
                logger.warning(f"[BOLTS] Failed to load nudge bolts: {_e}")

        # Send all bolt bundles
        logger.info("[BOLTS] Sending %d bolt bundle(s) to client", len(_bolt_bundles))
        for _buid, _bbolts in _bolt_bundles:
            logger.info("[BOLTS]   %s: %d bolt(s)", _buid, len(_bbolts))
            await controller._send({
                "type": "bolt_defs",
                "nudge_uid": _buid,
                "bolts": _bbolts,
            })

    elif msg_type == "start_bolt_worker":
        # Lightweight Worker activation for bolt system — no chat reset.
        # The bolt engine calls this when a worker_decode bolt fires outside
        # the droplet and no Worker is running yet.
        _bolt_uid = msg.get("nudge_uid")
        if _bolt_uid and getattr(controller, "_nudge_store", None):
            store = controller._nudge_store
            user_teams = getattr(controller, "_user_teams", None)
            _sid = controller.session_id

            if _sid not in _browser_mcp_sessions:
                _browser_mcp_sessions[_sid] = {
                    "store": store,
                    "user_email": controller.user_mail or "",
                    "user_teams": user_teams or [],
                    "uids": {_bolt_uid},
                    "loop": asyncio.get_running_loop(),
                    "controller": controller,
                    "pending_requests": {},
                    "active_mcp_nudges": set(),
                    "active_tool_names": {},
                }
            else:
                _browser_mcp_sessions[_sid]["uids"].add(_bolt_uid)

            if _bolt_uid not in _browser_mcp_sessions[_sid].get("active_mcp_nudges", set()):
                async def _deferred_bolt_activate(_ctx, _u):
                    await asyncio.sleep(0.05)
                    try:
                        res = await _activate_browser_mcp(_ctx, _u)
                        logger.info("[BOLT_WORKER] Activated for %s: %s", _u, res)
                    except Exception as exc:
                        logger.warning("[BOLT_WORKER] Activation failed for %s: %s", _u, exc)

                asyncio.create_task(
                    _deferred_bolt_activate(
                        _browser_mcp_sessions[_sid], _bolt_uid,
                    )
                )

    elif msg_type == "store_bolt_result":
        # Persist a bolt result as user + assistant messages.
        # Supports rich_mcp (fenced code block) or plain assistant_text.
        _user_text = msg.get("user_text", "")
        _rich_data = msg.get("rich_mcp")
        _asst_text = msg.get("assistant_text")
        if _user_text and (_rich_data or _asst_text) and controller.session:
            user_msg = ChatMessage(role=Role.USER, content=_user_text)
            controller.session.history.add_message(user_msg)
            if _rich_data:
                block_json = json.dumps(_rich_data, ensure_ascii=False)
                asst_content = f"```rich_mcp\n{block_json}\n```"
            else:
                asst_content = _asst_text
            asst_msg = ChatMessage(role=Role.ASSISTANT, content=asst_content)
            controller.session.history.add_message(asst_msg)
            try:
                data = controller._serialize_conversation()
                if data:
                    await controller._send({"type": "save_conversation", "data": data})
            except Exception as e:
                logger.warning("[BOLT] store_bolt_result save failed: %s", e)

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
        # Cancel any in-flight streaming
        if controller._streaming_task and not controller._streaming_task.done():
            await controller.stop_streaming()

        # Save current conversation first
        data = controller._serialize_conversation()
        if data:
            await controller._send({"type": "save_conversation", "data": data})

        # Built-in dev cleanup (dev MCPs, remote mode stays active)
        from llming_lodge.dev.dev_mcp import deactivate_all_dev_mcps
        try:
            await deactivate_all_dev_mcps(controller)
        except Exception as e:
            logger.warning("[WS] deactivate_all_dev_mcps error: %s", e)

        # App-specific cleanup
        if controller._on_new_chat:
            try:
                await controller._on_new_chat(controller)
            except Exception as e:
                logger.warning("[WS] on_new_chat error: %s", e)

        # Clear history and create new session
        controller.clear_history()
        controller._message_rich_mcp.clear()
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

        # Unlock model and restore user's preference from before droplet
        if getattr(controller, '_model_locked', False):
            controller._model_locked = False
            controller._model_locked_reason = ""
            await controller._send({"type": "model_unlocked"})
            # Restore auto-select state
            if getattr(controller, '_saved_auto_select', False):
                controller._auto_select = True
                controller._saved_auto_select = False
            saved = getattr(controller, '_saved_model_pref', None)
            if saved:
                if saved == "@auto":
                    controller._auto_select = True
                else:
                    await controller.switch_model(saved, _force=True)
                controller._saved_model_pref = None
                logger.info("[NEW_CHAT] Restored model preference: %s", saved)

        # Restore app-level default system prompt
        controller._base_system_prompt = controller._app_system_prompt
        if controller._base_system_prompt:
            controller.update_settings(system_prompt=controller._base_system_prompt)

        # Apply preset if provided (project or nudge)
        preset = msg.get("preset")
        nudge_meta = None
        if preset:
            preset_type = preset.get("type", "project")

            # Server-side nudge fetch: nudge_uid → fetch from registry or MongoDB
            if preset_type == "nudge" and preset.get("nudge_uid") and getattr(controller, "_nudge_store", None):
                try:
                    store = controller._nudge_store
                    user_teams = getattr(controller, "_user_teams", None)
                    is_admin = getattr(controller, "_is_nudge_admin", False)

                    # System nudges: load from in-memory registry
                    _preset_uid = preset["nudge_uid"]
                    if is_system_nudge_uid(_preset_uid):
                        nudge = get_system_nudge(_preset_uid)
                    else:
                        nudge = await store.get_for_user(_preset_uid, controller.user_mail or "", user_teams=user_teams, is_admin=is_admin)
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
                            "bolts": nudge.get("bolts", []),
                            "translations": nudge.get("translations"),
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

                            # Signal for remote mode polling to detect MCP readiness
                            _mcp_ready_event = asyncio.Event()
                            controller._mcp_ready_event = _mcp_ready_event

                            async def _deferred_mcp_activate(_ctx, _u, _evt):
                                """Run after a short yield so chat_cleared arrives first."""
                                await asyncio.sleep(0.2)
                                try:
                                    res = await _activate_browser_mcp(_ctx, _u)
                                    logger.info("[NUDGE] Auto-activated MCP for %s: %s", _u, res)
                                except Exception as exc:
                                    logger.warning("[NUDGE] MCP auto-activation failed for %s: %s", _u, exc)
                                finally:
                                    _evt.set()

                            asyncio.create_task(
                                _deferred_mcp_activate(
                                    _browser_mcp_sessions[new_session_id], _uid,
                                    _mcp_ready_event,
                                )
                            )

                        # Auto-activate server-side MCP if nudge has server_mcp capability.
                        # Unlike browser MCP, this doesn't need deferred activation —
                        # it's in-process and doesn't require WS round-trip.
                        _server_mcp_class = (nudge.get("capabilities") or {}).get("server_mcp")
                        if _server_mcp_class:
                            async def _deferred_server_mcp(_ctrl, _ndg):
                                await asyncio.sleep(0.2)
                                try:
                                    res = await _activate_server_mcp(_ctrl, _ndg)
                                    logger.info("[NUDGE] Auto-activated server MCP: %s", res)
                                except Exception as exc:
                                    logger.warning("[NUDGE] Server MCP activation failed: %s", exc)

                            asyncio.create_task(_deferred_server_mcp(controller, nudge))

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

        # Send bolt definitions to client if nudge has bolts
        if nudge_meta and nudge_meta.get("bolts"):
            await controller._send({
                "type": "bolt_defs",
                "nudge_uid": nudge_meta["uid"],
                "bolts": nudge_meta["bolts"],
            })

    elif msg_type == "load_conversation":
        # Cancel any in-flight streaming
        if controller._streaming_task and not controller._streaming_task.done():
            await controller.stop_streaming()
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

    elif msg_type == "console_logs":
        if hasattr(controller, '_pending_console_logs') and controller._pending_console_logs:
            try:
                controller._pending_console_logs.set_result(msg)
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
                stt_result = await service.transcribe(audio_bytes, filename=filename, content_type=content_type, language=lang)
                await controller._send({"type": "transcription_result", "text": stt_result.text})
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
                result = await service.synthesize(text, voice=controller._tts_voice or "nova", language=controller._locale, with_timings=True)
                audio_b64 = base64.b64encode(result.audio_bytes).decode("ascii")
                await controller._send({
                    "type": "tts_audio",
                    "audio_b64": audio_b64,
                    "word_timings": [{"word": w.text, "start": w.start, "end": w.end} for w in result.word_timings],
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
            from llming_models.media import OpenAIMediaProvider
            service = controller._get_speech_service()
            voice = controller._tts_voice or controller._tts_default_voice or "cedar"
            instructions = controller._base_system_prompt or "You are a helpful assistant."
            lang_name = OpenAIMediaProvider._LOCALE_NAMES.get(controller._locale, "")
            if lang_name:
                instructions += f"\nSpeak in {lang_name}."

            # Gather tool definitions for Realtime function calling
            # Only include tools explicitly flagged for realtime use
            rt_tools = []
            from llming_models.tools.tool_definition import ToolSource
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
        entry.last_heartbeat = time.monotonic()
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

    elif msg_type == "nudge_unpublish":
        await _handle_nudge_unpublish(controller, msg)

    elif msg_type == "nudge_revert":
        await _handle_nudge_revert(controller, msg)

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
        # Look up by current session_id first, then fall back to searching all
        # sessions by controller or request_id.  After page reload/navigation
        # the session_id changes but the browser MCP context may still be
        # keyed under the old session_id.
        ctx = _browser_mcp_sessions.get(controller.session_id)
        if not ctx or not request_id or request_id not in (ctx or {}).get("pending_requests", {}):
            # Fallback: find the context that owns this request_id
            for _sid, _ctx in _browser_mcp_sessions.items():
                if request_id and request_id in _ctx.get("pending_requests", {}):
                    ctx = _ctx
                    # Migrate to current session_id so future lookups work
                    if _sid != controller.session_id:
                        _browser_mcp_sessions[controller.session_id] = _browser_mcp_sessions.pop(_sid)
                        ctx["controller"] = controller
                        logger.info("[BROWSER_MCP] Migrated context from session %s → %s", _sid, controller.session_id)
                    break
        if ctx and request_id and request_id in ctx.get("pending_requests", {}):
            pending = ctx["pending_requests"].pop(request_id)
            future = pending["future"] if isinstance(pending, dict) else pending
            if not future.done():
                future.set_result(msg)
        else:
            logger.warning("[BROWSER_MCP] Unexpected result for request_id=%s", request_id)

    # ── MCP Droplet Trust management ─────────────────────
    elif msg_type == "mcp_trust:grant":
        nudge_uid = msg.get("nudge_uid")
        if nudge_uid:
            coll = _get_trust_coll(controller)
            if coll:
                await _ensure_trust_indexes(coll)
                await coll.update_one(
                    {"user_email": (controller.user_mail or "").lower(), "nudge_uid": nudge_uid},
                    {"$set": {"trusted": True, "granted_at": datetime.now(timezone.utc)}},
                    upsert=True,
                )
                logger.info("[MCP_TRUST] Granted trust for nudge %s by %s", nudge_uid, controller.user_mail)

    elif msg_type == "mcp_trust:revoke":
        nudge_uid = msg.get("nudge_uid")
        if nudge_uid:
            coll = _get_trust_coll(controller)
            if coll:
                await _ensure_trust_indexes(coll)
                await coll.delete_one({
                    "user_email": (controller.user_mail or "").lower(),
                    "nudge_uid": nudge_uid,
                })
                logger.info("[MCP_TRUST] Revoked trust for nudge %s by %s", nudge_uid, controller.user_mail)

    elif msg_type == "mcp_trust:list":
        uids = await _get_trusted_nudge_uids(controller)
        await controller._send({"type": "mcp_trust_list", "nudge_uids": uids})

    elif msg_type == "mcp_trust:execute_once":
        # One-time execution of trust-gated actions (user approved via prompt)
        nudge_uid = msg.get("nudge_uid")
        ctx = _browser_mcp_sessions.get(controller.session_id)
        if nudge_uid and ctx:
            rich = {
                "inject_messages": msg.get("inject_messages"),
                "trigger_llm_call": msg.get("trigger_llm_call"),
            }
            await _execute_trust_gated_actions(controller, ctx, nudge_uid, rich)

    # ── API Key management ────────────────────────────────
    elif msg_type == "apikeys:list":
        coll = _get_api_keys_coll(controller)
        if coll is None:
            await controller._send({"type": "apikeys:list", "keys": []})
            return
        await _ensure_api_keys_indexes(coll)
        user_email = controller.user_mail or ""
        docs = await coll.find({"user_email": user_email}).to_list(100)
        keys = []
        for d in docs:
            keys.append({
                "key_id": d["key_id"],
                "name": d.get("name", ""),
                "key_prefix": d.get("key_prefix", ""),
                "permissions": d.get("permissions", []),
                "created_at": d.get("created_at", "").isoformat() if isinstance(d.get("created_at"), datetime) else str(d.get("created_at", "")),
            })
        await controller._send({"type": "apikeys:list", "keys": keys})

    elif msg_type == "apikeys:create":
        coll = _get_api_keys_coll(controller)
        if coll is None:
            await controller._send({"type": "error", "error_type": "ApiKeyError", "message": "API keys not available"})
            return
        await _ensure_api_keys_indexes(coll)
        user_email = controller.user_mail or ""
        name = msg.get("name", "Unnamed Key").strip() or "Unnamed Key"
        permissions = msg.get("permissions", ["manage_droplets", "automate_chat"])
        # Generate key
        raw_key = "llming_" + secrets.token_hex(16)
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        key_id = str(uuid4())
        doc = {
            "key_id": key_id,
            "user_email": user_email,
            "name": name,
            "key_prefix": raw_key[:10],
            "key_hash": key_hash,
            "permissions": permissions,
            "created_at": datetime.now(timezone.utc),
        }
        await coll.insert_one(doc)
        await controller._send({
            "type": "apikeys:created",
            "key_id": key_id,
            "name": name,
            "full_key": raw_key,
            "permissions": permissions,
        })

    elif msg_type == "apikeys:delete":
        coll = _get_api_keys_coll(controller)
        if coll is None:
            return
        user_email = controller.user_mail or ""
        key_id = msg.get("key_id", "")
        result = await coll.delete_one({"key_id": key_id, "user_email": user_email})
        if result.deleted_count:
            await controller._send({"type": "apikeys:deleted", "key_id": key_id})
        else:
            await controller._send({"type": "error", "error_type": "ApiKeyError", "message": "Key not found"})

    # ── App Extensions ──────────────────────────────────────
    elif msg_type == "app_ext:activate":
        ext_mgr = getattr(controller, "_app_ext_manager", None)
        ext_name = msg.get("name", "")
        if ext_mgr and ext_name:
            config = await ext_mgr.activate(ext_name, controller)
            await controller._send({
                "type": "app_ext:activated",
                "name": ext_name,
                "config": config or {},
            })
        else:
            await controller._send({
                "type": "app_ext:activated",
                "name": ext_name,
                "config": {},
            })

    elif msg_type == "app_ext:deactivate":
        ext_mgr = getattr(controller, "_app_ext_manager", None)
        ext_name = msg.get("name", "")
        if ext_mgr and ext_name:
            await ext_mgr.deactivate(ext_name, controller)

    elif msg_type == "app_ext:message":
        ext_mgr = getattr(controller, "_app_ext_manager", None)
        ext_name = msg.get("name", "")
        payload = msg.get("payload", {})
        if ext_mgr and ext_name:
            response = await ext_mgr.handle_message(ext_name, controller, payload)
            if response is not None:
                await controller._send({
                    "type": "app_ext:message",
                    "name": ext_name,
                    "payload": response,
                })

    else:
        # Check for custom WS message handlers registered by external code
        custom_handlers = getattr(controller, "_custom_ws_handlers", None)
        if custom_handlers and msg_type in custom_handlers:
            await custom_handlers[msg_type](msg, controller)
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

    from llming_docs.manager import ALL_DOC_PLUGIN_TYPES, TYPE_TOOL_PREFIXES

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
    from llming_docs.manager import _MCP_SERVERS, _TYPE_ALIASES

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

        # Migrate browser MCP session context to new session ID
        if old_session_id in _browser_mcp_sessions:
            bctx = _browser_mcp_sessions.pop(old_session_id)
            bctx["controller"] = controller
            _browser_mcp_sessions[conv_id] = bctx
            controller.tool_config["activate_mcp_nudge"] = {"_session_id": conv_id}

        # Re-register in registry
        registry = SessionRegistry.get()
        registry.remove(old_session_id)
        registry._sessions[conv_id] = entry

        # Switch model if needed — but only if the saved version matches the
        # current server config version.  When model defaults change (e.g. new
        # default model), bumping model_defaults_version causes all clients to
        # use the new default instead of restoring a stale selection.
        saved_version = data.get("model_defaults_version")
        current_version = llm_manager.user_config.global_config.model_defaults_version
        loaded_model = data.get("model")
        if loaded_model and loaded_model != controller.model and saved_version == current_version:
            try:
                await controller.switch_model(loaded_model)
            except Exception as ex:
                logger.warning(f"[LOAD] Could not switch to model {loaded_model}: {ex}")
        elif loaded_model and saved_version != current_version:
            logger.info("[LOAD] Ignoring stored model %s (saved version %s != current %s)",
                        loaded_model, saved_version, current_version)
            # Ensure auto-select is enabled (the new default)
            if not controller._auto_select:
                await controller.switch_model("@auto")

        # Restore base system prompt
        if data.get("base_system_prompt"):
            controller._base_system_prompt = data["base_system_prompt"]
            controller.update_settings(system_prompt=data["base_system_prompt"])

        # Rebuild history (restore tool_calls metadata for serialization round-trips)
        controller.session.history = ChatHistory()
        controller._message_tool_calls = {}
        for msg_dict in data.get("messages", []):
            msg = ChatMessage.model_validate(msg_dict)
            controller.session.history.add_message(msg)
            # Restore tool_calls tracking for this message
            tc_data = msg_dict.get("tool_calls")
            if tc_data and msg.role == Role.ASSISTANT:
                controller._message_tool_calls[id(msg)] = tc_data
            # Restore avatar override
            av_data = msg_dict.get("avatar_override")
            if av_data and msg.role == Role.ASSISTANT:
                controller._message_avatar_overrides[id(msg)] = av_data

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
                _reload_uid = data["nudge_id"]

                # System nudges: load from registry (no MongoDB, no file cache)
                if is_system_nudge_uid(_reload_uid):
                    nudge = get_system_nudge(_reload_uid)
                else:
                    from llming_lodge.nudge_store import get_file_cache
                    cached = await get_file_cache().get_files(
                        _reload_uid, store, controller.user_mail or "",
                        user_teams=user_teams,
                    )
                    if cached:
                        preset_files = cached
                    nudge = await store.get_for_user(_reload_uid, controller.user_mail or "", user_teams=user_teams, is_admin=getattr(controller, "_is_nudge_admin", False))
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
                        "bolts": nudge.get("bolts", []),
                        "translations": nudge.get("translations"),
                    }
                    # Apply doc plugin config from nudge
                    if nudge.get("doc_plugins") is not None:
                        await _apply_doc_plugins(controller, entry, nudge["doc_plugins"])

                    # Re-activate browser MCP if nudge has JS files
                    _nudge_has_js = any(
                        (f.get("name", "").endswith(".js") or f.get("name", "").endswith(".mjs"))
                        for f in nudge.get("files", [])
                    )
                    if _nudge_has_js:
                        _uid = nudge["uid"]
                        if conv_id not in _browser_mcp_sessions:
                            _browser_mcp_sessions[conv_id] = {
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
                            _browser_mcp_sessions[conv_id]["uids"].add(_uid)
                            _browser_mcp_sessions[conv_id]["controller"] = controller

                        async def _deferred_mcp_reactivate(_ctx, _u):
                            await asyncio.sleep(0.3)
                            try:
                                res = await _activate_browser_mcp(_ctx, _u)
                                logger.info("[LOAD] Re-activated MCP for %s: %s", _u, res)
                            except Exception as exc:
                                logger.warning("[LOAD] MCP re-activation failed for %s: %s", _u, exc)

                        asyncio.create_task(
                            _deferred_mcp_reactivate(
                                _browser_mcp_sessions[conv_id], _uid,
                            )
                        )

                    # Re-activate server-side MCP if nudge has server_mcp capability
                    _server_mcp_class = (nudge.get("capabilities") or {}).get("server_mcp")
                    if _server_mcp_class:
                        async def _deferred_server_mcp_reload(_ctrl, _ndg):
                            await asyncio.sleep(0.3)
                            try:
                                res = await _activate_server_mcp(_ctrl, _ndg)
                                logger.info("[LOAD] Re-activated server MCP: %s", res)
                            except Exception as exc:
                                logger.warning("[LOAD] Server MCP re-activation failed: %s", exc)

                        asyncio.create_task(_deferred_server_mcp_reload(controller, nudge))
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
    query = msg.get("query", "")
    category = msg.get("category", "")
    mine = msg.get("mine", False)
    page = msg.get("page", 0)
    try:
        results, has_more = await store.search(
            controller.user_mail or "",
            query=query,
            category=category,
            mine=mine,
            page=page,
            page_size=24,
            user_teams=user_teams,
            include_master=is_admin,
            all_users=all_users,
        )

        # Inject matching system nudges on first page (not "mine" tab)
        if page == 0 and not mine and not all_users:
            sys_keys = getattr(controller, "_system_nudge_keys", None)
            sys_matches = search_system_nudges(sys_keys, query=query, category=category)
            if sys_matches:
                sys_uids = {n["uid"] for n in sys_matches}
                # Deduplicate (system nudges first, then MongoDB results)
                results = [_sys_nudge_meta(n) for n in sys_matches] + [
                    r for r in results if r.get("uid") not in sys_uids
                ]

        await controller._send({
            "type": "nudge_search_result",
            "nudges": results,
            "page": page,
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

        # System nudges: serve from in-memory registry (read-only)
        if is_system_nudge_uid(uid):
            nudge = get_system_nudge(uid)
            if nudge:
                nudge = dict(nudge)  # copy so we don't mutate registry
                nudge["_has_live"] = False
                nudge["_live_matches"] = True
            await controller._send({"type": "nudge_detail", "nudge": nudge})
            return

        mode = msg.get("mode")
        if mode:
            nudge = await store.get(uid, mode, controller.user_mail or "", user_teams=user_teams)
        else:
            nudge = await store.get_for_user(uid, controller.user_mail or "", user_teams=user_teams, is_admin=getattr(controller, "_is_nudge_admin", False))

        # Enrich dev nudges with live publication status for editors
        if nudge and nudge.get("mode") == "dev" and store._user_can_edit(nudge, controller.user_mail or "", user_teams):
            live = await store.get(uid, "live", controller.user_mail or "", user_teams)
            nudge["_has_live"] = live is not None
            if live:
                compare_keys = [
                    "name", "description", "system_prompt", "model", "language",
                    "suggestions", "capabilities", "category", "sub_category",
                    "visibility", "icon", "is_master", "auto_discover",
                    "auto_discover_when", "doc_plugins", "version",
                ]
                matches = all(nudge.get(k) == live.get(k) for k in compare_keys)
                # Also compare files by name+size (not content, too heavy)
                dev_files = [(f.get("name"), f.get("size")) for f in (nudge.get("files") or [])]
                live_files = [(f.get("name"), f.get("size")) for f in (live.get("files") or [])]
                if dev_files != live_files:
                    matches = False
                nudge["_live_matches"] = matches
                nudge["_live_version"] = live.get("version", "")
            else:
                nudge["_live_matches"] = False

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
    # System nudges are read-only
    if is_system_nudge_uid(msg.get("data", {}).get("uid", "")):
        await controller._send({"type": "nudge_error", "error": "System nudges cannot be modified"})
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
    if is_system_nudge_uid(msg.get("uid", "")):
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
    if is_system_nudge_uid(msg.get("uid", "")):
        return
    user_teams = getattr(controller, "_user_teams", None)
    try:
        uid = msg.get("uid", "")
        is_admin = getattr(controller, "_is_nudge_admin", False)
        ok = await store.flush_to_live(uid, controller.user_mail or "", user_teams=user_teams, is_admin=is_admin)
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


async def _handle_nudge_unpublish(controller: WebSocketChatController, msg: dict):
    store = getattr(controller, "_nudge_store", None)
    if not store:
        return
    if is_system_nudge_uid(msg.get("uid", "")):
        return
    user_teams = getattr(controller, "_user_teams", None)
    try:
        uid = msg.get("uid", "")
        is_admin = getattr(controller, "_is_nudge_admin", False)
        ok = await store.unpublish(uid, controller.user_mail or "", user_teams=user_teams, is_admin=is_admin)
        await controller._send({
            "type": "nudge_unpublished",
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
        logger.error(f"[NUDGE] Unpublish failed: {e}", exc_info=True)


async def _handle_nudge_revert(controller: WebSocketChatController, msg: dict):
    store = getattr(controller, "_nudge_store", None)
    if not store:
        return
    if is_system_nudge_uid(msg.get("uid", "")):
        return
    user_teams = getattr(controller, "_user_teams", None)
    try:
        uid = msg.get("uid", "")
        is_admin = getattr(controller, "_is_nudge_admin", False)
        ok = await store.revert_to_live(uid, controller.user_mail or "", user_teams=user_teams, is_admin=is_admin)
        await controller._send({
            "type": "nudge_reverted",
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
        logger.error(f"[NUDGE] Revert failed: {e}", exc_info=True)


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
        # System nudge UIDs are always valid (visible to everyone)
        sys_valid = [u for u in uids if is_system_nudge_uid(u) and get_system_nudge(u)]
        db_uids = [u for u in uids if not is_system_nudge_uid(u)]
        valid_uids = sys_valid
        if db_uids:
            valid_uids += await store.validate_visible(db_uids, controller.user_mail or "", user_teams)
        await controller._send({
            "type": "nudge_favorites_validated",
            "valid_uids": valid_uids,
        })
    except Exception as e:
        logger.error(f"[NUDGE] Favorites validation failed: {e}", exc_info=True)
