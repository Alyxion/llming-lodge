"""Remote debug API for chat sessions and nudges.

Provides REST endpoints to inspect and control active chat sessions,
and CRUD endpoints for nudges (MongoDB, no session needed).
Protected by API key + IP whitelist + DEBUG_CHAT_REMOTE env flag.

Endpoints:
  GET  /debug/sessions          — list all active sessions
  GET  /debug/sessions/{id}     — session detail (history, state, tools)
  POST /debug/sessions/{id}/send — send a message
  GET  /debug/sessions/{id}/status — check streaming status
  POST /debug/sessions/{id}/model  — switch model
  POST /debug/sessions/{id}/new_chat — start new chat (optionally with preset)
  GET  /debug/sessions/{id}/conversations — list IDB conversations (via browser WS)
  POST /debug/sessions/{id}/load_conversation — load a conversation by ID

  POST /debug/sessions/{id}/scroll            — scroll chat (bottom / message index)
  POST /debug/sessions/{id}/open_droplet      — open a droplet (nudge) in new chat
  POST /debug/sessions/{id}/open_project      — open a project view
  POST /debug/sessions/{id}/open_conversation — switch to a conversation

  POST /debug/sessions/{id}/inject_file    — inject file via browser (same as attach)
  POST /debug/sessions/{id}/remove_file_ui — remove pending file via browser

  GET    /debug/sessions/{id}/presets           — list presets (nudges/projects)
  GET    /debug/sessions/{id}/presets/{pid}     — get full preset (incl. files)
  PUT    /debug/sessions/{id}/presets           — save/update a preset
  DELETE /debug/sessions/{id}/presets/{pid}     — delete a preset

  GET  /debug/sessions/{id}/idb_files              — list IDB file store entries
  GET  /debug/sessions/{id}/idb_files/{hash}       — get single file metadata
  GET  /debug/sessions/{id}/idb_file_refs/{conv_id} — get file refs for a conversation

  POST /debug/sessions/{id}/doc — unified document control (browser + AI commands)

  GET  /debug/nudges           — list nudges (query, category, mode, pagination)
  GET  /debug/nudges/{uid}     — get full nudge by uid (create if not found)
  PUT  /debug/nudges/{uid}     — create or update nudge fields
  PUT  /debug/nudges/{uid}/files — replace nudge files array
"""

import ipaddress
import logging
import os
import time
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from llming_lodge.server import API_PREFIX

from llming_lodge.api.chat_session_api import _sync_pdf_viewer_mcp
from llming_lodge.llm_base_models import Role

logger = logging.getLogger(__name__)

# ── Security ────────────────────────────────────────────────────────

_API_KEY: Optional[str] = None
_IP_NETWORKS: list = []
_ENABLED: bool = False


def _load_config():
    """Load config from env (called once at import, re-callable for tests)."""
    global _API_KEY, _IP_NETWORKS, _ENABLED
    _ENABLED = os.environ.get("DEBUG_CHAT_REMOTE", "0") == "1"
    _API_KEY = os.environ.get("DEBUG_CHAT_API_KEY", "")
    raw = os.environ.get("DEBUG_CHAT_IP_WHITELIST", "127.0.0.1,::1")
    _IP_NETWORKS = []
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        try:
            _IP_NETWORKS.append(ipaddress.ip_network(entry, strict=False))
        except ValueError:
            logger.warning(f"[DEBUG-API] Invalid IP/network in whitelist: {entry}")


_load_config()


def _check_auth(request: Request):
    """FastAPI dependency: verify enabled + API key + IP whitelist."""
    if not _ENABLED:
        raise HTTPException(404, "Not found")

    # Check API key (header or query param)
    key = request.headers.get("x-debug-key") or request.query_params.get("key")
    if not key or key != _API_KEY:
        raise HTTPException(403, "Invalid API key")

    # Check IP
    client_ip = request.client.host if request.client else "unknown"
    try:
        addr = ipaddress.ip_address(client_ip)
        if not any(addr in net for net in _IP_NETWORKS):
            logger.warning(f"[DEBUG-API] Blocked IP: {client_ip}")
            raise HTTPException(403, f"IP {client_ip} not whitelisted")
    except ValueError:
        raise HTTPException(403, f"Cannot parse IP: {client_ip}")


# ── Request models ──────────────────────────────────────────────────

class SendMessageRequest(BaseModel):
    text: str
    images: Optional[list] = None


class SwitchModelRequest(BaseModel):
    model: str


class NudgeUpdateRequest(BaseModel):
    """Partial update for a nudge. Only provided fields are merged."""
    name: Optional[str] = None
    description: Optional[str] = None
    icon: Optional[str] = None
    category: Optional[str] = None
    sub_category: Optional[str] = None
    system_prompt: Optional[str] = None
    model: Optional[str] = None
    language: Optional[str] = None
    creator_email: Optional[str] = None
    creator_name: Optional[str] = None
    team_id: Optional[str] = None
    visibility: Optional[list[str]] = None
    suggestions: Optional[list] = None
    capabilities: Optional[list] = None
    files: Optional[list] = None


# ── Router ──────────────────────────────────────────────────────────

def build_debug_router(*, nudge_store=None) -> APIRouter:
    """Build the debug API router. Include only when DEBUG_CHAT_REMOTE=1."""
    from .chat_session_api import SessionRegistry

    router = APIRouter(
        prefix=f"{API_PREFIX}/debug",
        dependencies=[Depends(_check_auth)],
    )

    # ── List sessions ───────────────────────────────────────

    @router.get("/sessions")
    async def list_sessions():
        registry = SessionRegistry.get()
        sessions = []
        now = time.monotonic()
        for sid, entry in registry._sessions.items():
            ctrl = entry.controller
            msg_count = len(ctrl.session.history.messages) if ctrl.session else 0
            sessions.append({
                "session_id": sid,
                "user_id": entry.user_id,
                "user_name": entry.user_name,
                "model": ctrl.model,
                "streaming": bool(ctrl._streaming_task and not ctrl._streaming_task.done()),
                "ws_connected": entry.websocket is not None,
                "message_count": msg_count,
                "idle_seconds": round(now - entry.last_activity),
                "title": ctrl._conversation_title,
            })
        return {"count": len(sessions), "sessions": sessions}

    # ── Session detail ──────────────────────────────────────

    @router.get("/sessions/{session_id}")
    async def get_session(session_id: str, include_history: bool = True):
        registry = SessionRegistry.get()
        entry = registry.get_session(session_id)
        if not entry:
            raise HTTPException(404, f"Session {session_id} not found")

        ctrl = entry.controller
        result = {
            "session_id": session_id,
            "user_id": entry.user_id,
            "user_name": entry.user_name,
            "model": ctrl.model,
            "provider": ctrl.provider,
            "temperature": ctrl.temperature,
            "max_input_tokens": ctrl.max_input_tokens,
            "max_output_tokens": ctrl.max_output_tokens,
            "system_prompt": ctrl._base_system_prompt,
            "streaming": bool(ctrl._streaming_task and not ctrl._streaming_task.done()),
            "ws_connected": entry.websocket is not None,
            "title": ctrl._conversation_title,
            "enabled_tools": ctrl.enabled_tools,
            "available_tools": ctrl.available_tools,
            "budget": await ctrl.available_budget_async(),
            "condensed_summary": ctrl.session._condensed_summary if ctrl.session else None,
        }

        # Upload manager / attached files
        um = entry.upload_manager
        if um:
            result["upload_manager"] = {
                "file_count": len(um.files),
                "files": [
                    {
                        "file_id": f.file_id,
                        "name": f.name,
                        "size": f.size,
                        "mime_type": f.mime_type,
                        "consumed": f.consumed,
                        "has_text_content": bool(f.text_content),
                        "text_content_length": len(f.text_content) if f.text_content else 0,
                        "has_raw_data": f.raw_data is not None,
                    }
                    for f in um.files
                ],
            }
        else:
            result["upload_manager"] = None

        # Active preset/nudge/project info
        result["active_project_id"] = getattr(ctrl, "_project_id", None)
        result["active_nudge_id"] = getattr(ctrl, "_nudge_id", None)
        result["active_nudge_uid"] = getattr(ctrl, "_nudge_uid", None)

        # File cache stats
        from llming_lodge.nudge_store import get_file_cache
        fc = get_file_cache()
        result["file_cache"] = {
            "entry_count": len(fc._entries),
            "entries": {
                uid: {
                    "file_count": len(e.files),
                    "files": [{"name": f.name, "size": f.size, "text_len": len(f.text_content)} for f in e.files],
                    "updated_at": e.updated_at,
                    "mode": e.mode,
                    "idle_seconds": round(time.monotonic() - e.last_access),
                }
                for uid, e in fc._entries.items()
            },
        }

        # Full system prompt (including any document context appended)
        result["effective_system_prompt_length"] = len(ctrl.system_prompt) if ctrl.system_prompt else 0
        result["effective_system_prompt_tail"] = ctrl.system_prompt[-500:] if ctrl.system_prompt and len(ctrl.system_prompt) > 500 else ctrl.system_prompt

        # Context preamble (silently prepended to system prompt at API call time)
        preamble = getattr(ctrl.session, "_context_preamble", None) or ""
        result["context_preamble_length"] = len(preamble)
        result["context_preamble_tail"] = preamble[-1000:] if len(preamble) > 1000 else preamble
        # Auto-discover state
        suffix = getattr(ctrl.session, "_system_prompt_suffix", None) or ""
        result["auto_discover"] = {
            "catalog": getattr(ctrl, "_auto_discover_catalog", None),
            "suffix_length": len(suffix),
            "suffix_active": "consult_nudge" in suffix,
            "tool_enabled": "consult_nudge" in ctrl.enabled_tools,
            "tool_config": ctrl.tool_config.get("consult_nudge") if ctrl.tool_config else None,
        }
        result["mcp_prompt_hints_length"] = len(getattr(ctrl, "_mcp_prompt_hints_block", ""))
        result["mcp_client_renderers"] = [
            r.get("lang", "") for r in getattr(ctrl.session, "_mcp_client_renderers", [])
        ]

        if include_history:
            messages = []
            for msg in ctrl.session.history.messages:
                m = {
                    "role": msg.role.value if hasattr(msg.role, "value") else str(msg.role),
                    "content": msg.content[:500] if msg.content else "",
                    "content_length": len(msg.content) if msg.content else 0,
                    "stale": msg.content_stale,
                    "has_images": bool(msg.images),
                    "image_count": len(msg.images) if msg.images else 0,
                }
                messages.append(m)
            result["messages"] = messages
            result["message_count"] = len(messages)

        return result

    # ── Send message ────────────────────────────────────────

    @router.post("/sessions/{session_id}/send")
    async def send_message(session_id: str, req: SendMessageRequest):
        registry = SessionRegistry.get()
        entry = registry.get_session(session_id)
        if not entry:
            raise HTTPException(404, f"Session {session_id} not found")

        ctrl = entry.controller
        if ctrl._streaming_task and not ctrl._streaming_task.done():
            raise HTTPException(409, "Already streaming")

        import asyncio

        # Tell the browser to render the user bubble (debug API bypasses JS sendMessage)
        if entry.websocket:
            await ctrl._send({
                "type": "user_message",
                "text": req.text,
                "images": req.images,
            })

        # Run message intercept hook (dev MCPs, etc.) before sending to LLM
        if req.text and not req.images and getattr(ctrl, "_on_message_intercept", None):
            try:
                intercept_result = await ctrl._on_message_intercept(req.text, ctrl)
            except Exception:
                intercept_result = None
            if intercept_result is not None:
                from llming_lodge.chat_controller import llm_manager
                model_info = llm_manager.get_model_info(ctrl.model)
                await ctrl._send({
                    "type": "response_started",
                    "model": ctrl.model,
                    "model_icon": model_info.model_icon if model_info else "",
                    "model_label": model_info.label if model_info else ctrl.model,
                })
                await ctrl._send({"type": "text_chunk", "content": intercept_result})
                await ctrl._send({"type": "response_completed"})
                await ctrl._send({
                    "type": "tools_updated",
                    "tools": ctrl.get_all_known_tools(),
                })
                return {
                    "status": "intercepted",
                    "session_id": session_id,
                    "text": req.text,
                    "response": intercept_result,
                }

        # Fire and forget — the response streams via WS to the browser
        task = asyncio.create_task(ctrl.send_message(req.text, images=req.images))

        return {
            "status": "sent",
            "session_id": session_id,
            "text": req.text,
            "note": "Response streams via WebSocket to the connected browser",
        }

    # ── Check status ────────────────────────────────────────

    @router.get("/sessions/{session_id}/status")
    async def get_status(session_id: str):
        registry = SessionRegistry.get()
        entry = registry.get_session(session_id)
        if not entry:
            raise HTTPException(404, f"Session {session_id} not found")

        ctrl = entry.controller
        streaming = bool(ctrl._streaming_task and not ctrl._streaming_task.done())

        last_msg = None
        if ctrl.session.history.messages:
            m = ctrl.session.history.messages[-1]
            last_msg = {
                "role": m.role.value if hasattr(m.role, "value") else str(m.role),
                "content_preview": m.content[:200] if m.content else "",
                "content_length": len(m.content) if m.content else 0,
            }

        return {
            "session_id": session_id,
            "streaming": streaming,
            "partial_text": ctrl._text_content[:500] if streaming else None,
            "partial_text_length": len(ctrl._text_content) if streaming else None,
            "tool_calls_active": [
                {"name": tc.name, "status": tc.status.value if hasattr(tc.status, "value") else str(tc.status)}
                for tc in ctrl._tool_calls
            ] if streaming else [],
            "message_count": len(ctrl.session.history.messages),
            "last_message": last_msg,
            "ws_connected": entry.websocket is not None,
        }

    # ── Switch model ────────────────────────────────────────

    @router.post("/sessions/{session_id}/model")
    async def switch_model(session_id: str, req: SwitchModelRequest):
        registry = SessionRegistry.get()
        entry = registry.get_session(session_id)
        if not entry:
            raise HTTPException(404, f"Session {session_id} not found")

        ctrl = entry.controller
        old_model = ctrl.model
        await ctrl.switch_model(req.model)

        return {
            "session_id": session_id,
            "old_model": old_model,
            "new_model": ctrl.model,
        }

    # ── List available models ───────────────────────────────

    @router.get("/models")
    async def list_models():
        from llming_lodge.chat_controller import llm_manager
        models = []
        for info in llm_manager.get_available_llms():
            models.append({
                "model": info.model,
                "label": info.label,
                "provider": llm_manager.get_provider_for_model(info.model),
                "icon": info.model_icon,
            })
        return {"models": models}

    @router.get("/models/cascade")
    async def list_models_cascade():
        """Show full cascade resolution for every model (debug/admin)."""
        from llming_lodge.chat_controller import llm_manager
        return {
            "cascade_order": llm_manager.provider_cascade,
            "active_providers": [p for p in llm_manager.provider_cascade if p in llm_manager.providers],
            "models": llm_manager.get_cascade_debug_info(),
        }

    # ── New chat ────────────────────────────────────────────

    @router.post("/sessions/{session_id}/new_chat")
    async def new_chat(session_id: str, request: Request):
        """Start a new chat, optionally with a preset (project or nudge).

        Body (optional): { "preset": { ... } }
        For nudges: { "preset": { "type": "nudge", "nudge_uid": "..." } }
        For projects: { "preset": { "type": "project", "id": "..." } }
          Projects with only an 'id' will auto-fetch full data (incl. files) from browser IDB.
        """
        from .chat_session_api import _handle_client_message
        registry = SessionRegistry.get()
        entry = registry.get_session(session_id)
        if not entry:
            raise HTTPException(404, f"Session {session_id} not found")

        msg: dict = {"type": "new_chat"}
        try:
            body = await request.json()
            if body.get("preset"):
                msg["preset"] = body["preset"]
        except Exception:
            pass  # No body or invalid JSON = plain new chat

        # For project presets with only an ID, fetch full data from browser IDB
        preset = msg.get("preset")
        if preset and preset.get("type") == "project" and preset.get("id") and not preset.get("files"):
            if entry.websocket:
                import asyncio
                ctrl = entry.controller
                future = asyncio.get_event_loop().create_future()
                ctrl._pending_preset_detail = future
                await ctrl._send({
                    "type": "ui_action",
                    "action": "get_preset",
                    "preset_id": preset["id"],
                })
                try:
                    proj_data = await asyncio.wait_for(future, timeout=5.0)
                    if proj_data:
                        # Build preset from IDB data (same as browser's _handleSubmit)
                        proj_files = []
                        for f in (proj_data.get("files") or []):
                            if f.get("text_content"):
                                pf = {k: v for k, v in f.items() if k != "content"}
                            else:
                                pf = f
                            proj_files.append(pf)
                        msg["preset"] = {
                            "id": proj_data["id"],
                            "type": "project",
                            "system_prompt": proj_data.get("system_prompt", ""),
                            "model": proj_data.get("model"),
                            "language": proj_data.get("language", "auto"),
                            "files": proj_files,
                            "doc_plugins": proj_data.get("doc_plugins"),
                        }
                except asyncio.TimeoutError:
                    pass
                finally:
                    ctrl._pending_preset_detail = None

        await _handle_client_message(entry.controller, entry, msg)

        return {
            "status": "cleared",
            "new_session_id": entry.controller.session_id,
            "preset": msg.get("preset", {}).get("type"),
        }

    # ── UI Navigation (browser-side via WS) ─────────────────

    def _ui_action_helper(session_id: str):
        """Resolve session + check WS, return (entry, controller)."""
        registry = SessionRegistry.get()
        entry = registry.get_session(session_id)
        if not entry:
            raise HTTPException(404, f"Session {session_id} not found")
        if not entry.websocket:
            raise HTTPException(409, "No WebSocket connected")
        return entry, entry.controller

    @router.post("/sessions/{session_id}/scroll")
    async def scroll_chat(session_id: str, target: str = "bottom", index: int = -1):
        """Scroll chat view. target='bottom' or target='message' with index.

        index: 0-based message index, or negative (-1 = last message).
        """
        entry, ctrl = _ui_action_helper(session_id)

        import asyncio
        future = asyncio.get_event_loop().create_future()
        ctrl._pending_scroll_ack = future

        if target == "message":
            await ctrl._send({"type": "ui_action", "action": "scroll_to_message", "index": index})
        else:
            await ctrl._send({"type": "ui_action", "action": "scroll_to_bottom"})

        try:
            result = await asyncio.wait_for(future, timeout=5.0)
            return result
        except asyncio.TimeoutError:
            return {"ok": False, "note": "Timeout"}
        finally:
            ctrl._pending_scroll_ack = None

    @router.post("/sessions/{session_id}/open_droplet")
    async def open_droplet(session_id: str, nudge_uid: str = "", nudge_id: str = ""):
        """Open a new chat with a droplet/nudge. Uses the browser's _startNudgeChat flow.

        nudge_uid: MongoDB nudge UID (preferred for server-side nudges)
        nudge_id:  IDB nudge ID (legacy, for browser-side nudges)
        """
        if not nudge_uid and not nudge_id:
            raise HTTPException(400, "Provide nudge_uid or nudge_id")
        entry, ctrl = _ui_action_helper(session_id)

        import asyncio
        future = asyncio.get_event_loop().create_future()
        ctrl._pending_open_droplet_ack = future

        msg = {"type": "ui_action", "action": "open_droplet"}
        if nudge_uid:
            msg["nudge_uid"] = nudge_uid
        else:
            msg["nudge_id"] = nudge_id
        await ctrl._send(msg)

        try:
            result = await asyncio.wait_for(future, timeout=10.0)
            # After the browser triggers new_chat, the controller session_id changes
            result["new_session_id"] = ctrl.session_id
            return result
        except asyncio.TimeoutError:
            return {"ok": False, "note": "Timeout"}
        finally:
            ctrl._pending_open_droplet_ack = None

    @router.post("/sessions/{session_id}/open_project")
    async def open_project(session_id: str, project_id: str):
        """Open a project view in the browser (like clicking a project in the sidebar)."""
        entry, ctrl = _ui_action_helper(session_id)

        import asyncio
        future = asyncio.get_event_loop().create_future()
        ctrl._pending_open_project_ack = future

        await ctrl._send({"type": "ui_action", "action": "open_project", "project_id": project_id})

        try:
            result = await asyncio.wait_for(future, timeout=5.0)
            return result
        except asyncio.TimeoutError:
            return {"ok": False, "note": "Timeout"}
        finally:
            ctrl._pending_open_project_ack = None

    @router.post("/sessions/{session_id}/open_conversation")
    async def open_conversation(session_id: str, conversation_id: str):
        """Switch to a specific conversation (like clicking it in the sidebar)."""
        entry, ctrl = _ui_action_helper(session_id)

        import asyncio
        future = asyncio.get_event_loop().create_future()
        ctrl._pending_open_conversation_ack = future

        await ctrl._send({"type": "ui_action", "action": "open_conversation", "conversation_id": conversation_id})

        try:
            result = await asyncio.wait_for(future, timeout=10.0)
            return result
        except asyncio.TimeoutError:
            return {"ok": False, "note": "Timeout"}
        finally:
            ctrl._pending_open_conversation_ack = None

    # ── Toggle tool ─────────────────────────────────────────

    @router.post("/sessions/{session_id}/toggle_tool")
    async def toggle_tool(session_id: str, name: str, enabled: bool = True):
        registry = SessionRegistry.get()
        entry = registry.get_session(session_id)
        if not entry:
            raise HTTPException(404, f"Session {session_id} not found")

        ctrl = entry.controller
        ctrl.toggle_tool(name, enabled)

        return {
            "session_id": session_id,
            "tool": name,
            "enabled": enabled,
            "enabled_tools": ctrl.enabled_tools,
        }

    # ── File management ─────────────────────────────────────

    @router.post("/sessions/{session_id}/paste_image")
    async def paste_image(session_id: str, request: Request):
        """Paste image(s) into a session. Body: {images: [base64_string, ...]}"""
        from llming_lodge.server import SessionDataStore

        registry = SessionRegistry.get()
        entry = registry.get_session(session_id)
        if not entry:
            raise HTTPException(404, f"Session {session_id} not found")

        data = await request.json()
        images = data.get("images", [])
        SessionDataStore.set_pasted_images(session_id, images)
        return {"ok": True, "count": len(images)}

    @router.post("/sessions/{session_id}/attach_file")
    async def attach_file(session_id: str, request: Request):
        """Attach a file to a session via debug API.

        Body: {filename: str, content_base64: str, mime_type?: str}
        Content is base64-encoded file bytes.
        """
        import base64 as _b64

        registry = SessionRegistry.get()
        entry = registry.get_session(session_id)
        if not entry:
            raise HTTPException(404, f"Session {session_id} not found")
        if not entry.upload_manager:
            raise HTTPException(500, "No upload manager for session")

        data = await request.json()
        filename = data.get("filename", "unknown.txt")
        content_b64 = data.get("content_base64", "")
        if not content_b64:
            raise HTTPException(400, "content_base64 required")

        content = _b64.b64decode(content_b64)
        att = await entry.upload_manager.store_file(filename, content, entry.user_id)

        # Sync document context
        entry.controller.sync_document_context(entry.upload_manager)

        # Notify browser via WS
        if entry.websocket:
            await entry.controller._send({
                "type": "files_updated",
                "files": [{
                    "fileId": f.file_id,
                    "name": f.name,
                    "size": f.size,
                    "mimeType": f.mime_type,
                } for f in entry.upload_manager.files],
            })

        return {
            "ok": True,
            "file": {
                "fileId": att.file_id,
                "name": att.name,
                "size": att.size,
                "mimeType": att.mime_type,
            },
            "total_files": len(entry.upload_manager.files),
        }

    @router.get("/sessions/{session_id}/files")
    async def list_files(session_id: str):
        """List files attached to a session."""
        registry = SessionRegistry.get()
        entry = registry.get_session(session_id)
        if not entry:
            raise HTTPException(404, f"Session {session_id} not found")

        files = []
        if entry.upload_manager:
            for f in entry.upload_manager.files:
                files.append({
                    "fileId": f.file_id,
                    "name": f.name,
                    "size": f.size,
                    "mimeType": f.mime_type,
                    "hasText": bool(f.text_content),
                })
        return {"files": files}

    @router.delete("/sessions/{session_id}/files/{file_id}")
    async def remove_file(session_id: str, file_id: str):
        """Remove a file from a session and rebuild document context."""
        registry = SessionRegistry.get()
        entry = registry.get_session(session_id)
        if not entry:
            raise HTTPException(404, f"Session {session_id} not found")

        if entry.upload_manager:
            entry.upload_manager.remove_file(file_id, entry.user_id)

        # Sync document context
        entry.controller.sync_document_context(entry.upload_manager)

        return {"ok": True, "file_id": file_id}

    @router.post("/sessions/{session_id}/notify_file_uploaded")
    async def notify_file_uploaded(session_id: str):
        """Trigger the file_uploaded flow (sync doc context + PDF viewer MCP).

        Call this after uploading a file via the HTTP upload endpoint to
        rebuild the document context and register PDF tools — same as when
        the browser sends a 'file_uploaded' WS message.
        """
        registry = SessionRegistry.get()
        entry = registry.get_session(session_id)
        if not entry:
            raise HTTPException(404, f"Session {session_id} not found")

        ctrl = entry.controller
        ctrl.sync_document_context(entry.upload_manager)
        await _sync_pdf_viewer_mcp(ctrl, entry)
        await ctrl._send_context_info()
        return {"ok": True}

    @router.post("/sessions/{session_id}/inject_file")
    async def inject_file(session_id: str, request: Request):
        """Inject a file through the browser's normal upload path.

        This goes through the exact same code path as a user clicking
        "attach" — the browser receives the file data, creates a File
        object, and calls _handleFileSelection().

        Body: {path: str} — absolute path to a local file.
        """
        import asyncio
        import base64 as _b64
        import mimetypes
        from pathlib import Path

        registry = SessionRegistry.get()
        entry = registry.get_session(session_id)
        if not entry:
            raise HTTPException(404, f"Session {session_id} not found")
        if not entry.websocket:
            raise HTTPException(409, "No WebSocket connected")

        data = await request.json()
        file_path = data.get("path", "")
        if not file_path:
            raise HTTPException(400, "path required")

        p = Path(file_path).resolve()
        if not p.is_file():
            raise HTTPException(400, f"File not found: {file_path}")
        # Safety: only allow files under user's home directory
        home = Path.home()
        if not str(p).startswith(str(home)):
            raise HTTPException(403, "Path must be under user home directory")

        content = p.read_bytes()
        mime_type = mimetypes.guess_type(p.name)[0] or "application/octet-stream"

        ctrl = entry.controller
        future = asyncio.get_event_loop().create_future()
        ctrl._pending_inject_file_ack = future

        await ctrl._send({
            "type": "ui_action",
            "action": "inject_file",
            "filename": p.name,
            "mime_type": mime_type,
            "data_base64": _b64.b64encode(content).decode(),
        })

        try:
            result = await asyncio.wait_for(future, timeout=15.0)
            return {"ok": result.get("ok", False), "filename": p.name, "size": len(content), "mime_type": mime_type}
        except asyncio.TimeoutError:
            return {"ok": False, "note": "Timeout waiting for browser ack"}
        finally:
            ctrl._pending_inject_file_ack = None

    @router.post("/sessions/{session_id}/remove_file_ui")
    async def remove_file_ui(session_id: str, request: Request):
        """Remove a file through the browser UI (same as clicking X).

        Body: {file_id: str}
        """
        import asyncio

        registry = SessionRegistry.get()
        entry = registry.get_session(session_id)
        if not entry:
            raise HTTPException(404, f"Session {session_id} not found")
        if not entry.websocket:
            raise HTTPException(409, "No WebSocket connected")

        data = await request.json()
        file_id = data.get("file_id", "")
        if not file_id:
            raise HTTPException(400, "file_id required")

        ctrl = entry.controller
        future = asyncio.get_event_loop().create_future()
        ctrl._pending_remove_file_ack = future

        await ctrl._send({
            "type": "ui_action",
            "action": "remove_file",
            "file_id": file_id,
        })

        try:
            result = await asyncio.wait_for(future, timeout=5.0)
            return {"ok": result.get("ok", False), "file_id": file_id}
        except asyncio.TimeoutError:
            return {"ok": False, "note": "Timeout waiting for browser ack"}
        finally:
            ctrl._pending_remove_file_ack = None

    @router.get("/sessions/{session_id}/context")
    async def get_context_info(session_id: str):
        """Get computed context size breakdown for a session."""
        registry = SessionRegistry.get()
        entry = registry.get_session(session_id)
        if not entry:
            raise HTTPException(404, f"Session {session_id} not found")

        ctrl = entry.controller
        info = ctrl._compute_context_info()
        if not info:
            return {"error": "Unable to compute context info"}

        # Add file details
        files = []
        if entry.upload_manager:
            for f in entry.upload_manager.files:
                files.append({
                    "fileId": f.file_id,
                    "name": f.name,
                    "size": f.size,
                    "mimeType": f.mime_type,
                    "textLength": len(f.text_content) if f.text_content else 0,
                    "estTokens": len(f.text_content) // 4 if f.text_content else 0,
                })

        return {**info, "files": files}

    # ── Conversations (IDB via WS) ─────────────────────────

    @router.get("/sessions/{session_id}/conversations")
    async def list_conversations(session_id: str):
        """List conversations from the browser's IndexedDB.

        Sends a ui_action to the browser requesting the conversation list,
        which the browser returns via WS. Since this is async, we instead
        expose the conversations the server has seen from save_conversation
        events and from the IDB store.
        """
        registry = SessionRegistry.get()
        entry = registry.get_session(session_id)
        if not entry:
            raise HTTPException(404, f"Session {session_id} not found")

        if not entry.websocket:
            raise HTTPException(409, "No WebSocket connected")

        # Ask the browser to send conversation list via WS
        import asyncio
        ctrl = entry.controller

        # Use a future to receive the response
        conv_future = asyncio.get_event_loop().create_future()
        ctrl._pending_conv_list = conv_future

        await ctrl._send({"type": "ui_action", "action": "list_conversations"})

        try:
            result = await asyncio.wait_for(conv_future, timeout=5.0)
            return {"conversations": result}
        except asyncio.TimeoutError:
            return {"conversations": [], "note": "Timeout waiting for browser response"}
        finally:
            ctrl._pending_conv_list = None

    @router.post("/sessions/{session_id}/load_conversation")
    async def load_conversation_api(session_id: str, request: Request):
        """Load a conversation by ID via browser UI action.

        Body: {conversation_id: str}
        This sends a ui_action to the browser to select the conversation.
        """
        registry = SessionRegistry.get()
        entry = registry.get_session(session_id)
        if not entry:
            raise HTTPException(404, f"Session {session_id} not found")

        if not entry.websocket:
            raise HTTPException(409, "No WebSocket connected")

        data = await request.json()
        conv_id = data.get("conversation_id", "")
        if not conv_id:
            raise HTTPException(400, "conversation_id required")

        await entry.controller._send({
            "type": "ui_action",
            "action": "load_conversation",
            "conversation_id": conv_id,
        })

        return {"ok": True, "conversation_id": conv_id}

    # ── UI toggle helpers ────────────────────────────────────

    @router.post("/sessions/{session_id}/ui_action")
    async def ui_action(session_id: str, request: Request):
        """Send a UI action to the browser via WebSocket.

        Body: {action: "toggle_sidebar" | "toggle_settings" | "open_model_menu" |
               "close_dropdowns" | "show_context_info" | "trigger_quick_action" |
               "load_conversation" | "open_lightbox" | "list_conversations",
               ...params}

        These are forwarded as WS messages of type 'ui_action' to the browser.
        The frontend JS handles them and toggles the relevant UI elements.
        """
        registry = SessionRegistry.get()
        entry = registry.get_session(session_id)
        if not entry:
            raise HTTPException(404, f"Session {session_id} not found")

        data = await request.json()
        action = data.get("action", "")
        ctrl = entry.controller

        if not entry.websocket:
            raise HTTPException(409, "No WebSocket connected")

        # Forward to browser
        await ctrl._send({"type": "ui_action", **data})

        return {"ok": True, "action": action}

    # ── Inject content (test rendering) ─────────────────────

    @router.post("/sessions/{session_id}/inject")
    async def inject_content(session_id: str, request: Request):
        """Inject content directly into the chat UI as an assistant response.

        Bypasses the LLM entirely — useful for testing renderers, doc plugins,
        and markdown output.

        Body: {text: str, role?: "assistant"}
        The text is sent to the browser via the normal response_started →
        text_chunk → response_completed WS flow.
        """
        registry = SessionRegistry.get()
        entry = registry.get_session(session_id)
        if not entry:
            raise HTTPException(404, f"Session {session_id} not found")

        if not entry.websocket:
            raise HTTPException(409, "No WebSocket connected")

        data = await request.json()
        text = data.get("text", "")
        if not text:
            raise HTTPException(400, "text required")

        ctrl = entry.controller

        # 1. response_started
        await ctrl._send({
            "type": "response_started",
            "model": ctrl.model,
            "model_icon": "",
            "provider": ctrl.provider or "",
        })

        # 2. text_chunk (full text at once)
        await ctrl._send({
            "type": "text_chunk",
            "content": text,
        })

        # 3. response_completed
        await ctrl._send({
            "type": "response_completed",
            "full_text": text,
            "generated_image": None,
            "tool_calls": [],
        })

        return {
            "ok": True,
            "session_id": session_id,
            "text_length": len(text),
            "note": "Content injected as assistant response",
        }

    # ── Presets (Nudges & Projects) ──────────────────────

    @router.get("/sessions/{session_id}/presets")
    async def list_presets(session_id: str, type: str = ""):
        """List presets (nudges/projects) from the browser's IndexedDB.

        Query params:
          type — 'nudge', 'project', or '' for all
        """
        registry = SessionRegistry.get()
        entry = registry.get_session(session_id)
        if not entry:
            raise HTTPException(404, f"Session {session_id} not found")
        if not entry.websocket:
            raise HTTPException(409, "No WebSocket connected")

        import asyncio
        ctrl = entry.controller
        future = asyncio.get_event_loop().create_future()
        ctrl._pending_preset_list = future

        await ctrl._send({
            "type": "ui_action",
            "action": "list_presets",
            "preset_type": type or None,
        })

        try:
            result = await asyncio.wait_for(future, timeout=5.0)
            return {"presets": result}
        except asyncio.TimeoutError:
            return {"presets": [], "note": "Timeout waiting for browser response"}
        finally:
            ctrl._pending_preset_list = None

    @router.get("/sessions/{session_id}/presets/{preset_id}")
    async def get_preset(session_id: str, preset_id: str):
        """Get full preset data (including files with text_content) from IDB."""
        registry = SessionRegistry.get()
        entry = registry.get_session(session_id)
        if not entry:
            raise HTTPException(404, f"Session {session_id} not found")
        if not entry.websocket:
            raise HTTPException(409, "No WebSocket connected")

        import asyncio
        ctrl = entry.controller
        future = asyncio.get_event_loop().create_future()
        ctrl._pending_preset_detail = future

        await ctrl._send({
            "type": "ui_action",
            "action": "get_preset",
            "preset_id": preset_id,
        })

        try:
            result = await asyncio.wait_for(future, timeout=5.0)
            if result is None:
                raise HTTPException(404, f"Preset {preset_id} not found in browser IDB")
            return {"preset": result}
        except asyncio.TimeoutError:
            return {"preset": None, "note": "Timeout waiting for browser response"}
        finally:
            ctrl._pending_preset_detail = None

    @router.put("/sessions/{session_id}/presets")
    async def save_preset(session_id: str, request: Request):
        """Save/update a preset in the browser's IDB.

        Body: full preset object (must include id, type, name, etc.)
        """
        registry = SessionRegistry.get()
        entry = registry.get_session(session_id)
        if not entry:
            raise HTTPException(404, f"Session {session_id} not found")
        if not entry.websocket:
            raise HTTPException(409, "No WebSocket connected")

        data = await request.json()
        if not data.get("id"):
            raise HTTPException(400, "Preset must include 'id'")

        import asyncio
        ctrl = entry.controller
        future = asyncio.get_event_loop().create_future()
        ctrl._pending_preset_save = future

        await ctrl._send({
            "type": "ui_action",
            "action": "save_preset",
            "data": data,
        })

        try:
            ok = await asyncio.wait_for(future, timeout=5.0)
            return {"ok": ok, "preset_id": data["id"]}
        except asyncio.TimeoutError:
            return {"ok": False, "note": "Timeout waiting for browser response"}
        finally:
            ctrl._pending_preset_save = None

    @router.delete("/sessions/{session_id}/presets/{preset_id}")
    async def delete_preset(session_id: str, preset_id: str):
        """Delete a preset from the browser's IDB."""
        registry = SessionRegistry.get()
        entry = registry.get_session(session_id)
        if not entry:
            raise HTTPException(404, f"Session {session_id} not found")
        if not entry.websocket:
            raise HTTPException(409, "No WebSocket connected")

        import asyncio
        ctrl = entry.controller
        future = asyncio.get_event_loop().create_future()
        ctrl._pending_preset_delete = future

        await ctrl._send({
            "type": "ui_action",
            "action": "delete_preset",
            "preset_id": preset_id,
        })

        try:
            ok = await asyncio.wait_for(future, timeout=5.0)
            return {"ok": ok, "preset_id": preset_id}
        except asyncio.TimeoutError:
            return {"ok": False, "note": "Timeout waiting for browser response"}
        finally:
            ctrl._pending_preset_delete = None

    # ── IDB File Store (content-addressable files) ───────

    @router.get("/sessions/{session_id}/idb_files")
    async def list_idb_files(session_id: str):
        """List all files in the browser's IDB file store (without data blobs)."""
        registry = SessionRegistry.get()
        entry = registry.get_session(session_id)
        if not entry:
            raise HTTPException(404, f"Session {session_id} not found")
        if not entry.websocket:
            raise HTTPException(409, "No WebSocket connected")

        import asyncio
        ctrl = entry.controller
        future = asyncio.get_event_loop().create_future()
        ctrl._pending_idb_file_list = future

        await ctrl._send({"type": "ui_action", "action": "list_idb_files"})

        try:
            result = await asyncio.wait_for(future, timeout=5.0)
            return {"files": result}
        except asyncio.TimeoutError:
            return {"files": [], "note": "Timeout waiting for browser response"}
        finally:
            ctrl._pending_idb_file_list = None

    @router.get("/sessions/{session_id}/idb_files/{hash}")
    async def get_idb_file(session_id: str, hash: str):
        """Get single file metadata (no blob) from IDB file store."""
        registry = SessionRegistry.get()
        entry = registry.get_session(session_id)
        if not entry:
            raise HTTPException(404, f"Session {session_id} not found")
        if not entry.websocket:
            raise HTTPException(409, "No WebSocket connected")

        import asyncio
        ctrl = entry.controller
        future = asyncio.get_event_loop().create_future()
        ctrl._pending_idb_file_detail = future

        await ctrl._send({"type": "ui_action", "action": "get_idb_file", "hash": hash})

        try:
            result = await asyncio.wait_for(future, timeout=5.0)
            if result is None:
                raise HTTPException(404, f"File {hash} not found in browser IDB")
            return {"file": result}
        except asyncio.TimeoutError:
            return {"file": None, "note": "Timeout waiting for browser response"}
        finally:
            ctrl._pending_idb_file_detail = None

    @router.get("/sessions/{session_id}/idb_file_refs/{conversation_id}")
    async def get_idb_file_refs(session_id: str, conversation_id: str):
        """Get file refs saved with a conversation in IDB."""
        registry = SessionRegistry.get()
        entry = registry.get_session(session_id)
        if not entry:
            raise HTTPException(404, f"Session {session_id} not found")
        if not entry.websocket:
            raise HTTPException(409, "No WebSocket connected")

        import asyncio
        ctrl = entry.controller
        future = asyncio.get_event_loop().create_future()
        ctrl._pending_idb_file_refs = future

        await ctrl._send({
            "type": "ui_action",
            "action": "get_idb_file_refs",
            "conversation_id": conversation_id,
        })

        try:
            result = await asyncio.wait_for(future, timeout=5.0)
            return {"file_refs": result}
        except asyncio.TimeoutError:
            return {"file_refs": [], "note": "Timeout waiting for browser response"}
        finally:
            ctrl._pending_idb_file_refs = None

    # ── Nudges (MongoDB — no session needed) ─────────────

    def _get_nudge_store():
        """Return the nudge store passed at router build time, or find one from a session."""
        if nudge_store:
            return nudge_store
        # Fallback: try to find from any active session
        registry = SessionRegistry.get()
        for _sid, entry in registry._sessions.items():
            store = getattr(entry.controller, "_nudge_store", None)
            if store:
                return store
        return None

    @router.get("/nudges")
    async def list_nudges(
        query: str = "",
        category: str = "",
        mode: str = "",
        page: int = 0,
        page_size: int = 50,
    ):
        """List nudges from MongoDB. No session needed."""
        store = _get_nudge_store()
        if not store:
            raise HTTPException(503, "Nudge store not available (no nudge_mongo_uri configured)")

        await store._ensure_indexes()
        coll, _ = store._ensure_colls()

        filt: dict = {}
        if mode:
            filt["mode"] = mode
        if category:
            filt["category"] = category
        if query:
            filt["$or"] = [
                {"name": {"$regex": query, "$options": "i"}},
                {"description": {"$regex": query, "$options": "i"}},
                {"creator_email": {"$regex": query, "$options": "i"}},
            ]

        cursor = (
            coll.find(filt, {"files": 0})
            .sort("updated_at", -1)
            .skip(page * page_size)
            .limit(page_size + 1)
        )
        results = []
        async for doc in cursor:
            doc.pop("_id", None)
            results.append(doc)

        has_more = len(results) > page_size
        return {
            "nudges": results[:page_size],
            "page": page,
            "page_size": page_size,
            "has_more": has_more,
        }

    @router.get("/nudges/{uid}")
    async def get_nudge(uid: str, mode: str = "dev"):
        """Get full nudge document by uid (including files)."""
        store = _get_nudge_store()
        if not store:
            raise HTTPException(503, "Nudge store not available")

        await store._ensure_indexes()
        coll, _ = store._ensure_colls()

        doc = await coll.find_one({"uid": uid, "mode": mode})
        if not doc:
            raise HTTPException(404, f"Nudge {uid} (mode={mode}) not found")
        doc.pop("_id", None)
        return {"nudge": doc}

    @router.put("/nudges/{uid}")
    async def update_nudge(uid: str, req: NudgeUpdateRequest, mode: str = "dev"):
        """Create or update a nudge. Creates with defaults if not found."""
        store = _get_nudge_store()
        if not store:
            raise HTTPException(503, "Nudge store not available")

        await store._ensure_indexes()
        coll, _ = store._ensure_colls()

        existing = await coll.find_one({"uid": uid, "mode": mode})

        # Merge only provided (non-None) fields
        updates = {k: v for k, v in req.model_dump().items() if v is not None}
        if not updates:
            raise HTTPException(400, "No fields to update")

        # Strip text_content from files before persisting (extracted on load)
        if "files" in updates:
            for f in updates["files"]:
                if isinstance(f, dict):
                    f.pop("text_content", None)

        if existing:
            await coll.update_one(
                {"uid": uid, "mode": mode},
                {"$set": updates},
            )
        else:
            # Create new nudge with defaults
            from datetime import datetime, timezone
            now = datetime.now(timezone.utc).isoformat()
            doc = {
                "uid": uid,
                "mode": mode,
                "type": "nudge",
                "name": updates.get("name", f"Nudge {uid[:8]}"),
                "description": "",
                "icon": None,
                "system_prompt": "",
                "model": None,
                "language": "auto",
                "creator_email": "",
                "creator_name": "",
                "category": "",
                "sub_category": "",
                "visibility": ["*"],  # visible to all by default
                "suggestions": "",
                "capabilities": {},
                "files": [],
                "created_at": now,
                "updated_at": now,
            }
            doc.update(updates)
            await coll.insert_one(doc)

        store._cache_evict(uid)
        if "files" in updates:
            from llming_lodge.nudge_store import get_file_cache
            get_file_cache().evict(uid)

        # Return updated doc (exclude heavy files from response)
        updated = await coll.find_one({"uid": uid, "mode": mode}, {"files": 0})
        updated.pop("_id", None)
        created = not existing
        return {"nudge": updated, "created": created, "updated_fields": list(updates.keys())}

    @router.put("/nudges/{uid}/files")
    async def update_nudge_files(uid: str, request: Request, mode: str = "dev"):
        """Replace the files array of a nudge.

        Body: {files: [{name, mime_type, size, content}, ...]}
        ``content`` is a base64-encoded data URL or raw base64 string.
        ``text_content`` is stripped before persisting (extracted on load).
        """
        store = _get_nudge_store()
        if not store:
            raise HTTPException(503, "Nudge store not available")

        await store._ensure_indexes()
        coll, _ = store._ensure_colls()

        existing = await coll.find_one({"uid": uid, "mode": mode})
        if not existing:
            raise HTTPException(404, f"Nudge {uid} (mode={mode}) not found")

        data = await request.json()
        files = data.get("files")
        if files is None:
            raise HTTPException(400, "files array required")

        # Strip text_content (extracted on-the-fly at load time)
        for f in files:
            if isinstance(f, dict):
                f.pop("text_content", None)

        from datetime import datetime, timezone
        await coll.update_one(
            {"uid": uid, "mode": mode},
            {"$set": {"files": files, "updated_at": datetime.now(timezone.utc).isoformat()}},
        )
        store._cache_evict(uid)
        from llming_lodge.nudge_store import get_file_cache
        get_file_cache().evict(uid)

        return {
            "ok": True,
            "uid": uid,
            "mode": mode,
            "file_count": len(files),
            "files": [{"name": f.get("name"), "size": f.get("size", 0)} for f in files if isinstance(f, dict)],
        }

    # ── AI document editing ────────────────────────────────

    @router.post("/sessions/{session_id}/open_doc_window")
    async def open_doc_window(session_id: str, request: Request):
        """Send a UI action to open a document in window mode.

        Body: { "document_id": "..." }
        """
        registry = SessionRegistry.get()
        entry = registry.get_session(session_id)
        if not entry:
            raise HTTPException(404, f"Session {session_id} not found")

        data = await request.json()
        document_id = data.get("document_id", "")

        ctrl = entry.controller
        await ctrl._send({
            "type": "ui_action",
            "action": "open_doc_window",
            "document_id": document_id,
        })
        return {"status": "sent", "document_id": document_id}

    @router.post("/sessions/{session_id}/edit_region")
    async def edit_region(session_id: str, request: Request):
        """Send an AI edit request via WS and wait for the result.

        Body: { "document_id": "...", "selected_text": "...", "action": "fix_grammar", "custom_prompt": "..." }
        """
        import asyncio

        registry = SessionRegistry.get()
        entry = registry.get_session(session_id)
        if not entry:
            raise HTTPException(404, f"Session {session_id} not found")

        data = await request.json()
        request_id = f"debug_{int(time.time() * 1000)}"

        ctrl = entry.controller

        # Create a future to wait for the result
        result_future: asyncio.Future = asyncio.get_event_loop().create_future()
        original_send = ctrl._send

        async def _intercept_send(msg):
            await original_send(msg)
            if isinstance(msg, dict) and msg.get("type") == "ai_edit_result" and msg.get("request_id") == request_id:
                if not result_future.done():
                    result_future.set_result(msg)

        ctrl._send = _intercept_send

        try:
            from llming_lodge.api.chat_session_api import _handle_client_message

            await _handle_client_message(ctrl, entry, {
                "type": "ai_edit_request",
                "request_id": request_id,
                "document_id": data.get("document_id", ""),
                "document_type": data.get("document_type", "text_doc"),
                "selected_text": data.get("selected_text", ""),
                "full_context": data.get("full_context", ""),
                "action": data.get("action", "fix_grammar"),
                "custom_prompt": data.get("custom_prompt"),
                "language": data.get("language", "en"),
            })

            result = await asyncio.wait_for(result_future, timeout=35)
            return result
        except asyncio.TimeoutError:
            raise HTTPException(504, "AI edit timed out")
        finally:
            ctrl._send = original_send

    # ── Unified Document Command ───────────────────────────────

    @router.post("/sessions/{session_id}/doc")
    async def doc_command(session_id: str, request: Request):
        """Execute a document command.

        Browser-side commands (via WS round-trip):
          list_documents, open_windowed, close_window, maximize, restore,
          get_state, get_content, select_text, get_selection, set_cursor,
          type_text, scroll_doc

        Server-side AI commands (direct handler call):
          ai_edit, ai_task, ai_typeahead
        """
        import asyncio
        import uuid

        entry, ctrl = _ui_action_helper(session_id)
        body = await request.json()
        command = body.get("command", "")

        if not command:
            raise HTTPException(400, "Missing 'command' field")

        # --- Server-side AI commands ---
        if command in ("ai_edit", "ai_task", "ai_typeahead"):
            return await _handle_ai_doc_command(ctrl, command, body)

        # --- Browser-side commands via WS ---
        request_id = f"dbg_{uuid.uuid4().hex[:8]}"

        if not hasattr(ctrl, '_pending_doc_cmds'):
            ctrl._pending_doc_cmds = {}

        future = asyncio.get_event_loop().create_future()
        ctrl._pending_doc_cmds[request_id] = future

        ws_msg = {**body, "type": "doc_command", "request_id": request_id}
        await ctrl._send(ws_msg)

        timeout = 15.0 if command == "open_windowed" else 5.0
        try:
            result = await asyncio.wait_for(future, timeout=timeout)
            result.pop("type", None)
            result.pop("request_id", None)
            return result
        except asyncio.TimeoutError:
            return {"ok": False, "error": "Timeout waiting for browser response"}
        finally:
            ctrl._pending_doc_cmds.pop(request_id, None)

    async def _handle_ai_doc_command(ctrl, command: str, body: dict):
        """Handle AI edit/task/typeahead by calling the handler directly."""
        import asyncio

        from llming_lodge.api.ai_edit_handler import AIEditHandler

        if not hasattr(ctrl, '_ai_edit_handler'):
            ctrl._ai_edit_handler = AIEditHandler(ctrl)
        handler = ctrl._ai_edit_handler

        result_future = asyncio.get_event_loop().create_future()
        original_send = ctrl._send

        if command == "ai_edit":
            expect_type = "ai_edit_result"
        elif command == "ai_task":
            expect_type = "ai_task_result"
        else:
            expect_type = "ai_typeahead_suggestion"

        request_id = body.get("request_id", f"dbg_{command}_{int(time.time() * 1000)}")

        async def _intercept_send(msg):
            await original_send(msg)
            if isinstance(msg, dict) and msg.get("type") == expect_type and msg.get("request_id") == request_id:
                if not result_future.done():
                    result_future.set_result(msg)

        ctrl._send = _intercept_send

        try:
            if command == "ai_edit":
                await handler.handle_edit_request({
                    "request_id": request_id,
                    "action": body.get("action", "fix_grammar"),
                    "selected_text": body.get("text", ""),
                    "full_context": body.get("full_context", ""),
                    "custom_prompt": body.get("custom_prompt", ""),
                    "language": body.get("language", "en"),
                })
            elif command == "ai_task":
                await handler.handle_task_request({
                    "request_id": request_id,
                    "task_description": body.get("task_description", ""),
                    "full_context": body.get("full_context", ""),
                    "document_type": body.get("document_type", "text_doc"),
                    "language": body.get("language", "en"),
                })
            elif command == "ai_typeahead":
                await handler.handle_typeahead_request({
                    "request_id": request_id,
                    "text_before_cursor": body.get("text_before_cursor", ""),
                    "document_name": body.get("document_name", ""),
                    "language": body.get("language", "en"),
                })

            timeout = {"ai_edit": 30, "ai_task": 60, "ai_typeahead": 10}.get(command, 30)
            result = await asyncio.wait_for(result_future, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            return {"ok": False, "error": f"Timeout waiting for {command} result"}
        finally:
            ctrl._send = original_send

    logger.info("[DEBUG-API] Chat debug endpoints registered")
    return router
