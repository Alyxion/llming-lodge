"""Remote debug API for chat sessions.

Provides REST endpoints to inspect and control active chat sessions.
Protected by API key + IP whitelist + DEBUG_CHAT_REMOTE env flag.

Endpoints:
  GET  /debug/sessions          — list all active sessions
  GET  /debug/sessions/{id}     — session detail (history, state, tools)
  POST /debug/sessions/{id}/send — send a message
  GET  /debug/sessions/{id}/status — check streaming status
  POST /debug/sessions/{id}/model  — switch model
  GET  /debug/sessions/{id}/conversations — list IDB conversations (via browser WS)
  POST /debug/sessions/{id}/load_conversation — load a conversation by ID
"""

import ipaddress
import logging
import os
import time
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

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


# ── Router ──────────────────────────────────────────────────────────

def build_debug_router() -> APIRouter:
    """Build the debug API router. Include only when DEBUG_CHAT_REMOTE=1."""
    from .chat_session_api import SessionRegistry

    router = APIRouter(
        prefix="/api/llming-lodge/debug",
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
            "budget": ctrl.available_budget,
            "condensed_summary": ctrl.session._condensed_summary if ctrl.session else None,
        }

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

    # ── New chat ────────────────────────────────────────────

    @router.post("/sessions/{session_id}/new_chat")
    async def new_chat(session_id: str):
        from .chat_session_api import _handle_client_message
        registry = SessionRegistry.get()
        entry = registry.get_session(session_id)
        if not entry:
            raise HTTPException(404, f"Session {session_id} not found")

        await _handle_client_message(entry.controller, entry, {"type": "new_chat"})

        return {
            "status": "cleared",
            "new_session_id": entry.controller.session_id,
        }

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

    logger.info("[DEBUG-API] Chat debug endpoints registered")
    return router
