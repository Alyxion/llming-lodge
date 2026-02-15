"""Server integration helpers (framework-agnostic).

Host apps call these to register static assets and API routes.
"""

import os
import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)

# API path constants
API_PREFIX = "/api/llming-lodge"
API_UPLOAD = f"{API_PREFIX}/upload"
API_IMAGE_PASTE = f"{API_PREFIX}/image-paste"
API_LOAD_CONVERSATION = f"{API_PREFIX}/load-conversation"


class SessionDataStore:
    """Thread-safe store for session-scoped runtime data.

    Manages:
    - **assets**: binary blobs served via HTTP (e.g. contact photos from MCP tools)
    - **pasted_images**: base64 images posted by JS before the WS message references them
    - **pending_loads**: conversation data posted by JS for server-side hydration
    """

    _lock = threading.Lock()
    _assets: dict[str, dict[str, tuple[bytes, str]]] = {}
    _pasted_images: dict[str, list[str]] = {}
    _pending_loads: dict[str, dict] = {}

    # ── Assets ────────────────────────────────────────────────

    @classmethod
    def register_asset_bucket(cls, scope_key: str) -> None:
        with cls._lock:
            cls._assets[scope_key] = {}

    @classmethod
    def remove_asset_bucket(cls, scope_key: str) -> None:
        with cls._lock:
            cls._assets.pop(scope_key, None)

    @classmethod
    def put_asset(cls, scope_key: str, path: str, data: bytes, content_type: str) -> None:
        with cls._lock:
            bucket = cls._assets.get(scope_key)
            if bucket is not None:
                bucket[path] = (data, content_type)

    @classmethod
    def get_asset(cls, scope_key: str, path: str) -> Optional[tuple[bytes, str]]:
        with cls._lock:
            bucket = cls._assets.get(scope_key)
            if bucket is None:
                return None
            return bucket.get(path)

    # ── Pasted images ─────────────────────────────────────────

    @classmethod
    def set_pasted_images(cls, session_id: str, images: list[str]) -> None:
        with cls._lock:
            cls._pasted_images[session_id] = images

    @classmethod
    def pop_pasted_images(cls, session_id: str) -> list[str]:
        with cls._lock:
            return cls._pasted_images.pop(session_id, [])

    @classmethod
    def clear_pasted_images(cls, session_id: str) -> None:
        with cls._lock:
            cls._pasted_images.pop(session_id, None)

    # ── Pending conversation loads ────────────────────────────

    @classmethod
    def set_pending_load(cls, session_id: str, data: dict) -> None:
        with cls._lock:
            cls._pending_loads[session_id] = data

    @classmethod
    def pop_pending_load(cls, session_id: str) -> Optional[dict]:
        with cls._lock:
            return cls._pending_loads.pop(session_id, None)


def get_static_path() -> str:
    """Return absolute path to the static assets directory."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')


def get_chat_static_path() -> str:
    """Return absolute path to the chat static assets directory."""
    return os.path.join(get_static_path(), 'chat')


def build_http_router():
    """Build the FastAPI APIRouter with file upload, image paste, and asset endpoints.

    This contains the HTTP routes that were previously in ui/chat.py._build_router().
    The host app should include this router once at startup.
    """
    from fastapi import APIRouter, UploadFile, File, Header, HTTPException, Request
    from fastapi.responses import Response
    from pydantic import BaseModel

    from llming_lodge.documents import UploadManager, MAX_FILE_SIZE
    from llming_lodge.api.chat_session_api import SessionRegistry, _handle_client_message

    router = APIRouter(prefix=API_PREFIX)

    # ---- File uploads ----

    @router.post("/upload/{session_id}")
    async def upload_files(
        session_id: str,
        files: list[UploadFile] = File(...),
        x_user_id: str = Header(...),
    ):
        mgr = UploadManager.get(session_id)
        if not mgr or mgr.user_id != x_user_id:
            raise HTTPException(403, "Invalid session")
        results = []
        for f in files:
            content = await f.read()
            if len(content) > MAX_FILE_SIZE:
                raise HTTPException(413, f"File too large: {f.filename}")
            att = await mgr.store_file(f.filename, content, x_user_id)
            results.append({
                "name": att.name,
                "size": att.size,
                "fileId": att.file_id,
                "mimeType": att.mime_type,
            })
        return {"files": results}

    # ---- Image paste ----

    class ImagePastePayload(BaseModel):
        images: list[str] = []

    @router.post("/image-paste/{session_id}")
    async def image_paste(session_id: str, payload: ImagePastePayload):
        SessionDataStore.set_pasted_images(session_id, payload.images)
        logger.info(f"[IMAGE-PASTE-API] Stored {len(payload.images)} images for session {session_id}")
        return {"ok": True, "count": len(payload.images)}

    # ---- Conversation load ----

    @router.post("/load-conversation/{session_id}")
    async def load_conversation(session_id: str, request: Request):
        """Receive conversation data from JS (fetched from IDB) via HTTP POST."""
        data = await request.json()
        SessionDataStore.set_pending_load(session_id, data)
        logger.info(f"[LOAD-API] Received conversation {session_id} ({len(data.get('messages', []))} msgs)")
        return {"ok": True}

    # ---- Session-scoped assets (photos, etc.) ----

    @router.get("/asset/{scope_key}/{path:path}")
    async def serve_asset(scope_key: str, path: str):
        """Serve a session-scoped binary asset (e.g. contact photo)."""
        entry = SessionDataStore.get_asset(scope_key, path)
        if entry is None:
            return Response(status_code=404, content="Not found")
        data, content_type = entry
        return Response(
            content=data,
            media_type=content_type,
            headers={"Cache-Control": "private, max-age=3600"},
        )

    # ---- WebSocket echo (debug) ----

    from starlette.websockets import WebSocket as _WS, WebSocketDisconnect as _WSD

    @router.websocket("/ws-echo")
    async def ws_echo(ws: _WS):
        """Minimal echo endpoint for debugging WS connectivity."""
        await ws.accept()
        logger.info("[WS-ECHO] Client connected")
        try:
            while True:
                data = await ws.receive_text()
                await ws.send_text(f"echo: {data}")
        except _WSD:
            logger.info("[WS-ECHO] Client disconnected")

    # ---- WebSocket chat endpoint (Chat) ----

    import json as _json
    import time as _time

    @router.websocket("/ws/{session_id}")
    async def websocket_chat(ws: _WS, session_id: str):
        """WebSocket endpoint for Chat static frontend."""
        logger.info(f"[WS] Connection attempt for session {session_id}")

        registry = SessionRegistry.get()
        entry = registry.get_session(session_id)
        if not entry:
            logger.warning(f"[WS] Session {session_id} not found in registry "
                           f"(active sessions: {registry.active_count})")
            await ws.accept()
            await ws.send_json({
                "type": "error",
                "error_type": "SessionNotFound",
                "message": f"Session {session_id} not found",
            })
            await ws.close(code=4004, reason="Session not found")
            return

        await ws.accept()
        controller = entry.controller
        controller.set_websocket(ws)
        entry.websocket = ws

        # Wire condensation callbacks
        controller._wire_condensation()

        # Send session init
        init_msg = controller.build_session_init(user_name=entry.user_name)
        await ws.send_json(init_msg)

        # Send initial context info
        await controller._send_context_info()

        logger.info(f"[WS] Client connected to session {session_id}")

        try:
            while True:
                raw = await ws.receive_text()
                try:
                    msg = _json.loads(raw)
                except _json.JSONDecodeError:
                    await ws.send_json({"type": "error", "error_type": "InvalidJSON", "message": "Invalid JSON"})
                    continue

                entry.last_activity = _time.monotonic()
                await _handle_client_message(controller, entry, msg)

        except _WSD:
            logger.info(f"[WS] Client disconnected from session {session_id}")
        except Exception as e:
            logger.error(f"[WS] Error in session {session_id}: {type(e).__name__}: {e}")
        finally:
            controller.set_websocket(None)
            entry.websocket = None

    return router


def get_ws_router():
    """Return the combined FastAPI APIRouter for WebSocket + HTTP endpoints.

    Includes: file upload, image paste, conversation load, assets, WS chat.
    """
    return build_http_router()
