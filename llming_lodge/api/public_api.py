"""Public REST API for droplet management and remote chat.

Stateless Bearer-token auth using user API keys stored in MongoDB.
Endpoints:
  GET    /droplets            — list user's droplets
  GET    /droplets/{uid}      — get single droplet
  POST   /droplets            — create new droplet
  PUT    /droplets/{uid}      — update droplet fields (partial merge)
  POST   /droplets/{uid}/publish   — flush dev → live
  POST   /droplets/{uid}/unpublish — delete live version
  PUT    /droplets/{uid}/zip  — upload complete droplet as ZIP
  PUT    /droplets/_new/zip   — create new droplet from ZIP

  POST   /chat/send           — send a chat message (SSE stream by default)
  GET    /chat/tasks/{task_id} — poll for task result (non-blocking mode)
"""

import base64
import hashlib
import io
import logging
import mimetypes
import uuid
import zipfile
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)


# ── Validation constants (match client-side limits) ──────────────────

MAX_FILES = 40
MAX_SINGLE_FILE = 5 * 1024 * 1024      # 5 MB
MAX_TOTAL_SIZE = 10 * 1024 * 1024       # 10 MB
MAX_ZIP_SIZE = 15 * 1024 * 1024         # 15 MB (ZIP overhead)

ALLOWED_KNOWLEDGE_EXT = {".pdf", ".docx", ".xlsx", ".txt", ".md", ".csv"}
ALLOWED_MCP_JS_EXT = {".js", ".mjs"}
ALLOWED_MCP_DATA_EXT = {
    ".csv", ".tsv", ".json", ".xml", ".txt", ".md",
    ".pdf", ".docx", ".xlsx", ".xls", ".yaml", ".yml",
}
ALLOWED_ALL_EXT = ALLOWED_KNOWLEDGE_EXT | ALLOWED_MCP_JS_EXT | ALLOWED_MCP_DATA_EXT

# The manifest file name inside the ZIP
MANIFEST_NAME = "droplet.json"


# ── Auth ─────────────────────────────────────────────────────────────

async def _check_api_key(request: Request, nudge_store, required_permission: str):
    """Verify Bearer token and required permission. Returns user_email."""
    auth = request.headers.get("authorization", "")
    if not auth.lower().startswith("bearer "):
        raise HTTPException(401, "Missing Authorization: Bearer <key>")
    token = auth[7:].strip()
    if not token.startswith("llming_"):
        raise HTTPException(401, "Invalid API key format (expected llming_ prefix)")

    key_hash = hashlib.sha256(token.encode()).hexdigest()

    from nice_droplets.utils.mongo_helpers import get_async_mongo_client
    client = get_async_mongo_client(nudge_store._mongo_uri)
    db = client[nudge_store._mongo_db]
    coll = db["api_keys"]

    doc = await coll.find_one({"key_hash": key_hash})
    if not doc:
        raise HTTPException(401, "Invalid API key")

    permissions = doc.get("permissions", [])
    if required_permission not in permissions:
        raise HTTPException(403, f"Key lacks '{required_permission}' permission")

    return doc["user_email"]


# ── Request models ───────────────────────────────────────────────────

class DropletUpdate(BaseModel):
    """Partial update for a droplet. Only provided fields are merged."""
    name: Optional[str] = None
    description: Optional[str] = None
    icon: Optional[str] = None
    category: Optional[str] = None
    sub_category: Optional[str] = None
    system_prompt: Optional[str] = None
    model: Optional[str] = None
    language: Optional[str] = None
    team_id: Optional[str] = None
    visibility: Optional[list[str]] = None
    suggestions: Optional[list] = None
    capabilities: Optional[dict] = None


class ChatSendRequest(BaseModel):
    text: str
    droplet_uid: Optional[str] = None


# ── Router builder ───────────────────────────────────────────────────

def build_public_api_router(nudge_store) -> APIRouter:
    """Build the public API router. Always mounted."""
    router = APIRouter(prefix="/api/llming/v1")

    # ── Droplet endpoints ────────────────────────────────

    @router.get("/droplets")
    async def list_droplets(request: Request):
        user_email = await _check_api_key(request, nudge_store, "manage_droplets")
        results, _ = await nudge_store.search(
            user_email,
            mine=True,
            page_size=200,
        )
        return {"droplets": results}

    @router.get("/droplets/{uid}")
    async def get_droplet(uid: str, request: Request):
        user_email = await _check_api_key(request, nudge_store, "manage_droplets")
        doc = await nudge_store.get_for_user(uid, user_email)
        if not doc:
            raise HTTPException(404, f"Droplet {uid} not found")
        return doc

    @router.post("/droplets")
    async def create_droplet(request: Request):
        user_email = await _check_api_key(request, nudge_store, "manage_droplets")
        body = await request.json()
        body.setdefault("type", "nudge")
        body.setdefault("creator_email", user_email)
        result = await nudge_store.save(body, user_email)
        return result

    @router.put("/droplets/{uid}")
    async def update_droplet(uid: str, body: DropletUpdate, request: Request):
        user_email = await _check_api_key(request, nudge_store, "manage_droplets")
        # Fetch existing
        existing = await nudge_store.get_for_user(uid, user_email)
        if not existing:
            raise HTTPException(404, f"Droplet {uid} not found")
        # Merge only provided fields
        updates = body.model_dump(exclude_unset=True)
        existing.update(updates)
        result = await nudge_store.save(existing, user_email)
        return result

    @router.post("/droplets/{uid}/publish")
    async def publish_droplet(uid: str, request: Request):
        user_email = await _check_api_key(request, nudge_store, "manage_droplets")
        ok = await nudge_store.flush_to_live(uid, user_email, is_admin=True)
        if not ok:
            raise HTTPException(400, "Publish failed (not found or no permission)")
        return {"status": "published", "uid": uid}

    @router.post("/droplets/{uid}/unpublish")
    async def unpublish_droplet(uid: str, request: Request):
        user_email = await _check_api_key(request, nudge_store, "manage_droplets")
        ok = await nudge_store.unpublish(uid, user_email)
        if not ok:
            raise HTTPException(400, "Unpublish failed (not found or no permission)")
        return {"status": "unpublished", "uid": uid}

    # ── ZIP upload ────────────────────────────────────────

    @router.put("/droplets/{uid}/zip")
    async def upload_droplet_zip(uid: str, request: Request):
        """Upload a complete droplet as a ZIP file.

        ZIP structure:
          droplet.json          — manifest with droplet fields (name, system_prompt, etc.)
          files/                — attachment files (knowledge + MCP)
            report.pdf
            helper.js
            data.csv

        The manifest can contain any Nudge field except 'files' — those come
        from the files/ directory inside the ZIP.  If uid is '_new', a new
        droplet is created.
        """
        user_email = await _check_api_key(request, nudge_store, "manage_droplets")

        # Read body as raw bytes
        body = await request.body()
        content_type = request.headers.get("content-type", "")
        if "zip" not in content_type and not body[:4] == b"PK\x03\x04":
            raise HTTPException(400, "Expected a ZIP file (Content-Type: application/zip)")
        if len(body) > MAX_ZIP_SIZE:
            raise HTTPException(413, f"ZIP too large ({len(body)} bytes, max {MAX_ZIP_SIZE})")

        try:
            zf = zipfile.ZipFile(io.BytesIO(body))
        except zipfile.BadZipFile:
            raise HTTPException(400, "Invalid ZIP file")

        # Parse manifest
        import json as _json
        manifest = {}
        if MANIFEST_NAME in zf.namelist():
            try:
                manifest = _json.loads(zf.read(MANIFEST_NAME))
            except (ValueError, KeyError) as e:
                raise HTTPException(400, f"Invalid {MANIFEST_NAME}: {e}")
        manifest.pop("files", None)  # files come from ZIP, not manifest
        manifest.pop("_id", None)

        # Collect files from files/ directory
        files = []
        total_size = 0
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = info.filename
            # Skip manifest and dotfiles/macos metadata
            if name == MANIFEST_NAME or name.startswith("__MACOSX") or "/." in name:
                continue
            # Strip files/ prefix if present
            basename = name.split("/")[-1] if "/" in name else name
            if not basename:
                continue

            ext = "." + basename.rsplit(".", 1)[-1].lower() if "." in basename else ""
            if ext not in ALLOWED_ALL_EXT:
                raise HTTPException(
                    400,
                    f"File '{basename}' has disallowed extension '{ext}'. "
                    f"Allowed: {sorted(ALLOWED_ALL_EXT)}",
                )

            raw = zf.read(name)
            if len(raw) > MAX_SINGLE_FILE:
                raise HTTPException(
                    413,
                    f"File '{basename}' is {len(raw)} bytes (max {MAX_SINGLE_FILE})",
                )
            total_size += len(raw)
            if total_size > MAX_TOTAL_SIZE:
                raise HTTPException(413, f"Total file size exceeds {MAX_TOTAL_SIZE} bytes")

            mime = mimetypes.guess_type(basename)[0] or "application/octet-stream"
            # Store as data URL (same format as browser uploads) so browser
            # MCP Worker can decode JS files correctly.
            content_b64 = f"data:{mime};base64," + base64.b64encode(raw).decode("ascii")

            # For text files, store text_content directly
            text_content = ""
            if ext in {".txt", ".md", ".csv", ".tsv", ".json", ".xml", ".yaml", ".yml"}:
                try:
                    text_content = raw.decode("utf-8")
                except UnicodeDecodeError:
                    text_content = raw.decode("latin-1")

            files.append({
                "file_id": str(uuid.uuid4()),
                "name": basename,
                "size": len(raw),
                "mime_type": mime,
                "content": content_b64,
                "text_content": text_content,
            })

        if len(files) > MAX_FILES:
            raise HTTPException(400, f"Too many files ({len(files)}, max {MAX_FILES})")

        # Extract text from binary files (PDF, DOCX, XLSX)
        from llming_lodge.documents import extract_text
        for f in files:
            if f["text_content"] or f["mime_type"].startswith("text/"):
                continue
            try:
                raw = base64.b64decode(f["content"])
                f["text_content"] = extract_text(raw, f["mime_type"])
            except Exception as e:
                logger.warning("[ZIP] Text extraction failed for %s: %s", f["name"], e)

        # Build droplet document
        is_new = uid == "_new"
        if is_new:
            doc = {
                "type": "nudge",
                "mode": "dev",
                "creator_email": user_email,
            }
        else:
            doc = await nudge_store.get_for_user(uid, user_email, is_admin=True) or {}
            if not doc:
                raise HTTPException(404, f"Droplet {uid} not found")

        doc.update(manifest)
        doc["files"] = files
        doc.setdefault("type", "nudge")
        doc.setdefault("creator_email", user_email)

        result = await nudge_store.save(doc, user_email, is_admin=True)
        return {
            **result,
            "files_count": len(files),
            "total_size": total_size,
        }

    # ── Remote chat endpoints ────────────────────────────

    @router.post("/chat/send")
    async def chat_send(body: ChatSendRequest, request: Request, stream: bool = True):
        user_email = await _check_api_key(request, nudge_store, "automate_chat")

        from nice_droplets.utils.mongo_helpers import get_async_mongo_client
        import uuid
        client = get_async_mongo_client(nudge_store._mongo_uri)
        db = client[nudge_store._mongo_db]
        coll = db["remote_tasks"]

        # If droplet_uid provided, insert a select_droplet task first and wait
        if body.droplet_uid:
            select_id = str(uuid.uuid4())
            await coll.insert_one({
                "task_id": select_id,
                "user_email": user_email,
                "type": "select_droplet",
                "payload": {"droplet_uid": body.droplet_uid},
                "status": "pending",
                "chunks": [],
                "response": None,
                "created_at": datetime.now(timezone.utc),
            })
            # Wait for droplet selection + MCP activation to complete
            import asyncio
            for _ in range(100):  # up to 20 seconds
                doc = await coll.find_one({"task_id": select_id})
                if doc and doc.get("status") in ("completed", "error"):
                    break
                await asyncio.sleep(0.2)
            else:
                raise HTTPException(504, "Timeout waiting for droplet selection")
            if doc and doc.get("status") == "error":
                raise HTTPException(400, f"Droplet selection failed: {doc.get('error', '?')}")

        task_id = str(uuid.uuid4())
        task_doc = {
            "task_id": task_id,
            "user_email": user_email,
            "type": "send_message",
            "payload": {"text": body.text},
            "status": "pending",
            "chunks": [],
            "response": None,
            "created_at": datetime.now(timezone.utc),
        }
        await coll.insert_one(task_doc)

        if not stream:
            return {"task_id": task_id, "status": "pending"}

        # SSE streaming mode — poll MongoDB for chunks
        import asyncio

        async def event_stream():
            chunks_sent = 0
            max_wait = 120  # seconds
            start = asyncio.get_event_loop().time()
            while (asyncio.get_event_loop().time() - start) < max_wait:
                doc = await coll.find_one({"task_id": task_id})
                if not doc:
                    yield f"data: {_json_dumps({'type': 'error', 'message': 'Task not found'})}\n\n"
                    return

                # Send new chunks
                all_chunks = doc.get("chunks", [])
                if len(all_chunks) > chunks_sent:
                    for chunk in all_chunks[chunks_sent:]:
                        yield f"data: {_json_dumps({'type': 'chunk', 'text': chunk})}\n\n"
                    chunks_sent = len(all_chunks)

                if doc.get("status") == "completed":
                    resp = doc.get("response", {})
                    yield f"data: {_json_dumps({'type': 'done', 'text': resp.get('text', ''), 'model': resp.get('model', '')})}\n\n"
                    return

                if doc.get("status") == "error":
                    yield f"data: {_json_dumps({'type': 'error', 'message': doc.get('error', 'Unknown error')})}\n\n"
                    return

                await asyncio.sleep(0.2)

            yield f"data: {_json_dumps({'type': 'error', 'message': 'Timeout waiting for response'})}\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @router.get("/chat/tasks/{task_id}")
    async def chat_task_status(task_id: str, request: Request):
        user_email = await _check_api_key(request, nudge_store, "automate_chat")

        from nice_droplets.utils.mongo_helpers import get_async_mongo_client
        client = get_async_mongo_client(nudge_store._mongo_uri)
        db = client[nudge_store._mongo_db]
        coll = db["remote_tasks"]

        doc = await coll.find_one({"task_id": task_id, "user_email": user_email})
        if not doc:
            raise HTTPException(404, "Task not found")

        result = {
            "task_id": doc["task_id"],
            "status": doc["status"],
            "chunks": doc.get("chunks", []),
        }
        if doc["status"] == "completed":
            result["response"] = doc.get("response")
        return result

    return router


def _json_dumps(obj) -> str:
    """Compact JSON serialization for SSE data lines."""
    import json
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
