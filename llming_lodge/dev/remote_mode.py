"""Remote mode — poll MongoDB for tasks from external scripts.

Allows external tools (curl, scripts, CI) to send messages to an active
chat session via the REST API while the browser stays connected.
"""

import asyncio
import hashlib
import logging
import os
import secrets
from datetime import datetime, timezone, timedelta
from urllib.parse import urlparse
from uuid import uuid4

logger = logging.getLogger(__name__)


async def activate_remote_mode(controller) -> str:
    """Activate remote mode: poll MongoDB for tasks from external scripts."""
    if getattr(controller, "_remote_mode", False):
        expires = getattr(controller, "_remote_mode_expires", None)
        remaining = ""
        if expires:
            delta = expires - datetime.now(timezone.utc)
            remaining = f" ({int(delta.total_seconds() // 60)} min remaining)"
        return f"Remote mode is already active{remaining}."

    controller._remote_mode = True
    controller._remote_mode_expires = datetime.now(timezone.utc) + timedelta(minutes=30)

    # Upsert a temporary "remote mode" API key (replaced each activation)
    from llming_lodge.api.chat_session_api import (
        _get_api_keys_coll, _ensure_api_keys_indexes,
    )
    api_key = ""
    coll = _get_api_keys_coll(controller)
    if coll is not None:
        await _ensure_api_keys_indexes(coll)
        user_email = controller.user_mail or ""
        raw_key = "llming_" + secrets.token_hex(16)
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        await coll.update_one(
            {"user_email": user_email, "name": "_remote"},
            {"$set": {
                "key_prefix": raw_key[:10],
                "key_hash": key_hash,
                "permissions": ["manage_droplets", "automate_chat"],
                "created_at": datetime.now(timezone.utc),
            }, "$setOnInsert": {
                "key_id": str(uuid4()),
                "user_email": user_email,
                "name": "_remote",
            }},
            upsert=True,
        )
        api_key = raw_key
    else:
        api_key = "<could not create key -- no database>"

    # Start polling task
    controller._remote_poll_task = asyncio.create_task(_poll_remote_tasks(controller))

    base_url = os.environ.get('WEBSITE_URL', 'http://localhost:8080')
    _host = urlparse(base_url).hostname or ""
    insecure = " -k" if _host in ("localhost", "0.0.0.0", "127.0.0.1") else ""

    return (
        f"\U0001f517 **Remote mode activated** for 30 minutes.\n\n"
        f"Session: `{controller.session_id}`\n\n"
        f"API Key:\n```\n{api_key}\n```\n\n"
        f"Send messages via API:\n```bash\n"
        f"curl -N{insecure} \\\n"
        f"  -H \"Authorization: Bearer {api_key}\" \\\n"
        f"  -H \"Content-Type: application/json\" \\\n"
        f"  -d '{{\"text\": \"Hello\"}}' \\\n"
        f"  {base_url}/api/llming/v1/chat/send\n```\n\n"
        f"Type `/dev remote stop` to deactivate."
    )


def deactivate_remote_mode(controller) -> str | None:
    """Deactivate remote mode. Returns message if was active, else None."""
    was_active = getattr(controller, "_remote_mode", False)
    controller._remote_mode = False
    controller._remote_mode_expires = None
    poll_task = getattr(controller, "_remote_poll_task", None)
    if poll_task and not poll_task.done():
        poll_task.cancel()
        controller._remote_poll_task = None
    if was_active:
        return "Remote mode deactivated."
    return None


async def _poll_remote_tasks(controller) -> None:
    """Background loop: poll MongoDB for pending remote tasks."""
    from llming_lodge.api.chat_session_api import _get_remote_tasks_coll, _ensure_remote_tasks_indexes

    coll = _get_remote_tasks_coll(controller)
    if coll is None:
        logger.warning("[REMOTE] No MongoDB collection available")
        return
    await _ensure_remote_tasks_indexes(coll)

    user_email = controller.user_mail or ""
    controller._remote_poll_task = asyncio.current_task()

    try:
        while getattr(controller, "_remote_mode", False):
            # Check expiration
            expires = getattr(controller, "_remote_mode_expires", None)
            if expires and datetime.now(timezone.utc) > expires:
                controller._remote_mode = False
                await controller._send({
                    "type": "text_chunk",
                    "content": "\n\n*Remote mode expired.*",
                })
                break

            # Claim a pending task
            task_doc = await coll.find_one_and_update(
                {"user_email": user_email, "status": "pending"},
                {"$set": {"status": "processing"}},
            )
            if task_doc:
                task_id = task_doc["task_id"]
                task_type = task_doc.get("type", "send_message")
                payload = task_doc.get("payload", {})

                if task_type == "select_droplet":
                    await _handle_select_droplet(controller, coll, task_id, payload)
                elif task_type == "send_message":
                    await _handle_send_message(controller, coll, task_id, payload)
                else:
                    logger.warning("[REMOTE] Unknown task type: %s", task_type)
                    await coll.update_one(
                        {"task_id": task_id},
                        {"$set": {"status": "error", "error": f"Unknown type: {task_type}"}},
                    )
            else:
                await asyncio.sleep(2)
    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.error("[REMOTE] Polling error: %s", e)
    finally:
        controller._remote_poll_task = None


async def _handle_select_droplet(controller, coll, task_id: str, payload: dict) -> None:
    """Handle a select_droplet remote task."""
    droplet_uid = payload.get("droplet_uid", "")
    if not droplet_uid:
        await coll.update_one(
            {"task_id": task_id},
            {"$set": {"status": "error", "error": "Missing droplet_uid"}},
        )
        return

    try:
        await controller._send({
            "type": "open_droplet",
            "nudge_uid": droplet_uid,
        })
        _evt = getattr(controller, "_mcp_ready_event", None)
        if _evt:
            try:
                await asyncio.wait_for(_evt.wait(), timeout=20.0)
            except asyncio.TimeoutError:
                logger.warning("[REMOTE] MCP activation timed out for %s", droplet_uid)
        else:
            await asyncio.sleep(3)
        await coll.update_one(
            {"task_id": task_id},
            {"$set": {"status": "completed", "response": {
                "droplet_uid": droplet_uid,
                "tools": list(controller.enabled_tools),
            }}},
        )
        logger.info("[REMOTE] Selected droplet %s, tools: %s",
                     droplet_uid, controller.enabled_tools)
    except Exception as e:
        logger.error("[REMOTE] select_droplet error: %s", e)
        await coll.update_one(
            {"task_id": task_id},
            {"$set": {"status": "error", "error": str(e)}},
        )


async def _handle_send_message(controller, coll, task_id: str, payload: dict) -> None:
    """Handle a send_message remote task."""
    text = payload.get("text", "")
    if not text:
        return

    await controller._send({
        "type": "user_message",
        "text": text,
    })
    controller._remote_task_id = task_id
    try:
        await controller.send_message(text)
    except Exception as e:
        logger.error("[REMOTE] send_message error: %s", e)
        await coll.update_one(
            {"task_id": task_id},
            {"$set": {"status": "error", "error": str(e)}},
        )
    finally:
        controller._remote_task_id = None
