"""Lodge debug commands — registered via @command decorator.

Importing this module registers all commands in the default CommandRegistry.
These become REST endpoints AND MCP tools automatically.
"""

import asyncio
import logging
import time

from llming_com.command import command, CommandScope, CommandError

logger = logging.getLogger(__name__)


# ── Global commands ───────────────────────────────────────────────────


# list_sessions is provided by llming-com base_commands.
# Lodge overrides it to add chat-specific fields (model, streaming, title).

@command("list_sessions", app="lodge", description="List all active chat sessions with model, streaming status, and title",
         scope=CommandScope.GLOBAL, http_method="GET")
async def list_sessions(registry):
    now = time.monotonic()
    sessions = []
    for sid, entry in registry.list_sessions().items():
        ctrl = entry.controller
        msg_count = len(ctrl.session.history.messages) if ctrl.session else 0
        streaming = bool(ctrl._streaming_task and not ctrl._streaming_task.done())
        sessions.append({
            "session_id": sid,
            "user_id": entry.user_id,
            "user_name": entry.user_name,
            "user_email": entry.user_email,
            "model": ctrl.model,
            "streaming": streaming,
            "ws_connected": entry.websocket is not None,
            "message_count": msg_count,
            "idle_seconds": round(now - entry.last_activity),
            "title": ctrl._conversation_title,
        })
    return {"count": len(sessions), "sessions": sessions}


@command("list_models", app="lodge", description="List available LLM models",
         scope=CommandScope.GLOBAL, http_method="GET",
         tags=["models"])
async def list_models():
    from llming_lodge.chat_controller import llm_manager
    models = []
    for info in llm_manager.get_available_llms():
        try:
            provider = llm_manager.get_provider_for_model(info.model)
        except ValueError:
            provider = info.provider
        models.append({
            "model": info.model,
            "name": info.name,
            "label": info.label,
            "provider": provider,
            "icon": info.model_icon,
            "speed": info.speed,
            "quality": info.quality,
        })
    return {"models": models}


# ── Session-scoped commands ───────────────────────────────────────────


@command("get_session", app="lodge", description="Get session detail with message history",
         scope=CommandScope.SESSION, http_method="GET")
async def get_session(controller, entry, session_id: str, full: bool = False):
    ctrl = controller
    msg_count = len(ctrl.session.history.messages) if ctrl.session else 0
    streaming = bool(ctrl._streaming_task and not ctrl._streaming_task.done())

    result = {
        "session_id": session_id,
        "user_id": entry.user_id,
        "user_name": entry.user_name,
        "user_email": entry.user_email,
        "model": ctrl.model,
        "streaming": streaming,
        "ws_connected": entry.websocket is not None,
        "message_count": msg_count,
        "enabled_tools": list(ctrl.enabled_tools) if ctrl.enabled_tools else [],
    }

    # Include message history
    messages = []
    for msg in ctrl.session.history.messages:
        m = {
            "role": msg.role.value if hasattr(msg.role, "value") else str(msg.role),
            "content": msg.content if full else (msg.content[:500] if msg.content else ""),
            "content_length": len(msg.content) if msg.content else 0,
            "stale": msg.content_stale,
            "has_images": bool(msg.images),
            "image_count": len(msg.images) if msg.images else 0,
        }
        if full:
            tc_data = ctrl._message_tool_calls.get(id(msg))
            if tc_data:
                m["tool_calls"] = tc_data
            av_data = ctrl._message_avatar_overrides.get(id(msg))
            if av_data:
                m["avatar_override"] = av_data
        messages.append(m)
    result["messages"] = messages

    return result


@command("send_message", app="lodge", description="Send a chat message to the LLM",
         scope=CommandScope.SESSION, http_method="POST")
async def send_message(controller, entry, session_id: str,
                       text: str, images: list | None = None):
    if controller._streaming_task and not controller._streaming_task.done():
        raise CommandError(409, "Already streaming")

    # Render user bubble in browser
    if entry.websocket:
        await controller._send({
            "type": "user_message",
            "text": text,
            "images": images,
        })

    # Dev command intercept
    intercept_result = None
    if text and not images:
        from llming_lodge.dev.dev_commands import handle_dev_command
        try:
            intercept_result = await handle_dev_command(text, controller)
        except Exception:
            pass
        if intercept_result is None and controller._on_message_intercept:
            try:
                intercept_result = await controller._on_message_intercept(text, controller)
            except Exception:
                pass
        if intercept_result is not None:
            from llming_lodge.chat_controller import llm_manager
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
            return {
                "status": "intercepted",
                "session_id": session_id,
                "text": text,
                "response": intercept_result,
            }

    asyncio.create_task(controller.send_message(text, images=images))
    return {
        "status": "sent",
        "session_id": session_id,
        "text": text,
        "note": "Response streams via WebSocket to the connected browser",
    }


@command("get_status", app="lodge", description="Check streaming status of a session",
         scope=CommandScope.SESSION, http_method="GET")
async def get_status(controller, entry, session_id: str):
    streaming = bool(controller._streaming_task and not controller._streaming_task.done())
    last_msg = None
    if controller.session.history.messages:
        m = controller.session.history.messages[-1]
        last_msg = {
            "role": m.role.value if hasattr(m.role, "value") else str(m.role),
            "content_preview": m.content[:200] if m.content else "",
            "content_length": len(m.content) if m.content else 0,
        }
    return {
        "session_id": session_id,
        "streaming": streaming,
        "partial_text": controller._text_content[:500] if streaming else None,
        "message_count": len(controller.session.history.messages),
        "last_message": last_msg,
        "ws_connected": entry.websocket is not None,
    }


@command("switch_model", app="lodge", description="Switch the LLM model for a session",
         scope=CommandScope.SESSION, http_method="POST",
         tags=["models"])
async def switch_model(controller, session_id: str, model: str):
    old_model = controller.model
    await controller.switch_model(model)
    return {
        "session_id": session_id,
        "old_model": old_model,
        "new_model": controller.model,
    }


@command("run_js", app="lodge", description="Execute JavaScript in the browser",
         scope=CommandScope.SESSION, http_method="POST",
         requires_websocket=True)
async def run_js(controller, session_id: str, code: str):
    if not code:
        raise CommandError(400, "Missing 'code'")
    await controller._send({
        "type": "ui_action",
        "action": "run_js",
        "code": code,
    })
    return {"status": "ok"}


@command("get_console", app="lodge", description="Get browser console logs",
         scope=CommandScope.SESSION, http_method="GET",
         requires_websocket=True)
async def get_console(controller, session_id: str):
    console_buffer = controller._console_log_buffer if controller._console_log_buffer else []
    return {
        "logs": list(console_buffer),
        "total_buffered": len(console_buffer),
    }


@command("list_conversations", app="lodge",
         description="List conversations from the browser's IndexedDB",
         scope=CommandScope.SESSION, http_method="GET",
         requires_websocket=True)
async def list_conversations(controller, session_id: str):
    future = asyncio.get_event_loop().create_future()
    controller._pending_conv_list = future
    await controller._send({"type": "ui_action", "action": "list_conversations"})
    try:
        result = await asyncio.wait_for(future, timeout=5.0)
        return {"conversations": result}
    except asyncio.TimeoutError:
        return {"conversations": [], "note": "Timeout waiting for browser response"}
    finally:
        controller._pending_conv_list = None


@command("load_conversation", app="lodge",
         description="Load a conversation by ID from browser IndexedDB",
         scope=CommandScope.SESSION, http_method="POST",
         requires_websocket=True)
async def load_conversation(controller, session_id: str, conversation_id: str):
    if not conversation_id:
        raise CommandError(400, "conversation_id required")
    await controller._send({
        "type": "ui_action",
        "action": "load_conversation",
        "conversation_id": conversation_id,
    })
    return {"ok": True, "conversation_id": conversation_id}
