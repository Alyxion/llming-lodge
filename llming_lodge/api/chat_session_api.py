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
from llming_lodge.budget.budget_limit import BudgetLimit
from llming_lodge.llm_base_models import Role, ChatMessage
from llming_lodge.tools.tool_call import ToolCallInfo, ToolCallStatus
from llming_lodge.chat_controller import ChatController, llm_manager
from llming_lodge.tools.tool_definition import MCPServerConfig
from llming_lodge.tools.tool_registry import get_default_registry
from llming_lodge.documents import UploadManager
from llming_lodge.utils.image_utils import sniff_image_mime
from llming_lodge.utils import LlmMarkdownPostProcessor

logger = logging.getLogger(__name__)


# ── Session Registry ────────────────────────────────────────────────


@dataclass
class SessionEntry:
    """A registered chat session with its controller and metadata."""
    controller: "WebSocketChatController"
    user_id: str
    user_name: str = ""
    websocket: Optional[WebSocket] = None
    created_at: float = field(default_factory=time.monotonic)
    last_activity: float = field(default_factory=time.monotonic)
    upload_manager: Optional[UploadManager] = None
    mcp_servers: Optional[List[MCPServerConfig]] = None
    _cleanup_done: bool = False


class SessionRegistry:
    """Maps session_id → SessionEntry. Thread-safe via asyncio."""

    _instance: Optional["SessionRegistry"] = None

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
        mcp_servers: Optional[List[MCPServerConfig]] = None,
    ) -> SessionEntry:
        entry = SessionEntry(
            controller=controller,
            user_id=user_id,
            user_name=user_name,
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
    ):
        super().__init__(
            user_id=user_id,
            user_mail=user_mail,
            budget_limits=budget_limits,
            system_prompt=system_prompt,
            context_preamble=context_preamble,
            mcp_servers=mcp_servers,
            initial_model=initial_model,
        )
        self.session_id = session_id
        self._ws: Optional[WebSocket] = None
        self._base_system_prompt = system_prompt or self.system_prompt
        self._conversation_title: Optional[str] = None
        self._title_msg_count: int = 0

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

    def _on_text_chunk(self, content: str) -> None:
        asyncio.create_task(self._send({
            "type": "text_chunk",
            "content": content,
        }))

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

        # Trigger title generation and save in background
        asyncio.create_task(self._post_response_tasks())

    async def _post_response_tasks(self) -> None:
        """After response: generate title, save conversation, send context info."""
        try:
            await self._generate_title()
        except Exception as e:
            logger.debug(f"[TITLE] Title generation failed: {e}")

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

            base_tokens = len(self._base_system_prompt or "") // 4
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

            total = base_tokens + doc_tokens + history_tokens + image_tokens + tool_tokens
            pct_exact = min(100.0, total / max_input * 100) if max_input > 0 else 0.0
            est_cost = (total * model_info.input_token_price + 500 * model_info.output_token_price) / 1_000_000

            return {
                "pct": round(pct_exact),
                "pctExact": round(pct_exact, 2),
                "historyTokens": history_tokens,
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

    # ── Document context sync (mirrors ChatView._sync_document_context) ───

    def sync_document_context(self, upload_manager: Optional[UploadManager]) -> None:
        """Rebuild system prompt = base prompt + attached document texts."""
        from llming_lodge.documents import extract_text, truncate_to_token_budget

        base = self._base_system_prompt or self.system_prompt or ""

        # Strip any previous document section
        marker = "\n\n---\n[Attached documents context]"
        idx = base.find(marker)
        if idx != -1:
            base = base[:idx]

        doc_section = ""
        if upload_manager:
            raw_docs = []
            for f in upload_manager.files:
                if f.mime_type.startswith("image/"):
                    continue
                if not f.text_content:
                    f.text_content = extract_text(f.path, f.mime_type)
                raw_docs.append((f.name, f.text_content))
            if raw_docs:
                truncated = truncate_to_token_budget(raw_docs)
                doc_texts = [f"### {name}\n{text}" for name, text in truncated]
                doc_section = marker + "\n" + "\n\n".join(doc_texts)

        self._base_system_prompt = base
        self.update_settings(system_prompt=base + doc_section)

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
            title = first_user.content[:30].strip() if first_user else "Untitled"

        now = datetime.now().isoformat()
        created_at = messages[0].timestamp.isoformat() if messages else now

        return {
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

            response = await client.ainvoke(messages)
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

    def build_session_init(self, user_name: str = "") -> dict:
        """Build the session_init message sent when WS connects."""
        import math

        models = []
        for info in llm_manager.get_available_llms():
            # Normalize cost to 1–10 scale (log scale, $0.10→1, $75→10)
            price = info.output_token_price or 0
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

        quick_actions = [
            {
                "id": "@sys.docs",
                "label": "Analyze a document",
                "desc": "Upload PDFs, Word, or Excel files",
                "icon": "description",
                "engagement": "I'd like to analyze a document.",
                "prompt": "You are a document analyst. Help the user understand, summarize, and extract insights from uploaded documents. Ask clarifying questions about what they need.",
            },
            {
                "id": "@sys.image",
                "label": "Create an image",
                "desc": "Generate images from your description",
                "icon": "palette",
                "engagement": "I want to create an image.",
                "prompt": "You are a creative image generator. Help the user describe and generate images. Ask about style, subject, mood, and details to create the perfect image. Start by asking what they want to visualize.",
            },
            {
                "id": "@sys.ideas",
                "label": "Brainstorm ideas",
                "desc": "Explore concepts and strategies",
                "icon": "lightbulb",
                "engagement": "I'd like to brainstorm ideas.",
                "prompt": "You are a creative consultant. Help the user brainstorm. Start by asking what topic or challenge they want to explore, then guide them with questions and suggestions.",
            },
            {
                "id": "@sys.code",
                "label": "Write code",
                "desc": "Build scripts, apps, and automations",
                "icon": "code",
                "engagement": "I need help writing code.",
                "prompt": "You are a coding assistant. Help the user write code. Start by asking what they want to build, which language, and any constraints. Then guide them step by step.",
            },
            {
                "id": "@sys.summary",
                "label": "Summarize text",
                "desc": "Condense articles, notes, or reports",
                "icon": "summarize",
                "engagement": "I need to summarize something.",
                "prompt": "You are a summary assistant. Help the user condense text. Start by asking them to paste or upload the text they want summarized, and what format they prefer (bullet points, paragraph, key takeaways).",
            },
        ]

        return {
            "type": "session_init",
            "session_id": self.session_id,
            "user_name": user_name,
            "models": models,
            "current_model": self.model,
            "tools": self.get_all_known_tools(),
            "budget": self.available_budget,
            "system_prompt": self._base_system_prompt or self.system_prompt,
            "temperature": self.temperature,
            "max_input_tokens": self.max_input_tokens,
            "max_output_tokens": self.max_output_tokens,
            "quick_actions": quick_actions,
        }


# ── WebSocket Router Builder ────────────────────────────────────────


def build_ws_router():
    """Build a FastAPI APIRouter with the WebSocket chat endpoint."""
    from fastapi import APIRouter

    router = APIRouter(prefix="/api/llming-lodge")

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
        init_msg = controller.build_session_init(user_name=entry.user_name)
        await ws.send_json(init_msg)

        # Send initial context info
        await controller._send_context_info()

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
        if text:
            try:
                await controller.send_message(text, images=images)
            except Exception as e:
                logger.error(f"[WS] send_message error: {e}")

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
        if kwargs:
            controller.update_settings(**kwargs)
            await controller._send_context_info()

    elif msg_type == "toggle_tool":
        name = msg.get("name", "")
        enabled = msg.get("enabled", True)
        if name:
            controller.toggle_tool(name, enabled)
            await controller._send({
                "type": "tools_updated",
                "tools": controller.get_all_known_tools(),
            })
            await controller._send_context_info()

    elif msg_type == "new_chat":
        # Save current conversation first
        data = controller._serialize_conversation()
        if data:
            await controller._send({"type": "save_conversation", "data": data})

        # Clear history and create new session
        controller.clear_history()
        old_session_id = controller.session_id
        new_session_id = str(uuid4())
        controller.session_id = new_session_id
        controller._conversation_title = None
        controller._title_msg_count = 0

        # Restore default system prompt
        if controller._base_system_prompt:
            controller.update_settings(system_prompt=controller._base_system_prompt)

        # Clear pasted images
        from llming_lodge.server import SessionDataStore
        SessionDataStore.clear_pasted_images(old_session_id)

        # Clean up old upload manager, create new one
        if entry.upload_manager:
            entry.upload_manager.cleanup()
        entry.upload_manager = UploadManager.create(new_session_id, entry.user_id)

        # Re-register in registry under new ID
        registry = SessionRegistry.get()
        registry.remove(old_session_id)
        registry._sessions[new_session_id] = entry

        # Re-wire condensation callbacks
        controller._wire_condensation()

        await controller._send({
            "type": "chat_cleared",
            "new_session_id": new_session_id,
        })

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

    elif msg_type == "file_uploaded":
        # Frontend uploaded a file via HTTP; rebuild document context
        controller.sync_document_context(entry.upload_manager)
        await controller._send_context_info()

    elif msg_type == "file_removed":
        file_id = msg.get("file_id", "")
        if file_id and entry.upload_manager:
            entry.upload_manager.remove_file(file_id, entry.user_id)
        controller.sync_document_context(entry.upload_manager)
        await controller._send_context_info()

    elif msg_type == "conversation_list":
        # Response from browser with IDB conversation list
        convs = msg.get("conversations", [])
        if hasattr(controller, '_pending_conv_list') and controller._pending_conv_list:
            try:
                controller._pending_conv_list.set_result(convs)
            except Exception:
                pass

    elif msg_type == "heartbeat":
        await controller._send({"type": "heartbeat_ack"})

    else:
        logger.warning(f"[WS] Unknown message type: {msg_type}")


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

        # Restore enabled tools
        if data.get("enabled_tools"):
            controller.update_settings(tools=data["enabled_tools"])

        # Re-wire condensation callbacks
        controller._wire_condensation()

        await controller._send_context_info()

        logger.info(f"[LOAD] Loaded conversation {conv_id} with {len(data.get('messages', []))} messages")

    except Exception as e:
        logger.error(f"[LOAD] Failed to load conversation: {e}", exc_info=True)
        await controller._send({
            "type": "error",
            "error_type": "LoadError",
            "message": f"Failed to load conversation: {e}",
        })
