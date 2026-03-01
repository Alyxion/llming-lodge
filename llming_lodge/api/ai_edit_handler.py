"""AI document editing handler — one-shot LLM calls outside chat history."""

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Optional

from llming_lodge.api.writing_styles import get_style_context
from llming_lodge.providers.llm_provider_models import ReasoningEffort

if TYPE_CHECKING:
    from llming_lodge.api.chat_session_api import WebSocketChatController

logger = logging.getLogger(__name__)

# Action → system prompt fragment
_EDIT_ACTIONS: dict[str, str] = {
    "fix_grammar": "Fix grammar, spelling, and punctuation errors. Preserve the original meaning, tone, and style exactly.",
    "formal": "Rewrite the text in a polished, professional tone suitable for business correspondence. Keep the core message intact.",
    "casual": "Rewrite the text in a warm, approachable, and conversational tone while preserving the key message.",
    "shorter": "Condense the text to its essential points. Remove filler words and redundancy. Keep every key fact.",
    "expand": "Elaborate on the text with additional detail, examples, or explanation. Maintain the original voice and style.",
    "improve": "Improve clarity, flow, and readability. Fix any issues and make the text more compelling without changing its meaning.",
    "simplify": "Rewrite the text in simple, clear language. Avoid jargon and complex sentence structures. Keep the meaning intact.",
    "bullet_points": "Convert the text into a well-structured bulleted list. Each bullet should be concise and self-contained.",
    "translate_de": "Translate the text into German. Produce fluent, natural-sounding German — not a word-for-word translation.",
    "translate_en": "Translate the text into English. Produce fluent, natural-sounding English — not a word-for-word translation.",
    "translate_fr": "Translate the text into French. Produce fluent, natural-sounding French — not a word-for-word translation.",
    "translate_it": "Translate the text into Italian. Produce fluent, natural-sounding Italian — not a word-for-word translation.",
    "translate_zh": "Translate the text into Mandarin Chinese. Produce fluent, natural-sounding Chinese — not a word-for-word translation.",
    "translate_es": "Translate the text into Spanish. Produce fluent, natural-sounding Spanish — not a word-for-word translation.",
}


class AIEditHandler:
    """Handles AI edit / task / typeahead requests for a single session."""

    def __init__(self, controller: "WebSocketChatController",
                 user_name: str = "", user_mail: str = "") -> None:
        self._ctrl = controller
        self._user_name = user_name
        self._user_mail = user_mail or controller.user_mail or ""
        self._pending_typeahead: Optional[asyncio.Task] = None
        self._pending_edit: Optional[asyncio.Task] = None
        self._pending_task: Optional[asyncio.Task] = None

    def _make_client(self, temperature: float, max_tokens: int,
                     model_override: str | None = None,
                     reasoning_effort: ReasoningEffort | None = None):
        """Create a one-shot LLM client via the session's provider."""
        session = self._ctrl.session
        provider = session._provider
        return provider.create_client(
            provider=session.config.provider,
            model=model_override or session.config.model,
            base_url=session.config.base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            toolboxes=[],
            reasoning_effort=reasoning_effort,
        )

    async def _invoke_with_budget(self, messages, temperature: float, max_tokens: int,
                                  timeout: float, operation_type: str,
                                  model_override: str | None = None,
                                  reasoning_effort: ReasoningEffort | None = None):
        """Invoke LLM with budget reserve/return/log flow."""
        session = self._ctrl.session
        reserved = False

        # Resolve model info for budget pricing
        if model_override:
            _model_info = next(
                (m for m in session._provider.get_models()
                 if m.name == model_override or m.model == model_override),
                session.model_info,
            )
        else:
            _model_info = session.model_info

        # 1. Reserve budget
        if session.budget_manager:
            est_input = session._estimate_tokens(messages)
            await session.budget_manager.reserve_budget_async(
                input_tokens=est_input, max_output_tokens=max_tokens,
                input_token_price=_model_info.input_token_price,
                output_token_price=_model_info.output_token_price,
            )
            reserved = True

        try:
            # 2. Stream (never use ainvoke — it doesn't handle tool loops)
            client = self._make_client(temperature, max_tokens, model_override=model_override,
                                       reasoning_effort=reasoning_effort)
            t0 = time.monotonic()
            parts: list[str] = []
            last_meta: dict = {}

            async def _stream():
                async for chunk in client.astream(messages):
                    if chunk.content:
                        parts.append(chunk.content)
                    meta = getattr(chunk, 'response_metadata', None)
                    if meta:
                        last_meta.update(meta)

            await asyncio.wait_for(_stream(), timeout=timeout)
            duration_ms = (time.monotonic() - t0) * 1000

            # 3. Extract actual tokens from last chunk metadata
            usage = last_meta.get('usage', last_meta)
            actual_in = (usage.get('input_tokens') or usage.get('prompt_tokens')
                         or usage.get('total_input_tokens') or 0)
            actual_out = (usage.get('output_tokens') or usage.get('completion_tokens')
                          or usage.get('total_output_tokens') or 0)

            # 4. Return unused + log
            if session.budget_manager:
                await session.budget_manager.return_unused_budget_async(
                    reserved_output_tokens=max_tokens, actual_output_tokens=actual_out,
                    output_token_price=_model_info.output_token_price,
                )
                actual_cost = (
                    actual_in * (_model_info.input_token_price / 1_000_000) +
                    actual_out * (_model_info.output_token_price / 1_000_000)
                )
                model_name = f"{session.config.provider}.{model_override or session.config.model}"
                for limit in session.budget_manager.limits.values():
                    await limit.log_usage_async(
                        model_name=model_name, tokens_input=actual_in,
                        tokens_output=actual_out, costs=actual_cost,
                        duration_ms=duration_ms, user_id=session.user_id,
                        operation_type=operation_type,
                    )
                logger.info("[AI_EDIT] %s — model=%s in=%d out=%d cost=%.5f dur=%.0fms",
                            operation_type, model_name, actual_in, actual_out, actual_cost, duration_ms)

            from llming_lodge.messages import LlmAIMessage
            return LlmAIMessage(content="".join(parts), response_metadata=last_meta)

        except (asyncio.CancelledError, Exception) as e:
            # Return full reserved budget on failure/cancel
            if reserved and session.budget_manager:
                await session.budget_manager.return_unused_budget_async(
                    reserved_output_tokens=max_tokens, actual_output_tokens=0,
                    output_token_price=_model_info.output_token_price,
                )
            raise

    async def handle_edit_request(self, msg: dict) -> None:
        """Process an ai_edit_request — spawns cancellable task."""
        # Cancel any existing edit
        if self._pending_edit and not self._pending_edit.done():
            self._pending_edit.cancel()

        request_id = msg.get("request_id", "")
        action = msg.get("action", "fix_grammar")
        selected_text = msg.get("selected_text", "")
        full_context = msg.get("full_context", "")
        custom_prompt = msg.get("custom_prompt", "")
        language = msg.get("language", "en")

        if not selected_text:
            await self._ctrl._send({
                "type": "ai_edit_result",
                "request_id": request_id,
                "status": "error",
                "error": "No text selected",
            })
            return

        async def _run():
            style_ctx = get_style_context(language)
            action_instruction = custom_prompt if action == "custom" else _EDIT_ACTIONS.get(action, _EDIT_ACTIONS["fix_grammar"])

            system = (
                "You are a writing assistant. "
                f"{action_instruction} "
                "Output ONLY the rewritten text — no explanations, no quotes, no markdown.\n"
                f"{self._user_identity_hint()}"
                f"{style_ctx}"
            )
            human = f"Document context (for reference only):\n{full_context[:2000]}\n\n---\nText to rewrite:\n{selected_text}"

            try:
                from llming_lodge.messages import LlmSystemMessage, LlmHumanMessage
                from llming_lodge.budget import InsufficientBudgetError

                response = await self._invoke_with_budget(
                    [LlmSystemMessage(content=system), LlmHumanMessage(content=human)],
                    temperature=0.3, max_tokens=2000, timeout=30,
                    operation_type="text_enhancement",
                )

                result_text = response.content.strip()
                await self._ctrl._send({
                    "type": "ai_edit_result",
                    "request_id": request_id,
                    "status": "ok",
                    "original_text": selected_text,
                    "result_text": result_text,
                })
            except asyncio.CancelledError:
                await self._ctrl._send({
                    "type": "ai_edit_result",
                    "request_id": request_id,
                    "status": "cancelled",
                })
            except InsufficientBudgetError:
                await self._ctrl._send({
                    "type": "ai_edit_result",
                    "request_id": request_id,
                    "status": "error",
                    "error": "budget_exceeded",
                })
            except Exception as e:
                logger.warning("[AI_EDIT] Edit request failed: %s", e)
                await self._ctrl._send({
                    "type": "ai_edit_result",
                    "request_id": request_id,
                    "status": "error",
                    "error": str(e),
                })

        self._pending_edit = asyncio.create_task(_run())

    def _user_identity_hint(self) -> str:
        """Return a short user-identity line for system prompts."""
        parts = []
        if self._user_name:
            parts.append(self._user_name)
        if self._user_mail:
            parts.append(self._user_mail)
        if parts:
            return f"\nThe user is: {', '.join(parts)}.\n"
        return ""

    async def handle_task_request(self, msg: dict) -> None:
        """Process an ai_task_request via the session's full pipeline.

        Uses the same model, tools, MCPs, and web access as the main chat
        so the generate feature has full capabilities.  The doc-plugin
        preamble is skipped; the LLM returns markdown which the frontend
        renders using the same MarkdownRenderer as the main chat.
        """
        # Cancel any existing task
        if self._pending_task and not self._pending_task.done():
            self._pending_task.cancel()

        request_id = msg.get("request_id", "")
        task_description = msg.get("task_description", "")
        full_context = msg.get("full_context", "")
        document_type = msg.get("document_type", "text_doc")
        language = msg.get("language", "en")

        if not task_description:
            await self._ctrl._send({
                "type": "ai_task_result",
                "request_id": request_id,
                "status": "error",
                "error": "No task description",
            })
            return

        async def _run():
            style_ctx = get_style_context(language)

            system = (
                "You are a writing assistant. Generate content using markdown. "
                "Use standard markdown: headings (#, ##), lists (-, *), bold (**), "
                "italic (*), tables, LaTeX math ($...$, $$...$$), "
                "mermaid diagrams (```mermaid). "
                "Output ONLY the NEW content to be inserted — do NOT repeat or "
                "include the existing document. No explanations.\n"
                f"{self._user_identity_hint()}"
                f"{style_ctx}"
            )
            human = (
                f"Existing document (for context only, do NOT repeat it):\n{full_context[:4000]}\n\n---\n"
                f"Task: {task_description}"
            )

            session = self._ctrl.session
            history_len = len(session.history.messages)

            try:
                from llming_lodge.budget import InsufficientBudgetError

                # Always use streaming — ainvoke doesn't handle tool loops.
                # Collect chunks silently (no WS forwarding to frontend).
                async def _collect():
                    stream = await session.chat_async(
                        message=human,
                        streaming=True,
                        system_prompt=system,
                        max_tokens=4000,
                        skip_preamble=True,
                    )
                    parts = []
                    async for chunk in stream:
                        if chunk.content and not getattr(chunk, 'tool_call', None):
                            parts.append(chunk.content)
                    return "".join(parts)

                content = (await asyncio.wait_for(_collect(), timeout=90)).strip()
                await self._ctrl._send({
                    "type": "ai_task_result",
                    "request_id": request_id,
                    "status": "ok",
                    "content": content,
                })
            except asyncio.CancelledError:
                await self._ctrl._send({
                    "type": "ai_task_result",
                    "request_id": request_id,
                    "status": "cancelled",
                })
            except InsufficientBudgetError:
                await self._ctrl._send({
                    "type": "ai_task_result",
                    "request_id": request_id,
                    "status": "error",
                    "error": "budget_exceeded",
                })
            except Exception as e:
                logger.warning("[AI_EDIT] Task request failed: %s", e)
                await self._ctrl._send({
                    "type": "ai_task_result",
                    "request_id": request_id,
                    "status": "error",
                    "error": str(e),
                })
            finally:
                # Remove task messages from chat history to avoid polluting it
                session.history.messages = session.history.messages[:history_len]

        self._pending_task = asyncio.create_task(_run())

    # Model used for ghost text — must be fast (<2s), minimal cost
    TYPEAHEAD_MODEL = "gpt-5-nano"

    async def handle_typeahead_request(self, msg: dict) -> None:
        """Process an ai_typeahead_request — runs as cancellable task."""
        self._cancel_typeahead()

        request_id = msg.get("request_id", "")
        text_before = msg.get("text_before_cursor", "")
        text_after = msg.get("text_after_cursor", "")

        if not text_before or len(text_before.strip()) < 30:
            logger.debug("[GHOST] Skipped — text too short (%d chars)", len(text_before.strip()))
            return

        logger.info("[GHOST] Starting request %s — %d chars before, %d chars after",
                    request_id, len(text_before), len(text_after))

        async def _run():
            context = text_before[-500:]
            after_context = text_after[:500].strip() if text_after else ""

            after_hint = ""
            if after_context:
                after_hint = (
                    f"\nThe document continues after the cursor with: {after_context[:300]}\n"
                    "Use this to match the topic, tone, and language — but do NOT repeat it."
                )

            system = (
                "You are a ghost-text autocomplete engine. "
                "The user is typing in a document. Your job: output ONLY the text that "
                "continues naturally from the exact point where the text ends. "
                "CRITICAL: If the last word is incomplete (e.g. 'Inter'), you MUST "
                "finish that word first (e.g. 'esse'). Never start a new word when "
                "the previous word is unfinished. "
                "Stay in the SAME language as the text. "
                "Output a few words to one short sentence — no quotes, no explanations, "
                "no prefixes, no commentary about the text."
                f"{after_hint}"
            )

            try:
                from llming_lodge.messages import LlmSystemMessage, LlmHumanMessage

                response = await self._invoke_with_budget(
                    [LlmSystemMessage(content=system), LlmHumanMessage(content=context)],
                    temperature=0.2, max_tokens=40, timeout=5,
                    operation_type="ghost_suggestion",
                    model_override=self.TYPEAHEAD_MODEL,
                    reasoning_effort=ReasoningEffort.NONE,
                )

                suggestion = response.content.strip().strip("\"'")
                if suggestion:
                    logger.info("[GHOST] Suggestion ready %s — '%s'", request_id, suggestion[:50])
                    await self._ctrl._send({
                        "type": "ai_typeahead_suggestion",
                        "request_id": request_id,
                        "suggestion": suggestion,
                        "replace_count": 0,
                    })
                else:
                    logger.info("[GHOST] Empty suggestion %s", request_id)
            except asyncio.CancelledError:
                logger.info("[GHOST] Cancelled %s", request_id)
            except Exception as e:
                logger.warning("[GHOST] Failed %s: %s", request_id, e)

        self._pending_typeahead = asyncio.create_task(_run())

    def _cancel_typeahead(self) -> None:
        if self._pending_typeahead and not self._pending_typeahead.done():
            logger.info("[GHOST] Cancelling pending typeahead")
            self._pending_typeahead.cancel()
            self._pending_typeahead = None

    def cancel_typeahead(self, msg: dict) -> None:
        """Cancel any pending typeahead request."""
        self._cancel_typeahead()

    async def cancel_edit(self, msg: dict) -> None:
        """Cancel any pending edit request."""
        if self._pending_edit and not self._pending_edit.done():
            self._pending_edit.cancel()
            self._pending_edit = None

    async def cancel_task(self, msg: dict) -> None:
        """Cancel any pending task request."""
        if self._pending_task and not self._pending_task.done():
            self._pending_task.cancel()
            self._pending_task = None
