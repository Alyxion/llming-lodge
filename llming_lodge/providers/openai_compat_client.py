"""
OpenAI-compatible Chat Completions API client.

Used by providers with OpenAI-compatible APIs (DeepSeek, Together, Mistral)
that don't support OpenAI's Responses API.
"""
from __future__ import annotations

import itertools
import json
import logging
from typing import (
    Any,
    AsyncIterator,
    Iterator,
    List,
    Union,
    Optional,
    Dict,
)

from openai import OpenAI, AsyncOpenAI

from llming_lodge.llm_base_client import LlmClient
from llming_lodge.messages import (
    LlmAIMessage,
    LlmHumanMessage,
    LlmSystemMessage,
    LlmMessageChunk,
)
from llming_lodge.llm_base_models import Role

logger = logging.getLogger(__name__)


def _convert_messages(
    messages: List[Union[LlmSystemMessage, LlmHumanMessage, LlmAIMessage]],
) -> List[Dict[str, Any]]:
    """Convert internal messages to OpenAI Chat Completions format."""
    role_map = {
        LlmSystemMessage: "system",
        LlmHumanMessage: "user",
        LlmAIMessage: "assistant",
    }
    converted = []
    for m in messages:
        role = role_map[type(m)]
        converted.append({"role": role, "content": m.content})
    return converted


class OpenAICompatibleClient(LlmClient):
    """Client for OpenAI-compatible APIs (DeepSeek, Together, Mistral).

    Uses the standard Chat Completions API (not Responses API).
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        streaming: bool = False,
        base_url: Optional[str] = None,
    ) -> None:
        super().__init__(model, temperature, max_tokens, streaming)
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._aclient = AsyncOpenAI(api_key=api_key, base_url=base_url)

    def _build_kwargs(
        self,
        messages: List[Union[LlmSystemMessage, LlmHumanMessage, LlmAIMessage]],
        stream: bool = False,
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": _convert_messages(messages),
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        if stream:
            kwargs["stream"] = True
            kwargs["stream_options"] = {"include_usage": True}
        return kwargs

    # ── synchronous invoke ──────────────────────────────────────────────── #

    def invoke(
        self,
        messages: List[Union[LlmSystemMessage, LlmHumanMessage, LlmAIMessage]],
    ) -> LlmAIMessage:
        kwargs = self._build_kwargs(messages)
        response = self._client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or ""
        metadata = {}
        if response.usage:
            metadata = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            }
        return LlmAIMessage(content=content, response_metadata=metadata)

    # ── async invoke ────────────────────────────────────────────────────── #

    async def ainvoke(
        self,
        messages: List[Union[LlmSystemMessage, LlmHumanMessage, LlmAIMessage]],
    ) -> LlmAIMessage:
        kwargs = self._build_kwargs(messages)
        response = await self._aclient.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or ""
        metadata = {}
        if response.usage:
            metadata = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            }
        return LlmAIMessage(content=content, response_metadata=metadata)

    # ── synchronous streaming ───────────────────────────────────────────── #

    def stream(
        self,
        messages: List[Union[LlmSystemMessage, LlmHumanMessage, LlmAIMessage]],
    ) -> Iterator[LlmMessageChunk]:
        kwargs = self._build_kwargs(messages, stream=True)
        chunk_index = itertools.count()

        response_stream = self._client.chat.completions.create(**kwargs)
        for chunk in response_stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            content = delta.content if delta and delta.content else ""
            if content:
                yield LlmMessageChunk(
                    content=content,
                    role=Role.ASSISTANT,
                    index=next(chunk_index),
                    is_final=False,
                    response_metadata={},
                )

        yield LlmMessageChunk(
            content="",
            role=Role.ASSISTANT,
            index=next(chunk_index),
            is_final=True,
            response_metadata={},
        )

    # ── async streaming ─────────────────────────────────────────────────── #

    async def astream(
        self,
        messages: List[Union[LlmSystemMessage, LlmHumanMessage, LlmAIMessage]],
        usage_callback: Optional[callable] = None,
    ) -> AsyncIterator[LlmMessageChunk]:
        kwargs = self._build_kwargs(messages, stream=True)
        chunk_index = itertools.count()

        total_input_tokens = 0
        total_output_tokens = 0

        response_stream = await self._aclient.chat.completions.create(**kwargs)
        async for chunk in response_stream:
            # Capture usage from the final chunk (stream_options.include_usage)
            if chunk.usage:
                total_input_tokens = chunk.usage.prompt_tokens or 0
                total_output_tokens = chunk.usage.completion_tokens or 0

            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            content = delta.content if delta and delta.content else ""
            if content:
                yield LlmMessageChunk(
                    content=content,
                    role=Role.ASSISTANT,
                    index=next(chunk_index),
                    is_final=False,
                    response_metadata={},
                )

        if usage_callback and (total_input_tokens or total_output_tokens):
            try:
                usage_callback(total_input_tokens, total_output_tokens)
            except Exception as e:
                logger.warning(f"Usage callback error: {e}")

        yield LlmMessageChunk(
            content="",
            role=Role.ASSISTANT,
            index=next(chunk_index),
            is_final=True,
            response_metadata={
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
            },
        )
