"""
Native Google Gemini client using google-genai SDK.
"""
from __future__ import annotations

import itertools
import logging
from typing import (
    Any,
    AsyncIterator,
    Iterator,
    List,
    Union,
    Optional,
    Dict,
    Tuple,
)

from google import genai
from google.genai import types

from llming_lodge.llm_base_client import LlmClient
from llming_lodge.messages import (
    LlmAIMessage,
    LlmHumanMessage,
    LlmSystemMessage,
    LlmMessageChunk,
)
from llming_lodge.llm_base_models import Role

logger = logging.getLogger(__name__)


def _build_contents(
    messages: List[Union[LlmSystemMessage, LlmHumanMessage, LlmAIMessage]],
) -> Tuple[Optional[str], List[types.Content]]:
    """Convert internal messages to google-genai Content format.

    Returns:
        Tuple of (system_instruction, list of Content objects)
    """
    system_instruction = None
    contents: List[types.Content] = []

    for m in messages:
        if isinstance(m, LlmSystemMessage):
            system_instruction = m.content
        elif isinstance(m, LlmHumanMessage):
            parts = [types.Part.from_text(text=m.content)]
            if m.images:
                import base64
                for img_base64 in m.images:
                    # Detect image type
                    if img_base64.startswith("/9j/"):
                        mime_type = "image/jpeg"
                    elif img_base64.startswith("iVBOR"):
                        mime_type = "image/png"
                    else:
                        mime_type = "image/png"
                    parts.append(types.Part.from_bytes(
                        data=base64.b64decode(img_base64),
                        mime_type=mime_type,
                    ))
            contents.append(types.Content(role="user", parts=parts))
        elif isinstance(m, LlmAIMessage):
            contents.append(types.Content(
                role="model",
                parts=[types.Part.from_text(text=m.content)],
            ))

    return system_instruction, contents


def _extract_text_and_images(response_parts) -> Tuple[str, List[str]]:
    """Extract text and base64 images from response parts."""
    import base64

    text_parts = []
    images = []

    for part in response_parts:
        if hasattr(part, "text") and part.text:
            text_parts.append(part.text)
        elif hasattr(part, "inline_data") and part.inline_data:
            data = part.inline_data.data
            mime = getattr(part.inline_data, "mime_type", "image/png")
            if isinstance(data, bytes):
                b64 = base64.b64encode(data).decode("utf-8")
            else:
                b64 = data
            images.append(f"data:{mime};base64,{b64}")
            logger.debug(f"[GOOGLE] Extracted inline image ({len(b64)} chars)")

    return "".join(text_parts), images


class GoogleClient(LlmClient):
    """Native Google Gemini client using google-genai SDK."""

    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        streaming: bool = False,
    ) -> None:
        super().__init__(model, temperature, max_tokens, streaming)
        self._api_key = api_key
        self._client = genai.Client(api_key=api_key)

    def _build_config(self) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
        )

    # ── synchronous invoke ──────────────────────────────────────────────── #

    def invoke(
        self,
        messages: List[Union[LlmSystemMessage, LlmHumanMessage, LlmAIMessage]],
    ) -> LlmAIMessage:
        system_instruction, contents = _build_contents(messages)
        config = self._build_config()

        response = self._client.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=config.temperature,
                max_output_tokens=config.max_output_tokens,
                system_instruction=system_instruction,
            ),
        )

        text = ""
        images = []
        if response.candidates and response.candidates[0].content:
            text, images = _extract_text_and_images(response.candidates[0].content.parts)

        metadata = {}
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            metadata = {
                "input_tokens": getattr(response.usage_metadata, "prompt_token_count", 0),
                "output_tokens": getattr(response.usage_metadata, "candidates_token_count", 0),
            }

        return LlmAIMessage(content=text, images=images or None, response_metadata=metadata)

    # ── async invoke ────────────────────────────────────────────────────── #

    async def ainvoke(
        self,
        messages: List[Union[LlmSystemMessage, LlmHumanMessage, LlmAIMessage]],
    ) -> LlmAIMessage:
        system_instruction, contents = _build_contents(messages)
        config = self._build_config()

        response = await self._client.aio.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=config.temperature,
                max_output_tokens=config.max_output_tokens,
                system_instruction=system_instruction,
            ),
        )

        text = ""
        images = []
        if response.candidates and response.candidates[0].content:
            text, images = _extract_text_and_images(response.candidates[0].content.parts)

        metadata = {}
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            metadata = {
                "input_tokens": getattr(response.usage_metadata, "prompt_token_count", 0),
                "output_tokens": getattr(response.usage_metadata, "candidates_token_count", 0),
            }

        return LlmAIMessage(content=text, images=images or None, response_metadata=metadata)

    # ── synchronous streaming ───────────────────────────────────────────── #

    def stream(
        self,
        messages: List[Union[LlmSystemMessage, LlmHumanMessage, LlmAIMessage]],
    ) -> Iterator[LlmMessageChunk]:
        system_instruction, contents = _build_contents(messages)
        config = self._build_config()
        chunk_index = itertools.count()

        for chunk in self._client.models.generate_content_stream(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=config.temperature,
                max_output_tokens=config.max_output_tokens,
                system_instruction=system_instruction,
            ),
        ):
            if chunk.candidates and chunk.candidates[0].content:
                text, images = _extract_text_and_images(chunk.candidates[0].content.parts)
                yield LlmMessageChunk(
                    content=text,
                    role=Role.ASSISTANT,
                    index=next(chunk_index),
                    is_final=False,
                    response_metadata={},
                    images=images if images else None,
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
        system_instruction, contents = _build_contents(messages)
        config = self._build_config()
        chunk_index = itertools.count()

        async for chunk in await self._client.aio.models.generate_content_stream(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=config.temperature,
                max_output_tokens=config.max_output_tokens,
                system_instruction=system_instruction,
            ),
        ):
            if chunk.candidates and chunk.candidates[0].content:
                text, images = _extract_text_and_images(chunk.candidates[0].content.parts)
                if text or images:
                    yield LlmMessageChunk(
                        content=text,
                        role=Role.ASSISTANT,
                        index=next(chunk_index),
                        is_final=False,
                        response_metadata={},
                        images=images if images else None,
                    )

        # Note: google-genai doesn't provide per-chunk usage in streaming,
        # usage_callback is not called here
        yield LlmMessageChunk(
            content="",
            role=Role.ASSISTANT,
            index=next(chunk_index),
            is_final=True,
            response_metadata={},
        )
