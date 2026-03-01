"""
OpenAI LLM client implementation (Responses API edition).
"""
from __future__ import annotations

import asyncio
import functools
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
)

from openai import OpenAI, AsyncOpenAI, AzureOpenAI, AsyncAzureOpenAI
from openai.types.responses import (
    ResponseCompletedEvent,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseUsage,
    ResponseIncompleteEvent,
    # Reasoning model events (gpt-5, o1, etc.)
    ResponseReasoningTextDeltaEvent,
    ResponseReasoningTextDoneEvent,
    ResponseReasoningSummaryTextDeltaEvent,
    ResponseReasoningSummaryTextDoneEvent,
    ResponseContentPartAddedEvent,
    ResponseContentPartDoneEvent,
)

import json

logger = logging.getLogger(__name__)

from llming_lodge.llm_base_client import LlmClient
from llming_lodge.messages import (
    LlmAIMessage,
    LlmHumanMessage,
    LlmSystemMessage,
    LlmMessageChunk,
)
from llming_lodge.tools.tool_call import ToolCallInfo, ToolCallStatus
from llming_lodge.tools.llm_toolbox import LlmToolbox
from llming_lodge.tools.llm_tool import LlmTool
from llming_lodge.providers.llm_provider_models import ReasoningEffort


def _convert_messages(
    messages: List[Union[LlmSystemMessage, LlmHumanMessage, LlmAIMessage]],
    max_image_history: int = 10
) -> List[Dict[str, Any]]:
    """
    Convert internal message objects into the format expected by
    the Responses API:  [{"role": "...", "content": "..."}]

    For messages with images, converts to multimodal format:
    {"role": "user", "content": [
        {"type": "text", "text": "..."},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
    ]}

    Args:
        messages: List of message objects
        max_image_history: Maximum number of recent messages to include images from.
                          Older messages will have their images stripped.
    """
    role_map = {
        LlmSystemMessage: "system",
        LlmHumanMessage: "user",
        LlmAIMessage: "assistant",
    }

    # Count messages with images from the end to determine which get images
    messages_with_images_indices = set()
    image_count = 0
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        # Check both user and assistant messages for images
        if hasattr(msg, 'images') and msg.images:
            if image_count < max_image_history:
                messages_with_images_indices.add(i)
                image_count += 1

    converted: List[Dict[str, Any]] = []
    for i, m in enumerate(messages):
        role = role_map[type(m)]

        # Check if this message has images that should be included
        has_images = hasattr(m, 'images') and m.images and i in messages_with_images_indices

        if has_images:
            # Multimodal format for Responses API
            # Use different content types based on role:
            # - User messages: input_text, input_image
            # - Assistant messages: output_text (images added as user context in next turn)
            if role == "assistant":
                # For assistant messages with generated images, add them as a follow-up
                # user message showing what was generated (API limitation)
                converted.append({"role": role, "content": m.content})
                # Add the generated image as a user context message
                image_content = [{"type": "input_text", "text": "[Previously generated image]"}]
                for img_base64 in m.images:
                    if img_base64.startswith("/9j/"):
                        media_type = "image/jpeg"
                    elif img_base64.startswith("iVBOR"):
                        media_type = "image/png"
                    else:
                        media_type = "image/png"
                    image_content.append({
                        "type": "input_image",
                        "image_url": f"data:{media_type};base64,{img_base64}"
                    })
                converted.append({"role": "user", "content": image_content})
            else:
                # User messages use input_text and input_image
                content_parts = [{"type": "input_text", "text": m.content}]
                for img_base64 in m.images:
                    if img_base64.startswith("/9j/"):
                        media_type = "image/jpeg"
                    elif img_base64.startswith("iVBOR"):
                        media_type = "image/png"
                    else:
                        media_type = "image/png"
                    content_parts.append({
                        "type": "input_image",
                        "image_url": f"data:{media_type};base64,{img_base64}"
                    })
                converted.append({"role": role, "content": content_parts})
        else:
            # Simple text format
            converted.append({"role": role, "content": m.content})

    return converted


class OpenAILlmClient(LlmClient):
    """
    OpenAI Responses-API based implementation of `LlmClient`.
    """

    def __init__(
        self,
        api_key: Optional[str],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        streaming: bool = False,
        base_url: Optional[str] = None,
        toolboxes: Optional[List[LlmToolbox]] = None,
        api_type: Optional[str] = None,
        api_version: Optional[str] = None,
        reasoning_effort: Optional[ReasoningEffort] = None,
    ) -> None:
        import os
        super().__init__(model, temperature, max_tokens, streaming)
        self.toolboxes = toolboxes or []
        self.reasoning_effort = reasoning_effort

        # Fallback to environment variables if not set
        api_type = api_type or os.environ.get("OPENAI_API_TYPE", "openai").lower()
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        api_version = api_version or os.environ.get("OPENAI_API_VERSION")

        if not api_key:
            raise ValueError("OpenAI API key must be provided either as argument or OPENAI_API_KEY env var.")

        self.api_type = api_type
        self.api_version = api_version

        # Azure support
        if api_type == "azure":
            base_url = base_url or os.environ.get("AZURE_OPENAI_ENDPOINT")
            self._client = AzureOpenAI(
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=base_url,
            )
            self._aclient = AsyncAzureOpenAI(
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=base_url,
            )
        else:
            base_url = base_url or os.environ.get("OPENAI_API_BASE")
            self._client = OpenAI(
                api_key=api_key,
                base_url=base_url,  # None → default api.openai.com
            )
            self._aclient = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
            )

    @property
    def client(self) -> OpenAI | AzureOpenAI:
        return self._client

    @property
    def aclient(self) -> AsyncOpenAI | AsyncAzureOpenAI:
        return self._aclient

    def _get_reasoning_kwargs(self) -> dict:
        """Build reasoning configuration for API calls.

        Returns a dict with the reasoning parameter if configured, otherwise empty.
        For GPT-5 models:
        - NONE: Maps to 'minimal' (gpt-5 models don't support 'none', lowest is 'minimal')
        - MINIMAL/LOW/MEDIUM/HIGH: Sets reasoning effort level

        Note: When tools are present (e.g., web_search), 'minimal' reasoning is not supported.
        In that case, reasoning is automatically increased to 'low'.
        """
        if self.reasoning_effort is None:
            return {}

        # Map NONE to 'minimal' because gpt-5 models don't support 'none'
        # Valid values for gpt-5-nano: 'minimal', 'low', 'medium', 'high'
        effort_value = self.reasoning_effort.value
        if self.reasoning_effort == ReasoningEffort.NONE:
            effort_value = "minimal"

        # When tools are present, 'minimal' reasoning is not supported by OpenAI.
        # Automatically increase to 'low' to maintain tool compatibility.
        if effort_value == "minimal" and self._build_tools_list():
            effort_value = "low"

        return {"reasoning": {"effort": effort_value}}

    @staticmethod
    def _enforce_strict_schema(schema: dict) -> dict:
        """Recursively enforce OpenAI strict-mode rules on a JSON schema.

        Handles all strict-mode requirements:
        - Every schema node must have a ``type`` key
        - Every object must have ``additionalProperties: false``
        - Every object must have ``properties`` and ``required`` listing all keys
        - Every array must have ``items``
        - ``default`` values are moved into the description
        - ``enum`` values are removed from non-string types if needed
        """
        _recurse = OpenAILlmClient._enforce_strict_schema
        schema = dict(schema)
        typ = schema.get("type")

        # Ensure every node has a type (required by strict mode)
        if not typ and "anyOf" not in schema and "oneOf" not in schema and "allOf" not in schema:
            schema["type"] = "string"
            typ = "string"

        if typ == "object":
            schema["additionalProperties"] = False
            if "properties" not in schema:
                schema["properties"] = {}
            schema["required"] = list(schema["properties"].keys())
            schema["properties"] = {
                k: _recurse(dict(v)) for k, v in schema["properties"].items()
            }

        elif typ == "array":
            if "items" not in schema:
                schema["items"] = {"type": "string"}
            schema["items"] = _recurse(dict(schema["items"]))

        # anyOf / oneOf / allOf
        for combo_key in ("anyOf", "oneOf", "allOf"):
            if combo_key in schema:
                schema[combo_key] = [_recurse(dict(s)) for s in schema[combo_key]]

        # Move default to description
        if "default" in schema:
            default_val = schema.pop("default")
            if isinstance(default_val, bool):
                default_str = str(default_val).lower()
            elif isinstance(default_val, str):
                default_str = f'"{default_val}"' if default_val else "empty"
            else:
                default_str = str(default_val)
            desc = schema.get("description", "")
            schema["description"] = f"{desc} (default: {default_str})"

        return schema

    def _build_tools_list(self) -> List[Dict[str, Any]]:
        """Build tools list from toolboxes for API calls.

        Handles both string tools (e.g., "web_search") and LlmToolbox objects.
        """
        logger.debug(f"[OPENAI] _build_tools_list: {len(self.toolboxes)} toolboxes")

        tools_list = []
        for toolbox in self.toolboxes:
            # Handle string tools directly (e.g., "web_search")
            if isinstance(toolbox, str):
                if toolbox == "web_search":
                    tools_list.append({"type": "web_search"})
                continue

            # Handle LlmToolbox objects
            for tool in toolbox.tools:
                if isinstance(tool, str) and tool == "web_search":
                    tools_list.append({"type": "web_search"})
                elif isinstance(tool, dict):
                    # Provider-native tool as dict (e.g. {"type": "web_search", ...})
                    tools_list.append(tool)
                elif isinstance(tool, LlmTool):
                    # Recursively enforce OpenAI strict-mode rules on the schema
                    params = self._enforce_strict_schema(dict(tool.parameters))

                    tool_schema = {
                        "type": "function",
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": params,
                        "strict": True
                    }
                    tools_list.append(tool_schema)

        tool_names = [t.get('name') or t.get('type') for t in tools_list]
        logger.debug(f"[OPENAI] Built tools list ({len(tool_names)}): {tool_names}")
        return tools_list

    def _has_web_search(self) -> bool:
        """Check if web_search tool is enabled."""
        for toolbox in self.toolboxes:
            if isinstance(toolbox, str) and toolbox == "web_search":
                return True
            if hasattr(toolbox, 'tools'):
                for tool in toolbox.tools:
                    if isinstance(tool, str) and tool == "web_search":
                        return True
                    if isinstance(tool, dict) and tool.get("type") == "web_search":
                        return True
        return False

    def _is_reasoning_model(self) -> bool:
        """Check if the current model is a GPT-5 reasoning model.

        GPT-5 models don't support the temperature parameter.
        """
        return self.model.startswith("gpt-5") or self.model.startswith("o1") or self.model.startswith("o3") or self.model.startswith("o4")

    def _skip_temperature(self) -> bool:
        """Check if temperature parameter should be skipped.

        Temperature is not supported for:
        - Web search tool
        - GPT-5 reasoning models (gpt-5-mini, gpt-5-nano, gpt-5.2)
        - o1/o3 reasoning models
        """
        return self._has_web_search() or self._is_reasoning_model()

    def _build_tool_func_map(self) -> Dict[str, Any]:
        """Build a map of tool names to their callable functions."""
        tool_func_map = {}
        for toolbox in self.toolboxes:
            if isinstance(toolbox, str):
                continue  # String tools don't have callables
            if hasattr(toolbox, 'tools'):
                for tool in toolbox.tools:
                    if isinstance(tool, LlmTool) and callable(tool.func):
                        tool_func_map[tool.name] = tool.func
        return tool_func_map

    # --------------------------------------------------------------------- #
    # synchronous (single-shot)                                             #
    # --------------------------------------------------------------------- #
    def invoke(
        self,
        messages: List[Union[LlmSystemMessage, LlmHumanMessage, LlmAIMessage]],
    ) -> LlmAIMessage:
        # Use correct param for Azure (engine), OpenAI (model)
        create_kwargs = dict(
            input=_convert_messages(messages),
            max_output_tokens=self.max_tokens,
            store=False,  # stateless call – we don't need OpenAI to keep history
            **self._get_reasoning_kwargs(),  # Add reasoning configuration
        )

        # Temperature not supported with web_search or reasoning models (gpt-5, o1, o3)
        if not self._skip_temperature():
            create_kwargs["temperature"] = self.temperature

        if self.api_type == "azure":
            create_kwargs["model"] = self.model
        else:
            create_kwargs["model"] = self.model

        # Build tools array from toolboxes
        tools_list = self._build_tools_list()
        if tools_list:
            create_kwargs["tools"] = tools_list

        response = self._client.responses.create(**create_kwargs)

        # Handle function call output or text
        # GPT-5 may return multiple output items (reasoning + message)
        visible_text = ""
        function_call_result = None
        function_name = None

        for output_item in response.output:
            item_type = getattr(output_item, "type", None)

            # Skip reasoning items (GPT-5 internal thinking)
            if item_type == "reasoning":
                continue

            # Handle message type with content
            if item_type == "message" and hasattr(output_item, "content") and output_item.content is not None:
                visible_text += "".join(
                    part.text
                    for part in output_item.content
                    if hasattr(part, "text")
                )
            # Handle legacy content structure (non-typed)
            elif hasattr(output_item, "content") and output_item.content is not None:
                visible_text += "".join(
                    part.text
                    for part in output_item.content
                    if hasattr(part, "text")
                )
            # Handle function calls
            elif item_type == "function_call":
                function_name = getattr(output_item, "name", None)
                args = json.loads(getattr(output_item, "arguments", "{}"))
                for toolbox in self.toolboxes:
                    for tool in toolbox.tools:
                        if isinstance(tool, LlmTool) and tool.name == function_name:
                            func = tool.func
                            if callable(func):
                                function_call_result = func(**args)

        if function_name is not None:
            return LlmAIMessage(
                content=json.dumps({"function_call_result": function_call_result, "function": function_name}),
                response_metadata=response.usage.model_dump(),
            )

        return LlmAIMessage(
            content=visible_text,
            response_metadata=response.usage.model_dump(),
        )

    # --------------------------------------------------------------------- #
    # asynchronous (single-shot)                                            #
    # --------------------------------------------------------------------- #
    async def ainvoke(
        self,
        messages: List[Union[LlmSystemMessage, LlmHumanMessage, LlmAIMessage]],
    ) -> LlmAIMessage:
        # Use correct param for Azure (engine), OpenAI (model)
        create_kwargs = dict(
            input=_convert_messages(messages),
            max_output_tokens=self.max_tokens,
            store=False,
            **self._get_reasoning_kwargs(),  # Add reasoning configuration
        )

        # Temperature not supported with web_search or reasoning models (gpt-5, o1, o3)
        if not self._skip_temperature():
            create_kwargs["temperature"] = self.temperature

        if self.api_type == "azure":
            create_kwargs["model"] = self.model
        else:
            create_kwargs["model"] = self.model

        # Build tools array from toolboxes
        tools_list = self._build_tools_list()
        if tools_list:
            create_kwargs["tools"] = tools_list

        try:
            response = await self._aclient.responses.create(**create_kwargs)
        except Exception as api_err:
            if "invalid_function_parameters" in str(api_err).lower() or "invalid schema" in str(api_err).lower():
                # Log the offending tool schemas for debugging
                import json as _json
                for t in tools_list:
                    if t.get("name") and t.get("parameters"):
                        logger.debug(f"[OPENAI] Tool schema '{t['name']}': {_json.dumps(t['parameters'], indent=2)[:500]}")
            raise

        # Handle function call output or text
        # GPT-5 may return multiple output items (reasoning + message)
        visible_text = ""
        function_call_result = None
        function_name = None

        for output_item in response.output:
            item_type = getattr(output_item, "type", None)

            # Skip reasoning items (GPT-5 internal thinking)
            if item_type == "reasoning":
                continue

            # Handle message type with content
            if item_type == "message" and hasattr(output_item, "content") and output_item.content is not None:
                visible_text += "".join(
                    part.text
                    for part in output_item.content
                    if hasattr(part, "text")
                )
            # Handle legacy content structure (non-typed)
            elif hasattr(output_item, "content") and output_item.content is not None:
                visible_text += "".join(
                    part.text
                    for part in output_item.content
                    if hasattr(part, "text")
                )
            # Handle function calls
            elif item_type == "function_call":
                function_name = getattr(output_item, "name", None)
                args = json.loads(getattr(output_item, "arguments", "{}"))
                for toolbox in self.toolboxes:
                    for tool in toolbox.tools:
                        if isinstance(tool, LlmTool) and tool.name == function_name:
                            func = tool.func
                            if callable(func):
                                function_call_result = func(**args)

        if function_name is not None:
            return LlmAIMessage(
                content=json.dumps({"function_call_result": function_call_result, "function": function_name}),
                response_metadata=response.usage.model_dump(),
            )

        return LlmAIMessage(
            content=visible_text,
            response_metadata=response.usage.model_dump(),
        )

    # --------------------------------------------------------------------- #
    # synchronous streaming                                                 #
    # --------------------------------------------------------------------- #
    def stream(
        self,
        messages: List[Union[LlmSystemMessage, LlmHumanMessage, LlmAIMessage]],
    ) -> Iterator[LlmMessageChunk]:
        """
        Yield `LlmMessageChunk`s as `response.text.delta` events arrive,
        and emit one final empty chunk when `response.completed` arrives
        to let the caller know the stream is finished.
        """
        stream_kwargs = dict(
            model=self.model,
            input=_convert_messages(messages),
            max_output_tokens=self.max_tokens,
            **self._get_reasoning_kwargs(),  # Add reasoning configuration
        )

        # Temperature not supported with web_search or reasoning models (gpt-5, o1, o3)
        if not self._skip_temperature():
            stream_kwargs["temperature"] = self.temperature

        # Build tools array from toolboxes
        tools_list = self._build_tools_list()
        if tools_list:
            stream_kwargs["tools"] = tools_list

        with self._client.responses.stream(**stream_kwargs) as stream:
            chunk_index = itertools.count()
            tool_json_fragments = {}
            function_call_info = {}
            for event in stream:
                if isinstance(event, ResponseTextDeltaEvent):
                    event_data = event.dict() if hasattr(event, 'dict') else event.model_dump() if hasattr(event, 'model_dump') else {}
                    content = event_data.get('text') or event_data.get('value') or next((v for v in event_data.values() if isinstance(v, str)), '')
                    yield LlmMessageChunk(
                        content=content,          # the delta text
                        role="assistant",
                        index=next(chunk_index),
                        is_final=False,
                        response_metadata={},         # responses API includes usage later
                    )
                elif isinstance(event, ResponseOutputItemAddedEvent):
                    idx = event.output_index
                    tool_json_fragments[idx] = ""
                    if hasattr(event, "item") and hasattr(event.item, "name"):
                        function_call_info[idx] = {
                            "name": event.item.name,
                        }
                elif isinstance(event, ResponseFunctionCallArgumentsDeltaEvent):
                    tool_json_fragments[event.output_index] += event.delta
                elif isinstance(event, ResponseFunctionCallArgumentsDoneEvent):
                    idx = event.output_index
                    args = json.loads(tool_json_fragments.pop(idx))
                    fn_name = function_call_info[idx]["name"]
                    call_id = function_call_info[idx].get("call_id", "")

                    # Yield pending state
                    yield LlmMessageChunk(
                        content="",
                        role="assistant",
                        index=next(chunk_index),
                        is_final=False,
                        response_metadata={},
                        tool_call=ToolCallInfo(
                            name=fn_name,
                            call_id=call_id,
                            status=ToolCallStatus.PENDING,
                            arguments=args,
                        ),
                    )

                    # Execute the tool
                    result = None
                    error_msg = None
                    for toolbox in self.toolboxes:
                        for tool in toolbox.tools:
                            if isinstance(tool, LlmTool) and tool.name == fn_name:
                                func = tool.func
                                if callable(func):
                                    try:
                                        result = func(**args)
                                    except Exception as e:
                                        logger.error(f"Tool execution error for {fn_name}: {e}")
                                        error_msg = str(e)

                    # Yield completed state
                    yield LlmMessageChunk(
                        content="",
                        role="assistant",
                        index=next(chunk_index),
                        is_final=False,
                        response_metadata={},
                        tool_call=ToolCallInfo(
                            name=fn_name,
                            call_id=call_id,
                            status=ToolCallStatus.ERROR if error_msg else ToolCallStatus.COMPLETED,
                            arguments=args,
                            result=result,
                            error=error_msg,
                        ),
                    )
                elif isinstance(event, (ResponseTextDoneEvent, ResponseCompletedEvent)):
                    # Mark the end of the stream
                    yield LlmMessageChunk(
                        content="",
                        role="assistant",
                        index=next(chunk_index),
                        is_final=True,
                        response_metadata={},
                    )
                    break  # safety
                # Handle reasoning model events (gpt-5, o1, etc.) - skip reasoning tokens
                elif isinstance(event, (ResponseReasoningTextDeltaEvent, ResponseReasoningTextDoneEvent,
                                       ResponseReasoningSummaryTextDeltaEvent, ResponseReasoningSummaryTextDoneEvent,
                                       ResponseContentPartAddedEvent, ResponseContentPartDoneEvent)):
                    pass  # Reasoning events - consume but don't yield

    # --------------------------------------------------------------------- #
    # asynchronous streaming                                                #
    # --------------------------------------------------------------------- #
    async def astream(
        self,
        messages: list,
        usage_callback: Optional[callable] = None,
    ) -> AsyncIterator[LlmMessageChunk]:
        """Stream responses with auto-continue after tool execution.

        Args:
            messages: List of messages to send
            usage_callback: Optional callback(input_tokens, output_tokens) called after each API iteration
        """
        from typing import Callable
        # Build tools array from toolboxes
        tools_list = self._build_tools_list()
        tool_func_map = self._build_tool_func_map()

        # Build conversation input - will be extended with tool results
        conversation_input = _convert_messages(messages)

        chunk_index = itertools.count()
        max_tool_iterations = 10
        iteration = 0

        # Track cumulative usage across all iterations
        total_input_tokens = 0
        total_output_tokens = 0

        while iteration < max_tool_iterations:
            iteration += 1

            stream_kwargs = dict(
                model=self.model,
                input=conversation_input,
                max_output_tokens=self.max_tokens,
                **self._get_reasoning_kwargs(),
            )

            # Temperature not supported with web_search or reasoning models (gpt-5, o1, o3)
            if not self._skip_temperature():
                stream_kwargs["temperature"] = self.temperature

            if tools_list:
                stream_kwargs["tools"] = tools_list

            tool_json_fragments = {}
            function_call_info = {}
            executed_tools = []  # Track tools executed in this iteration
            iteration_usage = None  # Track usage for this iteration

            async with self._aclient.responses.stream(**stream_kwargs) as stream:
                async for event in stream:
                    # Text tokens
                    if isinstance(event, ResponseTextDeltaEvent):
                        event_data = event.model_dump() if hasattr(event, 'model_dump') else event.dict() if hasattr(event, 'dict') else {}
                        content = event_data.get('delta') or event_data.get('text') or getattr(event, 'delta', '') or getattr(event, 'text', '')
                        if content:
                            yield LlmMessageChunk(
                                content=content,
                                role="assistant",
                                index=next(chunk_index),
                                is_final=False,
                                response_metadata={},
                            )
                    # New output item (may be a tool-call shell)
                    elif isinstance(event, ResponseOutputItemAddedEvent):
                        idx = event.output_index
                        tool_json_fragments[idx] = ""
                        if hasattr(event, "item") and hasattr(event.item, "name"):
                            function_call_info[idx] = {
                                "name": event.item.name,
                                "call_id": event.item.call_id,
                            }
                    # Streaming JSON arguments
                    elif isinstance(event, ResponseFunctionCallArgumentsDeltaEvent):
                        tool_json_fragments[event.output_index] += event.delta
                    # JSON closed — execute the Python function
                    elif isinstance(event, ResponseFunctionCallArgumentsDoneEvent):
                        idx = event.output_index
                        args = json.loads(tool_json_fragments.pop(idx))
                        fn_name = function_call_info[idx]["name"]
                        call_id = function_call_info[idx]["call_id"]

                        # Yield a "pending" chunk with structured tool info
                        yield LlmMessageChunk(
                            content="",
                            role="assistant",
                            index=next(chunk_index),
                            is_final=False,
                            response_metadata={},
                            tool_call=ToolCallInfo(
                                name=fn_name,
                                call_id=call_id,
                                status=ToolCallStatus.PENDING,
                                arguments=args,
                            ),
                        )

                        # Give UI a chance to render the pending state before blocking on function execution
                        await asyncio.sleep(0)

                        # Execute the function (may take time for image generation, web search, etc.)
                        # Run in executor to avoid blocking the event loop - this is critical for MCP tools
                        # which use run_coroutine_threadsafe to execute async operations on this same loop
                        result = None
                        error_msg = None
                        if fn_name in tool_func_map:
                            func = tool_func_map[fn_name]
                            if callable(func):
                                try:
                                    loop = asyncio.get_running_loop()
                                    result = await loop.run_in_executor(
                                        None, functools.partial(func, **args)
                                    )
                                except Exception as e:
                                    logger.error(f"Tool execution error for {fn_name}: {e}")
                                    error_msg = str(e)

                        # Track executed tool for conversation continuation
                        executed_tools.append({
                            "name": fn_name,
                            "call_id": call_id,
                            "arguments": args,
                            "result": result,
                            "error": error_msg,
                        })

                        # Yield the result with structured tool info
                        yield LlmMessageChunk(
                            content="",
                            role="assistant",
                            index=next(chunk_index),
                            is_final=False,
                            response_metadata={},
                            tool_call=ToolCallInfo(
                                name=fn_name,
                                call_id=call_id,
                                status=ToolCallStatus.ERROR if error_msg else ToolCallStatus.COMPLETED,
                                arguments=args,
                                result=result,
                                error=error_msg,
                            ),
                        )
                    elif isinstance(event, ResponseCompletedEvent):
                        # Capture usage from completed response (including cache details)
                        if hasattr(event, 'response') and hasattr(event.response, 'usage'):
                            usage = event.response.usage
                            # OpenAI reports cached tokens in input_tokens_details
                            cached = 0
                            details = getattr(usage, 'input_tokens_details', None)
                            if details:
                                cached = getattr(details, 'cached_tokens', 0) or 0
                            iteration_usage = {
                                'input_tokens': getattr(usage, 'input_tokens', 0),
                                'output_tokens': getattr(usage, 'output_tokens', 0),
                                'cached_input_tokens': cached,
                            }
                    # Reasoning model events (gpt-5, o1, etc.) - consume but don't yield
                    elif isinstance(event, (ResponseReasoningTextDeltaEvent, ResponseReasoningTextDoneEvent,
                                           ResponseReasoningSummaryTextDeltaEvent, ResponseReasoningSummaryTextDoneEvent,
                                           ResponseContentPartAddedEvent, ResponseContentPartDoneEvent,
                                           ResponseOutputItemDoneEvent, ResponseTextDoneEvent, ResponseUsage)):
                        pass
                    elif isinstance(event, ResponseIncompleteEvent):
                        # Capture usage even from incomplete responses
                        if hasattr(event, 'response') and hasattr(event.response, 'usage'):
                            usage = event.response.usage
                            cached = 0
                            details = getattr(usage, 'input_tokens_details', None)
                            if details:
                                cached = getattr(details, 'cached_tokens', 0) or 0
                            iteration_usage = {
                                'input_tokens': getattr(usage, 'input_tokens', 0),
                                'output_tokens': getattr(usage, 'output_tokens', 0),
                                'cached_input_tokens': cached,
                            }
                        reason = getattr(event, 'reason', 'unknown')
                        logger.warning(f"Response incomplete: {reason}")

                # Also try to get usage from stream's final response if not already captured
                if not iteration_usage:
                    try:
                        final_response = await stream.get_final_response()
                        if hasattr(final_response, 'usage') and final_response.usage:
                            cached = 0
                            details = getattr(final_response.usage, 'input_tokens_details', None)
                            if details:
                                cached = getattr(details, 'cached_tokens', 0) or 0
                            iteration_usage = {
                                'input_tokens': getattr(final_response.usage, 'input_tokens', 0),
                                'output_tokens': getattr(final_response.usage, 'output_tokens', 0),
                                'cached_input_tokens': cached,
                            }
                    except Exception:
                        pass  # Stream may not support get_final_response

            # Track cumulative usage from this iteration
            if iteration_usage:
                total_input_tokens += iteration_usage['input_tokens']
                total_output_tokens += iteration_usage['output_tokens']
                iter_cached = iteration_usage.get('cached_input_tokens', 0)

                if iter_cached:
                    logger.info(f"[OPENAI CACHE] cached={iter_cached} of {iteration_usage['input_tokens']} input tokens")

                # Call usage callback if provided
                if usage_callback:
                    try:
                        usage_callback(iteration_usage['input_tokens'],
                                       iteration_usage['output_tokens'],
                                       cached_input_tokens=iter_cached)
                    except Exception as e:
                        logger.warning(f"Usage callback error: {e}")

            # After stream ends, check if we need to continue with tool results
            if executed_tools:
                # Add function calls and their outputs to conversation for next iteration
                # OpenAI Responses API requires both the function_call and function_call_output
                from llming_lodge.utils.image_utils import is_likely_image_data

                for tool in executed_tools:
                    # First add the function call (as returned by the model)
                    conversation_input.append({
                        "type": "function_call",
                        "call_id": tool["call_id"],
                        "name": tool["name"],
                        "arguments": json.dumps(tool["arguments"]),
                    })
                    # Then add the function call output
                    # IMPORTANT: Don't send base64 images back to the model - causes context overflow!
                    # Instead, send a text summary. The actual result is still available in tool["result"]
                    # for the UI/caller to display.
                    result_to_send = tool["result"]
                    if result_to_send is not None and isinstance(result_to_send, str) and is_likely_image_data(result_to_send):
                        # Yield the image data as a special chunk for the caller to display
                        yield LlmMessageChunk(
                            content=result_to_send,  # Base64 image data
                            role="assistant",
                            index=next(chunk_index),
                            is_final=False,
                            tool_call=ToolCallInfo(
                                call_id=tool["call_id"],
                                name=tool["name"],
                                arguments=tool["arguments"],
                                status=ToolCallStatus.COMPLETED
                            )
                        )
                        # Send a text summary to the model instead of the huge base64 data
                        result_to_send = f"[Image generated successfully]"
                        logger.debug(f"[OPENAI] Yielded image to caller, filtered from conversation (original: {len(tool['result'])} chars)")

                    result_str = json.dumps(result_to_send) if result_to_send is not None else "null"
                    conversation_input.append({
                        "type": "function_call_output",
                        "call_id": tool["call_id"],
                        "output": result_str,
                    })
                # Continue loop to get model's response to tool results
                continue
            else:
                # No tools executed, we're done
                break

        # Final chunk to mark stream end with total usage
        yield LlmMessageChunk(
            content="",
            role="assistant",
            index=next(chunk_index),
            is_final=True,
            response_metadata={
                'total_input_tokens': total_input_tokens,
                'total_output_tokens': total_output_tokens,
            },
        )

    async def generate_image(
        self,
        prompt: str,
        size: str = "1024x1024",
        quality: str = "medium",
        n: int = 1,
        model: str = "gpt-image-1"
    ) -> str:
        """
        Generate an image.

        Args:
            prompt: Text description of the image to generate
            size: Image size (e.g. "1024x1024", "1536x1024", "1024x1536")
            quality: "low", "medium", or "high" for gpt-image-1; "standard" or "hd" for dall-e-3
            n: Number of images to generate
            model: Model/deployment name

        Returns:
            Base64-encoded image data
        """
        kwargs = dict(model=model, prompt=prompt, size=size, quality=quality, n=n)
        # DALL-E models need explicit response_format; gpt-image-1 always returns base64
        if model.startswith("dall-e"):
            kwargs["response_format"] = "b64_json"
        response = await self._aclient.images.generate(**kwargs)
        return response.data[0].b64_json

    def generate_image_sync(
        self,
        prompt: str,
        size: str = "1024x1024",
        quality: str = "medium",
        n: int = 1,
        model: str = "gpt-image-1"
    ) -> str:
        """
        Generate an image (synchronous version).

        Args:
            prompt: Text description of the image to generate
            size: Image size (e.g. "1024x1024", "1536x1024", "1024x1536")
            quality: "low", "medium", or "high" for gpt-image-1; "standard" or "hd" for dall-e-3
            n: Number of images to generate
            model: Model/deployment name

        Returns:
            Base64-encoded image data
        """
        kwargs = dict(model=model, prompt=prompt, size=size, quality=quality, n=n)
        # DALL-E models need explicit response_format; gpt-image-1 always returns base64
        if model.startswith("dall-e"):
            kwargs["response_format"] = "b64_json"
        response = self._client.images.generate(**kwargs)
        return response.data[0].b64_json


if __name__ == "__main__":
    import asyncio
    import os
    import dotenv

    def find_env(max_level=10):
        start_dir = os.path.dirname(os.path.abspath(__file__))
        for i in range(max_level):            
            try:
                if os.path.exists(os.path.join(start_dir, ".env")):
                    dotenv.load_dotenv(start_dir+"/.env")
                    return
                start_dir = os.path.dirname(start_dir)
            except:
                break

    find_env()
    client = AzureOpenAI(
                api_key=os.environ.get("OPENAI_API_KEY"),
                api_version=os.environ.get("OPENAI_API_VERSION"),
            )    

        
    def test_completion():
        # test basic completion
        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "When was Microsoft founded?"}
            ]
        )
        print(f"[sync] {response.choices[0].message.content}")

    def test_responses_api():
        response = client.responses.create(
            model="gpt-5-mini",
            input=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "When was Microsoft founded?"}
            ]
        )
        print(f"[sync] {response.choices[0].message.content}")

    test_responses_api()

    exit()