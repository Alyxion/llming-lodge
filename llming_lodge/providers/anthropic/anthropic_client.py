"""
Native Anthropic LLM client implementation with image and tool support.
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

from anthropic import Anthropic, AsyncAnthropic

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

logger = logging.getLogger(__name__)


def _convert_messages(
    messages: List[Union[LlmSystemMessage, LlmHumanMessage, LlmAIMessage]],
    max_image_history: int = 10
) -> tuple[Optional[str], List[Dict[str, Any]]]:
    """
    Convert internal message objects into the format expected by Anthropic.

    For messages with images, converts to multimodal format:
    {"role": "user", "content": [
        {"type": "text", "text": "..."},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "..."}}
    ]}

    Args:
        messages: List of message objects
        max_image_history: Maximum number of recent messages to include images from.

    Returns:
        Tuple of (system_prompt, converted_messages)
    """
    # Extract system message if present
    system_prompt = None
    regular_messages = []
    for m in messages:
        if isinstance(m, LlmSystemMessage):
            system_prompt = m.content
        else:
            regular_messages.append(m)

    # Count messages with images from the end to determine which get images
    messages_with_images_indices = set()
    image_count = 0
    for i in range(len(regular_messages) - 1, -1, -1):
        msg = regular_messages[i]
        if isinstance(msg, LlmHumanMessage) and msg.images:
            if image_count < max_image_history:
                messages_with_images_indices.add(i)
                image_count += 1

    converted: List[Dict[str, Any]] = []
    for i, m in enumerate(regular_messages):
        if isinstance(m, LlmHumanMessage):
            role = "user"
        elif isinstance(m, LlmAIMessage):
            role = "assistant"
        else:
            continue  # Skip unknown message types

        # Check if this is a human message with images that should be included
        if isinstance(m, LlmHumanMessage) and m.images and i in messages_with_images_indices:
            # Multimodal format
            content_parts = [{"type": "text", "text": m.content}]
            for img_base64 in m.images:
                # Detect image type from base64 header or default to png
                if img_base64.startswith("/9j/"):
                    media_type = "image/jpeg"
                elif img_base64.startswith("iVBOR"):
                    media_type = "image/png"
                elif img_base64.startswith("R0lG"):
                    media_type = "image/gif"
                elif img_base64.startswith("UklG"):
                    media_type = "image/webp"
                else:
                    media_type = "image/png"  # Default

                content_parts.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": img_base64
                    }
                })
            converted.append({"role": role, "content": content_parts})
        else:
            # Simple text format
            converted.append({"role": role, "content": m.content})

    return system_prompt, converted


def _convert_tools(toolboxes: List[LlmToolbox]) -> List[Dict[str, Any]]:
    """Convert toolboxes to Anthropic tool format.

    Handles both regular function tools (LlmTool) and provider-native tools
    like web_search which use special Anthropic tool types.
    """
    tools = []
    for toolbox in toolboxes:
        for tool in toolbox.tools:
            if isinstance(tool, LlmTool):
                tool_schema = {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.parameters
                }
                tools.append(tool_schema)
            elif isinstance(tool, str):
                # Provider-native tool specified as string (e.g., "web_search")
                if tool == "web_search":
                    # Anthropic's native web search tool
                    tools.append({
                        "type": "web_search_20250305",
                        "name": "web_search",
                        "max_uses": 5,
                    })
                else:
                    logger.warning(f"Unknown native tool: {tool}")
            elif isinstance(tool, dict):
                # Provider-native tool with config (e.g., {"type": "web_search", ...})
                tool_type = tool.get("type", "")
                if tool_type == "web_search" or tool_type == "web_search_20250305":
                    # Anthropic's native web search tool with config
                    web_search_config = {
                        "type": "web_search_20250305",
                        "name": "web_search",
                    }
                    # Copy allowed config options
                    if "max_uses" in tool:
                        web_search_config["max_uses"] = tool["max_uses"]
                    if "allowed_domains" in tool:
                        web_search_config["allowed_domains"] = tool["allowed_domains"]
                    if "blocked_domains" in tool:
                        web_search_config["blocked_domains"] = tool["blocked_domains"]
                    if "user_location" in tool:
                        web_search_config["user_location"] = tool["user_location"]
                    tools.append(web_search_config)
                else:
                    logger.warning(f"Unknown native tool type: {tool_type}")
    return tools


class AnthropicClient(LlmClient):
    """
    Native Anthropic API client with image and tool support.
    """

    def __init__(
        self,
        api_key: Optional[str],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        streaming: bool = False,
        toolboxes: Optional[List[LlmToolbox]] = None,
    ):
        """
        Initialize Anthropic client.

        Args:
            api_key: Anthropic API key (or from ANTHROPIC_API_KEY env var)
            model: Model name (e.g., "claude-sonnet-4-5-20250929")
            temperature: Sampling temperature
            max_tokens: Maximum output tokens
            streaming: Whether to stream by default
            toolboxes: Optional list of tool collections
        """
        super().__init__(model, temperature, max_tokens, streaming)

        import os
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY must be provided")

        self._client = Anthropic(api_key=api_key)
        self._aclient = AsyncAnthropic(api_key=api_key)
        self.toolboxes = toolboxes or []

    def _build_kwargs(self, messages: list) -> Dict[str, Any]:
        """Build kwargs for API call."""
        system_prompt, converted = _convert_messages(messages)

        kwargs = {
            "model": self.model,
            "messages": converted,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens or 4096,
        }

        if system_prompt:
            kwargs["system"] = system_prompt

        # Add tools if available
        if self.toolboxes:
            tools = _convert_tools(self.toolboxes)
            if tools:
                kwargs["tools"] = tools

        return kwargs

    def _execute_tool(self, tool_name: str, tool_input: dict) -> Any:
        """Execute a tool and return its result."""
        for toolbox in self.toolboxes:
            for tool in toolbox.tools:
                if isinstance(tool, LlmTool) and tool.name == tool_name:
                    if callable(tool.func):
                        return tool.func(**tool_input)
        return None

    def invoke(
        self,
        messages: list[Union[LlmSystemMessage, LlmHumanMessage, LlmAIMessage]]
    ) -> LlmAIMessage:
        """Synchronously invoke the model."""
        kwargs = self._build_kwargs(messages)
        response = self._client.messages.create(**kwargs)

        # Handle tool use
        while response.stop_reason == "tool_use":
            # Find tool use blocks
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = self._execute_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result) if result is not None else "null"
                    })

            # Add assistant message and tool results
            kwargs["messages"].append({
                "role": "assistant",
                "content": [{"type": b.type, **({"text": b.text} if b.type == "text" else {"id": b.id, "name": b.name, "input": b.input})} for b in response.content]
            })
            kwargs["messages"].append({
                "role": "user",
                "content": tool_results
            })

            response = self._client.messages.create(**kwargs)

        # Extract text content
        text_content = ""
        for block in response.content:
            if hasattr(block, "text"):
                text_content += block.text

        return LlmAIMessage(
            content=text_content,
            response_metadata={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "model": response.model,
                "stop_reason": response.stop_reason,
            }
        )

    async def ainvoke(
        self,
        messages: list[Union[LlmSystemMessage, LlmHumanMessage, LlmAIMessage]]
    ) -> LlmAIMessage:
        """Asynchronously invoke the model."""
        kwargs = self._build_kwargs(messages)
        response = await self._aclient.messages.create(**kwargs)

        # Handle tool use
        while response.stop_reason == "tool_use":
            # Find tool use blocks
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = self._execute_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result) if result is not None else "null"
                    })

            # Add assistant message and tool results
            kwargs["messages"].append({
                "role": "assistant",
                "content": [{"type": b.type, **({"text": b.text} if b.type == "text" else {"id": b.id, "name": b.name, "input": b.input})} for b in response.content]
            })
            kwargs["messages"].append({
                "role": "user",
                "content": tool_results
            })

            response = await self._aclient.messages.create(**kwargs)

        # Extract text content
        text_content = ""
        for block in response.content:
            if hasattr(block, "text"):
                text_content += block.text

        return LlmAIMessage(
            content=text_content,
            response_metadata={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "model": response.model,
                "stop_reason": response.stop_reason,
            }
        )

    def stream(
        self,
        messages: list[Union[LlmSystemMessage, LlmHumanMessage, LlmAIMessage]]
    ) -> Iterator[LlmMessageChunk]:
        """Stream responses from the model synchronously."""
        kwargs = self._build_kwargs(messages)
        chunk_index = itertools.count()

        with self._client.messages.stream(**kwargs) as stream:
            for text in stream.text_stream:
                yield LlmMessageChunk(
                    content=text,
                    role="assistant",
                    index=next(chunk_index),
                    is_final=False,
                    response_metadata={}
                )

        yield LlmMessageChunk(
            content="",
            role="assistant",
            index=next(chunk_index),
            is_final=True,
            response_metadata={}
        )

    async def astream(
        self,
        messages: list[Union[LlmSystemMessage, LlmHumanMessage, LlmAIMessage]],
        usage_callback: Optional[callable] = None,
    ) -> AsyncIterator[LlmMessageChunk]:
        """Stream responses from the model asynchronously with tool support.

        Args:
            messages: List of messages to send
            usage_callback: Optional callback(input_tokens, output_tokens) called after each API iteration
        """
        import asyncio
        import functools

        kwargs = self._build_kwargs(messages)
        chunk_index = itertools.count()

        # Build tool function map for execution
        tool_func_map = {}
        for toolbox in self.toolboxes:
            for tool in toolbox.tools:
                if isinstance(tool, LlmTool):
                    tool_func_map[tool.name] = tool.func

        max_tool_iterations = 10
        iteration = 0

        # Track cumulative usage across all iterations
        total_input_tokens = 0
        total_output_tokens = 0

        while iteration < max_tool_iterations:
            iteration += 1
            tool_use_blocks = []
            current_tool_use = None
            current_tool_input = ""

            async with self._aclient.messages.stream(**kwargs) as stream:
                async for event in stream:
                    # Handle different event types
                    if hasattr(event, 'type'):
                        if event.type == 'content_block_start':
                            block = event.content_block
                            if hasattr(block, 'type') and block.type == 'tool_use':
                                # Tool use starting
                                current_tool_use = {
                                    'id': block.id,
                                    'name': block.name,
                                    'input': ''
                                }
                                # Emit pending indicator with structured tool info
                                yield LlmMessageChunk(
                                    content="",
                                    role="assistant",
                                    index=next(chunk_index),
                                    is_final=False,
                                    response_metadata={},
                                    tool_call=ToolCallInfo(
                                        name=block.name,
                                        call_id=block.id,
                                        status=ToolCallStatus.PENDING,
                                    ),
                                )

                        elif event.type == 'content_block_delta':
                            delta = event.delta
                            if hasattr(delta, 'type'):
                                if delta.type == 'text_delta':
                                    # Text content
                                    yield LlmMessageChunk(
                                        content=delta.text,
                                        role="assistant",
                                        index=next(chunk_index),
                                        is_final=False,
                                        response_metadata={}
                                    )
                                elif delta.type == 'input_json_delta' and current_tool_use:
                                    # Tool input being streamed
                                    current_tool_input += delta.partial_json

                        elif event.type == 'content_block_stop':
                            if current_tool_use:
                                # Tool use complete - parse input and store
                                try:
                                    current_tool_use['input'] = json.loads(current_tool_input) if current_tool_input else {}
                                except json.JSONDecodeError:
                                    current_tool_use['input'] = {}
                                tool_use_blocks.append(current_tool_use)
                                current_tool_use = None
                                current_tool_input = ""

                # Get final message for stop reason and usage
                final_message = await stream.get_final_message()

            # Track usage from this iteration
            if hasattr(final_message, 'usage'):
                iter_input = getattr(final_message.usage, 'input_tokens', 0)
                iter_output = getattr(final_message.usage, 'output_tokens', 0)
                total_input_tokens += iter_input
                total_output_tokens += iter_output

                # Call usage callback if provided
                if usage_callback:
                    try:
                        usage_callback(iter_input, iter_output)
                    except Exception as e:
                        logger.warning(f"Usage callback error: {e}")

            # Check if we need to execute tools
            if final_message.stop_reason == "tool_use" and tool_use_blocks:
                tool_results = []

                for tool_block in tool_use_blocks:
                    tool_name = tool_block['name']
                    tool_input = tool_block['input']
                    tool_id = tool_block['id']

                    # Execute the tool
                    result = None
                    error_msg = None
                    if tool_name in tool_func_map:
                        func = tool_func_map[tool_name]
                        if callable(func):
                            try:
                                # Run in executor to avoid blocking
                                loop = asyncio.get_running_loop()
                                result = await loop.run_in_executor(
                                    None, functools.partial(func, **tool_input)
                                )
                            except Exception as e:
                                logger.error(f"Tool execution error for {tool_name}: {e}")
                                error_msg = str(e)

                    # Emit function result with structured tool info
                    yield LlmMessageChunk(
                        content="",
                        role="assistant",
                        index=next(chunk_index),
                        is_final=False,
                        response_metadata={},
                        tool_call=ToolCallInfo(
                            name=tool_name,
                            call_id=tool_id,
                            status=ToolCallStatus.ERROR if error_msg else ToolCallStatus.COMPLETED,
                            arguments=tool_input,
                            result=result,
                            error=error_msg,
                        ),
                    )

                    # For API continuation, still need string result
                    result_str = json.dumps(result) if result is not None else "null"

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": result_str
                    })

                # Add assistant message using final_message content (includes both text and tool_use)
                # This ensures we don't lose any text that was streamed before tool calls
                assistant_content = []
                for block in final_message.content:
                    if block.type == "text":
                        assistant_content.append({"type": "text", "text": block.text})
                    elif block.type == "tool_use":
                        assistant_content.append({
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.input
                        })

                kwargs["messages"].append({
                    "role": "assistant",
                    "content": assistant_content
                })
                kwargs["messages"].append({
                    "role": "user",
                    "content": tool_results
                })

                # Continue loop to get model's response to tool results
                continue
            else:
                # No more tool use, we're done
                break

        # Final chunk with total usage
        yield LlmMessageChunk(
            content="",
            role="assistant",
            index=next(chunk_index),
            is_final=True,
            response_metadata={
                'total_input_tokens': total_input_tokens,
                'total_output_tokens': total_output_tokens,
            }
        )
