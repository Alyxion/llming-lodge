"""Chat session handling with LLM providers."""
import logging
import time
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union

from pydantic import BaseModel, Field

from .messages import LlmAIMessage, LlmHumanMessage, LlmSystemMessage, LlmMessageChunk
from .tools.tool_definition import MCPServerConfig
from .llm_base_models import ChatHistory, ChatMessage, Role
from .providers import get_provider
from .providers.llm_provider_models import ReasoningEffort
from .budget import LLMBudgetManager, InsufficientBudgetError

logger = logging.getLogger(__name__)


class LLMConfig(BaseModel):
    """Configuration for LLM providers."""
    provider: str = Field(..., description="Provider name (openai, anthropic, mistral, google, together, or custom)")
    model: str = Field(..., description="Model name to use")
    base_url: Optional[str] = None
    temperature: float = Field(0.7, description="Temperature for responses")
    max_tokens: Optional[int] = 4096
    max_input_tokens: Optional[int] = 64000
    reasoning_effort: Optional[ReasoningEffort] = Field(None, description="Reasoning effort level (None = use model default)")

    max_history_images: int = Field(20, description="Maximum number of images kept active in history. Oldest are flagged stale.")
    condense_threshold_pct: float = Field(0.80, description="Trigger conversation condensation when context usage exceeds this fraction of max_input_tokens")
    condense_model: Optional[str] = Field(None, description="Model to use for condensation. If None, uses the cheapest available model from the same provider. Use a small/fast model to save cost and time.")
    condense_max_tokens: int = Field(5000, description="Hard cap on condensation summary output tokens. Prevents runaway costs on large contexts.")

    # Tool configuration
    tools: Optional[List[str]] = Field(None, description="List of tool names to enable. None = use model defaults.")
    tool_config: Optional[Dict[str, Any]] = Field(
        None,
        description="Per-tool configuration. Keys are tool names, values are tool-specific settings. "
                    "Example: {'generate_image': {'size': '1024x1024', 'quality': 'standard'}, "
                    "'web_search': {'max_results': 5}}"
    )
    mcp_servers: Optional[List[MCPServerConfig]] = Field(
        None,
        description="List of MCP server configurations. Tools from these servers will be discovered "
                    "and made available. Example: [MCPServerConfig(command='python', args=['-m', 'my_mcp_server'])]"
    )


class ChatSession:
    """Manages chat interactions with different LLM providers."""

    def __init__(
        self,
        config: LLMConfig,
        system_prompt: Optional[str] = None,
        budget_manager: Optional[LLMBudgetManager] = None,
        user_id: Optional[str] = None
    ):
        """Initialize chat session.
        
        Args:
            config: LLM configuration
            system_prompt: Optional system prompt to set context
            budget_manager: Optional budget manager for tracking costs
            user_id: Optional user ID for logging usage
        """
        self.config = config
        self.history = ChatHistory()
        self.budget_manager = budget_manager
        self._system_prompt = system_prompt
        self._context_preamble: Optional[str] = None  # silently prepended to system prompt at API call time
        self._client = None
        self._last_client_config = None
        self.user_id = user_id
        
        # Get provider implementation
        provider_class = get_provider(config.provider)
        self._provider = provider_class()
        
        # Get model info
        model_info = next(
            (info for info in self._provider.get_models() if info.name == config.model or info.model == config.model),
            None
        )
        if not model_info:
            raise ValueError(f"Model {config.model} not found in provider {config.provider}")
        
        self.model_info = model_info

        # Conversation condensation
        self._condensed_summary: Optional[str] = None
        self._is_condensing: bool = False
        self.on_condense_start: Optional[Any] = None   # Callable[[], None]
        self.on_condense_end: Optional[Any] = None     # Callable[[], None]
        self.on_condense_progress: Optional[Any] = None  # Callable[[float], None] — 0.0 to 1.0

        # MCP connections (lazily initialized)
        self._mcp_connections: Dict[str, Any] = {}
        self._mcp_tools_discovered = False
        # Tracks tool grouping per MCP server: server_id -> {label, description, category, exclude_providers, tool_names}
        self._mcp_server_groups: Dict[str, Dict[str, Any]] = {}

    def invalidate_client(self) -> None:
        """Invalidate the client and force a new client creation."""
        self._client = None
        self._last_client_config = None

    async def _discover_mcp_tools(self, early: bool = False) -> None:
        """Discover tools from configured MCP servers.

        Connects to all MCP servers in parallel and registers discovered tools.
        When early=True (called via discover_tools()), tools are discovered
        but only enabled if the server's enabled_by_default is True.

        Args:
            early: If True, this is an early/pre-message discovery call.
        """
        if self._mcp_tools_discovered or not self.config.mcp_servers:
            return

        import asyncio
        from .tools.mcp import create_connection, MCPError
        from .tools.tool_registry import get_default_registry

        registry = get_default_registry()

        # Store the current event loop for async MCP tool execution
        registry.set_event_loop(asyncio.get_running_loop())

        # Parse server configs upfront
        server_metas = []
        for idx, server_config in enumerate(self.config.mcp_servers):
            if hasattr(server_config, 'command'):
                server_id = server_config.command or server_config.url or server_config.label
                meta = dict(
                    enabled=getattr(server_config, 'enabled_by_default', False),
                    category=getattr(server_config, 'category', None),
                    exclude=getattr(server_config, 'exclude_providers', None),
                    label=getattr(server_config, 'label', None),
                    description=getattr(server_config, 'description', None),
                    default_tools=getattr(server_config, 'default_enabled_tools', None),
                )
            else:
                server_id = server_config.get('command') or server_config.get('url') or server_config.get('label')
                meta = dict(
                    enabled=server_config.get('enabled_by_default', False),
                    category=server_config.get('category'),
                    exclude=server_config.get('exclude_providers'),
                    label=server_config.get('label'),
                    description=server_config.get('description'),
                    default_tools=server_config.get('default_enabled_tools'),
                )
            meta['server_id'] = server_id
            meta['group_id'] = meta['label'] or server_id or f"mcp_{idx}"
            server_metas.append(meta)

        # ── Phase 1: start all connections + list tools in PARALLEL ──
        async def _connect_and_list(server_config, meta):
            connection = create_connection(server_config)
            await connection.start()
            tools = await connection.list_tools()
            return connection, tools

        results = await asyncio.gather(
            *[_connect_and_list(cfg, meta)
              for cfg, meta in zip(self.config.mcp_servers, server_metas)],
            return_exceptions=True,
        )

        # ── Phase 2: register tools (fast, in-memory) ──
        for meta, result in zip(server_metas, results):
            if isinstance(result, Exception):
                logger.error(f"Failed to connect to MCP server {meta['group_id']}: {result}")
                continue

            connection, tools = result
            group_tool_names = []

            for tool in tools:
                if meta['category'] and tool.ui:
                    tool.ui.category = meta['category']
                elif meta['category']:
                    from .tools.tool_definition import ToolUIMetadata
                    tool.ui = ToolUIMetadata(category=meta['category'])
                if meta['exclude']:
                    tool.exclude_providers = meta['exclude']

                registry.register(tool)
                self._mcp_connections[tool.name] = connection
                registry._mcp_connections[tool.name] = connection
                group_tool_names.append(tool.name)

                # Decide whether to enable this tool
                should_enable = False
                if not early:
                    should_enable = True
                elif meta['enabled']:
                    if meta['default_tools'] is not None:
                        should_enable = tool.name in meta['default_tools']
                    else:
                        should_enable = True

                if should_enable:
                    if self.config.tools is None:
                        self.config.tools = []
                    if tool.name not in self.config.tools:
                        self.config.tools = list(self.config.tools) + [tool.name]

            self._mcp_server_groups[meta['group_id']] = {
                "label": meta['label'] or meta['group_id'],
                "description": meta['description'],
                "category": meta['category'] or "General",
                "exclude_providers": meta['exclude'],
                "tool_names": group_tool_names,
            }

        self._mcp_tools_discovered = True

    async def discover_tools(self) -> None:
        """Early tool discovery (before first message).

        Connects to MCP servers and discovers tools, but only enables
        tools from servers with enabled_by_default=True.
        """
        await self._discover_mcp_tools(early=True)

    async def close_mcp_connections(self) -> None:
        """Close all MCP connections."""
        for connection in self._mcp_connections.values():
            try:
                await connection.close()
            except Exception as e:
                logger.warning(f"Error closing MCP connection: {e}")
        self._mcp_connections.clear()
        self._mcp_tools_discovered = False

    @property
    def system_prompt(self) -> Optional[str]:
        """Get the current system prompt."""
        return self._system_prompt

    @system_prompt.setter
    def system_prompt(self, value: Optional[str]) -> None:
        """Set a new system prompt."""
        self._system_prompt = value

    def _get_client(self, temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> 'LlmClient':
        """Get or create a client with the specified parameters."""
        # Check if the model has an enforced temperature
        if self.model_info.enforced_temperature is not None:
            # Use the enforced temperature instead of the provided one
            temperature = self.model_info.enforced_temperature

        # Determine reasoning effort: config override > model default
        reasoning_effort = self.config.reasoning_effort
        if reasoning_effort is None and self.model_info.reasoning:
            # Use model's default reasoning effort if it's a reasoning model
            reasoning_effort = self.model_info.default_reasoning_effort

        client_config = self.config.model_dump()
        client_config.update({
            'temperature': temperature if temperature is not None else self.config.temperature,
            'max_tokens': max_tokens if max_tokens is not None else self.config.max_tokens,
            'reasoning_effort': reasoning_effort
        })

        # Remove tool config from client config (we'll pass toolboxes separately)
        tools = client_config.pop('tools', None)
        tool_config = client_config.pop('tool_config', None)
        mcp_servers = client_config.pop('mcp_servers', None)

        # Create cache key including tool settings
        tools_key = tuple(tools) if tools else None
        tool_config_key = tuple(sorted(tool_config.items())) if tool_config else None
        # Handle both MCPServerConfig objects and dicts (from model_dump)
        if mcp_servers:
            mcp_key = tuple(
                (s.command if hasattr(s, 'command') else s.get('command')) or
                (s.url if hasattr(s, 'url') else s.get('url')) or
                (s.label if hasattr(s, 'label') else s.get('label'))
                for s in mcp_servers
            )
        else:
            mcp_key = None
        cache_key = (frozenset(client_config.items()), tools_key, tool_config_key, mcp_key)
        if self._client and self._last_client_config == cache_key:
            return self._client

        # Build toolboxes based on config
        toolboxes = self._build_toolboxes()
        client_config['toolboxes'] = toolboxes

        # Create new client
        self._client = self._provider.create_client(**client_config)
        self._last_client_config = cache_key
        return self._client

    def _build_toolboxes(self) -> List:
        """Build toolboxes based on tools config.

        Uses ToolRegistry and ToolboxAdapter for flexible tool management.
        Supports per-tool configuration via tool_config.
        """
        from .tools.toolbox_adapter import get_toolboxes_for_config
        from .providers.openai.openai_client import OpenAILlmClient
        import os

        # Create cost callback for budget tracking
        def tool_cost_callback(tool_name: str, cost_usd: float):
            if self.budget_manager:
                for limit in self.budget_manager.limits.values():
                    limit.reserve_budget(cost_usd)
                    limit.log_usage(
                        model_name=f"openai.{tool_name}",
                        tokens_input=0,
                        tokens_output=0,
                        costs=cost_usd,
                        duration_ms=0,
                        user_id=self.user_id
                    )
                logger.debug(f"[TOOL COST] {tool_name}: ${cost_usd:.4f} reserved and logged")

        # Create OpenAI client for DALL-E image generation if needed
        openai_client = None
        if self.config.provider == "openai":
            openai_client = OpenAILlmClient(
                api_key=os.environ.get('OPENAI_API_KEY'),
                model=self.config.model,
                max_tokens=self.config.max_tokens
            )

        # Get model's default tools
        model_default_tools = getattr(self.model_info, 'default_tools', None)

        logger.info(f"[TOOLBOXES] Building toolboxes: config.tools={self.config.tools}, model_defaults={model_default_tools}")

        # Use the adapter to get toolboxes
        result = get_toolboxes_for_config(
            tools=self.config.tools,
            provider=self.config.provider,
            model_default_tools=model_default_tools,
            tool_config=self.config.tool_config,
            cost_callback=tool_cost_callback,
            openai_client=openai_client,
        )

        # Log what toolboxes were created
        toolbox_summary = []
        for tb in result:
            tool_names = [t.name if hasattr(t, 'name') else str(t) for t in tb.tools]
            toolbox_summary.append(f"{tb.name}: {tool_names}")
        logger.info(f"[TOOLBOXES] Created {len(result)} toolboxes: {toolbox_summary}")

        return result

    def _prepare_messages(
        self,
        system_prompt: Optional[str] = None
    ) -> List[Union[LlmSystemMessage, LlmHumanMessage, LlmAIMessage]]:
        """Convert chat history to message format while respecting token limits."""
        # Use provided system prompt or fall back to default
        current_system_prompt = system_prompt if system_prompt is not None else self._system_prompt

        # Silently prepend context preamble (user identity, etc.)
        if self._context_preamble:
            current_system_prompt = self._context_preamble + "\n\n" + (current_system_prompt or "")

        # Inject condensed summary into system prompt if available
        if self._condensed_summary:
            current_system_prompt = (current_system_prompt or "") + \
                "\n\n---\n[Condensed conversation history]\n" + self._condensed_summary

        # Calculate system prompt tokens
        system_tokens = 0
        if current_system_prompt:
            system_msg = LlmSystemMessage(content=current_system_prompt)
            system_tokens = self._estimate_tokens([system_msg])

        # Calculate available tokens for chat history
        max_input = self.config.max_input_tokens or self.model_info.max_input_tokens
        available_tokens = max_input - system_tokens

        # Process messages from newest to oldest, skipping stale content
        messages = []
        current_tokens = 0

        for msg in reversed(self.history.messages):
            # Skip messages whose content was condensed into the summary
            if msg.content_stale:
                continue
            content = msg.content
            # Skip images flagged as stale (exceeded max_history_images limit)
            images = msg.images if (msg.images and not msg.images_stale) else None
            if msg.role == Role.USER:
                new_msg = LlmHumanMessage(content=content, images=images)
            elif msg.role == Role.ASSISTANT:
                new_msg = LlmAIMessage(content=content, images=images)
            else:
                continue

            msg_tokens = self._estimate_tokens([new_msg])
            if current_tokens + msg_tokens > available_tokens:
                break

            messages.insert(0, new_msg)
            current_tokens += msg_tokens

        # Add system message if we have one
        if current_system_prompt:
            if self.model_info.supports_system_prompt:
                messages.insert(0, system_msg)
            else:
                # Insert system prompt as a separate human message
                messages.insert(0, LlmHumanMessage(content=current_system_prompt))

        return messages

    def _extract_generated_images(self, response: str) -> List[str]:
        """Extract base64 image data from generated image function results.

        Args:
            response: The full response text containing function call results

        Returns:
            List of base64-encoded image strings (without data URI prefix)
        """
        import json

        images = []

        # Check if response contains generate_image function result
        if '"generate_image"' not in response or '"function_call_result"' not in response:
            return images

        # Try to find and parse JSON objects containing function_call_result
        # Handle concatenated JSON objects by finding balanced braces
        i = 0
        while i < len(response):
            if response[i] == '{':
                # Find the matching closing brace
                brace_count = 0
                start = i
                for j in range(i, len(response)):
                    if response[j] == '{':
                        brace_count += 1
                    elif response[j] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            # Found a complete JSON object
                            json_str = response[start:j+1]
                            try:
                                data = json.loads(json_str)
                                if (data.get('function') == 'generate_image' and
                                    data.get('function_call_result')):
                                    result = data['function_call_result']
                                    if isinstance(result, str) and len(result) > 100:
                                        images.append(result)
                            except json.JSONDecodeError:
                                pass
                            i = j
                            break
            i += 1

        return images

    def _clean_response_for_history(self, response: str) -> str:
        """Remove large base64 image data from response for cleaner history storage.

        Args:
            response: The full response text

        Returns:
            Response with base64 data replaced by placeholder
        """
        import re

        # Replace base64 image data in function results with placeholder
        # Match the pattern: "function_call_result": "<very long base64>"
        def replace_base64(_match):
            return '"function_call_result": "[IMAGE_GENERATED]"'

        # Pattern to match base64 data (long alphanumeric strings)
        pattern = r'"function_call_result":\s*"[A-Za-z0-9+/=]{1000,}"'
        cleaned = re.sub(pattern, replace_base64, response)

        return cleaned

    def add_message(self, role: Union[Role, str], content: str, images: Optional[List[str]] = None) -> None:
        """Add a message to the chat history.

        Args:
            role: The role of the message sender
            content: The text content of the message
            images: Optional list of base64-encoded images
        """
        if isinstance(role, str):
            role = Role(role)
        if role != Role.SYSTEM:  # Don't store system messages in history
            self.history.add_message(ChatMessage(role=role, content=content, images=images))
            if images:
                self._enforce_image_limit()

    def _enforce_image_limit(self) -> None:
        """Mark oldest images as stale when total exceeds max_history_images.

        Iterates from newest to oldest, counting images. Once the limit is
        reached, all remaining messages with images are flagged stale.
        """
        limit = self.config.max_history_images
        count = 0
        for msg in reversed(self.history.messages):
            if not msg.images:
                continue
            msg_images = len(msg.images)
            if count + msg_images <= limit:
                msg.images_stale = False
                count += msg_images
            else:
                if not msg.images_stale:
                    logger.debug(f"[IMAGE-LIMIT] Marking message {msg.id} images as stale ({msg_images} images)")
                msg.images_stale = True

    def _estimate_full_context_tokens(self) -> int:
        """Estimate total token usage from the FULL history without truncation.

        Unlike _prepare_messages() which drops old messages to fit the budget,
        this counts ALL non-stale messages + system prompt + condensed summary.
        Used to decide whether condensation is needed.
        """
        msgs = []

        # System prompt (with condensed summary if present)
        sp = self._system_prompt or ""
        if self._condensed_summary:
            sp += "\n\n---\n[Condensed conversation history]\n" + self._condensed_summary
        if sp:
            msgs.append(LlmSystemMessage(content=sp))

        # All non-stale history messages
        for msg in self.history.messages:
            if msg.content_stale:
                continue
            images = msg.images if (msg.images and not msg.images_stale) else None
            if msg.role == Role.USER:
                msgs.append(LlmHumanMessage(content=msg.content, images=images))
            elif msg.role == Role.ASSISTANT:
                msgs.append(LlmAIMessage(content=msg.content, images=images))

        return self._estimate_tokens(msgs)

    async def check_and_condense(self, force: bool = False) -> bool:
        """Check if context usage is near the limit and condense if needed.

        Called after the AI responds. If token usage exceeds condense_threshold_pct
        of max_input_tokens, summarizes the entire conversation into a condensed
        summary and marks all existing messages as content_stale.

        Args:
            force: If True, skip the threshold check and condense immediately.

        Returns True if condensation was performed.
        """
        if self._is_condensing:
            return False

        max_input = self.config.max_input_tokens or self.model_info.max_input_tokens

        # Estimate tokens from the FULL history (not _prepare_messages which truncates)
        current_tokens = self._estimate_full_context_tokens()

        logger.debug(f"[CONDENSE] Check: {current_tokens}/{max_input} tokens "
                     f"({current_tokens/max_input*100:.1f}%), threshold {self.config.condense_threshold_pct*100:.0f}%")

        if not force:
            threshold = max_input * self.config.condense_threshold_pct
            if current_tokens < threshold:
                return False

        logger.info(f"[CONDENSE] Context at {current_tokens}/{max_input} tokens "
                     f"({current_tokens/max_input*100:.1f}%), threshold {self.config.condense_threshold_pct*100:.0f}% — condensing")

        self._is_condensing = True
        if self.on_condense_start:
            self.on_condense_start()

        try:
            # Build conversation text from non-stale messages
            conv_parts = []
            for msg in self.history.messages:
                if msg.content_stale:
                    continue
                role = "User" if msg.role == Role.USER else "Assistant"
                text = msg.content
                if msg.images and not msg.images_stale:
                    text += f" [+{len(msg.images)} image(s)]"
                conv_parts.append(f"{role}: {text}")

            # Include previous summary as context if it exists
            if self._condensed_summary:
                conversation_text = (
                    f"[Previous summary]\n{self._condensed_summary}\n\n"
                    f"[New messages]\n" + "\n\n".join(conv_parts)
                )
            else:
                conversation_text = "\n\n".join(conv_parts)

            # Scale summary budget: 30% of input, hard-capped by condense_max_tokens
            input_token_est = len(conversation_text) // 4
            hard_cap = self.config.condense_max_tokens
            target_words = max(300, min(hard_cap * 75 // 100, input_token_est * 30 // 100))
            summary_max_tokens = max(1024, min(hard_cap, input_token_est * 40 // 100))

            # Pick the condensation model: explicit config > cheapest from same provider > current model
            condense_model = self.config.condense_model
            if not condense_model:
                all_models = self._provider.get_models()
                # Pick cheapest model with enough output capacity
                candidates = [
                    m for m in all_models
                    if m.max_output_tokens >= summary_max_tokens and m.model != self.config.model
                ]
                if candidates:
                    candidates.sort(key=lambda m: m.input_token_price)
                    condense_model = candidates[0].model
                    logger.info(f"[CONDENSE] Auto-selected cheapest model: {condense_model} "
                                f"(${candidates[0].input_token_price}/M input)")
                else:
                    condense_model = self.config.model

            # Generate summary using a one-shot LLM call (no tools, cheap model, no reasoning)
            client = self._provider.create_client(
                provider=self.config.provider,
                model=condense_model,
                base_url=self.config.base_url,
                temperature=0.3,
                max_tokens=summary_max_tokens,
                toolboxes=[],
                reasoning_effort=ReasoningEffort.NONE,
            )
            summary_messages = [
                LlmSystemMessage(content=(
                    "You are a conversation summarizer. Your task is to produce a DETAILED and LONG "
                    "summary that captures everything needed to continue the conversation without "
                    "losing any context.\n\n"
                    f"TARGET LENGTH: {target_words} words minimum. This is critical — do NOT "
                    "write a short summary. Use the full budget.\n\n"
                    "MUST INCLUDE:\n"
                    "- Every fact, decision, and preference stated by the user\n"
                    "- Technical details, configurations, code snippets, file paths\n"
                    "- The current state of any ongoing work or discussion\n"
                    "- Any open questions or pending items\n"
                    "- The tone and relationship context\n\n"
                    "FORMAT: Use structured sections with headers and bullet points. "
                    "Write in third person. Do not include meta-commentary about the summarization."
                )),
                LlmHumanMessage(content=conversation_text),
            ]

            logger.info(f"[CONDENSE] Input ~{input_token_est} tokens, target ~{target_words} words, "
                        f"max_tokens={summary_max_tokens}, model={condense_model}")

            # Stream the summary so we can report progress
            summary_parts = []
            generated_tokens = 0
            async for chunk in client.astream(summary_messages):
                if chunk.content:
                    summary_parts.append(chunk.content)
                    generated_tokens += max(1, len(chunk.content) // 4)
                    if self.on_condense_progress:
                        pct = min(0.99, generated_tokens / summary_max_tokens)
                        self.on_condense_progress(pct)

            summary_text = "".join(summary_parts).strip()

            if not summary_text or len(summary_text) < 50:
                logger.warning(f"[CONDENSE] Summary too short or empty ({len(summary_text)} chars), skipping")
                return False

            self._condensed_summary = summary_text
            summary_tokens = len(self._condensed_summary) // 4

            # Mark all current non-stale messages as content_stale
            marked = 0
            for msg in self.history.messages:
                if not msg.content_stale:
                    msg.content_stale = True
                    marked += 1

            logger.info(f"[CONDENSE] Done — summary {len(self._condensed_summary)} chars (~{summary_tokens} tokens), "
                         f"ratio {summary_tokens/max(1,input_token_est)*100:.0f}%, "
                         f"marked {marked} messages as stale")

        except Exception as e:
            logger.error(f"[CONDENSE] Failed: {e}")
        finally:
            self._is_condensing = False
            if self.on_condense_end:
                self.on_condense_end()

        return True

    def get_history(self) -> ChatHistory:
        """Get the chat history."""
        return self.history

    def clear_history(self) -> None:
        """Clear the chat history and condensed summary."""
        self.history.clear()
        self._condensed_summary = None

    def _estimate_tokens(self, messages: List[Union[LlmSystemMessage, LlmHumanMessage, LlmAIMessage]]) -> int:
        """Estimate tokens in messages including text and images.

        Image token estimation uses base64 data length to approximate pixel count,
        then calculates tokens based on the provider's pricing model.
        Fallback: ~765 tokens for a standard 1024x1024 image.
        """
        from tiktoken import get_encoding
        encoding = get_encoding("cl100k_base")

        text_tokens = 0
        image_tokens = 0

        for msg in messages:
            # Count text tokens
            if isinstance(msg, LlmSystemMessage):
                text_tokens += len(encoding.encode(f"System: {msg.content}\n"))
            elif isinstance(msg, LlmHumanMessage):
                text_tokens += len(encoding.encode(f"Human: {msg.content}\n"))
                if hasattr(msg, 'images') and msg.images:
                    for img in msg.images:
                        image_tokens += self._estimate_image_tokens(img)
            elif isinstance(msg, LlmAIMessage):
                text_tokens += len(encoding.encode(f"Assistant: {msg.content}\n"))
                # Generated images in assistant messages are output, not counted as input

        return text_tokens + image_tokens

    def _estimate_image_tokens(self, image_data: str) -> int:
        """Estimate token cost of a single image based on its base64 size.

        Uses base64 byte length to approximate pixel count, then applies
        provider-specific token pricing. Falls back to 765 tokens.
        """
        # Strip data URI prefix if present
        raw = image_data
        if ',' in raw:
            raw = raw.split(',', 1)[1]

        # base64 encodes 3 bytes into 4 chars; JPEG ~0.5 bytes/pixel at medium quality
        # So: pixels ≈ (base64_len * 3/4) / 0.5 = base64_len * 1.5
        base64_len = len(raw)
        approx_pixels = int(base64_len * 1.5)

        provider = self.config.provider
        if provider == 'anthropic':
            # Anthropic: ~1 token per 750 pixels, minimum 85
            return max(85, approx_pixels // 750)
        else:
            # OpenAI / others: high detail = 170 tokens per 512x512 tile + 85 base
            # Approximate side length from pixel count (assume square)
            side = int(approx_pixels ** 0.5)
            tiles = max(1, (side + 511) // 512) ** 2
            return 170 * tiles + 85

    def chat(
        self,
        message: str,
        *,
        streaming: bool = False,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        images: Optional[List[str]] = None
    ) -> Union[LlmAIMessage, Iterator[LlmMessageChunk]]:
        """Send a message and get a response synchronously.

        Args:
            message: The message to send
            streaming: If True, returns an Iterator[LlmMessageChunk] that yields response chunks.
                     If False, returns a LlmAIMessage containing the complete response.
            system_prompt: Optional system prompt to override the default
            temperature: Optional temperature to override the default
            max_tokens: Optional max_tokens to override the default
            images: Optional list of base64-encoded images to include with the message
        """
        start_time = time.time()

        self.add_message(Role.USER, message, images=images)
        
        messages = self._prepare_messages(system_prompt)
        estimated_input_tokens = self._estimate_tokens(messages)
        max_output_tokens = max_tokens if max_tokens is not None else (self.config.max_tokens or 4096)
        
        try:
            # Reserve budget if budget manager is configured
            if self.budget_manager:
                self.budget_manager.reserve_budget(
                    input_tokens=estimated_input_tokens,
                    max_output_tokens=max_output_tokens,
                    input_token_price=self.model_info.input_token_price,
                    output_token_price=self.model_info.output_token_price
                )

            # Get client with current parameters
            client = self._get_client(temperature=temperature, max_tokens=max_tokens)

            if streaming:
                # Streaming mode - return an Iterator
                full_response = ""
                text_only_response = ""  # Track only text tokens, not function results
                last_chunk = None
                def stream_response():
                    nonlocal full_response, text_only_response, last_chunk
                    for chunk in client.stream(messages):
                        content = chunk.content
                        full_response += content
                        last_chunk = chunk
                        # Only count text tokens, not function call results (which may contain huge base64 data)
                        chunk_role = getattr(chunk, 'role', 'assistant')
                        if chunk_role not in ('function', 'function_pending'):
                            text_only_response += content
                        chunk_obj = LlmMessageChunk(
                            content=content,
                            role=chunk.role if hasattr(chunk, 'role') else Role.ASSISTANT,
                            index=last_chunk.index if hasattr(last_chunk, 'index') else 0,
                            is_final=False,
                            response_metadata=last_chunk.response_metadata if hasattr(last_chunk, 'response_metadata') else {}
                        )
                        yield chunk_obj

                    # Return unused budget if budget manager is configured
                    if self.budget_manager and last_chunk:
                        # Only count text output tokens, not function call results
                        actual_output_tokens = self._estimate_tokens([LlmAIMessage(content=text_only_response)])
                        self.budget_manager.return_unused_budget(
                            reserved_output_tokens=max_output_tokens,
                            actual_output_tokens=actual_output_tokens,
                            output_token_price=self.model_info.output_token_price
                        )
                        
                        # Log usage information in all budget limits
                        import time
                        execution_time = time.time() - start_time
                        for limit in self.budget_manager.limits.values():
                            limit.log_usage(
                                model_name=f"{self.config.provider}.{self.config.model}",
                                tokens_input=estimated_input_tokens,
                                tokens_output=actual_output_tokens,
                                costs=(estimated_input_tokens * (self.model_info.input_token_price / 1_000_000) + 
                                      actual_output_tokens * (self.model_info.output_token_price / 1_000_000)),
                                duration_ms=execution_time * 1000,  # Convert to milliseconds
                                user_id=self.user_id
                            )
                    
                    self.add_message(Role.ASSISTANT, full_response)
                
                return stream_response()
            else:
                # Non-streaming mode - return a string
                response = client.invoke(messages)
                response_text = response.content
                
                # Return unused budget if budget manager is configured
                if self.budget_manager:
                    # Always use our own token calculation
                    actual_output_tokens = self._estimate_tokens([LlmAIMessage(content=response_text)])
                    self.budget_manager.return_unused_budget(
                        reserved_output_tokens=max_output_tokens,
                        actual_output_tokens=actual_output_tokens,
                        output_token_price=self.model_info.output_token_price
                    )
                    
                    # Log usage information in all budget limits
                    execution_time = time.time() - start_time
                    for limit in self.budget_manager.limits.values():
                        limit.log_usage(
                            model_name=f"{self.config.provider}.{self.config.model}",
                            tokens_input=estimated_input_tokens,
                            tokens_output=actual_output_tokens,
                            costs=(estimated_input_tokens * (self.model_info.input_token_price / 1_000_000) + 
                                  actual_output_tokens * (self.model_info.output_token_price / 1_000_000)),
                            duration_ms=execution_time * 1000,  # Convert to milliseconds
                            user_id=self.user_id
                        )
                
                self.add_message(Role.ASSISTANT, response_text)
                return LlmAIMessage(content=response_text, response_metadata=response.response_metadata or {})
            
        except Exception as e:
            if isinstance(e, InsufficientBudgetError):
                raise
            raise RuntimeError(f"Error in chat completion: {str(e)}")

    async def chat_async(
        self,
        message: str,
        *,
        streaming: bool = False,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        images: Optional[List[str]] = None
    ) -> Union[LlmAIMessage, AsyncIterator[LlmMessageChunk]]:
        """Send a message and get a response asynchronously.

        Args:
            message: The message to send
            streaming: If True, returns an AsyncIterator[LlmMessageChunk] that yields response chunks.
                     If False, returns a LlmAIMessage containing the complete response.
            system_prompt: Optional system prompt to override the default
            temperature: Optional temperature to override the default
            max_tokens: Optional max_tokens to override the default
            images: Optional list of base64-encoded images to include with the message
        """
        import time
        start_time = time.time()

        # Discover MCP tools if configured (only on first call)
        logger.debug(f"[SESSION] chat_async: mcp_servers={self.config.mcp_servers}, discovered={self._mcp_tools_discovered}")
        if self.config.mcp_servers and not self._mcp_tools_discovered:
            logger.debug(f"[SESSION] Triggering MCP discovery for {len(self.config.mcp_servers)} servers")
            await self._discover_mcp_tools()

        self.add_message(Role.USER, message, images=images)

        messages = self._prepare_messages(system_prompt)
        estimated_input_tokens = self._estimate_tokens(messages)
        max_output_tokens = max_tokens if max_tokens is not None else (self.config.max_tokens or 4096)

        try:
            # Reserve budget if budget manager is configured
            if self.budget_manager:
                self.budget_manager.reserve_budget(
                    input_tokens=estimated_input_tokens,
                    max_output_tokens=max_output_tokens,
                    input_token_price=self.model_info.input_token_price,
                    output_token_price=self.model_info.output_token_price
                )

            # Get client with current parameters
            client = self._get_client(temperature=temperature, max_tokens=max_output_tokens)

            if streaming:
                # Streaming mode - return an AsyncIterator
                # Clients (OpenAI/Anthropic) now handle tool execution and auto-continue internally
                full_response = ""
                text_only_response = ""  # Track only text tokens for budget
                last_chunk = None

                # Track live usage across API iterations via callback
                live_usage = {'input_tokens': 0, 'output_tokens': 0}

                def usage_callback(input_tokens: int, output_tokens: int):
                    """Called after each API iteration with that iteration's usage."""
                    live_usage['input_tokens'] += input_tokens
                    live_usage['output_tokens'] += output_tokens
                    logger.debug(f"[USAGE] Iteration: +{input_tokens} input, +{output_tokens} output (total: {live_usage})")

                async def stream_response():
                    nonlocal full_response, text_only_response, last_chunk

                    # State machine for filtering base64 images from streaming chunks
                    from llming_lodge.utils.image_utils import is_likely_image_data

                    filter_state = {
                        "state": "normal",  # "normal", "buffering", "in_base64"
                        "buffer": "",
                    }

                    def filter_base64_from_chunk(content: str) -> str:
                        """Filter base64 image data from streaming content.

                        Replaces markdown ![...](data:image/...;base64,...) with [image].
                        Full base64 data is still kept in full_response for history processing.
                        """
                        if not content:
                            return content

                        state = filter_state["state"]
                        buf = filter_state["buffer"]
                        output = ""

                        for char in content:
                            if state == "in_base64":
                                # We're inside base64 data, look for closing )
                                if char == ")":
                                    state = "normal"
                                    output += "[image]\n"
                                    logger.debug(f"[STREAM] Filtered base64 image from chunk")
                                # else: skip the base64 character
                            elif state == "buffering":
                                # We saw ![, buffering to check for data:image
                                buf += char
                                # Check if we've confirmed it's a base64 image
                                if "](data:image/" in buf and "base64," in buf:
                                    # It's a base64 image - output text before ![
                                    img_start = buf.find("![")
                                    if img_start > 0:
                                        output += buf[:img_start]
                                    state = "in_base64"
                                    buf = ""
                                elif len(buf) > 200 and "](data:" not in buf:
                                    # Not an image, flush buffer
                                    output += buf
                                    buf = ""
                                    state = "normal"
                                elif ")" in buf and "base64," not in buf:
                                    # It's a regular markdown link, not base64
                                    output += buf
                                    buf = ""
                                    state = "normal"
                            else:  # normal state
                                if char == "!" and buf == "":
                                    # Potential start of markdown image
                                    buf = char
                                    state = "buffering"
                                elif buf == "!" and char == "[":
                                    # Confirmed ![ - continue buffering
                                    buf += char
                                    state = "buffering"
                                elif buf == "!":
                                    # Was ! but not followed by [, flush
                                    output += buf + char
                                    buf = ""
                                else:
                                    output += char

                        filter_state["state"] = state
                        filter_state["buffer"] = buf
                        return output

                    async for chunk in client.astream(messages, usage_callback=usage_callback):
                        last_chunk = chunk

                        # Accumulate text content (not tool call events)
                        if chunk.content and not chunk.tool_call:
                            full_response += chunk.content
                            text_only_response += chunk.content

                            # Filter base64 from the streaming chunk for display
                            filtered_content = filter_base64_from_chunk(chunk.content)
                            if filtered_content != chunk.content:
                                # Create a new chunk with filtered content
                                chunk = LlmMessageChunk(
                                    content=filtered_content,
                                    role=chunk.role,
                                    index=chunk.index if hasattr(chunk, 'index') else 0,
                                    is_final=chunk.is_final if hasattr(chunk, 'is_final') else False,
                                    response_metadata=chunk.response_metadata if hasattr(chunk, 'response_metadata') else {},
                                    tool_call=chunk.tool_call if hasattr(chunk, 'tool_call') else None,
                                    tool_result=chunk.tool_result if hasattr(chunk, 'tool_result') else None,
                                    image_data=chunk.image_data if hasattr(chunk, 'image_data') else None,
                                )

                        # Yield the chunk (filtered for display, but full_response has unfiltered content for history)
                        yield chunk

                    # Return unused budget if budget manager is configured
                    if self.budget_manager and last_chunk:
                        # Use actual tokens from API response if available, otherwise estimate
                        metadata = last_chunk.response_metadata or {}
                        actual_input_tokens = metadata.get('total_input_tokens') or live_usage['input_tokens'] or estimated_input_tokens
                        actual_output_tokens = metadata.get('total_output_tokens') or live_usage['output_tokens'] or self._estimate_tokens([LlmAIMessage(content=text_only_response)])

                        await self.budget_manager.return_unused_budget_async(
                            reserved_output_tokens=max_output_tokens,
                            actual_output_tokens=actual_output_tokens,
                            output_token_price=self.model_info.output_token_price
                        )

                        # Log usage information in all budget limits
                        execution_time = time.time() - start_time
                        actual_cost = (actual_input_tokens * (self.model_info.input_token_price / 1_000_000) +
                                      actual_output_tokens * (self.model_info.output_token_price / 1_000_000))
                        for limit in self.budget_manager.limits.values():
                            await limit.log_usage_async(
                                model_name=f"{self.config.provider}.{self.config.model}",
                                tokens_input=actual_input_tokens,
                                tokens_output=actual_output_tokens,
                                costs=actual_cost,
                                duration_ms=execution_time * 1000,  # Convert to milliseconds
                                user_id=self.user_id
                            )
                        logger.debug(f"[USAGE] Final: {actual_input_tokens} input, {actual_output_tokens} output, ${actual_cost:.6f}")

                    # Extract any generated images from the response and add to history
                    generated_images = self._extract_generated_images(full_response)

                    # Add the text response (without huge base64 data) along with generated images
                    clean_response = self._clean_response_for_history(full_response)
                    self.add_message(Role.ASSISTANT, clean_response, images=generated_images if generated_images else None)

                    # Check if context needs condensation
                    await self.check_and_condense()

                return stream_response()
            else:
                # Non-streaming mode - return a string
                response = await client.ainvoke(messages)
                response_text = response.content

                # Return unused budget if budget manager is configured
                if self.budget_manager:
                    # Use actual tokens from API response if available, otherwise estimate
                    metadata = response.response_metadata or {}
                    actual_input_tokens = metadata.get('input_tokens') or metadata.get('total_input_tokens') or estimated_input_tokens
                    actual_output_tokens = metadata.get('output_tokens') or metadata.get('total_output_tokens') or self._estimate_tokens([LlmAIMessage(content=response_text)])

                    await self.budget_manager.return_unused_budget_async(
                        reserved_output_tokens=max_output_tokens,
                        actual_output_tokens=actual_output_tokens,
                        output_token_price=self.model_info.output_token_price
                    )

                    # Log usage information in all budget limits
                    execution_time = time.time() - start_time
                    actual_cost = (actual_input_tokens * (self.model_info.input_token_price / 1_000_000) +
                                  actual_output_tokens * (self.model_info.output_token_price / 1_000_000))
                    for limit in self.budget_manager.limits.values():
                        await limit.log_usage_async(
                            model_name=f"{self.config.provider}.{self.config.model}",
                            tokens_input=actual_input_tokens,
                            tokens_output=actual_output_tokens,
                            costs=actual_cost,
                            duration_ms=execution_time * 1000,  # Convert to milliseconds
                            user_id=self.user_id
                        )
                    logger.debug(f"[USAGE] Final: {actual_input_tokens} input, {actual_output_tokens} output, ${actual_cost:.6f}")

                self.add_message(Role.ASSISTANT, response_text)

                # Check if context needs condensation
                await self.check_and_condense()

                return LlmAIMessage(content=response_text, response_metadata=response.response_metadata or {})

        except Exception as e:
            if isinstance(e, InsufficientBudgetError):
                raise
            raise RuntimeError(f"Error in async chat completion: {str(e)}")

    def copy_history_from(self, other: 'ChatSession') -> None:
        """Copy history from another session."""
        self.clear_history()
        for msg in other.history.messages:
            self.add_message(msg.role, msg.content)
            
    @classmethod
    def create_with_history(
        cls,
        config: LLMConfig,
        history: ChatHistory,
        system_prompt: Optional[str] = None,
        budget_manager: Optional[LLMBudgetManager] = None,
        user_id: Optional[str] = None
    ) -> 'ChatSession':
        """Create a new session with existing history.
        
        Args:
            config: LLM configuration
            history: Chat history to initialize with
            system_prompt: Optional system prompt
            budget_manager: Optional budget manager for tracking costs
            user_id: Optional user ID for logging usage
            
        Returns:
            New ChatSession instance with copied history
        """
        session = cls(config=config, system_prompt=system_prompt, budget_manager=budget_manager, user_id=user_id)
        for msg in history.messages:
            if msg.role != Role.SYSTEM:  # Don't copy system messages
                session.add_message(msg.role, msg.content)
        return session
