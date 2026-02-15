"""Base LLM client implementation."""
from abc import ABC, abstractmethod
from typing import AsyncIterator, Iterator, Optional, Union

from tiktoken import get_encoding

from .messages import LlmAIMessage, LlmHumanMessage, LlmSystemMessage, LlmMessageChunk


class LlmClient(ABC):
    """Base class for LLM clients."""

    def __init__(
        self,
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        streaming: bool = False
    ):
        """Initialize LLM client.
        
        Args:
            model: Model name to use
            temperature: Temperature for responses
            max_tokens: Maximum tokens to generate
            streaming: Whether to enable streaming mode
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.streaming = streaming

    def estimate_tokens(self, text: str, role: Optional[str] = None) -> int:
        """Estimate number of tokens in text using cl100k base tokenizer.
        
        Args:
            text: Text to estimate tokens for
            role: Optional role prefix (e.g. "system", "user", "assistant")
            
        Returns:
            Estimated token count
        """
        encoding = get_encoding("cl100k_base")
        if role:
            # Add role prefix to better estimate actual token usage
            text = f"{role}: {text}"
        return len(encoding.encode(text))

    async def estimate_tokens_async(self, text: str, role: Optional[str] = None) -> int:
        """Async version of estimate_tokens.
        
        Args:
            text: Text to estimate tokens for
            role: Optional role prefix (e.g. "system", "user", "assistant")
            
        Returns:
            Estimated token count
        """
        return self.estimate_tokens(text, role)

    @abstractmethod
    def invoke(self, messages: list[Union[LlmSystemMessage, LlmHumanMessage, LlmAIMessage]]) -> LlmAIMessage:
        """Synchronously invoke the model.
        
        Args:
            messages: List of messages to send
            
        Returns:
            Model response text
        """
        pass

    @abstractmethod
    async def ainvoke(self, messages: list[Union[LlmSystemMessage, LlmHumanMessage, LlmAIMessage]]) -> LlmAIMessage:
        """Asynchronously invoke the model.
        
        Args:
            messages: List of messages to send
            
        Returns:
            Model response text
        """
        pass

    @abstractmethod
    def stream(self, messages: list[Union[LlmSystemMessage, LlmHumanMessage, LlmAIMessage]]) -> Iterator[LlmMessageChunk]:
        """Stream responses from the model synchronously.
        
        Args:
            messages: List of messages to send
            
        Returns:
            Iterator yielding response chunks
        """
        pass

    @abstractmethod
    async def astream(
        self,
        messages: list[Union[LlmSystemMessage, LlmHumanMessage, LlmAIMessage]],
        usage_callback: Optional[callable] = None,
    ) -> AsyncIterator[LlmMessageChunk]:
        """Stream responses from the model asynchronously.

        Args:
            messages: List of messages to send
            usage_callback: Optional callback(input_tokens, output_tokens) called after each API iteration.

        Returns:
            AsyncIterator yielding response chunks
        """
        pass
