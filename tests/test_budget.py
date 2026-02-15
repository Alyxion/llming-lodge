import os
import pytest
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

from llming_lodge.budget.budget_manager import LLMBudgetManager
from llming_lodge.budget.budget_types import InsufficientBudgetError, TokenUsage, LimitPeriod
from llming_lodge.budget.memory_budget_limit import MemoryBudgetLimit
from llming_lodge.session import ChatSession, LLMConfig
from llming_lodge.providers import get_provider
from llming_lodge.messages import LlmAIMessage, LlmHumanMessage, LlmMessageChunk


async def collect_stream(generator):
    """Helper to collect streaming response."""
    chunks = []
    full_text = ""
    async for chunk in generator:
        chunks.append(chunk)
        full_text += chunk.content
    return chunks, full_text


def test_budget_manager_initialization():
    limits = [
        MemoryBudgetLimit(
            name="total",
            amount=10.0,
            period=LimitPeriod.TOTAL
        )
    ]
    budget = LLMBudgetManager(limits)
    assert budget.available_budget == pytest.approx(10.0)


def test_budget_reservation():
    limits = [
        MemoryBudgetLimit(
            name="total",
            amount=2.0,
            period=LimitPeriod.TOTAL
        )
    ]
    budget = LLMBudgetManager(limits)
    # Example: 300 input tokens at 1000.0/1M tokens and 500 output tokens at 2000.0/1M tokens
    budget.reserve_budget(input_tokens=300, max_output_tokens=500, input_token_price=1000.0, output_token_price=2000.0)
    # Cost calculation:
    # Input: 300 * (1000.0/1M tokens) = 0.3
    # Output: 500 * (2000.0/1M tokens) = 1.0
    # Total: 1.3
    # Remaining: 2.0 - 1.3 = 0.7
    expected = 0.7
    assert budget.available_budget == pytest.approx(expected)


def test_insufficient_budget():
    limits = [
        MemoryBudgetLimit(
            name="total",
            amount=0.5,
            period=LimitPeriod.TOTAL
        )
    ]
    budget = LLMBudgetManager(limits)
    with pytest.raises(InsufficientBudgetError) as exc_info:
        # Would require 1.3 (0.3 for input + 1.0 for output)
        budget.reserve_budget(input_tokens=300, max_output_tokens=500, input_token_price=1000.0, output_token_price=2000.0)
    assert exc_info.value.limit_name == "total"


def test_return_unused_budget():
    limits = [
        MemoryBudgetLimit(
            name="total",
            amount=2.0,
            period=LimitPeriod.TOTAL
        )
    ]
    budget = LLMBudgetManager(limits)
    # Reserve: 300 input tokens at 1000.0/1M tokens and 500 output tokens at 2000.0/1M tokens
    budget.reserve_budget(input_tokens=300, max_output_tokens=500, input_token_price=1000.0, output_token_price=2000.0)
    # Return: Only used 200 of 500 output tokens, so return 300 * (2000.0/1M tokens) = 0.6
    budget.return_unused_budget(500, 200, 2000.0)
    expected = 2.0 - (0.3 + 0.4)  # 2 - (0.3 + 0.4)
    assert budget.available_budget == pytest.approx(expected)


def test_invalid_return_amount():
    limits = [
        MemoryBudgetLimit(
            name="total",
            amount=2.0,
            period=LimitPeriod.TOTAL
        )
    ]
    budget = LLMBudgetManager(limits)
    budget.reserve_budget(input_tokens=300, max_output_tokens=500, input_token_price=1000.0, output_token_price=2000.0)
    with pytest.raises(ValueError):
        # Cannot return more tokens than reserved
        budget.return_unused_budget(500, 600, 2000.0)


def test_budget_reset():
    limits = [
        MemoryBudgetLimit(
            name="total",
            amount=2.0,
            period=LimitPeriod.TOTAL
        )
    ]
    budget = LLMBudgetManager(limits)
    budget.reserve_budget(input_tokens=300, max_output_tokens=500, input_token_price=1000.0, output_token_price=2000.0)
    budget.reset()
    assert budget.available_budget == pytest.approx(2.0)


def test_token_usage():
    usage = TokenUsage(
        input_tokens=100,
        output_tokens=150,
        input_cost=0.1,
        output_cost=0.3
    )
    assert usage.total_tokens == 250
    assert usage.total_cost == pytest.approx(0.4)


@pytest.mark.asyncio
async def test_memory_budget_async_behavior():
    """Test the async behavior of memory budget implementation."""
    # Initialize with a small budget
    limit = MemoryBudgetLimit(
        name="test_memory",
        amount=1.0,
        period=LimitPeriod.TOTAL
    )

    # Test initial state
    available = await limit.get_available_budget_async()
    assert available == 1.0

    # Test async budget reservation
    success = await limit.reserve_budget_async(0.3)
    assert success
    available = await limit.get_available_budget_async()
    assert available == pytest.approx(0.7)

    # Test multiple async reservations
    success = await limit.reserve_budget_async(0.2)
    assert success
    available = await limit.get_available_budget_async()
    assert available == pytest.approx(0.5)

    # Test async budget return
    await limit.return_budget_async(0.1)
    available = await limit.get_available_budget_async()
    assert available == pytest.approx(0.6)

    # Test async reset
    await limit.reset_async()
    available = await limit.get_available_budget_async()
    assert available == 1.0

    # Test concurrent async operations
    import asyncio

    async def concurrent_reservation():
        # Try to reserve 0.4 concurrently
        success = await limit.reserve_budget_async(0.4)
        if success:
            await asyncio.sleep(0.1)  # Simulate some async work
            await limit.return_budget_async(0.4)

    # Create multiple tasks that try to reserve budget concurrently
    tasks = [concurrent_reservation() for _ in range(5)]
    await asyncio.gather(*tasks)

    # After all tasks complete, budget should be back to initial amount
    available = await limit.get_available_budget_async()
    assert available == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_budget_manager_async_behavior():
    """Test the async behavior of budget manager."""
    limits = [
        MemoryBudgetLimit(
            name="total",
            amount=2.0,
            period=LimitPeriod.TOTAL
        )
    ]
    budget = LLMBudgetManager(limits)

    # Test initial state
    available = await budget.available_budget_async()
    assert available == pytest.approx(2.0)

    # Test async budget reservation
    await budget.reserve_budget_async(input_tokens=300, max_output_tokens=500, input_token_price=1000.0, output_token_price=2000.0)
    available = await budget.available_budget_async()
    assert available == pytest.approx(0.7)

    # Test async return of unused budget
    await budget.return_unused_budget_async(reserved_output_tokens=500, actual_output_tokens=200, output_token_price=2000.0)
    available = await budget.available_budget_async()
    expected = 2.0 - (0.3 + 0.4)
    assert available == pytest.approx(expected)

    # Test async reset
    await budget.reset_async()
    available = await budget.available_budget_async()
    assert available == pytest.approx(2.0)

    # Test concurrent operations
    import asyncio

    async def concurrent_operation():
        try:
            await budget.reserve_budget_async(input_tokens=100, max_output_tokens=200, input_token_price=1000.0, output_token_price=2000.0)
            await asyncio.sleep(0.1)
            await budget.return_unused_budget_async(reserved_output_tokens=200, actual_output_tokens=100, output_token_price=2000.0)
        except InsufficientBudgetError:
            pass

    tasks = [concurrent_operation() for _ in range(5)]
    await asyncio.gather(*tasks)

    available = await budget.available_budget_async()
    assert 0 <= available <= 2.0, "Budget should remain within valid range"


def test_multiple_limits():
    limits = [
        MemoryBudgetLimit(
            name="total",
            amount=10.0,
            period=LimitPeriod.TOTAL
        ),
        MemoryBudgetLimit(
            name="daily",
            amount=5.0,
            period=LimitPeriod.DAILY
        )
    ]
    budget = LLMBudgetManager(limits)

    # Available budget should be the minimum of all limits
    assert budget.available_budget == pytest.approx(5.0)

    # Now try to reserve more than the daily limit - should raise InsufficientBudgetError
    with pytest.raises(InsufficientBudgetError):
        budget.reserve_budget(input_tokens=2000, max_output_tokens=2000, input_token_price=1000.0, output_token_price=2000.0)


@pytest.mark.parametrize("provider,model", [
    ("openai", "gpt-5.2"),
    ("anthropic", "claude-sonnet-4-5-20250929"),
    ("google", "gemini-2.5-flash"),
    ("mistral", "mistral-small-latest"),
])
def test_chat_session_with_budget(provider, model):
    """Test budget management with different providers."""
    provider_class = get_provider(provider)
    provider_instance = provider_class()
    if not provider_instance.is_available:
        pytest.skip(f"Provider {provider} is not available (API key not set)")

    config = LLMConfig(
        provider=provider,
        model=model,
        max_tokens=100
    )
    limits = [
        MemoryBudgetLimit(
            name="total",
            amount=10.0,
            period=LimitPeriod.TOTAL
        )
    ]
    budget = LLMBudgetManager(limits)

    session = ChatSession(
        config=config,
        system_prompt="You are a helpful assistant.",
        budget_manager=budget
    )

    assert budget.available_budget == pytest.approx(10.0)

    response = session.chat("What is 2+2?", streaming=False)
    assert isinstance(response, LlmAIMessage)
    assert len(response.content) > 0
    assert budget.available_budget < 10.0


@pytest.mark.parametrize("provider,model", [
    ("openai", "gpt-5.2"),
    ("anthropic", "claude-sonnet-4-5-20250929"),
    ("google", "gemini-2.5-flash"),
    ("mistral", "mistral-small-latest"),
])
def test_chat_session_budget_exceeded(provider, model):
    """Test budget exceeded scenario with different providers."""
    provider_class = get_provider(provider)
    provider_instance = provider_class()
    if not provider_instance.is_available:
        pytest.skip(f"Provider {provider} is not available (API key not set)")

    config = LLMConfig(
        provider=provider,
        model=model,
        max_tokens=2000
    )
    limits = [
        MemoryBudgetLimit(
            name="total",
            amount=0.001,
            period=LimitPeriod.TOTAL
        )
    ]
    budget = LLMBudgetManager(limits)

    session = ChatSession(
        config=config,
        budget_manager=budget
    )

    with pytest.raises(InsufficientBudgetError) as exc_info:
        session.chat("Tell me a very long story about artificial intelligence.")
    assert exc_info.value.limit_name == "total"


@pytest.mark.parametrize("provider,model", [
    ("openai", "gpt-5.2"),
    ("anthropic", "claude-sonnet-4-5-20250929"),
    ("google", "gemini-2.5-flash"),
    ("mistral", "mistral-small-latest"),
])
@pytest.mark.asyncio
async def test_chat_session_async(provider, model):
    """Test async non-streaming responses with budget management."""
    provider_class = get_provider(provider)
    provider_instance = provider_class()
    if not provider_instance.is_available:
        pytest.skip(f"Provider {provider} is not available (API key not set)")

    config = LLMConfig(
        provider=provider,
        model=model,
        max_tokens=100
    )
    limits = [
        MemoryBudgetLimit(
            name="total",
            amount=10.0,
            period=LimitPeriod.TOTAL
        )
    ]
    budget = LLMBudgetManager(limits)

    session = ChatSession(
        config=config,
        system_prompt="You are a helpful assistant.",
        budget_manager=budget
    )

    assert budget.available_budget == pytest.approx(10.0)

    response = await session.chat_async("What is 2+2?")
    assert isinstance(response, LlmAIMessage)
    assert len(response.content) > 0
    assert budget.available_budget < 10.0


@pytest.mark.parametrize("provider,model", [
    ("openai", "gpt-5.2"),
    ("anthropic", "claude-sonnet-4-5-20250929"),
    ("google", "gemini-2.5-flash"),
    ("mistral", "mistral-small-latest"),
])
@pytest.mark.asyncio
async def test_chat_session_streaming(provider, model):
    """Test streaming responses in async mode."""
    provider_class = get_provider(provider)
    provider_instance = provider_class()
    if not provider_instance.is_available:
        pytest.skip(f"Provider {provider} is not available (API key not set)")

    config = LLMConfig(
        provider=provider,
        model=model,
        max_tokens=100
    )
    limits = [
        MemoryBudgetLimit(
            name="total",
            amount=10.0,
            period=LimitPeriod.TOTAL
        )
    ]
    budget = LLMBudgetManager(limits)

    session = ChatSession(
        config=config,
        system_prompt="You are a helpful assistant.",
        budget_manager=budget
    )

    assert budget.available_budget == pytest.approx(10.0)

    chunks, full_text = await collect_stream(await session.chat_async("Count from 1 to 5.", streaming=True))

    assert len(chunks) >= 1, "Should receive at least one chunk when streaming"
    assert all(isinstance(chunk, LlmMessageChunk) for chunk in chunks)
    assert full_text is not None and len(full_text) > 0

    assert budget.available_budget < 10.0
    assert session.history.messages[-1].content == full_text
