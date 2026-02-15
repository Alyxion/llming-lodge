"""Tests for memory management across providers."""
import pytest

from llming_lodge.llm_provider_manager import LLMManager
from llming_lodge.session import ChatSession, LLMConfig
from llming_lodge.messages import LlmAIMessage, LlmMessageChunk
from .conftest import BASIC_MODELS


@pytest.fixture
def manager(monkeypatch):
    """Test LLM manager."""
    return LLMManager()

@pytest.mark.asyncio
@pytest.mark.parametrize("provider,model", BASIC_MODELS)
async def test_memory_accumulation(manager, provider, model):
    """Test that memory accumulates correctly."""
    if provider not in manager.providers:
        pytest.skip(f"Provider {provider} not available (no API key)")

    config = manager.get_config_for_model(model)
    session = ChatSession(
        config=config,
        system_prompt="Reply in max 5 words."
    )

    full_response = ""
    async for chunk in await session.chat_async("Hello!", streaming=True):
        full_response += chunk.content

    assert len(full_response) > 0

    history = session.get_history()
    assert len(history.messages) == 2  # User + Assistant
    assert history.messages[0].role == "user"
    assert history.messages[1].role == "assistant"

@pytest.mark.asyncio
async def test_memory_persistence_across_providers(manager):
    """Test that memory persists when switching between providers."""
    available_providers = [p for p, m in BASIC_MODELS if p in manager.providers]
    if len(available_providers) < 2:
        pytest.skip("Need at least two providers for this test")

    provider1, provider2 = available_providers[:2]
    model1 = next(m for p, m in BASIC_MODELS if p == provider1)
    model2 = next(m for p, m in BASIC_MODELS if p == provider2)

    # Start with first provider
    config1 = manager.get_config_for_model(model1)
    session = ChatSession(
        config=config1,
        system_prompt="Reply in max 5 words."
    )

    full_response = ""
    async for chunk in await session.chat_async("Hello!", streaming=True):
        full_response += chunk.content
    assert len(full_response) > 0

    # Switch to second provider
    config2 = manager.get_config_for_model(model2)
    new_session = ChatSession.create_with_history(
        config=config2,
        history=session.get_history(),
        system_prompt="Reply in max 5 words."
    )

    full_response2 = ""
    async for chunk in await new_session.chat_async("What did I say before?", streaming=True):
        full_response2 += chunk.content
    assert len(full_response2) > 0

    # Verify history persistence
    history = new_session.get_history()
    assert len(history.messages) == 4  # 2 pairs of User + Assistant
    for i, msg in enumerate(history.messages):
        assert msg.role == ("user" if i % 2 == 0 else "assistant")

@pytest.mark.asyncio
async def test_memory_copying_on_provider_switch(manager):
    """Test that memory is correctly copied when switching providers."""
    available_providers = [p for p, m in BASIC_MODELS if p in manager.providers]
    if len(available_providers) < 2:
        pytest.skip("Need at least two providers for this test")

    provider1, provider2 = available_providers[:2]
    model1 = next(m for p, m in BASIC_MODELS if p == provider1)
    model2 = next(m for p, m in BASIC_MODELS if p == provider2)

    config1 = manager.get_config_for_model(model1)
    session1 = ChatSession(
        config=config1,
        system_prompt="Reply in max 5 words."
    )

    async for chunk in await session1.chat_async("Hello!", streaming=True):
        pass

    config2 = manager.get_config_for_model(model2)
    session2 = ChatSession(
        config=config2,
        system_prompt="Reply in max 5 words."
    )
    session2.copy_history_from(session1)

    history1 = session1.get_history()
    history2 = session2.get_history()
    assert len(history1.messages) == len(history2.messages)
    for msg1, msg2 in zip(history1.messages, history2.messages):
        assert msg1.role in ("user", "assistant")
        assert msg1.role == msg2.role
        assert msg1.content == msg2.content
