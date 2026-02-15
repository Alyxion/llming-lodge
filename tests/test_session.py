"""Tests for chat session functionality."""
import pytest

from llming_lodge.session import ChatSession, LLMConfig, Role
from llming_lodge.llm_base_models import ChatHistory, ChatMessage
from llming_lodge.llm_base_client import LlmClient
from llming_lodge.llm_provider_manager import LLMManager
from llming_lodge.messages import LlmAIMessage, LlmHumanMessage, LlmSystemMessage, LlmMessageChunk
from .conftest import BASIC_MODELS


@pytest.fixture
def manager():
    """Create LLM manager for tests."""
    return LLMManager()

@pytest.mark.parametrize("provider,model", BASIC_MODELS)
def test_create_with_history(manager, provider, model):
    """Test creating session with existing history."""
    if provider not in manager.providers:
        pytest.skip(f"Provider {provider} not available (no API key)")

    config = manager.get_config_for_model(model)
    history = ChatHistory()
    history.add_message(ChatMessage(role=Role.USER, content="Hello"))
    history.add_message(ChatMessage(role=Role.ASSISTANT, content="Hi there!"))

    session = ChatSession.create_with_history(
        config=config,
        history=history,
        system_prompt="You are a helpful assistant."
    )

    assert len(session.get_history().messages) == 2
    assert session.get_history().messages[0].role == Role.USER
    assert session.get_history().messages[1].role == Role.ASSISTANT
    assert session.system_prompt == "You are a helpful assistant."

@pytest.mark.asyncio
@pytest.mark.parametrize("provider,model", BASIC_MODELS)
async def test_chat_streaming(manager, provider, model):
    """Test streaming chat across all providers."""
    if provider not in manager.providers:
        pytest.skip(f"Provider {provider} not available (no API key)")

    config = manager.get_config_for_model(model)
    session = ChatSession(
        config=config,
        system_prompt="Reply in max 5 words."
    )

    full_response = ""
    chunk_count = 0
    async for chunk in await session.chat_async("Say hello", streaming=True):
        assert isinstance(chunk, LlmMessageChunk)
        full_response += chunk.content
        chunk_count += 1

    assert chunk_count > 0
    assert len(full_response) > 0

    history = session.get_history()
    assert len(history.messages) == 2  # User + Assistant
    assert history.messages[0].role == Role.USER
    assert history.messages[0].content == "Say hello"
    assert history.messages[1].role == Role.ASSISTANT
    assert len(history.messages[1].content) > 0

@pytest.mark.parametrize("provider,model", BASIC_MODELS)
def test_prepare_messages(manager, provider, model):
    """Test message preparation."""
    if provider not in manager.providers:
        pytest.skip(f"Provider {provider} not available (no API key)")

    config = manager.get_config_for_model(model)
    session = ChatSession(
        config=config,
        system_prompt="You are a helpful assistant."
    )

    session.add_message(Role.USER, "Hello!")
    session.add_message(Role.ASSISTANT, "Hi there!")

    messages = session._prepare_messages()
    assert len(messages) == 3  # System + User + Assistant
    assert messages[0].content == "You are a helpful assistant."
    assert messages[1].content == "Hello!"
    assert messages[2].content == "Hi there!"

    messages = session._prepare_messages(system_prompt="Different prompt")
    assert len(messages) == 3
    assert messages[0].content == "Different prompt"

@pytest.mark.parametrize("provider,model", BASIC_MODELS)
def test_init_with_system_prompt(manager, provider, model):
    """Test session initialization with system prompt."""
    if provider not in manager.providers:
        pytest.skip(f"Provider {provider} not available (no API key)")

    config = manager.get_config_for_model(model)
    session = ChatSession(
        config=config,
        system_prompt="You are a helpful assistant."
    )

    history = session.get_history()
    assert len(history.messages) == 0
    assert session.system_prompt == "You are a helpful assistant."

    messages = session._prepare_messages()
    assert len(messages) == 1
    assert isinstance(messages[0], LlmSystemMessage)
    assert messages[0].content == "You are a helpful assistant."

@pytest.mark.parametrize("provider,model", BASIC_MODELS)
def test_copy_history(manager, provider, model):
    """Test copying history between sessions."""
    if provider not in manager.providers:
        pytest.skip(f"Provider {provider} not available (no API key)")

    config = manager.get_config_for_model(model)
    session1 = ChatSession(
        config=config,
        system_prompt="You are a helpful assistant."
    )
    session1.add_message(Role.USER, "Hello!")
    session1.add_message(Role.ASSISTANT, "Hi there!")

    session2 = ChatSession(
        config=config,
        system_prompt="Different system prompt"
    )
    session2.copy_history_from(session1)

    history1 = session1.get_history()
    history2 = session2.get_history()

    assert len(history1.messages) == len(history2.messages)
    for msg1, msg2 in zip(history1.messages, history2.messages):
        assert msg1.role == msg2.role
        assert msg1.content == msg2.content

    assert session1.system_prompt != session2.system_prompt

def test_invalid_provider():
    """Test initialization with invalid provider."""
    config = LLMConfig(
        provider="invalid",
        model="test"
    )
    with pytest.raises(ValueError) as exc_info:
        ChatSession(config=config)
        assert "Provider invalid not registered" in str(exc_info.value)
