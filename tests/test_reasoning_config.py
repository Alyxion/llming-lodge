"""Tests for reasoning effort configuration."""
import os
import pytest
from unittest.mock import MagicMock, patch

from llming_lodge.providers.llm_provider_models import ReasoningEffort, LLMInfo, ModelSize
from llming_lodge.providers.openai.openai_client import OpenAILlmClient
from llming_lodge.providers.openai.openai_models import OPENAI_MODELS
from llming_lodge.session import LLMConfig, ChatSession
from llming_lodge.messages import LlmSystemMessage, LlmHumanMessage


class TestReasoningKwargs:
    """Test _get_reasoning_kwargs method."""

    def test_no_reasoning_effort_returns_empty_dict(self):
        """When reasoning_effort is None, should return empty dict."""
        with patch.object(OpenAILlmClient, '__init__', lambda self, **kwargs: None):
            client = OpenAILlmClient()
            client.reasoning_effort = None
            client.toolboxes = []  # No tools
            assert client._get_reasoning_kwargs() == {}

    def test_reasoning_effort_none_maps_to_minimal(self):
        """When reasoning_effort is NONE, should map to 'minimal' (GPT-5 doesn't support 'none')."""
        with patch.object(OpenAILlmClient, '__init__', lambda self, **kwargs: None):
            client = OpenAILlmClient()
            client.reasoning_effort = ReasoningEffort.NONE
            client.toolboxes = []  # No tools
            result = client._get_reasoning_kwargs()
            # NONE maps to 'minimal' because gpt-5 models don't support 'none', lowest is 'minimal'
            assert result == {"reasoning": {"effort": "minimal"}}

    def test_reasoning_effort_low(self):
        """When reasoning_effort is LOW, should set effort to low."""
        with patch.object(OpenAILlmClient, '__init__', lambda self, **kwargs: None):
            client = OpenAILlmClient()
            client.reasoning_effort = ReasoningEffort.LOW
            client.toolboxes = []  # No tools
            result = client._get_reasoning_kwargs()
            assert result == {"reasoning": {"effort": "low"}}

    def test_reasoning_effort_medium(self):
        """When reasoning_effort is MEDIUM, should set effort to medium."""
        with patch.object(OpenAILlmClient, '__init__', lambda self, **kwargs: None):
            client = OpenAILlmClient()
            client.reasoning_effort = ReasoningEffort.MEDIUM
            client.toolboxes = []  # No tools
            result = client._get_reasoning_kwargs()
            assert result == {"reasoning": {"effort": "medium"}}

    def test_reasoning_effort_high(self):
        """When reasoning_effort is HIGH, should set effort to high."""
        with patch.object(OpenAILlmClient, '__init__', lambda self, **kwargs: None):
            client = OpenAILlmClient()
            client.reasoning_effort = ReasoningEffort.HIGH
            client.toolboxes = []  # No tools
            result = client._get_reasoning_kwargs()
            assert result == {"reasoning": {"effort": "high"}}

    def test_reasoning_effort_minimal(self):
        """When reasoning_effort is MINIMAL, should set effort to minimal."""
        with patch.object(OpenAILlmClient, '__init__', lambda self, **kwargs: None):
            client = OpenAILlmClient()
            client.reasoning_effort = ReasoningEffort.MINIMAL
            client.toolboxes = []  # No tools
            result = client._get_reasoning_kwargs()
            assert result == {"reasoning": {"effort": "minimal"}}

    def test_reasoning_increased_when_tools_present_and_minimal(self):
        """When tools are present and reasoning is 'minimal', it should be increased to 'low'."""
        from llming_lodge.tools.llm_toolbox import LlmToolbox

        with patch.object(OpenAILlmClient, '__init__', lambda self, **kwargs: None):
            client = OpenAILlmClient()
            client.reasoning_effort = ReasoningEffort.MINIMAL  # 'minimal' conflicts with tools
            # Add web_search tool which conflicts with minimal reasoning
            client.toolboxes = [LlmToolbox(name="search", description="Search tools", tools=["web_search"])]
            result = client._get_reasoning_kwargs()
            # Should be increased to 'low' because 'minimal' doesn't work with tools
            assert result == {"reasoning": {"effort": "low"}}

    def test_reasoning_increased_when_tools_present_and_none(self):
        """When tools are present and reasoning is NONE (maps to 'minimal'), it should be increased to 'low'."""
        from llming_lodge.tools.llm_toolbox import LlmToolbox
        from llming_lodge.tools.llm_tool import LlmTool

        with patch.object(OpenAILlmClient, '__init__', lambda self, **kwargs: None):
            client = OpenAILlmClient()
            client.reasoning_effort = ReasoningEffort.NONE  # Maps to 'minimal', conflicts with tools
            # Add a custom function tool
            tool = LlmTool(
                name="test_tool",
                description="A test tool",
                parameters={"type": "object", "properties": {}},
                func=lambda: "test"
            )
            client.toolboxes = [LlmToolbox(name="custom", description="Custom tools", tools=[tool])]
            result = client._get_reasoning_kwargs()
            # Should be increased to 'low' because 'minimal' doesn't work with tools
            assert result == {"reasoning": {"effort": "low"}}

    def test_reasoning_kept_when_tools_present_and_higher_than_minimal(self):
        """When tools are present but reasoning is already 'low' or higher, it should not change."""
        from llming_lodge.tools.llm_toolbox import LlmToolbox

        with patch.object(OpenAILlmClient, '__init__', lambda self, **kwargs: None):
            client = OpenAILlmClient()
            client.reasoning_effort = ReasoningEffort.HIGH  # Already high, no need to change
            client.toolboxes = [LlmToolbox(name="search", description="Search tools", tools=["web_search"])]
            result = client._get_reasoning_kwargs()
            # Should stay 'high' because it's already above 'minimal'
            assert result == {"reasoning": {"effort": "high"}}


class TestModelDefaults:
    """Test that model defaults are correctly configured."""

    def test_gpt5_nano_has_reasoning_disabled(self):
        """GPT-5 Nano should have reasoning disabled by default for speed."""
        nano_model = next((m for m in OPENAI_MODELS if m.name == "gpt-5-nano"), None)
        assert nano_model is not None
        assert nano_model.reasoning is True  # It's a reasoning model
        assert nano_model.default_reasoning_effort == ReasoningEffort.NONE  # But reasoning is disabled by default

    def test_gpt5_mini_has_minimal_reasoning(self):
        """GPT-5 Mini should have minimal reasoning by default."""
        mini_model = next((m for m in OPENAI_MODELS if m.name == "gpt-5-mini"), None)
        assert mini_model is not None
        assert mini_model.reasoning is True
        assert mini_model.default_reasoning_effort == ReasoningEffort.MINIMAL

    def test_gpt5_2_has_low_reasoning(self):
        """GPT-5.2 should have low reasoning by default."""
        gpt5_model = next((m for m in OPENAI_MODELS if m.name == "gpt-5.2"), None)
        assert gpt5_model is not None
        assert gpt5_model.reasoning is True
        assert gpt5_model.default_reasoning_effort == ReasoningEffort.LOW


class TestLLMConfig:
    """Test LLMConfig with reasoning_effort."""

    def test_config_default_reasoning_effort_is_none(self):
        """LLMConfig should have reasoning_effort as None by default."""
        config = LLMConfig(provider="openai", model="gpt-5-nano")
        assert config.reasoning_effort is None

    def test_config_accepts_reasoning_effort(self):
        """LLMConfig should accept reasoning_effort parameter."""
        config = LLMConfig(
            provider="openai",
            model="gpt-5-nano",
            reasoning_effort=ReasoningEffort.HIGH
        )
        assert config.reasoning_effort == ReasoningEffort.HIGH

    def test_config_model_dump_includes_reasoning(self):
        """model_dump should include reasoning_effort field."""
        config = LLMConfig(
            provider="openai",
            model="gpt-5-nano",
            reasoning_effort=ReasoningEffort.LOW
        )
        dump = config.model_dump()
        assert "reasoning_effort" in dump
        assert dump["reasoning_effort"] == ReasoningEffort.LOW


class TestReasoningEffortEnum:
    """Test ReasoningEffort enum values."""

    def test_enum_values(self):
        """Check all enum values are correct strings."""
        assert ReasoningEffort.NONE.value == "none"
        assert ReasoningEffort.MINIMAL.value == "minimal"
        assert ReasoningEffort.LOW.value == "low"
        assert ReasoningEffort.MEDIUM.value == "medium"
        assert ReasoningEffort.HIGH.value == "high"

    def test_enum_is_string_subclass(self):
        """ReasoningEffort should be a string enum for easy serialization."""
        assert isinstance(ReasoningEffort.LOW, str)
        assert ReasoningEffort.LOW == "low"


# Integration tests (require API key)
@pytest.fixture(scope="module")
def openai_api_key():
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        pytest.skip("OPENAI_API_KEY environment variable not set")
    return key


@pytest.fixture(scope="module")
def openai_model():
    # Use a default model, can be overridden by env var
    return os.environ.get("OPENAI_MODEL", "gpt-5.2")


class TestReasoningIntegration:
    """Integration tests for reasoning configuration with real API."""

    def test_client_accepts_reasoning_effort(self, openai_api_key, openai_model):
        """OpenAILlmClient should accept reasoning_effort parameter."""
        client = OpenAILlmClient(
            api_key=openai_api_key,
            model=openai_model,
            temperature=1.0,
            reasoning_effort=ReasoningEffort.NONE
        )
        assert client.reasoning_effort == ReasoningEffort.NONE

    def test_client_accepts_reasoning_effort_low(self, openai_api_key, openai_model):
        """OpenAILlmClient should accept LOW reasoning_effort parameter."""
        client = OpenAILlmClient(
            api_key=openai_api_key,
            model=openai_model,
            temperature=1.0,
            reasoning_effort=ReasoningEffort.LOW
        )
        assert client.reasoning_effort == ReasoningEffort.LOW

    def test_invoke_with_reasoning_low(self, openai_api_key, openai_model):
        """Test synchronous invoke with LOW reasoning."""
        client = OpenAILlmClient(
            api_key=openai_api_key,
            model=openai_model,
            temperature=1.0,
            max_tokens=100,
            reasoning_effort=ReasoningEffort.LOW
        )

        messages = [
            LlmSystemMessage(content="Answer briefly."),
            LlmHumanMessage(content="What is the capital of France?"),
        ]

        result = client.invoke(messages)
        assert result.content is not None
        assert len(result.content) > 0
        assert "paris" in result.content.lower()

    @pytest.mark.asyncio
    async def test_ainvoke_with_reasoning_low(self, openai_api_key, openai_model):
        """Test async invoke with LOW reasoning."""
        client = OpenAILlmClient(
            api_key=openai_api_key,
            model=openai_model,
            temperature=1.0,
            max_tokens=100,
            reasoning_effort=ReasoningEffort.LOW
        )

        messages = [
            LlmSystemMessage(content="Answer briefly."),
            LlmHumanMessage(content="What is 2+2?"),
        ]

        result = await client.ainvoke(messages)
        assert result.content is not None
        assert len(result.content) > 0
        assert "4" in result.content

    @pytest.mark.asyncio
    async def test_astream_with_reasoning_low(self, openai_api_key, openai_model):
        """Test async streaming with LOW reasoning."""
        client = OpenAILlmClient(
            api_key=openai_api_key,
            model=openai_model,
            temperature=1.0,
            max_tokens=100,
            reasoning_effort=ReasoningEffort.LOW
        )

        messages = [
            LlmSystemMessage(content="Answer briefly."),
            LlmHumanMessage(content="Say hello."),
        ]

        chunks = []
        async for chunk in client.astream(messages):
            chunks.append(chunk.content)

        full_response = "".join(chunks)
        assert len(full_response) > 0
        # Should contain some greeting
        assert any(word in full_response.lower() for word in ["hello", "hi", "hey", "greetings"])
