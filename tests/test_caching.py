"""Tests for prompt caching and cached-token cost accounting."""
import pytest

from llming_models.providers.anthropic.anthropic_client import AnthropicClient


# =========================================================================
# AnthropicClient._mark_cache_control
# =========================================================================

class TestMarkCacheControl:
    """Unit tests for the static _mark_cache_control helper."""

    def test_string_content_converted_to_list(self):
        msg = {"role": "user", "content": "hello"}
        AnthropicClient._mark_cache_control(msg)
        assert isinstance(msg["content"], list)
        assert len(msg["content"]) == 1
        block = msg["content"][0]
        assert block["type"] == "text"
        assert block["text"] == "hello"
        assert block["cache_control"] == {"type": "ephemeral"}

    def test_list_content_last_block_tagged(self):
        msg = {"role": "user", "content": [
            {"type": "text", "text": "part1"},
            {"type": "text", "text": "part2"},
        ]}
        AnthropicClient._mark_cache_control(msg)
        # First block untouched
        assert "cache_control" not in msg["content"][0]
        # Last block gets cache_control
        assert msg["content"][1]["cache_control"] == {"type": "ephemeral"}

    def test_single_element_list(self):
        msg = {"role": "user", "content": [{"type": "text", "text": "only"}]}
        AnthropicClient._mark_cache_control(msg)
        assert msg["content"][0]["cache_control"] == {"type": "ephemeral"}

    def test_empty_list_is_noop(self):
        msg = {"role": "user", "content": []}
        AnthropicClient._mark_cache_control(msg)
        assert msg["content"] == []

    def test_image_block_gets_tagged(self):
        """cache_control is added to whatever the last block is, even images."""
        msg = {"role": "user", "content": [
            {"type": "text", "text": "describe this"},
            {"type": "image", "source": {"type": "base64", "data": "abc"}},
        ]}
        AnthropicClient._mark_cache_control(msg)
        assert msg["content"][-1]["cache_control"] == {"type": "ephemeral"}

    def test_idempotent(self):
        """Calling twice doesn't double-wrap or corrupt."""
        msg = {"role": "user", "content": "hello"}
        AnthropicClient._mark_cache_control(msg)
        AnthropicClient._mark_cache_control(msg)
        assert len(msg["content"]) == 1
        assert msg["content"][0]["cache_control"] == {"type": "ephemeral"}


# =========================================================================
# AnthropicClient._apply_cache_control
# =========================================================================

class TestApplyCacheControl:
    """Unit tests for the static _apply_cache_control strategy."""

    def test_empty_messages_is_noop(self):
        msgs = []
        AnthropicClient._apply_cache_control(msgs)
        assert msgs == []

    def test_single_message_gets_breakpoint(self):
        msgs = [{"role": "user", "content": "hi"}]
        AnthropicClient._apply_cache_control(msgs)
        assert isinstance(msgs[0]["content"], list)
        assert msgs[0]["content"][0]["cache_control"] == {"type": "ephemeral"}

    def test_two_messages_only_first_tagged(self):
        """With exactly 2 messages, only breakpoint 1 (first) is set.
        The penultimate IS the first, so no second breakpoint needed."""
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        AnthropicClient._apply_cache_control(msgs)
        # First message tagged
        assert isinstance(msgs[0]["content"], list)
        assert msgs[0]["content"][0]["cache_control"] == {"type": "ephemeral"}
        # Second message NOT tagged (it's the last, not penultimate)
        assert isinstance(msgs[1]["content"], str)

    def test_three_messages_first_and_penultimate(self):
        msgs = [
            {"role": "user", "content": "msg1"},
            {"role": "assistant", "content": "msg2"},
            {"role": "user", "content": "msg3"},
        ]
        AnthropicClient._apply_cache_control(msgs)
        # First message tagged
        assert msgs[0]["content"][0]["cache_control"] == {"type": "ephemeral"}
        # Penultimate (index 1) tagged
        assert msgs[1]["content"][0]["cache_control"] == {"type": "ephemeral"}
        # Last message NOT tagged
        assert isinstance(msgs[2]["content"], str)

    def test_five_messages_first_and_penultimate(self):
        msgs = [
            {"role": "user", "content": f"msg{i}"}
            for i in range(5)
        ]
        AnthropicClient._apply_cache_control(msgs)
        # First tagged
        assert msgs[0]["content"][0]["cache_control"] == {"type": "ephemeral"}
        # Penultimate (index 3) tagged
        assert msgs[3]["content"][0]["cache_control"] == {"type": "ephemeral"}
        # Others untouched
        for idx in [1, 2, 4]:
            assert isinstance(msgs[idx]["content"], str), f"msg[{idx}] should be untouched"

    def test_condensed_scenario(self):
        """After condensation, the first message is the condensed summary.
        It should still get the first breakpoint."""
        msgs = [
            {"role": "user", "content": "[Condensed summary of 200 messages...]"},
            {"role": "assistant", "content": "I understand."},
            {"role": "user", "content": "New question"},
        ]
        AnthropicClient._apply_cache_control(msgs)
        assert msgs[0]["content"][0]["cache_control"] == {"type": "ephemeral"}
        assert msgs[1]["content"][0]["cache_control"] == {"type": "ephemeral"}


# =========================================================================
# Usage callback signature & accumulation
# =========================================================================

class TestUsageCallback:
    """Tests for the usage_callback signature used by session.py."""

    def test_callback_accumulates_cached_tokens(self):
        live_usage = {'input_tokens': 0, 'output_tokens': 0, 'cached_input_tokens': 0}

        def usage_callback(input_tokens, output_tokens, cached_input_tokens=0):
            live_usage['input_tokens'] += input_tokens
            live_usage['output_tokens'] += output_tokens
            live_usage['cached_input_tokens'] += cached_input_tokens

        # Simulate two API iterations
        usage_callback(10000, 500, cached_input_tokens=7000)
        usage_callback(10500, 300, cached_input_tokens=9800)

        assert live_usage['input_tokens'] == 20500
        assert live_usage['output_tokens'] == 800
        assert live_usage['cached_input_tokens'] == 16800

    def test_callback_default_zero_cached(self):
        """When provider doesn't report cache info, cached_input_tokens defaults to 0."""
        live_usage = {'input_tokens': 0, 'output_tokens': 0, 'cached_input_tokens': 0}

        def usage_callback(input_tokens, output_tokens, cached_input_tokens=0):
            live_usage['input_tokens'] += input_tokens
            live_usage['output_tokens'] += output_tokens
            live_usage['cached_input_tokens'] += cached_input_tokens

        # Called without cached_input_tokens (backward compat)
        usage_callback(5000, 200)
        assert live_usage['cached_input_tokens'] == 0
        assert live_usage['input_tokens'] == 5000


# =========================================================================
# Cost calculation with cached tokens
# =========================================================================

class TestCachedCostCalculation:
    """Tests for the cost formula that accounts for cached input tokens."""

    @staticmethod
    def _compute_cost(
        input_tokens: int,
        output_tokens: int,
        cached_input_tokens: int,
        input_price: float,
        cached_price: float,
        output_price: float,
    ) -> float:
        """Replicate the session.py cost formula."""
        effective_cached_price = cached_price or input_price
        non_cached = input_tokens - cached_input_tokens
        return (
            non_cached * (input_price / 1_000_000)
            + cached_input_tokens * (effective_cached_price / 1_000_000)
            + output_tokens * (output_price / 1_000_000)
        )

    def test_no_caching_same_as_original(self):
        """With 0 cached tokens, formula matches the original."""
        # Original: input * input_price + output * output_price
        cost = self._compute_cost(
            input_tokens=10000, output_tokens=500, cached_input_tokens=0,
            input_price=3.00, cached_price=0.30, output_price=15.00,
        )
        expected = 10000 * (3.00 / 1e6) + 500 * (15.00 / 1e6)
        assert abs(cost - expected) < 1e-12

    def test_full_cache_uses_cached_price(self):
        """When all input is cached, uses cached_input_token_price."""
        cost = self._compute_cost(
            input_tokens=10000, output_tokens=500, cached_input_tokens=10000,
            input_price=3.00, cached_price=0.30, output_price=15.00,
        )
        expected = 10000 * (0.30 / 1e6) + 500 * (15.00 / 1e6)
        assert abs(cost - expected) < 1e-12

    def test_partial_cache_blended(self):
        """Mix of cached and non-cached input tokens."""
        cost = self._compute_cost(
            input_tokens=10000, output_tokens=500, cached_input_tokens=7000,
            input_price=3.00, cached_price=0.30, output_price=15.00,
        )
        expected = (
            3000 * (3.00 / 1e6)    # non-cached
            + 7000 * (0.30 / 1e6)  # cached
            + 500 * (15.00 / 1e6)  # output
        )
        assert abs(cost - expected) < 1e-12

    def test_cached_price_fallback_to_input_price(self):
        """When cached_input_token_price is 0, fall back to input_token_price."""
        cost = self._compute_cost(
            input_tokens=10000, output_tokens=500, cached_input_tokens=7000,
            input_price=3.00, cached_price=0.0, output_price=15.00,
        )
        # All input at full price (no savings)
        expected = 10000 * (3.00 / 1e6) + 500 * (15.00 / 1e6)
        assert abs(cost - expected) < 1e-12

    def test_savings_anthropic_sonnet(self):
        """Sonnet 4.5: $3.00 input, $0.30 cached, $15.00 output.
        With 70% cache hit, should save ~63% on input costs."""
        full_cost = self._compute_cost(
            input_tokens=10000, output_tokens=500, cached_input_tokens=0,
            input_price=3.00, cached_price=0.30, output_price=15.00,
        )
        cached_cost = self._compute_cost(
            input_tokens=10000, output_tokens=500, cached_input_tokens=7000,
            input_price=3.00, cached_price=0.30, output_price=15.00,
        )
        input_savings = 1.0 - (cached_cost / full_cost)
        assert input_savings > 0.40  # Substantial savings

    def test_savings_openai_gpt5(self):
        """GPT-5.2: $1.75 input, $0.175 cached, $14.00 output.
        With 80% cache hit, should save significantly."""
        full_cost = self._compute_cost(
            input_tokens=50000, output_tokens=1000, cached_input_tokens=0,
            input_price=1.75, cached_price=0.175, output_price=14.00,
        )
        cached_cost = self._compute_cost(
            input_tokens=50000, output_tokens=1000, cached_input_tokens=40000,
            input_price=1.75, cached_price=0.175, output_price=14.00,
        )
        savings_pct = (1.0 - cached_cost / full_cost) * 100
        assert savings_pct > 40  # Should be significant

    def test_all_models_have_cached_price(self):
        """All model configs should have cached_input_token_price set."""
        from llming_models.providers.anthropic.anthropic_models import ANTHROPIC_MODELS
        from llming_models.providers.openai.openai_models import OPENAI_MODELS

        for model in ANTHROPIC_MODELS:
            assert model.cached_input_token_price > 0, (
                f"{model.label} missing cached_input_token_price"
            )
            assert model.cached_input_token_price < model.input_token_price, (
                f"{model.label}: cached price should be less than input price"
            )

        for model in OPENAI_MODELS:
            assert model.cached_input_token_price > 0, (
                f"{model.label} missing cached_input_token_price"
            )
            assert model.cached_input_token_price < model.input_token_price, (
                f"{model.label}: cached price should be less than input price"
            )


# =========================================================================
# _build_kwargs cache integration
# =========================================================================

class TestBuildKwargsCaching:
    """Verify _build_kwargs sets up cache_control on system + messages."""

    @pytest.fixture
    def _patch_env(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")

    def test_system_prompt_has_cache_control(self, _patch_env):
        from llming_models.messages import LlmSystemMessage, LlmHumanMessage
        client = AnthropicClient(api_key="test-key-not-real", model="claude-sonnet-4-6")
        messages = [
            LlmSystemMessage(content="You are helpful."),
            LlmHumanMessage(content="Hello"),
        ]
        kwargs = client._build_kwargs(messages)

        # System prompt should be structured list with cache_control
        assert isinstance(kwargs["system"], list)
        assert kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}

    def test_messages_have_cache_breakpoints(self, _patch_env):
        from llming_models.messages import LlmSystemMessage, LlmHumanMessage, LlmAIMessage
        client = AnthropicClient(api_key="test-key-not-real", model="claude-sonnet-4-6")
        messages = [
            LlmSystemMessage(content="System prompt"),
            LlmHumanMessage(content="User msg 1"),
            LlmAIMessage(content="Assistant reply"),
            LlmHumanMessage(content="User msg 2"),
            LlmAIMessage(content="Another reply"),
            LlmHumanMessage(content="Latest question"),
        ]
        kwargs = client._build_kwargs(messages)

        # Messages (excludes system prompt)
        msgs = kwargs["messages"]
        assert len(msgs) == 5

        # First message should have cache_control
        first_content = msgs[0]["content"]
        assert isinstance(first_content, list)
        assert first_content[0]["cache_control"] == {"type": "ephemeral"}

        # Penultimate (index 3) should have cache_control
        pen_content = msgs[3]["content"]
        assert isinstance(pen_content, list)
        assert pen_content[0]["cache_control"] == {"type": "ephemeral"}

        # Last message should NOT have cache_control
        last_content = msgs[4]["content"]
        assert isinstance(last_content, str)
