"""Integration tests for prompt caching — hits real APIs.

Requires API keys in .env (loaded via dotenv):
  ANTHROPIC_API_KEY    — Anthropic direct
  OPENAI_API_KEY       — OpenAI direct
  AZURE_OPENAI_API_KEY — Azure OpenAI
  AZURE_OPENAI_ENDPOINT

OpenAI caching is automatic and best-effort.  It needs a large prompt
(well above 1024 tokens) and sometimes 2–3 warm-up calls before the
server-side cache is populated.  GPT-5 models cache more aggressively
than GPT-4.1 models.
"""
import asyncio
import os
import logging
from dataclasses import dataclass, field
from typing import List, Optional

import dotenv
import pytest
import pytest_asyncio

# Load .env from project root (two levels up from tests/)
dotenv.load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env"))

from llming_models.messages import LlmSystemMessage, LlmHumanMessage, LlmMessageChunk
from llming_models.providers.anthropic.anthropic_client import AnthropicClient
from llming_models.providers.openai.openai_client import OpenAILlmClient
from llming_models.session import ChatSession, LLMConfig
from llming_models.budget import LLMBudgetManager, MemoryBudgetLimit, LimitPeriod

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Large, stable text used as a "document" to fill the cache.
# ~20K tokens — large enough for reliable OpenAI automatic caching.
# Must be identical across calls so the cache can match.
# ---------------------------------------------------------------------------
_LARGE_DOCUMENT = (
    "The following is a detailed technical specification for the Acme Widget System v42.\n\n"
    + "\n".join(
        f"Section {i}: This section covers aspect number {i} of the widget architecture. "
        f"The component designated W-{i:04d} interfaces with subsystem S-{i % 7} via protocol P-{i % 3}. "
        f"Performance benchmarks show {100 + i * 3}ms latency under normal load and {200 + i * 5}ms under peak. "
        f"Memory footprint is approximately {10 + i}MB with {i * 2} concurrent connections supported. "
        f"Error rate must stay below {0.01 * (i % 5 + 1):.2f}% for SLA compliance. "
        f"The configuration requires parameters alpha={i * 0.1:.1f}, beta={i * 0.2:.1f}, gamma={i * 0.3:.1f}. "
        f"Redundancy factor is {i % 3 + 1}x with failover timeout of {i * 100}ms."
        for i in range(200)
    )
)


def _make_messages(question: str = "Summarize section 42 in one sentence."):
    """Build a message list with a large system prompt + document + short question."""
    return [
        LlmSystemMessage(content="You are a concise technical assistant. Answer in one sentence."),
        LlmHumanMessage(content=f"Here is the document:\n\n{_LARGE_DOCUMENT}\n\n{question}"),
    ]


# ---------------------------------------------------------------------------
# Helper: stream and collect usage from a client
# ---------------------------------------------------------------------------
async def _stream_and_collect(client, messages):
    """Stream a response and return (text, usage_dict)."""
    usage = {"input_tokens": 0, "output_tokens": 0, "cached_input_tokens": 0}

    def cb(inp, out, cached_input_tokens=0):
        usage["input_tokens"] += inp
        usage["output_tokens"] += out
        usage["cached_input_tokens"] += cached_input_tokens

    text_parts = []
    async for chunk in client.astream(messages, usage_callback=cb):
        if chunk.content:
            text_parts.append(chunk.content)

    return "".join(text_parts), usage


# =========================================================================
# Anthropic
# =========================================================================

@pytest.mark.asyncio
@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set")
async def test_anthropic_cache_reports_tokens():
    """Send the same large prompt twice to Anthropic.
    First call creates the cache, second call should read from it."""
    client = AnthropicClient(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        model="claude-haiku-4-5-20251001",  # cheapest
        max_tokens=100,
    )
    msgs = _make_messages()

    # Call 1: creates cache (cache_creation > 0, cache_read == 0)
    text1, usage1 = await _stream_and_collect(client, msgs)
    assert text1, "Expected a non-empty response"
    assert usage1["input_tokens"] > 0, "Should report input tokens"
    logger.info(f"Anthropic call 1: {usage1}")

    # Call 2: should read from cache (cache_read > 0)
    text2, usage2 = await _stream_and_collect(client, msgs)
    assert text2, "Expected a non-empty response"
    assert usage2["cached_input_tokens"] > 0, (
        f"Second call should have cached tokens, got: {usage2}"
    )
    logger.info(f"Anthropic call 2: {usage2}")

    # Cached tokens should be a substantial portion of input
    cache_pct = usage2["cached_input_tokens"] / usage2["input_tokens"] * 100
    logger.info(f"Anthropic cache hit: {cache_pct:.0f}%")
    assert cache_pct > 50, f"Expected >50% cache hit, got {cache_pct:.0f}%"


# =========================================================================
# OpenAI (direct)
# =========================================================================

@pytest.mark.asyncio
@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
async def test_openai_cache_reports_tokens():
    """Send the same large prompt 3 times to OpenAI.
    OpenAI automatic caching is best-effort; GPT-5-nano caches aggressively
    but may need 1–2 warm-up calls before reporting cached tokens."""
    client = OpenAILlmClient(
        api_key=os.environ["OPENAI_API_KEY"],
        model="gpt-5-nano",
        max_tokens=100,
        api_type="openai",
    )
    msgs = _make_messages()

    usages = []
    for i in range(3):
        _text, usage = await _stream_and_collect(client, msgs)
        logger.info(f"OpenAI call {i + 1}: {usage}")
        usages.append(usage)

    # At least one of the calls should report cached tokens
    any_cached = any(u["cached_input_tokens"] > 0 for u in usages)
    assert any_cached, (
        f"Expected at least one call to report cached tokens, got: {usages}"
    )

    # The best cache hit should cover >50% of input
    best = max(u["cached_input_tokens"] for u in usages)
    total = usages[0]["input_tokens"]
    cache_pct = best / total * 100
    logger.info(f"OpenAI best cache hit: {best}/{total} ({cache_pct:.0f}%)")
    assert cache_pct > 50, f"Expected >50% cache hit, got {cache_pct:.0f}%"


# =========================================================================
# Azure OpenAI
# =========================================================================

@pytest.mark.asyncio
@pytest.mark.skipif(
    not (os.environ.get("AZURE_OPENAI_API_KEY") and os.environ.get("AZURE_OPENAI_ENDPOINT")),
    reason="AZURE_OPENAI_API_KEY or AZURE_OPENAI_ENDPOINT not set",
)
async def test_azure_openai_cache_reports_tokens():
    """Send the same large prompt 3 times via Azure OpenAI.
    Azure automatic caching should report cached tokens after warm-up."""
    client = OpenAILlmClient(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        model="gpt-5-nano",
        max_tokens=100,
        api_type="azure",
        api_version="2025-03-01-preview",
        base_url=os.environ["AZURE_OPENAI_ENDPOINT"],
    )
    msgs = _make_messages()

    usages = []
    for i in range(3):
        _text, usage = await _stream_and_collect(client, msgs)
        logger.info(f"Azure call {i + 1}: {usage}")
        usages.append(usage)

    # At least one of the calls should report cached tokens
    any_cached = any(u["cached_input_tokens"] > 0 for u in usages)
    assert any_cached, (
        f"Expected at least one call to report cached tokens, got: {usages}"
    )

    best = max(u["cached_input_tokens"] for u in usages)
    total = usages[0]["input_tokens"]
    cache_pct = best / total * 100
    logger.info(f"Azure best cache hit: {best}/{total} ({cache_pct:.0f}%)")
    assert cache_pct > 50, f"Expected >50% cache hit, got {cache_pct:.0f}%"


# =========================================================================
# End-to-end cost verification through ChatSession
# =========================================================================

class SpyBudgetLimit(MemoryBudgetLimit):
    """MemoryBudgetLimit that records every log_usage_async call."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.logged_calls: List[dict] = []

    async def log_usage_async(self, **kwargs) -> None:
        self.logged_calls.append(kwargs)


@pytest.mark.asyncio
@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set")
async def test_anthropic_session_cost_uses_cached_price():
    """Full ChatSession pipeline: send the same question twice, verify the
    second call is cheaper because cached tokens use cached_input_token_price.

    Haiku 4.5: $1.00/M input, $0.10/M cached, $5.00/M output.
    """
    spy = SpyBudgetLimit(name="spy", amount=100.0, period=LimitPeriod.TOTAL)
    budget = LLMBudgetManager([spy])

    config = LLMConfig(
        provider="anthropic",
        model="claude-haiku-4-5-20251001",
        max_tokens=100,
        max_input_tokens=200000,
    )
    session = ChatSession(
        config=config,
        system_prompt="You are a concise technical assistant. Answer in one sentence.",
        budget_manager=budget,
    )

    question = f"Here is the document:\n\n{_LARGE_DOCUMENT}\n\nSummarize section 42 in one sentence."

    # Call 1: creates cache
    resp1 = await session.chat_async(question, streaming=True)
    async for _ in resp1:
        pass
    assert len(spy.logged_calls) == 1
    call1 = spy.logged_calls[0]
    logger.info(f"Session call 1: tokens_in={call1['tokens_input']}, "
                f"tokens_out={call1['tokens_output']}, cost={call1['costs']:.6f}")

    # Clear history so the same message is sent fresh (same prompt prefix)
    session.clear_history()

    # Call 2: should hit cache → lower cost
    resp2 = await session.chat_async(question, streaming=True)
    async for _ in resp2:
        pass
    assert len(spy.logged_calls) == 2
    call2 = spy.logged_calls[1]
    logger.info(f"Session call 2: tokens_in={call2['tokens_input']}, "
                f"tokens_out={call2['tokens_output']}, cost={call2['costs']:.6f}")

    # --- Verify the cost formula uses cached pricing ---
    model_info = session.model_info
    inp_price = model_info.input_token_price           # 1.00
    cached_price = model_info.cached_input_token_price  # 0.10
    out_price = model_info.output_token_price           # 5.00

    logger.info(f"Model prices: input=${inp_price}/M, cached=${cached_price}/M, output=${out_price}/M")

    # The Anthropic cache may already be warm from earlier tests, so call 1
    # might ALSO have cached tokens.  We verify the cost formula is correct
    # by checking that at least one call's cost is less than the non-cached formula.
    for i, call in enumerate([call1, call2], 1):
        in_tok = call["tokens_input"]
        out_tok = call["tokens_output"]
        logged_cost = call["costs"]

        # What the OLD (non-cached) formula would compute
        old_formula_cost = in_tok * (inp_price / 1e6) + out_tok * (out_price / 1e6)

        logger.info(f"Call {i}: logged=${logged_cost:.6f}, non_cached_formula=${old_formula_cost:.6f}")

        # The logged cost should NEVER exceed the non-cached formula
        # (caching can only reduce cost, never increase it)
        assert logged_cost <= old_formula_cost + 1e-9, (
            f"Call {i}: logged cost ${logged_cost:.6f} should not exceed "
            f"non-cached formula ${old_formula_cost:.6f}"
        )

    # At least one call should be SUBSTANTIALLY cheaper than non-cached
    # (proving cached_input_token_price is being used)
    best_savings = 0.0
    for call in [call1, call2]:
        in_tok = call["tokens_input"]
        out_tok = call["tokens_output"]
        old_cost = in_tok * (inp_price / 1e6) + out_tok * (out_price / 1e6)
        if old_cost > 0:
            savings = (1 - call["costs"] / old_cost) * 100
            best_savings = max(best_savings, savings)

    logger.info(f"Best savings vs non-cached formula: {best_savings:.0f}%")
    assert best_savings > 30, (
        f"Expected >30% savings from caching, got {best_savings:.0f}%. "
        f"Calls: {[c['costs'] for c in [call1, call2]]}"
    )
