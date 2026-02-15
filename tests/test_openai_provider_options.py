import pytest

from llming_lodge.providers.openai.openai_provider import OpenAIProvider
from llming_lodge.providers.openai.openai_client import OpenAILlmClient
from llming_lodge.providers.llm_provider_models import ReasoningEffort
from llming_lodge.messages import LlmSystemMessage, LlmHumanMessage
from llming_lodge.tools.llm_toolbox import LlmToolbox


"""
Tests for GPT-5 options and tool handling using the real OpenAI client.
Requires OPENAI_API_KEY to be set; tests skip if not configured.
"""


@pytest.fixture
def provider():
    # Use real key loaded by tests/__init__.py
    return OpenAIProvider()


def test_models_list_contains_gpt5_variants(provider):
    names = {m.model for m in provider.get_models()}
    assert "gpt-5.2" in names
    assert "gpt-5-mini" in names
    assert "gpt-5-nano" in names


def test_native_client_forwards_gpt5_options(provider):
    # Case A: pass as explicit kwargs via provider.create_client (real client)
    client: OpenAILlmClient = provider.create_client(
        model="gpt-5.2",
        temperature=0.1,
        max_tokens=32,
        streaming=False,
        toolboxes=[],
        reasoning_effort=ReasoningEffort.MEDIUM,
    )

    msgs = [
        LlmSystemMessage(content="You are a helpful assistant."),
        LlmHumanMessage(content="Say hi"),
    ]
    resp = client.invoke(msgs)
    assert hasattr(resp, "content")

    # Case B: with HIGH reasoning
    client2: OpenAILlmClient = provider.create_client(
        model="gpt-5.2",
        temperature=0.2,
        max_tokens=16,
        streaming=False,
        toolboxes=[],
        reasoning_effort=ReasoningEffort.HIGH,
    )
    resp2 = client2.invoke(msgs)
    assert hasattr(resp2, "content")


def test_basic_tool_usage_with_and_without_web_search(provider):
    # Case 1: with web_search tool present
    toolbox_with_search = LlmToolbox(
        name="search_tools",
        description="Web search toolbox",
        tools=["web_search"],
    )
    client: OpenAILlmClient = provider.create_client(
        model="gpt-5-mini",
        temperature=0.0,
        max_tokens=16,

        toolboxes=[toolbox_with_search],
    )
    resp = client.invoke([LlmSystemMessage(content="sys"), LlmHumanMessage(content="q")])
    assert hasattr(resp, "content")

    # Case 2: without web_search present
    client_no_tools: OpenAILlmClient = provider.create_client(
        model="gpt-5-nano",
        temperature=0.0,
        max_tokens=16,

        toolboxes=[],
    )
    resp2 = client_no_tools.invoke([LlmSystemMessage(content="sys"), LlmHumanMessage(content="q")])
    assert hasattr(resp2, "content")


def test_gpt5_nano_with_and_without_web_search(provider):
    # With web_search tool
    toolbox_with_search = LlmToolbox(
        name="nano_search",
        description="Web search toolbox",
        tools=["web_search"],
    )
    client_with: OpenAILlmClient = provider.create_client(
        model="gpt-5-nano",
        temperature=0.0,
        max_tokens=16,

        toolboxes=[toolbox_with_search],
    )
    resp = client_with.invoke([LlmSystemMessage(content="sys"), LlmHumanMessage(content="q")])
    assert hasattr(resp, "content")

    # Without any tools
    client_without: OpenAILlmClient = provider.create_client(
        model="gpt-5-nano",
        temperature=0.0,
        max_tokens=16,

        toolboxes=[],
    )
    resp2 = client_without.invoke([LlmSystemMessage(content="sys"), LlmHumanMessage(content="q")])
    assert hasattr(resp2, "content")

