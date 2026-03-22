"""Integration test for Claude Opus 4.6 via Azure AI Services (Foundry)."""

import os

import pytest

ENDPOINT = os.environ.get("AZURE_AI_SERVICES_ENDPOINT", "")
API_KEY = os.environ.get("AZURE_AI_SERVICES_KEY", "")

requires_azure_claude = pytest.mark.skipif(
    not ENDPOINT or not API_KEY,
    reason="AZURE_AI_SERVICES_ENDPOINT and AZURE_AI_SERVICES_KEY must be set",
)


@requires_azure_claude
def test_azure_claude_opus_chat_completion():
    """Send a simple chat completion to Claude Opus 4.6 via Azure AI Services."""
    from anthropic import AnthropicFoundry

    client = AnthropicFoundry(
        api_key=API_KEY,
        base_url=ENDPOINT.rstrip("/") + "/anthropic/",
    )

    message = client.messages.create(
        model="claude-opus-4-6-2",
        max_tokens=64,
        messages=[{"role": "user", "content": "Reply with exactly: AZURE_CLAUDE_OK"}],
    )

    assert message.content, "Expected at least one content block"
    text = message.content[0].text
    assert "AZURE_CLAUDE_OK" in text, f"Unexpected response: {text}"
    assert message.model, "Model field should be populated"
    print(f"Model: {message.model}, Response: {text}")


@requires_azure_claude
def test_azure_anthropic_provider_creates_client():
    """Verify AzureAnthropicProvider creates a working client."""
    from llming_models.providers.azure_anthropic.azure_anthropic_provider import (
        AzureAnthropicProvider,
    )
    from llming_models.messages import LlmHumanMessage

    provider = AzureAnthropicProvider()
    assert provider.is_available

    client = provider.create_client(
        model="claude-opus-4-6-2",
        temperature=0.0,
        max_tokens=64,
    )

    response = client.invoke(
        [LlmHumanMessage(content="Reply with exactly: PROVIDER_TEST_OK")]
    )
    assert "PROVIDER_TEST_OK" in response.content, f"Unexpected: {response.content}"
    print(f"Provider test response: {response.content}")


def test_azure_anthropic_provider_unavailable_without_env():
    """Provider reports unavailable when env vars are missing."""
    from unittest.mock import patch

    from llming_models.providers.azure_anthropic.azure_anthropic_provider import (
        AzureAnthropicProvider,
    )

    with patch.dict(os.environ, {}, clear=True):
        provider = AzureAnthropicProvider()
        assert not provider.is_available


def test_azure_anthropic_in_cascade():
    """azure_anthropic appears before anthropic in the default cascade."""
    from llming_models.config.llm_global_config import LLMGlobalConfig

    config = LLMGlobalConfig()
    cascade = config.provider_cascade
    assert "azure_anthropic" in cascade
    assert "anthropic" in cascade
    assert cascade.index("azure_anthropic") < cascade.index("anthropic")


def test_deployments_env_parsing():
    """AZURE_ANTHROPIC_DEPLOYMENTS env var configures available models."""
    from unittest.mock import patch
    from llming_models.providers.azure_anthropic.azure_anthropic_models import (
        get_azure_anthropic_models,
    )

    with patch.dict(os.environ, {
        "AZURE_ANTHROPIC_DEPLOYMENTS": "claude_opus=my-opus,claude_haiku=my-opus"
    }):
        models = get_azure_anthropic_models()

    names = [m.name for m in models]
    assert "claude_opus" in names
    assert "claude_haiku" in names
    assert "claude_sonnet" not in names
    # Both route to same deployment
    assert all(m.model == "my-opus" for m in models)


def test_deployments_env_empty():
    """No models when AZURE_ANTHROPIC_DEPLOYMENTS is unset."""
    from unittest.mock import patch
    from llming_models.providers.azure_anthropic.azure_anthropic_models import (
        get_azure_anthropic_models,
    )

    with patch.dict(os.environ, {}, clear=True):
        models = get_azure_anthropic_models()

    assert models == []
