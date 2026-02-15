"""Tests for LLM manager functionality."""
import pytest

from llming_lodge.llm_provider_manager import LLMManager
from llming_lodge.session import LLMConfig, ChatSession


@pytest.fixture
def manager(monkeypatch):
    """Test LLM manager using environment variables."""
    # Set test API keys for all providers
    monkeypatch.setenv('OPENAI_API_KEY', 'test_key')
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'test_key')
    monkeypatch.setenv('MISTRAL_API_KEY', 'test_key')
    monkeypatch.setenv('GEMINI_KEY', 'test_key')
    monkeypatch.setenv('TOGETHER_API_KEY', 'test_key')
    monkeypatch.setenv('TOGETHER_API_BASE', 'https://api.together.xyz')
    return LLMManager()

def test_init_with_partial_keys(monkeypatch):
    """Test initialization with partial API keys."""
    # Remove all environment variables first
    for key in ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'MISTRAL_API_KEY', 'GEMINI_KEY', 'TOGETHER_API_KEY', 'TOGETHER_API_BASE']:
        monkeypatch.delenv(key, raising=False)
    
    # Set only some keys
    monkeypatch.setenv('OPENAI_API_KEY', 'test_key')
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'test_key')
    
    manager = LLMManager()
    
    # Should only have OpenAI and Anthropic available
    available_providers = {llm.provider for llm in manager.get_available_llms()}
    assert available_providers == {"openai", "anthropic"}
    
    # Should raise error for unavailable models
    with pytest.raises(ValueError) as exc_info:
        manager.get_config_for_model("mistral_large")
    assert "Model mistral_large not found" in str(exc_info.value)

def test_get_available_llms(manager):
    """Test that only supported LLMs are returned."""
    llms = manager.get_available_llms()
    
    # Verify all supported providers are returned
    providers = {llm.provider for llm in llms}
    expected_providers = {"openai", "anthropic", "mistral", "google", "together"}
    assert providers == expected_providers

def test_get_config_for_invalid_model(manager):
    """Test getting config with invalid model."""
    with pytest.raises(ValueError) as exc_info:
        manager.get_config_for_model("invalid-model")
    assert "Model invalid-model not found" in str(exc_info.value)

def test_get_config_for_models(manager):
    """Test getting config for supported models."""
    for provider in manager.SUPPORTED_PROVIDERS:
        # Get first model for provider
        model = next(
            (info.name for info in manager.get_available_llms() if info.provider == provider),
            None
        )
        if model:
            config = manager.get_config_for_model(model)
            assert isinstance(config, LLMConfig)
            providers = manager.get_providers_for_model(model)
            assert config.provider in providers
            assert config.temperature == 0.7

def test_llm_info_fields(manager):
    """Test that LLM info includes pricing information."""
    llms = manager.get_available_llms()
    for llm in llms:
        assert hasattr(llm, 'input_token_price')
        assert hasattr(llm, 'output_token_price')
        assert isinstance(llm.input_token_price, float)
        assert isinstance(llm.output_token_price, float)
        assert llm.input_token_price > 0
        assert llm.output_token_price > 0

def test_model_consistency(manager):
    """Test that each provider uses the correct model."""
    llms = manager.get_available_llms()
    for llm in llms:
        if llm.provider == "openai":
            # OpenAI models: gpt-5.2, gpt-5.2-mini, gpt-5.2-nano
            assert "gpt-5" in llm.name
        elif llm.provider == "anthropic":
            assert "claude" in llm.name
        elif llm.provider == "mistral":
            assert "mistral" in llm.name
        elif llm.provider == "google":
            assert "gemini" in llm.name
        elif llm.provider == "together":
            assert "deepseek" in llm.name or "together" in llm.name

def test_get_config_with_specific_models(manager):
    """Test getting config with specific models."""
    provider_models = {
        "openai": "gpt-5.2",
        "anthropic": "claude_opus",
        "mistral": "mistral_large",
        "google": "gemini_pro",
        "together": "deepseek-ai/DeepSeek-V3"
    }

    for provider, model in provider_models.items():
        if provider in manager.providers:
            config = manager.get_config_for_model(model)
            assert isinstance(config, LLMConfig)
            providers = manager.get_providers_for_model(model)
            assert config.provider in providers

