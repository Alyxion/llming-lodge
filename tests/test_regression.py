"""Regression tests for LLM functionality."""
import time
import json
import os
import pytest
from datetime import datetime
from pathlib import Path
from llming_lodge.llm_provider_manager import LLMManager


@pytest.fixture
def manager():
    """Create LLM manager."""
    return LLMManager()


@pytest.fixture
def test_results(request):
    """Store test results for finalizer."""
    results = []
    
    def save_results():
        if not results:
            return
            
        # Create logs directory if it doesn't exist
        log_dir = Path("tests/logs")
        log_dir.mkdir(exist_ok=True)
        
        # Write results to JSON file
        log_file = log_dir / "latency_results.json"
        try:
            if log_file.exists():
                with open(log_file) as f:
                    existing_results = json.load(f)
            else:
                existing_results = []
        except json.JSONDecodeError:
            existing_results = []
        
        existing_results.extend(results)
        
        with open(log_file, "w") as f:
            json.dump(existing_results, f, indent=2)
    
    request.addfinalizer(save_results)
    return results


from .conftest import ALL_MODELS


@pytest.mark.regression
@pytest.mark.asyncio
@pytest.mark.timeout(60)  # Increase timeout to 60 seconds for regression tests
@pytest.mark.parametrize("provider,model", ALL_MODELS)
async def test_model_latency(manager, provider, model, test_results):
    """Test each model with a simple ping-pong exchange and measure latency."""
    if provider not in manager.providers:
        pytest.skip(f"Provider {provider} not available (no API key)")
    
    model_info = next(
        (info for info in manager.get_available_llms() 
         if info.provider == provider and (info.name == model or info.model == model)),
        None
    )
    if not model_info:
        pytest.skip(f"Model {model} not found for provider {provider}")
    
    config = manager.get_config_for_model(model_info.name)
    # Limit response length for consistent testing
    config.max_tokens = 100
    # Use system prompt based on model's capabilities and type
    if not model_info.supports_system_prompt:
        system_prompt = None
    elif "deepseek" in model_info.model.lower() or "DeepSeek-R1" in model_info.model:
        system_prompt = "You are a test assistant. Reply with exactly one word. No reasoning, no explanations, just the word 'pong' or 'ping' as requested."
    else:
        system_prompt = "You are a test assistant. Always provide extremely concise, one-word responses."
    session = manager.create_session(config, system_prompt=system_prompt)
    
    # Test sync chat with latency measurement (non-streaming)
    start_time = time.time()
    response = session.chat("Reply with 'pong'", streaming=False)
    sync_latency = time.time() - start_time
    assert isinstance(response, str)
    assert len(response) > 0
    
    # Test async chat with latency measurement (non-streaming)
    start_time = time.time()
    async_response = await session.chat_async("Reply with 'pong'", streaming=False)
    async_non_stream_latency = time.time() - start_time
    assert isinstance(async_response, str)
    assert len(async_response) > 0

    # Test async chat with latency measurement (streaming)
    start_time = time.time()
    full_response = ""
    async for chunk in await session.chat_async("Reply with 'ping'", streaming=True):
        assert isinstance(chunk, str)
        full_response += chunk
    async_stream_latency = time.time() - start_time
    assert len(full_response) > 0
    
    # Store results for finalizer
    results = {
        "model": model_info.model,
        "provider": model_info.provider,
        "sync_latency": round(sync_latency, 2),
        "async_non_stream_latency": round(async_non_stream_latency, 2),
        "async_stream_latency": round(async_stream_latency, 2),
        "sync_response_length": len(response),
        "async_non_stream_length": len(async_response),
        "async_stream_length": len(full_response),
        "timestamp": datetime.now().isoformat()
    }
    test_results.append(results)
    
    print(f"\nModel: {model_info.model}")
    print(f"Sync chat latency: {sync_latency:.2f} seconds")
    print(f"Async non-streaming chat latency: {async_non_stream_latency:.2f} seconds")
    print(f"Async streaming chat latency: {async_stream_latency:.2f} seconds")
    print(f"Response lengths: sync={len(response)}, async_non_stream={len(async_response)}, async_stream={len(full_response)}")
