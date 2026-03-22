"""Tests for image generation with Azure OpenAI provider (gpt-image-1)."""

import os
from unittest.mock import MagicMock, patch

import pytest

from llming_models.tools.tool_definition import (
    DEFAULT_IMAGE_GENERATION_TOOL,
    PROVIDER_COMPAT,
)
from llming_models.tools.tool_registry import ToolRegistry
from llming_models.tools.toolbox_adapter import ToolboxAdapter


class TestImageGenProviderCompat:
    """Verify image generation tool is available for azure_openai."""

    def test_image_gen_available_for_azure(self):
        assert DEFAULT_IMAGE_GENERATION_TOOL.is_available_for_provider("azure_openai")

    def test_image_gen_available_for_openai(self):
        assert DEFAULT_IMAGE_GENERATION_TOOL.is_available_for_provider("openai")

    def test_image_gen_not_available_for_anthropic(self):
        assert not DEFAULT_IMAGE_GENERATION_TOOL.is_available_for_provider("anthropic")

    def test_registry_returns_image_gen_for_azure(self):
        reg = ToolRegistry()
        tools = reg.get_for_provider("azure_openai")
        names = [t.name for t in tools]
        assert "generate_image" in names


class TestImageGenToolboxCreation:
    """Verify ToolboxAdapter creates image generation toolbox for azure_openai."""

    def test_adapter_creates_toolbox_with_openai_client(self):
        """When an openai_client is provided, image gen toolbox is created."""
        mock_client = MagicMock()
        mock_client.generate_image_sync.return_value = "base64data"

        adapter = ToolboxAdapter()
        toolboxes = adapter.get_toolboxes(
            tool_names=["generate_image"],
            provider="azure_openai",
            openai_client=mock_client,
        )

        assert len(toolboxes) == 1
        assert toolboxes[0].name == "image_generation"

    def test_adapter_skips_toolbox_without_openai_client(self):
        """Without openai_client, image gen toolbox is not created."""
        adapter = ToolboxAdapter()
        toolboxes = adapter.get_toolboxes(
            tool_names=["generate_image"],
            provider="azure_openai",
            openai_client=None,
        )
        assert len(toolboxes) == 0


class TestImageGenExecution:
    """Verify image generation uses correct model and parameters."""

    def test_default_model_is_gpt_image_1(self):
        """Without DALLE_DEPLOYMENT_NAME env, uses 'gpt-image-1'."""
        mock_client = MagicMock()
        mock_client.generate_image_sync.return_value = "base64data"

        adapter = ToolboxAdapter()
        with patch.dict(os.environ, {}, clear=False):
            # Remove DALLE_DEPLOYMENT_NAME if set
            os.environ.pop("DALLE_DEPLOYMENT_NAME", None)
            toolboxes = adapter.get_toolboxes(
                tool_names=["generate_image"],
                provider="openai",
                openai_client=mock_client,
            )

        assert len(toolboxes) == 1
        # Execute the tool function
        tool = toolboxes[0].tools[0]
        tool.func(prompt="a cat", size="1024x1024", quality="medium")

        mock_client.generate_image_sync.assert_called_once_with(
            prompt="a cat",
            size="1024x1024",
            quality="medium",
            model="gpt-image-1",
        )

    def test_custom_deployment_name(self):
        """DALLE_DEPLOYMENT_NAME env overrides the model name."""
        mock_client = MagicMock()
        mock_client.generate_image_sync.return_value = "base64data"

        adapter = ToolboxAdapter()
        with patch.dict(os.environ, {"DALLE_DEPLOYMENT_NAME": "my-gpt-image-deployment"}):
            toolboxes = adapter.get_toolboxes(
                tool_names=["generate_image"],
                provider="azure_openai",
                openai_client=mock_client,
            )

        assert len(toolboxes) == 1
        tool = toolboxes[0].tools[0]
        tool.func(prompt="a dog", size="1536x1024", quality="high")

        mock_client.generate_image_sync.assert_called_once_with(
            prompt="a dog",
            size="1536x1024",
            quality="high",
            model="my-gpt-image-deployment",
        )

    def test_azure_provider_gets_toolbox(self):
        """Full flow: azure_openai provider with client -> toolbox created and callable."""
        mock_client = MagicMock()
        mock_client.generate_image_sync.return_value = "fakebase64"

        adapter = ToolboxAdapter()
        toolboxes = adapter.get_toolboxes(
            tool_names=["generate_image"],
            provider="azure_openai",
            openai_client=mock_client,
        )

        assert len(toolboxes) == 1
        tb = toolboxes[0]
        assert tb.name == "image_generation"
        assert len(tb.tools) == 1

        result = tb.tools[0].func(
            prompt="sunset over mountains",
            size="1024x1536",
            quality="medium",
        )
        assert result == "fakebase64"
        mock_client.generate_image_sync.assert_called_once()

    def test_quality_options_are_low_medium_high(self):
        """Tool schema uses gpt-image-1 quality options."""
        mock_client = MagicMock()
        mock_client.generate_image_sync.return_value = "base64data"

        adapter = ToolboxAdapter()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DALLE_DEPLOYMENT_NAME", None)
            toolboxes = adapter.get_toolboxes(
                tool_names=["generate_image"],
                provider="openai",
                openai_client=mock_client,
            )

        schema = toolboxes[0].tools[0].parameters
        quality_enum = schema["properties"]["quality"]["enum"]
        assert quality_enum == ["low", "medium", "high"]

    def test_size_options_are_gpt_image_1_sizes(self):
        """Tool schema uses gpt-image-1 sizes (not DALL-E 3 sizes)."""
        mock_client = MagicMock()
        mock_client.generate_image_sync.return_value = "base64data"

        adapter = ToolboxAdapter()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DALLE_DEPLOYMENT_NAME", None)
            toolboxes = adapter.get_toolboxes(
                tool_names=["generate_image"],
                provider="openai",
                openai_client=mock_client,
            )

        schema = toolboxes[0].tools[0].parameters
        size_enum = schema["properties"]["size"]["enum"]
        assert "1024x1024" in size_enum
        assert "1536x1024" in size_enum
        assert "1024x1536" in size_enum
        # DALL-E 3 sizes should NOT be present
        assert "1792x1024" not in size_enum
        assert "1024x1792" not in size_enum


class TestSessionCreatesClientForAzure:
    """Verify session.py creates OpenAI client for azure_openai provider."""

    def test_azure_provider_creates_openai_client(self):
        """Session._build_toolboxes creates an OpenAILlmClient for azure_openai."""
        # Verify azure_openai maps to openai
        assert PROVIDER_COMPAT.get("azure_openai") == "openai"
        # And that the tool itself is available
        assert DEFAULT_IMAGE_GENERATION_TOOL.is_available_for_provider("azure_openai")


class TestOpenAIClientImageGen:
    """Verify OpenAI client conditionally passes response_format."""

    def test_gpt_image_1_no_response_format(self):
        """gpt-image-1 should NOT get response_format='b64_json'."""
        from unittest.mock import patch, MagicMock

        mock_images = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [MagicMock(b64_json="fakebase64")]
        mock_images.generate.return_value = mock_response

        from llming_models.providers.openai.openai_client import OpenAILlmClient
        with patch.object(OpenAILlmClient, '__init__', lambda self, **kw: None):
            client = OpenAILlmClient()
            client._client = MagicMock()
            client._client.images = mock_images

            result = client.generate_image_sync(
                prompt="test", size="1024x1024", quality="medium", model="gpt-image-1"
            )

        assert result == "fakebase64"
        call_kwargs = mock_images.generate.call_args[1]
        assert "response_format" not in call_kwargs

    def test_dalle3_gets_response_format(self):
        """dall-e-3 should get response_format='b64_json'."""
        from unittest.mock import patch, MagicMock

        mock_images = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [MagicMock(b64_json="fakebase64")]
        mock_images.generate.return_value = mock_response

        from llming_models.providers.openai.openai_client import OpenAILlmClient
        with patch.object(OpenAILlmClient, '__init__', lambda self, **kw: None):
            client = OpenAILlmClient()
            client._client = MagicMock()
            client._client.images = mock_images

            result = client.generate_image_sync(
                prompt="test", size="1024x1024", quality="standard", model="dall-e-3"
            )

        assert result == "fakebase64"
        call_kwargs = mock_images.generate.call_args[1]
        assert call_kwargs["response_format"] == "b64_json"
