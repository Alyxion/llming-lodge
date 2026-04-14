"""Tests for Gemini image generation with mocked google.genai."""
from __future__ import annotations

import base64
import builtins
import importlib
import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llming_lodge.tools.gemini_image import (
    GEMINI_IMAGE_PRICING,
    _get_google_api_key,
    generate_image_gemini,
    generate_image_with_gemini,
)


# ---------------------------------------------------------------------------
# GEMINI_IMAGE_PRICING
# ---------------------------------------------------------------------------


class TestGeminiImagePricing:
    """Tests for pricing constants."""

    def test_has_expected_ratios(self):
        expected_keys = ["1:1", "16:9", "9:16", "4:3", "3:4"]
        for key in expected_keys:
            assert key in GEMINI_IMAGE_PRICING, f"Missing key: {key}"

    def test_all_values_positive(self):
        for key, val in GEMINI_IMAGE_PRICING.items():
            assert isinstance(val, float)
            assert val > 0


# ---------------------------------------------------------------------------
# _get_google_api_key
# ---------------------------------------------------------------------------


class TestGetGoogleApiKey:
    """Tests for _get_google_api_key helper."""

    def test_explicit_key_returned(self):
        key = _get_google_api_key(api_key="my-explicit-key")
        assert key == "my-explicit-key"

    def test_explicit_key_takes_precedence(self):
        """Explicit key wins over client and env."""
        mock_client = MagicMock()
        mock_client._client = MagicMock()
        mock_client._client.google_api_key = "client-key"

        with patch.dict("os.environ", {"GEMINI_KEY": "env-key"}):
            key = _get_google_api_key(google_client=mock_client, api_key="explicit-key")
        assert key == "explicit-key"

    def test_from_google_client(self):
        mock_client = MagicMock()
        mock_client._client = MagicMock()
        mock_client._client.google_api_key = "client-api-key"

        with patch.dict("os.environ", {}, clear=True):
            key = _get_google_api_key(google_client=mock_client)
        assert key == "client-api-key"

    def test_from_gemini_key_env(self):
        with patch.dict("os.environ", {"GEMINI_KEY": "gemini-env-key"}, clear=True):
            key = _get_google_api_key()
        assert key == "gemini-env-key"

    def test_from_google_api_key_env(self):
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "google-env-key"}, clear=True):
            key = _get_google_api_key()
        assert key == "google-env-key"

    def test_gemini_key_takes_precedence_over_google_api_key(self):
        with patch.dict("os.environ", {"GEMINI_KEY": "gemini", "GOOGLE_API_KEY": "google"}, clear=True):
            key = _get_google_api_key()
        assert key == "gemini"

    def test_missing_key_raises_value_error(self):
        """When no key is available from any source, raise ValueError."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="Google API key is required"):
                _get_google_api_key()

    def test_client_without_google_api_key_falls_to_env(self):
        """Client exists but has no google_api_key attribute -> fall to env."""
        mock_client = MagicMock()
        mock_client._client = MagicMock(spec=[])  # No google_api_key

        with patch.dict("os.environ", {"GEMINI_KEY": "env-fallback"}, clear=True):
            key = _get_google_api_key(google_client=mock_client)
        assert key == "env-fallback"


# ---------------------------------------------------------------------------
# generate_image_with_gemini (async) -- mocked via sys.modules
# ---------------------------------------------------------------------------


def _build_genai_mocks(
    image_data: bytes | str | None = b"\x89PNG\r\n\x1a\nfake",
    no_candidates: bool = False,
    no_parts: bool = False,
    api_error: Exception | None = None,
):
    """Build mock google/google.genai/google.genai.types modules and return (mock_genai, mock_types, mock_google)."""
    mock_types = MagicMock()
    mock_types.GenerateContentConfig = MagicMock()
    mock_types.ImageConfig = MagicMock()

    mock_genai = MagicMock()
    mock_genai.types = mock_types

    mock_google = MagicMock()
    mock_google.genai = mock_genai

    # Build response
    if api_error:
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = api_error
        mock_genai.Client.return_value = mock_client
    elif no_candidates:
        mock_response = MagicMock()
        mock_response.candidates = []
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        mock_genai.Client.return_value = mock_client
    elif no_parts:
        mock_candidate = MagicMock()
        mock_candidate.content.parts = []
        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        mock_genai.Client.return_value = mock_client
    else:
        mock_inline_data = MagicMock()
        mock_inline_data.data = image_data
        mock_part = MagicMock()
        mock_part.inline_data = mock_inline_data
        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_part]
        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        mock_genai.Client.return_value = mock_client

    return mock_genai, mock_types, mock_google


class TestGenerateImageWithGemini:
    """Tests for generate_image_with_gemini with mocked google.genai."""

    @pytest.mark.asyncio
    async def test_success_returns_base64(self):
        image_bytes = b"fake_image_data_here"
        expected_b64 = base64.b64encode(image_bytes).decode("utf-8")

        mock_genai, mock_types, mock_google = _build_genai_mocks(image_data=image_bytes)

        with patch.dict(sys.modules, {
            "google": mock_google,
            "google.genai": mock_genai,
            "google.genai.types": mock_types,
        }):
            result = await generate_image_with_gemini(
                prompt="A sunset over mountains",
                api_key="test-key",
            )

        assert result == expected_b64

    @pytest.mark.asyncio
    async def test_no_image_in_response_raises(self):
        mock_genai, mock_types, mock_google = _build_genai_mocks(no_parts=True)

        with patch.dict(sys.modules, {
            "google": mock_google,
            "google.genai": mock_genai,
            "google.genai.types": mock_types,
        }):
            with pytest.raises(ValueError, match="No image data in response"):
                await generate_image_with_gemini(
                    prompt="A cat",
                    api_key="test-key",
                )

    @pytest.mark.asyncio
    async def test_no_candidates_raises(self):
        mock_genai, mock_types, mock_google = _build_genai_mocks(no_candidates=True)

        with patch.dict(sys.modules, {
            "google": mock_google,
            "google.genai": mock_genai,
            "google.genai.types": mock_types,
        }):
            with pytest.raises((ValueError, AttributeError)):
                await generate_image_with_gemini(
                    prompt="A dog",
                    api_key="test-key",
                )

    @pytest.mark.asyncio
    async def test_string_image_data_returned_as_is(self):
        """If image data is already a string (not bytes), return as-is."""
        mock_genai, mock_types, mock_google = _build_genai_mocks(image_data="already_base64_string")

        with patch.dict(sys.modules, {
            "google": mock_google,
            "google.genai": mock_genai,
            "google.genai.types": mock_types,
        }):
            result = await generate_image_with_gemini(
                prompt="A bird",
                api_key="test-key",
            )

        assert result == "already_base64_string"

    @pytest.mark.asyncio
    async def test_api_error_propagates(self):
        """API errors are re-raised."""
        mock_genai, mock_types, mock_google = _build_genai_mocks(
            api_error=RuntimeError("API quota exceeded")
        )

        with patch.dict(sys.modules, {
            "google": mock_google,
            "google.genai": mock_genai,
            "google.genai.types": mock_types,
        }):
            with pytest.raises(RuntimeError, match="API quota exceeded"):
                await generate_image_with_gemini(
                    prompt="Something",
                    api_key="test-key",
                )

    @pytest.mark.asyncio
    async def test_missing_genai_package_raises_import_error(self):
        """When google-genai is not installed, ImportError is raised."""
        # Remove google modules so the import fails
        saved = {}
        keys_to_remove = [k for k in sys.modules if k == "google" or k.startswith("google.")]
        for k in keys_to_remove:
            saved[k] = sys.modules.pop(k)

        real_import = builtins.__import__

        def failing_import(name, *args, **kwargs):
            if name == "google" or name.startswith("google."):
                raise ImportError(f"No module named '{name}'")
            return real_import(name, *args, **kwargs)

        try:
            with patch.object(builtins, "__import__", side_effect=failing_import):
                with pytest.raises(ImportError, match="google-genai package is required"):
                    await generate_image_with_gemini(
                        prompt="test",
                        api_key="test-key",
                    )
        finally:
            sys.modules.update(saved)

    @pytest.mark.asyncio
    async def test_custom_model_and_aspect_ratio(self):
        """Verify custom model and aspect_ratio are passed to the API."""
        mock_genai, mock_types, mock_google = _build_genai_mocks(image_data=b"img")

        with patch.dict(sys.modules, {
            "google": mock_google,
            "google.genai": mock_genai,
            "google.genai.types": mock_types,
        }):
            await generate_image_with_gemini(
                prompt="A landscape",
                api_key="test-key",
                aspect_ratio="16:9",
                model="custom-model-v2",
            )

        mock_client = mock_genai.Client.return_value
        call_args = mock_client.models.generate_content.call_args
        assert call_args.kwargs["model"] == "custom-model-v2"

    @pytest.mark.asyncio
    async def test_default_model(self):
        """Default model is used when none specified."""
        mock_genai, mock_types, mock_google = _build_genai_mocks(image_data=b"img")

        with patch.dict(sys.modules, {
            "google": mock_google,
            "google.genai": mock_genai,
            "google.genai.types": mock_types,
        }):
            await generate_image_with_gemini(
                prompt="A cat",
                api_key="test-key",
            )

        mock_client = mock_genai.Client.return_value
        call_args = mock_client.models.generate_content.call_args
        assert call_args.kwargs["model"] == "gemini-2.0-flash-exp-image-generation"


# ---------------------------------------------------------------------------
# generate_image_gemini (high-level async entry point)
# ---------------------------------------------------------------------------


class TestGenerateImageGemini:
    """Tests for generate_image_gemini which wraps key extraction + generation."""

    @pytest.mark.asyncio
    async def test_uses_explicit_api_key(self):
        with patch(
            "llming_models.tools.gemini_image.generate_image_with_gemini",
            new_callable=AsyncMock,
            return_value="base64data",
        ) as mock_gen:
            result = await generate_image_gemini(
                prompt="A cat",
                api_key="my-key",
            )

        assert result == "base64data"
        mock_gen.assert_called_once_with(
            prompt="A cat",
            api_key="my-key",
            aspect_ratio="1:1",
        )

    @pytest.mark.asyncio
    async def test_extracts_key_from_client(self):
        mock_client = MagicMock()
        mock_client._client = MagicMock()
        mock_client._client.google_api_key = "client-key"

        with patch(
            "llming_models.tools.gemini_image.generate_image_with_gemini",
            new_callable=AsyncMock,
            return_value="data",
        ) as mock_gen:
            await generate_image_gemini(prompt="A dog", google_client=mock_client)

        mock_gen.assert_called_once()
        assert mock_gen.call_args.kwargs["api_key"] == "client-key"

    @pytest.mark.asyncio
    async def test_no_key_raises(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="Google API key is required"):
                await generate_image_gemini(prompt="test")

    @pytest.mark.asyncio
    async def test_custom_aspect_ratio(self):
        with patch(
            "llming_models.tools.gemini_image.generate_image_with_gemini",
            new_callable=AsyncMock,
            return_value="data",
        ) as mock_gen:
            await generate_image_gemini(
                prompt="A landscape",
                api_key="key",
                aspect_ratio="16:9",
            )

        assert mock_gen.call_args.kwargs["aspect_ratio"] == "16:9"
