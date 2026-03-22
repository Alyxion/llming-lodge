"""Tests for provider compatibility in tool definitions and registry."""

import pytest

from llming_models.tools.tool_definition import (
    ToolDefinition,
    ToolSource,
    OPENAI_WEB_SEARCH_TOOL,
    ANTHROPIC_WEB_SEARCH_TOOL,
    DEFAULT_IMAGE_GENERATION_TOOL,
)
from llming_models.tools.tool_registry import ToolRegistry


# ── ToolDefinition.is_available_for_provider ────────────────────


class TestToolAvailability:
    """Test ToolDefinition.is_available_for_provider()."""

    def test_no_restrictions(self):
        """Tool with no requires_provider works for any provider."""
        tool = ToolDefinition(name="test", description="test")
        assert tool.is_available_for_provider("openai")
        assert tool.is_available_for_provider("azure_openai")
        assert tool.is_available_for_provider("anthropic")

    def test_requires_openai_matches_openai(self):
        tool = ToolDefinition(name="t", description="d", requires_provider="openai")
        assert tool.is_available_for_provider("openai")

    def test_requires_openai_rejects_anthropic(self):
        tool = ToolDefinition(name="t", description="d", requires_provider="openai")
        assert not tool.is_available_for_provider("anthropic")

    def test_azure_openai_compat_with_openai(self):
        """azure_openai is compatible with tools that require openai."""
        tool = ToolDefinition(name="t", description="d", requires_provider="openai")
        assert tool.is_available_for_provider("azure_openai")

    def test_requires_anthropic_rejects_azure(self):
        tool = ToolDefinition(name="t", description="d", requires_provider="anthropic")
        assert not tool.is_available_for_provider("azure_openai")

    def test_exclude_providers(self):
        tool = ToolDefinition(name="t", description="d", exclude_providers=["google"])
        assert tool.is_available_for_provider("openai")
        assert not tool.is_available_for_provider("google")

    def test_exclude_overrides_compat(self):
        """Exclude list takes precedence even if compat would match."""
        tool = ToolDefinition(
            name="t", description="d",
            requires_provider="openai",
            exclude_providers=["azure_openai"],
        )
        assert tool.is_available_for_provider("openai")
        assert not tool.is_available_for_provider("azure_openai")

    def test_web_search_openai_available_for_azure(self):
        """Real OPENAI_WEB_SEARCH_TOOL is available for azure_openai."""
        assert OPENAI_WEB_SEARCH_TOOL.is_available_for_provider("azure_openai")
        assert OPENAI_WEB_SEARCH_TOOL.is_available_for_provider("openai")
        assert not OPENAI_WEB_SEARCH_TOOL.is_available_for_provider("anthropic")

    def test_web_search_anthropic_not_available_for_azure(self):
        assert not ANTHROPIC_WEB_SEARCH_TOOL.is_available_for_provider("azure_openai")
        assert ANTHROPIC_WEB_SEARCH_TOOL.is_available_for_provider("anthropic")

    def test_image_gen_available_for_azure(self):
        """DALL-E image generation (requires openai) works with azure_openai."""
        assert DEFAULT_IMAGE_GENERATION_TOOL.is_available_for_provider("azure_openai")
        assert DEFAULT_IMAGE_GENERATION_TOOL.is_available_for_provider("openai")
        assert not DEFAULT_IMAGE_GENERATION_TOOL.is_available_for_provider("anthropic")


# ── ToolRegistry.get() with provider resolution ────────────────


class TestRegistryProviderLookup:
    """Test registry.get() resolves provider-specific keys with compat."""

    def _fresh_registry(self) -> ToolRegistry:
        return ToolRegistry()

    def test_get_web_search_for_openai(self):
        reg = self._fresh_registry()
        tool = reg.get("web_search", provider="openai")
        assert tool is not None
        assert tool.requires_provider == "openai"

    def test_get_web_search_for_anthropic(self):
        reg = self._fresh_registry()
        tool = reg.get("web_search", provider="anthropic")
        assert tool is not None
        assert tool.requires_provider == "anthropic"

    def test_get_web_search_for_azure_resolves_to_openai(self):
        """azure_openai should resolve to openai variant via compat mapping."""
        reg = self._fresh_registry()
        tool = reg.get("web_search", provider="azure_openai")
        assert tool is not None
        assert tool.requires_provider == "openai"

    def test_get_image_gen_for_azure(self):
        reg = self._fresh_registry()
        tool = reg.get("generate_image", provider="azure_openai")
        assert tool is not None
        assert tool.requires_provider == "openai"


# ── ToolRegistry.get_for_provider() filtering ─────────────────


class TestRegistryGetForProvider:
    """Test get_for_provider() correctly includes/excludes tools."""

    def test_azure_includes_openai_tools(self):
        reg = ToolRegistry()
        tools = reg.get_for_provider("azure_openai")
        names = [t.name for t in tools]
        assert "web_search" in names
        assert "generate_image" in names

    def test_anthropic_excludes_openai_tools(self):
        reg = ToolRegistry()
        tools = reg.get_for_provider("anthropic")
        names = [t.name for t in tools]
        # Should have anthropic web_search but not openai-only image gen
        assert "web_search" in names
        assert "generate_image" not in names

    def test_openai_has_both(self):
        reg = ToolRegistry()
        tools = reg.get_for_provider("openai")
        names = [t.name for t in tools]
        assert "web_search" in names
        assert "generate_image" in names
