"""Integration test: verify Claude Sonnet uses embed sections correctly.

Two levels:
1. **Schema/preamble tests** (always run) — verify the preamble and tool
   definitions contain the right keywords so the AI knows about embed.
2. **Live Claude Sonnet test** (regression marker) — sends a real prompt
   to Claude Sonnet and verifies it produces embed sections, not data copies.

Requires ANTHROPIC_API_KEY in .env for regression tests.
"""

import asyncio
import json
import os
import logging
from typing import Any

import dotenv
import pytest

dotenv.load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env"))

from llming_docs.document_store import DocumentSessionStore
from llming_docs.manager import DocPluginManager
from llming_docs.text_doc_mcp import TextDocMCP
from llming_docs.creator_mcp import DocumentCreatorMCP

logger = logging.getLogger(__name__)


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ═══════════════════════════════════════════════════════════════════════════
# PART 1: Preamble & tool schema verification (no API key needed)
# ═══════════════════════════════════════════════════════════════════════════


class TestPreambleContainsEmbedInstructions:
    """Verify the LLM preamble clearly instructs about embed sections."""

    def _get_preamble(self, types=None):
        mgr = DocPluginManager(enabled_types=types)
        return mgr.get_preamble()

    def test_preamble_mentions_embed_keyword(self):
        preamble = self._get_preamble()
        assert "embed" in preamble.lower()

    def test_preamble_mentions_ref_for_embed(self):
        preamble = self._get_preamble()
        assert "ref" in preamble

    def test_preamble_has_embed_example(self):
        """Preamble should have a concrete example of using embed with ref."""
        preamble = self._get_preamble()
        assert "text_doc_add_section" in preamble
        assert 'type="embed"' in preamble or "type='embed'" in preamble or '"embed"' in preamble

    def test_preamble_says_do_not_copy_data(self):
        """Preamble should explicitly say NOT to copy/fetch data."""
        preamble = self._get_preamble()
        lower = preamble.lower()
        assert "do not" in lower or "don't" in lower or "never" in lower
        # Should mention not fetching/copying
        assert any(phrase in lower for phrase in [
            "do not copy", "do not fetch", "don't copy", "don't fetch",
            "never copy", "never fetch", "not copy", "not fetch",
            "do not call", "just reference",
        ])

    def test_preamble_mentions_auto_conversion(self):
        """Preamble should explain that embeds auto-convert on export."""
        preamble = self._get_preamble()
        lower = preamble.lower()
        assert "png" in lower or "image" in lower
        assert "table" in lower

    def test_preamble_has_embedding_section(self):
        """There should be a dedicated 'Embedding Documents' section in the preamble."""
        preamble = self._get_preamble()
        assert "Embedding Documents" in preamble, "Missing 'Embedding Documents' section"
        # It should appear before presentation *details* (### Presentations)
        embed_idx = preamble.find("Embedding Documents")
        pres_details_idx = preamble.find("### Presentations")
        if pres_details_idx != -1:
            assert embed_idx < pres_details_idx


class TestToolSchemaContainsEmbed:
    """Verify MCP tool schemas include embed type and ref parameter."""

    def test_text_doc_add_section_has_embed_type(self):
        store = DocumentSessionStore()
        mcp = TextDocMCP(store)
        tools = _run(mcp.list_tools())
        add_tool = next(t for t in tools if t["name"] == "text_doc_add_section")

        # Check enum includes 'embed'
        type_prop = add_tool["inputSchema"]["properties"]["type"]
        assert "embed" in type_prop["enum"]

    def test_text_doc_add_section_has_ref_parameter(self):
        store = DocumentSessionStore()
        mcp = TextDocMCP(store)
        tools = _run(mcp.list_tools())
        add_tool = next(t for t in tools if t["name"] == "text_doc_add_section")

        # Check ref parameter exists
        props = add_tool["inputSchema"]["properties"]
        assert "ref" in props
        assert "document" in props["ref"]["description"].lower() or "embed" in props["ref"]["description"].lower()

    def test_text_doc_add_section_description_mentions_embed(self):
        store = DocumentSessionStore()
        mcp = TextDocMCP(store)
        tools = _run(mcp.list_tools())
        add_tool = next(t for t in tools if t["name"] == "text_doc_add_section")

        desc = add_tool["description"].lower()
        assert "embed" in desc
        assert "chart" in desc or "document" in desc or "reference" in desc

    def test_create_document_mentions_embed(self):
        store = DocumentSessionStore()
        mcp = DocumentCreatorMCP(store)
        tools = _run(mcp.list_tools())
        create_tool = next(t for t in tools if t["name"] == "create_document")

        desc = create_tool["description"]
        assert "embed" in desc.lower()
        assert "$ref" in desc or "ref" in desc

    def test_text_doc_add_section_embed_does_not_require_content(self):
        """Embed sections should NOT require content — only ref."""
        store = DocumentSessionStore()
        mcp = TextDocMCP(store)
        tools = _run(mcp.list_tools())
        add_tool = next(t for t in tools if t["name"] == "text_doc_add_section")

        required = add_tool["inputSchema"].get("required", [])
        assert "content" not in required


# ═══════════════════════════════════════════════════════════════════════════
# PART 2: Simulated tool flow (no API, verifies MCP behavior)
# ═══════════════════════════════════════════════════════════════════════════


class TestSimulatedEmbedFlow:
    """Simulate what a well-prompted AI should do: create chart, then embed it."""

    def test_ai_flow_chart_then_embed_in_word(self):
        """Simulate: AI creates chart, then creates Word doc with embed ref."""
        store = DocumentSessionStore()
        creator = DocumentCreatorMCP(store)
        text_mcp = TextDocMCP(store)

        # Step 1: AI creates a plotly chart (via create_document or math tool)
        chart_result = _run(creator.call_tool("create_document", {
            "type": "plotly",
            "name": "Revenue Chart",
            "data": json.dumps({
                "data": [{"x": [1, 2, 3], "y": [10, 20, 15], "type": "bar"}],
                "layout": {"title": {"text": "Revenue"}},
            }),
        }))
        chart_id = json.loads(chart_result)["document_id"]

        # Step 2: AI creates a Word doc with embed (correct behavior)
        word_result = _run(creator.call_tool("create_document", {
            "type": "text_doc",
            "name": "Revenue Report",
            "data": json.dumps({
                "sections": [
                    {"id": "h1", "type": "heading", "level": 1, "content": "Revenue Report"},
                    {"id": "intro", "type": "paragraph", "content": "See the chart below:"},
                    {"id": "chart", "type": "embed", "$ref": chart_id},
                    {"id": "conclusion", "type": "paragraph", "content": "Revenue is up 50%."},
                ],
            }),
        }))
        word_id = json.loads(word_result)["document_id"]

        # Verify the Word doc has the embed reference
        word_doc = store.get(word_id)
        sections = word_doc.data["sections"]
        embed_sections = [s for s in sections if s.get("type") == "embed"]
        assert len(embed_sections) == 1
        assert embed_sections[0]["$ref"] == chart_id

        # Verify the chart is still in the store (for resolution)
        chart_doc = store.get(chart_id)
        assert chart_doc is not None
        assert chart_doc.type == "plotly"

    def test_ai_flow_chart_then_embed_via_tool(self):
        """Simulate: AI uses text_doc_add_section to add embed."""
        store = DocumentSessionStore()
        creator = DocumentCreatorMCP(store)
        text_mcp = TextDocMCP(store)

        # Create chart
        chart_result = _run(creator.call_tool("create_document", {
            "type": "plotly",
            "name": "Sales Chart",
            "data": json.dumps({
                "data": [{"x": [1, 2, 3], "y": [5, 15, 10], "type": "scatter"}],
                "layout": {},
            }),
        }))
        chart_id = json.loads(chart_result)["document_id"]

        # Create Word doc
        word_result = _run(creator.call_tool("create_document", {
            "type": "text_doc",
            "name": "Sales Report",
            "data": json.dumps({"sections": [
                {"id": "t", "type": "heading", "level": 1, "content": "Sales Report"},
            ]}),
        }))
        word_id = json.loads(word_result)["document_id"]

        # AI adds embed via tool
        result = _run(text_mcp.call_tool("text_doc_add_section", {
            "document_id": word_id,
            "type": "embed",
            "ref": chart_id,
        }))
        assert json.loads(result)["status"] == "section_added"

        # Verify
        doc = store.get(word_id)
        embed = [s for s in doc.data["sections"] if s["type"] == "embed"]
        assert len(embed) == 1
        assert embed[0]["$ref"] == chart_id


# ═══════════════════════════════════════════════════════════════════════════
# PART 3: Real Claude Sonnet integration test
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.regression
class TestClaudeSonnetEmbedBehavior:
    """Call real Claude Sonnet to verify it uses embed sections.

    Requires ANTHROPIC_API_KEY. Skipped if not available.
    Run with: pytest -m regression tests/test_embed_prompting.py
    """

    @pytest.fixture
    def api_key(self):
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            pytest.skip("ANTHROPIC_API_KEY not set")
        return key

    @pytest.fixture
    def store(self):
        return DocumentSessionStore()

    @pytest.fixture
    def preamble(self, store):
        mgr = DocPluginManager(enabled_types=["plotly", "table", "text_doc"])
        return mgr.get_preamble()

    @pytest.fixture
    def tools(self, store):
        """Build the tool list the AI sees."""
        text_mcp = TextDocMCP(store)
        creator = DocumentCreatorMCP(store)
        return _run(text_mcp.list_tools()) + _run(creator.list_tools())

    def _call_sonnet(self, api_key, system, user_msg, tools):
        """Call Claude Sonnet and return tool_use blocks."""
        from anthropic import Anthropic

        client = Anthropic(api_key=api_key)

        # Convert MCP tool format to Anthropic API format
        api_tools = []
        for t in tools:
            api_tools.append({
                "name": t["name"],
                "description": t.get("description", ""),
                "input_schema": t["inputSchema"],
            })

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            system=system,
            messages=[{"role": "user", "content": user_msg}],
            tools=api_tools,
            tool_choice={"type": "any"},
        )

        tool_uses = [b for b in response.content if b.type == "tool_use"]
        return tool_uses, response

    def test_sonnet_uses_embed_for_chart_in_word(self, api_key, store, preamble, tools):
        """Given a chart ID, Claude Sonnet should use embed to add it to a Word doc."""
        # Pre-create a chart in the store
        chart = store.create(type="plotly", name="sin(x) Plot", data={
            "data": [{"x": [0, 1, 2, 3], "y": [0, 0.84, 0.91, 0.14], "type": "scatter"}],
            "layout": {"title": {"text": "sin(x)"}},
        })

        # Pre-create a Word doc
        word = store.create(type="text_doc", name="Report", data={
            "sections": [{"id": "h1", "type": "heading", "level": 1, "content": "Report"}],
        })

        system = (
            f"You are a document assistant.\n{preamble}\n\n"
            f"Active documents:\n"
            f"- Plotly chart: id={chart.id}, name='{chart.name}'\n"
            f"- Word document: id={word.id}, name='{word.name}'\n"
        )
        user = f"Add the sin(x) chart to the Word document."

        tool_uses, response = self._call_sonnet(api_key, system, user, tools)

        # The AI should call text_doc_add_section with type=embed and ref=chart.id
        assert len(tool_uses) >= 1, f"Expected tool call, got: {response.content}"

        # Find the text_doc_add_section call
        add_calls = [t for t in tool_uses if t.name == "text_doc_add_section"]
        assert len(add_calls) >= 1, (
            f"Expected text_doc_add_section call, got: "
            f"{[t.name for t in tool_uses]}"
        )

        call = add_calls[0]
        args = call.input
        assert args.get("type") == "embed", (
            f"Expected type='embed', got type='{args.get('type')}'. "
            f"Full args: {json.dumps(args, indent=2)}"
        )
        assert args.get("ref") == chart.id, (
            f"Expected ref='{chart.id}', got ref='{args.get('ref')}'. "
            f"Full args: {json.dumps(args, indent=2)}"
        )

        logger.info(
            "[SONNET_TEST] SUCCESS — AI used embed with ref=%s (chart_id=%s)",
            args.get("ref"), chart.id,
        )

    def test_sonnet_does_not_fetch_data(self, api_key, store, preamble, tools):
        """Claude should NOT call get_document or plotly_get_data — just embed."""
        chart = store.create(type="plotly", name="cos(x)", data={
            "data": [{"x": [0, 1, 2], "y": [1, 0.54, -0.42], "type": "scatter"}],
            "layout": {"title": {"text": "cos(x)"}},
        })
        word = store.create(type="text_doc", name="Doc", data={
            "sections": [{"id": "h", "type": "heading", "level": 1, "content": "Doc"}],
        })

        system = (
            f"You are a document assistant.\n{preamble}\n\n"
            f"Active documents:\n"
            f"- Plotly chart: id={chart.id}, name='{chart.name}'\n"
            f"- Word document: id={word.id}, name='{word.name}'\n"
        )
        user = "Add the cos(x) chart to the Word document."

        tool_uses, response = self._call_sonnet(api_key, system, user, tools)

        # Should NOT call get_document or any data-fetching tool
        fetch_calls = [t for t in tool_uses if t.name in (
            "get_document", "plotly_get_data", "plotly_list_charts",
        )]
        assert len(fetch_calls) == 0, (
            f"AI incorrectly fetched data instead of embedding. "
            f"Calls: {[t.name for t in tool_uses]}"
        )

    def test_sonnet_creates_word_with_embed_from_scratch(self, api_key, store, preamble, tools):
        """Given a chart, Claude should create a new Word doc with embed section."""
        chart = store.create(type="plotly", name="Revenue", data={
            "data": [{"x": ["Q1", "Q2", "Q3"], "y": [100, 150, 130], "type": "bar"}],
            "layout": {"title": {"text": "Revenue"}},
        })

        system = (
            f"You are a document assistant.\n{preamble}\n\n"
            f"Active documents:\n"
            f"- Plotly chart: id={chart.id}, name='{chart.name}'\n"
        )
        user = "Create a Word document that includes this chart with a title and description."

        tool_uses, response = self._call_sonnet(api_key, system, user, tools)

        # Should call create_document with text_doc type
        create_calls = [t for t in tool_uses if t.name == "create_document"]

        if create_calls:
            # Check the data contains an embed section
            args = create_calls[0].input
            assert args.get("type") in ("text_doc", "word")
            data = args.get("data", "{}")
            if isinstance(data, str):
                data = json.loads(data)
            sections = data.get("sections", [])
            embed_sections = [s for s in sections if s.get("type") == "embed"]
            assert len(embed_sections) >= 1, (
                f"No embed section found in created document. "
                f"Sections: {json.dumps(sections, indent=2)}"
            )
            assert embed_sections[0].get("$ref") == chart.id, (
                f"Embed $ref doesn't match chart ID. "
                f"Expected {chart.id}, got {embed_sections[0].get('$ref')}"
            )
        else:
            # Might have used text_doc_add_section instead — that's also fine
            add_calls = [t for t in tool_uses if t.name == "text_doc_add_section"]
            assert len(add_calls) >= 1, (
                f"Expected create_document or text_doc_add_section. "
                f"Got: {[t.name for t in tool_uses]}"
            )
