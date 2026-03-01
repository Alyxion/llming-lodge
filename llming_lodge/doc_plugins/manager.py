"""DocPluginManager — orchestrates document store and per-type MCP servers.

Each doc plugin type self-registers its preamble and MCP server definition
so that the LLM prompt and tooling are automatically derived from whichever
plugins are currently enabled.
"""

import importlib
import logging
from typing import List, Optional

from llming_lodge.tools.mcp.config import MCPServerConfig
from llming_lodge.doc_plugins.document_store import DocumentSessionStore
from llming_lodge.doc_plugins.creator_mcp import DocumentCreatorMCP

logger = logging.getLogger(__name__)

# ── Plugin registry ──────────────────────────────────────────────
# All known doc plugin types.  Order here determines preamble order.
ALL_DOC_PLUGIN_TYPES: list[str] = [
    "plotly", "latex", "table", "text_doc", "presentation", "html", "email_draft",
]

# Backward compat: old type names → new type names
_TYPE_ALIASES: dict[str, str] = {
    "word": "text_doc",
    "powerpoint": "presentation",
}

# Preamble fragment per type (auto-inserted into context_preamble)
_PREAMBLE_LINES: dict[str, str] = {
    "plotly":       "- ```plotly  — Plotly.js chart (JSON: {data: [...], layout: {...}})",
    "latex":        "- ```latex   — LaTeX formula",
    "table":        "- ```table   — Data table / spreadsheet (JSON: {columns: [...], rows: [...]})",
    "text_doc":     "- ```text_doc — Text document (JSON: {sections: [{type, content, ...}]})",
    "presentation": "- ```presentation — Presentation / slide deck (JSON: {title, author, slideNumbers, slides: [{title, layout, elements: [...]}]})",
    "html":         "- ```html_sandbox — Website / web app (HTML, CSS, JavaScript — JSON: {html, css, js, title})",
    "email_draft":  "- ```email_draft — Email draft (JSON: {subject, to: [...], cc: [...], bcc: [...], body_html, attachments: [{ref, name}]})",
}

# Per-type MCP server definitions (types without an entry only have frontend rendering)
_MCP_SERVERS: dict[str, dict] = {
    "plotly": {
        "module": "llming_lodge.doc_plugins.plotly_mcp",
        "class_name": "PlotlyDocumentMCP",
        "label": "Plotly Charts",
        "description": "Edit and refine Plotly chart documents",
    },
    "table": {
        "module": "llming_lodge.doc_plugins.table_mcp",
        "class_name": "TableDocumentMCP",
        "label": "Tables",
        "description": "Edit table and spreadsheet documents",
    },
    "text_doc": {
        "module": "llming_lodge.doc_plugins.text_doc_mcp",
        "class_name": "TextDocMCP",
        "label": "Text Documents",
        "description": "Edit structured text documents",
    },
    "presentation": {
        "module": "llming_lodge.doc_plugins.presentation_mcp",
        "class_name": "PresentationMCP",
        "label": "Presentations",
        "description": "Edit presentation slide decks",
    },
    "html": {
        "module": "llming_lodge.doc_plugins.html_mcp",
        "class_name": "HtmlDocumentMCP",
        "label": "Website",
        "description": "Create and edit websites, web apps, and interactive HTML/CSS/JS projects",
    },
    "email_draft": {
        "module": "llming_lodge.doc_plugins.email_mcp",
        "class_name": "EmailDraftMCP",
        "label": "Email Drafts",
        "description": "Compose and edit email drafts",
    },
}

# Tool-name prefix per doc type (for bulk-toggling when presets change)
TYPE_TOOL_PREFIXES: dict[str, str] = {
    "plotly": "plotly_",
    "table": "table_",
    "text_doc": "text_doc_",
    "presentation": "pptx_",
    "html": "html_",
    "email_draft": "email_",
}


class DocPluginManager:
    """Creates and manages document-related MCP servers for a session.

    Args:
        enabled_types: Which doc plugin types to enable.
            ``None`` → all types (default for random chat).
            ``[]``   → no doc plugins at all.
            ``["plotly", "table"]`` → only those types.
    """

    def __init__(
        self,
        enabled_types: Optional[List[str]] = None,
        presentation_templates: Optional[List] = None,
        requires_providers: Optional[List[str]] = None,
    ) -> None:
        self.store = DocumentSessionStore()
        self._mcp_instances: list = []
        self._enabled_types: list[str] = (
            [_TYPE_ALIASES.get(t, t) for t in enabled_types]
            if enabled_types is not None
            else list(ALL_DOC_PLUGIN_TYPES)
        )
        self._presentation_templates: list = list(presentation_templates or [])
        self._requires_providers: Optional[List[str]] = requires_providers

    # ── Public API ───────────────────────────────────────────────

    @property
    def enabled_types(self) -> list[str]:
        """Currently enabled doc plugin types."""
        return list(self._enabled_types)

    @property
    def presentation_templates(self) -> list:
        """Available presentation templates."""
        return list(self._presentation_templates)

    def set_enabled_types(self, types: Optional[List[str]]) -> None:
        """Update enabled types (e.g. when a preset is applied)."""
        if types is not None:
            self._enabled_types = [_TYPE_ALIASES.get(t, t) for t in types]
        else:
            self._enabled_types = list(ALL_DOC_PLUGIN_TYPES)

    def get_preamble(self) -> str:
        """Build LLM preamble text for currently enabled doc plugin types.

        Returns an empty string if no types are enabled.
        """
        if not self._enabled_types:
            return ""

        lines = [
            "\n\n## Rich Document Rendering",
            "The chat UI supports rendering rich documents inline. "
            "To render visualizations directly in your response, use fenced code blocks "
            "with these language identifiers:",
        ]
        for t in self._enabled_types:
            if t in _PREAMBLE_LINES:
                lines.append(_PREAMBLE_LINES[t])
        lines.append(
            "Always prefer these fenced code blocks for inline rendering. "
            "Use the create_document tool only when the user explicitly asks for a persistent "
            "document they can manage separately."
        )
        lines.append(
            "\n### Document Identity & Updates\n"
            "**ALWAYS** include an `\"id\"` field (UUID v4, e.g. `\"id\": \"a1b2c3d4-e5f6-...\"`) "
            "in every document block you create. This is mandatory — no exceptions.\n"
            "**ALWAYS** include a `\"name\"` field — a short, descriptive, human-readable filename "
            "(e.g. `\"name\": \"Q1 Revenue Chart\"`, `\"name\": \"Sales by Region Table\"`, "
            "`\"name\": \"Project Kickoff Deck\"`). This name appears in the sidebar document "
            "panel and as the default export filename. Keep it concise but specific.\n"
            "When the user asks you to update, enhance, or revise an existing document, "
            "**reuse the exact same `\"id\"`** from the original block. This signals to the "
            "system that you are providing an updated version of the same document rather than "
            "a new one. Keeping the same id preserves continuity and is greatly appreciated.\n"
            "Generate a fresh UUID only when creating a genuinely new document."
        )
        lines.append(
            "\n### Cross-Block References\n"
            "Reference any earlier block via `{\"$ref\": \"<id>\"}` in any JSON "
            "property of a later block. The referenced data merges as a base — explicit "
            "properties in the referencing object override.\n"
            "Example: a plotly block with `\"id\": \"<uuid>\"` can be embedded in a "
            "presentation slide element as `{\"type\": \"chart\", \"$ref\": \"<uuid>\"}`.\n"
            "Rules: only reference blocks that appear earlier in the conversation; "
            "do not create circular references."
        )
        if "plotly" in self._enabled_types:
            lines.append(
                "For charts: use Plotly.js spec with 'type' in each trace "
                "(e.g. 'pie', 'bar', 'scatter')."
            )
        if "presentation" in self._enabled_types:
            # Check if any template has layouts (template-native mode)
            _templates_with_layouts = [
                t for t in self._presentation_templates
                if hasattr(t, 'layouts') and t.layouts
            ]
            if _templates_with_layouts:
                # Template-native preamble — the LLM should use layout names and placeholders
                lines.append(
                    "\n### Presentations\n"
                    "When creating presentations:\n"
                    "1. **Build data blocks first** — output ```table and ```plotly blocks "
                    "with `\"id\"` fields BEFORE the ```presentation block. This lets data "
                    "appear inline AND be referenced in slides.\n"
                    "2. **Reference data via `$ref`** — in placeholder values use "
                    "`{\"type\": \"chart\", \"$ref\": \"my-chart-id\"}` or "
                    "`{\"type\": \"table\", \"$ref\": \"my-table-id\"}` to embed data.\n"
                    "3. **Keep text concise** — use bullet points, not paragraphs, to avoid overflow.\n"
                    "4. The presentation supports PPTX export and fullscreen viewing."
                )
                for tpl in _templates_with_layouts:
                    tpl_name = tpl.name if hasattr(tpl, 'name') else str(tpl)
                    tpl_label = tpl.label if hasattr(tpl, 'label') else tpl_name
                    lines.append(f"\n### Template: \"{tpl_name}\" ({tpl_label})")
                    lines.append(
                        f'Set `"template": "{tpl_name}"` at the top level when the user '
                        "explicitly requests this brand/template style. "
                        "Without an explicit request, use the default styling (no template field).\n"
                        "When using this template, slides use `layout` + `placeholders` instead of `elements`.\n"
                        "Available layouts:"
                    )
                    for layout in tpl.layouts:
                        ph_list = ", ".join(
                            f'`{ph.name}` ({"/".join(ph.accepts)})'
                            for ph in layout.placeholders
                        )
                        flags = ""
                        if layout.is_title:
                            flags = " *(use for first slide)*"
                        elif layout.is_end:
                            flags = " *(use for last slide)*"
                        lines.append(f'- **"{layout.name}"** ({layout.label}){flags}: {ph_list}')
                    lines.append(
                        '\nUse **"text"** as the default layout for text/list slides and '
                        '**"chart"** as the default for chart/data slides. '
                        'Only use specialized layouts (two_columns, text_half, picture) when '
                        'the content specifically calls for them.'
                    )
                    lines.append(
                        "\nPlaceholder values: a string for text, or an object for rich content:\n"
                        '- List: `{"type": "list", "items": ["Item 1", "Item 2"]}`\n'
                        '- Table: `{"type": "table", "headers": [...], "rows": [[...]]}`\n'
                        '- Chart: `{"type": "chart", "data": [...], "layout": {...}}` (Plotly spec, or `{"$ref": "id"}`)\n'
                        '- Image: `{"type": "image", "src": "url_or_data_uri"}`\n'
                        "\nExample slide:\n"
                        "```json\n"
                        '{"layout": "text", "placeholders": {"title": "Revenue", '
                        '"body": "Q1 2026 Results", '
                        '"content": {"type": "list", "items": ["Up 15%", "42 new customers"]}}}\n'
                        "```"
                    )

                # Also mention non-template format for completeness
                tpl_names_list = ", ".join(
                    f'"{t.name}"' for t in _templates_with_layouts
                )
                lines.append(
                    "\n### Default Presentations (no template)\n"
                    f"When NOT using a template ({tpl_names_list}), use the abstract `elements` format:\n"
                    "- Spec-level fields: `\"title\"`, `\"author\"`, `\"slideNumbers\": true`\n"
                    "- Slide fields: `\"title\"`, `\"layout\"` (\"title\"/\"end\"), `\"elements\"` array\n"
                    "- Element types: text, heading, list (items:[]), table (headers:[], rows:[[]]), "
                    "chart (Plotly spec), image (src, alt), subtitle (for title slides)."
                )
            else:
                # No template layouts — use the existing generic preamble
                lines.append(
                    "\n### Presentation Best Practices\n"
                    "When creating presentations:\n"
                    "1. **Build data blocks first** — output ```table and ```plotly blocks "
                    "with `\"id\"` fields BEFORE the ```presentation block. This lets data "
                    "appear inline AND be referenced in slides.\n"
                    "2. **Reference data via `$ref`** — in slide elements use "
                    "`{\"type\": \"chart\", \"$ref\": \"my-chart-id\"}` or "
                    "`{\"type\": \"table\", \"$ref\": \"my-table-id\"}` to embed data.\n"
                    "3. **Title slide** — set `\"layout\": \"title\"` on the first slide. "
                    "It renders with centered title, accent bar, and author line.\n"
                    "4. **Spec-level fields** — set `\"title\"`, `\"author\"`, "
                    "`\"slideNumbers\": true` at the top level of the spec.\n"
                    "5. **Keep text concise** — use bullet points, not paragraphs, to avoid overflow.\n"
                    "6. Element types: text, heading, list (items:[]), table (headers:[], rows:[[]]), "
                    "chart (Plotly spec), image (src, alt), subtitle (for title slides).\n"
                    "7. The presentation supports PPTX export and fullscreen viewing."
                )
                if self._presentation_templates:
                    tpl_names = ", ".join(
                        f'"{t.name}" ({t.label})' if hasattr(t, 'name') else str(t)
                        for t in self._presentation_templates
                    )
                    lines.append(
                        "\n### Presentation Templates\n"
                        f"Available templates: {tpl_names}.\n"
                        'Add `"template": "template_name"` to the top-level presentation spec ONLY '
                        "when the user explicitly requests a specific brand or template style.\n"
                        "Without an explicit request, use the default styling."
                    )
        if "email_draft" in self._enabled_types:
            lines.append(
                "\n### Email Drafts\n"
                "When composing emails:\n"
                "1. Use ````email_draft` fenced code blocks with JSON: "
                "`{subject, to: [...], cc: [...], body_html, attachments: [{ref, name}]}`.\n"
                "2. Write `body_html` as clean, professional HTML suitable for email clients. "
                "Use inline styles only (no `<style>` blocks). No JavaScript.\n"
                "3. To attach files from the chat, add `{\"ref\": \"filename.pdf\", \"name\": \"Report\"}` "
                "to the `attachments` array. You can reference chat attachments by filename "
                "or documents created in the chat by their document ID.\n"
                "4. The user will see a visual email preview with Draft and Send buttons — "
                "you do NOT need to send the email yourself.\n"
                "5. Always include `\"id\"` and `\"name\"` fields as with other document types."
            )
        return "\n".join(lines) + "\n"

    def get_mcp_configs(self) -> List[MCPServerConfig]:
        """Create MCP server instances and return their configs.

        Only creates MCPs for enabled types.  Call once per session, then
        merge the returned configs with the user's existing mcp_servers list.
        """
        if not self._enabled_types:
            return []

        configs: list[MCPServerConfig] = []

        # Creator MCP — present when any type is enabled
        creator = DocumentCreatorMCP(self.store)
        self._mcp_instances.append(creator)
        configs.append(MCPServerConfig(
            server_instance=creator,
            label="Documents",
            description="Create and manage rich documents (charts, tables, spreadsheets, etc.)",
            category="Documents",
            enabled_by_default=True,
            requires_providers=self._requires_providers,
        ))

        # Per-type MCP servers
        for doc_type in self._enabled_types:
            spec = _MCP_SERVERS.get(doc_type)
            if not spec:
                continue  # e.g. "latex" has no MCP, only frontend rendering
            try:
                mod = importlib.import_module(spec["module"])
                cls = getattr(mod, spec["class_name"])
                instance = cls(self.store)
                self._mcp_instances.append(instance)
                configs.append(MCPServerConfig(
                    server_instance=instance,
                    label=spec["label"],
                    description=spec["description"],
                    category="Documents",
                    enabled_by_default=False,
                    requires_providers=self._requires_providers,
                ))
            except (ImportError, AttributeError) as e:
                logger.warning("[DOC_PLUGINS] Could not load %s MCP: %s", doc_type, e)

        return configs

    async def cleanup(self) -> None:
        """Clean up MCP instances."""
        for mcp in self._mcp_instances:
            if hasattr(mcp, "close"):
                try:
                    await mcp.close()
                except Exception:
                    pass
        self._mcp_instances.clear()
