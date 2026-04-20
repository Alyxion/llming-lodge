# Document Plugins

LLMing-Lodge includes a rich document plugin system that lets the LLM create interactive, editable documents directly in the chat. Each document type renders inline with export options, workspace integration, and — where supported — fullscreen lightbox views.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Fenced Code Block Format](#fenced-code-block-format)
- [Document Types](#document-types)
  - [Plotly Charts](#plotly-charts)
  - [Tables](#tables)
  - [Excel Spreadsheets](#excel-spreadsheets)
  - [Word Documents](#word-documents)
  - [PowerPoint Presentations](#powerpoint-presentations)
  - [HTML Sandbox](#html-sandbox)
  - [LaTeX Formulas](#latex-formulas)
- [Cross-Block References](#cross-block-references)
- [Workspace Integration](#workspace-integration)
- [MCP Tools](#mcp-tools)
- [Configuration](#configuration)
- [Frontend Rendering](#frontend-rendering)

---

## Overview

When the LLM produces a fenced code block with a recognized language identifier (e.g. ` ```plotly `, ` ```table `), the chat UI renders it as an interactive document instead of plain code. Each document type provides:

- **Inline preview** in the chat message
- **Toolbar** with export buttons (download, source view, open in workspace)
- **Workspace panel** — a side pane for focused viewing/interaction
- **Lightbox** — fullscreen view (Plotly, PowerPoint)
- **MCP tools** — the LLM can read and edit documents after creation

## Architecture

The system spans three layers:

```
  Backend (Python)                      Frontend (JavaScript)
 +----------------------------+       +----------------------------+
 | DocPluginManager           |       | DocPluginRegistry          |
 |   get_preamble() ---------> LLM    |   register(lang, plugin)   |
 |   get_mcp_configs()        |       |   render(lang, el, data)   |
 |                            |       |   onBlockRendered(cb)      |
 | PlotlyDocumentMCP          |       |                            |
 | TableDocumentMCP           |       | builtin-plugins.js         |
 | TextDocMCP                 |       |   plotlyPlugin             |
 | PresentationMCP            |       |   tablePlugin              |
 | HtmlDocumentMCP            |       |   wordPlugin               |
 |                            |       |   powerpointPlugin         |
 | DocumentSessionStore       |       |   htmlPlugin               |
 +----------------------------+       |   latexPlugin              |
                                      +----------------------------+
```

**Backend** (`llming_lodge/doc_plugins/`):
- `manager.py` — orchestrates enabled plugins, generates LLM preamble, creates MCP servers
- `*_mcp.py` — one MCP server per type with read/edit tools
- `document_store.py` — per-session document persistence with versioning

**Frontend** (`llming_lodge/static/chat/plugins/`):
- `doc-plugin-registry.js` — plugin registration, render pipeline, cross-block reference resolution
- `builtin-plugins.js` — render functions and export logic for all seven document types

---

## Fenced Code Block Format

Every document block the LLM emits is a fenced code block with a language identifier and a JSON body:

````
```plotly
{
  "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "name": "Monthly Revenue",
  "data": [ ... ],
  "layout": { ... }
}
```
````

### Mandatory Fields

| Field  | Type   | Description |
|--------|--------|-------------|
| `id`   | string | UUID v4. Reuse the same ID when updating an existing document. |
| `name` | string | Human-readable label shown in the document list. |

If the LLM omits the `id`, the frontend auto-generates one via `crypto.randomUUID()`.

### Supported Language Identifiers

| Identifier      | Type               |
|-----------------|--------------------|
| `plotly`        | Plotly.js chart     |
| `table`         | Data table          |
| `excel`         | Multi-sheet spreadsheet |
| `word`          | Structured document |
| `powerpoint`    | Slide deck          |
| `pptx`          | Slide deck (alias)  |
| `html_sandbox`  | HTML/CSS/JS app     |
| `latex`         | LaTeX formula       |

---

## Document Types

### Plotly Charts

Interactive charts powered by [Plotly.js](https://plotly.com/javascript/).

**Language:** `plotly`

**JSON Format:**

```json
{
  "id": "c7a9f2e1-...",
  "name": "Sales Performance",
  "data": [
    {
      "x": [1, 2, 3, 4, 5],
      "y": [10, 15, 13, 17, 20],
      "type": "scatter",
      "mode": "lines+markers",
      "name": "Q1 Sales"
    }
  ],
  "layout": {
    "title": "Sales Performance",
    "xaxis": { "title": "Month" },
    "yaxis": { "title": "Revenue ($K)" }
  }
}
```

The `data` and `layout` fields follow the [Plotly.js schema](https://plotly.com/javascript/reference/) directly. All trace types are supported (scatter, bar, pie, heatmap, etc.).

**Inline Rendering:**
- Compact, responsive chart preview with theme-aware colors
- Click hint overlay (expand icon) on hover

**Toolbar:**
| Button | Action |
|--------|--------|
| Expand | Opens fullscreen lightbox |
| SVG | Downloads chart as SVG |
| PNG | Downloads chart as PNG |
| `{…}` | Toggles raw JSON source view |
| Side pane | Opens in workspace panel |

**Lightbox:** Full-size interactive chart with dark background, scroll-zoom enabled, SVG/PNG export at 1920x1080.

---

### Tables

Simple data tables with sorting and export.

**Language:** `table`

**JSON Format:**

```json
{
  "id": "t4b8f1c2-...",
  "name": "Sales by Region",
  "columns": ["Region", "Q1", "Q2", "Q3", "Q4"],
  "rows": [
    ["North", "100", "120", "115", "140"],
    ["South", "85", "95", "110", "125"],
    ["East", "200", "210", "205", "220"]
  ]
}
```

**Inline Rendering:** Styled HTML table with header row and scrollable wrapper.

**Toolbar:**
| Button | Action |
|--------|--------|
| XLSX | Downloads as formatted Excel file (bold headers, auto-fit columns, frozen header, auto-filter) |
| CSV | Downloads as comma-separated values |
| `{…}` | Raw JSON source view |
| Side pane | Opens in workspace panel |

**XLSX Export Details:** Built from raw XML without external dependencies. Includes:
- Bold header row with gray background fill
- Auto-fitted column widths (10-60 characters)
- Automatic number/string type detection
- Frozen header row and auto-filter

---

### Excel Spreadsheets

Multi-sheet spreadsheets with tab navigation.

**Language:** `excel`

**JSON Format:**

```json
{
  "id": "e3d7c9f2-...",
  "name": "Annual Budget",
  "sheets": [
    {
      "name": "Summary",
      "columns": ["Category", "2024", "2025"],
      "rows": [
        ["Revenue", "500000", "550000"],
        ["Expenses", "350000", "385000"]
      ]
    },
    {
      "name": "Details",
      "columns": ["Item", "Cost"],
      "rows": [
        ["Salaries", "200000"],
        ["Equipment", "50000"]
      ]
    }
  ],
  "activeSheet": "Summary"
}
```

Each sheet has the same structure as a table (`columns` + `rows`). Rows can be arrays or objects (keyed by column name).

**Inline Rendering:** Tab bar for sheet switching, styled table for active sheet.

**Toolbar:**
| Button | Action |
|--------|--------|
| XLSX | Downloads via SheetJS with all sheets preserved |
| `{…}` | Raw JSON source view |
| Side pane | Opens in workspace panel |

---

### Word Documents

Structured documents with headings, paragraphs, lists, and tables.

**Language:** `word`

**JSON Format:**

```json
{
  "id": "w2b5e8c1-...",
  "name": "Project Proposal",
  "sections": [
    {
      "type": "heading",
      "content": "Executive Summary",
      "level": 1
    },
    {
      "type": "paragraph",
      "content": "This project aims to improve customer retention through..."
    },
    {
      "type": "list",
      "items": ["Phase 1: Research", "Phase 2: Implementation", "Phase 3: Rollout"],
      "ordered": true
    },
    {
      "type": "table",
      "headers": ["Milestone", "Date", "Status"],
      "rows": [
        ["Research complete", "2025-03", "Done"],
        ["Beta launch", "2025-06", "In progress"]
      ]
    }
  ]
}
```

**Section Types:**

| Type | Required Fields | Notes |
|------|----------------|-------|
| `heading` | `content`, `level` (1-6) | Renders as `<h1>`-`<h6>` |
| `paragraph` | `content` | Plain text paragraph |
| `list` | `items` (array of strings) | Set `ordered: true` for numbered list |
| `table` | `headers`, `rows` | Nested tabular data |

**Toolbar:**
| Button | Action |
|--------|--------|
| HTML | Downloads as styled HTML document (Georgia serif, 800px max-width) |
| `{…}` | Raw JSON source view |
| Side pane | Opens in workspace panel |

---

### PowerPoint Presentations

Slide decks with a unified layout engine powering both preview and PPTX export.

**Language:** `powerpoint` (or `pptx`)

**JSON Format:**

```json
{
  "id": "p4f1a9e2-...",
  "name": "Quarterly Review",
  "title": "Q1 2025 Results",
  "author": "Sales Team",
  "slideNumbers": true,
  "slides": [
    {
      "title": "Q1 2025 Results",
      "layout": "title",
      "elements": [
        { "type": "subtitle", "content": "Fiscal Year 2025" }
      ]
    },
    {
      "title": "Key Metrics",
      "elements": [
        { "type": "list", "items": ["Revenue: $2.1M (+15%)", "Users: 50K (+22%)", "NPS: 72"] }
      ]
    },
    {
      "title": "Revenue Chart",
      "elements": [
        { "type": "chart", "$ref": "c7a9f2e1-..." }
      ]
    }
  ]
}
```

**Slide Layouts:**
- `title` — centered title, accent bar, subtitle, author (auto-detected for first slide)
- `content` — (default) title at top, elements flow vertically

**Element Types:**

| Type | Fields | Notes |
|------|--------|-------|
| `text` | `content` | Body text, 16pt |
| `heading` | `content` | Section heading, 22pt bold |
| `subtitle` | `content` | Subtitle text, 16pt gray |
| `list` | `items` (array) | Bullet points |
| `table` | `headers`, `rows` | Inline table |
| `chart` | `data`, `layout` or `$ref` | Plotly chart (see [cross-block references](#cross-block-references)) |
| `image` | `src`, `alt` | External image |

**Inline Rendering:**
- Slide preview in 16:9 aspect ratio (960x540 virtual px, CSS-scaled to fit)
- Navigation arrows + slide counter
- Double-click opens fullscreen lightbox

**Toolbar:**
| Button | Action |
|--------|--------|
| Expand | Opens fullscreen lightbox with keyboard navigation |
| PPTX | Downloads as PowerPoint file via PptxGenJS |
| `{…}` | Raw JSON source view |
| Side pane | Opens in workspace panel |

**Lightbox:** Full-screen slide viewer with arrow-key navigation, PPTX export button.

**PPTX Export:** Uses [PptxGenJS](https://gitbrent.github.io/PptxGenJS/) with the same layout coordinates. Charts are rasterized to PNG (800x450) via Plotly and embedded as images.

---

### HTML Sandbox

Full HTML/CSS/JS applications rendered in a sandboxed iframe.

**Language:** `html_sandbox`

**JSON Format:**

```json
{
  "id": "h9e2a5f3-...",
  "name": "Calculator App",
  "title": "Simple Calculator",
  "html": "<div id='calc'>...</div>",
  "css": "#calc { padding: 20px; font-family: sans-serif; }",
  "js": "document.getElementById('calc').addEventListener('click', ...);"
}
```

All three sections (`html`, `css`, `js`) are optional — omit empty ones.

**Inline Rendering:** Compact card with icon, title, and "Click to open in workspace" hint. Source tabs (HTML / CSS / JS) toggle below the card.

**Workspace Integration:** Automatically opens in the workspace side pane on creation. The iframe uses `sandbox="allow-scripts"` for isolation.

**Toolbar:**
| Button | Action |
|--------|--------|
| HTML | Downloads as complete HTML file |
| `{…}` | Raw JSON source view |

---

### LaTeX Formulas

Mathematical formulas rendered via [KaTeX](https://katex.org/).

**Language:** `latex`

**Format:** Either a raw LaTeX string or JSON:

Raw:
````
```latex
\int_{0}^{\infty} e^{-x^2} dx = \frac{\sqrt{\pi}}{2}
```
````

JSON:
```json
{
  "formula": "E = mc^2",
  "displayMode": true
}
```

`displayMode` defaults to `true` (block-level). Set to `false` for inline rendering.

**Toolbar:**
| Button | Action |
|--------|--------|
| `{…}` | Raw source view |
| Side pane | Opens in workspace panel |

---

## Cross-Block References

Documents can reference data from earlier blocks using `$ref` pointers. This is particularly useful for embedding Plotly charts and tables into PowerPoint slides.

### How It Works

1. Every block with an `id` field is registered in a `BlockDataStore`
2. Any JSON property can use `{"$ref": "<id>"}` to reference another block
3. The referenced data is merged as a base — explicit properties in the current block override

### Example

First, create a chart and a table:

````
```plotly
{
  "id": "revenue-chart",
  "name": "Revenue Chart",
  "data": [{ "x": ["Q1","Q2","Q3"], "y": [100,150,200], "type": "bar" }],
  "layout": { "title": "Quarterly Revenue" }
}
```

```table
{
  "id": "metrics-table",
  "name": "Key Metrics",
  "columns": ["Metric", "Value"],
  "rows": [["Revenue", "$450K"], ["Growth", "33%"]]
}
```
````

Then reference them in a presentation:

````
```powerpoint
{
  "id": "quarterly-deck",
  "name": "Q Report",
  "title": "Quarterly Report",
  "slides": [
    {
      "title": "Revenue",
      "elements": [{ "type": "chart", "$ref": "revenue-chart" }]
    },
    {
      "title": "Summary",
      "elements": [{ "type": "table", "$ref": "metrics-table" }]
    }
  ]
}
```
````

### Rules

- Only reference blocks that appear **earlier** in the conversation
- No circular references
- The `$ref` target's data merges as a base; local properties override
- Cross-type compatibility is applied automatically (e.g., a table referenced inside a chart element gets its columns/rows mapped)

---

## Workspace Integration

The **workspace panel** is a side pane that shares horizontal space with the chat area. It has two sections:

```
+------------------+-------------------------+
| Document List    | Preview / Workspace     |
| (220px)          | (flex: 1)               |
|                  |                         |
| [Chart A]        | <rendered content>      |
| [Table B] *      |                         |
| [Slides C]       |                         |
+------------------+-------------------------+
```

### How Documents Appear

- **Auto-open:** The workspace opens automatically when the first document is created
- **Document list:** All inline and persistent documents appear as cards, newest first
- **Click:** Select a card to preview it in the right pane
- **Double-click:** Open in fullscreen lightbox (if supported)
- **Toolbar button:** Each plugin's toolbar has a side-pane icon button that opens that specific document in the workspace

### Toggle

The workspace toggle button sits in the top-right of the topbar. It opens/closes the entire panel (doc list + preview) as one unit.

### Mobile (< 768px)

On narrow viewports, the workspace takes full width and the chat area is hidden. The topbar toggle switches between chat view and workspace view.

---

## MCP Tools

Each document type (except LaTeX) exposes MCP tools that let the LLM read and modify documents after creation. These tools are registered as in-process MCP servers.

### Tool Prefixes

| Type | Prefix | Example Tools |
|------|--------|---------------|
| Plotly | `plotly_` | `plotly_list_charts`, `plotly_get_data`, `plotly_update_layout` |
| Table | `table_` | `table_get_rows`, `table_update_cells`, `table_sort` |
| Excel | `excel_` | `excel_list_sheets`, `excel_get_range`, `excel_add_sheet` |
| Word | `word_` | `word_list_sections`, `word_add_section`, `word_move_section` |
| PowerPoint | `pptx_` | `pptx_list_slides`, `pptx_add_slide`, `pptx_reorder_slides` |
| HTML | `html_` | `html_get_source`, `html_update_html`, `html_update_css` |

### Plotly Tools

| Tool | Parameters | Description |
|------|-----------|-------------|
| `plotly_list_charts` | — | List all charts (id, name, version, trace count) |
| `plotly_get_data` | `document_id` | Get full data + layout |
| `plotly_get_trace` | `document_id`, `trace_index` | Get specific trace |
| `plotly_add_trace` | `document_id`, `trace` | Append a trace |
| `plotly_remove_trace` | `document_id`, `trace_index` | Remove trace by index |
| `plotly_update_trace` | `document_id`, `trace_index`, `updates` | Partial update of trace properties |
| `plotly_update_layout` | `document_id`, `layout` | Partial update of layout |
| `plotly_search_data` | `query` | Search across all charts |

### Table Tools

| Tool | Parameters | Description |
|------|-----------|-------------|
| `table_get_rows` | `document_id`, `start`, `count` | Paginated row access |
| `table_update_cells` | `document_id`, `updates` | Update cells: `[{row, col, value}]` |
| `table_add_rows` | `document_id`, `rows` | Append rows |
| `table_delete_rows` | `document_id`, `indices` | Delete rows by index |
| `table_add_column` | `document_id`, `name`, `default_value`, `position` | Add column |
| `table_remove_column` | `document_id`, `index` or `name` | Remove column |
| `table_search` | `document_id`, `query` | Case-insensitive search |
| `table_sort` | `document_id`, `column`, `descending` | Sort with auto number detection |

### Excel Tools

| Tool | Parameters | Description |
|------|-----------|-------------|
| `excel_list_sheets` | `document_id` | List sheets with metadata |
| `excel_get_sheet` | `document_id`, `sheet_name` | Get full sheet data |
| `excel_get_range` | `document_id`, `start_row/col`, `end_row/col`, `sheet_name` | Rectangular range |
| `excel_update_cells` | `document_id`, `updates`, `sheet_name` | Update cells |
| `excel_add_rows` | `document_id`, `rows`, `sheet_name` | Append rows |
| `excel_delete_rows` | `document_id`, `indices`, `sheet_name` | Delete rows |
| `excel_add_sheet` | `document_id`, `name`, `columns` | Create sheet |
| `excel_delete_sheet` | `document_id`, `sheet_name` | Remove sheet |
| `excel_search` | `document_id`, `query` | Search all sheets |

### Word Tools

| Tool | Parameters | Description |
|------|-----------|-------------|
| `word_list_sections` | `document_id` | List sections (id, type, preview) |
| `word_get_section` | `document_id`, `section_id` or `index` | Get section |
| `word_update_section` | `document_id`, `section_id`, `updates` | Update content/type/level |
| `word_add_section` | `document_id`, `type`, `content`, `position`, `level` | Add section |
| `word_delete_section` | `document_id`, `section_id` | Remove section |
| `word_move_section` | `document_id`, `section_id`, `new_position` | Reorder |
| `word_search` | `document_id`, `query` | Search content |

### PowerPoint Tools

| Tool | Parameters | Description |
|------|-----------|-------------|
| `pptx_list_slides` | `document_id` | List slides (id, title, element count) |
| `pptx_get_slide` | `document_id`, `slide_id` or `index` | Get full slide |
| `pptx_update_element` | `document_id`, `slide_id`, `element_index`, `updates` | Update element |
| `pptx_add_slide` | `document_id`, `title`, `layout`, `elements`, `position` | Add slide |
| `pptx_delete_slide` | `document_id`, `slide_id` | Remove slide |
| `pptx_reorder_slides` | `document_id`, `order` | New slide order (all IDs) |
| `pptx_search` | `document_id`, `query` | Search titles and content |

### HTML Tools

| Tool | Parameters | Description |
|------|-----------|-------------|
| `html_get_source` | `document_id` | Get HTML, CSS, JS, title |
| `html_get_html` | `document_id` | Get HTML only |
| `html_get_css` | `document_id` | Get CSS only |
| `html_get_js` | `document_id` | Get JS only |
| `html_update_html` | `document_id`, `html` | Replace HTML |
| `html_update_css` | `document_id`, `css` | Replace CSS |
| `html_update_js` | `document_id`, `js` | Replace JS |
| `html_search` | `document_id`, `query` | Search all sections |

---

## Configuration

### Enabling/Disabling Plugins

Use `DocPluginManager` to control which document types are available:

```python
from llming_docs.manager import DocPluginManager

# All types enabled (default)
manager = DocPluginManager()

# Specific types only
manager = DocPluginManager(enabled_types=["plotly", "table", "html"])

# No document plugins
manager = DocPluginManager(enabled_types=[])

# Change at runtime
manager.set_enabled_types(["plotly", "table", "excel"])
```

### Preamble Generation

The manager generates an LLM preamble that teaches the model how to use each enabled document type:

```python
preamble = manager.get_preamble()
# Returns instruction text with format specs, rules, and best practices
```

This preamble is injected into the system prompt and covers:
- Fenced code block syntax for each type
- Mandatory `id` and `name` fields
- Cross-block reference syntax
- PowerPoint best practices (build data first, reference via `$ref`)

### MCP Server Registration

Get MCP configs for the enabled types:

```python
configs = manager.get_mcp_configs()
# Returns list of MCPServerConfig for each enabled type
```

---

## Frontend Rendering

### Plugin Registry

The `DocPluginRegistry` class manages plugin lifecycle:

```javascript
const registry = new DocPluginRegistry();

// Register a plugin
registry.register('plotly', {
  render: async (container, data, blockId) => { ... },
  inline: true,        // render inline in chat (vs. document panel)
  loader: async () => { ... },  // load external dependencies
});

// Check if a language is supported
registry.has('plotly');  // true

// Render a block
await registry.render('plotly', container, jsonString, 'block-123');
```

### Render Pipeline

When `registry.render(lang, container, rawData, blockId)` is called:

1. **Load dependencies** — calls `plugin.loader()` once (deduplicated)
2. **Parse JSON** — if valid JSON, auto-assigns UUID if missing
3. **Register in BlockDataStore** — for cross-block reference resolution
4. **Resolve `$ref` pointers** — replaces references with actual data
5. **Apply cross-type compatibility** — maps data between types (e.g., table columns into chart traces)
6. **Call `plugin.render()`** — plugin creates DOM elements in the container
7. **Fire `onBlockRendered`** — notifies the app to update the document list

### Toolbar Pattern

Every plugin follows the same toolbar pattern:

```javascript
const toolbar = document.createElement('div');
toolbar.className = 'cv2-doc-plugin-export';

// Type-specific buttons (download, expand, etc.)
toolbar.innerHTML = `<button class="cv2-doc-export-btn">XLSX</button>`;

// Common buttons added by helpers:
_addSourceButton(toolbar, rawData);      // {…} source toggle
_addWorkspaceButton(toolbar, blockId);   // side-pane button

container.appendChild(toolbar);
```

### Theme Awareness

Plotly charts adapt to light/dark mode by reading the `cv2-dark` class on `#chat-app`:

- **Light mode:** dark text (#1f2937), light grid lines, transparent backgrounds
- **Dark mode:** light text (#e5e7eb), subtle grid lines, transparent backgrounds

### External Dependencies

| Library | Loaded For | CDN/Vendor Path |
|---------|-----------|-----------------|
| Plotly.js | plotly, powerpoint (charts) | `/chat-static/vendor/plotly.min.js` |
| SheetJS (XLSX) | excel | `/chat-static/vendor/xlsx.full.min.js` |
| PptxGenJS | powerpoint export | `/chat-static/vendor/pptxgenjs.bundle.min.js` |
| KaTeX | latex | Pre-loaded (assumed available) |

Dependencies are loaded on demand via `_loadScript()` with deduplication — a library is loaded only once regardless of how many blocks use it.
