# Browser MCP — JavaScript Visualizations in Droplets

Droplets can include JavaScript files that run as MCP tool servers directly in the browser. These tools let the AI interact with custom data, generate rich visualizations, and perform computations — all within a sandboxed Web Worker.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Writing MCP JavaScript](#writing-mcp-javascript)
  - [Tool Registration](#tool-registration)
  - [MCP SDK Compatibility](#mcp-sdk-compatibility)
  - [Multi-File Projects](#multi-file-projects)
  - [Accessing Data Files](#accessing-data-files)
  - [Return Values](#return-values)
- [Rich Visualizations](#rich-visualizations)
  - [Rich MCP Envelope](#rich-mcp-envelope)
  - [Visualization Payload](#visualization-payload)
  - [Trust-Gated Actions](#trust-gated-actions)
- [Sandbox Restrictions](#sandbox-restrictions)
- [Droplet Packaging](#droplet-packaging)
  - [File Structure](#file-structure)
  - [ZIP Upload via API](#zip-upload-via-api)
  - [Capabilities Field](#capabilities-field)
- [Activation Flow](#activation-flow)
- [Communication Protocol](#communication-protocol)
- [Built-in Document Plugins](#built-in-document-plugins)
- [API Reference](#api-reference)

---

## Overview

```
                        ┌──────────────────────┐
                        │   LLM (Claude, etc.) │
                        └──────┬───────────────┘
                               │ tool_call
                        ┌──────▼───────────────┐
                        │   Chat Server        │
                        │   (Python)           │
                        └──────┬───────────────┘
                               │ WebSocket
                        ┌──────▼───────────────┐
                        │   Browser            │
                        │   ┌────────────────┐ │
                        │   │  Web Worker    │ │
                        │   │  (sandboxed)   │ │
                        │   │                │ │
                        │   │  index.js      │ │
                        │   │  utils.js      │ │
                        │   │  data.csv      │ │
                        │   └────────────────┘ │
                        └──────────────────────┘
```

1. Droplet JS files are bundled and loaded into a Web Worker
2. The worker discovers tools and reports them to the server
3. The LLM calls tools by name; the server routes calls to the worker via WebSocket
4. Results are returned to the LLM (plain text or rich visualization)

---

## Quick Start

**1. Create `index.js`:**

```javascript
import { Server } from "@modelcontextprotocol/sdk/server/index.js";

const server = new Server({ name: "my-tools", version: "1.0.0" });

server.tool("greet", "Greet a user by name", {
  type: "object",
  properties: {
    name: { type: "string", description: "The user's name" }
  },
  required: ["name"]
}, async ({ name }) => {
  return `Hello, ${name}! Welcome to the chat.`;
});

await server.connect(new StdioServerTransport());
```

**2. Create `droplet.json`:**

```json
{
  "name": "Greeting Bot",
  "description": "A simple bot that greets users",
  "system_prompt": "You have a greet tool available. Use it when users ask to be greeted.",
  "icon": "waving_hand"
}
```

**3. Package and upload:**

```bash
zip -r greeting.zip droplet.json index.js

curl -X PUT https://example.com/api/llming/v1/droplets/_new/zip \
  -H "Authorization: Bearer llming_..." \
  -H "Content-Type: application/zip" \
  --data-binary @greeting.zip
```

The MCP SDK imports and `server.connect()` call are automatically stripped and replaced with browser-compatible shims. You write standard MCP SDK code — the runtime handles the rest.

---

## Writing MCP JavaScript

### Tool Registration

Register tools using the MCP SDK pattern:

```javascript
server.tool(name, description, inputSchema, handler)
```

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `name` | string | Tool name (used by the LLM to call it) |
| `description` | string | What the tool does (shown to the LLM) |
| `inputSchema` | object | JSON Schema for the tool's parameters |
| `handler` | async function | Receives the validated arguments, returns a result |

**Example with complex schema:**

```javascript
server.tool("query_sales", "Query sales data with filters", {
  type: "object",
  properties: {
    region: {
      type: "string",
      enum: ["north", "south", "east", "west"],
      description: "Sales region to query"
    },
    year: {
      type: "integer",
      minimum: 2020,
      maximum: 2030,
      description: "Fiscal year"
    },
    metric: {
      type: "string",
      enum: ["revenue", "units", "customers"],
      description: "Which metric to return"
    }
  },
  required: ["region", "year"]
}, async ({ region, year, metric = "revenue" }) => {
  // Access attached data files
  const csvText = self._getDataFileText("sales_data.csv");
  const rows = csvText.split("\n").slice(1); // skip header

  const filtered = rows
    .map(r => r.split(","))
    .filter(r => r[0] === region && r[1] === String(year));

  const colIndex = { revenue: 2, units: 3, customers: 4 }[metric];
  const total = filtered.reduce((sum, r) => sum + Number(r[colIndex]), 0);

  return `${metric} for ${region} in ${year}: ${total}`;
});
```

**Shorthand (no description):**

```javascript
server.tool("ping", { type: "object", properties: {} }, async () => "pong");
```

### MCP SDK Compatibility

The runtime rewrites MCP SDK code for browser execution:

| SDK Pattern | Browser Translation |
|---|---|
| `import { Server } from "@modelcontextprotocol/sdk/..."` | Stripped (shim provides `Server`) |
| `import { StdioServerTransport } from "..."` | Stripped |
| `new Server({name, version})` | Returns shim object |
| `server.tool(...)` | Routes to `_registerMcpTool()` |
| `server.setRequestHandler(schema, handler)` | Routes to internal dispatcher |
| `server.connect(...)` | No-op |
| `server.close()` | No-op |

**The `setRequestHandler` pattern** (lower-level SDK style) is also supported:

```javascript
server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: [
    {
      name: "my_tool",
      description: "Does something useful",
      inputSchema: { type: "object", properties: { x: { type: "number" } } }
    }
  ]
}));

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;
  if (name === "my_tool") {
    return {
      content: [{ type: "text", text: `Result: ${args.x * 2}` }]
    };
  }
});
```

Available schema constants: `ListToolsRequestSchema`, `CallToolRequestSchema`, `ListResourcesRequestSchema`, `ReadResourceRequestSchema`.

### Multi-File Projects

Local imports between files are supported:

```javascript
// utils.js
export function formatCurrency(amount) {
  return `$${amount.toLocaleString()}`;
}

export function parseCSV(text) {
  const lines = text.trim().split("\n");
  const headers = lines[0].split(",");
  return lines.slice(1).map(line => {
    const values = line.split(",");
    return Object.fromEntries(headers.map((h, i) => [h, values[i]]));
  });
}
```

```javascript
// index.js
import { formatCurrency, parseCSV } from "./utils.js";

server.tool("total_revenue", "Calculate total revenue", {
  type: "object", properties: {}
}, async () => {
  const data = parseCSV(self._getDataFileText("revenue.csv"));
  const total = data.reduce((s, r) => s + Number(r.amount), 0);
  return `Total revenue: ${formatCurrency(total)}`;
});
```

**How it works:**

1. Files without local imports are loaded first
2. Each file's `export` statements are captured into `self._mcp_modules[filename]`
3. `import { x } from "./file.js"` becomes `const { x } = self._mcp_modules["file"]`
4. Dependency order is resolved automatically (multi-level imports supported)

**Rules:**
- Only relative imports (`./file.js`) are supported between project files
- SDK imports (`@modelcontextprotocol/...`) are stripped automatically
- No npm packages or CDN imports — all code must be self-contained

### Accessing Data Files

Data files attached to the droplet (CSV, JSON, PDF, DOCX, etc.) are pre-loaded into the worker. Access them with these global APIs:

#### `self._listDataFiles()`

Returns metadata for all attached data files.

```javascript
const files = self._listDataFiles();
// [{name: "data.csv", mimeType: "text/csv", size: 1024}, ...]
```

#### `self._getDataFile(name)`

Returns a file by exact name with all data.

```javascript
const file = self._getDataFile("data.csv");
// {
//   name: "data.csv",
//   mimeType: "text/csv",
//   size: 1024,
//   base64: "Y29sMSxjb2wy...",   // raw base64-encoded content
//   text: "col1,col2\nval1,val2"  // extracted text (for text files, PDFs, etc.)
// }
```

#### `self._getDataFileText(name)`

Returns only the extracted plain text of a file. For text files this is the raw content; for PDFs/DOCX/XLSX this is the text extracted during upload.

```javascript
const csvContent = self._getDataFileText("sales.csv");
const pdfContent = self._getDataFileText("report.pdf"); // extracted text
```

#### `self._getDataFileBytes(name)`

Returns the raw binary content as a `Uint8Array`. Useful for binary processing.

```javascript
const bytes = self._getDataFileBytes("image.png");
// Uint8Array or null
```

### Return Values

#### Plain Text

The simplest return — a string sent directly to the LLM:

```javascript
async ({ query }) => {
  return `Found 42 results for "${query}"`;
}
```

#### MCP Content Array

The SDK-style return format is auto-unwrapped:

```javascript
async ({ query }) => {
  return {
    content: [
      { type: "text", text: `Found 42 results for "${query}"` }
    ]
  };
}
```

Both formats produce identical results for the LLM.

#### Rich Visualization

For interactive charts, tables, or custom HTML — see the [Rich Visualizations](#rich-visualizations) section below.

---

## Rich Visualizations

Tools can return rich, interactive content that renders in the chat UI. This is the most powerful feature of browser MCP — it lets droplets create charts, dashboards, and custom visualizations.

### Rich MCP Envelope

Wrap the tool result in a `__rich_mcp__` envelope:

```javascript
server.tool("sales_chart", "Generate a sales chart", {
  type: "object",
  properties: {
    year: { type: "integer" }
  }
}, async ({ year }) => {
  const data = parseCSV(self._getDataFileText("sales.csv"));
  const yearData = data.filter(r => r.year === String(year));

  return JSON.stringify({
    __rich_mcp__: {
      version: "1.0",
      llm_summary: "[Generated interactive sales chart for " + year + "]",
      render: {
        type: "visualization",
        title: `Sales Performance ${year}`,
        payload: {
          data: [{
            x: yearData.map(r => r.quarter),
            y: yearData.map(r => Number(r.revenue)),
            type: "bar",
            name: "Revenue",
            marker: { color: "#4B8FE7" }
          }],
          layout: {
            title: `Revenue by Quarter — ${year}`,
            xaxis: { title: "Quarter" },
            yaxis: { title: "Revenue ($)" }
          }
        }
      }
    }
  });
});
```

### Envelope Fields

| Field | Type | Required | Description |
|---|---|---|---|
| `__rich_mcp__` | object | yes | Marker object (triggers rich rendering) |
| `version` | string | no | Render spec version (default: `"1.0"`) |
| `min_viewer_version` | string | no | Minimum browser MCP version (default: `"1.0"`) |
| `llm_summary` | string | yes | Text summary sent to the LLM instead of the full envelope. The LLM never sees the visualization data — only this summary. |
| `render` | object | yes | Rendering specification |
| `render.type` | string | yes | Content type (see below) |
| `render.title` | string | no | Display title for the visualization |
| `render.payload` | object | yes | Type-specific rendering data |
| `inject_messages` | array | no | Trust-gated: messages to inject into chat history |
| `trigger_llm_call` | object | no | Trust-gated: trigger a follow-up LLM call |

### Visualization Payload

The `render.payload` follows the same format as the built-in document plugins (see [document-plugins.md](document-plugins.md)).

**Plotly chart:**

```json
{
  "render": {
    "type": "visualization",
    "title": "Revenue Trend",
    "payload": {
      "data": [
        {
          "x": ["Q1", "Q2", "Q3", "Q4"],
          "y": [100, 150, 130, 200],
          "type": "scatter",
          "mode": "lines+markers",
          "name": "2025"
        }
      ],
      "layout": {
        "xaxis": { "title": "Quarter" },
        "yaxis": { "title": "Revenue ($K)" }
      }
    }
  }
}
```

The payload supports all [Plotly.js trace types](https://plotly.com/javascript/reference/) — scatter, bar, pie, heatmap, sankey, treemap, 3D, maps, etc.

**Table:**

```json
{
  "render": {
    "type": "visualization",
    "title": "Sales Summary",
    "payload": {
      "columns": ["Region", "Revenue", "Growth"],
      "rows": [
        ["North", "$2.1M", "+15%"],
        ["South", "$1.8M", "+22%"]
      ]
    }
  }
}
```

**HTML sandbox:**

```json
{
  "render": {
    "type": "visualization",
    "title": "Interactive Dashboard",
    "payload": {
      "html": "<div id='app'>...</div>",
      "css": "body { font-family: sans-serif; }",
      "js": "// interactive logic here"
    }
  }
}
```

### Processing Pipeline

1. Tool handler returns `JSON.stringify({ __rich_mcp__: {...} })`
2. Server detects the `__rich_mcp__` marker in the result
3. Server extracts `llm_summary` and sends **only that** to the LLM
4. Server stores the full render spec in MongoDB (`rich_mcp_renders` collection)
5. A UUID reference is appended to the assistant's response as a fenced code block
6. The browser fetches the render spec by UUID and renders it using the appropriate plugin

This means the LLM never sees the visualization data — it only knows the summary. This keeps context windows clean and prevents the LLM from trying to parse chart JSON.

### Trust-Gated Actions

Rich MCP results can optionally include actions that require user approval:

**`inject_messages`** — add messages to the conversation:

```javascript
return JSON.stringify({
  __rich_mcp__: {
    llm_summary: "[Analysis complete]",
    render: { type: "visualization", payload: { ... } },
    inject_messages: [
      { role: "user", content: "Now explain the key insights from this chart." }
    ],
    trigger_llm_call: {
      prompt: "Based on the chart you just generated, provide 3 key insights."
    }
  }
});
```

**Behavior:**

1. The browser shows a trust prompt: *"This droplet wants to add messages and trigger AI response"*
2. User can choose **"Allow once"** or **"Always trust this droplet"**
3. If allowed, the messages are injected and the follow-up LLM call is triggered
4. Trust decisions are stored in MongoDB (`mcp_droplet_trust` collection) per user per droplet

---

## Sandbox Restrictions

All MCP JavaScript runs in a Web Worker with strict sandboxing. The following APIs are **blocked and frozen** before any user code loads:

| Category | Blocked APIs |
|---|---|
| Network | `fetch`, `XMLHttpRequest`, `WebSocket`, `EventSource`, `Request`, `Response`, `Headers` |
| Script loading | `importScripts`, dynamic `import()` |
| Storage | `indexedDB`, `caches` |
| Cross-origin | `BroadcastChannel` |
| Beacons | `navigator.sendBeacon` |

All blocks use `Object.defineProperty` with `configurable: false` — user code cannot circumvent them.

**What IS available:**
- Full ES6+ JavaScript (classes, async/await, generators, etc.)
- `JSON`, `Math`, `Date`, `Map`, `Set`, `Array`, etc.
- `console.log` / `console.error` (for debugging)
- `self._*` data file APIs (see [Accessing Data Files](#accessing-data-files))
- `TextEncoder`, `TextDecoder`
- `crypto.randomUUID()`, `crypto.getRandomValues()`
- `setTimeout`, `setInterval` (though tools should return promptly)

---

## Droplet Packaging

### File Structure

A droplet ZIP contains a manifest and files:

```
my-droplet/
├── droplet.json          # Manifest with droplet metadata
├── index.js              # MCP entry point (required for JS tools)
├── utils.js              # Additional JS modules (optional)
├── data.csv              # Data files (optional)
├── knowledge.pdf         # Knowledge documents (optional)
└── config.json           # Additional data files (optional)
```

**Entry point:** The runtime looks for the file named in `mcp_entry_point` (default: `"index.js"`).

**JS files** (`.js`, `.mjs`) are loaded as MCP tool code in the Web Worker.

**Data files** (everything else) are available via `self._getDataFile()`. Text-based files and extracted text from PDFs/DOCX/XLSX are also injected into the LLM's context as knowledge.

### ZIP Upload via API

See the [Remote API documentation](remote-api.md#upload-droplet-as-zip) for full details on the ZIP upload endpoint.

```bash
# Create new droplet from ZIP
curl -X PUT https://example.com/api/llming/v1/droplets/_new/zip \
  -H "Authorization: Bearer llming_..." \
  -H "Content-Type: application/zip" \
  --data-binary @my-droplet.zip

# Update existing droplet
curl -X PUT https://example.com/api/llming/v1/droplets/my-uid/zip \
  -H "Authorization: Bearer llming_..." \
  -H "Content-Type: application/zip" \
  --data-binary @my-droplet.zip
```

### Capabilities Field

The `capabilities` field on a droplet controls which tools are enabled:

```json
{
  "capabilities": {
    "query_sales": true,
    "delete_record": false,
    "optional_tool": null
  }
}
```

| Value | Effect |
|---|---|
| `true` | Tool is always enabled when this droplet is active |
| `false` | Tool is disabled (hidden from the LLM) |
| `null` | Use the user's global setting |

---

## Activation Flow

When a user opens a chat with a droplet that has JS files:

1. **Server** checks the nudge's `files` array for `.js` / `.mjs` files
2. **Server** separates JS source files from data files
3. **Server** extracts text content from data files (PDFs, DOCX, etc.)
4. **Server** sorts JS files by dependency depth (files without imports first)
5. **Server** sends a `start_browser_mcp` WebSocket message to the browser
6. **Browser** bundles all JS files with the MCP shim, creates a Web Worker
7. **Worker** initializes, discovers tools (via `server.tool()` or `setRequestHandler`)
8. **Browser** reports the discovered tool list back to the server
9. **Server** registers each tool in the global MCP tool registry
10. **Server** injects a preamble block into the LLM context:

```
## MCP Tools — Sales Assistant

CRITICAL: You have the following specialised tools available.
Always call these tools when the user's request matches their purpose.

- **query_sales**: Query sales data with filters
- **sales_chart**: Generate an interactive sales chart
```

### Mid-Conversation Activation

Droplets can also be activated during a conversation via the `activate_mcp_nudge(uid)` tool. The same flow above applies.

---

## Communication Protocol

### WebSocket Messages

**Server → Browser: activate worker**

```json
{
  "type": "start_browser_mcp",
  "request_id": "uuid",
  "nudge_uid": "my-droplet-uid",
  "entry_point": "index.js",
  "files": {
    "index.js": "import { Server } from ...\nserver.tool(...)",
    "utils.js": "export function parseCSV(text) { ... }"
  },
  "data_files": [
    {
      "name": "data.csv",
      "mime_type": "text/csv",
      "size": 1024,
      "content_base64": "Y29sMSxjb2wy...",
      "text_content": "col1,col2\nval1,val2"
    }
  ]
}
```

**Server → Browser: call tool**

```json
{
  "type": "browser_mcp_call",
  "request_id": "uuid",
  "nudge_uid": "my-droplet-uid",
  "tool_name": "query_sales",
  "arguments": { "region": "north", "year": 2025 }
}
```

**Browser → Server: tool result**

```json
{
  "type": "browser_mcp_result",
  "request_id": "uuid",
  "result": "Revenue for north in 2025: $2,100,000",
  "rich_mcp": false
}
```

Or for rich results:

```json
{
  "type": "browser_mcp_result",
  "request_id": "uuid",
  "result": "{\"__rich_mcp__\": { ... }}",
  "rich_mcp": true
}
```

**Browser → Server: error**

```json
{
  "type": "browser_mcp_result",
  "request_id": "uuid",
  "error": "TypeError: Cannot read property 'x' of undefined"
}
```

### Timeouts

| Operation | Timeout |
|---|---|
| Worker initialization | 10 seconds |
| Individual tool call | 30 seconds |

---

## Built-in Document Plugins

The chat UI includes built-in document rendering for several formats. These are separate from browser MCP — they render fenced code blocks that the LLM writes directly (not via tool calls). However, browser MCP tools can create these same visualizations via the [Rich MCP Envelope](#rich-mcp-envelope).

See [document-plugins.md](document-plugins.md) for the complete reference, including:

- **Plotly charts** — interactive charts with export to SVG/PNG
- **Tables** — sortable data tables with XLSX/CSV export
- **Excel** — multi-sheet spreadsheets with tab navigation
- **Word documents** — structured docs with DOCX export
- **PowerPoint** — slide decks with PPTX export and fullscreen viewer
- **HTML sandbox** — custom HTML/CSS/JS apps in sandboxed iframes
- **LaTeX** — mathematical formulas via KaTeX

---

## API Reference

### Global APIs in Worker Context

| API | Returns | Description |
|---|---|---|
| `self._listDataFiles()` | `[{name, mimeType, size}]` | List all attached data files |
| `self._getDataFile(name)` | `{name, mimeType, size, base64, text}` | Get file with all data |
| `self._getDataFileText(name)` | `string` | Get extracted plain text only |
| `self._getDataFileBytes(name)` | `Uint8Array \| null` | Get raw binary content |

### Server Shim

| Method | Description |
|---|---|
| `server.tool(name, desc, schema, handler)` | Register a tool (high-level API) |
| `server.tool(name, schema, handler)` | Register a tool (no description) |
| `server.setRequestHandler(schema, handler)` | Register handler (low-level SDK API) |
| `server.connect(transport)` | No-op (compatibility) |
| `server.close()` | No-op (compatibility) |

### Schema Constants

| Constant | Value |
|---|---|
| `ListToolsRequestSchema` | `{ method: "tools/list" }` |
| `CallToolRequestSchema` | `{ method: "tools/call" }` |
| `ListResourcesRequestSchema` | `{ method: "resources/list" }` |
| `ReadResourceRequestSchema` | `{ method: "resources/read" }` |

---

## Complete Example: Sales Dashboard Droplet

**`droplet.json`:**

```json
{
  "name": "Sales Dashboard",
  "description": "Interactive sales analysis with charts and data queries",
  "system_prompt": "You are a sales analyst. Use the available tools to query sales data and generate visualizations. Always create charts when discussing trends.",
  "icon": "analytics",
  "category": "business",
  "mcp_entry_point": "index.js"
}
```

**`utils.js`:**

```javascript
export function parseCSV(text) {
  const lines = text.trim().split("\n");
  const headers = lines[0].split(",").map(h => h.trim());
  return lines.slice(1).map(line => {
    const vals = line.split(",").map(v => v.trim());
    return Object.fromEntries(headers.map((h, i) => [h, vals[i]]));
  });
}

export function groupBy(arr, key) {
  return arr.reduce((acc, item) => {
    const k = item[key];
    (acc[k] = acc[k] || []).push(item);
    return acc;
  }, {});
}
```

**`index.js`:**

```javascript
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { parseCSV, groupBy } from "./utils.js";

const server = new Server({ name: "sales-dashboard", version: "1.0.0" });

let salesData = null;

function getData() {
  if (!salesData) {
    salesData = parseCSV(self._getDataFileText("sales.csv"));
  }
  return salesData;
}

// Simple text query
server.tool("query_sales", "Query sales data with optional filters", {
  type: "object",
  properties: {
    region: { type: "string", description: "Filter by region" },
    year: { type: "integer", description: "Filter by year" },
    limit: { type: "integer", description: "Max rows to return", default: 10 }
  }
}, async ({ region, year, limit = 10 }) => {
  let data = getData();
  if (region) data = data.filter(r => r.region === region);
  if (year) data = data.filter(r => r.year === String(year));

  const rows = data.slice(0, limit);
  return JSON.stringify(rows, null, 2);
});

// Rich visualization
server.tool("revenue_chart", "Generate a revenue chart by region and quarter", {
  type: "object",
  properties: {
    year: { type: "integer", description: "Year to chart", default: 2025 }
  }
}, async ({ year = 2025 }) => {
  const data = getData().filter(r => r.year === String(year));
  const byRegion = groupBy(data, "region");

  const traces = Object.entries(byRegion).map(([region, rows]) => ({
    x: rows.map(r => r.quarter),
    y: rows.map(r => Number(r.revenue)),
    type: "bar",
    name: region,
  }));

  return JSON.stringify({
    __rich_mcp__: {
      version: "1.0",
      llm_summary: `[Generated revenue chart for ${year} with ${traces.length} regions]`,
      render: {
        type: "visualization",
        title: `Revenue by Region — ${year}`,
        payload: {
          data: traces,
          layout: {
            barmode: "group",
            xaxis: { title: "Quarter" },
            yaxis: { title: "Revenue ($)" },
          }
        }
      }
    }
  });
});

// Summary tool
server.tool("sales_summary", "Get a summary of sales metrics", {
  type: "object",
  properties: {
    year: { type: "integer" }
  },
  required: ["year"]
}, async ({ year }) => {
  const data = getData().filter(r => r.year === String(year));
  const totalRevenue = data.reduce((s, r) => s + Number(r.revenue), 0);
  const regions = [...new Set(data.map(r => r.region))];
  const quarters = [...new Set(data.map(r => r.quarter))];

  return [
    `Year: ${year}`,
    `Total Revenue: $${totalRevenue.toLocaleString()}`,
    `Regions: ${regions.join(", ")}`,
    `Quarters with data: ${quarters.join(", ")}`,
    `Total records: ${data.length}`,
  ].join("\n");
});

await server.connect(new StdioServerTransport());
```

**`sales.csv`:**

```csv
region,year,quarter,revenue,units
north,2025,Q1,520000,1200
north,2025,Q2,580000,1350
south,2025,Q1,410000,980
south,2025,Q2,445000,1050
east,2025,Q1,680000,1600
east,2025,Q2,720000,1700
```

**Upload and test:**

```bash
zip -r sales-dashboard.zip droplet.json index.js utils.js sales.csv

curl -X PUT https://example.com/api/llming/v1/droplets/_new/zip \
  -H "Authorization: Bearer llming_..." \
  -H "Content-Type: application/zip" \
  --data-binary @sales-dashboard.zip

# Send a chat message using the droplet
curl -N https://example.com/api/llming/v1/chat/send \
  -H "Authorization: Bearer llming_..." \
  -H "Content-Type: application/json" \
  -d '{"text": "Show me a revenue chart for 2025", "droplet_uid": "<uid-from-upload>"}'
```
