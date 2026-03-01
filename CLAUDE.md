# llming-lodge

## Licensing

**NEVER use AGPL or GPL licensed libraries.** All dependencies must be permissively licensed (MIT, BSD, Apache-2.0, etc.). This is critical for the project's commercial use.

## No Disk I/O on Server

**NEVER write files to disk on the server.** All file data (uploads, nudge files, project files, document extraction) must stay in memory. No `tempfile`, no `write_bytes()`, no temp directories. The extractors (`extract_text`, `extract_pdf`, `extract_docx`, `extract_xlsx`) accept both `Path` and `bytes` — always pass `bytes` for server-side operations. `FileAttachment` stores data in `raw_data: bytes`, not on disk.

## Testing Chat via Debug API

**Do NOT use JavaScript injection (Chrome MCP `javascript_tool`) to trigger chat messages.**
Use the Chat Debug API instead — a single POST sends text and triggers the full send pipeline (including message intercepts like dev MCP activation commands).

**NEVER use Chrome MCP browser interaction for testing chat.** Always use the Debug API for all chat testing: sending messages, activating dev MCPs, checking responses. Only use Chrome MCP for visual verification when explicitly asked by the user.

### Setup

Set these env vars in the project `.env` (already configured for local dev):

```
DEBUG_CHAT_REMOTE=1
DEBUG_CHAT_API_KEY=<secure key>
```

The debug router is only mounted locally (`server=False`) and requires both env vars. It is never mounted in release/server mode.

### URL Prefix

All endpoints live under `/api/llming/debug/` (note: `API_PREFIX = "/api/llming"`).

Header required: `x-debug-key: <DEBUG_CHAT_API_KEY>`

### Endpoints

**List sessions:**
```bash
curl http://localhost:8080/api/llming/debug/sessions -H "x-debug-key: $DEBUG_CHAT_API_KEY"
```

**Send a message (trigger full chat round-trip):**
```bash
curl -X POST "http://localhost:8080/api/llming/debug/sessions/{SESSION_ID}/send" \
  -H "x-debug-key: $DEBUG_CHAT_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, what can you do?"}'
```

**Check streaming status:**
```bash
curl "http://localhost:8080/api/llming/debug/sessions/{SESSION_ID}/status" \
  -H "x-debug-key: $DEBUG_CHAT_API_KEY"
```

**Get full session detail (history, state, tools):**
```bash
curl "http://localhost:8080/api/llming/debug/sessions/{SESSION_ID}" \
  -H "x-debug-key: $DEBUG_CHAT_API_KEY"
```

### Workflow

1. Open the chat page in the browser (creates a session)
2. `GET /api/llming/debug/sessions` to find the session ID
3. `POST /api/llming/debug/sessions/{id}/send` with `{"text": "..."}` to send a message
4. `GET /api/llming/debug/sessions/{id}/status` to check if streaming is done
5. `GET /api/llming/debug/sessions/{id}` to see full conversation history

### Route Registration

The debug router must be registered at app startup in the host app's server module (before NiceGUI's catch-all page handler). Routes added via `include_router` after the server is running will be shadowed by the catch-all and return 404.

## Two Types of Documents

There are TWO distinct document concepts — always differentiate between them:

1. **Attachments / Knowledge** — static files (PDF, DOCX, XLSX, etc.) attached to a nudge or project. These are converted to text and injected into the system prompt automatically as context. Managed by `NudgeFileCache` (nudge_store.py) and `_inject_preset_files` (chat_session_api.py). Relevant for nudges and projects.

2. **Dynamic Documents** — documents the user actively creates during a chat session (plots, HTML sandboxes, Word docs, tables, etc.). Managed by the `DocPluginManager` and MCP-based document tools (Plotly, Tables, Excel, Word, PowerPoint, HTML Sandbox). These are NOT relevant for nudge/project file attachments.

When working on nudge/project file injection, focus only on type 1 (attachments/knowledge). The dynamic document system (type 2) is separate.

## Fonts and External Resources

**NEVER import fonts from Google Fonts CDN** (`fonts.googleapis.com` / `fonts.gstatic.com`). This is strictly forbidden due to tracking. Instead, download the font file (woff2), bundle it locally in `static/chat/vendor/fonts/`, use a local `@font-face` declaration, and add proper attribution to `vendor/THIRD-PARTY-NOTICES` + a license file in `vendor/LICENSES/`.

## Architecture

The chat frontend is modularized into ~14 JS files + ~9 CSS files using a prototype mixin + feature registry pattern:

1. `chat-features.js` — loaded first, declares empty `ChatApp` class + `_ChatAppProto` accumulator + `ChatFeatures` registry
2. Feature modules (e.g. `chat-voice.js`) — add methods to `_ChatAppProto` and register with `ChatFeatures`
3. `chat-app-core.js` — loaded last, applies all accumulated methods to `ChatApp.prototype`, defines constructor/init/render/bindEvents, boots the app

### Script load order (in chat_page.py)

Vendor scripts -> Plugin scripts -> Feature modules -> `chat-app-core.js` (must be last)

**CodeMirror load order**: CodeMirror addons (`javascript.min.js`, `lint.min.js`, etc.) must load in Phase 2 AFTER `codemirror.min.js` loads in Phase 1. They reference the global `CodeMirror` object at evaluation time — loading in parallel via `Promise.all` causes `CodeMirror is not defined`. See `_vendor_scripts` vs `_codemirror_addons` in `chat_page.py`.

## Chat Frontend Performance

### Scroll — critical rules

- **NEVER use `scroll-behavior: smooth` on `.cv2-messages`**. Mac trackpads send 60-120+ wheel events/sec; CSS smooth scroll queues overlapping animations causing multi-second UI hangs.
- `.cv2-messages` must have `overscroll-behavior-y: contain` (prevent scroll chaining) + `will-change: transform` (GPU compositor layer).
- `.cv2-doc-plugin-block` must have `contain: layout style paint` — isolates document blocks from parent layout during scroll reflow.
- `.cv2-doc-table-wrapper` must have `contain: layout style` — isolates table layout from scroll reflow.

### ResizeObserver

- Only create a ResizeObserver on windowed previews when a Plotly chart is present (`[data-plotly]`). Non-Plotly previews (tables, word, pptx) don't need it. Always debounce (100ms).
