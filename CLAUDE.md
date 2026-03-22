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

The debug router must be registered at app startup in the host app's server module (before any catch-all page handler). Routes added via `include_router` after the server is running will be shadowed by the catch-all and return 404.

## No NiceGUI Dependency

**llming-lodge has ZERO NiceGUI dependencies.** The chat page is served as pure static HTML with content-hashed JS/CSS references. All chat communication happens over a plain WebSocket (`/api/llming/ws/{session_id}`).

### Architecture

1. **`ChatPage.setup(app)`** — call once at startup to mount static files, API routes, and the chat page HTML route
2. **`ChatPage.create_session(user_config)`** — creates controller, loads nudges/MCP, returns `session_id`
3. **`GET /api/llming/chat/{session_id}`** — serves the static HTML page (requires valid `llming_auth` cookie)
4. **WebSocket cleanup** — full session cleanup (nudges, MCP, uploads, docs) happens automatically on WS disconnect

### Auth Flow (HMAC-signed cookie)

- `make_auth_cookie_value()` → returns `(session_id, signed_token)` — caller sets as HTTP cookie
- `verify_auth_cookie(request)` → checks the HMAC signature on the `llming_auth` cookie
- `sign_auth_token(session_id)` → creates `<session_id>.<hmac_sig>` token
- The cookie proves "this browser went through the host app's auth flow"
- Session IDs are server-generated UUIDs — unguessable
- Both cookie + valid session ID are required to access the chat page

### Host App Integration (NiceGUI bridge)

Host apps using NiceGUI for auth can bridge like this:
```python
# In @ui.page("/chat") handler, after auth:
session_id = await chat_page.create_session(user_config)
token = sign_auth_token(session_id)
ui.run_javascript(
    f'document.cookie="llming_auth={token};path=/;max-age=86400;samesite=lax";'
    f'location.replace("/api/llming/chat/{session_id}");'
)
```

### Dev Hot-Reload

In dev mode, `start_dev_file_watcher()` watches the chat static directory. When JS/CSS files change:
1. Content hashes are recomputed (cache busted)
2. A `{"type": "dev_reload"}` WebSocket message is sent to all connected clients
3. The browser reloads the page with fresh hashed URLs

### Cache Busting

All JS/CSS files are served with content-hash query strings: `/chat-static/chat-core.css?v=a1b2c3d4e5`. Hashes are computed at startup and invalidated by the dev file watcher.

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

### Script load order (in chat_page.py — `_build_script_phases()`)

The chat HTML page loads scripts in three sequential phases (each phase loads its scripts in parallel):

1. **Phase 1**: Vendor libs (marked, DOMPurify, KaTeX, CodeMirror base) + doc-plugin-registry + block-data-store + chat-features.js
2. **Phase 2**: CodeMirror addons + ai-edit-shared + builtin-plugins + bolt apps + all feature modules
3. **Phase 3**: `chat-app-core.js` (must be last — applies proto, boots app)

**CodeMirror load order**: CodeMirror addons (`javascript.min.js`, `lint.min.js`, etc.) must load in Phase 2 AFTER `codemirror.min.js` loads in Phase 1. They reference the global `CodeMirror` object at evaluation time — loading in parallel via `Promise.all` causes `CodeMirror is not defined`.

## Rich MCP Vendor Libraries (Sandbox Iframes)

Rich MCP tools (e.g. MathMCP 3D plots) render in **sandboxed iframes** using `srcdoc`. Because the iframe has `sandbox="allow-scripts"` (no `allow-same-origin`), it cannot fetch external resources. Instead, vendor libraries (Plotly, KaTeX) are **fetched as text by the parent page** and **inlined** into the srcdoc as `<script>` / `<style>` blocks.

### How it works

1. The MCP tool returns a `__rich_mcp__` envelope with `render.type = 'html_sandbox'` and `render.vendor_libs: ['plotly', 'katex_js', 'katex_css']`
2. Client-side `_resolveRichMcpVendorLibs()` in `builtin-plugins.js` fetches each lib as text from the paths in `_RICH_MCP_VENDOR_LIBS`
3. `_richMcpSandboxIframe()` injects the fetched text directly into the srcdoc: `<script>${plotlySource}</script>`
4. Results are cached in `_richMcpVendorCache` so subsequent renders don't re-fetch

### Path mapping

Vendor files live in `static/chat/vendor/` on disk. They are served at `/chat-static/vendor/` (mounted via `StaticFiles` in `server.py`).

```javascript
// builtin-plugins.js
const _RICH_MCP_VENDOR_LIBS = {
  plotly:    '/chat-static/vendor/plotly.min.js',
  katex_js:  '/chat-static/vendor/katex.min.js',
  katex_css: '/chat-static/vendor/katex.min.css',
};
```

### Adding a new vendor library

1. Download the minified file (must be MIT/BSD/Apache-2.0 licensed — **never AGPL/GPL**)
2. Place it in `static/chat/vendor/` (e.g. `static/chat/vendor/newlib.min.js`)
3. Add a license file: `static/chat/vendor/LICENSES/newlib-LICENSE`
4. Add attribution to `static/chat/vendor/THIRD-PARTY-NOTICES`
5. Add the mapping to `_RICH_MCP_VENDOR_LIBS` in `builtin-plugins.js`:
   ```javascript
   newlib: '/chat-static/vendor/newlib.min.js',
   ```
6. Reference it from the MCP tool's `render.vendor_libs` array: `['newlib']`

### Critical rules

- **`__rich_mcp__` envelopes must NEVER contain HTML, CSS, or JS.** The envelope stored in conversation history (IndexedDB) must only contain **pure data** (plot traces, layout config, LaTeX strings, step arrays, etc.). The client-side renderer builds HTML/CSS/JS from this data at render time. This ensures:
  - Style fixes and improvements automatically apply to ALL conversations (old and new)
  - Storage size stays small (no CSS/JS bloat in every message)
  - No stale rendering code baked into saved conversations
  - Same pattern as the nozzle code finder — data only, never presentation
- **Path prefix is `/chat-static/`**, NOT `/static/chat/`. The `static/chat/` directory is mounted at `/chat-static/` in `server.py`.
- **Always check `resp.ok`** before using fetch results. A catch-all page handler returns a full HTML page for 404s — if injected as a `<script>` tag in the srcdoc, it causes unexpected scripts to execute inside the sandboxed iframe, leading to cascading errors.
- **Never use CDN URLs** in vendor_libs — the sandbox iframe has no network access (no `allow-same-origin`). All libs must be inlined.
- Vendor files are fetched once and cached in memory (`_richMcpVendorCache`). Cache persists for the page session.

## E2E Testing with Playwright

### Overview

End-to-end tests run against a real server with mock Office 365 credentials. The mock user system (`LLMING_MOCK_USERS=1`) intercepts MS Graph API calls at the transport layer so all handlers, MCP tools, and the full chat pipeline work unchanged with synthetic data.

Tests use **pytest-playwright** (Python, not Node) to drive a real Chromium browser.

### Prerequisites

1. **Server running with mock users** — auto-reload **must be off** (file changes in `tests/` trigger a reload that kills mid-test sessions):
   ```bash
   # Option A: Inline launcher (recommended for CI)
   LLMING_MOCK_USERS=1 python -c "
   from lechler_ai.server import setup_app
   from nicegui import ui
   setup_app()
   ui.run(port=8080, reload=False, show=False)
   "

   # Option B: Normal app (auto-reload on — works if you don't edit files mid-test)
   LLMING_MOCK_USERS=1 poetry run python -m lechler_ai.app
   ```

2. **Redis + MongoDB running** (for token cache, user data, nudges)

3. **At least one LLM provider** configured (e.g. `ANTHROPIC_API_KEY` for Haiku)

4. **Playwright browsers installed**:
   ```bash
   poetry run python -m playwright install chromium
   ```

### Running tests

```bash
cd lechler_ai
poetry run pytest tests/e2e/ -v --timeout=180
```

Override the server URL:
```bash
TEST_BASE_URL=https://localhost:8443 poetry run pytest tests/e2e/ -v
```

### Writing a new test

#### 1. Use the `chat_page` fixture

Every test gets a fresh browser context (empty IndexedDB, no stale cookies). The `chat_page` fixture handles mock login and waits for the chat UI to be ready:

```python
from .conftest import chat_page, select_model, send_and_wait

def test_something(chat_page):
    page = chat_page  # already logged in, textarea visible
```

#### 2. Key selectors

| Element | Selector | Notes |
|---------|----------|-------|
| Text input | `#cv2-textarea` | Real `<textarea>`, use `fill()` |
| Send button | `#cv2-send-btn` | |
| Model button | `#cv2-model-btn` | Click to open dropdown |
| Model option | `.cv2-model-option[data-model="..."]` | Use `filter(has_text=...)` |
| Incognito toggle | `#cv2-incognito-toggle` | Visible before first message |
| New chat | `#cv2-new-chat` | Exits incognito, sends WS `new_chat` |
| Assistant message | `.cv2-msg-assistant` | Contains `.cv2-msg-body` + `.cv2-tool-area` |
| Tool call indicator | `.cv2-tool-area` | Contains tool name text |
| Document block | `.cv2-doc-plugin-block` | Plotly, tables, etc. |
| Email draft | `.cv2-email-draft` | Full email composer UI |
| Conversation item | `.cv2-conv-item` | In sidebar, has `data-id` |
| Chat app root | `#chat-app` | Gets `.cv2-incognito` class |

#### 3. Chat app globals

The chat JS app is accessible at `window.__chatApp`:

```python
# Check streaming state
page.evaluate("() => window.__chatApp.streaming")

# Read IndexedDB conversation count
page.evaluate("() => window.__chatApp.idb.getAll().then(a => a.length)")

# Check incognito
page.evaluate("() => window.__chatApp.incognito")

# Current model
page.evaluate("() => window.__chatApp.currentModel")
```

#### 4. Helper functions (from `conftest.py`)

| Helper | Usage |
|--------|-------|
| `select_model(page, "Haiku")` | Opens dropdown, clicks model by label text |
| `send_and_wait(page, "Hello")` | Fills textarea, clicks send, waits for streaming start→end |
| `get_conversation_count(page)` | Returns number of `.cv2-conv-item` in sidebar |
| `enable_tool(page, "Email Drafts")` | Clicks tool toggle via JS (works for hidden flyout menus) |

#### 5. Gotchas

- **Tool toggles reset after each response.** The server's tool discovery re-syncs `enabled_tools` after every response. Re-enable tools with `enable_tool()` before a follow-up message that needs them.

- **Tool pending indicators persist.** `.cv2-tool-pending` stays in the DOM even after the tool completes on the server. Don't use `wait_for_tools_done()` — verify artifacts directly (`.cv2-doc-plugin-block`, `.cv2-email-draft`).

- **Auto-reload kills sessions.** NiceGUI's WatchFiles monitors `lechler_ai/`, `salesbot/`, and `dependencies/`. Editing test files triggers a reload. Use `reload=False`.

- **`send_and_wait` relies on `window.__chatApp.streaming`.** It waits for streaming to start (`true`) then end (`false`). If the AI response is extremely fast (<100ms), there's a theoretical race — practically never happens with network latency.

- **Fresh browser context per test.** Each test gets a new browser context (pytest-playwright default). IndexedDB is empty, cookies are fresh. Tests are fully isolated.

- **Incognito sidebar state.** In incognito mode, `idb.put()` is a no-op but `_updateSidebarConversation()` still adds the conversation to the in-memory array. The sidebar shows it during the session, but it's not persisted. Verify persistence via `window.__chatApp.idb.getAll()`, not via sidebar DOM.

### Example: Testing a nudge

```python
def test_math_nudge(chat_page):
    page = chat_page

    # Activate the math nudge from sidebar
    page.locator('.cv2-nudge-inline[data-nudge-uid="math-uid"]').click()
    page.wait_for_timeout(2_000)

    # Send a math problem
    send_and_wait(page, "Solve x^2 - 5x + 6 = 0")

    # Verify the response contains the solution
    last_body = page.locator(".cv2-msg-assistant .cv2-msg-body").last
    body_text = last_body.inner_text().lower()
    assert "x = 2" in body_text or "x = 3" in body_text
```

### Example: Testing conversation persistence

```python
def test_conversation_saved(chat_page):
    page = chat_page
    select_model(page, "Haiku")
    send_and_wait(page, "Hello!")

    # Verify conversation was saved to IndexedDB
    stored = page.evaluate(
        "() => window.__chatApp.idb.getAll().then(a => a.length)"
    )
    assert stored == 1

    # Verify it appears in the sidebar
    assert get_conversation_count(page) >= 1
```

## Chat Frontend Performance

### Scroll — critical rules

- **NEVER use `scroll-behavior: smooth` on `.cv2-messages`**. Mac trackpads send 60-120+ wheel events/sec; CSS smooth scroll queues overlapping animations causing multi-second UI hangs.
- `.cv2-messages` must have `overscroll-behavior-y: contain` (prevent scroll chaining) + `will-change: transform` (GPU compositor layer).
- `.cv2-doc-plugin-block` must have `contain: layout style paint` — isolates document blocks from parent layout during scroll reflow.
- `.cv2-doc-table-wrapper` must have `contain: layout style` — isolates table layout from scroll reflow.

### ResizeObserver

- Only create a ResizeObserver on windowed previews when a Plotly chart is present (`[data-plotly]`). Non-Plotly previews (tables, word, pptx) don't need it. Always debounce (100ms).
