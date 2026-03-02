# LLMing-Lodge

<p align="center">
  <img src="llming_lodge/media/llming_lodge_relax_small.png" alt="LLMing-Lodge" />
</p>

A cozy lodge where LLMings stream, use tools, and stay within budget.

LLMing-Lodge is a full-stack AI chat platform: a streaming LLM backend with a rich web UI, full Model Context Protocol (MCP) support, interactive document generation, voice capabilities, and a pluggable multi-provider architecture. Run it standalone or embed it into any FastAPI/Starlette app.

## Features

### Multi-Provider LLM Backend

- **6+ providers**: OpenAI, Anthropic, Google Gemini, Mistral, DeepSeek, Together AI — switch models at runtime.
- **Generic OpenAI-compatible endpoints**: Register any OpenAI-compatible API via `LLMManager.register_openai_compatible()`.
- **Azure OpenAI**: Full support including the Realtime voice API.
- **Async streaming**: Token-by-token streaming with tool call events over WebSocket.
- **Model metadata**: Per-model speed, quality, cost ratings, context window size, and capability flags (vision, reasoning, image generation).

### Tools & MCP

Full [Model Context Protocol](https://modelcontextprotocol.io/) implementation with five transport types:

| Transport | Description |
|---|---|
| **Stdio** | Local subprocess MCP server (JSON-RPC over stdin/stdout) |
| **HTTP/SSE** | Remote MCP server over HTTP |
| **In-Process** | Direct Python calls, zero overhead |
| **Browser Web Worker** | JavaScript MCP servers running client-side in a Web Worker |
| **Provider-Native** | Provider-handled tools (e.g., OpenAI web search) |

Built-in tools include image generation (GPT Image) and web search. Nudges can package JavaScript MCP servers that run entirely in the browser — enabling client-side tooling without exposing anything server-side.

### Interactive Document Plugins

The LLM can generate rich, editable documents inline using fenced code blocks. Each document type has a dedicated MCP editing server that auto-activates on first use:

| Document Type | Capabilities |
|---|---|
| **Plotly Charts** | Interactive charts with per-trace editing tools |
| **Tables** | Data grids with row/column manipulation, XLSX export |
| **Text Documents** | Structured Word-like documents with DOCX export |
| **Presentations** | Slide decks with branded template support and PPTX export |
| **HTML Sandbox** | Full HTML/CSS/JS web apps in a sandboxed iframe with CodeMirror editor |
| **LaTeX** | Math formulas rendered via KaTeX |
| **Email Drafts** | Composable emails with send/draft actions |

Documents support **cross-block references** — a Plotly chart can be embedded inside a presentation slide via `{"$ref": "<uuid>"}`.

### Voice & Speech

- **Speech-to-Text**: Record voice messages transcribed via `gpt-4o-transcribe` (Whisper fallback). Live waveform visualization, 90-second max, voice activity detection.
- **Text-to-Speech**: Sentence-level streaming TTS — audio synthesis starts while the LLM is still generating. 13 voices, 7 locales.
- **Realtime Voice**: Azure OpenAI Realtime API with server-side VAD, push-to-talk, and tool calling — API keys never leave the server.
- **Speech Mode**: Continuous voice conversation with automatic recording cycles.

### Budget Management

Track and limit token/cost usage per period (hourly, daily, weekly, monthly, lifetime):

- **MongoDB backend** for persistent multi-process budgets.
- **In-memory backend** for lightweight single-process usage.
- Automatic cost calculation using provider-specific per-million-token pricing.
- Budget reservation before LLM calls with unused budget returned on cancellation.
- Optional `BudgetHandler` callback for external billing system integration.

### Nudges & Projects

**Projects** group conversations under shared settings (system prompt, model, language, attached files, enabled plugins). Stored client-side in IndexedDB.

**Nudges** extend projects with collaboration features:

- Creator attribution, descriptions, prompt starters, per-tool capability overrides.
- **Dev/live versioning**: creators iterate on a dev version, then publish to live.
- **Visibility ACLs**: control access via fnmatch globs (e.g., `["*@acme.com"]`).
- **Master nudges**: team-level nudges silently applied to every session.
- **Auto-discover nudges**: appear in the LLM's system prompt so it can suggest and activate them.
- **Browser MCP nudges**: package JavaScript MCP servers that run as Web Workers in the user's browser.
- Full CRUD + search + favorites via MongoDB (`NudgeStore`).

### Knowledge & Document Context

- Attach PDF, DOCX, XLSX, and plain text files to nudges or projects as knowledge context.
- Files are extracted to text and injected into the system prompt automatically (up to 30% of context window, max 150K tokens).
- **PDF visual inspection**: `PdfViewerMCP` renders PDF pages as JPEG at 150 DPI and injects them as vision input for chart/diagram/layout analysis.
- In-memory file cache with TTL-based eviction — no disk I/O on the server.

### Context Management

- **Automatic condensation**: When context usage exceeds a configurable threshold (default 80%), conversation history is condensed using the cheapest available model.
- **Image history limit**: Configurable max images in history (default 20); oldest are pruned.
- **Prompt inspector** (admin): Token-level breakdown of every context component — preamble, system prompt, master prompt, documents, condensed summary, auto-discover suffix, messages, and tool definitions.

### Chat Frontend

A modular vanilla-JS frontend (~14 JS + ~9 CSS files) using a prototype mixin pattern:

- Markdown rendering (marked.js + DOMPurify + KaTeX)
- Image upload, paste, drag-drop with lightbox viewer
- Conversation search (Cmd+K / Ctrl+K) over IndexedDB
- Context usage indicator
- Tool toggle menu with status display
- Model selector with speed/quality/cost ratings
- Conversation sidebar with draft persistence and restore
- Nudge explorer with search and favorites

### AI Text Editing

One-shot LLM calls for inline text editing outside of chat history:

Fix grammar, formalize, casualize, shorten, expand, improve, simplify, convert to bullet points, or translate (DE, EN, FR, IT, ZH, ES). Per-language business writing styles with appropriate greetings and closings.

Also supports AI-powered **typeahead** text completion.

### Internationalization

Built-in i18n with locale fallback chains:

`de-swg` (Swabian) → `de-de` → `en-us`, `fr-fr` → `en-us`, `it-it` → `en-us`, `hi-in` → `en-us`, `en-in` → `en-us`

Translations are injected into the frontend at session init for client-side lookups.

## Installation

```bash
pip install llming-lodge
```

Or from source with [Poetry](https://python-poetry.org/):

```bash
git clone https://github.com/Alyxion/llming-lodge.git
cd llming-lodge
poetry install
```

## Quick Start

### As a Library

```python
import asyncio
from llming_lodge import ChatSession, LLMConfig

async def main():
    config = LLMConfig(provider="openai", model="gpt-4o", temperature=0.7)
    session = ChatSession(config=config)

    async for chunk in session.stream_async("Hello, how are you?"):
        print(chunk.content, end="", flush=True)

asyncio.run(main())
```

### Standalone Server

Run a full chat UI with HTTPS — no host framework required:

```bash
llming-lodge            # starts on https://localhost:8443
```

A self-signed TLS certificate is auto-generated if none is provided.

### Embedded in a Host App

Mount the chat routes on any FastAPI or Starlette application:

```python
from llming_lodge.server import setup_routes

setup_routes(app, debug=True, nudge_store=store)
```

## Configuration

Set the provider keys you need as environment variables:

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | OpenAI models, image generation, TTS/STT |
| `ANTHROPIC_API_KEY` | Anthropic Claude models |
| `GEMINI_KEY` | Google Gemini models |
| `MISTRAL_API_KEY` | Mistral models |
| `TOGETHER_API_KEY` | Together AI / DeepSeek models |
| `MONGODB_URI` | Budget persistence + nudge storage (optional) |
| `PORT` | Server port (default: 8443) |
| `HTTPS` | Enable HTTPS (default: `1`; set to `0` for HTTP) |
| `SSL_CERTFILE` / `SSL_KEYFILE` | Custom TLS certificate paths |
| `SYSTEM_PROMPT` | Default system prompt for standalone mode |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint (for Realtime voice) |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key |

## Architecture

```
llming_lodge/
├── standalone.py              # CLI entry point (FastAPI + Uvicorn + TLS)
├── server.py                  # Route builder, static files, session store
├── session.py                 # ChatSession + LLMConfig
├── chat_controller.py         # Core chat logic (streaming, tools, context)
├── llm_provider_manager.py    # Multi-provider orchestration
├── providers/                 # Provider implementations (OpenAI, Anthropic, Gemini, ...)
├── api/
│   ├── chat_session_api.py    # WebSocket controller + session registry
│   ├── debug_api.py           # REST debug/admin API
│   ├── dev_dashboard.py       # System health monitoring dashboard
│   └── ai_edit_handler.py     # One-shot AI edit/typeahead
├── doc_plugins/               # Document MCP servers (Plotly, Tables, PPTX, ...)
├── documents/                 # File upload, text extraction, PDF viewer MCP
├── tools/                     # Tool registry, MCP connections (stdio, HTTP, in-process, browser)
├── budget/                    # Token/cost budget management
├── nudge_store.py             # MongoDB-backed nudge CRUD + ACL
├── speech_service.py          # STT + TTS via OpenAI
├── i18n/                      # Internationalization engine + translations
├── static/chat/               # Frontend JS + CSS modules
└── monitoring/                # Event loop heartbeat + system metrics
```

The frontend uses a **prototype mixin + feature registry pattern**: `chat-features.js` declares the registry, feature modules add methods, and `chat-app-core.js` (loaded last) assembles them into the `ChatApp` class.

Communication between frontend and backend is over a single WebSocket connection per session, using a typed JSON message protocol.

## License

**LLMing-Lodge** is licensed under the **Business Source License 1.1 (BSL 1.1)**.

- **Free for small organizations** (fewer than 20 employees).
- **Commercial license required** for larger organizations.
- **Converts to Apache 2.0** after 4 years per version.

See [LICENSE](LICENSE) and [LICENSE_COMMERCIAL.md](LICENSE_COMMERCIAL.md) for details.
