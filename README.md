# LLMing-Lodge

<p align="center">
  <img src="llming_lodge/media/llming_lodge_relax_small.png" alt="LLMing-Lodge" />
</p>

A cozy lodge where LLMings stream, use tools, and stay within budget.

## Features

- **Multi-Provider**: OpenAI, Anthropic, Google Gemini, Mistral, DeepSeek, Together AI — switch at runtime.
- **Streaming**: Async and sync streaming with tool call events.
- **Tools & MCP**: Built-in tool system with full [Model Context Protocol](https://modelcontextprotocol.io/) support (stdio, HTTP, in-process).
- **Budget Management**: Track and limit token usage with MongoDB or in-memory backends.
- **Chat Sessions**: Managed conversation history with automatic context condensation.
- **Document Processing**: Extract text from PDF, DOCX, and XLSX for context injection.
- **Standalone Server**: Run as an HTTPS chat server with a built-in web UI — no framework required.

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

Run a full chat UI with HTTPS and no host framework:

```bash
cp .env.example .env   # add your API keys
llming-lodge            # starts on https://localhost:8443
```

## Auth

Lodge doesn't own auth logic — it reads cookie names and verifies tokens through the shared `llming_com.get_auth()` singleton. That singleton is the same one used by any sibling page in the host (e.g. an llming-hub landing page), so **sign-in at one page carries over to the chat page without a second OAuth round-trip**.

- Lodge's `chat_page.py` reads `auth.session_cookie_name` / `auth.auth_cookie_name` / `auth.identity_cookie_name` from the instance — never from module-level constants.
- The host application is responsible for running the OAuth flow and setting the cookies on `path="/"` so every page under the domain can see them.
- See llming-com's [Unified auth across pages](../llming-com/README.md#unified-auth-across-pages) for the full flow and the `app_name` pattern for multi-app isolation on shared domains.

## Configuration

Copy `.env.example` and add the provider keys you need:

| Variable | Provider |
|---|---|
| `OPENAI_API_KEY` | OpenAI |
| `ANTHROPIC_API_KEY` | Anthropic |
| `GEMINI_KEY` | Google Gemini |
| `MISTRAL_API_KEY` | Mistral |
| `TOGETHER_API_KEY` | Together AI |
| `MONGODB_URI` | Budget persistence (optional) |

## License

**LLMing-Lodge** is licensed under the **Business Source License 1.1 (BSL 1.1)**.

- **Free for small organizations** (fewer than 20 employees).
- **Commercial license required** for larger organizations.
- **Converts to Apache 2.0** after 4 years per version.

See [LICENSE](LICENSE) and [LICENSE_COMMERCIAL.md](LICENSE_COMMERCIAL.md) for details.
