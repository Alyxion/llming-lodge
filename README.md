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

- **Free for small organizations** (fewer than 1,500 employees).
- **Commercial license required** for larger organizations.
- **Converts to Apache 2.0** after 4 years per version.

See [LICENSE](LICENSE) and [LICENSE_COMMERCIAL.md](LICENSE_COMMERCIAL.md) for details.
