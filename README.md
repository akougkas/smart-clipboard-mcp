# Smart Clipboard MCP Server

Give AI agents persistent memory with intelligent clipboard management.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![MCP Protocol](https://img.shields.io/badge/MCP-2025--06--18-orange.svg)](https://modelcontextprotocol.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Why Smart Clipboard?

When working with AI agents like Claude, context is lost between conversations. Code snippets, configurations, and solutions disappear. Smart Clipboard solves this by giving agents a persistent, searchable memory they can write to and read from across sessions.

**Built for developers who use AI agents to code.**

## Key Features

- **Persistent Memory**: Store code, configs, and solutions across Claude sessions
- **Semantic Search**: Find content using natural language, not keywords
- **Auto-Tagging**: AI automatically categorizes content for better retrieval
- **Fast**: <1ms response times with intelligent caching
- **Works Offline**: Falls back to local embeddings when APIs are unavailable
- **Universal AI Support**: Auto-detects OpenAI, Anthropic, Ollama, LM Studio, and 100+ providers

## Installation

### For Claude Desktop Users

```bash
# Install from GitHub
claude mcp add smart-clipboard --uvx https://github.com/akougkas/smart-clipboard-mcp.git

# Or clone and run locally
git clone https://github.com/akougkas/smart-clipboard-mcp.git
cd smart-clipboard-mcp
claude mcp add smart-clipboard stdio "uv --directory $(pwd) run smart-clipboard"
```

### For Development

```bash
git clone https://github.com/akougkas/smart-clipboard-mcp.git
cd smart-clipboard-mcp
uv sync
uv run smart-clipboard
```

## Usage

Once connected to Claude Desktop, the agent can:

```text
Store: "Save this Docker configuration for Python microservices"
Search: "Find that nginx reverse proxy config from yesterday"
List: "Show my recent code snippets"
```

### Available MCP Tools

- `clip_add` - Store content with automatic tagging
- `clip_search` - Semantic search through stored content
- `clip_list` - List recent clips
- `clip_remove` - Remove specific clips
- `clip_stats` - Storage statistics

## Configuration

The server auto-detects available AI providers. To use specific providers, set environment variables:

```bash
# OpenAI (recommended for best embeddings)
export OPENAI_API_KEY="sk-..."

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."

# Local models (Ollama/LM Studio)
export OLLAMA_API_BASE="http://localhost:11434"
```

Edit `config.yaml` for advanced settings:

```yaml
intelligence:
  embedding_model: "text-embedding-3-small"
  tagging_model: "gpt-3.5-turbo"

storage:
  path: "~/.smart-clipboard/lancedb"

cache:
  backend: "memory"  # or "redis" if available
```

## Architecture

```
Claude Desktop ──MCP──> Smart Clipboard ──> LanceDB (vectors)
                              │
                              └──> LiteLLM ──> AI Provider
```

- **MCP Protocol**: Native stdio communication with Claude
- **LanceDB**: Local vector database for semantic search
- **LiteLLM**: Universal router for 100+ AI providers
- **Smart Caching**: Memory cache for instant responses

## Performance

- Response time: <1ms (cached), <10ms (uncached)
- Search latency: <20ms across 100k clips
- Memory usage: <50MB for 10k clips
- Startup time: <2 seconds

## Development

```bash
# Run tests
PYTHONPATH=. uv run pytest

# Format code
uv run black smart_clipboard/

# Lint
uv run ruff check .
```

## Contributing

Contributions welcome! This aims to be a reference implementation for MCP servers.

1. Fork the repository
2. Create your feature branch
3. Add tests for your changes
4. Ensure tests pass
5. Submit a pull request

## Author

**Antonios Kougkas**
- GitHub: [@akougkas](https://github.com/akougkas)
- Website: [akougkas.io](https://akougkas.io)
- Email: a.kougkas@gmail.com

## License

MIT License - see [LICENSE](LICENSE) file.

## Acknowledgments

- [Model Context Protocol](https://modelcontextprotocol.io) by Anthropic
- [LanceDB](https://lancedb.com) for vector storage
- [LiteLLM](https://litellm.ai) for universal AI support