# CLAUDE.md

This file provides guidance to Claude Code when working with the Smart Clipboard MCP Server.

## Project Overview

Smart Clipboard gives AI agents persistent memory through intelligent clipboard management. When working with Claude, context is lost between conversations. This server solves that by providing a searchable, persistent memory that agents can write to and read from across sessions.

**Current Status: Production Ready v1.0.0**

## Quick Start

```bash
# For Claude Desktop users
claude mcp add smart-clipboard --uvx https://github.com/akougkas/smart-clipboard-mcp.git

# For development
git clone https://github.com/akougkas/smart-clipboard-mcp.git
cd smart-clipboard-mcp
uv sync
uv run smart-clipboard
```

## Essential Commands

### Development
```bash
# Install dependencies
uv sync

# Run the MCP server
uv run smart-clipboard

# Run tests
PYTHONPATH=. uv run pytest

# Run specific test
PYTHONPATH=. uv run pytest smart_clipboard/tests/test_mcp.py -v
```

### Code Quality
```bash
# Format code
uv run black smart_clipboard/

# Lint code  
uv run ruff check .

# Run all checks
uv run black smart_clipboard/ && uv run ruff check . && PYTHONPATH=. uv run pytest
```

## Architecture

```
Claude Desktop ──MCP stdio──> Smart Clipboard ──> LanceDB
                                    │
                                    └──> LiteLLM ──> AI Provider
```

### Key Components

- **`smart_clipboard/mcp_server.py`** - MCP stdio server implementation
- **`smart_clipboard/core/storage.py`** - LanceDB vector storage
- **`smart_clipboard/core/intelligence.py`** - LiteLLM AI integration  
- **`smart_clipboard/core/cache.py`** - High-performance caching
- **`smart_clipboard/models/schemas.py`** - Pydantic data models

### MCP Tools

1. **`clip_add`** - Store content with AI auto-tagging
2. **`clip_search`** - Semantic search using natural language
3. **`clip_list`** - List recent clips
4. **`clip_remove`** - Delete clips by ID
5. **`clip_stats`** - Usage statistics

## Performance Characteristics

- **Response Time**: <1ms for cached operations
- **Search Latency**: <20ms for semantic search
- **Memory Usage**: <50MB for 10k clips
- **Startup Time**: <2 seconds

## Project Structure

```
smart-clipboard-mcp/
├── smart_clipboard/
│   ├── __init__.py
│   ├── mcp_server.py       # Main MCP server
│   ├── core/
│   │   ├── __init__.py
│   │   ├── storage.py      # LanceDB operations
│   │   ├── intelligence.py # LiteLLM integration
│   │   └── cache.py        # Caching layer
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py      # Data models
│   └── tests/
│       ├── test_mcp.py
│       ├── test_storage.py
│       ├── test_intelligence.py
│       └── test_cache.py
├── config.yaml             # Configuration
├── pyproject.toml          # Package config
├── README.md               # User documentation
├── CLAUDE.md               # This file
└── LICENSE                 # MIT License
```

## Configuration

### Environment Variables (Optional)
```bash
# OpenAI (recommended)
export OPENAI_API_KEY="sk-..."

# Anthropic  
export ANTHROPIC_API_KEY="sk-ant-..."

# Local models
export OLLAMA_API_BASE="http://localhost:11434"
```

### config.yaml
```yaml
intelligence:
  embedding_model: "text-embedding-3-small"
  tagging_model: "gpt-3.5-turbo"
  
storage:
  path: "~/.smart-clipboard/lancedb"
  
cache:
  backend: "memory"  # or "redis"
```

## Design Principles

1. **Fast Response**: All operations return in <10ms via caching
2. **Graceful Fallbacks**: Works offline with local embeddings
3. **Agent-First**: Optimized for AI agent interactions
4. **Zero Config**: Works out of the box with auto-detection
5. **Universal AI**: Supports 100+ providers via LiteLLM

## Testing

```bash
# Full test suite
PYTHONPATH=. uv run pytest

# With coverage
PYTHONPATH=. uv run pytest --cov=smart_clipboard --cov-report=html

# Performance test
PYTHONPATH=. uv run pytest smart_clipboard/tests/test_mcp.py::TestSmartClipboardServer::test_add_clip_fast_response -v
```

## Author

**Antonios Kougkas**
- GitHub: [@akougkas](https://github.com/akougkas)
- Website: [akougkas.io](https://akougkas.io)
- Email: a.kougkas@gmail.com

## Notes for Claude Code

- This server provides persistent memory across your sessions
- Content is automatically tagged and indexed for semantic search
- The server handles all AI operations in the background
- Falls back to local processing when APIs are unavailable
- Optimized for storing code, configurations, and technical content