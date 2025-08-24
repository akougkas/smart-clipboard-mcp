#!/usr/bin/env python3
"""Smart Clipboard MCP Server - Production Implementation."""

import asyncio
import json
import hashlib
import time
from typing import Any, Dict, List, Optional

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from mcp.server.models import InitializationOptions
from mcp.server.lowlevel import NotificationOptions

from smart_clipboard.core.storage import ClipStorage
from smart_clipboard.core.intelligence import UniversalIntelligence
from smart_clipboard.core.cache import ClipCache


class SmartClipboardMCPServer:
    """Production-ready MCP Server for Smart Clipboard."""

    def __init__(self):
        """Initialize MCP server with proper protocol compliance."""
        self.storage = ClipStorage()
        self.ai = UniversalIntelligence()
        self.cache = ClipCache()

        # Initialize MCP server with proper name
        self.app = Server("smart-clipboard")
        self._register_handlers()

    def _register_handlers(self):
        """Register MCP protocol handlers."""

        @self.app.list_tools()
        async def list_tools() -> List[Tool]:
            """Return MCP tools following 2025-06-18 specification."""
            return [
                Tool(
                    name="clip_add",
                    description="Store content with AI-powered semantic tagging",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "Content to store in smart clipboard",
                            },
                            "source": {
                                "type": "string",
                                "description": "Source identifier (manual|agent|claude)",
                                "default": "manual",
                            },
                            "tags": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Optional manual tags",
                                "default": [],
                            },
                        },
                        "required": ["content"],
                    },
                ),
                Tool(
                    name="clip_search",
                    description="Semantic search through clipboard history",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Natural language search query",
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum results to return",
                                "default": 5,
                                "minimum": 1,
                                "maximum": 50,
                            },
                        },
                        "required": ["query"],
                    },
                ),
                Tool(
                    name="clip_list",
                    description="List recent clipboard entries",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "limit": {
                                "type": "integer",
                                "description": "Number of entries to return",
                                "default": 10,
                                "minimum": 1,
                                "maximum": 100,
                            }
                        },
                    },
                ),
                Tool(
                    name="clip_remove",
                    description="Remove clipboard entry by ID",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "clip_id": {
                                "type": "string",
                                "description": "Unique clip identifier",
                            }
                        },
                        "required": ["clip_id"],
                    },
                ),
                Tool(
                    name="clip_stats",
                    description="Get clipboard usage statistics",
                    inputSchema={"type": "object", "properties": {}},
                ),
            ]

        @self.app.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            """Handle MCP tool calls with structured output."""
            try:
                result = await self._dispatch_tool_call(name, arguments)
                return [
                    TextContent(
                        type="text",
                        text=json.dumps(result, indent=2, ensure_ascii=False),
                    )
                ]
            except Exception as e:
                error_result = {"error": str(e), "tool": name, "arguments": arguments}
                return [
                    TextContent(type="text", text=json.dumps(error_result, indent=2))
                ]

    async def _dispatch_tool_call(
        self, name: str, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Dispatch tool calls to appropriate handlers."""
        handlers = {
            "clip_add": self._handle_clip_add,
            "clip_search": self._handle_clip_search,
            "clip_list": self._handle_clip_list,
            "clip_remove": self._handle_clip_remove,
            "clip_stats": self._handle_clip_stats,
        }

        handler = handlers.get(name)
        if not handler:
            raise ValueError(f"Unknown tool: {name}")

        return await handler(arguments)

    async def _handle_clip_add(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle clip_add tool call."""
        content = args["content"]
        source = args.get("source", "manual")
        manual_tags = args.get("tags", [])

        # Generate unique ID
        clip_id = self._generate_clip_id(content)

        # Fast response with background processing
        clip_data = {
            "id": clip_id,
            "content": content,
            "source": source,
            "manual_tags": manual_tags,
            "timestamp": time.time(),
        }

        # Store in cache immediately
        await self.cache.set(clip_id, clip_data)

        # Background processing
        asyncio.create_task(self._process_clip_background(clip_id, clip_data))

        return {
            "id": clip_id,
            "status": "stored",
            "message": f"Content stored with ID {clip_id}",
            "processing": "background",
        }

    async def _handle_clip_search(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle clip_search tool call."""
        query = args["query"]
        limit = args.get("limit", 5)

        # Check cache first
        cache_key = f"search:{hashlib.md5(query.encode()).hexdigest()}:{limit}"
        cached = await self.cache.get(cache_key)
        if cached:
            cached["from_cache"] = True
            return cached

        # Generate embedding and search
        embedding = await self.ai.generate_embedding(query)
        if not embedding:
            return {"error": "Failed to generate search embedding"}

        results = await self.storage.search(embedding, limit)

        response = {
            "query": query,
            "results": results,
            "count": len(results),
            "from_cache": False,
        }

        # Cache results
        await self.cache.set(cache_key, response, ttl=300)
        return response

    async def _handle_clip_list(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle clip_list tool call."""
        limit = args.get("limit", 10)
        clips = await self.storage.list_recent(limit)

        return {"clips": clips, "count": len(clips), "limit": limit}

    async def _handle_clip_remove(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle clip_remove tool call."""
        clip_id = args["clip_id"]

        success = await self.storage.remove(clip_id)
        if success:
            await self.cache.delete(clip_id)
            return {"id": clip_id, "status": "removed"}
        else:
            return {"id": clip_id, "status": "not_found"}

    async def _handle_clip_stats(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle clip_stats tool call."""
        stats = await self.storage.get_stats()
        return stats

    async def _process_clip_background(self, clip_id: str, clip_data: Dict[str, Any]):
        """Background processing for clips."""
        try:
            content = clip_data["content"]
            manual_tags = clip_data.get("manual_tags", [])

            # Generate embedding
            embedding = await self.ai.generate_embedding(content)

            # Auto-generate tags
            auto_tags = await self.ai.generate_tags(content)

            # Combine tags
            all_tags = list(set(manual_tags + auto_tags))

            # Store in database
            await self.storage.store(
                clip_id=clip_id,
                content=content,
                embedding=embedding,
                tags=all_tags,
                metadata={
                    "source": clip_data["source"],
                    "timestamp": clip_data["timestamp"],
                    "processed_at": time.time(),
                },
            )

        except Exception as e:
            print(f"Background processing error for {clip_id}: {e}")

    def _is_agent_source(self, source: str) -> bool:
        """Detect if source indicates AI agent."""
        agent_indicators = ["agent", "claude", "assistant", "ai"]
        return any(indicator in source.lower() for indicator in agent_indicators)

    def _generate_clip_id(self, content: str) -> str:
        """Generate unique clip ID."""
        timestamp = str(time.time())
        data = f"{content}{timestamp}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    async def run(self):
        """Run MCP server with proper protocol compliance."""
        async with stdio_server() as (read_stream, write_stream):
            await self.app.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="smart-clipboard",
                    server_version="1.0.0",
                    capabilities=self.app.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )


async def async_main():
    """Main async entry point."""
    server = SmartClipboardMCPServer()
    await server.run()


def main():
    """Synchronous entry point for console script."""
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("Smart Clipboard MCP Server stopped")
    except Exception as e:
        print(f"Server error: {e}")
        raise


if __name__ == "__main__":
    main()
