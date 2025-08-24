"""Tests for MCP server functionality."""

import asyncio
import json
import pytest
from unittest.mock import Mock, patch, AsyncMock

# Import the server components
from smart_clipboard.mcp_server import SmartClipboardMCPServer as SmartClipboardServer
from smart_clipboard.core.storage import ClipStorage
from smart_clipboard.core.intelligence import UniversalIntelligence
from smart_clipboard.core.cache import ClipCache


class TestSmartClipboardServer:
    """Test the main MCP server functionality."""

    @pytest.fixture
    async def server(self):
        """Create a test server instance."""
        server = SmartClipboardServer()
        # Override with mock components for testing
        server.storage = Mock(spec=ClipStorage)
        server.ai = Mock(spec=UniversalIntelligence)
        server.cache = Mock(spec=ClipCache)
        return server

    @pytest.mark.asyncio
    async def test_add_clip_fast_response(self, server):
        """Test that add_clip returns quickly via fast path."""
        # Setup mocks
        server.ai.should_store = AsyncMock(return_value=True)
        server.cache.set = AsyncMock()

        content = "Test content for fast response"

        # Measure response time
        start_time = asyncio.get_event_loop().time()
        result = await server._add_clip(content, "manual")
        end_time = asyncio.get_event_loop().time()

        # Verify response time is under 10ms
        response_time = end_time - start_time
        assert (
            response_time < 0.01
        ), f"Response took {response_time:.3f}s, should be <0.01s"

        # Verify response structure
        assert result["status"] == "stored"
        assert "id" in result
        assert len(result["id"]) == 12  # MD5 hash truncated to 12 chars

        # Verify cache was called
        server.cache.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_agent_mode_detection(self, server):
        """Test agent mode is properly detected."""
        server.ai.should_store = AsyncMock(return_value=True)
        server.cache.set = AsyncMock()

        # Test various agent source indicators
        agent_sources = ["agent", "claude", "Claude Code", "AI Assistant"]

        for source in agent_sources:
            result = await server._add_clip("Test content", source)
            assert result["status"] == "stored"
            assert "agent mode" in result["message"]

    @pytest.mark.asyncio
    async def test_agent_mode_content_filtering(self, server):
        """Test agent mode filters uninteresting content."""
        server.ai.should_store = AsyncMock(return_value=False)
        server.cache.set = AsyncMock()

        result = await server._add_clip("short", "agent")

        assert result["status"] == "skipped"
        assert "not significant" in result["reason"]
        server.cache.set.assert_not_called()

    @pytest.mark.asyncio
    async def test_search_clips_caching(self, server):
        """Test search results are properly cached."""
        # Setup mocks
        server.cache.get = AsyncMock(return_value=None)  # Cache miss first time
        server.cache.set = AsyncMock()
        server.ai.generate_embedding = AsyncMock(return_value=[0.1] * 1536)
        server.storage.search = AsyncMock(
            return_value=[
                {"id": "test1", "content": "test content", "tags": [], "score": 0.9}
            ]
        )

        query = "test query"

        # First search - should query storage
        result1 = await server._search_clips(query, 5)

        assert result1["from_cache"] is False
        server.storage.search.assert_called_once()
        server.cache.set.assert_called_once()

        # Second search - should return from cache
        server.cache.get = AsyncMock(
            return_value={
                "query": query,
                "results": [
                    {"id": "test1", "content": "test", "tags": [], "score": 0.9}
                ],
                "count": 1,
                "from_cache": False,
            }
        )

        result2 = await server._search_clips(query, 5)
        assert result2["from_cache"] is True

    @pytest.mark.asyncio
    async def test_list_clips(self, server):
        """Test listing clips functionality."""
        # Mock storage response
        mock_clips = [
            {
                "id": "1",
                "content": "First clip",
                "tags": ["test"],
                "timestamp": 1234567890,
            },
            {
                "id": "2",
                "content": "Second clip",
                "tags": ["example"],
                "timestamp": 1234567891,
            },
        ]
        server.storage.list_recent = AsyncMock(return_value=mock_clips)

        result = await server._list_clips(10)

        assert result["count"] == 2
        assert result["limit"] == 10
        assert len(result["clips"]) == 2
        server.storage.list_recent.assert_called_once_with(10)

    @pytest.mark.asyncio
    async def test_remove_clip(self, server):
        """Test clip removal functionality."""
        clip_id = "test_clip_123"

        # Test successful removal
        server.storage.remove = AsyncMock(return_value=True)
        server.cache.delete = AsyncMock()

        result = await server._remove_clip(clip_id)

        assert result["status"] == "removed"
        assert result["id"] == clip_id
        server.storage.remove.assert_called_once_with(clip_id)
        server.cache.delete.assert_called_once_with(clip_id)

        # Test clip not found
        server.storage.remove = AsyncMock(return_value=False)

        result = await server._remove_clip(clip_id)

        assert result["status"] == "not_found"
        assert "error" in result

    @pytest.mark.asyncio
    async def test_get_stats(self, server):
        """Test statistics functionality."""
        mock_stats = {
            "total_clips": 42,
            "agent_clips": 30,
            "manual_clips": 12,
            "avg_content_length": 156.7,
            "storage_path": "/test/path",
        }
        server.storage.get_stats = AsyncMock(return_value=mock_stats)

        result = await server._get_stats()

        assert result == mock_stats
        server.storage.get_stats.assert_called_once()

    def test_generate_clip_id(self, server):
        """Test clip ID generation."""
        content = "Test content for ID generation"

        clip_id1 = server._generate_clip_id(content)
        clip_id2 = server._generate_clip_id(content)

        # Should be 12 characters
        assert len(clip_id1) == 12
        assert len(clip_id2) == 12

        # Should be different due to timestamp
        assert clip_id1 != clip_id2

    def test_is_agent_source(self, server):
        """Test agent source detection."""
        # Agent sources
        assert server._is_agent_source("agent") is True
        assert server._is_agent_source("claude") is True
        assert server._is_agent_source("Claude Code") is True
        assert server._is_agent_source("AI Assistant") is True

        # Human sources
        assert server._is_agent_source("manual") is False
        assert server._is_agent_source("user") is False
        assert server._is_agent_source("") is False


@pytest.mark.asyncio
async def test_background_processing():
    """Test background processing doesn't block main thread."""
    server = SmartClipboardMCPServer()

    # Mock all dependencies
    server.cache.get = AsyncMock(
        return_value={
            "content": "Test content for background processing",
            "source": "manual",
            "is_agent": False,
            "timestamp": 1234567890,
        }
    )
    server.ai.generate_embedding = AsyncMock(return_value=[0.1] * 1536)
    server.ai.generate_tags = AsyncMock(return_value=["test", "background"])
    server.storage.store = AsyncMock(return_value=True)

    # Process should complete without blocking
    start_time = asyncio.get_event_loop().time()
    await server._process_clip_background("test_clip_123")
    end_time = asyncio.get_event_loop().time()

    # Should complete quickly
    processing_time = end_time - start_time
    assert processing_time < 0.1, f"Background processing took {processing_time:.3f}s"

    # Verify all steps were called
    server.cache.get.assert_called_once_with("test_clip_123")
    server.ai.generate_embedding.assert_called_once()
    server.storage.store.assert_called_once()


@pytest.mark.asyncio
async def test_error_handling():
    """Test error handling in various scenarios."""
    server = SmartClipboardMCPServer()

    # Test storage failure
    server.ai.should_store = AsyncMock(return_value=True)
    server.cache.set = AsyncMock(side_effect=Exception("Cache error"))

    # Should not crash on cache error
    result = await server._add_clip("Test content", "manual")
    assert "id" in result  # Should still return clip ID

    # Test search with missing embedding
    server.cache.get = AsyncMock(return_value=None)
    server.ai.generate_embedding = AsyncMock(return_value=None)

    result = await server._search_clips("test query", 5)
    assert "error" in result
