"""Simple tests for cache functionality."""

import pytest
import asyncio
from unittest.mock import patch, Mock

from smart_clipboard.core.cache import ClipCache, MemoryCache


class TestMemoryCache:
    """Test in-memory cache implementation."""

    @pytest.fixture
    def cache(self):
        """Create memory cache for testing."""
        return MemoryCache(max_size=100)

    @pytest.mark.asyncio
    async def test_set_and_get(self, cache):
        """Test basic set and get operations."""
        await cache.set("test_key", {"data": "test_value"})
        result = await cache.get("test_key")

        assert result == {"data": "test_value"}

    @pytest.mark.asyncio
    async def test_ttl_expiration(self, cache):
        """Test TTL expiration."""
        await cache.set("expire_key", "value", ttl=1)

        # Should be available immediately
        result = await cache.get("expire_key")
        assert result == "value"

        # Wait for expiration
        await asyncio.sleep(1.1)

        # Should be None after expiration
        result = await cache.get("expire_key")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete(self, cache):
        """Test cache deletion."""
        await cache.set("delete_key", "value")
        assert await cache.get("delete_key") == "value"

        await cache.delete("delete_key")
        assert await cache.get("delete_key") is None

    @pytest.mark.asyncio
    async def test_clear(self, cache):
        """Test cache clearing."""
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")

        await cache.clear()

        assert await cache.get("key1") is None
        assert await cache.get("key2") is None

    def test_stats(self, cache):
        """Test cache statistics."""
        stats = cache.get_stats()

        assert "total_entries" in stats
        assert "max_size" in stats
        assert "usage_percent" in stats


class TestClipCache:
    """Test main cache interface."""

    @pytest.mark.asyncio
    async def test_memory_cache_fallback(self):
        """Test fallback to memory cache when Redis unavailable."""
        with patch("redis.asyncio.Redis", side_effect=ImportError):
            cache = ClipCache()

            await cache.set("test", "value")
            result = await cache.get("test")

            assert result == "value"

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test cache error handling."""
        cache = ClipCache()

        # Mock backend to raise errors
        cache.backend.get = Mock(side_effect=Exception("Cache error"))
        cache.backend.set = Mock(side_effect=Exception("Cache error"))

        # Should not crash on errors
        result = await cache.get("test_key")
        assert result is None

        await cache.set("test_key", "value")  # Should not crash
