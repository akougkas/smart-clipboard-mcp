"""Fast caching layer for immediate responses."""

import json
import time
import asyncio
from typing import Any, Optional, Dict


class ClipCache:
    """High-performance cache with Redis fallback to in-memory."""

    def __init__(self):
        self.backend = None
        self._initialize_backend()

    def _initialize_backend(self):
        """Initialize cache backend with Redis fallback to memory."""
        try:
            import redis.asyncio as redis

            self.backend = RedisCache()
        except ImportError:
            print("Redis not available, using in-memory cache")
            self.backend = MemoryCache()
        except Exception as e:
            print(f"Redis connection failed: {e}, using in-memory cache")
            self.backend = MemoryCache()

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        try:
            return await self.backend.get(key)
        except Exception as e:
            print(f"Cache get error: {e}")
            return None

    async def set(self, key: str, value: Any, ttl: int = 3600):
        """Set value in cache with TTL."""
        try:
            await self.backend.set(key, value, ttl)
        except Exception as e:
            print(f"Cache set error: {e}")

    async def delete(self, key: str):
        """Delete key from cache."""
        try:
            await self.backend.delete(key)
        except Exception as e:
            print(f"Cache delete error: {e}")

    async def clear(self):
        """Clear all cache entries."""
        try:
            await self.backend.clear()
        except Exception as e:
            print(f"Cache clear error: {e}")


class RedisCache:
    """Redis-based cache implementation."""

    def __init__(self):
        import redis.asyncio as redis

        self.redis = redis.Redis(
            host="localhost",
            port=6379,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
        self._connected = None

    async def _ensure_connected(self) -> bool:
        """Ensure Redis connection is working."""
        if self._connected is not None:
            return self._connected

        try:
            await self.redis.ping()
            self._connected = True
            return True
        except Exception:
            self._connected = False
            return False

    async def get(self, key: str) -> Optional[Any]:
        """Get from Redis cache."""
        if not await self._ensure_connected():
            return None

        try:
            value = await self.redis.get(f"smart_clipboard:{key}")
            return json.loads(value) if value else None
        except Exception:
            return None

    async def set(self, key: str, value: Any, ttl: int = 3600):
        """Set in Redis cache with TTL."""
        if not await self._ensure_connected():
            return

        try:
            serialized = json.dumps(value, default=str)
            await self.redis.setex(f"smart_clipboard:{key}", ttl, serialized)
        except Exception:
            pass

    async def delete(self, key: str):
        """Delete from Redis cache."""
        if not await self._ensure_connected():
            return

        try:
            await self.redis.delete(f"smart_clipboard:{key}")
        except Exception:
            pass

    async def clear(self):
        """Clear all smart_clipboard keys."""
        if not await self._ensure_connected():
            return

        try:
            keys = await self.redis.keys("smart_clipboard:*")
            if keys:
                await self.redis.delete(*keys)
        except Exception:
            pass


class MemoryCache:
    """High-performance in-memory cache with LRU eviction."""

    def __init__(self, max_size: int = 10000):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_order: Dict[str, float] = {}
        self.max_size = max_size
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Any]:
        """Get from memory cache."""
        async with self._lock:
            if key in self.cache:
                entry = self.cache[key]

                # Check if expired
                if time.time() > entry["expires_at"]:
                    del self.cache[key]
                    del self.access_order[key]
                    return None

                # Update access time for LRU
                self.access_order[key] = time.time()
                return entry["value"]

            return None

    async def set(self, key: str, value: Any, ttl: int = 3600):
        """Set in memory cache with TTL and LRU eviction."""
        async with self._lock:
            current_time = time.time()

            # Clean expired entries if cache is getting full
            if len(self.cache) >= self.max_size * 0.8:
                await self._cleanup_expired()

            # Evict oldest entries if still too full
            if len(self.cache) >= self.max_size:
                await self._evict_lru()

            # Store new entry
            self.cache[key] = {"value": value, "expires_at": current_time + ttl}
            self.access_order[key] = current_time

    async def delete(self, key: str):
        """Delete from memory cache."""
        async with self._lock:
            if key in self.cache:
                del self.cache[key]
                del self.access_order[key]

    async def clear(self):
        """Clear all cache entries."""
        async with self._lock:
            self.cache.clear()
            self.access_order.clear()

    async def _cleanup_expired(self):
        """Remove expired entries."""
        current_time = time.time()
        expired_keys = [
            key
            for key, entry in self.cache.items()
            if current_time > entry["expires_at"]
        ]

        for key in expired_keys:
            del self.cache[key]
            if key in self.access_order:
                del self.access_order[key]

    async def _evict_lru(self):
        """Evict least recently used entries."""
        if not self.access_order:
            return

        # Sort by access time and remove oldest 20%
        sorted_keys = sorted(self.access_order.items(), key=lambda x: x[1])
        evict_count = max(1, len(sorted_keys) // 5)

        for key, _ in sorted_keys[:evict_count]:
            if key in self.cache:
                del self.cache[key]
            del self.access_order[key]

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        current_time = time.time()
        expired_count = sum(
            1 for entry in self.cache.values() if current_time > entry["expires_at"]
        )

        return {
            "total_entries": len(self.cache),
            "expired_entries": expired_count,
            "active_entries": len(self.cache) - expired_count,
            "max_size": self.max_size,
            "usage_percent": round((len(self.cache) / self.max_size) * 100, 1),
        }
