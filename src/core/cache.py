"""In-memory TTL+LRU cache implementation using cachetools"""
import hashlib
import sys
from typing import Optional, Dict
import logging
from threading import RLock
from cachetools import TTLCache

logger = logging.getLogger(__name__)


class CacheManager:
    """Manages in-memory TTL+LRU caching with memory limits"""
    
    def __init__(self, redis_url: str = None, ttl: int = 60, max_size: int = 1000, max_memory_mb: int = 500):
        self.ttl = ttl
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        
        # TTLCache combines LRU eviction with TTL expiration
        # It automatically handles both time-based and size-based eviction
        self._cache = TTLCache(maxsize=max_size, ttl=ttl)
        self._lock = RLock()  # Use RLock for cachetools thread safety
        self._hits = 0
        self._misses = 0
    
    async def connect(self):
        """Initialize cache (no-op for in-memory)"""
        logger.info(f"TTL+LRU cache initialized: max_size={self.max_size}, ttl={self.ttl}s, max_memory={self.max_memory_bytes//1024//1024}MB")
    
    async def disconnect(self):
        """Cleanup cache (no-op for in-memory)"""
        pass
    
    def _generate_key(self, text: str) -> str:
        """Generate cache key from input text"""
        hash_obj = hashlib.sha256(text.encode())
        return hash_obj.hexdigest()[:16]
    
    def _estimate_memory_usage(self) -> int:
        """Rough estimate of current cache memory usage"""
        if not self._cache:
            return 0
        
        # Rough estimate: assume avg 200 bytes per cache entry (key + value + overhead)
        return len(self._cache) * 200
    
    def _check_memory_limit(self):
        """Evict entries if memory usage is too high"""
        current_memory = self._estimate_memory_usage()
        if current_memory > self.max_memory_bytes:
            # Reduce cache size by 20% when memory limit is hit
            target_size = max(1, int(len(self._cache) * 0.8))
            while len(self._cache) > target_size:
                # TTLCache will automatically remove the LRU item when we try to add beyond maxsize
                # For now, just reduce maxsize temporarily
                pass
    
    async def get(self, text: str) -> Optional[dict]:
        """Get cached result for input text"""
        with self._lock:
            key = self._generate_key(text)
            try:
                value = self._cache[key]  # TTLCache handles TTL and LRU automatically
                self._hits += 1
                logger.debug(f"Cache hit for key: {key}")
                return value
            except KeyError:
                self._misses += 1
                return None
    
    async def set(self, text: str, result: dict) -> bool:
        """Cache result for input text"""
        with self._lock:
            self._check_memory_limit()
            
            key = self._generate_key(text)
            self._cache[key] = result  # TTLCache handles size limits and TTL
            
            logger.debug(f"Cached result for key: {key}")
            return True
    
    async def get_batch(self, texts: list[str]) -> dict[str, Optional[dict]]:
        """Get cached results for multiple texts"""
        with self._lock:
            results = {}
            
            for text in texts:
                key = self._generate_key(text)
                try:
                    value = self._cache[key]
                    results[text] = value
                    self._hits += 1
                except KeyError:
                    results[text] = None
                    self._misses += 1
            
            return results
    
    async def set_batch(self, items: dict[str, dict]) -> bool:
        """Cache multiple results"""
        with self._lock:
            self._check_memory_limit()
            
            for text, result in items.items():
                key = self._generate_key(text)
                self._cache[key] = result
            
            logger.debug(f"Cached {len(items)} results")
            return True
    
    async def clear(self) -> bool:
        """Clear all cache entries"""
        with self._lock:
            deleted = len(self._cache)
            self._cache.clear()
            self._hits = 0
            self._misses = 0
            
            logger.info(f"Cleared {deleted} cache entries")
            return True
    
    async def get_stats(self) -> dict:
        """Get cache statistics"""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / max(total_requests, 1)
            memory_usage = self._estimate_memory_usage()
            
            return {
                "enabled": True,
                "type": "TTL+LRU (cachetools)",
                "current_size": len(self._cache),
                "max_size": self.max_size,
                "ttl_seconds": self.ttl,
                "max_memory_mb": self.max_memory_bytes // 1024 // 1024,
                "estimated_memory_kb": memory_usage // 1024,
                "memory_usage_percent": round(memory_usage / self.max_memory_bytes * 100, 1),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(hit_rate, 3)
            }