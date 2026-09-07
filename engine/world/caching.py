"""Caching interfaces for performance-critical queries.

Invariants:
  - Cache entries are immutable after creation
  - Invalidation is explicit (no automatic stale detection)
  - Cache size is bounded
  - Thread-unsafe; designed for single-threaded use
"""

from __future__ import annotations

from typing import Any, Callable, Generic, Optional, TypeVar, Dict, List
from dataclasses import dataclass
from abc import ABC, abstractmethod

T = TypeVar("T")
K = TypeVar("K")


@dataclass
class CacheEntry(Generic[T]):
    """A cache entry with invalidation tracking."""
    key: str
    value: T
    tick_created: int
    tick_accessed: int


class Cache(ABC, Generic[K, T]):
    """Abstract cache interface."""

    @abstractmethod
    def get(self, key: K) -> Optional[T]:
        """Get value from cache."""
        pass

    @abstractmethod
    def put(self, key: K, value: T) -> None:
        """Store value in cache."""
        pass

    @abstractmethod
    def invalidate(self, key: K) -> None:
        """Invalidate a cache entry."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all cache entries."""
        pass


class LRUCache(Cache[K, T]):
    """Least-Recently-Used cache with bounded size.

    When capacity is reached, least recently accessed entry is evicted.
    """

    def __init__(self, capacity: int = 1000):
        self._capacity = capacity
        self._cache: Dict[K, T] = {}
        self._access_order: List[K] = []

    def get(self, key: K) -> Optional[T]:
        """Get value from cache (updates access order)."""
        if key not in self._cache:
            return None

        # Update access order
        self._access_order.remove(key)
        self._access_order.append(key)

        return self._cache[key]

    def put(self, key: K, value: T) -> None:
        """Store value in cache."""
        if key in self._cache:
            # Update existing
            self._cache[key] = value
            self._access_order.remove(key)
            self._access_order.append(key)
        else:
            # Add new
            if len(self._cache) >= self._capacity:
                # Evict LRU
                lru_key = self._access_order.pop(0)
                del self._cache[lru_key]

            self._cache[key] = value
            self._access_order.append(key)

    def invalidate(self, key: K) -> None:
        """Remove entry from cache."""
        if key in self._cache:
            del self._cache[key]
            self._access_order.remove(key)

    def clear(self) -> None:
        """Clear all entries."""
        self._cache.clear()
        self._access_order.clear()

    def size(self) -> int:
        """Get current cache size."""
        return len(self._cache)


class QueryCache:
    """Cache for expensive query results.

    Caches results of component queries, tag lookups, etc.
    Queries are cached by a string key and must be explicitly invalidated
    when entities/components change.
    """

    def __init__(self, capacity: int = 100):
        self._cache: LRUCache[str, List[Any]] = LRUCache(capacity)
        self._tick_created: Dict[str, int] = {}

    def get_cached(
        self,
        query_key: str,
        current_tick: int,
        compute_fn: Callable[[], List[Any]],
    ) -> List[Any]:
        """Get cached query result or compute if not cached.

        Args:
            query_key: Unique key for this query
            current_tick: Current simulation tick
            compute_fn: Function to compute result if not cached

        Returns:
            Query result (cached or freshly computed)
        """
        cached = self._cache.get(query_key)
        if cached is not None:
            return cached

        result = compute_fn()
        self._cache.put(query_key, result)
        self._tick_created[query_key] = current_tick
        return result

    def invalidate_pattern(self, pattern: str) -> None:
        """Invalidate all queries matching a pattern.

        Pattern is treated as a substring match for simplicity.

        Args:
            pattern: Substring to match against query keys
        """
        # Find all keys matching pattern
        matching_keys = [
            key for key in self._tick_created.keys()
            if pattern in key
        ]
        for key in matching_keys:
            self._cache.invalidate(key)
            del self._tick_created[key]

    def invalidate_all(self) -> None:
        """Clear all cached queries."""
        self._cache.clear()
        self._tick_created.clear()

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": self._cache.size(),
            "cached_queries": len(self._tick_created),
        }


class TransformCache:
    """Cache for coordinate frame transforms.

    Transforms are expensive to compute (matrix multiplications).
    This cache stores computed transforms and invalidates them when
    entities move.
    """

    def __init__(self, capacity: int = 500):
        self._cache: LRUCache[tuple, Any] = LRUCache(capacity)

    def get_transform(
        self,
        source_frame: str,
        target_frame: str,
        compute_fn: Callable[[], Any],
    ) -> Optional[Any]:
        """Get cached transform or compute if not cached.

        Args:
            source_frame: Source coordinate frame
            target_frame: Target coordinate frame
            compute_fn: Function to compute transform if not cached

        Returns:
            Transform object (cached or freshly computed)
        """
        key = (source_frame, target_frame)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        result = compute_fn()
        if result is not None:
            self._cache.put(key, result)
        return result

    def invalidate_frame(self, frame: str) -> None:
        """Invalidate all transforms involving a frame.

        Args:
            frame: Coordinate frame that changed
        """
        # Find all transform caches involving this frame
        to_invalidate = [
            key for key in self._cache._cache.keys()
            if frame in key
        ]
        for key in to_invalidate:
            self._cache.invalidate(key)

    def clear(self) -> None:
        """Clear all transforms."""
        self._cache.clear()

    def size(self) -> int:
        """Get cache size."""
        return self._cache.size()


class ComponentCache:
    """Cache for component queries.

    Maintains cached lists of entities with specific components.
    Invalidated when components are added/removed.
    """

    def __init__(self, capacity: int = 200):
        self._cache: LRUCache[str, List[str]] = LRUCache(capacity)

    def get_entities_with_component(
        self,
        component_type: str,
        compute_fn: Callable[[], List[str]],
    ) -> List[str]:
        """Get cached entities with component or compute if not cached.

        Args:
            component_type: Type name of component
            compute_fn: Function to compute entity list if not cached

        Returns:
            List of entity IDs with the component
        """
        cached = self._cache.get(component_type)
        if cached is not None:
            return cached

        result = compute_fn()
        self._cache.put(component_type, result)
        return result

    def invalidate_component(self, component_type: str) -> None:
        """Invalidate cache for a component type."""
        self._cache.invalidate(component_type)

    def invalidate_all(self) -> None:
        """Clear all component caches."""
        self._cache.clear()

    def size(self) -> int:
        """Get cache size."""
        return self._cache.size()
