"""Tests for dependency graph and caching systems."""

import pytest
from engine.world import (
    DependencyGraph,
    LRUCache,
    QueryCache,
    TransformCache,
    ComponentCache,
)


class TestDependencyGraph:
    """Dependency graph tests."""

    def test_register_system(self):
        """Test registering a system."""
        graph = DependencyGraph()
        calls = []
        graph.register_system("physics", lambda: calls.append("physics"))
        assert "physics" in graph.all_systems()

    def test_duplicate_system_raises(self):
        """Test that duplicate system name raises error."""
        graph = DependencyGraph()
        graph.register_system("physics", lambda: None)
        with pytest.raises(ValueError, match="already registered"):
            graph.register_system("physics", lambda: None)

    def test_simple_dependency(self):
        """Test simple dependency ordering."""
        graph = DependencyGraph()
        calls = []
        graph.register_system("broad", lambda: calls.append("broad"))
        graph.register_system("narrow", lambda: calls.append("narrow"), depends_on={"broad"})

        order = graph.get_execution_order()
        assert order == ["broad", "narrow"]

    def test_multiple_dependencies(self):
        """Test system with multiple dependencies."""
        graph = DependencyGraph()
        calls = []
        graph.register_system("a", lambda: calls.append("a"))
        graph.register_system("b", lambda: calls.append("b"))
        graph.register_system("c", lambda: calls.append("c"), depends_on={"a", "b"})

        order = graph.get_execution_order()
        assert "a" in order[:2] and "b" in order[:2]
        assert order[-1] == "c"

    def test_cycle_detection(self):
        """Test that cycles are detected."""
        graph = DependencyGraph()
        graph.register_system("a", lambda: None)
        graph.register_system("b", lambda: None, depends_on={"a"})

        with pytest.raises(ValueError, match="would create a cycle"):
            graph.update_dependencies("a", {"b"})

    def test_deterministic_order(self):
        """Test that execution order is deterministic."""
        graph1 = DependencyGraph()
        graph2 = DependencyGraph()

        for graph in [graph1, graph2]:
            graph.register_system("z", lambda: None)
            graph.register_system("a", lambda: None)
            graph.register_system("m", lambda: None)

        order1 = graph1.get_execution_order()
        order2 = graph2.get_execution_order()
        assert order1 == order2

    def test_execute_all(self):
        """Test executing all systems in order."""
        graph = DependencyGraph()
        calls = []
        graph.register_system("a", lambda: calls.append("a"))
        graph.register_system("b", lambda: calls.append("b"), depends_on={"a"})
        graph.register_system("c", lambda: calls.append("c"), depends_on={"b"})

        graph.execute_all()
        assert calls == ["a", "b", "c"]

    def test_execute_system(self):
        """Test executing a single system."""
        graph = DependencyGraph()
        calls = []
        graph.register_system("physics", lambda: calls.append("physics"))
        graph.execute_system("physics")
        assert calls == ["physics"]

    def test_unregister_system(self):
        """Test unregistering a system."""
        graph = DependencyGraph()
        graph.register_system("physics", lambda: None)
        graph.unregister_system("physics")
        assert "physics" not in graph.all_systems()

    def test_unregister_depended_on_raises(self):
        """Test that unregistering a depended-on system raises error."""
        graph = DependencyGraph()
        graph.register_system("physics", lambda: None)
        graph.register_system("rendering", lambda: None, depends_on={"physics"})

        with pytest.raises(ValueError, match="depend on it"):
            graph.unregister_system("physics")

    def test_tier_ordering(self):
        """Test that systems execute by tier."""
        graph = DependencyGraph()
        calls = []
        graph.register_system("physics", lambda: calls.append("physics"), tier=1)
        graph.register_system("broad", lambda: calls.append("broad"), tier=0)

        order = graph.get_execution_order()
        assert order.index("broad") < order.index("physics")

    def test_execution_error_handling(self):
        """Test that execution errors are caught."""
        graph = DependencyGraph()
        def failing():
            raise ValueError("test error")
        graph.register_system("fail", failing)

        with pytest.raises(RuntimeError, match="execution failed"):
            graph.execute_all()


class TestLRUCache:
    """LRU cache tests."""

    def test_put_and_get(self):
        """Test basic put/get operations."""
        cache = LRUCache[str, int](capacity=10)
        cache.put("key1", 42)
        assert cache.get("key1") == 42

    def test_missing_key_returns_none(self):
        """Test that missing keys return None."""
        cache = LRUCache[str, int](capacity=10)
        assert cache.get("missing") is None

    def test_lru_eviction(self):
        """Test that LRU item is evicted when capacity is reached."""
        cache = LRUCache[str, int](capacity=3)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)

        # Access 'a' to make it recently used
        cache.get("a")

        # Add new item, 'b' should be evicted (LRU)
        cache.put("d", 4)

        assert cache.get("b") is None
        assert cache.get("a") == 1
        assert cache.get("d") == 4

    def test_update_existing(self):
        """Test updating an existing key."""
        cache = LRUCache[str, int](capacity=10)
        cache.put("key", 1)
        cache.put("key", 2)
        assert cache.get("key") == 2

    def test_invalidate(self):
        """Test invalidating a cache entry."""
        cache = LRUCache[str, int](capacity=10)
        cache.put("key1", 42)
        cache.invalidate("key1")
        assert cache.get("key1") is None

    def test_clear(self):
        """Test clearing the cache."""
        cache = LRUCache[str, int](capacity=10)
        cache.put("key1", 1)
        cache.put("key2", 2)
        cache.clear()
        assert cache.size() == 0


class TestQueryCache:
    """Query cache tests."""

    def test_cache_hit(self):
        """Test that cached queries are reused."""
        cache = QueryCache(capacity=10)
        calls = [0]

        def compute():
            calls[0] += 1
            return ["entity1", "entity2"]

        result1 = cache.get_cached("query1", current_tick=0, compute_fn=compute)
        result2 = cache.get_cached("query1", current_tick=1, compute_fn=compute)

        assert result1 == result2
        assert calls[0] == 1  # Compute only called once

    def test_cache_miss(self):
        """Test that cache miss calls compute."""
        cache = QueryCache(capacity=10)
        calls = [0]

        def compute():
            calls[0] += 1
            return ["entity1"]

        cache.get_cached("query1", current_tick=0, compute_fn=compute)
        cache.get_cached("query2", current_tick=0, compute_fn=compute)

        assert calls[0] == 2

    def test_invalidate_pattern(self):
        """Test invalidating by pattern."""
        cache = QueryCache(capacity=10)
        cache.get_cached("physics_entities", current_tick=0, compute_fn=lambda: [])
        cache.get_cached("visual_entities", current_tick=0, compute_fn=lambda: [])

        # Invalidate all "physics_*" queries
        cache.invalidate_pattern("physics")

        # Calling again should recompute
        calls = [0]
        def compute():
            calls[0] += 1
            return []

        cache.get_cached("physics_entities", current_tick=1, compute_fn=compute)
        assert calls[0] == 1  # Had to recompute

    def test_invalidate_all(self):
        """Test clearing all caches."""
        cache = QueryCache(capacity=10)
        cache.get_cached("query1", current_tick=0, compute_fn=lambda: [1])
        cache.get_cached("query2", current_tick=0, compute_fn=lambda: [2])

        cache.invalidate_all()

        calls = [0]
        def compute():
            calls[0] += 1
            return []

        cache.get_cached("query1", current_tick=1, compute_fn=compute)
        assert calls[0] == 1  # Had to recompute

    def test_stats(self):
        """Test getting cache statistics."""
        cache = QueryCache(capacity=10)
        cache.get_cached("query1", current_tick=0, compute_fn=lambda: [])
        cache.get_cached("query2", current_tick=0, compute_fn=lambda: [])

        stats = cache.stats()
        assert stats["size"] == 2
        assert stats["cached_queries"] == 2


class TestTransformCache:
    """Transform cache tests."""

    def test_get_transform(self):
        """Test getting/caching transforms."""
        cache = TransformCache(capacity=50)
        calls = [0]

        def compute_transform():
            calls[0] += 1
            return {"matrix": [[1, 0, 0, 0]]}

        result1 = cache.get_transform("WORLD", "LOCAL", compute_transform)
        result2 = cache.get_transform("WORLD", "LOCAL", compute_transform)

        assert result1 == result2
        assert calls[0] == 1  # Computed only once

    def test_invalidate_frame(self):
        """Test invalidating frames."""
        cache = TransformCache(capacity=50)

        calls = [0]
        def compute():
            calls[0] += 1
            return {"matrix": [[1, 0, 0, 0]]}

        # Cache some transforms
        cache.get_transform("WORLD", "LOCAL", compute)
        cache.get_transform("WORLD", "CAMERA", compute)

        assert calls[0] == 2

        # Invalidate WORLD frame
        cache.invalidate_frame("WORLD")

        # Should recompute
        cache.get_transform("WORLD", "LOCAL", compute)
        assert calls[0] == 3

    def test_none_result_not_cached(self):
        """Test that None results are not cached."""
        cache = TransformCache(capacity=50)
        calls = [0]

        def compute():
            calls[0] += 1
            return None

        result1 = cache.get_transform("A", "B", compute)
        result2 = cache.get_transform("A", "B", compute)

        assert result1 is None
        assert result2 is None
        assert calls[0] == 2  # Recomputed each time


class TestComponentCache:
    """Component cache tests."""

    def test_get_entities_with_component(self):
        """Test caching entity queries."""
        cache = ComponentCache(capacity=50)
        calls = [0]

        def compute():
            calls[0] += 1
            return ["entity1", "entity2"]

        result1 = cache.get_entities_with_component("Physics", compute)
        result2 = cache.get_entities_with_component("Physics", compute)

        assert result1 == result2
        assert calls[0] == 1

    def test_invalidate_component(self):
        """Test invalidating component caches."""
        cache = ComponentCache(capacity=50)
        calls = [0]

        def compute():
            calls[0] += 1
            return ["entity1"]

        cache.get_entities_with_component("Physics", compute)
        assert calls[0] == 1

        # Invalidate and recompute
        cache.invalidate_component("Physics")
        cache.get_entities_with_component("Physics", compute)
        assert calls[0] == 2

    def test_invalidate_all(self):
        """Test clearing all component caches."""
        cache = ComponentCache(capacity=50)

        calls = [0]
        def compute():
            calls[0] += 1
            return []

        cache.get_entities_with_component("Physics", compute)
        cache.get_entities_with_component("Visual", compute)
        assert calls[0] == 2

        cache.invalidate_all()
        cache.get_entities_with_component("Physics", compute)
        cache.get_entities_with_component("Visual", compute)
        assert calls[0] == 4

    def test_size(self):
        """Test getting cache size."""
        cache = ComponentCache(capacity=50)

        cache.get_entities_with_component("Physics", lambda: [])
        assert cache.size() == 1

        cache.get_entities_with_component("Visual", lambda: [])
        assert cache.size() == 2
