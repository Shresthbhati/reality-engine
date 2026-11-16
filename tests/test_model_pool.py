"""B13: LRU eviction under a VRAM budget, loads once, never an OOM crash."""
import pytest

from engine.hardware.model_pool import ModelPool, ResourceUnavailable


class _M:
    def __init__(self, n): self.n, self.unloaded = n, False
    def unload(self): self.unloaded = True


def test_third_model_evicts_least_recently_used():
    p = ModelPool(budget_mb=100)
    made = {}
    def load(n): made[n] = _M(n); return made[n]
    with p.acquire("a", load, 50): pass
    with p.acquire("b", load, 50): pass
    with p.acquire("a", load, 50): pass          # touch a -> b is now LRU
    with p.acquire("c", load, 50): pass
    assert p.loaded() == ["a", "c"] and made["b"].unloaded and not made["a"].unloaded
    assert p.loads == {"a": 1, "b": 1, "c": 1}


def test_model_larger_than_budget_is_resource_unavailable():
    with pytest.raises(ResourceUnavailable):
        with ModelPool(budget_mb=100).acquire("big", lambda n: _M(n), 500): pass


def test_unknown_vram_is_refused_not_guessed():
    with pytest.raises(ResourceUnavailable):
        with ModelPool().acquire("mystery", lambda n: _M(n)): pass
