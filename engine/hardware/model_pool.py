"""LRU model pool under a VRAM budget (BUILD_SPEC B13).

``acquire(name, loader, vram_mb)`` yields a loaded model, evicting least-recently-used ones until it fits.
If it cannot fit even alone it raises ``ResourceUnavailable`` (callers mark the stage ``skipped``) -- never an OOM crash.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Optional

#: measured on the 6 GB RTX 4050 (BUILD_SPEC B13)
MEASURED_VRAM_MB = {"grounding_dino_tiny_fp16": 1700.0, "dpt_hybrid": 680.0, "depth_anything_v2_small_fp16": 420.0, "mask_rcnn": 1550.0}


class ResourceUnavailable(RuntimeError):
    pass


class ModelPool:
    def __init__(self, budget_mb: float = 5000.0) -> None:   # 6 GB minus 1 GB headroom
        self.budget_mb = float(budget_mb)
        self._m: "OrderedDict[str, tuple[Any, float]]" = OrderedDict()
        self.loads: dict[str, int] = {}

    @property
    def used_mb(self) -> float:
        return sum(v for _, v in self._m.values())

    def loaded(self) -> list[str]:
        return list(self._m)

    def _evict_one(self) -> None:
        _, (model, _) = self._m.popitem(last=False)
        unload = getattr(model, "unload", None)
        if callable(unload):
            unload()

    @contextmanager
    def acquire(self, name: str, loader: Callable[[str], Any], vram_mb: Optional[float] = None) -> Iterator[Any]:
        if name in self._m:
            self._m.move_to_end(name)
            yield self._m[name][0]
            return
        need = vram_mb if vram_mb is not None else MEASURED_VRAM_MB.get(name)
        if need is None:
            raise ResourceUnavailable(f"no measured VRAM for {name!r}; pass vram_mb")
        need = float(need)
        if need > self.budget_mb:
            raise ResourceUnavailable(f"{name} needs {need:.0f} MB, whole budget is {self.budget_mb:.0f} MB")
        while self.used_mb + need > self.budget_mb and self._m:
            self._evict_one()
        model = loader(name)
        self.loads[name] = self.loads.get(name, 0) + 1
        self._m[name] = (model, need)
        yield model
