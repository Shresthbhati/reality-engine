"""Hierarchical spatial representation: city > district > block > building > storey > room > element > part.

Each node owns a LOCAL frame in the ``FrameGraph`` and is positioned in its parent's frame by a recorded transform. Coordinates are
stored in the nearest local frame and only composed to a common frame on demand, so a chair at (2.0, 1.0, 0.0) in a room frame keeps
millimetre precision even when that room sits 5 km from the city origin. ``local_precision_m`` measures the float32 spacing that would
be lost by storing the same point in a city-origin frame.

What this module does NOT do: it does not claim any real building or city was reconstructed. ``readiness`` reports which levels are
populated and always states that no real-data validation has run here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from engine.multiengine.alignment import FrameGraph, Sim3

LEVELS = ("city", "district", "block", "building", "storey", "room", "element", "part")
_RANK = {name: i for i, name in enumerate(LEVELS)}


class HierarchyError(ValueError):
    pass


@dataclass(frozen=True)
class SpatialNode:
    node_id: str
    level: str
    frame_id: str
    parent_id: str | None


class SpatialHierarchy:
    def __init__(self, graph: FrameGraph) -> None:
        self.graph = graph
        self._nodes: dict[str, SpatialNode] = {}

    def add(
        self, node_id: str, level: str, frame_id: str, *, parent_id: str | None = None,
        to_parent: Sim3 | None = None, source: str = "",
    ) -> SpatialNode:
        """Add a node. A child needs its parent, a strictly lower level (levels may be skipped, never inverted), and the recorded
        transform (``to_parent``, with ``source``) from its local frame into the parent's frame."""
        if level not in _RANK:
            raise HierarchyError(f"unknown level {level!r}; expected one of {LEVELS}")
        if node_id in self._nodes:
            raise HierarchyError(f"node {node_id!r} already exists")
        self.graph.frame(frame_id)  # raises if the local frame was not declared
        if parent_id is None:
            if to_parent is not None:
                raise HierarchyError("a root node has no parent transform")
        else:
            parent = self._nodes.get(parent_id)
            if parent is None:
                raise HierarchyError(f"unknown parent {parent_id!r}")
            if _RANK[level] <= _RANK[parent.level]:
                raise HierarchyError(f"a {level} cannot sit inside a {parent.level}")
            if to_parent is None or not source:
                raise HierarchyError("a child node needs its transform into the parent frame and where that transform came from")
            self.graph.add_transform(frame_id, parent.frame_id, to_parent, source=source, scale_anchored=True)
        node = SpatialNode(node_id, level, frame_id, parent_id)
        self._nodes[node_id] = node
        return node

    def node(self, node_id: str) -> SpatialNode:
        try:
            return self._nodes[node_id]
        except KeyError:
            raise HierarchyError(f"unknown node {node_id!r}") from None

    def ancestors(self, node_id: str) -> list[SpatialNode]:
        out, cur = [], self.node(node_id)
        while cur.parent_id is not None:
            cur = self._nodes[cur.parent_id]
            out.append(cur)
        return out

    def root_of(self, node_id: str) -> SpatialNode:
        anc = self.ancestors(node_id)
        return anc[-1] if anc else self.node(node_id)

    def children(self, node_id: str) -> list[SpatialNode]:
        return sorted((n for n in self._nodes.values() if n.parent_id == node_id), key=lambda n: n.node_id)

    def to_root(self, points: Any, node_id: str) -> np.ndarray:
        n = self.node(node_id)
        return self.graph.transform_points(np.asarray(points, dtype=float), n.frame_id, self.root_of(node_id).frame_id)

    def convert(self, points: Any, src: str, dst: str) -> np.ndarray:
        return self.graph.transform_points(np.asarray(points, dtype=float), self.node(src).frame_id, self.node(dst).frame_id)

    def validate(self) -> list[str]:
        """Every structural problem: a node with no metric path to its parent frame (a graph can be edited after ``add``)."""
        problems = []
        for n in self._nodes.values():
            if n.parent_id is not None:
                ok, why = self.graph.comparable(n.frame_id, self._nodes[n.parent_id].frame_id, metric=True)
                if not ok:
                    problems.append(f"{n.node_id}: {why}")
        return problems

    def readiness(self) -> dict[str, Any]:
        present = sorted({n.level for n in self._nodes.values()}, key=_RANK.__getitem__)
        missing = [lv for lv in LEVELS if lv not in present]
        return {
            "levels_present": present,
            "levels_missing": missing,
            "room_to_building_chain_tested_structurally": all(lv in present for lv in ("building", "storey", "room")),
            "city_scale_claimable": False,
            "why": "structural tests only: no real building or city dataset has been reconstructed or validated with this hierarchy",
            "structural_problems": self.validate(),
        }


def local_precision_m(offset_m: float, dtype: Any = np.float32) -> float:
    """Spacing between representable values at ``offset_m`` from the frame origin: the best coordinate precision a frame whose
    origin is that far away can offer in ``dtype``."""
    return float(np.spacing(dtype(abs(offset_m))))
