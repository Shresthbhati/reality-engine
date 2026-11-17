"""Spatial relations between entities (BUILD_SPEC B15).

Pairs come from a uniform grid (0.5 m cells) over entity boxes, never all N^2. Relations are derived from boxes only:
``on``, ``attached_to`` (a wall), ``inside`` (container), ``adjacent_to``. ``contains`` is only the inverse of an
OBSERVED ``inside``. A container with no interior observation is ``not_observed``, never empty.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import product
from typing import Iterable, Optional

import numpy as np

CELL_M = 0.5
ON_TOL_M = 0.03
ATTACH_TOL_M = 0.03
ADJ_GAP_M = 0.10
ON_OVERLAP = 0.50
INSIDE_SHARE = 0.90

RELATION_KINDS = ("part_of", "attached_to", "on", "inside", "contains", "adjacent_to", "connects", "hosted_by", "supports")


@dataclass(frozen=True)
class Box:
    id: str
    lo: tuple[float, float, float]
    hi: tuple[float, float, float]
    is_container: bool = False
    is_wall: bool = False
    interior_observed: bool = False


def _grid(boxes: list[Box]) -> dict:
    g: dict = defaultdict(list)
    for i, b in enumerate(boxes):
        lo = np.floor(np.array(b.lo) / CELL_M).astype(int); hi = np.floor(np.array(b.hi) / CELL_M).astype(int)
        for c in product(*(range(int(a), int(z) + 1) for a, z in zip(lo, hi))):
            g[c].append(i)
    return g


def _overlap_xy(a: Box, b: Box) -> float:
    ox = min(a.hi[0], b.hi[0]) - max(a.lo[0], b.lo[0]); oy = min(a.hi[1], b.hi[1]) - max(a.lo[1], b.lo[1])
    if ox <= 0 or oy <= 0:
        return 0.0
    return ox * oy / max((a.hi[0] - a.lo[0]) * (a.hi[1] - a.lo[1]), 1e-12)


def _gap(a: Box, b: Box) -> float:
    d = np.maximum(np.array(a.lo) - np.array(b.hi), np.array(b.lo) - np.array(a.hi))
    return float(np.linalg.norm(np.maximum(d, 0.0)))


def _inside_share(a: Box, b: Box) -> float:
    lo = np.maximum(a.lo, b.lo); hi = np.minimum(a.hi, b.hi)
    inter = float(np.prod(np.maximum(hi - lo, 0.0))); vol = float(np.prod(np.array(a.hi) - np.array(a.lo)))
    return inter / vol if vol > 0 else 0.0


def derive_relations(boxes: Iterable[Box]) -> list[tuple[str, str, str]]:
    """``(a, kind, b)`` triples; ``z`` is up."""
    bx = list(boxes)
    pairs: set[tuple[int, int]] = set()
    for members in _grid(bx).values():
        for i in members:
            for j in members:
                if i < j:
                    pairs.add((i, j))
    out: list[tuple[str, str, str]] = []
    for i, j in sorted(pairs):
        found = False
        for a, b in ((bx[i], bx[j]), (bx[j], bx[i])):
            if b.is_container and not a.is_wall and _inside_share(a, b) >= INSIDE_SHARE:
                out.append((a.id, "inside", b.id)); found = True
                if b.interior_observed:
                    out.append((b.id, "contains", a.id))
            elif b.is_wall and not a.is_wall and _gap(a, b) <= ATTACH_TOL_M:
                out.append((a.id, "attached_to", b.id)); found = True
            elif not a.is_wall and abs(a.lo[2] - b.hi[2]) <= ON_TOL_M and _overlap_xy(a, b) >= ON_OVERLAP:
                out.append((a.id, "on", b.id)); found = True
        if not found and _gap(bx[i], bx[j]) < ADJ_GAP_M:
            out.append((bx[i].id, "adjacent_to", bx[j].id))
    return out


def interior_state(box: Box) -> Optional[str]:
    """``"not_observed"`` for a container whose inside was never seen; None otherwise."""
    return "not_observed" if box.is_container and not box.interior_observed else None
