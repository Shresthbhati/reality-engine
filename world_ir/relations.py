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


def _overlap_horiz(a: Box, b: Box, axes: tuple[int, int] = (0, 1)) -> float:
    ax0, ax1 = axes
    ox = min(a.hi[ax0], b.hi[ax0]) - max(a.lo[ax0], b.lo[ax0])
    oy = min(a.hi[ax1], b.hi[ax1]) - max(a.lo[ax1], b.lo[ax1])
    if ox <= 0 or oy <= 0:
        return 0.0
    return ox * oy / max((a.hi[ax0] - a.lo[ax0]) * (a.hi[ax1] - a.lo[ax1]), 1e-12)


def _gap(a: Box, b: Box) -> float:
    d = np.maximum(np.array(a.lo) - np.array(b.hi), np.array(b.lo) - np.array(a.hi))
    return float(np.linalg.norm(np.maximum(d, 0.0)))


def _inside_share(a: Box, b: Box) -> float:
    lo = np.maximum(a.lo, b.lo); hi = np.minimum(a.hi, b.hi)
    inter = float(np.prod(np.maximum(hi - lo, 0.0))); vol = float(np.prod(np.array(a.hi) - np.array(a.lo)))
    return inter / vol if vol > 0 else 0.0


def derive_relations(boxes: Iterable[Box], *, up_axis: int = 2) -> list[tuple[str, str, str]]:
    """``(a, kind, b)`` triples; ``up_axis`` is ``1`` for ``Frame.Y_UP`` or ``2`` for Z-up."""
    if up_axis not in (1, 2):
        raise ValueError(f"up_axis must be 1 (Y-up) or 2 (Z-up), got {up_axis!r}")
    horiz_axes = (0, 2) if up_axis == 1 else (0, 1)
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
            elif not a.is_wall and abs(a.lo[up_axis] - b.hi[up_axis]) <= ON_TOL_M and _overlap_horiz(a, b, horiz_axes) >= ON_OVERLAP:
                out.append((a.id, "on", b.id)); found = True
        if not found and _gap(bx[i], bx[j]) < ADJ_GAP_M:
            out.append((bx[i].id, "adjacent_to", bx[j].id))
    return out


def validate_scene_graph_consistency(
    boxes: Iterable[Box],
    relations: Iterable[tuple[str, str, str]],
    *,
    max_attach_gap_m: float = 0.25,
) -> list[dict[str, str]]:
    """Verify that derived/asserted spatial relationships agree with 3D box geometry.

    Returns a list of consistency violations (empty when consistent). Checks:
    - ``attached_to`` / ``hosted_by`` targets exist and are within ``max_attach_gap_m``
    - ``part_of`` has no contradictory multi-parent assignments
    """
    by_id = {b.id: b for b in boxes}
    violations: list[dict[str, str]] = []
    parents: dict[str, set[str]] = defaultdict(set)
    for subj, kind, obj in relations:
        if subj not in by_id or obj not in by_id:
            violations.append({
                "subject": subj, "kind": kind, "object": obj,
                "reason": "dangling entity reference in relationship",
            })
            continue
        if kind in ("attached_to", "hosted_by"):
            gap = _gap(by_id[subj], by_id[obj])
            if gap > max_attach_gap_m:
                violations.append({
                    "subject": subj, "kind": kind, "object": obj,
                    "reason": f"3D gap {gap:.3f}m exceeds max_attach_gap_m {max_attach_gap_m:.3f}m",
                })
        if kind == "part_of":
            parents[subj].add(obj)
            if len(parents[subj]) > 1:
                violations.append({
                    "subject": subj, "kind": kind, "object": obj,
                    "reason": f"contradictory parent_of targets: {sorted(parents[subj])}",
                })
    return violations


def interior_state(box: Box) -> Optional[str]:
    """``"not_observed"`` for a container whose inside was never seen; None otherwise."""
    return "not_observed" if box.is_container and not box.interior_observed else None
