"""Evidence-backed spatial relations between positioned canonical entities.

Only relations that follow from the resolved axis-aligned boxes alone are derived:

* ``OVERLAPS``   -- the boxes interpenetrate on all three axes by more than the tolerance;
* ``ADJACENT_TO`` -- the boxes overlap on two axes and are separated (or touching) by at most the tolerance on the third.

"On top of" / "supported by" are NOT derived: they need a known vertical axis, and an SfM-gauge frame has none. The tolerance is a
fraction of the smaller box's smallest extent (a stated starting value, not tuned) so it is unit-free, and two entities are compared only
when their positions carry the SAME scale state (values in different scales are never compared). Each relation records its basis.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from engine.multiengine.canonical import POSITION, CanonicalEntity
from engine.multiengine.conflicts import box_from_entity
from engine.multiengine.worldir_bridge import SpatialRelation, scale_of
from world_ir.schema_v1 import RelationshipKind

REL_TOLERANCE = 0.1  # fraction of the smaller box's smallest extent; untuned starting value


def derive_relations(canonicals: Sequence[CanonicalEntity], matrix: Any, *, rel_tolerance: float = REL_TOLERANCE) -> list[SpatialRelation]:
    boxes = []
    for ce in canonicals:
        b = box_from_entity(ce)
        pos = ce.determinations.get(POSITION)
        if b is not None and pos is not None:
            boxes.append((b, scale_of(matrix, pos.supporting).get("scale_state"), tuple(pos.supporting)))
    out: list[SpatialRelation] = []
    for i, (a, sa, oa) in enumerate(boxes):
        for b, sb, ob in boxes[i + 1:]:
            if sa is None or sa != sb:  # undeclared or different scale: not comparable
                continue
            tol = rel_tolerance * min(min(h - l for l, h in zip(x.lo, x.hi, strict=True)) for x in (a, b))
            ov = [min(a.hi[k], b.hi[k]) - max(a.lo[k], b.lo[k]) for k in range(3)]  # >0 overlap, <0 gap
            if all(o > tol for o in ov):
                kind, why = RelationshipKind.OVERLAPS, f"boxes overlap by {min(ov):.4g} (all axes) > tolerance {tol:.4g}"
            elif sum(o > 0 for o in ov) == 2 and min(ov) >= -tol:
                kind, why = RelationshipKind.ADJACENT_TO, f"boxes overlap on two axes and are separated by {max(0.0, -min(ov)):.4g} <= tolerance {tol:.4g} on the third"
            else:
                continue
            basis = f"{why}; axis-aligned boxes from resolved position+dimensions in scale state {sa!r}; tolerance = {rel_tolerance} x smaller box's smallest extent"
            obs = tuple(sorted(set(oa) | set(ob)))
            out.append(SpatialRelation(a.entity_id, kind, b.entity_id, basis, obs))
            out.append(SpatialRelation(b.entity_id, kind, a.entity_id, basis, obs))
    return out
