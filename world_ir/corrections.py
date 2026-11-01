"""User corrections as VERIFIED evidence (plan F6).

A correction is a human stating a fact the pipeline could not measure: a real wall length, a real room height, a real
area. It is stronger than any automatic result and must survive a rebuild, but it must never be confused with a
measurement the pipeline made.

Storage is split by lifetime:

* the entity carries the value in ``custom_properties[property]`` and the D1 shape in ``custom_properties["dimensions"]``
  with ``state="VERIFIED"``, ``basis="user-entered (…)"`` and a ``user_correction`` stamp;
* the world carries a replayable record in ``metadata["user_corrections"]``: kind, entity type, property, value, the
  entity's measured anchor (plan centroid / slab midpoint) and a note.

On a rebuild the compiler produces new entities, so ``carry_corrections`` replays each record onto the entity with the
same id, or -- when ids changed -- the nearest entity of the same type to the stored anchor. A record that has no
matching surface is skipped with a reason; nothing is guessed onto the wrong wall.

Only a METRIC world can carry a VERIFIED dimension. In a RELATIVE world the numbers are model units, so a typed metre
value is a scale reference (D2 ``known_length``), not a correction -- the caller must anchor scale first.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence

from world_ir.measure import DIMENSION_STATES, dimension_from_metadata, scale_state_of

#: properties a user may state; anything else is refused (no silent invention of new shape)
CORRECTABLE_PROPERTIES = frozenset({
    "length_m", "width_m", "height_m", "depth_m", "area_m2", "floor_area_m2", "thickness_m",
    "floor_dimension_u_m", "floor_dimension_v_m",
})
#: a rebuild may move a surface; beyond this the correction is not replayed onto a different entity
CARRY_MATCH_TOL_M = 1.0


class CorrectionError(ValueError):
    """A correction that cannot be applied honestly (relative world, unknown property, bad value)."""


def _anchor_of(entity) -> Optional[List[float]]:
    cp = getattr(entity, "custom_properties", None) or {}
    slab = cp.get("wall_slab") or {}
    start, end = slab.get("start_xyz_m"), slab.get("end_xyz_m")
    if start and end and len(start) == 3 and len(end) == 3:
        return [(float(a) + float(b)) / 2.0 for a, b in zip(start, end)]
    for key in ("bounds_min", "bounds_max"):
        if cp.get(key) is not None:
            lo, hi = cp.get("bounds_min"), cp.get("bounds_max")
            if lo and hi and len(lo) == len(hi):
                return [(float(a) + float(b)) / 2.0 for a, b in zip(lo, hi)]
    poly = cp.get("footprint_polygon_xyz_m")
    if poly:
        arr = [[float(c) for c in p] for p in poly]
        return [sum(p[i] for p in arr) / len(arr) for i in range(3)]
    t = getattr(entity, "transform", None) or {}
    pos = t.get("position") if isinstance(t, dict) else None
    if pos and len(pos) == 3:
        return [float(pos["x"]), float(pos["y"]), float(pos["z"])] if isinstance(pos, dict) else [float(v) for v in pos]
    return None


def make_correction(world, entity_id: str, property: str, value: float, note: str = "") -> dict:
    """Validate and shape one dimension correction (not yet applied). Raises CorrectionError."""
    entity = (world.entities or {}).get(entity_id)
    if entity is None:
        raise CorrectionError(f"entity '{entity_id}' is not in this world")
    if property not in CORRECTABLE_PROPERTIES:
        raise CorrectionError(f"'{property}' is not a correctable dimension; allowed: {sorted(CORRECTABLE_PROPERTIES)}")
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)) or value <= 0:
        raise CorrectionError("the corrected value must be a finite number > 0")
    state = scale_state_of(getattr(world, "metadata", None))
    if state != "metric":
        raise CorrectionError(
            "this world is not in metres: enter one known distance (scale) first, then correct dimensions")
    # A user-stated fact replaces a typical-size estimate: VERIFIED sits above the accepted prior's ESTIMATED cap.
    dim =dimension_from_metadata(float(value), getattr(world, "metadata", None), state="VERIFIED",
                                  basis=f"user-entered ({note.strip()})" if note.strip() else "user-entered",
                                  kind="area" if property.endswith("_m2") else "distance")
    if dim["state"] != "VERIFIED":
        raise CorrectionError(f"the correction could only be recorded as {dim['state']}; check the world's scale state")
    return {
        "kind": "dimension", "entity_id": entity_id, "entity_type": getattr(entity.type, "value", str(entity.type)),
        "property": property, "value": float(value), "dimension": dim, "note": note.strip() or None,
        "created_at": time.time(), "anchor": _anchor_of(entity),
    }


def apply_correction(world, record: dict, *, entity: Any = None) -> bool:
    """Apply one record to ``world`` (optionally onto a resolved ``entity``). Returns True when applied."""
    if record.get("kind") != "dimension":
        return False
    prop = record.get("property")
    value = record.get("value")
    if prop not in CORRECTABLE_PROPERTIES or not isinstance(value, (int, float)) or value <= 0:
        return False
    ent = entity if entity is not None else (world.entities or {}).get(record.get("entity_id"))
    if ent is None:
        return False
    cp = getattr(ent, "custom_properties", None)
    if not isinstance(cp, dict):
        return False
    dim = dimension_from_metadata(float(value), getattr(world, "metadata", None), state="VERIFIED",
                                  basis=record.get("basis") or (f"user-entered ({record['note']})" if record.get("note")
                                                                else "user-entered"),
                                  kind="area" if prop.endswith("_m2") else "distance")
    if dim["state"] != "VERIFIED":
        return False
    cp[prop] = float(value)
    dims = cp.get("dimensions") if isinstance(cp.get("dimensions"), dict) else {}
    dims = dict(dims)
    dims[prop] = dim
    cp["dimensions"] = dims
    stamp = cp.get("user_corrections") if isinstance(cp.get("user_corrections"), list) else []
    cp["user_corrections"] = (stamp + [{"property": prop, "value": float(value), "note": record.get("note"),
                                        "at": record.get("created_at")}])[-20:]
    return True


def record_correction(world, record: dict) -> dict:
    """Apply a new correction and append it to the world's replayable registry."""
    if not apply_correction(world, record):
        raise CorrectionError("the correction could not be applied to this world")
    existing = [r for r in (world.metadata.get("user_corrections") or [])
                if not (r.get("entity_id") == record.get("entity_id") and r.get("property") == record.get("property"))]
    world.metadata["user_corrections"] = existing + [record]
    return record


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.dist([float(v) for v in a], [float(v) for v in b])


def carry_corrections(world, records: Iterable[dict]) -> dict:
    """Replay saved corrections onto a freshly compiled world. Exact id first, else the nearest same-type surface
    within ``CARRY_MATCH_TOL_M``; otherwise skip with a reason. Never applies a fact to a different surface."""
    report = {"applied": [], "skipped": []}
    by_type: Dict[str, List[Any]] = {}
    for ent in (getattr(world, "entities", None) or {}).values():
        by_type.setdefault(getattr(ent.type, "value", str(ent.type)), []).append(ent)
    saved: List[dict] = []
    for rec in records or []:
        if not isinstance(rec, dict) or rec.get("kind") != "dimension":
            continue
        ent = (getattr(world, "entities", None) or {}).get(rec.get("entity_id"))
        matched = "id"
        if ent is None:
            anchor = rec.get("anchor")
            if not anchor:
                report["skipped"].append({"record": rec, "reason": "entity disappeared and the record has no anchor"})
                continue
            candidates = by_type.get(str(rec.get("entity_type")), [])
            scored = sorted(((_distance(anchor, a), e) for e in candidates if (a := _anchor_of(e)) is not None),
                            key=lambda p: p[0])
            if not scored or scored[0][0] > CARRY_MATCH_TOL_M:
                report["skipped"].append({"record": rec, "reason": "no surface of the same type within "
                                                                 f"{CARRY_MATCH_TOL_M} m of the stored position"})
                continue
            ent, matched = scored[0][1], f"nearest ({scored[0][0]:.2f} m)"
        if apply_correction(world, rec, entity=ent):
            report["applied"].append({"entity_id": ent.id, "property": rec.get("property"), "value": rec.get("value"),
                                      "matched_by": matched})
            saved.append(rec)
        else:
            report["skipped"].append({"record": rec, "reason": "the world's scale state cannot carry VERIFIED"})
    if saved:
        world.metadata["user_corrections"] = saved
    if report["applied"] or report["skipped"]:
        world.metadata["correction_carry"] = {"applied": len(report["applied"]), "skipped": len(report["skipped"])}
    return report
