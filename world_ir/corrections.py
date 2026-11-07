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


def reclassify_entity(
    world,
    entity_id: str,
    new_class: str,
    note: str = "",
    user_id: str = "",
) -> dict:
    """User correction: reclassify an entity to a new ontology class.

    Writes a user-basis decision and updates semantic labels, entity_class, and name.
    """
    ent = (getattr(world, "entities", None) or {}).get(entity_id)
    if ent is None:
        raise CorrectionError(f"entity '{entity_id}' not found in world")
    if not new_class or not isinstance(new_class, str):
        raise CorrectionError("new_class must be a non-empty string")

    old_class = (ent.custom_properties or {}).get("entity_class") or (
        ent.semantic_labels[0] if ent.semantic_labels else "unknown"
    )
    clean_class = new_class.strip()

    cp = getattr(ent, "custom_properties", None) or {}
    cp = dict(cp)
    cp["entity_class"] = clean_class
    cp["classification_basis"] = "user"

    decision = {
        "kind": "reclassify",
        "basis": "user",
        "entity_id": entity_id,
        "entity_type": getattr(ent.type, "value", str(ent.type)),
        "anchor": _anchor_of(ent),
        "old_class": old_class,
        "new_class": clean_class,
        "note": note.strip() or None,
        "user_id": user_id or "user",
        "timestamp": time.time(),
    }
    cp["user_decision"] = decision
    ent.custom_properties = cp

    # Update semantic labels
    labels = [clean_class]
    for part in clean_class.split("/"):
        if part and part not in labels:
            labels.append(part)
    ent.semantic_labels = labels

    # Update friendly display name
    leaf_name = clean_class.split("/")[-1].replace("_", " ").title()
    ent.name = leaf_name

    # Record in world metadata
    decisions = list(world.metadata.get("user_decisions") or [])
    decisions.append(decision)
    world.metadata["user_decisions"] = decisions

    return decision


def _get_box(ent) -> Optional[Tuple[List[float], List[float]]]:
    cp = getattr(ent, "custom_properties", None) or {}
    if "bounding_box" in cp and isinstance(cp["bounding_box"], (list, tuple)) and len(cp["bounding_box"]) == 2:
        return [list(cp["bounding_box"][0]), list(cp["bounding_box"][1])]
    if "bounds" in cp and isinstance(cp["bounds"], (list, tuple)) and len(cp["bounds"]) == 2:
        return [list(cp["bounds"][0]), list(cp["bounds"][1])]
    anchor = _anchor_of(ent)
    if anchor is not None:
        return [
            [anchor[0] - 0.5, anchor[1] - 0.5, anchor[2] - 0.5],
            [anchor[0] + 0.5, anchor[1] + 0.5, anchor[2] + 0.5],
        ]
    return [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]


def split_entity(
    world,
    entity_id: str,
    split_axis: str = "x",
    split_ratio: float = 0.5,
    note: str = "",
    user_id: str = "",
) -> Tuple[Any, Any, dict]:
    """User correction: split an entity into two distinct entities along an axis.

    Writes a user-basis decision, replaces source entity with two sub-entities, and records provenance.
    """
    ent = (getattr(world, "entities", None) or {}).get(entity_id)
    if ent is None:
        raise CorrectionError(f"entity '{entity_id}' not found in world")

    axis = split_axis.lower()
    if axis not in ("x", "y", "z"):
        raise CorrectionError(f"invalid split_axis '{split_axis}'; allowed: 'x', 'y', 'z'")
    if not (0.05 <= split_ratio <= 0.95):
        raise CorrectionError(f"split_ratio {split_ratio} must be between 0.05 and 0.95")

    axis_idx = {"x": 0, "y": 1, "z": 2}[axis]
    box = _get_box(ent)
    b_min, b_max = list(box[0]), list(box[1])

    split_coord = b_min[axis_idx] + split_ratio * (b_max[axis_idx] - b_min[axis_idx])

    # Box 1: [b_min, split_coord on axis]
    box1_max = list(b_max)
    box1_max[axis_idx] = split_coord
    box1 = (tuple(b_min), tuple(box1_max))

    # Box 2: [split_coord on axis, b_max]
    box2_min = list(b_min)
    box2_min[axis_idx] = split_coord
    box2 = (tuple(box2_min), tuple(b_max))

    from world_ir.schema_v1 import Entity as WEntity, Provenance as WProv

    id1 = f"{entity_id}_part1"
    id2 = f"{entity_id}_part2"

    cp1 = dict(ent.custom_properties or {})
    cp1["bounding_box"] = [list(box1[0]), list(box1[1])]
    cp1["split_from"] = entity_id
    cp1["decision_basis"] = "user"

    cp2 = dict(ent.custom_properties or {})
    cp2["bounding_box"] = [list(box2[0]), list(box2[1])]
    cp2["split_from"] = entity_id
    cp2["decision_basis"] = "user"

    ent1 = WEntity(
        id=id1,
        type=ent.type,
        name=f"{ent.name} (Part 1)",
        semantic_labels=list(ent.semantic_labels),
        provenance=WProv.RECONSTRUCTED,
        confidence=float(ent.confidence),
        custom_properties=cp1,
        observations=list(ent.observations),
    )
    ent2 = WEntity(
        id=id2,
        type=ent.type,
        name=f"{ent.name} (Part 2)",
        semantic_labels=list(ent.semantic_labels),
        provenance=WProv.RECONSTRUCTED,
        confidence=float(ent.confidence),
        custom_properties=cp2,
        observations=list(ent.observations),
    )

    # Rewire existing relationships in the world
    for other in world.entities.values():
        if other.id == entity_id:
            continue
        for rel in other.relationships:
            if rel.target_id == entity_id:
                rel.target_id = id1

    # Replace in world
    del world.entities[entity_id]
    world.entities[id1] = ent1
    world.entities[id2] = ent2

    decision = {
        "kind": "split",
        "basis": "user",
        "source_entity_id": entity_id,
        "entity_type": getattr(ent.type, "value", str(ent.type)),
        "anchor": _anchor_of(ent),
        "created_entity_ids": [id1, id2],
        "split_axis": axis,
        "split_ratio": float(split_ratio),
        "note": note.strip() or None,
        "user_id": user_id or "user",
        "timestamp": time.time(),
    }
    decisions = list(world.metadata.get("user_decisions") or [])
    decisions.append(decision)
    world.metadata["user_decisions"] = decisions

    return ent1, ent2, decision


def merge_entities(
    world,
    primary_entity_id: str,
    secondary_entity_ids: Sequence[str],
    new_name: Optional[str] = None,
    note: str = "",
    user_id: str = "",
) -> Tuple[Any, dict]:
    """User correction: merge multiple entities into one primary entity.

    Combines bounds, observations, subcomponents, and rewires all relationships.
    Writes a user-basis decision.
    """
    primary = (getattr(world, "entities", None) or {}).get(primary_entity_id)
    if primary is None:
        raise CorrectionError(f"primary entity '{primary_entity_id}' not found in world")

    secondaries = []
    for sid in secondary_entity_ids:
        sec = (getattr(world, "entities", None) or {}).get(sid)
        if sec is None:
            raise CorrectionError(f"secondary entity '{sid}' not found in world")
        if sid != primary_entity_id:
            secondaries.append(sec)

    if not secondaries:
        raise CorrectionError("at least one secondary entity must be provided to merge")

    # Union bounding boxes
    all_ents = [primary] + secondaries
    boxes = [_get_box(e) for e in all_ents]
    min_x = min(b[0][0] for b in boxes)
    min_y = min(b[0][1] for b in boxes)
    min_z = min(b[0][2] for b in boxes)
    max_x = max(b[1][0] for b in boxes)
    max_y = max(b[1][1] for b in boxes)
    max_z = max(b[1][2] for b in boxes)

    cp = dict(primary.custom_properties or {})
    cp["bounding_box"] = [[min_x, min_y, min_z], [max_x, max_y, max_z]]
    cp["merged_from"] = [s.id for s in secondaries]
    cp["decision_basis"] = "user"
    primary.custom_properties = cp

    if new_name:
        primary.name = new_name.strip()

    # Merge observations and components
    seen_obs_ids = {o.id for o in primary.observations}
    for sec in secondaries:
        for obs in sec.observations:
            if obs.id not in seen_obs_ids:
                primary.observations.append(obs)
                seen_obs_ids.add(obs.id)
        for cid in sec.component_ids:
            if cid not in primary.component_ids:
                primary.component_ids.append(cid)
        for gid in sec.geometry_ids:
            if gid not in primary.geometry_ids:
                primary.geometry_ids.append(gid)

    # Rewire all external relationships targeting any secondary entity to primary
    sec_ids = {s.id for s in secondaries}
    for other in world.entities.values():
        if other.id in sec_ids:
            continue
        for rel in other.relationships:
            if rel.target_id in sec_ids:
                rel.target_id = primary_entity_id

    # Remove secondaries from world
    for sec in secondaries:
        if sec.id in world.entities:
            del world.entities[sec.id]

    decision = {
        "kind": "merge",
        "basis": "user",
        "primary_entity_id": primary_entity_id,
        "entity_type": getattr(primary.type, "value", str(primary.type)),
        "anchor": _anchor_of(primary),
        "merged_entity_ids": list(sec_ids),
        "note": note.strip() or None,
        "user_id": user_id or "user",
        "timestamp": time.time(),
    }
    decisions = list(world.metadata.get("user_decisions") or [])
    decisions.append(decision)
    world.metadata["user_decisions"] = decisions

    return primary, decision


def carry_decisions(world, decisions: Iterable[dict]) -> dict:
    """Replay saved user decisions (reclassify, split, merge) onto a freshly compiled world (T-4.2).

    Exact entity id first; if not found, falls back to the nearest entity of the same type within
    CARRY_MATCH_TOL_M. Re-applies the decision and preserves the decision registry in world.metadata["user_decisions"].
    """
    report: Dict[str, list] = {"applied": [], "skipped": []}
    by_type: Dict[str, List[Any]] = {}
    for ent in (getattr(world, "entities", None) or {}).values():
        by_type.setdefault(getattr(ent.type, "value", str(ent.type)), []).append(ent)

    saved: List[dict] = []
    for dec in decisions or []:
        if not isinstance(dec, dict) or dec.get("basis") != "user":
            continue
        kind = dec.get("kind")
        if kind == "reclassify":
            target_id = dec.get("entity_id")
            ent = (getattr(world, "entities", None) or {}).get(target_id)
            matched = "id"
            if ent is None:
                anchor = dec.get("anchor")
                if not anchor:
                    report["skipped"].append({"decision": dec, "reason": "entity not found and no anchor"})
                    saved.append(dec)
                    continue
                candidates = by_type.get(str(dec.get("entity_type")), [])
                scored = sorted(
                    ((_distance(anchor, a), e) for e in candidates if (a := _anchor_of(e)) is not None),
                    key=lambda p: p[0],
                )
                if not scored or scored[0][0] > CARRY_MATCH_TOL_M:
                    report["skipped"].append({
                        "decision": dec,
                        "reason": f"no entity of type {dec.get('entity_type')} within {CARRY_MATCH_TOL_M} m",
                    })
                    saved.append(dec)
                    continue
                ent, matched = scored[0][1], f"nearest ({scored[0][0]:.2f} m)"

            clean_class = str(dec.get("new_class") or "").strip()
            if clean_class:
                cp = getattr(ent, "custom_properties", None) or {}
                cp = dict(cp)
                cp["entity_class"] = clean_class
                cp["classification_basis"] = "user"
                cp["user_decision"] = dec
                ent.custom_properties = cp

                labels = [clean_class]
                for part in clean_class.split("/"):
                    if part and part not in labels:
                        labels.append(part)
                ent.semantic_labels = labels
                leaf_name = clean_class.split("/")[-1].replace("_", " ").title()
                ent.name = leaf_name

                report["applied"].append({
                    "entity_id": ent.id,
                    "kind": "reclassify",
                    "new_class": clean_class,
                    "matched_by": matched,
                })
            saved.append(dec)
        elif kind == "split":
            source_id = dec.get("source_entity_id")
            ent = (getattr(world, "entities", None) or {}).get(source_id)
            if ent is not None:
                try:
                    split_entity(
                        world,
                        ent.id,
                        dec.get("split_axis", "x"),
                        dec.get("split_ratio", 0.5),
                        note=dec.get("note") or "",
                        user_id=dec.get("user_id") or "user",
                    )
                    report["applied"].append({"entity_id": ent.id, "kind": "split"})
                except Exception as exc:
                    report["skipped"].append({"decision": dec, "reason": str(exc)})
            saved.append(dec)
        elif kind == "merge":
            p_id = dec.get("primary_entity_id")
            m_ids = dec.get("merged_entity_ids") or []
            if p_id in (world.entities or {}) and any(sid in (world.entities or {}) for sid in m_ids):
                try:
                    available_secondaries = [sid for sid in m_ids if sid in world.entities]
                    merge_entities(
                        world,
                        p_id,
                        available_secondaries,
                        note=dec.get("note") or "",
                        user_id=dec.get("user_id") or "user",
                    )
                    report["applied"].append({"entity_id": p_id, "kind": "merge"})
                except Exception as exc:
                    report["skipped"].append({"decision": dec, "reason": str(exc)})
            saved.append(dec)
        else:
            saved.append(dec)

    if saved:
        world.metadata["user_decisions"] = saved
    return report
