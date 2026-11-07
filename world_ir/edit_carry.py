"""Manual edits survive reprocessing (plan T-4.2, rule 18: a late reconstruction result never overwrites an accepted manual edit).

Every accepted edit (rename, reclassify, move, a property the user set) is recorded in ``world.metadata["user_edits"]`` as it is made.
When a world is rebuilt from the evidence, ``carry_edits`` replays them onto the fresh world: exact id first, else the nearest element of
the SAME type within ``CARRY_TOL_M`` of where the edited element was, else skipped with the reason. The user's value always wins over the
automatic one, and an edit is never applied to an element of a different type or one that is far from the original.

Typed dimension corrections have their own registry (``world_ir.corrections``); this one covers everything else a person can change.
"""
from __future__ import annotations

import math
import time
from typing import Any, Dict, Iterable, List, Optional

from world_ir.corrections import _anchor_of

CARRY_TOL_M = 0.5
MAX_EDITS = 500
PIPELINE_OWNED = {"reconstruction", "wall_slab", "slab", "bootstrap"}
FIELDS = ("name", "semantic_labels", "transform", "entity_class")


def _type(entity) -> str:
    return getattr(entity.type, "value", str(entity.type))


def _anchor(entity) -> Optional[List[float]]:
    a = _anchor_of(entity)
    if a is not None:
        return a
    pos = (entity.transform or {}).get("position")
    return [float(pos["x"]), float(pos["y"]), float(pos["z"])] if pos else None


def _field_ok(field: str) -> bool:
    if field in FIELDS:
        return True
    if field.startswith("custom_properties."):
        return field.split(".", 1)[1].split(".")[0] not in PIPELINE_OWNED
    return False


def record_edit(world, entity, field: str, value: Any, *, at: Optional[float] = None) -> dict:
    """Remember that the user set ``field`` of ``entity`` to ``value``; the last edit per (entity, field) replaces earlier ones."""
    if field.startswith("custom_properties.") and field.split(".", 1)[1].split(".")[0] in PIPELINE_OWNED:
        raise ValueError(f"{field!r} is pipeline-owned and cannot be a user edit")
    if not _field_ok(field):
        raise ValueError(f"{field!r} is not an editable field")
    rec = {"kind": "edit", "field": field, "value": value, "entity_id": entity.id, "entity_type": _type(entity),
           "anchor": _anchor(entity), "at": at if at is not None else time.time()}
    kept = [r for r in (world.metadata.get("user_edits") or []) if not (r["entity_id"] == entity.id and r["field"] == field)]
    world.metadata["user_edits"] = (kept + [rec])[-MAX_EDITS:]
    return rec


def _apply(entity, rec: dict) -> None:
    field, value = rec["field"], rec["value"]
    if field == "name":
        entity.name = str(value)
    elif field == "semantic_labels":
        entity.semantic_labels = list(value)
    elif field == "transform":
        entity.transform = dict(value)
    elif field == "entity_class":
        cp = dict(entity.custom_properties or {})
        cp["entity_class"], cp["classification_basis"] = str(value), "user"
        entity.custom_properties = cp
    else:
        key = field.split(".", 1)[1]
        cp = dict(entity.custom_properties or {})
        cp[key] = value
        entity.custom_properties = cp


def carry_edits(world, records: Iterable[dict]) -> Dict[str, list]:
    """Replay ``records`` onto a freshly built ``world``; returns ``{"applied": [...], "skipped": [...]}`` and keeps the carried ones."""
    report: Dict[str, list] = {"applied": [], "skipped": []}
    by_type: Dict[str, list] = {}
    for e in world.entities.values():
        by_type.setdefault(_type(e), []).append(e)
    kept: List[dict] = []
    for rec in records or []:
        if not isinstance(rec, dict) or rec.get("kind") != "edit" or not _field_ok(str(rec.get("field", ""))):
            continue
        ent = world.entities.get(rec["entity_id"])
        matched = "id"
        if ent is not None and _type(ent) != rec.get("entity_type"):
            ent = None                                    # the id now names something else: never apply the edit to it
        if ent is None:
            anchor = rec.get("anchor")
            cands = by_type.get(str(rec.get("entity_type")), [])
            scored = sorted(((math.dist(anchor, a), e) for e in cands if anchor and (a := _anchor(e)) is not None), key=lambda p: p[0])
            if not scored or scored[0][0] > CARRY_TOL_M:
                report["skipped"].append({"record": rec, "reason": f"no {rec.get('entity_type')} within {CARRY_TOL_M} m of where it was edited"})
                continue
            ent, matched = scored[0][1], f"nearest ({scored[0][0]:.2f} m)"
        _apply(ent, rec)
        report["applied"].append({"entity_id": ent.id, "field": rec["field"], "matched_by": matched})
        kept.append({**rec, "entity_id": ent.id, "anchor": _anchor(ent) or rec.get("anchor")})
    if kept:
        world.metadata["user_edits"] = kept
    if report["applied"] or report["skipped"]:
        world.metadata["edit_carry"] = {"applied": len(report["applied"]), "skipped": len(report["skipped"])}
    return report
