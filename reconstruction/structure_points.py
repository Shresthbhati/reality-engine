"""The few points that DEFINE a room: its wall/floor/ceiling corners and the corners of its door/window openings.

Derived only from the compiled wall slabs (``custom_properties["wall_slab"]``: start/end on the floor, direction, up,
height, carved openings), so nothing is invented: no wall slab, no point. Unobserved gaps are not openings and are
skipped. Positions are in the world's own unit (relative or metric); the duplicate tolerance is a fraction of the
points' bounding diagonal, so it does not depend on that unit.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

import numpy as np

#: Points closer than this fraction of the bounding diagonal are one point (adjacent walls share their corners).
DEDUPE_FRACTION = 0.01
_KIND_RANK = {"corner": 0, "opening_corner": 1, "dimension": 2}


def _entities(world_or_entities: Any) -> Iterable[Any]:
    ents = getattr(world_or_entities, "entities", world_or_entities)
    return list(ents.values()) if isinstance(ents, dict) else list(ents or [])


def structure_points(world_ir_or_entities: Any) -> List[Dict[str, Any]]:
    """[{id, kind: corner|opening_corner|dimension, position: [x, y, z], entity_ids, label}] ; [] when no structure."""
    cands: List[Dict[str, Any]] = []
    for ent in _entities(world_ir_or_entities):
        cp = getattr(ent, "custom_properties", None) or {}
        slab = cp.get("wall_slab")
        if not slab or ent.type.value != "wall":
            continue
        a, b = np.asarray(slab["start_xyz_m"], float), np.asarray(slab["end_xyz_m"], float)
        d, up = np.asarray(slab["direction"], float), np.asarray(slab["up"], float)
        h = float(slab["height_m"])
        for p in (a, b, a + up * h, b + up * h):
            cands.append({"kind": "corner", "position": p, "entity_ids": [ent.id, cp.get("room_id")], "label": None})
        for op in slab.get("openings") or []:
            o = a + d * float(op["offset_m"]) + up * float(op["sill_m"])
            w, oh = d * float(op["width_m"]), up * float(op["height_m"])
            for p in (o, o + w, o + oh, o + w + oh):
                cands.append({"kind": "opening_corner", "position": p, "entity_ids": [op["entity_id"], ent.id],
                              "label": op.get("kind")})
    if not cands:
        return []
    pos = np.array([c["position"] for c in cands])
    tol = DEDUPE_FRACTION * float(np.linalg.norm(pos.max(axis=0) - pos.min(axis=0)))

    merged: List[Dict[str, Any]] = []
    for c in cands:
        near = next((m for m in merged if np.linalg.norm(m["position"] - c["position"]) <= tol), None)
        if near is None:
            merged.append({**c, "entity_ids": []})
            near = merged[-1]
        elif _KIND_RANK[c["kind"]] < _KIND_RANK[near["kind"]]:
            near["kind"], near["label"] = c["kind"], c["label"]
        near["entity_ids"] += [e for e in c["entity_ids"] if e and e not in near["entity_ids"]]
    return [{"id": f"sp-{i + 1:03d}", "kind": m["kind"], "position": [float(v) for v in m["position"]],
             "entity_ids": m["entity_ids"], "label": m["label"]} for i, m in enumerate(merged)]
