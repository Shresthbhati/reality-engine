"""Door-based room connectivity as explicit relationships, with UNKNOWN instead of assumed (Sprint 2, workstream 6).

``world_ir.validation`` only said "rooms form more than one component". This module derives WHICH rooms each door/opening joins
from evidence already in the world (explicit ``connected_room_ids``, connects/adjoins relationships, or the rooms whose boundary
contains the opening's host wall) and persists the result:

* ``opening --connects--> room`` and ``room --connects--> room`` relationships, tagged ``derived_from="door_room_connectivity"``
  with the basis that justified each (replaced, never duplicated, on a re-run);
* ``world.metadata["room_connectivity"]``: edges, components, openings whose other side is UNKNOWN, rooms with no door evidence.

An opening with one evidenced side is ``one_side_known`` -- the other side is not guessed (not "exterior", not "the neighbour").
No geometry is invented. Wall/closure checks live in ``engine/validation/room_geometry.py`` and are intentionally not duplicated.
"""

from __future__ import annotations

from typing import Any, Dict, List, Set

from world_ir.schema_v1 import Provenance, Relationship, RelationshipKind

DERIVED_FROM = "door_room_connectivity"
META_KEY = "room_connectivity"
_SPACES = {"room", "corridor"}
_PASSAGES = {"door", "opening"}
_LINKS = {"connects", "adjoins", "adjacent_to"}


def _t(e: Any) -> str:
    return str(getattr(e.type, "value", e.type))


def _k(rel: Any) -> str:
    return str(getattr(rel.kind, "value", rel.kind))


def _own(rel: Any) -> bool:
    return (getattr(rel, "metadata", None) or {}).get("derived_from") == DERIVED_FROM


def _rooms_of_opening(world: Any, oid: str, spaces: Set[str]) -> Dict[str, str]:
    """room id -> basis, from evidence only (never from our own previously derived relationships)."""
    ent = world.entities[oid]
    cp = ent.custom_properties or {}
    found: Dict[str, str] = {}
    for key in ("connected_room_ids", "room_ids", "between_room_ids"):
        for r in cp.get(key) or []:
            if r in spaces:
                found.setdefault(r, f"custom_properties.{key}")
    for rel in ent.relationships or []:
        if not _own(rel) and _k(rel) in _LINKS and rel.target_id in spaces:
            found.setdefault(rel.target_id, f"relationship:{_k(rel)}")
    for sid in spaces:
        for rel in world.entities[sid].relationships or []:
            if not _own(rel) and rel.target_id == oid and _k(rel) in _LINKS:
                found.setdefault(sid, f"relationship:{_k(rel)} (from space)")
    host = cp.get("host_wall_id") or cp.get("surface_id") or (cp.get("bootstrap") or {}).get("host_entity_id")
    if host in world.entities:
        for sid in spaces:
            if host in (world.entities[sid].custom_properties.get("boundary_element_ids") or ()):
                found.setdefault(sid, "host_wall_in_room_boundary")
        hcp = world.entities[host].custom_properties or {}
        for key in ("space_id", "room_id"):
            if hcp.get(key) in spaces:
                found.setdefault(hcp[key], f"host_wall.{key}")
    return found


def derive_door_connectivity(world: Any) -> dict:
    """Pure: the door-based room graph and what is unknown. Does not modify ``world``."""
    ents = world.entities or {}
    spaces = {i for i in ents if _t(ents[i]) in _SPACES}
    rooms = sorted(i for i in spaces if _t(ents[i]) == "room")
    edges: List[dict] = []
    unresolved: List[dict] = []
    for oid in sorted(i for i in ents if _t(ents[i]) in _PASSAGES):
        found = _rooms_of_opening(world, oid, spaces)
        ids = sorted(found)
        if len(ids) >= 2:
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    edges.append({"opening_id": oid, "room_a": ids[i], "room_b": ids[j],
                                  "basis": sorted({found[ids[i]], found[ids[j]]})})
        else:
            unresolved.append({"opening_id": oid, "known_rooms": ids,
                               "status": "one_side_known" if ids else "unknown",
                               "reason": ("only one side of the opening is evidenced; the other side is not assumed"
                                          if ids else "no room is evidenced on either side of the opening")})
    adj: Dict[str, Set[str]] = {r: set() for r in rooms}
    for e in edges:
        for a, b in ((e["room_a"], e["room_b"]), (e["room_b"], e["room_a"])):
            if a in adj and b in adj:
                adj[a].add(b)
    seen: Set[str] = set()
    comps: List[List[str]] = []
    for r in rooms:
        if r in seen:
            continue
        comp, todo = [], [r]
        while todo:
            x = todo.pop()
            if x not in seen:
                seen.add(x)
                comp.append(x)
                todo.extend(adj[x] - seen)
        comps.append(sorted(comp))
    with_door = {x for e in edges for x in (e["room_a"], e["room_b"])}
    status = ("not_applicable" if len(rooms) < 2 else "connected" if len(comps) == 1
              else "unknown" if not edges else "partial")
    return {"status": status, "rooms": rooms, "edges": edges, "components": comps,
            "unresolved_openings": unresolved, "rooms_without_door_evidence": [r for r in rooms if r not in with_door],
            "note": "connectivity is derived from door/opening evidence only; anything not evidenced is unknown, never assumed"}


def apply_door_connectivity(world: Any) -> dict:
    """Persist :func:`derive_door_connectivity` as relationships + ``metadata['room_connectivity']``. Idempotent."""
    ents = world.entities or {}
    for e in ents.values():                       # replace what a previous run derived
        if any(_own(r) for r in e.relationships or []):
            e.relationships = [r for r in e.relationships if not _own(r)]
    result = derive_door_connectivity(world)
    for ed in result["edges"]:
        meta = {"derived_from": DERIVED_FROM, "via_opening_id": ed["opening_id"], "basis": ed["basis"]}
        for src, dst in ((ed["opening_id"], ed["room_a"]), (ed["opening_id"], ed["room_b"]),
                         (ed["room_a"], ed["room_b"]), (ed["room_b"], ed["room_a"])):
            if not any(r.target_id == dst and _own(r) and r.metadata.get("via_opening_id") == ed["opening_id"]
                       for r in ents[src].relationships):
                ents[src].relationships.append(Relationship(
                    kind=RelationshipKind.CONNECTS, target_id=dst, confidence=None,
                    provenance=Provenance.INFERRED, metadata=dict(meta)))
    world.metadata[META_KEY] = result
    return result


def refresh_topology_metadata(world: Any) -> None:
    """Derive door connectivity and surface competing hypotheses / unresolved geometry into the persisted world."""
    from world_ir.hypotheses import surface_unresolved

    apply_door_connectivity(world)
    surface_unresolved(world)
