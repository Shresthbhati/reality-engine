"""Cross-session entity re-identification (world memory campaign,
Phases 3/4: Entity Memory, Cross-Session Entity Re-Identification).

Given two WorldIR snapshots of the same site captured at different
times (e.g. two independent reconstructions), determine which entities
in `after` correspond to which entities in `before` -- "is this the
same wall, observed again" versus "this is a new entity".

This is deliberately NOT embedding-based. The campaign brief explicitly
prohibits fake embeddings, and there is no real vision-embedding model
installed anywhere in this repository -- adding a stub embedding layer
here would be exactly the "fake embeddings" violation the brief calls
out. Instead this uses the two kinds of real evidence WorldIR already
has for every promoted entity: semantic type (EntityType) and geometric
position (the same `SpatialIndex` position resolution used by
`engine/scene_graph/spatial_index.py` -- transform.position, falling
back to geometry AABB centroid). A learned appearance-embedding matcher
can be added later as an additional, clearly-labeled signal; it does
not block a real, useful, deterministic geometric matcher today.

Match classification, in decreasing confidence order:
  MATCH          -- same EntityType, within MATCH_DISTANCE_M.
  POSSIBLE_MATCH -- same EntityType, within POSSIBLE_MATCH_DISTANCE_M
                    but beyond MATCH_DISTANCE_M (plausibly the same
                    object that moved slightly, or a noisier second
                    reconstruction of the same static structure).
  NO_MATCH       -- no same-type candidate within POSSIBLE_MATCH_DISTANCE_M
                    of this entity, but a candidate elsewhere in `after`
                    exists (it's not new, it's just not been matched).
  UNRESOLVED     -- the entity in `before` has no resolvable position
                    (nothing to compare geometrically), so re-identification
                    genuinely cannot be attempted -- never silently
                    treated as a confident non-match.

Every match carries the evidence behind it (distance, both entity ids,
both types) -- "same class" is necessary but never sufficient on its
own (campaign rule 9: "similarity is not identity"), and this module
never merges entity records; it only reports a classified correspondence
for a caller (e.g. a memory-consolidation layer) to act on.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

from engine.scene_graph.spatial_index import Point, SpatialIndex, entity_position
from world_ir.schema_v1 import Entity
from world_ir.world_v1 import WorldIR

#: Within this distance and same EntityType: confident match. Chosen as
#: a coarse default matching the AABB-floor tolerance the physics/export
#: pipeline already uses elsewhere in this repo (10x the 1cm geometry
#: floor) -- not tuned against any real dataset, so treat it as a
#: starting point a caller can override, not a validated constant.
MATCH_DISTANCE_M = 0.5

#: Beyond MATCH_DISTANCE_M but within this: plausible match, lower
#: confidence. Wide enough to catch "the same wall, reconstructed a
#: second time with different noise" without conflating distinct rooms.
POSSIBLE_MATCH_DISTANCE_M = 2.0


class MatchKind(str, Enum):
    MATCH = "match"
    POSSIBLE_MATCH = "possible_match"
    NO_MATCH = "no_match"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class EntityMatch:
    before_entity_id: str
    #: None for NO_MATCH/UNRESOLVED -- there is no candidate to name.
    after_entity_id: Optional[str]
    kind: MatchKind
    #: None when there is no geometric comparison to report (UNRESOLVED,
    #: or NO_MATCH with zero same-type candidates in `after` at all).
    distance_m: Optional[float]
    reason: str

    def to_dict(self) -> dict:
        return {
            "before_entity_id": self.before_entity_id,
            "after_entity_id": self.after_entity_id,
            "kind": self.kind.value,
            "distance_m": self.distance_m,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ReIdentificationResult:
    matches: Tuple[EntityMatch, ...]

    @property
    def matched(self) -> Tuple[EntityMatch, ...]:
        return tuple(m for m in self.matches if m.kind is MatchKind.MATCH)

    @property
    def possible_matches(self) -> Tuple[EntityMatch, ...]:
        return tuple(m for m in self.matches if m.kind is MatchKind.POSSIBLE_MATCH)

    @property
    def unmatched(self) -> Tuple[EntityMatch, ...]:
        return tuple(m for m in self.matches if m.kind is MatchKind.NO_MATCH)

    @property
    def unresolved(self) -> Tuple[EntityMatch, ...]:
        return tuple(m for m in self.matches if m.kind is MatchKind.UNRESOLVED)

    def new_entity_ids(self, after: WorldIR) -> Tuple[str, ...]:
        """Entities present in `after` that no `before` entity matched or
        possibly-matched onto -- candidates for "newly observed"."""
        claimed = {m.after_entity_id for m in self.matches if m.after_entity_id is not None}
        return tuple(sorted(eid for eid in after.entities if eid not in claimed))

    def to_dict(self) -> dict:
        return {"matches": [m.to_dict() for m in self.matches]}


def reidentify_entities(
    before: WorldIR, after: WorldIR,
    match_distance_m: float = MATCH_DISTANCE_M,
    possible_match_distance_m: float = POSSIBLE_MATCH_DISTANCE_M,
) -> ReIdentificationResult:
    """For every entity in `before`, find its best same-type geometric
    correspondence in `after` and classify it. Deterministic: entities
    processed in sorted id order; ties among equidistant candidates
    broken by candidate entity id.
    """
    if match_distance_m < 0 or possible_match_distance_m < match_distance_m:
        raise ValueError(
            f"invalid thresholds: match_distance_m={match_distance_m}, "
            f"possible_match_distance_m={possible_match_distance_m}"
        )

    after_index = SpatialIndex(after)
    matches: List[EntityMatch] = []

    for entity_id in sorted(before.entities):
        entity = before.entities[entity_id]
        position = entity_position(before, entity)

        if position is None:
            matches.append(EntityMatch(
                before_entity_id=entity_id, after_entity_id=None, kind=MatchKind.UNRESOLVED,
                distance_m=None,
                reason="entity has no transform.position and no geometry with real bounds "
                       "-- nothing to geometrically compare",
            ))
            continue

        candidates = after_index.nearest(position, k=len(after.entities) or 1, predicate=lambda e, t=entity.type: e.type == t)
        if not candidates:
            matches.append(EntityMatch(
                before_entity_id=entity_id, after_entity_id=None, kind=MatchKind.NO_MATCH,
                distance_m=None,
                reason=f"no entity of type {entity.type.value} exists in the 'after' world at all",
            ))
            continue

        best_entity, best_distance = candidates[0]
        if best_distance <= match_distance_m:
            matches.append(EntityMatch(
                before_entity_id=entity_id, after_entity_id=best_entity.id, kind=MatchKind.MATCH,
                distance_m=best_distance,
                reason=f"same type ({entity.type.value}) within {match_distance_m}m ({best_distance:.3f}m)",
            ))
        elif best_distance <= possible_match_distance_m:
            matches.append(EntityMatch(
                before_entity_id=entity_id, after_entity_id=best_entity.id, kind=MatchKind.POSSIBLE_MATCH,
                distance_m=best_distance,
                reason=(
                    f"same type ({entity.type.value}) but {best_distance:.3f}m apart "
                    f"(beyond match threshold {match_distance_m}m, within possible threshold "
                    f"{possible_match_distance_m}m)"
                ),
            ))
        else:
            matches.append(EntityMatch(
                before_entity_id=entity_id, after_entity_id=None, kind=MatchKind.NO_MATCH,
                distance_m=best_distance,
                reason=(
                    f"closest same-type candidate ({best_entity.id}) is {best_distance:.3f}m away, "
                    f"beyond the possible-match threshold {possible_match_distance_m}m"
                ),
            ))

    return ReIdentificationResult(matches=tuple(matches))


# --------------------------------------------------------------------------------------------
# Identity carried across VERSIONS of one world, from measured spatial continuity
# --------------------------------------------------------------------------------------------

#: relations (engine.pipeline.spatial_continuity) that mean "the same physical surface, one-to-one"
_SAME_ENTITY = ("preserved", "refined")
#: relations that are NOT the same entity but are related; their lineage is recorded, identity is never forced
_LINEAGE = ("split", "merge", "regrouped", "extended", "reduced", "ambiguous")


def identity_plan(relations: list, cand_world: WorldIR, prev_ids) -> Tuple[dict, list]:
    """Which candidate entity IS which previous entity, from spatial-continuity relations.

    Returns ``(mapping, lineage)``. ``mapping`` (candidate id -> previous id) holds ONLY one-to-one
    preserved/refined relations of the same type whose previous id is not already taken by another
    candidate entity; everything else keeps its own id. ``lineage`` records the related-but-not-identical
    cases (split, merge, regrouped, extended, reduced, ambiguous) so they stay explicit -- ambiguity is
    never resolved by guessing.
    """
    mapping: dict = {}
    taken = set(cand_world.entities) - {c for r in relations for c in r.get("cand", [])}
    for r in relations:
        prev, cand = r.get("prev", []), r.get("cand", [])
        if r.get("kind") not in _SAME_ENTITY or len(prev) != 1 or len(cand) != 1:
            continue
        p, c = prev[0], cand[0]
        ent = cand_world.entities.get(c)
        if ent is None or p not in prev_ids or p in mapping.values() or (p in taken and p != c):
            continue
        mapping[c] = p
    lineage = [{"relation": r["kind"], "previous": list(r.get("prev", [])), "candidate": list(r.get("cand", [])),
                "type": r.get("type")}
               for r in relations if r.get("kind") in _LINEAGE]
    return mapping, lineage


def rekey_entities(world: WorldIR, mapping: dict) -> dict:
    """Give candidate entities the ids of the previous entities they ARE, updating every reference
    (entity table, relationships, transforms, temporal events, body states). Returns the applied mapping.
    Pure id renaming: no content changes. Entities already holding the target id are never overwritten."""
    applied = {c: p for c, p in mapping.items() if c in world.entities and (p not in world.entities or p == c)}
    applied = {c: p for c, p in applied.items() if c != p}
    if not applied:
        return {}
    world.entities = {applied.get(k, k): v for k, v in world.entities.items()}
    for new_id, ent in world.entities.items():
        ent.id = new_id
        for rel in ent.relationships:
            rel.target_id = applied.get(rel.target_id, rel.target_id)
    world.transforms = {applied.get(k, k): v for k, v in world.transforms.items()}
    for ev in world.temporal_events.values():
        if getattr(ev, "entity_id", None):
            ev.entity_id = applied.get(ev.entity_id, ev.entity_id)
    body = getattr(world.temporal_state, "body_states", None)
    if isinstance(body, dict):
        world.temporal_state.body_states = {applied.get(k, k): v for k, v in body.items()}
    return applied


def carry_identity(prev_entity_ids, relations: list, cand_world: WorldIR) -> dict:
    """Plan + apply (on the given world) + record provenance of every identity decision.

    Returns ``{"carried": {cand_id: prev_id}, "lineage": [...]}``. Each carried entity gets
    ``custom_properties["identity"]`` stating it was matched by spatial continuity (kind + previous id); each
    related-but-distinct entity gets ``custom_properties["continuity"]`` naming the relation and the previous
    entities it relates to."""
    mapping, lineage = identity_plan(relations, cand_world, set(prev_entity_ids))
    kind_of = {r["cand"][0]: r["kind"] for r in relations if len(r.get("cand", [])) == 1 and r["kind"] in _SAME_ENTITY}
    for c, p in mapping.items():
        cand_world.entities[c].custom_properties["identity"] = {
            "carried_from": p, "by": "spatial_continuity", "relation": kind_of.get(c)}
    for item in lineage:
        for c in item["candidate"]:
            ent = cand_world.entities.get(c)
            if ent is not None:
                ent.custom_properties["continuity"] = {"relation": item["relation"], "previous": item["previous"]}
    applied = rekey_entities(cand_world, mapping)
    return {"carried": applied, "lineage": lineage}
