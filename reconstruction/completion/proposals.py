"""Completion proposals with PRIOR_GENERATED provenance (Blueprint Task 10.3 & Stage S18).

Constrained completion proposes unobserved or occluded geometry (e.g. the unobserved backside
of furniture against a wall, unobserved underside of a table, or extrapolated boundary of a
partially occluded wall) using learned or geometric priors.

CRITICAL INVARIANTS (Docs Blueprint 11 section 3, Blueprint 12 stage S18):
1. Proposal geometry and entities MUST ALWAYS carry `provenance = Provenance.GENERATED`
   and `statement_state = StatementState.PREDICTED`.
2. Evidence state MUST be `"PRIOR_GENERATED"` (`custom_properties["evidence_state"] = "PRIOR_GENERATED"`).
3. Invariant 3: It must NEVER be labelled `OBSERVED` or `PARTIALLY_OBSERVED`, and must NEVER
   cite sensor observation IDs as if it were directly sensed.
4. Proposals are visibly marked (`visibly_marked = True`, dashed styling / pattern) so that
   viewers and downstream consumers can clearly distinguish proposed completion from real observation.
5. Acceptance Gate: evaluate continuity, normal consistency, collision, and evidence honesty.
   Accepting a proposal incorporates it into the world while strictly preserving its GENERATED /
   PRIOR_GENERATED provenance -- never auto-promoting to OBSERVED.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple
from uuid import uuid4

import numpy as np

from provenance import Provenance
from world_ir.schema_v1 import (
    Entity,
    EntityType,
    Geometry,
    GeometryType,
    Vector3,
)
from world_ir.statement_state import StatementState
from world_ir.validation import validate_world_ir
from world_ir.world_v1 import WorldIR


class CompletionMethod(str, Enum):
    """Method used to generate completion proposals."""
    SYMMETRY_PRIOR = "symmetry_prior"
    BOUNDING_BOX_PRIOR = "bounding_box_prior"
    PLANAR_EXTRAPOLATION = "planar_extrapolation"
    BACKSIDE_CLOSURE = "backside_closure"
    POINT_CLOUD_PRIOR = "point_cloud_prior"


@dataclass
class CompletionGateVerdict:
    """Acceptance gate assessment for a completion proposal."""
    passed: bool
    continuity_score: float
    normal_consistency: float
    collision_free: bool
    evidence_honesty: bool
    rejection_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "continuity_score": round(self.continuity_score, 4),
            "normal_consistency": round(self.normal_consistency, 4),
            "collision_free": self.collision_free,
            "evidence_honesty": self.evidence_honesty,
            "rejection_reasons": list(self.rejection_reasons),
        }

    @staticmethod
    def from_dict(data: dict) -> "CompletionGateVerdict":
        return CompletionGateVerdict(
            passed=bool(data.get("passed", False)),
            continuity_score=float(data.get("continuity_score", 0.0)),
            normal_consistency=float(data.get("normal_consistency", 0.0)),
            collision_free=bool(data.get("collision_free", False)),
            evidence_honesty=bool(data.get("evidence_honesty", False)),
            rejection_reasons=list(data.get("rejection_reasons", [])),
        )


@dataclass
class CompletionProposal:
    """A proposed completion for unobserved / occluded parts of an entity."""
    proposal_id: str
    target_entity_id: str
    completion_method: CompletionMethod
    rationale: str
    candidate_entity: Entity
    candidate_geometry: Geometry
    confidence: float
    unobserved_facets: list[str] = field(default_factory=list)
    gate_verdict: Optional[CompletionGateVerdict] = None
    is_accepted: bool = False

    def to_dict(self) -> dict:
        return {
            "proposal_id": self.proposal_id,
            "target_entity_id": self.target_entity_id,
            "completion_method": self.completion_method.value,
            "rationale": self.rationale,
            "candidate_entity": self.candidate_entity.to_dict(),
            "candidate_geometry": self.candidate_geometry.to_dict(),
            "confidence": round(self.confidence, 4),
            "unobserved_facets": list(self.unobserved_facets),
            "gate_verdict": self.gate_verdict.to_dict() if self.gate_verdict else None,
            "is_accepted": self.is_accepted,
        }

    @staticmethod
    def from_dict(data: dict) -> "CompletionProposal":
        gate_data = data.get("gate_verdict")
        return CompletionProposal(
            proposal_id=data["proposal_id"],
            target_entity_id=data["target_entity_id"],
            completion_method=CompletionMethod(data["completion_method"]),
            rationale=data.get("rationale", ""),
            candidate_entity=Entity(**{k: v for k, v in data["candidate_entity"].items() if k in Entity.__dataclass_fields__}),
            candidate_geometry=Geometry.from_dict(data["candidate_geometry"]),
            confidence=float(data.get("confidence", 0.5)),
            unobserved_facets=list(data.get("unobserved_facets", [])),
            gate_verdict=CompletionGateVerdict.from_dict(gate_data) if gate_data else None,
            is_accepted=bool(data.get("is_accepted", False)),
        )


def _ensure_proposal_invariants(
    candidate_entity: Entity,
    candidate_geometry: Geometry,
    target_entity_id: str,
    method: CompletionMethod,
    confidence: float,
) -> None:
    """Enforce strict PRIOR_GENERATED invariants on completion proposals."""
    # Provenance and statement state MUST be GENERATED / PREDICTED
    candidate_entity.provenance = Provenance.GENERATED
    candidate_entity.statement_state = StatementState.PREDICTED
    candidate_geometry.provenance = Provenance.GENERATED

    # Never claim sensor observations
    candidate_entity.observations = []
    candidate_geometry.observations = []

    # Bounded honest confidence
    clamped_conf = max(0.1, min(0.75, float(confidence)))
    candidate_entity.confidence = clamped_conf
    candidate_geometry.confidence = clamped_conf

    # Required custom property flags for UI and validation
    candidate_entity.custom_properties["evidence_state"] = "PRIOR_GENERATED"
    candidate_entity.custom_properties["is_completion_proposal"] = True
    candidate_entity.custom_properties["visibly_marked"] = True
    candidate_entity.custom_properties["completion_method"] = method.value
    candidate_entity.custom_properties["target_entity_id"] = target_entity_id
    candidate_entity.custom_properties["display_style"] = {
        "pattern": "dashed",
        "style": "dashed",
        "opacity": 0.65,
        "tint_hex": "#8b5cf6",  # Distinct violet tint for prior-generated completions
    }


def propose_backside_completion(
    world: WorldIR,
    entity_id: str,
    depth_m: float = 0.35,
    unobserved_axis: str = "-z",
) -> Optional[CompletionProposal]:
    """Propose completion of unobserved backside/underside of an entity (e.g. against a wall)."""
    target = world.entities.get(entity_id)
    if target is None:
        return None

    # Find geometry bounds
    target_geom = None
    for gid in target.geometry_ids:
        g = world.geometries.get(gid)
        if g is not None and g.bounds_min is not None and g.bounds_max is not None:
            target_geom = g
            break

    if target_geom is None or target_geom.bounds_min is None or target_geom.bounds_max is None:
        return None

    b_min = target_geom.bounds_min
    b_max = target_geom.bounds_max

    # Compute completed bounds by extending in the unobserved axis direction
    if unobserved_axis == "-z":
        comp_min = Vector3(b_min.x, b_min.y, b_min.z - depth_m)
        comp_max = Vector3(b_max.x, b_max.y, b_min.z)
        facets = ["rear_face", "wall_facing_surface"]
    elif unobserved_axis == "+z":
        comp_min = Vector3(b_min.x, b_min.y, b_max.z)
        comp_max = Vector3(b_max.x, b_max.y, b_max.z + depth_m)
        facets = ["front_face"]
    elif unobserved_axis == "-y":
        comp_min = Vector3(b_min.x, b_min.y - depth_m, b_min.z)
        comp_max = Vector3(b_max.x, b_min.y, b_max.z)
        facets = ["bottom_underside"]
    elif unobserved_axis == "-x":
        comp_min = Vector3(b_min.x - depth_m, b_min.y, b_min.z)
        comp_max = Vector3(b_min.x, b_max.y, b_max.z)
        facets = ["left_side"]
    else:
        comp_min = Vector3(b_max.x, b_min.y, b_min.z)
        comp_max = Vector3(b_max.x + depth_m, b_max.y, b_max.z)
        facets = ["right_side"]

    prop_id = f"prop-comp-{uuid4().hex[:8]}"
    geom_id = f"geom-{prop_id}"
    comp_geom = Geometry(
        id=geom_id,
        type=GeometryType.BOX,
        bounds_min=comp_min,
        bounds_max=comp_max,
        provenance=Provenance.GENERATED,
        confidence=0.60,
    )

    comp_entity = Entity(
        id=f"ent-{prop_id}",
        type=target.type,
        name=f"Proposed Completion: {target.name or entity_id} ({unobserved_axis})",
        geometry_ids=[geom_id],
        provenance=Provenance.GENERATED,
        confidence=0.60,
    )

    _ensure_proposal_invariants(
        candidate_entity=comp_entity,
        candidate_geometry=comp_geom,
        target_entity_id=entity_id,
        method=CompletionMethod.BACKSIDE_CLOSURE,
        confidence=0.60,
    )

    proposal = CompletionProposal(
        proposal_id=prop_id,
        target_entity_id=entity_id,
        completion_method=CompletionMethod.BACKSIDE_CLOSURE,
        rationale=f"Synthesized unobserved backside along {unobserved_axis} using bounding prior",
        candidate_entity=comp_entity,
        candidate_geometry=comp_geom,
        confidence=0.60,
        unobserved_facets=facets,
    )
    return proposal


def propose_symmetry_completion(
    world: WorldIR,
    entity_id: str,
    symmetry_axis: str = "x",
) -> Optional[CompletionProposal]:
    """Propose completion by bilateral reflection of an observed partial object."""
    target = world.entities.get(entity_id)
    if target is None:
        return None

    target_geom = None
    for gid in target.geometry_ids:
        g = world.geometries.get(gid)
        if g is not None and g.bounds_min is not None and g.bounds_max is not None:
            target_geom = g
            break

    if target_geom is None or target_geom.bounds_min is None or target_geom.bounds_max is None:
        return None

    b_min = target_geom.bounds_min
    b_max = target_geom.bounds_max

    # Mirror bounds across symmetry plane at axis center or edge
    if symmetry_axis == "x":
        center_x = (b_min.x + b_max.x) / 2.0
        extent_x = b_max.x - b_min.x
        comp_min = Vector3(b_max.x, b_min.y, b_min.z)
        comp_max = Vector3(b_max.x + extent_x, b_max.y, b_max.z)
        facets = ["symmetric_half_x"]
    elif symmetry_axis == "z":
        extent_z = b_max.z - b_min.z
        comp_min = Vector3(b_min.x, b_min.y, b_max.z)
        comp_max = Vector3(b_max.x, b_max.y, b_max.z + extent_z)
        facets = ["symmetric_half_z"]
    else:
        return None

    prop_id = f"prop-comp-{uuid4().hex[:8]}"
    geom_id = f"geom-{prop_id}"
    comp_geom = Geometry(
        id=geom_id,
        type=GeometryType.BOX,
        bounds_min=comp_min,
        bounds_max=comp_max,
        provenance=Provenance.GENERATED,
        confidence=0.55,
    )

    comp_entity = Entity(
        id=f"ent-{prop_id}",
        type=target.type,
        name=f"Proposed Symmetry Completion: {target.name or entity_id}",
        geometry_ids=[geom_id],
        provenance=Provenance.GENERATED,
        confidence=0.55,
    )

    _ensure_proposal_invariants(
        candidate_entity=comp_entity,
        candidate_geometry=comp_geom,
        target_entity_id=entity_id,
        method=CompletionMethod.SYMMETRY_PRIOR,
        confidence=0.55,
    )

    proposal = CompletionProposal(
        proposal_id=prop_id,
        target_entity_id=entity_id,
        completion_method=CompletionMethod.SYMMETRY_PRIOR,
        rationale=f"Mirrored unobserved opposite half across {symmetry_axis}-axis symmetry prior",
        candidate_entity=comp_entity,
        candidate_geometry=comp_geom,
        confidence=0.55,
        unobserved_facets=facets,
    )
    return proposal


def propose_planar_extrapolation(
    world: WorldIR,
    entity_id: str,
    extend_dist_m: float = 1.0,
) -> Optional[CompletionProposal]:
    """Propose extrapolation for an incomplete planar surface (e.g. wall, floor, ceiling)."""
    target = world.entities.get(entity_id)
    if target is None:
        return None

    target_geom = None
    for gid in target.geometry_ids:
        g = world.geometries.get(gid)
        if g is not None and g.bounds_min is not None and g.bounds_max is not None:
            target_geom = g
            break

    if target_geom is None or target_geom.bounds_min is None or target_geom.bounds_max is None:
        return None

    b_min = target_geom.bounds_min
    b_max = target_geom.bounds_max

    # Determine dominant wall plane
    dx = b_max.x - b_min.x
    dz = b_max.z - b_min.z

    if dx >= dz:
        # Extrapolate along X
        comp_min = Vector3(b_max.x, b_min.y, b_min.z)
        comp_max = Vector3(b_max.x + extend_dist_m, b_max.y, b_max.z)
        facets = ["wall_extension_x"]
    else:
        # Extrapolate along Z
        comp_min = Vector3(b_min.x, b_min.y, b_max.z)
        comp_max = Vector3(b_max.x, b_max.y, b_max.z + extend_dist_m)
        facets = ["wall_extension_z"]

    prop_id = f"prop-comp-{uuid4().hex[:8]}"
    geom_id = f"geom-{prop_id}"
    comp_geom = Geometry(
        id=geom_id,
        type=GeometryType.BOX,
        bounds_min=comp_min,
        bounds_max=comp_max,
        provenance=Provenance.GENERATED,
        confidence=0.65,
    )

    comp_entity = Entity(
        id=f"ent-{prop_id}",
        type=target.type,
        name=f"Proposed Wall Extrapolation: {target.name or entity_id}",
        geometry_ids=[geom_id],
        provenance=Provenance.GENERATED,
        confidence=0.65,
    )

    _ensure_proposal_invariants(
        candidate_entity=comp_entity,
        candidate_geometry=comp_geom,
        target_entity_id=entity_id,
        method=CompletionMethod.PLANAR_EXTRAPOLATION,
        confidence=0.65,
    )

    proposal = CompletionProposal(
        proposal_id=prop_id,
        target_entity_id=entity_id,
        completion_method=CompletionMethod.PLANAR_EXTRAPOLATION,
        rationale="Extrapolated incomplete wall segment along boundary plane",
        candidate_entity=comp_entity,
        candidate_geometry=comp_geom,
        confidence=0.65,
        unobserved_facets=facets,
    )
    return proposal


def evaluate_completion_acceptance_gate(
    world: WorldIR,
    proposal: CompletionProposal,
    max_penetration_m: float = 0.05,
) -> CompletionGateVerdict:
    """Evaluate acceptance gate for a completion proposal: continuity, normal consistency, collision, and evidence honesty."""
    rejection_reasons = []

    # 1. Evidence Honesty Invariant (Docs Blueprint 11 & Task 10.3)
    # MUST be GENERATED and PRIOR_GENERATED, never OBSERVED
    ent = proposal.candidate_entity
    geom = proposal.candidate_geometry

    evidence_honest = True
    if ent.provenance in (Provenance.OBSERVED, Provenance.RECONSTRUCTED):
        evidence_honest = False
        rejection_reasons.append(
            f"Completion proposal entity has forbidden provenance {ent.provenance}; must be GENERATED"
        )

    ev_state = ent.custom_properties.get("evidence_state")
    if ev_state != "PRIOR_GENERATED":
        evidence_honest = False
        rejection_reasons.append(
            f"Completion proposal entity has evidence_state {ev_state!r}; must be 'PRIOR_GENERATED'"
        )

    if ent.observations or geom.observations:
        evidence_honest = False
        rejection_reasons.append(
            "Completion proposal claims sensor observations; completion must have empty observations"
        )

    # 2. Continuity with target entity
    target = world.entities.get(proposal.target_entity_id)
    continuity_score = 0.0
    if target is not None and target.geometry_ids:
        t_geom = world.geometries.get(target.geometry_ids[0])
        if t_geom and t_geom.bounds_min and t_geom.bounds_max and geom.bounds_min and geom.bounds_max:
            # Check proximity / contact between target and proposal bounds
            dx = max(0.0, max(t_geom.bounds_min.x, geom.bounds_min.x) - min(t_geom.bounds_max.x, geom.bounds_max.x))
            dy = max(0.0, max(t_geom.bounds_min.y, geom.bounds_min.y) - min(t_geom.bounds_max.y, geom.bounds_max.y))
            dz = max(0.0, max(t_geom.bounds_min.z, geom.bounds_min.z) - min(t_geom.bounds_max.z, geom.bounds_max.z))
            gap = math.sqrt(dx * dx + dy * dy + dz * dz)
            if gap <= 0.05:
                continuity_score = 1.0
            elif gap <= 0.20:
                continuity_score = max(0.1, 1.0 - (gap / 0.20))
            else:
                rejection_reasons.append(f"Proposal is disconnected from target entity by {gap:.2f}m")

    # 3. Collision with other observed entities (non-target)
    collision_free = True
    if geom.bounds_min and geom.bounds_max:
        for other_id, other_ent in world.entities.items():
            if other_id == proposal.target_entity_id or other_id == ent.id:
                continue
            if other_ent.provenance not in (Provenance.OBSERVED, Provenance.RECONSTRUCTED):
                continue
            for ogid in other_ent.geometry_ids:
                og = world.geometries.get(ogid)
                if not (og and og.bounds_min and og.bounds_max):
                    continue
                # Compute penetration volume
                ix_min = max(geom.bounds_min.x, og.bounds_min.x)
                ix_max = min(geom.bounds_max.x, og.bounds_max.x)
                iy_min = max(geom.bounds_min.y, og.bounds_min.y)
                iy_max = min(geom.bounds_max.y, og.bounds_max.y)
                iz_min = max(geom.bounds_min.z, og.bounds_min.z)
                iz_max = min(geom.bounds_max.z, og.bounds_max.z)

                if ix_min < ix_max and iy_min < iy_max and iz_min < iz_max:
                    overlap_x = ix_max - ix_min
                    overlap_y = iy_max - iy_min
                    overlap_z = iz_max - iz_min
                    min_penetration = min(overlap_x, overlap_y, overlap_z)
                    if min_penetration > max_penetration_m:
                        collision_free = False
                        rejection_reasons.append(
                            f"Proposal collides with observed entity {other_id} by {min_penetration:.3f}m"
                        )
                        break

    # 4. Normal consistency
    normal_consistency = 1.0
    if geom.bounds_min and geom.bounds_max:
        if geom.bounds_min.x > geom.bounds_max.x or geom.bounds_min.y > geom.bounds_max.y or geom.bounds_min.z > geom.bounds_max.z:
            normal_consistency = 0.0
            rejection_reasons.append("Proposal bounds inverted; invalid normals")

    passed = evidence_honest and (continuity_score >= 0.5) and collision_free and (normal_consistency >= 0.9)

    verdict = CompletionGateVerdict(
        passed=passed,
        continuity_score=continuity_score,
        normal_consistency=normal_consistency,
        collision_free=collision_free,
        evidence_honesty=evidence_honest,
        rejection_reasons=rejection_reasons,
    )
    proposal.gate_verdict = verdict
    return verdict


def generate_completion_proposals_for_world(
    world: WorldIR,
    min_confidence: float = 0.4,
) -> List[CompletionProposal]:
    """Scan world entities and generate candidate completion proposals for occluded/unobserved parts."""
    proposals: List[CompletionProposal] = []

    for eid, entity in sorted(world.entities.items()):
        # Skip existing proposals or unclassified entities
        if entity.custom_properties.get("is_completion_proposal"):
            continue
        if entity.confidence < min_confidence:
            continue

        ev_state = entity.custom_properties.get("evidence_state")
        # Generate backside completion for furniture or structures
        if entity.type in (EntityType.STRUCTURE, EntityType.UNKNOWN):
            p = propose_backside_completion(world, eid, depth_m=0.30, unobserved_axis="-z")
            if p is not None:
                evaluate_completion_acceptance_gate(world, p)
                proposals.append(p)
        # Generate wall extrapolation for wall entities that are partially observed or occluded
        elif entity.type == EntityType.WALL:
            if ev_state in ("PARTIALLY_OBSERVED", "OCCLUDED", "UNOBSERVED") or entity.provenance == Provenance.INFERRED:
                p = propose_planar_extrapolation(world, eid, extend_dist_m=0.75)
                if p is not None:
                    evaluate_completion_acceptance_gate(world, p)
                    proposals.append(p)

    return proposals


def attach_completion_proposals(
    world: WorldIR,
    proposals: Sequence[CompletionProposal],
) -> WorldIR:
    """Attach completion proposals to the WorldIR, storing candidate geometries and entities visibly tagged."""
    w = copy.deepcopy(world)
    if "completion_proposals" not in w.metadata:
        w.metadata["completion_proposals"] = []

    for prop in proposals:
        # Enforce invariants again to guarantee safety
        _ensure_proposal_invariants(
            candidate_entity=prop.candidate_entity,
            candidate_geometry=prop.candidate_geometry,
            target_entity_id=prop.target_entity_id,
            method=prop.completion_method,
            confidence=prop.confidence,
        )

        w.geometries[prop.candidate_geometry.id] = prop.candidate_geometry
        w.entities[prop.candidate_entity.id] = prop.candidate_entity

        # Register proposal in metadata
        w.metadata["completion_proposals"].append(prop.to_dict())

    return w


def accept_completion_proposal(
    world: WorldIR,
    proposal_id: str,
) -> WorldIR:
    """Accept and commit a completion proposal into the world.

    INVARIANT PRESERVATION:
    Accepting a proposal incorporated it into the world as a verified prior-completion.
    It strictly PRESERVES `provenance = Provenance.GENERATED` and `evidence_state = "PRIOR_GENERATED"`.
    It is NEVER auto-promoted to OBSERVED!
    """
    w = copy.deepcopy(world)
    proposals_data = w.metadata.get("completion_proposals", [])

    matched = False
    for p_dict in proposals_data:
        if p_dict.get("proposal_id") == proposal_id:
            prop = CompletionProposal.from_dict(p_dict)
            verdict = evaluate_completion_acceptance_gate(w, prop)
            if not verdict.passed:
                raise ValueError(
                    f"Cannot accept completion proposal {proposal_id}: gate check failed ({'; '.join(verdict.rejection_reasons)})"
                )

            p_dict["is_accepted"] = True
            p_dict["accepted_at"] = True
            matched = True

            # Mark entity as accepted while strictly preserving PRIOR_GENERATED
            ent_id = prop.candidate_entity.id
            if ent_id in w.entities:
                target_ent = w.entities[ent_id]
                target_ent.custom_properties["is_accepted_completion"] = True
                target_ent.custom_properties["evidence_state"] = "PRIOR_GENERATED"
                target_ent.provenance = Provenance.GENERATED
                target_ent.statement_state = StatementState.PREDICTED
            break

    if not matched:
        raise KeyError(f"Proposal {proposal_id} not found in world completion_proposals metadata")

    return w
