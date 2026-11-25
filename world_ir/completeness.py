"""Multi-dimensional reconstruction completeness assessment for WorldIR.

Measures each dimension separately (never collapsing into a single arbitrary score
and never counting missing/unobserved entities as successfully reconstructed).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List

from provenance import Provenance
from world_ir.constraints import check_architectural_constraints
from world_ir.observations import AssociationState, ledger_of
from world_ir.schema_v1 import EntityType
from world_ir.world_v1 import WorldIR
from world_ir.uncertainty_model import EpistemicState, propagate_entity_uncertainty


@dataclass(frozen=True)
class WorldCompletenessReport:
    entity_count: int
    geometric_coverage: float
    surface_completeness: float
    entity_detection_coverage: float
    topological_completeness: float
    semantic_completeness: float
    evidence_coverage: float
    uncertainty_breakdown: Dict[str, int]
    unresolved_contradictions: List[str]
    unknown_gaps: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def assess_world_completeness(world: WorldIR) -> WorldCompletenessReport:
    """Assess multi-dimensional completeness and surface what remains unknown."""
    entities = list((world.entities or {}).values())
    n = len(entities)
    if n == 0:
        return WorldCompletenessReport(
            entity_count=0,
            geometric_coverage=0.0,
            surface_completeness=0.0,
            entity_detection_coverage=0.0,
            topological_completeness=0.0,
            semantic_completeness=0.0,
            evidence_coverage=0.0,
            uncertainty_breakdown={s.value: 0 for s in EpistemicState},
            unresolved_contradictions=[],
            unknown_gaps=["No entities have been compiled into this world yet."],
        )

    with_geom = sum(
        1 for e in entities
        if any(gid in world.geometries for gid in (e.geometry_ids or ()))
    )
    surfaces = [e for e in entities if e.type in (EntityType.WALL, EntityType.FLOOR, EntityType.CEILING, EntityType.ROOF)]
    surfaces_observed = sum(
        1 for s in surfaces
        if s.provenance in (Provenance.OBSERVED, Provenance.RECONSTRUCTED)
        and any(gid in world.geometries for gid in (s.geometry_ids or ()))
    )

    _, decisions = ledger_of(world)
    with_evidence = 0
    for e in entities:
        has_obs = bool(e.observations) or bool((e.custom_properties or {}).get("observation_ids"))
        has_ledger = any(dec.entity_id == e.id for dec in decisions)
        if has_obs or has_ledger:
            with_evidence += 1

    with_semantic = sum(
        1 for e in entities
        if e.semantic_labels and not all("unknown" in lbl.lower() or "fallback" in lbl.lower() for lbl in e.semantic_labels)
    )

    with_topo = sum(1 for e in entities if e.relationships or getattr(e, "parent_id", None))

    constraints = check_architectural_constraints(world)
    contradictions: List[str] = [
        f"{v.code}: {v.message}" for v in constraints if v.severity == "error"
    ]
    for dec in decisions:
        if dec.state is AssociationState.AMBIGUOUS:
            contradictions.append(f"ambiguous_association ({dec.observation_id}): {dec.reason}")

    breakdown: Dict[str, int] = {s.value: 0 for s in EpistemicState}
    unknown_gaps: List[str] = []
    for e in entities:
        prof = propagate_entity_uncertainty(world, e.id)
        breakdown[prof.existence.state.value] += 1
        if prof.existence.state is EpistemicState.UNKNOWN:
            unknown_gaps.append(f"{e.id}: existence provenance/confidence is unknown")
        elif prof.geometry.state is EpistemicState.UNKNOWN:
            unknown_gaps.append(f"{e.id}: missing 3D geometry")

    for v in constraints:
        if v.severity == "warning":
            unknown_gaps.append(f"{v.code}: {v.message}")

    return WorldCompletenessReport(
        entity_count=n,
        geometric_coverage=round(with_geom / n, 4),
        surface_completeness=round(surfaces_observed / len(surfaces), 4) if surfaces else 0.0,
        entity_detection_coverage=round(with_evidence / n, 4),
        topological_completeness=round(with_topo / n, 4),
        semantic_completeness=round(with_semantic / n, 4),
        evidence_coverage=round(with_evidence / n, 4),
        uncertainty_breakdown=breakdown,
        unresolved_contradictions=contradictions,
        unknown_gaps=unknown_gaps[:50],
    )
