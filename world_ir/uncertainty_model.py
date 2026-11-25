"""Component-wise uncertainty representation and propagation for WorldIR entities.

Distinguishes epistemic states without ever converting UNKNOWN into 0.0 or 1.0:
  - DIRECTLY_OBSERVED
  - GEOMETRICALLY_INFERRED
  - SEMANTICALLY_INFERRED
  - WEAKLY_SUPPORTED
  - CONFLICTING
  - UNKNOWN
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
from typing import Any, Dict, Optional

from provenance import Provenance
from world_ir.observations import AssociationState, ledger_of
from world_ir.schema_v1 import Entity
from world_ir.world_v1 import WorldIR


class EpistemicState(str, Enum):
    DIRECTLY_OBSERVED = "directly_observed"
    GEOMETRICALLY_INFERRED = "geometrically_inferred"
    SEMANTICALLY_INFERRED = "semantically_inferred"
    WEAKLY_SUPPORTED = "weakly_supported"
    CONFLICTING = "conflicting"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class UncertaintyFacet:
    state: EpistemicState
    confidence: Optional[float]  # None when unknown; never fabricated
    basis: str


@dataclass(frozen=True)
class EntityUncertaintyProfile:
    entity_id: str
    existence: UncertaintyFacet
    position: UncertaintyFacet
    dimensions: UncertaintyFacet
    orientation: UncertaintyFacet
    geometry: UncertaintyFacet
    topology: UncertaintyFacet
    semantic: UncertaintyFacet

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "existence": {**asdict(self.existence), "state": self.existence.state.value},
            "position": {**asdict(self.position), "state": self.position.state.value},
            "dimensions": {**asdict(self.dimensions), "state": self.dimensions.state.value},
            "orientation": {**asdict(self.orientation), "state": self.orientation.state.value},
            "geometry": {**asdict(self.geometry), "state": self.geometry.state.value},
            "topology": {**asdict(self.topology), "state": self.topology.state.value},
            "semantic": {**asdict(self.semantic), "state": self.semantic.state.value},
        }


def _clean_conf(val: Any) -> Optional[float]:
    if isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(float(val)):
        return max(0.0, min(1.0, float(val)))
    return None


def _has_conflict(world: WorldIR, entity_id: str) -> bool:
    _, decisions = ledger_of(world)
    for dec in decisions:
        if dec.entity_id == entity_id and dec.state is AssociationState.AMBIGUOUS:
            return True
        if any(cid == entity_id for cid, _ in dec.candidates) and dec.state is AssociationState.AMBIGUOUS:
            return True
    ent = world.entities.get(entity_id)
    if ent is not None:
        cp = ent.custom_properties or {}
        if cp.get("has_conflict") or cp.get("conflict"):
            return True
    return False


def _base_state(ent: Entity, has_conflict: bool) -> EpistemicState:
    if has_conflict:
        return EpistemicState.CONFLICTING
    conf = _clean_conf(ent.confidence)
    prov = ent.provenance
    if prov is Provenance.OBSERVED:
        return EpistemicState.WEAKLY_SUPPORTED if (conf is not None and conf < 0.5) else EpistemicState.DIRECTLY_OBSERVED
    if prov is Provenance.RECONSTRUCTED:
        if len(ent.observations) == 0 and not (ent.custom_properties or {}).get("observation_ids"):
            return EpistemicState.GEOMETRICALLY_INFERRED
        return EpistemicState.WEAKLY_SUPPORTED if (conf is not None and conf < 0.5) else EpistemicState.DIRECTLY_OBSERVED
    if prov in (Provenance.INFERRED, Provenance.GENERATED):
        return EpistemicState.GEOMETRICALLY_INFERRED if ent.geometry_ids else EpistemicState.SEMANTICALLY_INFERRED
    return EpistemicState.UNKNOWN


def propagate_entity_uncertainty(world: WorldIR, entity_id: str) -> EntityUncertaintyProfile:
    """Compute component-wise uncertainty for an entity, propagating upstream uncertainty
    from linked geometries, observations, and host/parent entities without inventing numbers."""
    ent = world.entities[entity_id]
    conflict = _has_conflict(world, entity_id)
    base = _base_state(ent, conflict)
    ent_conf = _clean_conf(ent.confidence)

    # Propagate upstream geometry confidence
    geom_confs = [
        _clean_conf(world.geometries[gid].confidence)
        for gid in (ent.geometry_ids or ())
        if gid in world.geometries
    ]
    valid_geom_confs = [c for c in geom_confs if c is not None]
    geom_conf = min(valid_geom_confs) if valid_geom_confs else ent_conf
    if not ent.geometry_ids:
        geom_facet = UncertaintyFacet(EpistemicState.UNKNOWN, None, "no geometry attached")
    else:
        g_state = EpistemicState.CONFLICTING if conflict else (
            EpistemicState.UNKNOWN if geom_conf is None else base
        )
        geom_facet = UncertaintyFacet(g_state, geom_conf, "derived from attached geometry")

    # Propagate upstream parent/host confidence if present
    parent_id = getattr(ent, "parent_id", None)
    parent_conf = _clean_conf(world.entities[parent_id].confidence) if parent_id and parent_id in world.entities else None
    upstream_confs = [c for c in (ent_conf, geom_conf, parent_conf) if c is not None]
    pos_conf = min(upstream_confs) if upstream_confs else None

    if ent.transform is None and not ent.geometry_ids:
        pos_facet = UncertaintyFacet(EpistemicState.UNKNOWN, None, "no transform or geometry")
        ori_facet = UncertaintyFacet(EpistemicState.UNKNOWN, None, "no orientation recorded")
    else:
        pos_facet = UncertaintyFacet(
            EpistemicState.CONFLICTING if conflict else (EpistemicState.UNKNOWN if pos_conf is None else base),
            pos_conf,
            "propagated from entity and upstream geometry/parent",
        )
        ori_facet = UncertaintyFacet(
            EpistemicState.UNKNOWN if ent.transform is None or pos_conf is None else base,
            pos_conf if ent.transform is not None else None,
            "derived from entity transform" if ent.transform is not None else "orientation not recorded",
        )

    # Dimensions facet
    cp = ent.custom_properties or {}
    dims = cp.get("dimensions") if isinstance(cp.get("dimensions"), dict) else {}
    if dims:
        dim_states = [str(v.get("state", "")).upper() for v in dims.values() if isinstance(v, dict)]
        if any(s == "VERIFIED" for s in dim_states):
            dim_facet = UncertaintyFacet(EpistemicState.DIRECTLY_OBSERVED, pos_conf, "verified user/metric dimensions")
        elif any(s == "ESTIMATED" for s in dim_states):
            dim_facet = UncertaintyFacet(EpistemicState.SEMANTICALLY_INFERRED, pos_conf, "estimated dimensions")
        else:
            dim_facet = UncertaintyFacet(base, pos_conf, "recorded dimensions")
    elif ent.geometry_ids:
        dim_facet = UncertaintyFacet(geom_facet.state, geom_conf, "from geometry bounds")
    else:
        dim_facet = UncertaintyFacet(EpistemicState.UNKNOWN, None, "no dimensions recorded")

    # Topology facet
    if ent.relationships:
        rel_confs = [_clean_conf(r.confidence) for r in ent.relationships]
        valid_rel = [c for c in rel_confs if c is not None]
        topo_conf = min(valid_rel) if valid_rel else ent_conf
        topo_facet = UncertaintyFacet(
            EpistemicState.CONFLICTING if conflict else (EpistemicState.UNKNOWN if topo_conf is None else base),
            topo_conf,
            f"{len(ent.relationships)} spatial relationship(s)",
        )
    else:
        topo_facet = UncertaintyFacet(EpistemicState.UNKNOWN, None, "no spatial relationships recorded")

    # Semantic facet
    if ent.semantic_labels:
        is_fallback = any("unknown" in lbl.lower() or "fallback" in lbl.lower() for lbl in ent.semantic_labels)
        sem_state = EpistemicState.WEAKLY_SUPPORTED if is_fallback else (
            EpistemicState.SEMANTICALLY_INFERRED if ent.provenance in (Provenance.INFERRED, Provenance.GENERATED) else base
        )
        sem_facet = UncertaintyFacet(sem_state, ent_conf, ", ".join(ent.semantic_labels[:3]))
    else:
        sem_facet = UncertaintyFacet(EpistemicState.UNKNOWN, None, "no semantic labels")

    existence_facet = UncertaintyFacet(base, ent_conf, f"provenance={ent.provenance.value}")

    return EntityUncertaintyProfile(
        entity_id=entity_id,
        existence=existence_facet,
        position=pos_facet,
        dimensions=dim_facet,
        orientation=ori_facet,
        geometry=geom_facet,
        topology=topo_facet,
        semantic=sem_facet,
    )
