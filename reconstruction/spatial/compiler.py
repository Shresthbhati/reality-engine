"""Spatial compiler: observations -> association -> constraints -> topology -> completeness -> WorldIR.

`compile_spatial` is pure (no I/O). `to_worldir` maps the result into the canonical WorldIR without adding
anything the evidence did not supply:
  * only ESTABLISHED entities (>= `min_images_to_establish` views) are compiled unless `include_tentative`;
  * every entity carries its evidence ids and the full per-dimension belief table (status / sigma / confidence /
    support) under `custom_properties["spatial_beliefs"]`; confidence fields of the ConfidenceVector are filled
    only where a producer stated a probability, everything else stays None;
  * unknown wall height => the geometry is a plan footprint at the storey floor and `z_extent_known` is False;
    no height is invented;
  * the raw structural observations ride inside the world (`metadata["spatial_evidence"]`) so every evidence id
    an entity cites resolves after a store round trip;
  * contradictions, issues and the completeness report are stored with the world, not dropped.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from provenance import Provenance
from world_ir import (
    Entity,
    EntityType,
    Geometry,
    GeometryType,
    Relationship,
    RelationshipKind,
    Vector3,
    WorldIR,
)

from .association import AssociationConfig, AssociationResult, associate_structure
from .completeness import CompletenessReport, assess_completeness
from .constraints import (
    ConstraintConfig,
    ConstraintReport,
    evaluate_alignment,
    solve_spatial_constraints,
)
from .model import Kind, SpatialEntity, StructuralObservation
from .refinement import RefinementReport, refine_walls_with_points
from .topology import TopologyReport, validate_topology
from .uncertainty import beliefs_from_dict, beliefs_to_dict, confidence_vector, provenance_for

_ETYPE = {Kind.WALL: EntityType.WALL, Kind.DOOR: EntityType.DOOR, Kind.WINDOW: EntityType.WINDOW,
          Kind.OPENING: EntityType.OPENING, Kind.STAIR: EntityType.STAIRS}
SPATIAL_EVIDENCE_KEY = "spatial_evidence"
SPATIAL_REPORT_KEY = "spatial_compile_report"


@dataclass
class SpatialCompileResult:
    observations: list[StructuralObservation]
    association: AssociationResult
    constraints: ConstraintReport
    topology: TopologyReport
    completeness: CompletenessReport
    refinement: RefinementReport | None
    entities: list[SpatialEntity]                  # refined; includes tentative ones
    storeys: dict[str, float]

    def compiled_entities(self, include_tentative: bool = False) -> list[SpatialEntity]:
        k = self.association.config.min_images_to_establish
        return [e for e in self.entities if include_tentative or len(e.image_ids) >= k]


def compile_spatial(
    observations: Sequence[StructuralObservation], *, storeys: Mapping[str, float] | None = None,
    points_xyz: np.ndarray | None = None, point_evidence: Sequence[Sequence[str]] | None = None,
    frame_transforms: Mapping[str, tuple[float, float, float]] | None = None, target_frame: str = "world",
    assoc_cfg: AssociationConfig = AssociationConfig(), constraint_cfg: ConstraintConfig = ConstraintConfig(),
    reference_matches: tuple[int, int] | None = None, impose_junctions: bool = True,
) -> SpatialCompileResult:
    assoc = associate_structure(observations, assoc_cfg, target_frame=target_frame, frame_transforms=frame_transforms)
    ents = assoc.entities
    refinement = None
    extra = list(assoc.conflicts)
    if points_xyz is not None and len(points_xyz):
        refinement = refine_walls_with_points(ents, points_xyz, point_evidence)
        ents = refinement.entities
        extra += refinement.conflicts
    # only established entities shape the solution: a single-view guess must not pull an established wall
    k = assoc_cfg.min_images_to_establish
    est = [e for e in ents if len(e.image_ids) >= k]
    tent = [e for e in ents if len(e.image_ids) < k]
    cons = solve_spatial_constraints(est, constraint_cfg, storeys=storeys, conflicts=assoc.conflicts,
                                   impose_junctions=impose_junctions)
    refined = cons.entities + tent
    topo = validate_topology(refined, cons, extra, known_evidence_ids={o.evidence_id for o in observations},
                             cfg=constraint_cfg)
    cons.constraints += evaluate_alignment(cons.entities, [list(r.wall_ids) for r in topo.rooms], constraint_cfg)
    comp = assess_completeness(assoc, refined, topo, n_observations=len(observations), points_xyz=points_xyz,
                               reference_matches=reference_matches)
    return SpatialCompileResult(list(observations), assoc, cons, topo, comp, refinement, refined, dict(storeys or {}))


# ----------------------------------------------------------------------------------------------------- WorldIR

def _prov_rel(kind: RelationshipKind, target: str, conf: float | None, note: str) -> Relationship:
    return Relationship(kind=kind, target_id=target, confidence=conf, provenance=Provenance.INFERRED,
                        metadata={"derived_from": note})


def to_worldir(result: SpatialCompileResult, world: WorldIR | None = None, *, name: str = "spatial-compiled",
               include_tentative: bool = False) -> WorldIR:
    w = world if world is not None else WorldIR(name=name)
    ents = result.compiled_entities(include_tentative)
    keep = {e.entity_id for e in ents}
    for e in ents:
        floor = result.storeys.get(e.storey, 0.0) if e.storey else 0.0
        z0 = e.z0 if e.z0 is not None else floor
        z1 = e.z1 if e.z1 is not None else z0
        xs, ys = (e.a[0], e.b[0]), (e.a[1], e.b[1])
        gid = f"geom-{e.entity_id}"
        w.geometries[gid] = Geometry(
            id=gid, type=GeometryType.PLANE,
            bounds_min=Vector3(min(xs), min(ys), min(z0, z1)), bounds_max=Vector3(max(xs), max(ys), max(z0, z1)),
            provenance=Provenance.ESTIMATED, confidence=None,
            quality_metrics={"plan_segment": [list(e.a), list(e.b)], "z_extent_known": e.z0 is not None and e.z1 is not None,
                             "end_a_observed": e.end_a_observed, "end_b_observed": e.end_b_observed})
        props = {
            "plan_segment": [list(e.a), list(e.b)], "z_extent_known": e.z0 is not None and e.z1 is not None,
            "spatial_beliefs": beliefs_to_dict(e.beliefs), "evidence_ids": list(e.evidence_ids),
            "image_ids": list(e.image_ids), "sigma_a_m": e.sigma_a, "sigma_b_m": e.sigma_b,
            "end_a_observed": e.end_a_observed, "end_b_observed": e.end_b_observed,
            "storey_id": f"storey-{e.storey}" if e.storey else None, "alternatives": e.alternatives,
            "partially_visible": e.partially_visible, "occluded": e.occluded,
            "host_alternatives": list(e.host_alternatives), "sigma_parts": e.sigma_parts,
        }
        if e.host_id:
            props["host_wall_id"] = e.host_id
        rels = []
        if e.host_id and e.host_id in keep:
            rels.append(_prov_rel(RelationshipKind.ATTACHED_TO, e.host_id, None, "spatial_compiler.host_association"))
        w.entities[e.entity_id] = Entity(
            id=e.entity_id, type=_ETYPE[e.kind], name=e.entity_id, geometry_ids=[gid], relationships=rels,
            provenance=provenance_for(e), confidence=None,
            confidence_basis="per-dimension beliefs in custom_properties.spatial_beliefs; only producer-stated "
                             "probabilities fill confidence_vector", confidence_vector=confidence_vector(e),
            custom_properties=props)
    for wi, which, wj, typ in result.constraints.junctions:
        if wi in keep and wj in keep:
            w.entities[wi].relationships.append(_prov_rel(RelationshipKind.ADJACENT_TO, wj, None, f"junction:{typ}"))
    # storeys and rooms
    for s, z in sorted(result.storeys.items()):
        sid = f"storey-{s}"
        w.entities[sid] = Entity(id=sid, type=EntityType.STOREY, name=s, provenance=Provenance.INFERRED,
                                 custom_properties={"floor_height_m": z})
    for r in result.topology.rooms:
        if not set(r.wall_ids) <= keep:
            continue
        rid = r.room_id
        w.entities[rid] = Entity(
            id=rid, type=EntityType.ROOM, name=rid, provenance=Provenance.INFERRED, confidence=None,
            relationships=[_prov_rel(RelationshipKind.CONTAINS, x, None, "spatial_compiler.room_enclosure") for x in r.wall_ids],
            custom_properties={"boundary_element_ids": list(r.wall_ids), "polygon": [list(p) for p in r.polygon],
                               "floor_area_m2": r.area_m2, "floor_area_sigma_m2": r.area_sigma_m2,
                               "boundary_has_unobserved_end": r.boundary_has_unobserved_end,
                               "evidence_ids": list(r.evidence_ids), "storey_id": f"storey-{r.storey}" if r.storey else None})
        if r.storey and f"storey-{r.storey}" in w.entities:
            w.entities[f"storey-{r.storey}"].relationships.append(
                _prov_rel(RelationshipKind.CONTAINS, rid, None, "spatial_compiler.storey_membership"))
    # stairs connect the storeys their z-extent reaches, and the rooms their base / top points lie inside
    from .constraints import point_in_polygon
    for e in ents:
        if e.kind is not Kind.STAIR or e.z0 is None or e.z1 is None or len(result.storeys) < 2:
            continue
        lo = min(result.storeys, key=lambda s: abs(result.storeys[s] - e.z0))
        hi = min(result.storeys, key=lambda s: abs(result.storeys[s] - e.z1))
        if lo == hi:
            continue
        for sname, pt in ((lo, e.a), (hi, e.b)):
            if f"storey-{sname}" in w.entities:
                w.entities[e.entity_id].relationships.append(
                    _prov_rel(RelationshipKind.CONNECTS, f"storey-{sname}", None, "spatial_compiler.stair_z_extent"))
            for r in result.topology.rooms:
                if r.storey == sname and r.room_id in w.entities and point_in_polygon(pt, r.polygon):
                    w.entities[e.entity_id].relationships.append(
                        _prov_rel(RelationshipKind.CONNECTS, r.room_id, None, "spatial_compiler.stair_end_in_room"))
    for a, b, wall, ops in result.topology.room_adjacency:
        if a in w.entities and b in w.entities:
            w.entities[a].relationships.append(_prov_rel(RelationshipKind.ADJACENT_TO, b, None, f"shared_wall:{wall}"))
    w.metadata[SPATIAL_EVIDENCE_KEY] = [o.to_dict() for o in result.observations]
    w.metadata[SPATIAL_REPORT_KEY] = {
        "topology_status": result.topology.status(),
        "issues": [i.to_dict() for i in result.topology.issues],
        "constraints": [c.to_dict() for c in result.constraints.constraints],
        "contradictions": [c.to_dict() for c in result.constraints.contradictions],
        "association": [r.to_dict() for r in result.association.records],
        "completeness": result.completeness.to_dict(),
        "excluded_tentative": sorted(e.entity_id for e in result.entities if e.entity_id not in keep),
    }
    return w


def inspect_spatial_world(world: WorldIR) -> dict:
    """What a reader of a (re)loaded world can verify, with no access to the compile result."""
    ev_ids = {o["evidence_id"] for o in world.metadata.get(SPATIAL_EVIDENCE_KEY, [])}
    out = {"entities": {}, "dangling_relationships": [], "unresolved_evidence": [], "geometry_missing": [],
           "rooms": [], "n_entities": len(world.entities), "n_geometries": len(world.geometries)}
    for eid, e in sorted(world.entities.items()):
        cp = e.custom_properties or {}
        out["entities"][eid] = {"type": e.type.value, "provenance": e.provenance.value,
                                "beliefs": {k: v["status"] for k, v in (cp.get("spatial_beliefs") or {}).items()}}
        for r in e.relationships:
            if r.target_id not in world.entities:
                out["dangling_relationships"].append((eid, r.target_id))
        for g in e.geometry_ids:
            if g not in world.geometries:
                out["geometry_missing"].append((eid, g))
        for x in cp.get("evidence_ids", ()):
            if x not in ev_ids:
                out["unresolved_evidence"].append((eid, x))
        if e.type is EntityType.ROOM:
            out["rooms"].append({"id": eid, "area_m2": cp.get("floor_area_m2"), "sigma": cp.get("floor_area_sigma_m2"),
                                 "walls": cp.get("boundary_element_ids")})
        if cp.get("spatial_beliefs"):
            beliefs_from_dict(cp["spatial_beliefs"])   # must round-trip
    json.dumps(world.metadata.get(SPATIAL_REPORT_KEY, {}))
    out["topology_status"] = world.metadata.get(SPATIAL_REPORT_KEY, {}).get("topology_status")
    return out


def compile_multiview(local_observations, poses, *, storeys=None, mv_cfg=None, warm_start=None, impose_junctions: bool = True,
                      **kw):
    """Image-frame observations + prior poses -> joint optimization -> compile. Returns (compile result, MultiViewResult).

    Splits each wall's position uncertainty into its sources: `measurement` (raw observation sigmas only), `optimization`
    (what the pose uncertainty, propagated through the transform, adds), `model` (the un-modelled floor) and records how
    many of its observations were AMBIGUOUS in association. The parts describe one sigma budget, not a second estimate."""
    import math

    from .multiview import MultiViewConfig, optimize_multiview
    mv = optimize_multiview(local_observations, poses, mv_cfg or MultiViewConfig(), warm_start=warm_start)
    res = compile_spatial(mv.observations, storeys=storeys, impose_junctions=impose_junctions, **kw)
    ambiguous = {r.evidence_id for r in res.association.records if r.state == "AMBIGUOUS"}
    floor = res.association.config.offset_floor_m
    for e in res.entities:
        if e.kind is not Kind.WALL or e.beliefs["position"].sigma is None:
            continue
        raws = [mv.raw[x]["sigma_m"] for x in e.evidence_ids if x in mv.raw]
        meas = 1.0 / math.sqrt(sum(1.0 / (2 * r * r) for r in raws)) if raws else None   # 2 end points per observation
        total = e.beliefs["position"].sigma
        opt = math.sqrt(max(total ** 2 - meas ** 2, 0.0)) if meas is not None else None
        e.sigma_parts = {"measurement": meas, "optimization": opt, "model": floor,
                         "association_ambiguous_observations": float(len(ambiguous & set(e.evidence_ids)))}
    return res, mv
