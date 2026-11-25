"""Spatial reasoning, WorldIR geometries, constraints, uncertainty, and completeness benchmarks."""

from __future__ import annotations

import pytest

from provenance import Provenance
from world_ir.completeness import assess_world_completeness
from world_ir.constraints import check_architectural_constraints
from world_ir.corrections import merge_entities, split_entity
from world_ir.observations import AssociationDecision, AssociationState, attach_ledger
from world_ir.schema_v1 import Entity, EntityType, Geometry, GeometryType, Observation, Relationship, RelationshipKind, Vector3
from world_ir.uncertainty_model import EpistemicState, propagate_entity_uncertainty
from world_ir.validation import validate_world_ir
from world_ir.world_v1 import WorldIR
from worldstore.store import WorldStore


def _make_wall(eid: str, x1: float, y1: float, x2: float, y2: float, *, height: float = 2.8, conf: float | None = 0.9, prov: Provenance = Provenance.OBSERVED):
    lo = Vector3(min(x1, x2) - 0.05, min(y1, y2) - 0.05, 0.0)
    hi = Vector3(max(x1, x2) + 0.05, max(y1, y2) + 0.05, height)
    geom = Geometry(id=f"{eid}_geom", type=GeometryType.BOX, bounds_min=lo, bounds_max=hi, provenance=prov, confidence=conf)
    ent = Entity(
        id=eid,
        type=EntityType.WALL,
        name=eid,
        geometry_ids=[geom.id],
        provenance=prov,
        confidence=conf,
        semantic_labels=["architecture/wall"],
        observations=[Observation(id=f"obs_{eid}", sensor_type="camera")] if prov in (Provenance.OBSERVED, Provenance.RECONSTRUCTED) else [],
        custom_properties={"wall_slab": {"start_xyz_m": [x1, y1, 0.0], "end_xyz_m": [x2, y2, 0.0]}},
    )
    return ent, geom


def test_split_and_merge_preserve_world_geometries_and_worldstore_roundtrip(tmp_path):
    world = WorldIR("bench_geom_world")
    w1, g1 = _make_wall("wall_1", 0.0, 0.0, 4.0, 0.0, conf=0.85)
    world.entities[w1.id] = w1
    world.geometries[g1.id] = g1

    # Split along X: must create two real geometries in world.geometries and remove orphan g1
    e1, e2, _ = split_entity(world, "wall_1", split_axis="x", split_ratio=0.5, note="split test")
    assert "wall_1" not in world.entities
    assert g1.id not in world.geometries
    assert len(e1.geometry_ids) == 1 and e1.geometry_ids[0] in world.geometries
    assert len(e2.geometry_ids) == 1 and e2.geometry_ids[0] in world.geometries

    g_part1 = world.geometries[e1.geometry_ids[0]]
    g_part2 = world.geometries[e2.geometry_ids[0]]
    assert g_part1.bounds_max.x == pytest.approx(g_part2.bounds_min.x, abs=1e-6)
    assert validate_world_ir(world).is_valid()

    # Merge back: must update world.geometries bounds to union
    merged, _ = merge_entities(world, e1.id, [e2.id], note="re-merge")
    mg = world.geometries[merged.geometry_ids[0]]
    assert mg.bounds_min.x == pytest.approx(-0.05, abs=1e-6)
    assert mg.bounds_max.x == pytest.approx(4.05, abs=1e-6)

    # WorldStore save + reload roundtrip
    store = WorldStore(str(tmp_path / "ws"))
    sv = store.save_version(world, parent=None)
    reloaded = store.load_version(sv.version_id)
    assert merged.id in reloaded.entities
    assert mg.id in reloaded.geometries


def test_irregular_non_manhattan_room_enclosure_and_incomplete_room():
    # Closed triangular (non-Manhattan) room
    w_closed = WorldIR("tri_room")
    walls = [
        _make_wall("w1", 0.0, 0.0, 4.0, 0.0),
        _make_wall("w2", 4.0, 0.0, 2.0, 3.0),
        _make_wall("w3", 2.0, 3.0, 0.0, 0.0),
    ]
    for e, g in walls:
        w_closed.entities[e.id] = e
        w_closed.geometries[g.id] = g
    room = Entity(
        id="room_tri",
        type=EntityType.ROOM,
        name="Triangular Room",
        provenance=Provenance.RECONSTRUCTED,
        confidence=0.8,
        custom_properties={"boundary_element_ids": ["w1", "w2", "w3"]},
    )
    w_closed.entities[room.id] = room
    v_closed = check_architectural_constraints(w_closed)
    assert not any(v.code == "room_enclosure_incomplete" for v in v_closed)

    # Remove one wall -> incomplete enclosure warning (not an error; remains representable)
    del w_closed.entities["w3"]
    del w_closed.geometries["w3_geom"]
    room.custom_properties["boundary_element_ids"] = ["w1", "w2"]
    v_open = check_architectural_constraints(w_closed)
    assert any(v.code == "room_enclosure_incomplete" and v.severity == "warning" for v in v_open)


def test_contradictory_cycle_and_wall_gap_detection():
    world = WorldIR("cycle_gap")
    w1, g1 = _make_wall("w1", 0.0, 0.0, 2.0, 0.0)
    w2, g2 = _make_wall("w2", 2.25, 0.0, 4.0, 0.0)  # 0.25m gap (> 0.05m tol, <= 0.5m)
    w1.relationships.append(Relationship(RelationshipKind.CONTAINS, "w2"))
    w2.relationships.append(Relationship(RelationshipKind.CONTAINS, "w1"))
    world.entities[w1.id] = w1
    world.entities[w2.id] = w2
    world.geometries[g1.id] = g1
    world.geometries[g2.id] = g2

    violations = check_architectural_constraints(world)
    codes = {v.code for v in violations}
    assert "contradictory_relationship" in codes
    assert "wall_connectivity_gap" in codes


def test_uncertainty_propagation_preserves_unknown_and_detects_conflicts():
    world = WorldIR("unc_world")
    w_obs, g_obs = _make_wall("w_obs", 0.0, 0.0, 3.0, 0.0, conf=0.9, prov=Provenance.OBSERVED)
    w_unk = Entity(id="w_unk", type=EntityType.WALL, name="Unknown Wall", provenance=Provenance.UNKNOWN, confidence=None)
    w_conf, g_conf = _make_wall("w_conf", 0.0, 1.0, 3.0, 1.0, conf=0.7, prov=Provenance.RECONSTRUCTED)

    world.entities = {w_obs.id: w_obs, w_unk.id: w_unk, w_conf.id: w_conf}
    world.geometries = {g_obs.id: g_obs, g_conf.id: g_conf}
    attach_ledger(
        world,
        (),
        [
            AssociationDecision(
                observation_id="obs_x",
                state=AssociationState.AMBIGUOUS,
                entity_id="w_conf",
                probability=0.5,
                reason="Conflicting depth planes",
                candidates=(("w_conf", 0.5), ("w_other", 0.5)),
            )
        ],
    )

    p_obs = propagate_entity_uncertainty(world, "w_obs")
    p_unk = propagate_entity_uncertainty(world, "w_unk")
    p_conf = propagate_entity_uncertainty(world, "w_conf")

    assert p_obs.existence.state is EpistemicState.DIRECTLY_OBSERVED
    assert p_unk.existence.state is EpistemicState.UNKNOWN
    assert p_unk.existence.confidence is None  # NEVER converted to 0.0 or 1.0
    assert p_unk.geometry.state is EpistemicState.UNKNOWN
    assert p_conf.existence.state is EpistemicState.CONFLICTING


def test_multi_dimensional_completeness_benchmark_against_baseline():
    # Empty baseline
    empty_report = assess_world_completeness(WorldIR("empty"))
    assert empty_report.geometric_coverage == 0.0
    assert empty_report.unknown_gaps

    # Partial vs complete room benchmark
    world = WorldIR("complete_room")
    for idx, (x1, y1, x2, y2) in enumerate([
        (0.0, 0.0, 4.0, 0.0),
        (4.0, 0.0, 4.0, 3.0),
        (4.0, 3.0, 0.0, 3.0),
        (0.0, 3.0, 0.0, 0.0),
    ], start=1):
        e, g = _make_wall(f"w{idx}", x1, y1, x2, y2, conf=0.9)
        e.relationships.append(Relationship(RelationshipKind.PART_OF, "room_1"))
        world.entities[e.id] = e
        world.geometries[g.id] = g
    world.entities["room_1"] = Entity(
        id="room_1",
        type=EntityType.ROOM,
        name="Living Room",
        provenance=Provenance.RECONSTRUCTED,
        confidence=0.88,
        semantic_labels=["space/living_room"],
        observations=[Observation(id="obs_r1", sensor_type="camera")],
        custom_properties={"boundary_element_ids": ["w1", "w2", "w3", "w4"]},
    )

    rep = assess_world_completeness(world)
    assert rep.surface_completeness == 1.0
    assert rep.semantic_completeness == 1.0
    assert rep.evidence_coverage == 1.0
    assert rep.unresolved_contradictions == []
