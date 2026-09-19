"""Tests for room inference and promotion (evidence/promote_rooms.py).

Fixtures are plane summaries derived from the same hand-computable
synthetic room family as tests/test_geometric_reasoning.py, so every
expected value -- ring vertices, area, dimensions, height, relationship
counts -- is closed-form, not "is a float". Includes the wall-base-ring
pitfall (a horizontal plane through the wall bases with the room's full
extent but no interior support) that the walkable-floor check exists
for.
"""

from __future__ import annotations

import math

import pytest

from evidence.promote_planes import (
    positions_by_plane,
    promote_plane_to_entity,
)
from evidence.promote_rooms import (
    MIN_FLOOR_INTERIOR_INLIERS,
    MIN_WALLS,
    PlaneSummary,
    RoomInferenceError,
    detect_rooms,
    plane_summary_from,
    promote_room_to_entity,
)
from perception.geometry.orientation import classify_planes
from perception.geometry.planes import detect_planes
from provenance import Provenance
from reconstruction.backend.interface import ReconstructedPoint, ReconstructionResult
from world_ir import RelationshipKind, WorldIR


def _point(x, y, z, counter=[0]) -> ReconstructedPoint:
    counter[0] += 1
    return ReconstructedPoint(
        position=(x, y, z), track_id=f"pt-{counter[0]:05d}", source_evidence_ids=["ev-1"]
    )


_CAMS = [(1.25, 1.3, 1.125), (0.9, 1.2, 1.5), (1.6, 1.25, 0.8)]
_UP = (0.0, 1.0, 0.0)


def _two_room_scene() -> ReconstructionResult:
    """2.5x2.25x2 m room: four walls (L x=0, R x=2.5, B z=2.25, F z=0
    with a doorway gap x 0.5..1.5), a separate doorway-segment plane on
    the same z=0 line (must merge with F's boundary line), and an
    interior horizontal slab at y=1 (x 0.3..2.2, z 0.3..1.95) that
    classifies as a second floor and must NOT spawn a room. All grid
    steps divide the extents, so every corner is an exact grid point:
    hand-checkable ring 2.5x2.25, area 5.625 m^2, height 2 m. Cameras
    inside."""
    counter = [0]
    points = []
    for x_i in range(26):       # floor: x 0..2.5 step 0.1
        for z_i in range(16):   # z 0..2.25 step 0.15
            points.append(_point(x_i * 0.1, 0.0, z_i * 0.15, counter))
    for x_i in range(26):       # ceiling: y=2
        for z_i in range(16):
            points.append(_point(x_i * 0.1, 2.0, z_i * 0.15, counter))
    for y_i in range(11):       # L / R walls: x=0 / x=2.5, y 0..2 step 0.2, z 0..2.25 step 0.15
        for z_i in range(16):
            points.append(_point(0.0, y_i * 0.2, z_i * 0.15, counter))
            points.append(_point(2.5, y_i * 0.2, z_i * 0.15, counter))
    for x_i in range(26):       # B wall: z=2.25, x 0..2.5, y 0..2
        for y_i in range(11):
            points.append(_point(x_i * 0.1, y_i * 0.2, 2.25, counter))
    for x_i in range(26):       # F wall: z=0, x 0..2.5 EXCEPT the doorway x 0.5..1.5
        if 0.5 < x_i * 0.1 < 1.5:
            continue
        for y_i in range(11):
            points.append(_point(x_i * 0.1, y_i * 0.2, 0.0, counter))
    for x_i in range(9):        # doorway segment: z=0, x 0.5..1.5, y 0.2..1.8 (separate plane)
        for y_i in range(9):
            points.append(_point(0.5 + x_i * 0.125, 0.2 + y_i * 0.2, 0.0, counter))
    for x_i in range(20):       # slab: y=1, x 0.3..2.2, z 0.3..1.95 (0.1/0.15 steps... 19x12)
        for z_i in range(12):
            points.append(_point(0.3 + x_i * 0.1, 1.0, 0.3 + z_i * 0.15, counter))
    return ReconstructionResult(points=points, camera_poses=[], registration_status="success")


def _promote_planes(result, world) -> dict:
    """Detect, classify, and promote all wall/floor/ceiling planes.
    Returns {plane_id: entity_id}."""
    detected = detect_planes(result, seed=42)
    oriented = classify_planes(detected.planes, _CAMS, _UP)
    positions = positions_by_plane(result, oriented)
    ids = {}
    for plane in oriented:
        if plane.role == "unknown":
            continue
        entity_id = f"ent-{plane.plane.plane_id}"
        promote_plane_to_entity(
            plane, result, world, entity_id, other_planes=oriented,
            plane_positions=positions,
        )
        ids[plane.plane.plane_id] = entity_id
    return ids, oriented, positions


def _summaries(oriented, positions, entity_ids) -> list:
    summaries = []
    for plane in oriented:
        if plane.role == "unknown":
            continue
        summary = plane_summary_from(plane, positions[plane.plane.plane_id])
        object.__setattr__(summary, "entity_id", entity_ids[plane.plane.plane_id])
        summaries.append(summary)
    return summaries


def _full_pipeline():
    result = _two_room_scene()
    world = WorldIR()
    entity_ids, oriented, positions = _promote_planes(result, world)
    summaries = _summaries(oriented, positions, entity_ids)
    rooms = detect_rooms(summaries, _UP)
    return world, rooms, summaries


# ---------------------------------------------------------------- detection


class TestDetectRooms:
    def test_finds_the_closed_room(self):
        _, rooms, _ = _full_pipeline()
        detected = [r for r in rooms if r.status == "detected"]
        assert len(detected) == 1
        room = detected[0]
        assert room.floor.role == "floor"
        assert room.ceiling is not None
        wall_ids = {w.plane_id for w in room.walls}
        # L, R, B, and the doorway-line wall (F's line with the doorway
        # segment merged in) bound the room: four boundary lines.
        assert len(wall_ids) == 4

    def test_ring_is_the_hand_computable_footprint(self):
        _, rooms, _ = _full_pipeline()
        room = next(r for r in rooms if r.status == "detected")
        area = room.floor_area_m2()
        dim_u, dim_v = room.floor_dimensions_m()
        # Ring: (0,0)-(2.5,0)-(2.5,2.25)-(0,2.25) in (x, z).
        assert area == pytest.approx(5.625)
        assert (dim_u, dim_v) == (pytest.approx(2.5), pytest.approx(2.25))

    def test_height_is_floor_to_wall_top(self):
        _, rooms, _ = _full_pipeline()
        room = next(r for r in rooms if r.status == "detected")
        assert room.height_m() == pytest.approx(2.0)

    def test_horizontal_slab_makes_no_room(self):
        """The pitfall: a horizontal slab at camera-waist height
        classifies as a floor (cameras above it). No walls rest on it,
        so it must yield an honest failed candidate, never a room."""
        _, rooms, _ = _full_pipeline()
        detected = [r for r in rooms if r.status == "detected"]
        assert len(detected) == 1
        slab_candidates = [r for r in rooms if r.status != "detected"]
        assert slab_candidates, "the slab candidate should be reported, not hidden"
        assert all(r.ring is None for r in slab_candidates)

    def test_two_walls_cannot_make_a_room(self):
        summaries = [
            PlaneSummary(
                plane_id="floor", role="floor", normal=(0.0, 1.0, 0.0), d=0.0,
                inlier_rms_m=0.005,
                bounds_min=(0.0, 0.0, 0.0), bounds_max=(4.0, 0.0, 4.0),
                inlier_positions=((1.0, 0.0, 1.0), (2.0, 0.0, 2.0), (3.0, 0.0, 3.0)),
                entity_id="ent-floor",
            ),
            PlaneSummary(
                plane_id="w1", role="wall", normal=(1.0, 0.0, 0.0), d=0.0,
                inlier_rms_m=0.005,
                bounds_min=(0.0, 0.0, 0.0), bounds_max=(0.0, 2.5, 4.0),
                inlier_positions=((0.0, 1.0, 1.0),), entity_id="ent-w1",
            ),
            PlaneSummary(
                plane_id="w2", role="wall", normal=(0.0, 0.0, 1.0), d=0.0,
                inlier_rms_m=0.005,
                bounds_min=(0.0, 0.0, 0.0), bounds_max=(4.0, 2.5, 0.0),
                inlier_positions=((1.0, 1.0, 0.0),), entity_id="ent-w2",
            ),
        ]
        rooms = detect_rooms(summaries, _UP)
        assert len(rooms) == 1
        assert rooms[0].status == "NO_CLOSED_RING"
        assert rooms[0].ring is None
        assert any(str(MIN_WALLS) in n for n in rooms[0].notes)

    def test_open_facade_is_unclosed_not_a_room(self):
        """Three walls + a gap: segments exist, but the ring cannot close."""
        summaries = [
            PlaneSummary(
                plane_id="floor", role="floor", normal=(0.0, 1.0, 0.0), d=0.0,
                inlier_rms_m=0.005,
                bounds_min=(0.0, 0.0, 0.0), bounds_max=(4.0, 0.0, 4.0),
                inlier_positions=tuple(
                    (x, 0.0, z) for x in (1.0, 2.0, 3.0) for z in (1.0, 2.0, 3.0)
                ),
                entity_id="ent-floor",
            ),
        ]
        # Three walls sharing corners on two sides only -- an open U.
        # w1: x=0 (z 0..4), w2: z=4 (x 0..4), w3: x=4 (z 0..4). The ring
        # dies at z=0 where no fourth wall exists.
        for pid, normal, bmin, bmax in [
            ("w1", (1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 2.5, 4.0)),
            ("w2", (0.0, 0.0, -1.0), (0.0, 0.0, 4.0), (4.0, 2.5, 4.0)),
            ("w3", (-1.0, 0.0, 0.0), (4.0, 0.0, 0.0), (4.0, 2.5, 4.0)),
        ]:
            summaries.append(PlaneSummary(
                plane_id=pid, role="wall", normal=normal, d=0.0,
                inlier_rms_m=0.005, bounds_min=bmin, bounds_max=bmax,
                inlier_positions=((bmin[0] + 0.1, 1.0, bmin[2] + 0.1),),
                entity_id=f"ent-{pid}",
            ))
        rooms = detect_rooms(summaries, _UP)
        assert rooms[0].status == "UNCLOSED_RING"
        assert rooms[0].ring is None

    def test_floating_walls_are_excluded_with_notes(self):
        summaries = [
            PlaneSummary(
                plane_id="floor", role="floor", normal=(0.0, 1.0, 0.0), d=0.0,
                inlier_rms_m=0.005,
                bounds_min=(0.0, 0.0, 0.0), bounds_max=(4.0, 0.0, 4.0),
                inlier_positions=((1.0, 0.0, 1.0), (2.0, 0.0, 2.0), (3.0, 0.0, 3.0)),
                entity_id="ent-floor",
            ),
            PlaneSummary(
                plane_id="loft", role="wall", normal=(1.0, 0.0, 0.0), d=0.0,
                inlier_rms_m=0.005,
                bounds_min=(0.0, 1.5, 0.0), bounds_max=(0.0, 2.5, 4.0),
                inlier_positions=((0.0, 2.0, 1.0),), entity_id="ent-loft",
            ),
        ]
        rooms = detect_rooms(summaries, _UP)
        assert rooms[0].status == "NO_CLOSED_RING"
        assert any("does not rest on this floor" in n for n in rooms[0].notes)

    def test_degenerate_up_raises(self):
        with pytest.raises(RoomInferenceError, match="up vector"):
            detect_rooms([], (0.0, 0.0, 0.0))

    def test_deterministic_across_calls(self):
        _, rooms_a, _ = _full_pipeline()
        _, rooms_b, _ = _full_pipeline()
        assert [(r.room_id, r.status, [w.plane_id for w in r.walls]) for r in rooms_a] == [
            (r.room_id, r.status, [w.plane_id for w in r.walls]) for r in rooms_b
        ]


# ---------------------------------------------------------------- promotion


class TestPromoteRoom:
    def _detected_room(self):
        _, rooms, _ = _full_pipeline()
        return next(r for r in rooms if r.status == "detected")

    def test_promotion_writes_entity_with_correct_type_and_provenance(self):
        world, _, _ = _full_pipeline()
        room = self._detected_room()
        result = promote_room_to_entity(room, world, "room-001", "Main Room")
        assert result.entity.type.value == "room"
        assert result.entity.provenance is Provenance.INFERRED
        assert world.entities["room-001"] is result.entity

    def test_contains_and_part_of_edges_are_written_both_ways(self):
        world, _, _ = _full_pipeline()
        room = self._detected_room()
        result = promote_room_to_entity(room, world, "room-001")
        part_ids = set(result.contains_ids)
        assert len(part_ids) == 6  # floor + 4 walls + ceiling
        # CONTAINS on the room...
        contains = {
            r.target_id for r in result.entity.relationships
            if r.kind is RelationshipKind.CONTAINS
        }
        assert contains == part_ids
        # ...and PART_OF on every part, pointing back.
        for pid in part_ids:
            back = {
                r.target_id for r in world.entities[pid].relationships
                if r.kind is RelationshipKind.PART_OF
            }
            assert "room-001" in back
        # Every edge is INFERRED with derivation metadata.
        for r in result.entity.relationships:
            assert r.provenance is Provenance.INFERRED
            assert r.metadata["derived_from"] == "wall_floor_ring_closure"

    def test_measurements_are_hand_computable(self):
        world, _, _ = _full_pipeline()
        room = self._detected_room()
        result = promote_room_to_entity(room, world, "room-001")
        by_name = dict(result.measurements)
        assert by_name["floor_area_m2"].value == pytest.approx(5.625)
        dims = sorted([by_name["floor_dimension_u_m"].value, by_name["floor_dimension_v_m"].value])
        assert dims == [pytest.approx(2.25), pytest.approx(2.5)]
        assert by_name["height_m"].value == pytest.approx(2.0)
        for m in by_name.values():
            assert m.unit.startswith("meter")
            assert m.provenance is Provenance.ESTIMATED
            assert 0.0 < m.confidence <= 1.0
            assert m.precision > 0.0

    def test_measurements_also_land_in_custom_properties(self):
        world, _, _ = _full_pipeline()
        room = self._detected_room()
        result = promote_room_to_entity(room, world, "room-001")
        assert result.entity.custom_properties["floor_area_m2"] == pytest.approx(5.625)

    def test_observation_records_the_derivation(self):
        world, _, _ = _full_pipeline()
        room = self._detected_room()
        result = promote_room_to_entity(room, world, "room-001")
        obs = result.entity.observations[0]
        assert obs.sensor_type == "room_inference"
        assert obs.metadata["boundary_vertex_count"] == 4
        assert obs.metadata["ceiling_plane_id"] is not None

    def test_refuses_unclosed_candidates(self):
        _, rooms, _ = _full_pipeline()
        world = WorldIR()
        unclosed = next(r for r in rooms if r.status != "detected")
        with pytest.raises(RoomInferenceError, match="refusing to promote"):
            promote_room_to_entity(unclosed, world, "room-bad")

    def test_refuses_when_plane_entities_are_missing(self):
        room = self._detected_room()
        empty_world = WorldIR()
        with pytest.raises(RoomInferenceError, match="not present in WorldIR"):
            promote_room_to_entity(room, empty_world, "room-001")

    def test_world_round_trips_through_serialization(self):
        world, _, _ = _full_pipeline()
        room = self._detected_room()
        promote_room_to_entity(room, world, "room-001")
        data = world.to_dict()
        restored = WorldIR.from_dict(data)
        assert restored.entities["room-001"].custom_properties == (
            world.entities["room-001"].custom_properties
        )
        kinds_restored = {
            (r.kind.value, r.target_id)
            for r in restored.entities["room-001"].relationships
        }
        kinds_original = {
            (r.kind.value, r.target_id)
            for r in world.entities["room-001"].relationships
        }
        assert kinds_restored == kinds_original

    def test_scene_graph_query_answers_whats_inside(self):
        """End-to-end: the promoted room answers 'what does this room
        contain?' from WorldIR relationships alone."""
        from engine.scene_graph.graph import SceneGraph

        world, _, _ = _full_pipeline()
        room = self._detected_room()
        promote_room_to_entity(room, world, "room-001")
        graph = SceneGraph(world)
        contents = graph.contents_of("room-001")
        assert len(contents) == 6
