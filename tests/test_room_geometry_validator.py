"""Workstream 2: coherent room geometry validation (``engine/validation/room_geometry.py``).

SYNTHETIC worlds only (hand-built boxes + the canonical interior compile). Every check has a positive case (violation
found), a negative case (clean room passes) and an "unknown when evidence is missing" case; the adversarial cases are the
ones a reviewer would try first: duplicate wall injected, gap in the loop, door off the wall, camera inside a wall,
floating furniture. Nothing here says anything about real-photo accuracy.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from engine.pipeline.stages.base import StageContext
from engine.pipeline.stages.registry import SegmentReport
from engine.pipeline.stages.s11_compile_validate import CompileValidate
from engine.validation.room_geometry import (
    CHECKS,
    validate_for_stage,
    validate_room_geometry,
)
from provenance import Provenance
from world_ir import WorldIR
from world_ir.schema_v1 import (
    Entity,
    EntityType,
    Geometry,
    GeometryType,
    Relationship,
    RelationshipKind,
    Vector3,
)

T = 0.1  # slab thickness of the synthetic walls


class Room:
    """A hand-built room (axis-aligned slab walls, floor, ceiling, footprint) in either a Y-up or a Z-up world."""

    def __init__(self, up: str = "y", scale: float = 1.0, metric: bool = False):
        self.up, self.k = up, scale
        self.world = WorldIR(id="w-geom", name="geom")
        self.world.metadata["compiled_from"] = {"up": [0.0, 1.0, 0.0] if up == "y" else [0.0, 0.0, 1.0]}
        if metric:
            self.world.metadata["scale"] = {"state": "metric"}

    def xyz(self, a, b, v):
        a, b, v = a * self.k, b * self.k, v * self.k
        return (a, v, b) if self.up == "y" else (a, b, v)

    def box(self, gid, lo, hi):
        """lo/hi are (a, b, v) triples in unscaled units."""
        p0, p1 = self.xyz(*lo), self.xyz(*hi)
        self.world.geometries[gid] = Geometry(
            id=gid, type=GeometryType.BOX,
            bounds_min=Vector3(*[min(x, y) for x, y in zip(p0, p1)]), bounds_max=Vector3(*[max(x, y) for x, y in zip(p0, p1)]))
        return gid

    def ent(self, eid, etype, lo=None, hi=None, rels=(), **props):
        e = Entity(id=eid, name=eid, type=etype, provenance=Provenance.OBSERVED,
                   geometry_ids=[self.box(f"g_{eid}", lo, hi)] if lo is not None else [],
                   relationships=list(rels), custom_properties=dict(props))
        self.world.entities[eid] = e
        return e

    def room(self, rid="room-A", a0=0.0, b0=0.0, a1=4.0, b1=3.0, floor=0.0, h=2.5, walls=("s", "n", "w", "e"),
             storey=None):
        wall_ids = []
        spec = {"s": ((a0, b0 - T / 2, floor), (a1, b0 + T / 2, floor + h)),
                "n": ((a0, b1 - T / 2, floor), (a1, b1 + T / 2, floor + h)),
                "w": ((a0 - T / 2, b0 - T / 2, floor), (a0 + T / 2, b1 + T / 2, floor + h)),
                "e": ((a1 - T / 2, b0 - T / 2, floor), (a1 + T / 2, b1 + T / 2, floor + h))}
        for side in walls:
            wid = f"{rid}-wall-{side}"
            self.ent(wid, EntityType.WALL, *spec[side], room_id=rid)
            wall_ids.append(wid)
        self.ent(f"{rid}-floor", EntityType.FLOOR, (a0, b0, floor - 0.05), (a1, b1, floor), room_id=rid)
        self.ent(f"{rid}-ceiling", EntityType.CEILING, (a0, b0, floor + h), (a1, b1, floor + h + 0.05), room_id=rid)
        poly = [self.xyz(a, b, floor) for a, b in ((a0, b0), (a1, b0), (a1, b1), (a0, b1))]
        self.ent(rid, EntityType.ROOM, (a0, b0, floor), (a1, b1, floor + h),
                 rels=[Relationship(RelationshipKind.CONTAINS, w) for w in wall_ids],
                 floor_height_m=floor * self.k, height_m=h * self.k, floor_area_m2=(a1 - a0) * (b1 - b0) * self.k ** 2,
                 footprint_polygon_xyz_m=[list(p) for p in poly], boundary_element_ids=wall_ids)
        if storey:
            if storey not in self.world.entities:
                self.ent(storey, EntityType.STOREY)
            self.world.entities[storey].relationships.append(Relationship(RelationshipKind.CONTAINS, rid))
        return rid

    def door(self, did, host, a0, a1, v0=0.0, v1=2.0, b=None, side="s", etype=EntityType.DOOR, rid="room-A"):
        """A door box on a wall along the first horizontal axis (``b`` overrides its perpendicular position)."""
        bb = {"s": 0.0, "n": 3.0}[side] if b is None else b
        return self.ent(did, etype, (a0, bb - T / 2, v0), (a1, bb + T / 2, v1), host_wall_id=host, room_id=rid)

    def furniture(self, fid, lo, hi, rels=(), **props):
        return self.ent(fid, EntityType.FURNITURE, lo, hi, rels=rels, **props)


def run(r: Room, cams=None):
    return validate_room_geometry(r.world, camera_centres=cams)


def codes(rep, check=None, severity=None):
    return {f.code for f in rep.findings if (check is None or f.check == check) and (severity is None or f.severity == severity)}


@pytest.fixture(params=["y", "z"])
def room(request):
    r = Room(up=request.param)
    r.room()
    return r


# ------------------------------------------------------------------ regression: clean worlds produce no errors
def test_clean_room_has_no_errors_and_passes_applicable_checks(room):
    rep = run(room, cams={"c1": room.xyz(2.0, 1.5, 1.4)})
    assert rep.errors == [] and rep.warnings == []
    assert rep.checks["room_closure"].state == "pass"
    assert rep.checks["duplicate_walls"].state == "pass"
    assert rep.checks["footprint_validity"].state == "pass"
    assert rep.checks["camera_conflicts"].state == "pass"
    assert rep.rooms["room-A"]["closure"]["state"] == "closed"
    assert set(rep.checks) == set(CHECKS)


def test_clean_room_report_is_json_serialisable_and_deterministic(room):
    import json
    a, b = run(room).to_dict(), run(room).to_dict()
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


@pytest.mark.parametrize("scale", [0.01, 1.0, 37.0])
def test_tolerances_are_relative_so_scale_does_not_change_findings(scale):
    """A unit-less reconstruction of any scale is judged like a metric one: same defect, same code, same pass."""
    clean, broken = Room(scale=scale), Room(scale=scale)
    clean.room()
    broken.room(walls=("s", "n", "w"))
    assert codes(run(clean), severity="error") == set()
    assert "room_boundary_open" in codes(run(broken), severity="error")


def test_canonical_interior_compiles_to_a_world_with_no_geometry_errors():
    """Regression: the known-good canonical interior (2 rooms, 2 storeys, compiled by the real compiler)."""
    from engine.compiler import CompileOptions, compile_reconstruction_to_world
    from tests.test_canonical_interior import UP, _canonical_interior_scene

    result = _canonical_interior_scene()
    for promote_building in (False, True):
        world, _ = compile_reconstruction_to_world(result, CompileOptions(
            seed=42, up=UP, promote_corridors=True, promote_windows=True, promote_stairs=True,
            promote_building=promote_building))
        rep = validate_room_geometry(world, camera_centres={c.evidence_id: c.position for c in result.camera_poses})
        assert rep.errors == [], [f.message for f in rep.errors]
        assert rep.checks["room_closure"].state == "pass"
        assert {r["closure"]["state"] for r in rep.rooms.values()} == {"closed"}
        assert rep.checks["camera_conflicts"].state == "pass"


# ------------------------------------------------------------------ duplicate walls
def test_duplicate_wall_injected_is_an_error(room):
    room.ent("dup", EntityType.WALL, (0.0, -T / 2 + 0.01, 0.0), (4.0, T / 2 + 0.01, 2.5), room_id="room-A")
    rep = run(room)
    assert "duplicate_wall" in codes(rep, "duplicate_walls", "error")
    f = next(f for f in rep.findings if f.code == "duplicate_wall")
    assert "dup" in f.element_ids and f.evidence["entity_ids"] and "overlap" in f.evidence
    assert rep.checks["duplicate_walls"].state == "fail"


def test_partial_collinear_overlap_below_half_is_not_a_duplicate(room):
    room.ent("stub", EntityType.WALL, (3.5, -T / 2, 0.0), (5.5, T / 2, 2.5), room_id="room-A")   # overlaps 0.5 of 2.0
    assert "duplicate_wall" not in codes(run(room))


def test_parallel_wall_far_away_is_not_a_duplicate(room):
    room.ent("par", EntityType.WALL, (0.0, 1.5, 0.0), (4.0, 1.5 + T, 2.5), room_id="room-A")
    assert codes(run(room), "duplicate_walls") == set()


def test_slab_and_its_evidence_plane_are_one_wall_not_a_duplicate(room):
    ev = room.ent("struct-plane-1", EntityType.WALL, (0.0, 0.0, 0.0), (4.0, 0.0, 2.5), room_id="room-A")
    room.world.entities["room-A-wall-s"].relationships.append(Relationship(RelationshipKind.ATTACHED_TO, ev.id))
    assert "duplicate_wall" not in codes(run(room))


def test_two_rooms_storing_one_partition_twice_is_a_warning_not_an_error():
    r = Room()
    r.room("room-A", 0, 0, 4, 3)
    r.room("room-B", 4.02, 0, 8, 3)   # B's west wall 4.02 vs A's east wall 4.0: the same partition stored twice
    rep = run(r)
    assert "duplicate_wall_shared_partition" in codes(rep, severity="warning")
    assert "duplicate_wall" not in codes(rep)


def test_tilted_wall_is_unknown_not_pass():
    r = Room()
    r.room()
    r.ent("tilt", EntityType.WALL, (0.0, 0.0, 0.0), (3.0, 3.0, 2.5), room_id="room-A")   # fat box: a diagonal wall
    rep = run(r)
    assert "wall_orientation_unreadable" in codes(rep, "duplicate_walls")
    assert rep.checks["duplicate_walls"].state == "partial"
    assert all(f.severity == "info" for f in rep.findings if f.code == "wall_orientation_unreadable")


# ------------------------------------------------------------------ room closure
def test_gap_in_wall_loop_is_open_error(room):
    r = Room(up=room.up)
    r.room(walls=("s", "w", "e"))                      # north wall missing
    rep = run(r)
    assert rep.rooms["room-A"]["closure"]["state"] == "open"
    assert "room_boundary_open" in codes(rep, "room_closure", "error")
    assert rep.checks["room_closure"].state == "fail"


def test_gap_between_collinear_wall_halves_is_found():
    r = Room()
    r.room(walls=("n", "w", "e"))
    r.ent("s1", EntityType.WALL, (0.0, -T / 2, 0.0), (1.4, T / 2, 2.5), room_id="room-A", thickness_m=T)
    r.ent("s2", EntityType.WALL, (2.6, -T / 2, 0.0), (4.0, T / 2, 2.5), room_id="room-A", thickness_m=T)
    r.world.entities["room-A"].custom_properties["boundary_element_ids"] += ["s1", "s2"]
    r.world.entities["room-A"].relationships += [Relationship(RelationshipKind.CONTAINS, "s1"),
                                                 Relationship(RelationshipKind.CONTAINS, "s2")]
    rep = run(r)
    gaps = rep.rooms["room-A"]["closure"]["gaps"]
    assert {g["wall"] for g in gaps} == {"s1", "s2"} and rep.rooms["room-A"]["closure"]["state"] == "open"


def test_gap_on_walls_that_declare_unobserved_spans_is_a_warning():
    r = Room()
    r.room(walls=("s", "w", "e"))
    for w in ("room-A-wall-w", "room-A-wall-e"):
        r.world.entities[w].custom_properties["unobserved_spans_m"] = [[0.0, 3.0]]
    rep = run(r)
    assert rep.rooms["room-A"]["closure"]["state"] == "open"
    assert codes(rep, "room_closure", "warning") == {"room_boundary_open"} and rep.errors == []


def test_two_walls_cannot_close_a_room():
    r = Room()
    r.room(walls=("s", "n"))
    rep = run(r)
    assert rep.rooms["room-A"]["closure"]["state"] == "open"
    assert "room_boundary_open" in codes(rep, severity="error")


def test_room_with_no_linked_walls_is_unknown_never_closed():
    r = Room()
    r.room()
    r.world.entities["room-A"].relationships.clear()
    r.world.entities["room-A"].custom_properties.pop("boundary_element_ids")
    for w in [e for e in r.world.entities.values() if e.type is EntityType.WALL]:
        w.custom_properties.pop("room_id")
    rep = run(r)
    assert rep.rooms["room-A"]["closure"]["state"] == "unknown"
    assert rep.checks["room_closure"].state == "unknown" and rep.errors == []


def test_room_with_unreadable_wall_is_unknown_never_closed():
    r = Room()
    r.room()
    r.ent("tilt", EntityType.WALL, (0.0, 0.0, 0.0), (3.0, 3.0, 2.5), room_id="room-A")
    assert run(r).rooms["room-A"]["closure"]["state"] == "unknown"


def test_world_without_recorded_up_axis_is_unknown():
    r = Room()
    r.room()
    del r.world.metadata["compiled_from"]
    rep = run(r)
    assert rep.rooms["room-A"]["closure"]["state"] == "unknown" and rep.errors == []
    assert rep.context["up_axis"] is None


# ------------------------------------------------------------------ footprint validity
def _set_poly(r: Room, pts):
    r.world.entities["room-A"].custom_properties["footprint_polygon_xyz_m"] = [list(r.xyz(a, b, 0.0)) for a, b in pts]


def test_self_intersecting_footprint_is_an_error(room):
    _set_poly(room, [(0, 0), (4, 3), (4, 0), (0, 3)])    # bow-tie
    assert "footprint_self_intersecting" in codes(run(room), "footprint_validity", "error")


def test_zero_area_footprint_is_an_error(room):
    _set_poly(room, [(0, 0), (4, 0), (4, 0.001), (0, 0.001)])
    assert "footprint_zero_area" in codes(run(room), severity="error")


def test_degenerate_footprint_two_distinct_vertices(room):
    _set_poly(room, [(0, 0), (0, 0), (4, 3), (4, 3)])
    assert "footprint_degenerate" in codes(run(room), severity="error")


def test_valid_non_convex_l_shape_footprint_passes(room):
    _set_poly(room, [(0, 0), (4, 0), (4, 1.5), (2, 1.5), (2, 3), (0, 3)])
    room.world.entities["room-A"].custom_properties.pop("floor_area_m2")
    rep = run(room)
    assert codes(rep, "footprint_validity") == set() and rep.checks["footprint_validity"].state == "pass"


def test_footprint_area_mismatch_is_a_warning(room):
    room.world.entities["room-A"].custom_properties["floor_area_m2"] = 30.0
    assert codes(run(room), "footprint_validity", "warning") == {"footprint_area_mismatch"}


def test_room_without_polygon_is_unknown(room):
    room.world.entities["room-A"].custom_properties.pop("footprint_polygon_xyz_m")
    rep = run(room)
    assert "footprint_unknown" in codes(rep) and rep.checks["footprint_validity"].state == "unknown"


def test_floor_plane_polygon_is_used_when_no_world_polygon(room):
    cp = room.world.entities["room-A"].custom_properties
    cp.pop("footprint_polygon_xyz_m")
    cp["boundary_polygon_2d"] = [[0, 0], [4, 3], [4, 0], [0, 3]]
    assert "footprint_self_intersecting" in codes(run(room))


# ------------------------------------------------------------------ openings
def test_door_on_its_wall_is_clean(room):
    room.door("d1", "room-A-wall-s", 1.0, 1.9)
    rep = run(room)
    assert codes(rep, "openings") == set() and rep.checks["openings"].state == "pass"


def test_door_off_the_wall_plane_is_an_error(room):
    room.door("d1", "room-A-wall-s", 1.0, 1.9, b=1.0)           # a metre inside the room, hosted on the south wall
    assert "opening_off_wall_plane" in codes(run(room), "openings", "error")


def test_door_beyond_the_wall_end_is_an_error(room):
    room.door("d1", "room-A-wall-s", 3.4, 4.4)
    assert "opening_outside_host_extent" in codes(run(room), severity="error")


def test_door_wider_than_host_wall_is_an_error(room):
    room.ent("short", EntityType.WALL, (0.0, 2.0, 0.0), (0.6, 2.02, 2.5), room_id="room-A")
    room.door("d1", "short", 0.0, 1.5, b=2.01)
    assert "opening_host_too_short" in codes(run(room), severity="error")


def test_door_without_a_host_is_an_error(room):
    room.ent("d1", EntityType.DOOR, (1.0, 0.0, 0.0), (1.9, T, 2.0), room_id="room-A")
    assert "opening_not_hosted" in codes(run(room), severity="error")


def test_door_hosted_on_a_non_wall_is_an_error(room):
    room.door("d1", "room-A-floor", 1.0, 1.9)
    assert "opening_not_hosted" in codes(run(room), severity="error")


def test_overlapping_openings_on_one_wall_are_an_error(room):
    room.door("d1", "room-A-wall-s", 1.0, 2.0)
    room.door("w1", "room-A-wall-s", 1.5, 2.5, v0=0.9, v1=1.8, etype=EntityType.WINDOW)
    assert "openings_overlap" in codes(run(room), severity="error")


def test_adjacent_openings_do_not_overlap(room):
    room.door("d1", "room-A-wall-s", 0.5, 1.4)
    room.door("d2", "room-A-wall-s", 2.0, 2.9)
    assert "openings_overlap" not in codes(run(room))


def test_opening_above_wall_top_is_only_a_warning(room):
    room.door("d1", "room-A-wall-s", 1.0, 1.9, v0=0.0, v1=3.5)
    rep = run(room)
    assert "opening_outside_wall_height" in codes(rep, severity="warning") and rep.errors == []


def test_opening_on_unreadable_host_is_unknown(room):
    room.ent("tilt", EntityType.WALL, (0.0, 0.0, 0.0), (3.0, 3.0, 2.5), room_id="room-A")
    room.door("d1", "tilt", 1.0, 1.9)
    rep = run(room)
    assert "opening_undecidable" in codes(rep, "openings") and rep.checks["openings"].state == "unknown"


def test_no_openings_means_not_applicable_not_pass(room):
    assert run(room).checks["openings"].state == "not_applicable"


# ------------------------------------------------------------------ surface bounding
def test_floor_overhanging_its_walls_is_an_error(room):
    room.ent("room-A-floor", EntityType.FLOOR, (-1.5, 0.0, -0.05), (4.0, 3.0, 0.0), room_id="room-A")
    assert "surface_exceeds_walls" in codes(run(room), "surface_bounding", "error")


def test_floor_floating_away_from_every_wall_is_an_error(room):
    room.ent("room-A-floor", EntityType.FLOOR, (20.0, 20.0, -0.05), (21.0, 21.0, 0.0), room_id="room-A")
    assert codes(run(room), "surface_bounding", "error") & {"surface_exceeds_walls", "surface_not_bounded_by_walls"}


def test_ceiling_not_above_floor_is_an_error(room):
    room.ent("room-A-ceiling", EntityType.CEILING, (0.0, 0.0, -0.5), (4.0, 3.0, -0.45), room_id="room-A")
    assert "ceiling_not_above_floor" in codes(run(room), severity="error")


def test_floor_without_a_room_is_unknown_not_pass(room):
    room.ent("loose-floor", EntityType.FLOOR, (0.0, 0.0, -0.05), (4.0, 3.0, 0.0))
    rep = run(room)
    assert "surface_no_room" in codes(rep, "surface_bounding") and rep.checks["surface_bounding"].state == "partial"


def test_floor_of_a_room_with_unreadable_walls_is_unknown(room):
    room.ent("tilt", EntityType.WALL, (0.0, 0.0, 0.0), (3.0, 3.0, 2.5), room_id="room-A")
    rep = run(room)
    assert "surface_walls_unreadable" in codes(rep, "surface_bounding")


# ------------------------------------------------------------------ attachments
def test_furniture_resting_on_the_floor_against_a_wall_is_clean(room):
    room.furniture("sofa", (0.05, 1.0, 0.0), (1.0, 2.0, 0.8))
    rep = run(room)
    assert codes(rep, "attachments") == set() and rep.checks["attachments"].state == "pass"


def test_floating_furniture_is_an_error(room):
    room.furniture("table", (1.5, 1.0, 1.2), (2.5, 2.0, 1.9))
    rep = run(room)
    assert "furniture_floating" in codes(rep, "attachments", "error")
    assert next(f for f in rep.findings if f.code == "furniture_floating").evidence["height_above_floor"] == pytest.approx(1.2)


def test_raised_furniture_with_a_support_relation_is_not_floating(room):
    room.furniture("shelf", (1.5, 1.0, 1.2), (2.5, 2.0, 1.9), rels=[Relationship(RelationshipKind.MOUNTED_ON, "room-A-wall-n")])
    assert "furniture_floating" not in codes(run(room))


def test_furniture_stacked_on_other_furniture_is_not_floating(room):
    room.furniture("desk", (1.5, 1.0, 0.0), (2.5, 2.0, 0.75))
    room.furniture("lamp", (1.8, 1.3, 0.75), (2.0, 1.5, 1.2))
    assert "furniture_floating" not in codes(run(room))


def test_furniture_below_the_floor_is_an_error(room):
    room.furniture("rug", (1.5, 1.0, -0.9), (2.5, 2.0, -0.3))
    assert "furniture_below_floor" in codes(run(room), severity="error")


def test_furniture_above_the_ceiling_is_an_error(room):
    room.furniture("wardrobe", (1.5, 1.0, 0.0), (2.0, 1.5, 4.0))
    assert "furniture_above_ceiling" in codes(run(room), severity="error")


def test_furniture_crossing_a_wall_is_an_error(room):
    room.furniture("bed", (3.5, 1.0, 0.0), (4.6, 2.0, 0.6))     # pokes 0.6 through the east wall at a=4.0
    rep = run(room)
    f = next(f for f in rep.findings if f.code == "furniture_penetrates_wall")
    assert "room-A-wall-e" in f.element_ids and f.evidence["depth"] > f.evidence["tolerance"]


def test_box_intruding_a_wall_while_its_centre_is_in_the_room_is_only_a_warning(room):
    """A yawed object's axis-aligned box overstates its extent, so bounds alone never decide 'penetrating'."""
    room.furniture("sofa", (3.0, 1.0, 0.0), (4.3, 2.0, 0.8))
    rep = run(room)
    assert "furniture_bounds_intrude_wall" in codes(rep, "attachments", "warning")
    assert "furniture_penetrates_wall" not in codes(rep) and rep.errors == []


def test_furniture_touching_a_wall_face_is_not_penetrating(room):
    room.furniture("cabinet", (3.0, 1.0, 0.0), (4.0 - T / 2, 2.0, 1.0))
    assert "furniture_penetrates_wall" not in codes(run(room))


def test_furniture_without_floor_level_is_unknown(room):
    room.world.entities["room-A"].custom_properties.pop("floor_height_m")
    room.furniture("table", (1.5, 1.0, 1.2), (2.5, 2.0, 1.9))
    rep = run(room)
    assert "attachment_no_floor_level" in codes(rep, "attachments") and "furniture_floating" not in codes(rep)


# ------------------------------------------------------------------ camera conflicts
def test_camera_inside_a_wall_is_an_error(room):
    rep = run(room, cams={"c1": room.xyz(2.0, 0.02, 1.4)})                 # inside the south slab
    f = next(f for f in rep.findings if f.code == "camera_inside_wall")
    assert f.severity == "error" and f.evidence["camera"] == "c1" and "room-A-wall-s" in f.element_ids


def test_camera_inside_a_zero_thickness_evidence_plane_is_an_error():
    r = Room()
    r.room(walls=())
    r.ent("plane", EntityType.WALL, (0.0, 1.0, 0.0), (4.0, 1.0, 2.5))
    assert "camera_inside_wall" in codes(run(r, cams={"c": r.xyz(2.0, 1.0, 1.0)}))


def test_camera_beside_a_wall_and_above_it_is_clean(room):
    rep = run(room, cams={"near": room.xyz(2.0, 0.4, 1.4), "above": room.xyz(2.0, 0.0, 3.5)})
    assert "camera_inside_wall" not in codes(rep)


def test_camera_outside_the_room_it_is_a_station_of_is_an_error(room):
    room.world.entities["room-A"].custom_properties["camera_stations"] = ["c1"]
    rep = run(room, cams={"c1": room.xyz(9.0, 9.0, 1.4)})
    assert "camera_outside_claimed_room" in codes(rep, "camera_conflicts", "error")


def test_camera_above_the_claimed_room_is_an_error(room):
    room.world.entities["room-A"].custom_properties["camera_stations"] = ["c1"]
    assert "camera_outside_room_height" in codes(run(room, cams={"c1": room.xyz(2.0, 1.5, 6.0)}))


def test_camera_inside_claimed_room_is_clean(room):
    room.world.entities["room-A"].custom_properties["camera_stations"] = ["c1"]
    assert run(room, cams={"c1": room.xyz(2.0, 1.5, 1.4)}).checks["camera_conflicts"].state == "pass"


def test_no_cameras_supplied_is_unknown_not_pass(room):
    rep = run(room)
    assert rep.checks["camera_conflicts"].state == "unknown" and "cameras_not_supplied" in codes(rep)


def test_non_finite_camera_centres_are_ignored_not_trusted(room):
    rep = run(room, cams={"c1": (float("nan"), 0.0, 0.0)})
    assert rep.checks["camera_conflicts"].state == "unknown"


def test_unreadable_wall_makes_camera_wall_test_partial(room):
    room.ent("tilt", EntityType.WALL, (0.0, 0.0, 0.0), (3.0, 3.0, 2.5), room_id="room-A")
    rep = run(room, cams={"c1": room.xyz(2.0, 1.5, 1.4)})
    assert "camera_wall_test_partial" in codes(rep) and rep.checks["camera_conflicts"].state == "partial"


# ------------------------------------------------------------------ topology
def test_orphan_wall_in_a_world_with_rooms_is_a_warning(room):
    room.ent("stray", EntityType.WALL, (10.0, 10.0, 0.0), (12.0, 10.0 + T, 2.5))
    assert "orphan_element" in codes(run(room), "topology", "warning")


def test_everything_linked_means_no_orphans(room):
    assert "orphan_element" not in codes(run(room))


def test_orphans_with_no_room_in_world_are_unknown():
    r = Room()
    r.ent("w1", EntityType.WALL, (0.0, 0.0, 0.0), (4.0, T, 2.5))
    rep = run(r)
    assert "orphans_undecidable" in codes(rep) and rep.checks["topology"].state == "unknown"


def _two_rooms(connect: bool):
    r = Room()
    r.room("room-A", 0, 0, 4, 3, storey="storey-1")
    r.room("room-B", 4.0, 0, 8, 3, storey="storey-1")
    if connect:
        r.door("d1", "room-A-wall-e", 4.0 - T / 2, 4.0 + T / 2, b=0.0, side="s")   # placeholder, rebuilt below
        r.world.entities.pop("d1")
        r.ent("d1", EntityType.DOOR, (4.0 - T / 2, 1.0, 0.0), (4.0 + T / 2, 1.9, 2.0), host_wall_id="room-A-wall-e")
        r.world.entities["room-B-wall-w"].custom_properties["room_id"] = "room-B"
        r.world.entities["room-B"].relationships.append(Relationship(RelationshipKind.CONTAINS, "room-A-wall-e"))
    return r


def test_rooms_with_openings_elsewhere_but_none_between_them_are_flagged():
    r = Room()
    r.room("room-A", 0, 0, 4, 3, storey="storey-1")
    r.room("room-B", 10, 0, 14, 3, storey="storey-1")
    r.room("room-C", 20, 0, 24, 3, storey="storey-1")
    r.ent("d1", EntityType.DOOR, (1.0, 0.0, 0.0), (1.9, T, 2.0), host_wall_id="room-A-wall-s", room_id="room-A")
    rep = run(r)
    assert "rooms_not_connected" in codes(rep, "topology", "warning")
    f = next(f for f in rep.findings if f.code == "rooms_not_connected")
    assert [["room-A"], ["room-B"], ["room-C"]] == sorted(f.evidence["groups"])


def test_two_rooms_joined_by_a_shared_wall_door_are_connected():
    rep = run(_two_rooms(connect=True))
    assert "rooms_not_connected" not in codes(rep)


def test_two_rooms_with_no_opening_evidence_are_unknown_not_connected_or_disconnected():
    rep = run(_two_rooms(connect=False))
    assert "connectivity_unknown" in codes(rep) and "rooms_not_connected" not in codes(rep)
    assert rep.checks["topology"].state == "partial"


def test_connects_relationship_joins_rooms():
    r = _two_rooms(connect=False)
    r.world.entities["room-A"].relationships.append(Relationship(RelationshipKind.CONNECTS, "room-B"))
    assert "rooms_not_connected" not in codes(run(r)) and "connectivity_unknown" not in codes(run(r))


# ------------------------------------------------------------------ report + stage wiring
def test_every_unknown_finding_is_info_and_decided_ones_are_not():
    r = Room()
    r.room()
    r.ent("tilt", EntityType.WALL, (0.0, 0.0, 0.0), (3.0, 3.0, 2.5), room_id="room-A")
    rep = run(r)
    assert rep.unknowns and all(f.severity == "info" and f.status == "unknown" for f in rep.unknowns)
    assert all(f.status == "violation" for f in rep.findings if f.severity != "info")


def test_findings_carry_evidence_refs_to_geometry_and_photos():
    r = Room()
    r.room()
    r.world.entities["room-A-wall-s"].custom_properties["supporting_evidence_ids"] = ["ev-7"]
    r.ent("dup", EntityType.WALL, (0.0, 0.0, 0.0), (4.0, T, 2.5), room_id="room-A")
    f = next(f for f in run(r).findings if f.code == "duplicate_wall")
    assert "g_dup" in f.evidence["geometry_ids"] and "ev-7" in f.evidence["supporting_evidence_ids"]


def test_validator_does_not_mutate_the_world(room):
    import copy
    room.furniture("table", (1.5, 1.0, 1.2), (2.5, 2.0, 1.9))
    before = copy.deepcopy(room.world.to_dict())
    run(room)
    assert room.world.to_dict() == before


def test_stage_hook_records_report_and_degrades_on_error():
    r = Room()
    r.room(walls=("s", "w", "e"))
    warnings, metrics = validate_for_stage(r.world, camera_poses=[("c1", (2.0, 1.4, 1.5), (1, 0, 0, 0))])
    assert any("room_boundary_open" in w for w in warnings) and metrics["geometry_errors"] >= 1.0
    rec = r.world.metadata["room_geometry_validation"]
    assert rec["verdict"] == "violations" and rec["rooms"]["room-A"]["closure"]["state"] == "open"

    def seg(ctx) -> SegmentReport:
        return SegmentReport(warnings=warnings, metrics=metrics)

    ctx = StageContext(job_id="j", workdir=Path("."), artifacts={}, params={}, state={"segments": {"compile_validate": seg}})
    res = CompileValidate().run(ctx)
    assert res.status == "degraded" and res.metrics["geometry_errors"] >= 1.0


def test_stage_hook_on_clean_world_does_not_degrade(room):
    warnings, metrics = validate_for_stage(room.world, camera_poses=[("c1", room.xyz(2.0, 1.5, 1.4), (1, 0, 0, 0))])
    assert warnings == [] and metrics["geometry_errors"] == 0.0
    assert room.world.metadata["room_geometry_validation"]["verdict"] in ("no_violations_found", "no_violations_found_with_unknowns")


def test_stage_hook_unknowns_do_not_degrade_but_are_counted():
    r = Room()
    r.room()
    warnings, metrics = validate_for_stage(r.world)           # no cameras supplied
    assert warnings == [] and metrics["geometry_unknown"] >= 1.0


def test_validator_crash_is_recorded_and_degrades_never_silent(monkeypatch, room):
    import engine.validation.room_geometry as rg

    def boom(*a, **k):
        raise RuntimeError("kaput")
    monkeypatch.setattr(rg, "validate_room_geometry", boom)
    warnings, metrics = rg.validate_for_stage(room.world)
    assert warnings and "kaput" in warnings[0] and metrics["geometry_validator_error"] == 1.0
    assert room.world.metadata["room_geometry_validation"]["verdict"] == "validator_error"


def test_camera_pose_objects_are_accepted():
    from types import SimpleNamespace

    r = Room()
    r.room()
    pose = SimpleNamespace(evidence_id="c1", position=r.xyz(2.0, 0.02, 1.4))
    warnings, _ = validate_for_stage(r.world, camera_poses=[pose])
    assert any("camera_inside_wall" in w for w in warnings)


# ------------------------------------------------------------------ opening hosting as the compiler actually writes it
def test_window_wired_to_its_wall_by_part_of_is_hosted_and_not_a_constraint_error(room):
    """The compiler wires a promoted window to its wall with PART_OF (no host_wall_id). The P111 constraint used to read
    that as 'no host wall' and degrade every job with a window; both validators now accept a PART_OF -> WALL host."""
    from world_ir.constraints import check_architectural_constraints

    w = room.ent("win", EntityType.WINDOW, (1.0, -T / 2, 0.9), (2.0, T / 2, 1.9),
                 rels=[Relationship(RelationshipKind.PART_OF, "room-A-wall-s")])
    assert codes(run(room), "openings") == set()
    assert not [v for v in check_architectural_constraints(room.world) if v.code == "opening_without_host_wall"]
    assert w.custom_properties.get("host_wall_id") is None


def test_door_that_is_only_part_of_a_room_is_still_not_hosted(room):
    room.ent("d", EntityType.DOOR, (1.0, -T / 2, 0.0), (1.9, T / 2, 2.0), rels=[Relationship(RelationshipKind.PART_OF, "room-A")])
    assert "opening_not_hosted" in codes(run(room), severity="error")


def test_opening_bounds_fall_back_to_the_promoter_recorded_properties(room):
    room.ent("d", EntityType.DOOR, host_wall_id="room-A-wall-s",
             bounds_min=list(room.xyz(1.0, -T / 2, 0.0)), bounds_max=list(room.xyz(3.9, T / 2, 2.0)))
    # 2.9 wide on a 4.0 wall is fine; widen it past the wall end to prove the fallback is actually evaluated
    room.world.entities["d"].custom_properties["bounds_max"] = list(room.xyz(4.8, T / 2, 2.0))
    assert "opening_outside_host_extent" in codes(run(room))


def test_tilted_wall_beyond_the_axis_alignment_limit_is_unreadable_not_guessed():
    r = Room()
    r.room()
    r.ent("lean", EntityType.WALL, (0.0, 1.0, 0.0), (4.0, 1.6, 2.5), room_id="room-A")   # 0.6 over 4.0 = 8.5 deg
    rep = run(r)
    assert "wall_orientation_unreadable" in codes(rep) and "duplicate_wall" not in codes(rep)


def test_short_wall_with_declared_thickness_matching_its_box_is_readable_but_undeclared_is_not():
    r = Room()
    r.room(walls=())
    r.ent("short-declared", EntityType.WALL, (0.0, 0.0, 0.0), (1.2, T, 2.5), thickness_m=T)
    r.ent("short-undeclared", EntityType.WALL, (3.0, 0.0, 0.0), (4.2, T, 2.5))
    rep = run(r)
    assert {f.element_ids[0] for f in rep.findings if f.code == "wall_orientation_unreadable"} == {"short-undeclared"}
