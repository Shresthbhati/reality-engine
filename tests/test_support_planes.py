"""B09: a table top or step is a support surface, never the floor; a second storey gets its own floor."""
from perception.geometry.support_planes import split_floor_candidates


def test_table_top_is_a_support_surface():
    roles = split_floor_candidates([0.0, 0.01, 0.75], [20.0, 5.0, 3.0], footprint_area=25.0, wall_bases=[4, 3, 0])
    assert roles == ["floor", "floor", "support_surface"]


def test_staircase_steps_are_not_floors():
    roles = split_floor_candidates([0.0, 0.18, 0.36, 0.54], [20.0, 1.0, 1.0, 1.0], 25.0, [4, 0, 0, 0])
    assert roles == ["floor", "support_surface", "support_surface", "support_surface"]


def test_each_storey_is_judged_against_its_own_footprint():
    upper = split_floor_candidates([3.0, 3.75], [18.0, 2.0], footprint_area=20.0, wall_bases=[3, 0])
    assert upper == ["floor", "support_surface"]


def test_no_candidate_covers_enough_means_no_floor_invented():
    assert split_floor_candidates([0.0, 0.5], [2.0, 2.0], 25.0, [4, 4]) == ["support_surface", "support_surface"]


def test_every_level_that_reaches_walls_is_a_floor_not_just_the_lowest():
    # two levels of equal size, each reached by 4 wall bases (a multi-storey / split-level capture)
    assert split_floor_candidates([0.0, 1.5], [5.76, 5.76], 5.76, [4, 4]) == ["floor", "floor"]


def test_a_small_room_is_judged_against_its_own_walls_not_the_largest_floor():
    # 25 m2 hall + 4 m2 closet on another level: against the hall's footprint the closet would be a "table top"
    assert split_floor_candidates([0.0, 3.0], [25.0, 4.0], 25.0, [4, 3]) == ["floor", "support_surface"]
    assert split_floor_candidates([0.0, 3.0], [25.0, 4.0], [25.0, 4.5], [4, 3]) == ["floor", "floor"]


def test_a_slab_just_above_a_floor_is_not_a_second_floor():
    # both clusters reach walls and span the footprint, but 0.1 m apart: a raised slab, not another floor
    assert split_floor_candidates([0.0, 0.1], [20.0, 20.0], 20.0, [4, 4]) == ["floor", "support_surface"]


def test_classify_planes_labels_intermediate_plane_as_support_surface():
    from perception.architecture.classify import PlaneInput, classify_planes

    up = (0.0, 0.0, 1.0)
    floor = PlaneInput(
        plane_id="floor", normal=up, centroid=(1.5, 1.5, 0.0),
        bounds_min=(0.0, 0.0, 0.0), bounds_max=(3.0, 3.0, 0.0),
    )
    ceiling = PlaneInput(
        plane_id="ceiling", normal=up, centroid=(1.5, 1.5, 2.5),
        bounds_min=(0.0, 0.0, 2.5), bounds_max=(3.0, 3.0, 2.5),
    )
    table = PlaneInput(
        plane_id="table", normal=up, centroid=(1.5, 1.5, 0.75),
        bounds_min=(1.0, 1.0, 0.75), bounds_max=(2.0, 2.0, 0.75),
    )
    elements = classify_planes([floor, table, ceiling], up)
    by_id = {el.source_plane_id: el for el in elements}
    assert by_id["floor"].element_type == "floor"
    assert by_id["ceiling"].element_type == "ceiling"
    assert by_id["table"].element_type == "support_surface"


def test_find_floor_elevation_ignores_support_surfaces():
    from perception.architecture.fixtures import find_floor_elevation
    from world_ir.schema_v1 import Entity

    true_floor = Entity(
        id="floor_0",
        type="floor",
        transform=((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)),
        custom_properties={"elevation_m": 0.0, "role": "floor"},
    )
    table_surface = Entity(
        id="table_0",
        type="support_surface",
        transform=((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0.75), (0, 0, 0, 1)),
        custom_properties={"elevation_m": 0.75, "role": "support_surface"},
    )

    # Fixture at z=1.2 (e.g. wall switch)
    elev = find_floor_elevation((1.0, 0.0, 1.2), [true_floor, table_surface])
    assert elev == 0.0


def test_support_surfaces_in_full_reconstruction_workflow():
    from engine.compiler import CompileOptions, compile_reconstruction_to_world
    from provenance import Uncertainty
    from reconstruction.backend.interface import (
        ReconstructedCameraPose,
        ReconstructedPoint,
        ReconstructionResult,
    )
    from world_ir.schema_v1 import EntityType, RelationshipKind
    from world_ir.validation import validate_world_ir

    def _pt(x, y, z, c=[0]):
        c[0] += 1
        return ReconstructedPoint(
            position=(x, y, z),
            track_id=f"pt-{c[0]:05d}",
            source_evidence_ids=["ev-1"],
            uncertainty=Uncertainty(1.0, "synthetic"),
        )

    pts = []
    # Floor at y=0.0 (3x3m)
    for xi in range(31):
        for zi in range(31):
            pts.append(_pt(xi * 0.1, 0.0, zi * 0.1))
    # Ceiling at y=2.5
    for xi in range(31):
        for zi in range(31):
            pts.append(_pt(xi * 0.1, 2.5, zi * 0.1))
    # 4 walls bounding the room
    for yi in range(13):
        for zi in range(31):
            pts.append(_pt(0.0, yi * 0.2, zi * 0.1))
            pts.append(_pt(3.0, yi * 0.2, zi * 0.1))
        for xi in range(31):
            pts.append(_pt(xi * 0.1, yi * 0.2, 0.0))
            pts.append(_pt(xi * 0.1, yi * 0.2, 3.0))

    # Tabletop at y=0.75 (1x1m interior surface)
    for xi in range(11):
        for zi in range(11):
            pts.append(_pt(1.0 + xi * 0.1, 0.75, 1.0 + zi * 0.1))

    # Shelf at y=1.5 (1.3x0.6m interior surface)
    for xi in range(14):
        for zi in range(7):
            pts.append(_pt(0.2 + xi * 0.1, 1.5, 0.2 + zi * 0.1))

    cams = [
        ReconstructedCameraPose("ev-0", (1.5, 1.8, 1.5), (1.0, 0.0, 0.0, 0.0)),
        ReconstructedCameraPose("ev-1", (1.0, 1.8, 2.0), (1.0, 0.0, 0.0, 0.0)),
        ReconstructedCameraPose("ev-2", (2.0, 1.8, 1.0), (1.0, 0.0, 0.0, 0.0)),
    ]
    res = ReconstructionResult(points=pts, camera_poses=cams, registration_status="success")
    world, diag = compile_reconstruction_to_world(res, CompileOptions(seed=42))

    assert diag.planes_by_role["floor"] == 1
    assert diag.planes_by_role["ceiling"] == 1
    assert diag.planes_by_role["wall"] == 4
    assert diag.planes_by_role["support_surface"] == 2
    assert diag.rooms_detected == 1

    support_ents = [e for e in world.entities.values() if e.type is EntityType.SUPPORT_SURFACE]
    assert len(support_ents) == 2

    room_ents = [e for e in world.entities.values() if e.type is EntityType.ROOM]
    assert len(room_ents) == 1
    room = room_ents[0]

    # B09 room containment verification: room CONTAINS support surfaces, support surfaces PART_OF room
    contains_ids = {r.target_id for r in room.relationships if r.kind is RelationshipKind.CONTAINS}
    for se in support_ents:
        assert se.id in contains_ids
        part_of_ids = {r.target_id for r in se.relationships if r.kind is RelationshipKind.PART_OF}
        assert room.id in part_of_ids
        assert se.custom_properties.get("room_id") == room.id
        assert se.custom_properties.get("space_id") == room.id

    report = validate_world_ir(world)
    assert report.is_valid(), report.messages()


def test_stair_treads_support_surfaces_and_host_attachment():
    from engine.compiler import CompileOptions, compile_reconstruction_to_world
    from provenance import Uncertainty
    from reconstruction.backend.interface import (
        ReconstructedCameraPose,
        ReconstructedPoint,
        ReconstructionResult,
    )
    from world_ir.schema_v1 import EntityType, RelationshipKind, RelationshipKind
    from world_ir.validation import validate_world_ir

    def _pt(x, y, z, c=[0]):
        c[0] += 1
        return ReconstructedPoint(
            position=(x, y, z),
            track_id=f"pt-stair-{c[0]:05d}",
            source_evidence_ids=["ev-1"],
            uncertainty=Uncertainty(1.0, "synthetic"),
        )

    pts = []
    # Floor at y=0.0 (3x3m)
    for xi in range(31):
        for zi in range(31):
            pts.append(_pt(xi * 0.1, 0.0, zi * 0.1))
    # Ceiling at y=2.5
    for xi in range(31):
        for zi in range(31):
            pts.append(_pt(xi * 0.1, 2.5, zi * 0.1))
    # 4 walls
    for yi in range(13):
        for zi in range(31):
            pts.append(_pt(0.0, yi * 0.2, zi * 0.1))
            pts.append(_pt(3.0, yi * 0.2, zi * 0.1))
        for xi in range(31):
            pts.append(_pt(xi * 0.1, yi * 0.2, 0.0))
            pts.append(_pt(xi * 0.1, yi * 0.2, 3.0))

    # 3 distinct wide step treads (0.8m wide, 0.4m deep)
    for step_i in range(3):
        sy = (step_i + 1) * 0.25
        sz = 0.6 + step_i * 0.5
        for xi in range(12):
            for zi in range(7):
                pts.append(_pt(0.8 + xi * 0.1, sy, sz + zi * 0.06))

    cams = [
        ReconstructedCameraPose("ev-0", (1.5, 1.8, 1.5), (1.0, 0.0, 0.0, 0.0)),
        ReconstructedCameraPose("ev-1", (1.0, 1.8, 2.0), (1.0, 0.0, 0.0, 0.0)),
    ]
    res = ReconstructionResult(points=pts, camera_poses=cams, registration_status="success")
    world, diag = compile_reconstruction_to_world(res, CompileOptions(seed=42, promote_stairs=True))

    assert diag.planes_by_role["floor"] == 1
    assert diag.planes_by_role["ceiling"] == 1
    assert diag.planes_by_role["wall"] == 4
    # All step tread planes are support surfaces, never floor
    assert diag.planes_by_role.get("support_surface", 0) >= 1

    support_ents = [e for e in world.entities.values() if e.type is EntityType.SUPPORT_SURFACE]
    assert len(support_ents) >= 1
    for se in support_ents:
        part_of = {r.target_id for r in se.relationships if r.kind is RelationshipKind.PART_OF}
        assert any(pid.startswith("room") for pid in part_of)

    report = validate_world_ir(world)
    assert report.is_valid(), report.messages()

