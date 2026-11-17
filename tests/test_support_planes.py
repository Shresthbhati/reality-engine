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
