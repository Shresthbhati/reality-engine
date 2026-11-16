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
