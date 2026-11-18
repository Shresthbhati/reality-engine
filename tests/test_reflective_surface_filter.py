"""RE-PERC-000018: objects lifted BEHIND a flat mirror/glass candidate are reflections and are dropped."""
from types import SimpleNamespace as NS

import numpy as np

from perception.detection.grounded_scene import filter_implausible

BOX = (np.array([0.0, 0.0, 0.0]), np.array([5.0, 3.0, 5.0]))  # the room; its inside is toward +x


def cand(label, pos, half=(0.2, 0.2, 0.2)):
    p = NS(x=pos[0], y=pos[1], z=pos[2])
    return NS(label=label, position=p, evidence_ids=["a", "b"], confidence=0.9,
              bounds_min=NS(x=pos[0] - half[0], y=pos[1] - half[1], z=pos[2] - half[2]),
              bounds_max=NS(x=pos[0] + half[0], y=pos[1] + half[1], z=pos[2] + half[2]))


MIRROR = cand("mirror", (0.02, 1.5, 2.0), half=(0.02, 0.5, 0.5))  # flat pane on the x=0 wall, 1 m x 1 m


def names(cands, **kw):
    kept, dropped = filter_implausible(cands, BOX, n_views=3, **kw)
    return [c.label for c in kept], dropped


def test_object_behind_the_mirror_is_a_reflection():
    kept, dropped = names([MIRROR, cand("sofa", (2.0, 0.5, 2.0)), cand("lamp", (-1.0, 1.5, 2.0))])
    assert kept == ["mirror", "sofa"] and dropped == {"behind_reflective_surface": 1}


def test_object_behind_but_far_off_to_the_side_is_kept():
    kept, dropped = names([MIRROR, cand("lamp", (-0.5, 1.5, 4.8))])   # 2.8 m sideways, 0.5 m behind: outside the cone
    assert kept == ["mirror", "lamp"] and dropped == {}


def test_no_mirror_no_filtering():
    kept, dropped = names([cand("lamp", (-1.0, 1.5, 2.0))])
    assert kept == ["lamp"] and dropped == {}


def test_fat_box_defines_no_plane():
    chunky = cand("mirror", (0.5, 1.5, 2.0), half=(0.5, 0.5, 0.5))     # a cube is not a pane
    kept, _ = names([chunky, cand("lamp", (-1.0, 1.5, 2.0))])
    assert "lamp" in kept


def test_object_far_bigger_than_its_class_is_a_depth_blow_up():
    ont = NS(resolve_or_unknown=lambda label: NS(typical_dims_m={"w": (0.4, 0.6), "h": (0.8, 1.0)} if label == "chair" else None))
    chair_ok = cand("chair", (2.0, 0.5, 2.0), half=(0.3, 0.45, 0.3))      # 0.9 m: within 2.5 x 1.0
    chair_huge = cand("chair", (3.0, 0.5, 3.0), half=(2.0, 0.5, 0.3))     # 4 m long
    unknown_big = cand("thing", (1.0, 0.5, 1.0), half=(1.5, 0.5, 0.3))    # no declared dims: never rejected on size
    big_room = (np.array([0.0, 0.0, 0.0]), np.array([20.0, 6.0, 20.0]))     # so the scene-size rule does not fire first
    kept, dropped = filter_implausible([chair_ok, chair_huge, unknown_big], big_room, n_views=3, ontology=ont)
    assert [c.label for c in kept] == ["chair", "thing"] and dropped == {"implausible_size": 1}
