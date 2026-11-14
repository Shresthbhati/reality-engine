"""A floor/ceiling is flat, lies beyond the cameras that saw it, and is one plane per level.

classify_plane only checks which side of a plane the cameras are on, so every downward-facing slab that dense monocular
depth smeared across a room became a 'ceiling' (12 in a room with one). The checks here are per plane, so distinct levels
(storeys, mezzanines) are never removed."""

from __future__ import annotations

from perception.geometry.orientation import OrientedPlane, consolidate_horizontal_roles
from perception.geometry.planes import DetectedPlane
from provenance import Uncertainty

UP = (0.0, 1.0, 0.0)


def _plane(pid, role, height, n_inliers, positions, tilt=0.0):
    ids = [f"{pid}-{k:04d}" for k in range(n_inliers)]      # zero-padded: DetectedPlane requires sorted ids
    for k, tid in enumerate(ids):
        x = float(k % 7)
        positions[tid] = (x, height + tilt * x, float(k // 7))   # tilt = height gained per unit of x
    dp = DetectedPlane(plane_id=pid, normal=(0.0, 1.0, 0.0), d=-height, inlier_ids=ids, inlier_rms_distance_m=0.01,
                       uncertainty=Uncertainty(confidence=0.9, note="x"))
    return OrientedPlane(plane=dp, role=role, normal=(0.0, -1.0 if role == "ceiling" else 1.0, 0.0), d=height,
                         tilt_from_role_rad=0.0, uncertainty=Uncertainty(confidence=0.9, note="x"))


def _roles(planes):
    return {p.plane.plane_id: p.role for p in planes}


CAMS = [(0.0, 1.0, 0.0), (1.0, 1.3, 0.0), (2.0, 1.1, 1.0)]       # eye-level cameras inside a 2.5-high room


def test_tilted_depth_smear_is_not_a_ceiling_but_the_flat_ceiling_stays():
    pos = {}
    planes = [
        _plane("ceil", "ceiling", 2.5, 100, pos),
        _plane("smear-a", "ceiling", 1.6, 60, pos, tilt=0.3),      # spans ~1.7 vertically over ~7 laterally
        _plane("smear-b", "ceiling", 1.8, 55, pos, tilt=0.25),
        _plane("floor", "floor", 0.0, 120, pos),
    ]
    out, stats = consolidate_horizontal_roles(planes, UP, CAMS, pos)
    assert _roles(out) == {"ceil": "ceiling", "smear-a": "unknown", "smear-b": "unknown", "floor": "floor"}
    assert stats == {"demoted": 2, "merged": 0} and "tilted or depth-smeared" in out[1].uncertainty.note


def test_when_every_candidate_is_tilted_the_best_supported_one_is_the_surface():
    """Monocular depth tilts real floors too: a hard flatness veto deleted the floor and the multi-view room was rejected."""
    pos = {}
    planes = [_plane("floor-main", "floor", 0.0, 150, pos, tilt=-0.6), _plane("floor-bit", "floor", 0.1, 40, pos, tilt=-0.6),
              _plane("ceil-main", "ceiling", 2.5, 120, pos, tilt=0.6), _plane("ceil-bit", "ceiling", 2.2, 50, pos, tilt=0.6)]
    out, stats = consolidate_horizontal_roles(planes, UP, CAMS, pos)
    assert _roles(out) == {"floor-main": "floor", "floor-bit": "unknown", "ceil-main": "ceiling", "ceil-bit": "unknown"}
    assert stats == {"demoted": 2, "merged": 0} and "better-supported" in out[1].uncertainty.note


def test_a_ceiling_below_the_cameras_that_saw_it_is_not_a_ceiling():
    pos = {}
    planes = [_plane("ceil", "ceiling", 2.5, 100, pos), _plane("under-shelf", "ceiling", 0.4, 90, pos)]
    out, _ = consolidate_horizontal_roles(planes, UP, CAMS, pos)
    assert _roles(out) == {"ceil": "ceiling", "under-shelf": "unknown"}
    assert "not beyond them" in out[1].uncertainty.note


def test_fragments_of_one_level_merge_into_the_strongest_plane():
    pos = {}
    planes = [_plane("a", "ceiling", 2.50, 100, pos), _plane("b", "ceiling", 2.51, 40, pos),
              _plane("c", "ceiling", 2.49, 30, pos), _plane("floor", "floor", 0.0, 90, pos)]
    out, stats = consolidate_horizontal_roles(planes, UP, CAMS, pos)
    assert _roles(out) == {"a": "ceiling", "b": "unknown", "c": "unknown", "floor": "floor"}
    assert len(out[0].plane.inlier_ids) == 170 and stats == {"demoted": 0, "merged": 2}
    assert "merged into a" in out[1].uncertainty.note


def test_floor_copies_within_the_surfaces_own_roughness_are_one_floor_but_a_clear_gap_is_a_level():
    """Per-view monocular depth gives each photo its own copy of one floor: copies closer than the floor's own vertical
    spread merge; a copy beyond it stays a separate level."""
    pos = {}
    planes = [_plane("main", "floor", 0.0, 150, pos, tilt=0.04),        # spread ~0.22 over a 6-wide extent: still flat
              _plane("copy", "floor", 0.3, 60, pos), _plane("far", "floor", 0.8, 50, pos),
              _plane("ceil", "ceiling", 2.5, 100, pos)]
    out, stats = consolidate_horizontal_roles(planes, UP, CAMS, pos)
    assert _roles(out) == {"main": "floor", "copy": "unknown", "far": "floor", "ceil": "ceiling"}
    assert stats["merged"] == 1 and len(out[0].plane.inlier_ids) == 210


def test_distinct_flat_levels_all_survive_storeys_and_mezzanines():
    pos = {}
    cams = CAMS + [(0.0, 3.6, 0.0), (1.0, 3.9, 0.0)]                # cameras on both levels
    # two storeys 3.0 apart, plus a flat mezzanine floor only 0.6 above the ground floor (the canonical fixtures' shape)
    planes = [_plane("ceil-1", "ceiling", 2.9, 100, pos), _plane("ceil-2", "ceiling", 5.6, 90, pos),
              _plane("floor-1", "floor", 0.0, 100, pos), _plane("floor-2", "floor", 3.0, 90, pos),
              _plane("mezzanine", "floor", 0.6, 80, pos)]
    obs = {"ceil-1": CAMS, "floor-1": CAMS, "mezzanine": CAMS, "ceil-2": cams[3:], "floor-2": cams[3:]}
    out, stats = consolidate_horizontal_roles(planes, UP, cams, pos, observers=obs)
    assert _roles(out) == {"ceil-1": "ceiling", "ceil-2": "ceiling", "floor-1": "floor", "floor-2": "floor",
                           "mezzanine": "floor"}
    assert stats == {"demoted": 0, "merged": 0}


def test_walls_and_unknowns_pass_through_and_empty_input_is_safe():
    pos = {}
    wall = _plane("w", "wall", 1.0, 30, pos)
    out, stats = consolidate_horizontal_roles([wall], UP, CAMS, pos)
    assert out == [wall] and stats == {"demoted": 0, "merged": 0}
    assert consolidate_horizontal_roles([], UP, CAMS, {}) == ([], {"demoted": 0, "merged": 0})
