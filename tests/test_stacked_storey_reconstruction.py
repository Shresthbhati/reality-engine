"""A genuinely hard multi-storey fixture: storey 2 sits directly above
storey 1, sharing the same (x, z) footprint -- unlike every other
"multi-storey" fixture in this repo, which either hand-builds WorldIR
entities directly (tests/test_topology_coherence.py, bypassing
detect_planes/classify_planes entirely) or places rooms at disjoint
(x, z) footprints that also happen to differ in height
(tests/test_canonical_interior.py::_canonical_interior_scene), which
build_building_graph's height-based grouping separates trivially.

A real two-storey building's floors overlap in plan: storey 1's ceiling
and storey 2's floor are two closely-spaced, physically distinct
horizontal slabs at nearly the same (x, z) extent. This is the case
that actually stresses floor/ceiling discrimination, because a capture
walking through the whole building records camera poses on BOTH
floors -- unlike every synthetic fixture above, whose cameras sit in
one room only.

Convention matches tests/test_room_inference.py: up = (0, 1, 0) (y is
vertical), ReconstructedPoint(position=(x, y, z), ...).
"""
from __future__ import annotations

from perception.geometry.orientation import classify_planes
from perception.geometry.planes import detect_planes, plane_centroids
from reconstruction.backend.interface import ReconstructedPoint, ReconstructionResult

UP = (0.0, 1.0, 0.0)

# Storey 1: floor y=0.0, ceiling y=2.4. Storey 2: floor y=2.6 (a 0.2 m
# slab thickness above ceiling-1), ceiling y=5.0. Same (x, z) footprint
# for both storeys: x in [0, 3], z in [0, 2.5].
FOOTPRINT = (0.0, 3.0, 0.0, 2.5)
STOREY_1_FLOOR_Y = 0.0
STOREY_1_CEILING_Y = 2.4
STOREY_2_FLOOR_Y = 2.6
STOREY_2_CEILING_Y = 5.0

#: Cameras split across both floors -- the realistic capture case for a
#: walkthrough of the whole building, not a single room.
CAMS_STOREY_1 = [(1.25, 1.20, 1.1), (0.9, 1.15, 1.5), (1.6, 1.25, 0.8)]
CAMS_STOREY_2 = [(1.25, 3.80, 1.1), (0.9, 3.75, 1.5), (1.6, 3.85, 0.8)]


def _point(x: float, y: float, z: float, counter: list) -> ReconstructedPoint:
    counter[0] += 1
    return ReconstructedPoint(
        position=(x, y, z), track_id=f"pt-{counter[0]:06d}", source_evidence_ids=["ev-1"]
    )


def _grid(a_lo, a_hi, b_lo, b_hi, step=0.15):
    n_a = int(round((a_hi - a_lo) / step))
    n_b = int(round((b_hi - b_lo) / step))
    for i in range(n_a + 1):
        for j in range(n_b + 1):
            yield a_lo + i * step, b_lo + j * step


def _slab(y, counter, step=0.15):
    x_lo, x_hi, z_lo, z_hi = FOOTPRINT
    return [_point(x, y, z, counter) for x, z in _grid(x_lo, x_hi, z_lo, z_hi, step)]


def _wall_x(x, y_lo, y_hi, counter, step=0.15, gap=None):
    _, _, z_lo, z_hi = FOOTPRINT
    pts = []
    for z, y in _grid(z_lo, z_hi, y_lo, y_hi, step):
        if gap and gap[0] <= z <= gap[1] and y <= gap[2]:
            continue  # doorway
        pts.append(_point(x, y, z, counter))
    return pts


def _wall_z(z, y_lo, y_hi, counter, step=0.15, gap=None):
    x_lo, x_hi, _, _ = FOOTPRINT
    pts = []
    for x, y in _grid(x_lo, x_hi, y_lo, y_hi, step):
        if gap and gap[0] <= x <= gap[1] and y <= gap[2]:
            continue  # doorway
        pts.append(_point(x, y, z, counter))
    return pts


def _stacked_two_storey_building() -> ReconstructionResult:
    """Two real, vertically-stacked storeys sharing one footprint, plus
    camera poses recorded on both floors (a real walkthrough capture)."""
    counter = [0]
    x_lo, x_hi, z_lo, z_hi = FOOTPRINT
    points = []
    # Storey 1 envelope.
    points += _slab(STOREY_1_FLOOR_Y, counter)
    points += _slab(STOREY_1_CEILING_Y, counter)
    points += _wall_x(x_lo, STOREY_1_FLOOR_Y, STOREY_1_CEILING_Y, counter, gap=(0.9, 1.6, 2.0))
    points += _wall_x(x_hi, STOREY_1_FLOOR_Y, STOREY_1_CEILING_Y, counter)
    points += _wall_z(z_lo, STOREY_1_FLOOR_Y, STOREY_1_CEILING_Y, counter)
    points += _wall_z(z_hi, STOREY_1_FLOOR_Y, STOREY_1_CEILING_Y, counter)
    # Storey 2 envelope, directly above (same footprint).
    points += _slab(STOREY_2_FLOOR_Y, counter)
    points += _slab(STOREY_2_CEILING_Y, counter)
    points += _wall_x(x_lo, STOREY_2_FLOOR_Y, STOREY_2_CEILING_Y, counter, gap=(0.9, 1.6, 4.6))
    points += _wall_x(x_hi, STOREY_2_FLOOR_Y, STOREY_2_CEILING_Y, counter)
    points += _wall_z(z_lo, STOREY_2_FLOOR_Y, STOREY_2_CEILING_Y, counter)
    points += _wall_z(z_hi, STOREY_2_FLOOR_Y, STOREY_2_CEILING_Y, counter)
    return ReconstructionResult(points=points, camera_poses=[], registration_status="success")


def _approx_height(plane) -> float:
    """normal . point + d = 0 => y = -d / normal_y for a horizontal plane."""
    if abs(plane.normal[1]) > 0.5:
        return -plane.d / plane.normal[1]
    return float("nan")  # vertical (wall) plane


class TestStackedTwoStoreyClassification:
    """detect_planes + classify_planes on a real stacked-storey capture.

    classify_plane's floor/ceiling disambiguation
    (perception/geometry/orientation.py) used to orient every plane's
    normal toward the MEAN of every camera position in the whole capture
    (flip_normal_toward). That is fine for a single-room capture -- all
    cameras genuinely sit on one side of every horizontal plane -- but
    for a real multi-storey capture, cameras split across floors, so the
    global mean can land on the WRONG side of a plane whose true (local,
    same-floor) cameras all sit on the other side. Here the mean camera
    height (2.5) falls between storey-1's ceiling (2.4) and storey-2's
    floor (2.6) -- exactly the two slabs closest to that midpoint -- so
    both used to get their normals flipped against their true local
    camera side and misclassified as each other. Fixed by orienting each
    plane toward the camera closest to it in full 3D (using its own real
    inlier centroid, via perception.geometry.planes.plane_centroids), not
    the capture-wide mean -- height alone was tried first and rejected:
    it regressed tests/test_canonical_interior.py, where a second room's
    cameras can sit closer in HEIGHT to the first room's ceiling than
    that room's own (spatially distant) cameras do.
    """

    def test_average_camera_height_straddles_the_middle_two_slabs(self):
        # Precondition check on the fixture itself, not the code under
        # test: confirms this fixture actually creates the adversarial
        # camera distribution the docstring above describes.
        all_cams = CAMS_STOREY_1 + CAMS_STOREY_2
        avg_y = sum(c[1] for c in all_cams) / len(all_cams)
        assert STOREY_1_CEILING_Y < avg_y < STOREY_2_FLOOR_Y

    def test_outer_slabs_classify_correctly(self):
        """The two outer slabs (real floor-1, real ceiling-2) are far
        enough from the global camera mean that today's global-mean
        heuristic still gets them right. Pins current, correct behavior
        so a future fix does not regress it."""
        result = _stacked_two_storey_building()
        all_cams = CAMS_STOREY_1 + CAMS_STOREY_2
        detection = detect_planes(result, seed=42)
        oriented = classify_planes(
            detection.planes, all_cams, UP,
            plane_centroids=plane_centroids(result, detection.planes),
        )

        by_height = {
            round(_approx_height(o.plane), 3): o.role
            for o in oriented
            if _approx_height(o.plane) == _approx_height(o.plane)  # not NaN
        }
        assert by_height[round(STOREY_1_FLOOR_Y, 3)] == "floor"
        assert by_height[round(STOREY_2_CEILING_Y, 3)] == "ceiling"

    def test_middle_slabs_classify_correctly(self):
        """The hard case: storey-1's ceiling and storey-2's floor sit
        closest to the global camera mean, which used to misclassify
        them as each other before classify_plane started orienting by
        the camera nearest each plane's own real inlier centroid
        instead."""
        result = _stacked_two_storey_building()
        all_cams = CAMS_STOREY_1 + CAMS_STOREY_2
        detection = detect_planes(result, seed=42)
        oriented = classify_planes(
            detection.planes, all_cams, UP,
            plane_centroids=plane_centroids(result, detection.planes),
        )

        by_height = {
            round(_approx_height(o.plane), 3): o.role
            for o in oriented
            if _approx_height(o.plane) == _approx_height(o.plane)  # not NaN
        }
        assert by_height[round(STOREY_1_CEILING_Y, 3)] == "ceiling"
        assert by_height[round(STOREY_2_FLOOR_Y, 3)] == "floor"
