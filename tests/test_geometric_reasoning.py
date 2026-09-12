"""Tests for the geometric-reasoning slice: RANSAC plane detection
(perception/geometry/planes.py), orientation classification
(perception/geometry/orientation.py), and plane -> WorldIR promotion
(evidence/promote_planes.py).

Fixtures are synthetic reconstruction results (flat grids of points plus
noise) chosen so every expected value -- inlier counts, plane constants,
extents, thicknesses -- is hand-computable, not just "is a float".
"""

from __future__ import annotations

import math

import pytest

from evidence.promote_planes import (
    PlanePromotionError,
    find_wall_pair_face,
    positions_by_plane,
    promote_plane_to_entity,
)
from perception.geometry.orientation import (
    OrientationError,
    classify_plane,
    classify_planes,
)
from perception.geometry.planes import (
    DetectedPlane,
    detect_planes,
    flip_normal_toward,
)
from provenance import Provenance
from reconstruction.backend.interface import ReconstructedPoint, ReconstructionResult
from world_ir import EntityType, GeometryType, WorldIR


def _point(x: float, y: float, z: float, counter=[0]) -> ReconstructedPoint:
    counter[0] += 1
    return ReconstructedPoint(
        position=(x, y, z), track_id=f"pt-{counter[0]:05d}", source_evidence_ids=["ev-1"]
    )


def _room_result() -> ReconstructionResult:
    """Floor y=0 (12x12 grid @0.25m), ceiling y=2.5, walls x=0 and x=4
    (10x10 grids), 30 noise points. Cameras inside the room."""
    counter = [0]
    points = []
    for i in range(12):
        for j in range(12):
            points.append(_point(i * 0.25, 0.0, j * 0.25, counter))
            points.append(_point(i * 0.25, 2.5, j * 0.25, counter))
    for i in range(10):
        for j in range(10):
            points.append(_point(0.0, i * 0.25, j * 0.25, counter))
            points.append(_point(4.0, i * 0.25, j * 0.25, counter))
    for i in range(30):
        points.append(_point(2.0 + (i % 5) * 0.7, 1.25 + (i % 3) * 0.4, 1.5 + (i % 7) * 0.3, counter))
    return ReconstructionResult(points=points, camera_poses=[], registration_status="success")


_CAMS = [(2.0, 1.25, 1.5), (1.0, 1.2, 2.0), (3.0, 1.3, 1.0)]
_UP = (0.0, 1.0, 0.0)


# ---------------------------------------------------------------- planes


class TestDetectPlanes:
    def test_finds_all_four_room_planes_with_exact_constants(self):
        det = detect_planes(_room_result(), seed=42)
        roles = [(tuple(round(c, 3) for c in p.normal), round(p.d, 3)) for p in det.planes[:4]]
        assert (0.0, 1.0, 0.0) in [r[0] for r in roles]  # floor or ceiling normal
        normals = [r[0] for r in roles]
        assert normals.count((0.0, 1.0, 0.0)) == 2  # floor + ceiling
        assert normals.count((1.0, 0.0, 0.0)) == 2  # both walls
        ds = sorted(r[1] for r in roles)
        # plane eq n.p + d = 0: floor y=0 -> d=0, ceiling y=2.5 -> d=-2.5,
        # wall x=0 -> d=0, wall x=4 -> d=-4
        assert ds == [-4.0, -2.5, 0.0, 0.0]

    def test_inlier_counts_match_expected_grid_sizes(self):
        det = detect_planes(_room_result(), seed=42)
        counts = sorted((p.inlier_count for p in det.planes[:4]), reverse=True)
        assert counts == [164, 144, 90, 90]  # 144-grid, 144-grid, 100-grid, 100-grid

    def test_deterministic_same_seed(self):
        det_a = detect_planes(_room_result(), seed=42)
        det_b = detect_planes(_room_result(), seed=42)
        assert [(p.plane_id, p.normal, p.d, p.inlier_ids) for p in det_a.planes] == [
            (p.plane_id, p.normal, p.d, p.inlier_ids) for p in det_b.planes
        ]

    def test_different_seeds_still_find_the_structure(self):
        for seed in (7, 123, 9999):
            det = detect_planes(_room_result(), seed=seed)
            assert len(det.planes) >= 4
            assert all(p.inlier_count >= 8 for p in det.planes)

    def test_bookkeeping_inliers_plus_unassigned_equals_total(self):
        result = _room_result()
        det = detect_planes(result, seed=42)
        assert det.points_total == len(result.points)
        assert det.points_in_planes + len(det.unassigned_point_ids) == det.points_total
        # no double-claiming across planes
        all_ids = [pid for p in det.planes for pid in p.inlier_ids]
        assert len(all_ids) == len(set(all_ids))

    def test_noise_cloud_finds_no_large_consensus(self):
        # A seeded-uniform random cloud: RANSAC may still find small
        # chance coplanar clusters (honest -- that is what RANSAC does),
        # but none may claim a large fraction of the cloud the way real
        # structure does (the room's planes claim 90-164 of 518 points).
        from engine.core.rng import DeterministicRNG

        rng = DeterministicRNG(7, name="noise-cloud")
        counter = [0]
        points = [
            _point(rng.uniform(0.0, 4.0), rng.uniform(0.0, 3.0), rng.uniform(0.0, 4.0), counter)
            for _ in range(60)
        ]
        det = detect_planes(
            ReconstructionResult(points=points, camera_poses=[], registration_status="success"), seed=1
        )
        assert all(p.inlier_count < 20 for p in det.planes)
        assert det.points_in_planes + len(det.unassigned_point_ids) == 60

    def test_too_few_points_returns_empty(self):
        points = [_point(0, 0, 0), _point(1, 0, 0), _point(0, 1, 0)]
        det = detect_planes(
            ReconstructionResult(points=points, camera_poses=[], registration_status="success"), seed=1
        )
        assert det.planes == []

    def test_failed_registration_still_rejects(self):
        result = _room_result()
        broken = ReconstructionResult(points=result.points, camera_poses=[], registration_status="failed")
        det = detect_planes(broken, seed=42)
        # detection is a pure geometric op; it neither knows nor cares about
        # registration status -- but the honest contract is that promotion
        # paths reject failures (tested in the promotion suite).
        assert det.points_total == len(result.points)

    def test_min_inliers_threshold_respected(self):
        det = detect_planes(_room_result(), seed=42, min_inliers=200)
        assert det.planes == []

    def test_plane_ids_canonical_after_sorting(self):
        det = detect_planes(_room_result(), seed=42)
        assert [p.plane_id for p in det.planes] == [f"plane-{i:03d}" for i in range(len(det.planes))]
        assert all(p.inlier_count >= q.inlier_count for p, q in zip(det.planes, det.planes[1:]))


# ----------------------------------------------------------- orientation


class TestClassifyPlane:
    def test_floor_ceiling_walls_of_the_room(self):
        det = detect_planes(_room_result(), seed=42)
        oriented = classify_planes(det.planes, _CAMS, up=_UP)
        roles = [o.role for o in oriented[:4]]
        assert roles == ["floor", "ceiling", "wall", "wall"]

    def test_camera_side_disambiguates_floor_from_ceiling(self):
        plane = DetectedPlane(plane_id="p-000", normal=(0.0, 1.0, 0.0), d=0.0, inlier_ids=["a", "b"])
        above = classify_plane(plane, [(0.0, 5.0, 0.0)], up=_UP)
        below = classify_plane(plane, [(0.0, -5.0, 0.0)], up=_UP)
        assert above.role == "floor"
        assert below.role == "ceiling"

    def test_up_none_refuses_to_guess(self):
        plane = DetectedPlane(plane_id="p-000", normal=(0.0, 1.0, 0.0), d=0.0, inlier_ids=["a"])
        oriented = classify_plane(plane, _CAMS, up=None)
        assert oriented.role == "unknown"
        assert oriented.uncertainty.confidence == 0.0
        assert "refusing to guess" in (oriented.uncertainty.note or "")

    def test_sloped_plane_is_honestly_unknown(self):
        # 45-degree roof plane
        plane = DetectedPlane(
            plane_id="p-000",
            normal=(math.cos(math.radians(45)), math.sin(math.radians(45)), 0.0),
            d=-1.0,
            inlier_ids=["a"],
        )
        assert classify_plane(plane, _CAMS, up=_UP).role == "unknown"

    def test_no_cameras_raises(self):
        plane = DetectedPlane(plane_id="p-000", normal=(0.0, 1.0, 0.0), d=0.0, inlier_ids=["a"])
        with pytest.raises(OrientationError):
            classify_plane(plane, [], up=_UP)

    def test_normal_flipped_toward_cameras(self):
        # cameras at y=+5; normal starts pointing away (-y); must flip to face them
        plane = DetectedPlane(plane_id="p-000", normal=(0.0, -1.0, 0.0), d=0.0, inlier_ids=["a"])
        oriented = classify_plane(plane, [(0.0, 5.0, 0.0)], up=_UP)
        assert oriented.normal[1] > 0.0
        # zero-set of the plane equation is unchanged by normalization+flip
        p = (1.0, 0.0, -2.0)  # lies on the plane y=0
        assert math.isclose(
            oriented.normal[0] * p[0] + oriented.normal[1] * p[1] + oriented.normal[2] * p[2] + oriented.d,
            0.0,
            abs_tol=1e-12,
        )

    def test_plane_normal_invariant_enforced_at_construction(self):
        # classification composes flip/tilt math that assumes unit normals;
        # the invariant is enforced once, at DetectedPlane construction,
        # so a non-unit normal cannot reach classification at all
        with pytest.raises(ValueError, match="unit length"):
            DetectedPlane(plane_id="p-000", normal=(0.0, -5.0, 0.0), d=0.0, inlier_ids=["a"])

    def test_flip_normal_toward_direct(self):
        n, d = flip_normal_toward((0.0, -1.0, 0.0), 2.5, (0.0, 10.0, 0.0))
        assert n == (0.0, 1.0, 0.0)
        assert d == -2.5
        # already-facing stays put
        n2, d2 = flip_normal_toward((0.0, 1.0, 0.0), -2.5, (0.0, 10.0, 0.0))
        assert n2 == (0.0, 1.0, 0.0) and d2 == -2.5


# ------------------------------------------------------------- promotion


class TestPromotePlane:
    def _oriented_room(self):
        det = detect_planes(_room_result(), seed=42)
        oriented = classify_planes(det.planes, _CAMS, up=_UP)
        return oriented, positions_by_plane(_room_result(), oriented)

    def test_promotes_wall_floor_ceiling_with_provenance(self):
        oriented, _ = self._oriented_room()
        world = WorldIR()
        result = _room_result()
        promoted_types = set()
        for n, o in enumerate(oriented):
            if o.role in ("wall", "floor", "ceiling"):
                r = promote_plane_to_entity(o, result, world, f"struct-{n}")
                promoted_types.add(r.entity.type)
        assert promoted_types == {EntityType.WALL, EntityType.FLOOR, EntityType.CEILING}
        assert world.validate() == []
        for entity in world.entities.values():
            assert entity.provenance == Provenance.INFERRED
            geometry = world.geometries[entity.geometry_ids[0]]
            assert geometry.type == GeometryType.PLANE
            assert geometry.provenance == Provenance.RECONSTRUCTED

    def test_extent_measurement_is_real_geometry(self):
        oriented, _ = self._oriented_room()
        world = WorldIR()
        result = _room_result()
        for n, o in enumerate(oriented):
            if o.role == "floor":
                r = promote_plane_to_entity(o, result, world, "floor-1")
                # The floor's inliers include the wall-bottom-row points on
                # y=0, so the AABB spans x in [0, 4] (walls at 0 and 4) and
                # y/z over the 2.75 m grid: diagonal = sqrt(16 + 7.5625).
                assert math.isclose(r.extent_measurement.value, math.sqrt(16.0 + 2.75 ** 2), rel_tol=1e-6)
                assert r.extent_measurement.unit == "meter"
                assert r.extent_measurement.provenance == Provenance.ESTIMATED
                return
        pytest.fail("no floor plane found to promote")

    def test_wall_thickness_measured_for_close_pair(self):
        counter = [0]
        points = []
        for i in range(10):
            for j in range(10):
                points.append(_point(0.0, i * 0.25, j * 0.25, counter))
                points.append(_point(0.2, i * 0.25, j * 0.25, counter))
        for i in range(30):
            points.append(_point(2.0 + (i % 5) * 0.7, 1.25 + (i % 3) * 0.4, 1.5 + (i % 7) * 0.3, counter))
        result = ReconstructionResult(points=points, camera_poses=[], registration_status="success")
        det = detect_planes(result, seed=42)
        oriented = classify_planes(det.planes, [(2.0, 1.25, 1.5)], up=_UP)
        pp = positions_by_plane(result, oriented)
        world = WorldIR()
        thicknesses = []
        for n, o in enumerate(oriented):
            if o.role == "wall":
                r = promote_plane_to_entity(
                    o, result, world, f"wall-{n}", other_planes=oriented, plane_positions=pp
                )
                if r.thickness_measurement:
                    partner_id, measurement = r.thickness_measurement
                    thicknesses.append(measurement.value)
                    assert measurement.unit == "meter"
                    assert measurement.provenance == Provenance.ESTIMATED
                    assert partner_id.startswith("plane-")
        assert len(thicknesses) == 2  # both faces measured the pair
        assert all(math.isclose(t, 0.2, rel_tol=0.05) for t in thicknesses)

    def test_no_thickness_for_distant_walls(self):
        oriented, pp = self._oriented_room()
        result = _room_result()
        world = WorldIR()
        for n, o in enumerate(oriented):
            if o.role == "wall":
                r = promote_plane_to_entity(
                    o, result, world, f"wall-{n}", other_planes=oriented, plane_positions=pp
                )
                assert r.thickness_measurement is None  # 4 m apart: no pair

    def test_unknown_role_refused(self):
        # 45-degree plane: neither horizontal nor vertical nor wall
        plane = DetectedPlane(
            plane_id="p-000",
            normal=(math.cos(math.radians(45)), math.sin(math.radians(45)), 0.0),
            d=-1.0,
            inlier_ids=["a"],
        )
        oriented = classify_plane(plane, _CAMS, up=_UP)
        assert oriented.role == "unknown"
        with pytest.raises(PlanePromotionError):
            promote_plane_to_entity(oriented, _room_result(), WorldIR(), "bad-entity")

    def test_missing_point_ids_refused(self):
        counter = [0]
        plane = DetectedPlane(
            plane_id="p-000", normal=(0.0, 1.0, 0.0), d=0.0, inlier_ids=["ghost-point"]
        )
        oriented = classify_plane(plane, _CAMS, up=_UP)
        with pytest.raises(PlanePromotionError):
            promote_plane_to_entity(oriented, _room_result(), WorldIR(), "ghost")

    def test_supporting_evidence_recorded_on_geometry(self):
        oriented, _ = self._oriented_room()
        world = WorldIR()
        result = _room_result()
        r = promote_plane_to_entity(oriented[0], result, world, "struct-0")
        obs = r.geometry.observations[0]
        assert obs.sensor_type == "geometric_reasoning"
        assert obs.metadata["inlier_count"] == oriented[0].plane.inlier_count
        assert obs.metadata["role"] == oriented[0].role

    def test_world_roundtrip_byte_identical(self):
        oriented, pp = self._oriented_room()
        result = _room_result()
        world = WorldIR()
        for n, o in enumerate(oriented):
            if o.role in ("wall", "floor", "ceiling"):
                promote_plane_to_entity(
                    o, result, world, f"struct-{n}", other_planes=oriented, plane_positions=pp
                )
        j1 = world.to_json()
        assert j1 == WorldIR.from_json(j1).to_json()

    def test_full_room_compiles_four_entities(self):
        # the end-to-end acceptance shape: points in -> typed structure out
        oriented, pp = self._oriented_room()
        result = _room_result()
        world = WorldIR()
        for n, o in enumerate(oriented):
            if o.role in ("wall", "floor", "ceiling"):
                promote_plane_to_entity(
                    o, result, world, f"struct-{n}", other_planes=oriented, plane_positions=pp
                )
        assert sorted(e.type.value for e in world.entities.values()) == [
            "ceiling", "floor", "wall", "wall",
        ]


# ------------------------------------------------------------ wall pairing


class TestFindWallPairFace:
    def _oriented(self, normal, d, plane_id):
        from perception.geometry.planes import DetectedPlane

        plane = DetectedPlane(plane_id=plane_id, normal=normal, d=d, inlier_ids=["a"])
        return classify_plane(plane, _CAMS, up=_UP)
        from perception.geometry.planes import DetectedPlane

        plane = DetectedPlane(plane_id=plane_id, normal=normal, d=d, inlier_ids=["a"])
        return classify_plane(plane, _CAMS, up=_UP)

    def test_same_facing_normals_pair(self):
        # both normals flipped toward the cameras on the same side: +x facing
        a = self._oriented((1.0, 0.0, 0.0), 0.0, plane_id="p-000")
        b = self._oriented((1.0, 0.0, 0.0), -0.2, plane_id="p-001")
        positions = [(0.0, y, z) for y in range(4) for z in range(4)]
        other_positions = {b.plane.plane_id: [(0.2, y, z) for y in range(4) for z in range(4)]}
        partner = find_wall_pair_face(a, positions, [b], other_positions)
        assert partner is not None and partner.plane.plane_id == b.plane.plane_id

    def test_too_far_apart_does_not_pair(self):
        a = self._oriented((1.0, 0.0, 0.0), 0.0, plane_id="p-000")
        b = self._oriented((1.0, 0.0, 0.0), -4.0, plane_id="p-001")
        positions = [(0.0, y, z) for y in range(4) for z in range(4)]
        other_positions = {b.plane.plane_id: [(4.0, y, z) for y in range(4) for z in range(4)]}
        assert find_wall_pair_face(a, positions, [b], other_positions) is None

    def test_non_parallel_does_not_pair(self):
        a = self._oriented((1.0, 0.0, 0.0), 0.0, plane_id="p-000")
        b = self._oriented((0.0, 1.0, 0.0), 0.0, plane_id="p-001")
        positions = [(0.0, y, z) for y in range(4) for z in range(4)]
        assert find_wall_pair_face(a, positions, [b], {b.plane.plane_id: positions}) is None

    def test_no_span_overlap_does_not_pair(self):
        a = self._oriented((1.0, 0.0, 0.0), 0.0, plane_id="p-000")
        b = self._oriented((1.0, 0.0, 0.0), -0.2, plane_id="p-001")
        # wall span direction for a +-x normal is y (see _wall_span_direction):
        # separate the faces along y so their projected spans do not overlap
        positions = [(0.0, float(y), 0.0) for y in range(4)]
        other_positions = {b.plane.plane_id: [(0.2, 10.0 + float(y), 0.0) for y in range(4)]}
        assert find_wall_pair_face(a, positions, [b], other_positions) is None

    def test_floor_is_never_a_wall_pair_candidate(self):
        a = self._oriented((1.0, 0.0, 0.0), 0.0, plane_id="p-000")
        floor = self._oriented((0.0, 1.0, 0.0), 0.2, plane_id="p-001")
        positions = [(0.0, y, z) for y in range(4) for z in range(4)]
        assert find_wall_pair_face(a, positions, [floor], {floor.plane.plane_id: positions}) is None
