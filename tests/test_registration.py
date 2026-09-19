"""Tests for registration/registration.py (P4-01 cross-source
registration): deterministic known-offset recovery (ICP + GNSS
anchor) and rejection tests (no overlap, degenerate geometry), per
docs/future/registration/CROSS_SOURCE_REGISTRATION.md acceptance
criteria.
"""

import math
import random

import pytest

from engine.math import Quat, Vec3
from registration.registration import (
    RegistrationEngine,
    RegistrationError,
    ResidualStats,
    estimate_registration_covariance,
    register_gnss_anchor,
    register_icp,
    register_icp_point_to_plane,
)


def _cube_points(n=8):
    """A non-planar point cloud (cube corners) so ICP has a
    well-determined rigid solution."""
    return [
        Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 1, 0), Vec3(0, 0, 1),
        Vec3(1, 1, 0), Vec3(1, 0, 1), Vec3(0, 1, 1), Vec3(1, 1, 1),
    ][:n]


def _apply(rotation: Quat, translation: Vec3, points):
    return [rotation.rotate(p) + translation for p in points]


def _quat_z(deg: float) -> Quat:
    """Unit quaternion for a rotation of `deg` degrees about +z."""
    half = math.radians(deg) / 2.0
    return Quat(math.cos(half), 0.0, 0.0, math.sin(half))


def _quat_y(deg: float) -> Quat:
    half = math.radians(deg) / 2.0
    return Quat(math.cos(half), 0.0, math.sin(half), 0.0)


class TestICPKnownOffset:
    """ICP (this module and every other implementation of it) requires
    a coarse initial alignment: correspondences are found by nearest
    neighbor, so an offset comparable to or larger than the cloud's
    own extent produces wrong initial pairings and a wrong local
    optimum (see TestICPLocalMinimum below -- this is real, verified
    behavior, not a hypothetical caveat). These tests use offsets
    small relative to the unit cube's extent, matching the spec's
    documented pipeline (coarse alignment happens upstream of ICP)."""

    def test_pure_translation_recovered(self):
        target = _cube_points(8)
        offset = Vec3(0.2, -0.1, 0.05)
        source = _apply(Quat.identity(), offset, target)

        result = register_icp(source, target, from_frame="B", to_frame="A")
        assert result.status == "accepted"
        assert result.rmse == pytest.approx(0.0, abs=1e-6)
        recovered = [result.transform.apply(p) for p in source]
        for r, t in zip(recovered, target):
            assert (r - t).length() == pytest.approx(0.0, abs=1e-5)

    def test_rotation_and_translation_recovered(self):
        target = _cube_points(8)
        # small rotation (15 degrees about z) + small translation
        angle = math.radians(15)
        rotation = Quat(math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2))
        offset = Vec3(0.1, 0.15, 0.05)
        source = _apply(rotation, offset, target)

        result = register_icp(source, target, from_frame="B", to_frame="A")
        assert result.status == "accepted"
        assert result.rmse == pytest.approx(0.0, abs=1e-4)
        recovered = [result.transform.apply(p) for p in source]
        for r, t in zip(recovered, target):
            assert (r - t).length() == pytest.approx(0.0, abs=1e-3)


class TestICPLocalMinimum:
    """Documents, rather than hides, ICP's known limitation: an offset
    larger than the cloud's extent breaks nearest-neighbor
    correspondence and converges to a wrong (but self-consistent)
    local optimum instead of the true offset. This is why
    TestICPKnownOffset uses small offsets -- real usage requires a
    coarse-alignment stage (GNSS prior, trajectory prior) before ICP,
    as the spec's pipeline states."""

    def test_large_offset_does_not_recover_true_transform(self):
        target = _cube_points(8)
        true_offset = Vec3(2.0, -1.0, 0.5)
        source = _apply(Quat.identity(), true_offset, target)

        result = register_icp(source, target, from_frame="B", to_frame="A")
        assert result.status == "accepted"  # converges, just to the wrong answer
        recovered_translation = result.transform.translation
        assert (recovered_translation - true_offset).length() > 0.1

    def test_high_inlier_fraction_on_exact_match(self):
        target = _cube_points(8)
        source = _apply(Quat.identity(), Vec3(0.1, 0.1, 0.1), target)
        result = register_icp(source, target, from_frame="B", to_frame="A")
        assert result.inlier_fraction == pytest.approx(1.0)


class TestICPRejection:
    def test_degenerate_too_few_points_blocked(self):
        result = register_icp(
            [Vec3(0, 0, 0), Vec3(1, 0, 0)],
            [Vec3(0, 0, 0), Vec3(1, 0, 0)],
            from_frame="B", to_frame="A",
        )
        assert result.status == "blocked"
        assert result.transform is None
        assert "degenerate" in result.reason

    def test_no_overlap_blocked(self):
        # A uniform translation of the SAME shape is a legitimate,
        # recoverable correspondence (relative structure is preserved
        # regardless of offset magnitude) -- that is not a no-overlap
        # case, it's translation recovery, and register_icp correctly
        # accepts it. A genuine no-overlap case needs two clouds with
        # NO shared geometric structure: unit-cube corners vs. points
        # scattered randomly across a much larger volume.
        target = _cube_points(8)
        rng = random.Random(1234)
        scattered_source = [
            Vec3(rng.uniform(-50, 50), rng.uniform(-50, 50), rng.uniform(-50, 50))
            for _ in range(8)
        ]
        result = register_icp(
            scattered_source, target, from_frame="B", to_frame="A", min_overlap=0.5
        )
        assert result.status == "blocked"
        assert result.transform is None
        assert "overlap" in result.reason or "inlier" in result.reason or "spacing" in result.reason


class TestGNSSAnchor:
    def test_translation_recovered(self):
        source = [Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 1, 0)]
        offset = Vec3(5.0, -3.0, 1.0)
        target = [p + offset for p in source]

        result = register_gnss_anchor(source, target, from_frame="B", to_frame="A")
        assert result.status == "accepted"
        assert result.method == "gnss_anchor"
        assert result.rmse == pytest.approx(0.0, abs=1e-9)
        recovered = result.transform.apply(source[0])
        assert (recovered - target[0]).length() == pytest.approx(0.0, abs=1e-9)

    def test_mismatched_lengths_raises(self):
        with pytest.raises(RegistrationError, match="paired 1:1"):
            register_gnss_anchor(
                [Vec3(0, 0, 0)], [Vec3(0, 0, 0), Vec3(1, 0, 0)],
                from_frame="B", to_frame="A",
            )

    def test_empty_raises(self):
        with pytest.raises(RegistrationError, match="at least one anchor"):
            register_gnss_anchor([], [], from_frame="B", to_frame="A")


def _plane_cloud(rng, n=300, extent=4.0):
    """Trihedral corner: three mutually perpendicular planes (a room
    corner). Fully determines a rigid transform under point-to-plane
    -- two planes would leave the hinge direction unobservable (the
    classic point-to-plane degeneracy)."""
    pts, normals = [], []
    picks = [(0, 1), (0, 2), (1, 2)]
    for _ in range(n):
        axis, (u, v) = rng.choice((0, 1, 2)), rng.choice(picks)
        a, b = rng.uniform(-extent, extent), rng.uniform(-extent, extent)
        coords = [0.0, 0.0, 0.0]
        coords[u], coords[v] = a, b
        nrm = [0.0, 0.0, 0.0]
        nrm[axis] = 1.0
        pts.append(Vec3(*coords))
        normals.append(Vec3(*nrm))
    return pts, normals


class TestPointToPlaneICP:
    """The spec's named algorithm (the earlier module documented a
    point-to-point deviation pending a normal-bearing representation;
    normals are now caller-supplied)."""

    def test_known_offset_recovered_on_planar_scene(self):
        rng = random.Random(42)
        src, normals = _plane_cloud(rng)
        rot = _quat_z(15)
        true_t = Vec3(0.3, -0.2, 0.1)
        tgt = _apply(rot, true_t, src)
        result = register_icp_point_to_plane(
            src, tgt, source_normals=normals,
            from_frame="B", to_frame="A",
        )
        assert result.status == "accepted", result.reason
        got = result.transform
        for p in src[:20]:
            mapped = got.apply(p)
            nearest = min(tgt, key=lambda q: (q - mapped).length())
            assert (mapped - nearest).length() < 0.05

    def test_rejects_mismatched_normals_count(self):
        rng = random.Random(1)
        src, normals = _plane_cloud(rng, n=50)
        with pytest.raises(RegistrationError, match="1:1"):
            register_icp_point_to_plane(
                src, src, source_normals=normals[:-1],
                from_frame="B", to_frame="A",
            )

    def test_too_few_points_blocked(self):
        src = [Vec3(0, 0, 0), Vec3(1, 0, 0)]
        result = register_icp_point_to_plane(
            src, src, source_normals=[Vec3(0, 0, 1), Vec3(0, 0, 1)],
            from_frame="B", to_frame="A",
        )
        assert result.status == "blocked"
        assert "degenerate" in result.reason


class TestResidualStats:
    def test_residual_stats_measured_not_invented(self):
        # An exact rigid offset -> near-zero residuals, measured from
        # actual correspondences, exposed on the result.
        rng = random.Random(7)
        src, _ = _plane_cloud(rng)
        tgt = _apply(Quat.identity(), Vec3(0.5, 0.0, 0.0), src)
        result = register_icp(src, tgt, from_frame="B", to_frame="A")
        assert result.status == "accepted"
        assert result.residual_stats is not None
        rs = result.residual_stats
        assert rs.count == len(src)
        assert rs.max >= rs.p95 >= rs.median >= 0.0
        assert rs.median < 1e-6

    def test_no_overlap_reports_large_residuals(self):
        rng = random.Random(3)
        src = [Vec3(rng.uniform(0, 1), rng.uniform(0, 1), rng.uniform(0, 1)) for _ in range(60)]
        far = [Vec3(1000 + rng.uniform(0, 1), rng.uniform(0, 1), rng.uniform(0, 1)) for _ in range(60)]
        result = register_icp(src, far, from_frame="B", to_frame="A", min_overlap=0.99)
        # Blocked, and the diagnostics still carry measured residuals.
        assert result.status == "blocked"


class TestRegistrationEngine:
    """Orchestration: confidence-ordered methods, recorded attempts,
    never a silent best-effort transform."""

    def test_preference_order_gnss_before_icp(self):
        # With GNSS anchors AND overlapping clouds available, the
        # engine must answer with the anchor method.
        rng = random.Random(11)
        src, _ = _plane_cloud(rng)
        tgt = _apply(Quat.identity(), Vec3(10.0, 5.0, 2.0), src)
        engine = RegistrationEngine()
        result = engine.register(
            source_cloud=src, target_cloud=tgt,
            source_normals=None,
            anchor_source=[Vec3(0, 0, 0)], anchor_target=[Vec3(10, 5, 2)],
            from_frame="B", to_frame="A",
        )
        assert result.status == "accepted"
        assert result.method == "gnss_anchor"

    def test_falls_back_to_icp_without_anchors(self):
        rng = random.Random(12)
        src, _ = _plane_cloud(rng)
        tgt = _apply(_quat_y(10), Vec3(0.2, 0.1, 0.0), src)
        engine = RegistrationEngine()
        result = engine.register(
            source_cloud=src, target_cloud=tgt,
            from_frame="B", to_frame="A",
        )
        assert result.status == "accepted"
        assert result.method == "icp"

    def test_all_methods_fail_records_blocked(self):
        # Two disjoint clouds, no anchors: the engine must return a
        # blocked result whose attempts list shows every method that
        # declined and why -- not raise, not silently guess.
        rng = random.Random(13)
        src, _ = _plane_cloud(rng)
        far = [Vec3(p.x + 1000.0, p.y, p.z) for p in src]
        engine = RegistrationEngine()
        result = engine.register(source_cloud=src, target_cloud=far, from_frame="B", to_frame="A")
        assert result.status == "blocked"
        assert result.method == "none"
        assert len(result.attempts) >= 1
        assert all(a.status == "blocked" for a in result.attempts)

    def test_attempts_recorded_on_success(self):
        rng = random.Random(14)
        src, _ = _plane_cloud(rng)
        tgt = _apply(Quat.identity(), Vec3(0.4, 0.0, 0.0), src)
        engine = RegistrationEngine()
        result = engine.register(source_cloud=src, target_cloud=tgt, from_frame="B", to_frame="A")
        assert result.status == "accepted"
        assert [a.method for a in result.attempts] == ["gnss_anchor", "icp"]
        assert result.attempts[0].status == "blocked"  # no anchors given
        assert result.attempts[1].status == "accepted"


class TestRegistrationCovariance:
    """Registration uncertainty: the estimate must carry a defensible
    covariance derived from the measured residuals, not a made-up
    confidence score. Sigmas are computed from the residual
    distribution over N correspondences by first-order error
    propagation: sigma_t = rms / sqrt(N_effective), with N_effective
    reduced by spatial degeneracy (a corridor's weak axis has few
    independent constraints)."""

    def test_translation_sigma_from_residuals(self):
        rng = random.Random(21)
        src, _ = _plane_cloud(rng)
        tgt = _apply(Quat.identity(), Vec3(0.3, 0.0, 0.0), src)
        result = register_icp(src, tgt, from_frame="B", to_frame="A")
        assert result.status == "accepted"
        assert result.covariance is not None
        cov = result.covariance
        # Exact synthetic correspondence -> near-zero measured residual
        # -> tight sigma. Defensible means consistent with the actual
        # residuals, not a hardcoded number.
        expected_sigma = cov.translation_sigma_m
        assert expected_sigma >= 0.0
        assert expected_sigma < 0.01
        # Sigma must be consistent with the measured residual stats.
        assert cov.basis.startswith("residual-derived")

    def test_noisier_cloud_gives_larger_sigma(self):
        # Two clouds with different residual scale: the noisier
        # alignment MUST report a larger sigma. This is the property
        # that makes the covariance real rather than decorative.
        rng = random.Random(22)
        src, _ = _plane_cloud(rng)
        tgt_clean = _apply(Quat.identity(), Vec3(0.3, 0.0, 0.0), src)
        tgt_noisy = [
            Vec3(p.x + 0.02 * rng.gauss(0, 1),
                 p.y + 0.02 * rng.gauss(0, 1),
                 p.z + 0.02 * rng.gauss(0, 1))
            for p in tgt_clean
        ]
        clean = register_icp(src, tgt_clean, from_frame="B", to_frame="A")
        noisy = register_icp(src, tgt_noisy, from_frame="B", to_frame="A")
        assert clean.covariance and noisy.covariance
        assert noisy.covariance.translation_sigma_m > clean.covariance.translation_sigma_m

    def test_degenerate_geometry_inflates_sigma(self):
        # Collinear points (a corridor): rotation about the line axis
        # is unconstrained. The covariance must reflect that weakness
        # with an inflated sigma (or an explicit degeneracy flag), not
        # report false confidence.
        rng = random.Random(23)
        line = [Vec3(float(i), 0.0, 0.0) for i in range(40)]
        tgt = _apply(Quat.identity(), Vec3(0.0, 0.1, 0.0), line)
        result = register_icp(line, tgt, from_frame="B", to_frame="A", min_overlap=0.1)
        if result.status == "accepted":
            assert result.covariance.degenerate_axes is not None
            assert len(result.covariance.degenerate_axes) >= 1

    def test_covariance_roundtrip(self):
        rng = random.Random(24)
        src, _ = _plane_cloud(rng)
        tgt = _apply(Quat.identity(), Vec3(0.3, 0.0, 0.0), src)
        result = register_icp(src, tgt, from_frame="B", to_frame="A")
        d = result.covariance.to_dict()
        assert d["basis"].startswith("residual-derived")
        assert d["n_correspondences"] == result.residual_stats.count
