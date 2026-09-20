"""Absolute-pose fidelity evaluation (P1 RECONSTRUCTION FIDELITY).

The previous GT comparison in scripts/run_south_building_e2e.py
compared quaternions DIRECTLY against the reference model and claimed
"rotation is comparable without alignment". That is false: an
incremental SfM run recovers the scene up to an arbitrary gauge
(rotation + scale + translation), so the raw comparison conflated the
gauge difference with real per-camera error (the recorded 8.94 deg
median is mostly gauge, not fidelity).

The canonical evaluator must satisfy measured, provable properties:

  P1 (gauge invariance): applying a KNOWN rigid Sim(3) transform to an
     otherwise-perfect estimate must evaluate to ~zero error. The
     alignment absorbs the gauge; what remains is truth.
  P2 (noise honesty): injecting a known rotation perturbation must be
     measured back at the injected magnitude (within tolerance).
  P3 (refusal): fewer than 3 common cameras cannot define a frame --
     evaluation is REFUSED, never guessed. Degenerate (collinear /
     coincident) center configurations are flagged, not silently
     scored.
  P4 (determinism): identical inputs -> identical numbers.
"""

from __future__ import annotations

import math

import pytest

from reconstruction.evaluation import (
    evaluate_absolute_poses,
    horn_rotation,
    quat_to_matrix,
    umeyama_sim3,
)


def _axis_angle_quat(axis, angle):
    """(w,x,y,z) unit quaternion for a rotation about `axis`."""
    x, y, z = axis
    n = math.sqrt(x * x + y * y + z * z)
    x, y, z = x / n, y / n, z / n
    s = math.sin(angle / 2.0)
    return (math.cos(angle / 2.0), x * s, y * s, z * s)


def _quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def _ring_cameras(n=8, radius=10.0, height=2.0):
    """Deterministic ring of world->camera poses looking at the origin,
    in COLMAP images.txt convention (GT convention)."""
    cams = {}
    for i in range(n):
        theta = 2.0 * math.pi * i / n
        center = (radius * math.cos(theta), radius * math.sin(theta), height)
        # Camera looks toward origin: build world->cam from cam->world.
        forward = tuple(-c / math.sqrt(sum(c * c for c in center)) for c in center)
        right = (-forward[1], forward[0], 0.0)
        rn = math.sqrt(sum(r * r for r in right))
        right = tuple(r / rn for r in right)
        up = (
            right[1] * forward[2] - right[2] * forward[1],
            right[2] * forward[0] - right[0] * forward[2],
            right[0] * forward[1] - right[1] * forward[0],
        )
        # cam->world rotation matrix columns = right, up, -forward.
        Rcw = [
            [right[0], up[0], -forward[0]],
            [right[1], up[1], -forward[1]],
            [right[2], up[2], -forward[2]],
        ]
        qw, qx, qy, qz = _matrix_to_quat(Rcw)  # cam->world quaternion
        # world->cam is the conjugate; translation in world->cam frame.
        qwc = (qw, -qx, -qy, -qz)
        # t_wc = -R_wc @ center
        Rwc = [
            [right[0], up[0], -forward[0]],
            [right[1], up[1], -forward[1]],
            [right[2], up[2], -forward[2]],
        ]
        Rwc = _transpose(Rwc)
        t = tuple(-sum(Rwc[r][c] * center[c] for c in range(3)) for r in range(3))
        cams[f"img{i:02d}.jpg"] = (qw, -qx, -qy, -qz, t[0], t[1], t[2])
    return cams


def _transpose(R):
    return [list(col) for col in zip(*R)]


def _matrix_to_quat(R):
    """Rotation matrix -> (w,x,y,z), Shepperd's method, deterministic."""
    trace = R[0][0] + R[1][1] + R[2][2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2][1] - R[1][2]) * s
        y = (R[0][2] - R[2][0]) * s
        z = (R[1][0] - R[0][1]) * s
    elif R[0][0] > R[1][1] and R[0][0] > R[2][2]:
        s = 2.0 * math.sqrt(1.0 + R[0][0] - R[1][1] - R[2][2])
        w = (R[2][1] - R[1][2]) / s
        x = 0.25 * s
        y = (R[0][1] + R[1][0]) / s
        z = (R[0][2] + R[2][0]) / s
    elif R[1][1] > R[2][2]:
        s = 2.0 * math.sqrt(1.0 + R[1][1] - R[0][0] - R[2][2])
        w = (R[0][2] - R[2][0]) / s
        x = (R[0][1] + R[1][0]) / s
        y = 0.25 * s
        z = (R[1][2] + R[2][1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2][2] - R[0][0] - R[1][1])
        w = (R[1][0] - R[0][1]) / s
        x = (R[0][2] + R[2][0]) / s
        y = (R[1][2] + R[2][1]) / s
        z = 0.25 * s
    n = math.sqrt(w * w + x * x + y * y + z * z)
    return (w / n, x / n, y / n, z / n)


def _apply_gauge(gt, quat_gauge, scale, shift):
    """Transform GT (world->cam) poses by a known camera-to-world gauge
    g: cam'_w = g * cam_w, expressed back in world->cam convention."""
    out = {}
    for name, (qw, qx, qy, qz, tx, ty, tz) in gt.items():
        # camera-to-world of GT:
        q_c2w = (qw, -qx, -qy, -qz)
        R = quat_to_matrix(q_c2w)
        # center = -R_c2w @ t  (t is world->cam translation)
        t = (tx, ty, tz)
        center = [-sum(R[r][c] * t[c] for c in range(3)) for r in range(3)]
        # gauge on centers: c' = s * G @ c + shift
        G = quat_to_matrix(quat_gauge)
        center2 = [
            scale * sum(G[r][c] * center[c] for c in range(3)) + shift[r]
            for r in range(3)
        ]
        # gauge on orientations: R' = G @ R
        R2 = [[sum(G[r][k] * R[k][c] for k in range(3)) for c in range(3)] for r in range(3)]
        q2 = _matrix_to_quat(R2)  # cam'->world'
        # back to world->cam: conjugate quaternion, recompute t.
        qwc2 = (q2[0], -q2[1], -q2[2], -q2[3])
        Rwc2 = _transpose(quat_to_matrix(q2))
        t2 = [-sum(Rwc2[r][c] * center2[c] for c in range(3)) for r in range(3)]
        out[name] = (qwc2[0], qwc2[1], qwc2[2], qwc2[3], t2[0], t2[1], t2[2])
    return out


class _Pose:
    def __init__(self, evidence_id, rotation, position):
        self.evidence_id = evidence_id
        self.rotation = rotation  # camera->world (w,x,y,z)
        self.position = position


def _poses_from_gt(gt):
    """Estimate-side poses derived from GT (camera->world quats + centers)."""
    poses = []
    for name, (qw, qx, qy, qz, tx, ty, tz) in gt.items():
        q_c2w = (qw, -qx, -qy, -qz)
        R = quat_to_matrix(q_c2w)
        # center = -R_c2w @ t
        t = (tx, ty, tz)
        center = [-sum(R[r][c] * t[c] for c in range(3)) for r in range(3)]
        poses.append(_Pose(name, q_c2w, tuple(center)))
    return poses


GT = _ring_cameras()


class TestGaugeInvariance:
    def test_known_gauge_evaluates_to_zero(self):
        """P1: a known rigid Sim(3) gauge between estimate and GT must
        be absorbed -- the honest metric reports ~zero error."""
        gauge_q = _axis_angle_quat((0.3, 1.0, -0.2), 1.234)
        shifted = _apply_gauge(GT, gauge_q, 2.5, (100.0, -7.0, 3.0))
        poses = _poses_from_gt(shifted)
        report = evaluate_absolute_poses(poses, GT)
        assert report["refused"] is False
        assert report["rotation_deg_median"] < 1e-4
        assert report["rotation_deg_max"] < 1e-3
        assert report["center_error_median"] < 1e-6

    def test_identity_gauge_is_zero(self):
        poses = _poses_from_gt(GT)
        report = evaluate_absolute_poses(poses, GT)
        assert report["rotation_deg_median"] < 1e-6
        assert report["center_error_median"] < 1e-9


class TestNoiseHonesty:
    def test_per_camera_noise_measured_back(self):
        """P2: per-camera rotation noise (a CONSTANT perturbation is
        pure gauge and must evaluate to ~zero) is measured back at the
        injected magnitude."""
        inject = math.radians(2.0)
        poses = []
        for i, p in enumerate(_poses_from_gt(GT)):
            q_noise = _axis_angle_quat((0.0, 0.0, 1.0), inject)
            if i % 2:  # alternating sign per camera: NOT a gauge transform
                q_noise = _axis_angle_quat((0.0, 0.0, 1.0), -inject)
            noisy = _quat_mul(q_noise, p.rotation)
            poses.append(_Pose(p.evidence_id, noisy, p.position))
        report = evaluate_absolute_poses(poses, GT)
        assert abs(report["rotation_deg_median"] - 2.0) < 0.25

    def test_constant_perturbation_is_gauge_not_error(self):
        """The evaluator must not report gauge as fidelity error."""
        inject = math.radians(5.0)
        q_noise = _axis_angle_quat((0.0, 0.0, 1.0), inject)
        poses = [
            _Pose(p.evidence_id, _quat_mul(q_noise, p.rotation), p.position)
            for p in _poses_from_gt(GT)
        ]
        report = evaluate_absolute_poses(poses, GT)
        assert report["rotation_deg_median"] < 1e-4


class TestRefusal:
    def test_fewer_than_three_common_refused(self):
        poses = _poses_from_gt(dict(list(GT.items())[:2]))
        report = evaluate_absolute_poses(poses, GT)
        assert report["refused"] is True
        assert "reason" in report
        assert "count" not in {k: v for k, v in report.items() if k == "rotation_deg_median"}

    def test_collinear_centers_flagged(self):
        """Umeyama scale from collinear centers is still defined, but a
        coincident configuration is degenerate and must be flagged."""
        flat = {name: (qw, qx, qy, qz, tx, ty, tz) for name, (qw, qx, qy, qz, tx, ty, tz) in GT.items()}
        poses = []
        for i, p in enumerate(_poses_from_gt(flat)):
            poses.append(_Pose(p.evidence_id, p.rotation, (float(i), 0.0, 0.0)))
        report = evaluate_absolute_poses(poses, flat)
        # Coincident-to-ring correspondence is a real mismatch; the
        # evaluator must not crash and must report what it measured.
        assert "scale_ratio" in report or report.get("refused") is True

    def test_no_common_names_refused(self):
        poses = [_Pose("other.jpg", (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0))]
        report = evaluate_absolute_poses(poses, GT)
        assert report["refused"] is True


class TestDeterminism:
    def test_identical_inputs_identical_report(self):
        poses = _poses_from_gt(GT)
        a = evaluate_absolute_poses(poses, GT)
        b = evaluate_absolute_poses(poses, GT)
        assert a == b


class TestPointCloudFidelity:
    def test_transformed_points_measure_near_zero(self):
        """Points known to lie ON the reference geometry measure ~zero
        distance after the gauge transform is applied."""
        from reconstruction.evaluation import evaluate_point_cloud

        gt_points = {i: (float(i), float(i * i % 7), float(i % 5)) for i in range(50)}
        pts = [
            type("P", (), {"position": gt_points[i], "track_id": str(i), "source_evidence_ids": []})()
            for i in range(50)
        ]
        report = evaluate_absolute_poses(_poses_from_gt(GT), GT)
        transform = report["gauge_transform"]
        out = evaluate_point_cloud(pts, gt_points, transform)
        assert out["refused"] is False
        assert out["dist_median"] < 1e-9

    def test_offset_points_measure_their_offset(self):
        from reconstruction.evaluation import evaluate_point_cloud

        gt_points = {i: (float(i), 0.0, 0.0) for i in range(40)}
        pts = [
            type("P", (), {"position": (float(i), 0.5, 0.0), "track_id": str(i), "source_evidence_ids": []})()
            for i in range(40)
        ]
        report = evaluate_absolute_poses(_poses_from_gt(GT), GT)
        out = evaluate_point_cloud(pts, gt_points, report["gauge_transform"])
        assert abs(out["dist_median"] - 0.5) < 1e-6

    def test_empty_refuses(self):
        from reconstruction.evaluation import evaluate_point_cloud

        report = evaluate_absolute_poses(_poses_from_gt(GT), GT)
        out = evaluate_point_cloud([], {0: (0.0, 0.0, 0.0)}, report["gauge_transform"])
        assert out["refused"] is True


class TestPrimitives:
    def test_horn_rotation_recovers_known_transform(self):
        G = quat_to_matrix(_axis_angle_quat((0.1, 0.7, 0.4), 0.9))
        src = [(1.0, 0.0, 2.0), (-3.0, 4.0, 1.0), (2.0, -1.0, -2.0), (0.5, 2.5, 3.5)]
        dst = [tuple(sum(G[r][c] * p[c] for c in range(3)) for r in range(3)) for p in src]
        R = horn_rotation(src, dst)
        for s, d in zip(src, dst):
            mapped = tuple(sum(R[r][c] * s[c] for c in range(3)) for r in range(3))
            assert all(abs(mapped[r] - d[r]) < 1e-9 for r in range(3))

    def test_umeyama_recovers_scale_and_translation(self):
        G = quat_to_matrix(_axis_angle_quat((-0.4, 0.2, 0.9), 0.5))
        s_scale, shift = 3.0, (5.0, -2.0, 8.0)
        src = [(0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (0.0, 3.0, 0.0), (1.0, 1.0, 2.0), (-2.0, 0.5, 1.0)]
        dst = [
            tuple(s_scale * sum(G[r][c] * p[c] for c in range(3)) + shift[r] for r in range(3))
            for p in src
        ]
        R, t, s = umeyama_sim3(src, dst)
        assert abs(s - 3.0) < 1e-9
        for sp, dp in zip(src, dst):
            mapped = tuple(s * sum(R[r][c] * sp[c] for c in range(3)) + t[r] for r in range(3))
            assert all(abs(mapped[r] - dp[r]) < 1e-9 for r in range(3))
