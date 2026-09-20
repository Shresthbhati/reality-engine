"""Coverage recovery in the COLMAP backend (P1 fidelity, coverage gap).

Two measured defects this locks down:

  1. Robust SIFT defaults. On the committed real 32-photo south-building
     subset, baseline SIFT + exhaustive matching left 10/32 cameras
     unregistered (measured: 7 of them had ZERO verified matches to the
     main model -- repetitive architecture + downscale limits feature
     recall). COLMAP's affine-shape + domain-size-pooling SIFT and
     guided matching register all 32 (measured on this machine). The
     backend must default to the robust settings; the baseline remains
     opt-out.

  2. Multi-model merge. COLMAP's mapper may emit MULTIPLE sub-models
     (sparse/0, sparse/1, ...). The backend read only `0` and silently
     dropped the rest -- a direct no-silent-drop violation that also
     loses coverage. Sub-models are rigidly related; the canonical
     registration machinery (registration.cross_session) must resolve
     them, with refusals recorded, never identity placement.
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from evidence.session import EvidenceItem, EvidenceKind
from reconstruction.backend.colmap_backend import ColmapReconstructionBackend
from reconstruction.backend.interface import ReconstructionResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _photo(name: str) -> EvidenceItem:
    return EvidenceItem(
        id=name,
        kind=EvidenceKind.PHOTO,
        source_uri=f"file://C:/fake/{name}.jpg",
    )


def _ring_pose(name, radius=10.0, height=2.0, theta=None):
    """A camera pose on a ring (camera->world quat + center)."""
    if theta is None:
        theta = hash(name) % 360 / 180.0 * math.pi
    center = (radius * math.cos(theta), radius * math.sin(theta), height)
    forward = np.array([-c for c in center])
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    Rcw = np.column_stack([right, up, -forward])  # cam->world columns
    # Rotation matrix -> quaternion (w,x,y,z)
    trace = Rcw[0][0] + Rcw[1][1] + Rcw[2][2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (Rcw[2][1] - Rcw[1][2]) * s
        qy = (Rcw[0][2] - Rcw[2][0]) * s
        qz = (Rcw[1][0] - Rcw[0][1]) * s
    else:
        qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0
    return SimpleNamespace(
        evidence_id=name, position=center, rotation=(qw, qx, qy, qz),
        uncertainty=SimpleNamespace(confidence=0.9),
    )


def _axis_rotation_matrix(angle):
    """Rotation about Z by `angle` (3x3 numpy)."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _quat_to_mat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _mat_to_quat(R):
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


def _helix_points(n=60, scale=1.0):
    """A unique 3D helix curve: rotationally asymmetric, so ICP has a
    single tight optimum (a symmetric fixture let ICP 'align' a far
    cloud for wrong reasons -- measured). `scale` bakes a size gauge."""
    pts = []
    for i in range(n):
        th = math.pi * i / (n - 1)
        pts.append(SimpleNamespace(
            position=(scale * math.cos(th), scale * math.sin(th), scale * 0.15 * th),
            track_id=f"h{i:03d}",
            source_evidence_ids=[],
            uncertainty=SimpleNamespace(confidence=0.8),
        ))
    return pts


def _make_result(names, theta0=0.0, spread=2 * math.pi, jitter=0.0):
    poses = []
    points = []
    for i, n in enumerate(names):
        poses.append(_ring_pose(n, theta=theta0 + spread * i / max(1, len(names)) + jitter * i))
        # Points scattered around each camera's viewing axis (crude cloud).
        p = poses[-1].position
        # Helical height variation makes the cloud ASYMMETRIC -- a
        # rotationally-symmetric fixture lets ICP 'align' a far cloud
        # onto the arc for wrong reasons (measured: the symmetric
        # fixture aligned a 10000-offset model within gates).
        points.append(SimpleNamespace(
            position=(p[0] * 0.1, p[1] * 0.1, 0.4 * i),
            track_id=f"t-{n}",
            source_evidence_ids=[n],
            uncertainty=SimpleNamespace(confidence=0.8),
        ))
    return ReconstructionResult(points=points, camera_poses=poses, registration_status="partial")


def _submodel(names, gauge_R=None, gauge_t=None, point_scale=1.0):
    """A sub-model: ring cameras + helix point cloud, all baked into
    the frame given by the optional rigid gauge (R, t)."""
    base = _make_result(names)
    if gauge_R is None:
        points = [
            SimpleNamespace(position=p.position, track_id=p.track_id,
                            source_evidence_ids=p.source_evidence_ids,
                            uncertainty=SimpleNamespace(confidence=0.8))
            for p in _helix_points(scale=point_scale)
        ]
        return ReconstructionResult(points=points,
                                    camera_poses=list(base.camera_poses),
                                    registration_status="partial"), base
    gR = np.asarray(gauge_R, dtype=float)
    gt = np.asarray(gauge_t, dtype=float)
    points = []
    for p in _helix_points(scale=point_scale):
        p2 = gR @ np.array(p.position) + gt
        points.append(SimpleNamespace(position=tuple(p2), track_id=p.track_id,
                                      source_evidence_ids=p.source_evidence_ids,
                                      uncertainty=SimpleNamespace(confidence=0.8)))
    poses = []
    for pose in base.camera_poses:
        c2 = gR @ np.array(pose.position) + gt
        R2 = gR @ _quat_to_mat(pose.rotation)
        poses.append(SimpleNamespace(
            evidence_id=pose.evidence_id, position=tuple(c2),
            rotation=_mat_to_quat(R2),
            uncertainty=SimpleNamespace(confidence=0.9),
        ))
    return ReconstructionResult(points=points, camera_poses=poses,
                                registration_status="partial"), base


class TestRobustSiftDefaults:
    def test_defaults_enable_robust_sift(self):
        """Affine-shape + DSP extraction is ON by default: measured
        +22% verified points at unchanged fidelity on the real 32-photo
        subset."""
        from reconstruction.backend import colmap_backend

        assert colmap_backend.DEFAULT_ROBUST_SIFT is True
        args = colmap_backend._sift_extraction_args()
        assert "--SiftExtraction.estimate_affine_shape" in args
        assert args[args.index("--SiftExtraction.estimate_affine_shape") + 1] == "1"
        assert "--SiftExtraction.domain_size_pooling" in args
        assert args[args.index("--SiftExtraction.domain_size_pooling") + 1] == "1"

    def test_guided_matching_default_off(self):
        """Guided matching is OPT-IN ONLY: measured FALSE COVERAGE on
        repetitive architecture (31/32 cameras but 24.6 deg median
        rotation error vs 0.164 deg without -- the forced matches warp
        the model through similar facade windows)."""
        from reconstruction.backend import colmap_backend

        assert colmap_backend.DEFAULT_GUIDED_MATCHING is False
        args = colmap_backend._sift_matching_args()
        assert "--FeatureMatching.guided_matching" not in args
        opt_in = colmap_backend._sift_matching_args(guided=True)
        assert "--FeatureMatching.guided_matching" in opt_in

    def test_opt_out_restores_baseline(self):
        from reconstruction.backend import colmap_backend

        backend = ColmapReconstructionBackend(robust_sift=False)
        assert backend.robust_sift is False
        # Opt-out must actually remove the robust flags from the args.
        ex = colmap_backend._sift_extraction_args(robust=False)
        assert "--SiftExtraction.estimate_affine_shape" not in ex
        mt = colmap_backend._sift_matching_args(guided=False)
        assert "--FeatureMatching.guided_matching" not in mt

    def test_construction_stores_flags(self):
        backend = ColmapReconstructionBackend(robust_sift=True, guided_matching=True)
        assert backend.robust_sift is True
        assert backend.guided_matching is True


class TestMultiModelMerge:
    def test_single_model_unchanged(self, monkeypatch, tmp_path):
        """No sub-models on disk -> empty list (caller reports honestly)."""
        from reconstruction.backend import colmap_backend as cb

        models = cb._collect_submodels(Path(tmp_path) / "sparse", None)
        assert models == []

    def test_collect_submodels_sorted_and_text_converted(self, monkeypatch, tmp_path):
        """Sub-models are discovered in sorted numeric order and each is
        converted to TXT (model_converter) before parsing."""
        from reconstruction.backend import colmap_backend as cb

        sparse = Path(tmp_path) / "sparse"
        for m in ("1", "0", "2"):
            d = sparse / m
            d.mkdir(parents=True)
            (d / "images.txt").write_text("1 1 0 0 0 0 0 0 1 x.jpg\n", encoding="utf-8")
        converted = []

        def fake_convert(model_dir):
            converted.append(model_dir.name)

        monkeypatch.setattr(cb, "_convert_model_to_txt", fake_convert, raising=False)
        got = cb._collect_submodels(sparse, fake_convert)
        assert [m.name for m in got] == ["0", "1", "2"]
        assert converted == ["0", "1", "2"]

    def test_multi_model_results_are_rigidly_merged_not_dropped(self):
        """Multiple ReconstructionResults from sub-models become ONE via
        the canonical registration machinery; the recovered transform
        must be a genuine rotation+translation (not identity), and the
        second model's geometry must land back on the canonical
        geometry within the aligner's measured precision."""
        from reconstruction.merge import merge_submodel_results

        # Model 0: canonical frame. Model 1: the SAME helix geometry in
        # a rigidly transformed frame (180-deg Z rotation + offset).
        r0, _ = _submodel([f"a{i}" for i in range(6)])
        gauge_R = _axis_rotation_matrix(math.pi)
        gauge_t = np.array([40.0, -12.0, 5.0])
        r1, _ = _submodel([f"b{i}" for i in range(6)],
                          gauge_R=gauge_R, gauge_t=gauge_t)

        merged, report = merge_submodel_results({"0": r0, "1": r1}, reference="0")
        assert len(merged.camera_poses) == 12
        assert len(merged.points) == 120
        # The aligner must resolve the same-scene sub-model.
        assert report.status_by_session.get("1") == "aligned"
        t = report.transforms["1"]
        assert t is not None
        assert abs(t.rotation.w - 1.0) > 1e-6  # non-identity rotation
        # Model 1's points must land back ON the canonical helix within
        # the aligner's measured precision. Nearest-neighbor, not track
        # correspondence: a half-turn helix arc is exactly self-symmetric
        # (measured: the aligner may legitimately choose the symmetric
        # solution, permuting labels while preserving the geometry).
        from scipy.spatial import cKDTree

        canonical = np.array([p.position for p in _helix_points()])
        tree = cKDTree(canonical)
        merged_b = [p for p in merged.points if p.track_id.startswith("h")]
        moved_from_model1 = merged_b[60:]  # model 1 appended after model 0
        for p in moved_from_model1:
            d, _ = tree.query(np.array(p.position))
            assert d < 0.05

    def test_merge_refusal_excludes_with_reason(self):
        """When the aligner refuses under its gates, the second model's
        WRONG-FRAME data is EXCLUDED from the merged geometry and the
        refusal is reported -- never identity-placed, never silently
        dropped. A 100x-scale helix is rigidly unalignable onto the
        reference helix, so every ICP start must exceed the RMSE gate."""
        from reconstruction.merge import merge_submodel_results

        r0, _ = _submodel([f"a{i}" for i in range(6)])
        far_R = _axis_rotation_matrix(1.0)
        far, _ = _submodel([f"b{i}" for i in range(6)],
                           gauge_R=far_R, gauge_t=np.array([10000.0, 0.0, 0.0]),
                           point_scale=100.0)
        merged, report = merge_submodel_results({"0": r0, "1": far}, reference="0")
        assert report.status_by_session.get("1") == "unresolved"
        assert "1" in report.reasons and report.reasons["1"]
        # Only the reference model's geometry is in the merged result.
        assert len(merged.camera_poses) == 6
        assert len(merged.points) == 60
        assert merged.registration_status == "partial"

    def test_single_model_passthrough(self):
        from reconstruction.merge import merge_submodel_results

        r0 = _make_result([f"a{i}" for i in range(4)], theta0=0.0, spread=math.pi)
        merged, report = merge_submodel_results({"0": r0}, reference="0")
        assert len(merged.camera_poses) == 4
        assert merged.registration_status == r0.registration_status
        # The aligner records the reference frame itself.
        assert report.status_by_session == {"0": "reference"}


def _tiny_submodel(names, gauge_R, gauge_t, theta0=0.0):
    """A DEGENERATE sub-model: one point + one pose per name, rigidly
    placed. Mirrors the measured failure: a mapper fragment with a
    handful of points that ICP-fits any target with near-zero
    residual."""
    gR = np.asarray(gauge_R, dtype=float)
    gt = np.asarray(gauge_t, dtype=float)
    poses, points = [], []
    for i, n in enumerate(names):
        th = theta0 + 2.0 * math.pi * i / max(1, len(names))
        c = np.array([10.0 * math.cos(th), 10.0 * math.sin(th), 2.0])
        c2 = gR @ c + gt
        poses.append(SimpleNamespace(
            evidence_id=n, position=tuple(c2),
            rotation=_mat_to_quat(gR @ _quat_to_mat(_ring_pose(n, theta=th).rotation)),
            uncertainty=SimpleNamespace(confidence=0.8),
        ))
        h = _helix_points(n=60)[i * 7]
        p2 = gR @ np.array(h.position) + gt
        points.append(SimpleNamespace(
            position=tuple(p2), track_id=h.track_id,
            source_evidence_ids=[n], uncertainty=SimpleNamespace(confidence=0.8),
        ))
    return ReconstructionResult(points=points, camera_poses=poses,
                                registration_status="partial")


class TestMergeDiagnostics:
    def test_tiny_submodel_refused_not_merged(self):
        """Measured failure (run_20260920T200832): a 3-point sub-model
        ICP-snapped onto the 5k-point reference with zero residual and
        passed every gate, poisoning the whole gauge. A sub-model this
        degenerate CANNOT be verified as a same-scene registration, so
        it must be refused: excluded from geometry, reason recorded,
        merged status honest."""
        from reconstruction.merge import merge_submodel_results

        r0, _ = _submodel([f"a{i}" for i in range(6)])
        tiny = _tiny_submodel(
            [f"b{i}" for i in range(3)],
            gauge_R=_axis_rotation_matrix(1.1), gauge_t=np.array([30.0, 4.0, -2.0]),
        )
        assert len(tiny.points) == 3  # genuinely degenerate
        merged, report = merge_submodel_results({"0": r0, "1": tiny}, reference="0")
        assert report.status_by_session.get("1") == "unresolved"
        assert report.reasons.get("1")
        # Tiny model's geometry is EXCLUDED, not identity-placed.
        assert len(merged.camera_poses) == 6
        assert len(merged.points) == 60

    def test_real_submodel_still_merges(self):
        """The refusal must not throw out legitimate merges: a full
        60-point helix in a transformed frame still registers."""
        from reconstruction.merge import merge_submodel_results

        r0, _ = _submodel([f"a{i}" for i in range(6)])
        r1, _ = _submodel(
            [f"b{i}" for i in range(6)],
            gauge_R=_axis_rotation_matrix(math.pi),
            gauge_t=np.array([40.0, -12.0, 5.0]),
        )
        merged, report = merge_submodel_results({"0": r0, "1": r1}, reference="0")
        assert report.status_by_session.get("1") == "aligned"
        assert len(merged.camera_poses) == 12

    def test_merge_report_travels_on_result(self):
        """The merge diagnostics must SURVIVE into the returned result:
        the backend discarded `_merge_report`, so a false merge left no
        trace. Downstream (batch path, UI jobs panel) needs the record."""
        from reconstruction.merge import merge_submodel_results

        r0, _ = _submodel([f"a{i}" for i in range(6)])
        tiny = _tiny_submodel(
            [f"b{i}" for i in range(3)],
            gauge_R=_axis_rotation_matrix(0.7), gauge_t=np.array([9.0, 9.0, 9.0]),
        )
        merged, report = merge_submodel_results({"0": r0, "1": tiny}, reference="0")
        assert merged.merge_report is not None
        # The field is the serialized report (dict), ready for the
        # batch record / UI jobs panel.
        d = merged.merge_report
        assert d["status_by_session"].get("1") == "unresolved"
        assert "1" in d["reasons"]

    def test_passthrough_carries_no_report(self):
        """Single-model results must not pretend a merge happened."""
        from reconstruction.merge import merge_submodel_results

        r0 = _make_result([f"a{i}" for i in range(4)], theta0=0.0, spread=math.pi)
        merged, _ = merge_submodel_results({"0": r0}, reference="0")
        assert merged.merge_report is None
