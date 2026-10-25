"""Dense Level 3 is an evidence-driven, judged refinement: it is attempted only when the measured sparse model
justifies it and it is kept only when it is a better description of the scene than the sparse world.

Model-free: the sparse cloud, the dense cloud and the COLMAP dense chain are stand-ins with KNOWN geometry, so every
decision is provable. The six product cases are A..F below; the sparse world must survive every one of them.
"""

from __future__ import annotations

import math
import struct
from types import SimpleNamespace

import numpy as np
import pytest

from engine.pipeline import dense_gate as dg
from engine.pipeline import dense_judge as dj
from reconstruction.backend.interface import ReconstructedPoint
from reconstruction.dense_pipeline import DenseMVSRunError
from tests.test_dense_mvs_stage import _options, _patch_probe, _patch_run

RNG = np.random.default_rng(7)


def _surfaces(n: int, *, patches: bool) -> np.ndarray:
    """Points on a floor (y=0) and a wall (z=-2). ``patches`` = a sparse SfM-like cloud: dense clusters on the
    textured parts only, leaving bare stretches unsampled (which is where dense adds surface)."""
    if patches:
        u = np.concatenate([RNG.uniform(-2.0, -1.2, n // 2), RNG.uniform(1.2, 2.0, n - n // 2)])
    else:
        u = RNG.uniform(-2.0, 2.0, n)
    v = RNG.uniform(-2.0, 2.0, n)
    floor = np.stack([u[: n // 2], np.zeros(n // 2), v[: n // 2]], 1)
    wall = np.stack([u[n // 2:], RNG.uniform(0.0, 2.5, n - n // 2), np.full(n - n // 2, -2.0)], 1)
    return np.concatenate([floor, wall]) + RNG.normal(0, 0.003, (n, 3))


def _ply(xyz: np.ndarray) -> bytes:
    head = ("ply\nformat binary_little_endian 1.0\nelement vertex %d\nproperty float x\nproperty float y\n"
            "property float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
            % len(xyz)).encode("ascii")
    return head + b"".join(struct.pack("<3f3B", *map(float, p), 200, 200, 200) for p in xyz)


def _result(sparse: np.ndarray, n_cameras: int = 3):
    pts = [ReconstructedPoint(position=tuple(map(float, p)), track_id=f"t{i}", source_evidence_ids=["img-0", "img-1"])
           for i, p in enumerate(sparse)]
    cams = [SimpleNamespace(evidence_id=f"img-{i}") for i in range(n_cameras)]
    return SimpleNamespace(points=pts, camera_poses=cams)


def _rot_z(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])


def _stage(tmp_path, monkeypatch, result, dense_xyz=None, *, rotation=None, error=None, **kw):
    import engine.pipeline.vertical_slice as vs
    from world_ir.world_v1 import WorldIR

    _patch_probe(monkeypatch)
    _patch_run(monkeypatch, tmp_path, fused_bytes=None if dense_xyz is None else _ply(dense_xyz), error=error)
    world = WorldIR(id="w")
    options = _options(tmp_path)
    before = len(result.points)
    facts = vs._dense_mvs_stage(result, world, options, "relative", None, frame_rotation=rotation, judge=True, **kw)
    return facts, world, options, before


# ---- the judge itself -------------------------------------------------------------------------------------

def test_judge_accepts_a_denser_cloud_that_reproduces_the_structure_and_adds_surface():
    v = dj.judge_dense(_surfaces(300, patches=True), _surfaces(6000, patches=False))
    assert v["verdict"] == dj.ACCEPTED and v["sparse_retained"] >= 0.9 and v["new_fraction"] >= 0.25


def test_judge_rejects_a_cloud_in_the_wrong_frame_and_a_cloud_in_the_wrong_scale():
    sparse, dense = _surfaces(300, patches=True), _surfaces(6000, patches=False)
    rotated = dj.judge_dense(sparse, dense @ _rot_z(90).T)
    assert rotated["verdict"] == dj.REJECTED and rotated["deciding"] == "regression"
    scaled = dj.judge_dense(sparse, dense * 10.0)
    assert scaled["verdict"] == dj.REJECTED and scaled["deciding"] == "regression"


def test_judge_calls_a_same_density_cloud_equivalent_and_an_unjudgeable_one_rejected():
    sparse = _surfaces(2000, patches=False)
    same = dj.judge_dense(sparse, _surfaces(2400, patches=False))
    assert same["verdict"] == dj.EQUIVALENT and same["deciding"] == "improvement"
    tiny = dj.judge_dense(sparse, _surfaces(10, patches=False))
    assert tiny["verdict"] == dj.REJECTED and tiny["deciding"] == "judgeable"
    assert dj.judge_dense([], _surfaces(500, patches=False))["verdict"] == dj.REJECTED


def test_pure_density_gain_counts_when_the_structure_is_retained_but_a_modest_one_does_not():
    sparse = _surfaces(200, patches=False)                    # already samples every surface: ~no NEW surface
    big = dj.judge_dense(sparse, _surfaces(3000, patches=False))            # x15 denser, same surfaces
    assert big["verdict"] == dj.ACCEPTED and big["new_fraction"] < dj.MIN_NEW_FRACTION
    assert "density gain alone" in big["why"]
    modest = dj.judge_dense(sparse, _surfaces(900, patches=False))          # x4.5: not enough on its own
    assert modest["verdict"] == dj.EQUIVALENT
    wrong_frame = dj.judge_dense(sparse, _surfaces(3000, patches=False) @ _rot_z(90).T)
    assert wrong_frame["verdict"] == dj.REJECTED                            # density never rescues a regression


# ---- the product cases A..F -------------------------------------------------------------------------------

def _ready(n_pts=600):
    cams = [SimpleNamespace(evidence_id=f"ev-{i}", position=(6 * math.sin(a), 0.0, -6 * math.cos(a)),
                            rotation=(1.0, 0.0, 0.0, 0.0))
            for i, a in enumerate(np.linspace(-1.0, 1.0, 6))]
    pts = [SimpleNamespace(position=(0.01 * i, 0.0, 0.0), source_evidence_ids=["a", "b", "c", "d"])
           for i in range(n_pts)]
    return SimpleNamespace(camera_poses=cams, points=pts)


def _auto(tmp_path, monkeypatch, res, available=True):
    from engine.pipeline.vertical_slice import VerticalSliceOptions, _auto_dense_options

    ws = tmp_path / "staging"
    (ws / "images").mkdir(parents=True, exist_ok=True)
    (ws / "sparse" / "0").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("engine.pipeline.guidance.coverage_degrees", lambda poses, center: {"degrees": 100.0})
    _patch_probe(monkeypatch, available)
    return _auto_dense_options(res, len(res.camera_poses),
                               VerticalSliceOptions(colmap_session=SimpleNamespace(staging=ws), dense_auto=True))


def test_A_insufficient_sparse_quality_stays_sparse(tmp_path, monkeypatch):
    opts, facts, scratch = _auto(tmp_path, monkeypatch, _ready(n_pts=40))
    assert not opts.dense_mvs_enabled and scratch is None and "not justified" in facts["decision"]


def test_B_sufficient_sparse_quality_attempts_dense_and_probes_compute(tmp_path, monkeypatch):
    opts, facts, scratch = _auto(tmp_path, monkeypatch, _ready())
    try:
        assert opts.dense_mvs_enabled and facts["escalate"] and "escalated" in facts["decision"]
    finally:
        import shutil
        shutil.rmtree(scratch, ignore_errors=True)
    opts, facts, scratch = _auto(tmp_path / "x", monkeypatch, _ready(), available=False)      # no dense-capable COLMAP
    assert not opts.dense_mvs_enabled and scratch is None and "compute unavailable" in facts["decision"]


def test_C_an_improving_dense_cloud_is_accepted_and_lands_in_the_world_frame(tmp_path, monkeypatch):
    R = _rot_z(90)                                       # the canonical-frame rotation the sparse cloud went through
    sparse_world = _surfaces(300, patches=True)
    colmap_raw = _surfaces(6000, patches=False) @ np.linalg.inv(R).T      # fused.ply is in COLMAP's raw frame
    facts, world, options, before = _stage(tmp_path, monkeypatch, _result(sparse_world), colmap_raw, rotation=R)
    assert facts["status"] == "ran" and facts["judge"]["verdict"] == dj.ACCEPTED
    assert facts["dense_points_added"] == 6000 and facts["geometry_id"] in world.geometries
    # the cloud in the WORLD is in the world frame (bounds match the sparse world's, not the raw frame's)
    g = world.geometries[facts["geometry_id"]]
    assert g.bounds_min.z == pytest.approx(-2.0, abs=0.05) and g.bounds_max.x == pytest.approx(2.0, abs=0.05)


def test_C2_dense_points_join_the_fusion_input_with_dense_track_ids(tmp_path, monkeypatch):
    result = _result(_surfaces(300, patches=True))
    facts, _, _, before = _stage(tmp_path, monkeypatch, result, _surfaces(6000, patches=False))
    assert facts["status"] == "ran" and len(result.points) == before + 6000
    assert result.points[-1].track_id.startswith("dense_mvs:")           # frozen points are re-identified by copy


def test_D_a_failed_dense_run_leaves_the_sparse_candidate_untouched(tmp_path, monkeypatch):
    result = _result(_surfaces(300, patches=True))
    facts, world, options, before = _stage(tmp_path, monkeypatch, result,
                                           error=DenseMVSRunError("patch_match_stereo failed (exit 1): CUDA OOM"))
    assert facts["status"] == "failed" and "DenseMVSRunError" in facts["note"]
    assert len(result.points) == before and not world.geometries


def test_E_a_regressing_dense_candidate_is_rejected_and_writes_nothing(tmp_path, monkeypatch):
    result = _result(_surfaces(300, patches=True))
    facts, world, options, before = _stage(tmp_path, monkeypatch, result,
                                           _surfaces(6000, patches=False) @ _rot_z(90).T)     # frame not applied
    assert facts["status"] == "rejected" and facts["judge"]["deciding"] == "regression"
    assert len(result.points) == before and not world.geometries


def test_F_an_equivalent_dense_candidate_preserves_the_existing_world(tmp_path, monkeypatch):
    sparse = _surfaces(2000, patches=False)
    result = _result(sparse)
    facts, world, _, before = _stage(tmp_path, monkeypatch, result, _surfaces(2400, patches=False))
    assert facts["status"] == "equivalent" and len(result.points) == before and not world.geometries


def test_an_operator_forced_dense_run_is_not_judged(tmp_path, monkeypatch):
    import engine.pipeline.vertical_slice as vs
    from world_ir.world_v1 import WorldIR

    _patch_probe(monkeypatch)
    _patch_run(monkeypatch, tmp_path)
    facts = vs._dense_mvs_stage(_result(_surfaces(300, patches=True)), WorldIR(id="w"), _options(tmp_path),
                                "relative", None)
    assert facts["status"] == "ran" and facts["judge"] is None


# ---- product-path policy and level bookkeeping ------------------------------------------------------------

@pytest.mark.parametrize("value,expected", [(None, True), ("", True), ("1", True), ("0", False), ("false", False),
                                            ("OFF", False), (" no ", False)])
def test_dense_is_on_by_default_and_only_an_explicit_opt_out_disables_it(value, expected):
    env = {} if value is None else {"REALITY_DENSE_AUTO": value}
    assert dg.dense_auto_enabled(env) is expected


def test_the_job_runner_uses_the_policy_not_a_hidden_switch():
    import inspect

    from apps.api import jobs

    src = inspect.getsource(jobs)
    assert "dense_auto=dense_auto_enabled()" in src and '.strip() == "1"' not in src


def test_a_judged_dense_candidate_is_recorded_as_such_and_keeps_the_sparse_level(tmp_path):
    from tests.test_dense_gate import _progressive

    gate = {"escalate": True, "reasons": ["every measured criterion passed"]}
    for status, outcome in (("rejected", "rejected"), ("equivalent", "equivalent")):
        res = _progressive(tmp_path / status, {"dense_gate": gate,
                                               "dense_mvs": {"status": status, "note": f"{status}: reason"}})
        assert res.level == 2
        att = [a for a in res.attempts if a["level"] == 3][0]
        assert att["outcome"] == outcome and f"{status}: reason" in att["detail"]


def test_a_judged_dense_candidate_does_not_degrade_the_world():
    from apps.api.jobs import _stage_facts_degraded

    assert _stage_facts_degraded({"dense_mvs": {"status": "rejected"}, "depth": {"status": "ran"}}) == []
    assert _stage_facts_degraded({"dense_mvs": {"status": "equivalent"}}) == []
    assert _stage_facts_degraded({"dense_mvs": {"status": "disabled"}}) == []       # not attempted != degraded
    assert _stage_facts_degraded({"dense_mvs": {"status": "failed"}}) == ["dense_mvs stage status 'failed'"]
