"""Benchmark runner: PASSED only when the pipeline really processed the scene AND a metric was computed; BLOCKED/FAILED otherwise.

The pipelines used here are fakes that return a similarity-transformed copy of the ground truth, so the expected metric values are known exactly.
They are marked production=False and the aggregate says so; nothing here measures the real reconstruction pipeline (importing it pulls torch and
needs real photographs, neither available in this environment)."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.datasets import adapters as A
from benchmarks.datasets import manifest as M
from benchmarks.datasets import run_benchmark as R
from tests import benchmark_dataset_fixtures as F

REPO = Path(__file__).resolve().parent.parent
SCALE, SHIFT = 0.5, np.array([10.0, 0.0, 0.0])


def _rot_to_quat_wxyz(r: np.ndarray) -> tuple[float, float, float, float]:
    w = np.sqrt(max(0.0, 1 + r[0, 0] + r[1, 1] + r[2, 2])) / 2
    if w > 1e-6:
        return (w, (r[2, 1] - r[1, 2]) / (4 * w), (r[0, 2] - r[2, 0]) / (4 * w), (r[1, 0] - r[0, 1]) / (4 * w))
    x = np.sqrt(max(0.0, 1 + r[0, 0] - r[1, 1] - r[2, 2])) / 2          # 180-degree rotations (w = 0)
    return (0.0, x, r[0, 1] / (2 * x), r[0, 2] / (2 * x))


def fake_pipeline(sm: M.SceneManifest, *, drop: int = 0, production: bool = False, points=None, calls: list | None = None) -> R.PipelineHandle:
    """Returns the GT camera poses and points in a frame with scale 0.5 and a shift (a pure similarity of the truth)."""
    gt = {i.id: M.pose_c2w_opencv(i, sm.frame) for i in sm.images if i.pose_c2w is not None}

    def run(items):
        if calls is not None:
            calls.append(items)
        poses = []
        for eid, g in list(gt.items())[drop:]:
            poses.append((eid, tuple(SCALE * g[:3, 3] + SHIFT), _rot_to_quat_wxyz(g[:3, :3])))
        pts = points if points is not None else [(0, 0, 0), (4, 0, 0), (4, 3, 0), (0, 3, 0)]   # the GT quad's corners
        return SimpleNamespace(world=object(), camera_poses=tuple(poses), points=tuple(tuple(SCALE * np.array(p) + SHIFT) for p in pts),
                               registration_status="registered", scale_state="relative", cameras_registered=len(poses), cameras_input=len(items))
    return R.PipelineHandle("fake.pipeline", production, run)


def _scannetpp(tmp_path, split="test", n=6):
    sid = F.make_scannetpp(tmp_path, n_frames=n)
    sm = A.ScanNetPPAdapter(tmp_path, terms_acknowledged=True).load_scene(sid)
    sm.split = split
    return sm


def _scannet(tmp_path, split="test"):
    sid = F.make_scannet(tmp_path)
    sm = A.ScanNetAdapter(tmp_path, terms_acknowledged=True).load_scene(sid)
    sm.split = split
    return sm


def test_passed_when_processed_and_scored_with_known_values(tmp_path):
    sm = _scannetpp(tmp_path)
    calls: list = []
    r = R.run_scene(sm, tmp_path, fake_pipeline(sm, calls=calls))
    assert r["status"] == "PASSED" and r["processed"] is True and r["pipeline"] == {"entry": "fake.pipeline", "production": False}
    reg = r["metrics"]["camera_registration"]
    assert reg["status"] == "OK" and reg["ate_rmse_m"] < 1e-9 and reg["scale"] == pytest.approx(1 / SCALE) and reg["registered_fraction"] == 1.0
    assert reg["rotation_err_max_deg"] < 1e-3                       # OpenGL GT axes are known -> rotation reported and consistent with the conversion
    assert reg["gt_pose_provenance"] == ["estimated"]
    geo = r["metrics"]["geometry"]
    assert geo["status"] == "OK" and geo["chamfer_m"] < 1e-9 and geo["gt"]["provenance_class"] == "measured" and geo["fscore"]["0.05"]["fscore"] == 1.0
    for k in ("mesh_accuracy_completeness", "structure_completeness", "opening_completeness", "dimension_error"):
        assert r["metrics"][k]["status"] == "BLOCKED" and r["metrics"][k]["reason"]               # never a number, always a stated reason
    # the pipeline received evidence exactly as the API job builds it, without any GT intrinsics or poses
    items = calls[0]
    assert len(items) == len(sm.images) and all(it.kind.value == "photo" and it.source_uri.startswith("file:") and it.sha256 for it in items)
    assert all(not it.metadata for it in items)


def test_unknown_camera_axes_withhold_rotation_error(tmp_path):
    sm = _scannet(tmp_path)
    r = R.run_scene(sm, tmp_path, fake_pipeline(sm))
    reg = r["metrics"]["camera_registration"]
    assert r["status"] == "PASSED" and reg["ate_rmse_m"] < 1e-9
    assert "rotation_err_max_deg" not in reg and reg["rotation"].startswith("NOT_REPORTED")
    assert r["metrics"]["geometry"]["gt"]["provenance_class"] == "estimated"                       # ScanNet GT is a fused mesh, reported as such


def test_geometry_error_is_actually_measured(tmp_path):
    sm = _scannetpp(tmp_path)
    off = [(0, 0, 0.5), (4, 0, 0.5), (4, 3, 0.5), (0, 3, 0.5)]               # the GT corners lifted 0.5 m (in the GT scale)
    r = R.run_scene(sm, tmp_path, fake_pipeline(sm, points=off))
    geo = r["metrics"]["geometry"]
    # after the recovered x2 similarity each point sits exactly 0.5 m above its GT corner (the GT cloud is the 4 corners)
    assert geo["accuracy_m"] == pytest.approx(0.5, abs=1e-6) and geo["completeness_m"] == pytest.approx(0.5, abs=1e-6)
    assert geo["fscore"]["0.1"]["precision"] == 0.0 and geo["fscore"]["0.25"]["recall"] == 0.0


# ---------------------------------------------------------------------- BLOCKED / FAILED
def _never(*_a, **_k):
    raise AssertionError("the pipeline must not run for a BLOCKED scene")


def test_blocked_without_running_pipeline_for_bad_data(tmp_path):
    sm = _scannetpp(tmp_path)
    handle = R.PipelineHandle("never", False, _never)
    (tmp_path / sm.images[0].path).write_bytes(b"tampered")
    r = R.run_scene(sm, tmp_path, handle)
    assert r["status"] == "BLOCKED" and r["processed"] is False and "sha256" in r["reason"]
    (tmp_path / sm.images[1].path).unlink()
    assert "does not exist" in R.run_scene(sm, tmp_path, handle)["reason"]
    few = _scannetpp(tmp_path / "few", n=2)                          # 2 frames, 0 usable beyond: too few images
    r = R.run_scene(few, tmp_path / "few", handle)
    assert r["status"] == "BLOCKED" and "image(s)" in r["reason"]
    F.make_multiscan(tmp_path / "ms", with_frames=False)
    ms = A.MultiScanAdapter(tmp_path / "ms", terms_acknowledged=True).load_scene("scene_00000_00")
    r = R.run_scene(ms, tmp_path / "ms", handle)
    assert r["status"] == "BLOCKED" and "decode" in r["reason"]    # the video-only dataset layout is reported, not papered over


def test_blocked_when_pipeline_unavailable_and_default_loader_uses_the_production_entry(tmp_path, monkeypatch):
    sm = _scannetpp(tmp_path)

    def unavailable():
        raise R.PipelineUnavailable("no torch here")
    r = R.run_scene(sm, tmp_path, None, pipeline_loader=unavailable)
    assert r["status"] == "BLOCKED" and r["processed"] is False and "no torch" in r["reason"]
    monkeypatch.setitem(sys.modules, "engine.pipeline.vertical_slice", None)                       # make the import fail
    with pytest.raises(R.PipelineUnavailable):
        R.production_pipeline()
    r = R.run_scene(sm, tmp_path, None)
    assert r["status"] == "BLOCKED" and "cannot be imported" in r["reason"] and "pipeline" not in r


def test_production_entry_is_the_one_the_api_job_calls():
    """Static anchor (importing the real module needs ~20 s and torch): the entry exists, and apps/api/jobs.py calls the same function."""
    mod, fn = R.PIPELINE_ENTRY.rsplit(".", 1)
    src = (REPO / (mod.replace(".", "/") + ".py")).read_text(encoding="utf-8")
    assert re.search(rf"^def {fn}\(\s*evidence_items", src, re.M)
    jobs = (REPO / "apps" / "api" / "jobs.py").read_text(encoding="utf-8")
    assert "_vs.vertical_slice(evidence, vs_opts" in jobs and "EvidenceItem(id=ev.id, kind=EvidenceKind.PHOTO" in jobs


def test_failed_when_pipeline_raises_or_returns_nothing(tmp_path):
    sm = _scannetpp(tmp_path)

    def boom(items):
        raise RuntimeError("COLMAP could not register")
    r = R.run_scene(sm, tmp_path, R.PipelineHandle("x", True, boom))
    assert r["status"] == "FAILED" and r["processed"] is False and "COLMAP could not register" in r["reason"] and r["traceback_tail"]
    none = R.PipelineHandle("x", True, lambda items: SimpleNamespace(world=object(), camera_poses=(), points=()))
    assert R.run_scene(sm, tmp_path, none)["status"] == "FAILED"
    noworld = R.PipelineHandle("x", True, lambda items: SimpleNamespace(world=None, camera_poses=(("a", (0, 0, 0), (1, 0, 0, 0)),), points=()))
    assert R.run_scene(sm, tmp_path, noworld)["status"] == "FAILED"


def test_processed_but_nothing_scorable_is_blocked_not_passed(tmp_path):
    sm = _scannetpp(tmp_path)
    handle = fake_pipeline(sm)                                       # built while the GT still exists: the pipeline itself works fine
    sm.annotations, sm.data_kinds = [], ["real_image"]
    for i in sm.images:
        i.pose_c2w, i.pose_provenance = None, None
    r = R.run_scene(sm, tmp_path, handle)
    assert r["status"] == "BLOCKED" and r["processed"] is True and "no metric could be computed" in r["reason"]
    assert r["metrics"]["camera_registration"]["status"] == "BLOCKED" and r["metrics"]["geometry"]["status"] == "BLOCKED"


def test_partial_registration_is_reported_and_too_few_blocks_alignment(tmp_path):
    sm = _scannetpp(tmp_path, n=7)                                   # 6 usable images
    r = R.run_scene(sm, tmp_path, fake_pipeline(sm, drop=2))
    reg = r["metrics"]["camera_registration"]
    assert r["status"] == "PASSED" and reg["n_registered"] == 4 and reg["registered_fraction"] == pytest.approx(4 / 6)
    r = R.run_scene(sm, tmp_path, fake_pipeline(sm, drop=4))         # 2 cameras left: cannot align -> geometry BLOCKED too
    assert r["status"] == "BLOCKED" and r["processed"] is True
    assert "needs a Sim(3) alignment" in r["metrics"]["geometry"]["reason"]


def test_extractors_enable_structure_and_dimension_metrics(tmp_path):
    sm = _scannetpp(tmp_path)
    floor = {"kind": "floor", "center": (2, 1.5, 0), "u": (1, 0, 0), "v": (0, 1, 0), "half_u": 2, "half_v": 1.5}
    door = {"kind": "door", "center": (1, 0, 1), "width_m": 0.9, "height_m": 2.0}
    sm.annotations += [M.Annotation(id="st", kind="structure", provenance_class="fitted", data={"elements": [floor, door]}),
                       M.Annotation(id="dim", kind="dimension", provenance_class="measured", data={"name": "len", "value_m": 4.0}, uncertainty_m=0.005)]
    r = R.run_scene(sm, tmp_path, fake_pipeline(sm), structure_extractor=lambda res: [floor, door], dimension_extractor=lambda res: {"len": 4.01})
    assert r["status"] == "PASSED"
    assert r["metrics"]["structure_completeness"]["floor"]["completeness"] == pytest.approx(1.0)
    assert r["metrics"]["opening_completeness"]["recall"] == 1.0
    assert r["metrics"]["dimension_error"]["mae_m"] == pytest.approx(0.01)
    r2 = R.run_scene(sm, tmp_path, fake_pipeline(sm))                # without extractors the same GT yields BLOCKED, not zeros
    assert r2["metrics"]["structure_completeness"]["status"] == "BLOCKED" and r2["metrics"]["dimension_error"]["status"] == "BLOCKED"


# ---------------------------------------------------------------- run + aggregate JSON
def test_run_benchmark_writes_machine_readable_reports_and_never_blends(tmp_path):
    root = tmp_path / "data"
    a = _scannetpp(root / "pp", split="test")
    b = _scannet(root / "sn", split="test")
    skipped = _scannetpp(root / "pp2", split="development")
    out = tmp_path / "out"
    # one root per dataset in real life; here each manifest is run against its own root
    results = []
    for sm, rt in ((a, root / "pp"), (b, root / "sn")):
        results.append(R.run_scene(sm, rt, fake_pipeline(sm)))
    (root / "pp2" / skipped.images[0].path).write_bytes(b"tampered")                                    # data no longer matches its manifest
    results.append({**R.run_scene(skipped, root / "pp2", _scannet_never()), "split": "test"})            # BLOCKED, pipeline never called
    agg = R.aggregate(results, production=False)
    assert agg["production_pipeline"] is False and "NOT the production pipeline" in agg["warning"]
    assert agg["status_counts"] == {"PASSED": 2, "BLOCKED": 1, "FAILED": 0}
    assert [b["scene_id"] for b in agg["blocked"]] == [skipped.scene_id] and "sha256" in agg["blocked"][0]["reason"]
    keys = sorted(agg["groups"])
    assert len(keys) == 2 and all(k.startswith("split=test|data_kinds=") for k in keys)
    assert any("data_kinds=physical_gt+real_image" in k and "gt_provenance=estimated,fitted,measured" in k for k in keys)
    assert any("data_kinds=rgbd|gt_provenance=estimated" in k for k in keys)                            # RGB-D/estimated is NOT mixed with physical GT
    for g in agg["groups"].values():
        assert g["n_scenes"] == 1 and "camera_registration.ate_rmse_m" in g["mean"]
    # end-to-end writer
    agg2 = R.run_benchmark([a], root / "pp", out, splits=("test",), pipeline=fake_pipeline(a))
    assert json.loads((out / "aggregate.json").read_text(encoding="utf-8")) == agg2
    per = json.loads((out / "scenes" / f"{a.scene_id}.json").read_text(encoding="utf-8"))
    assert per["status"] == "PASSED" and per["scene_id"] == a.scene_id
    agg3 = R.run_benchmark([a, skipped], root / "pp", tmp_path / "out2", splits=("development",), pipeline=fake_pipeline(a))
    assert agg3["skipped_not_in_selected_splits"] == [a.scene_id] and agg3["n_scenes"] == 1


def _scannet_never():
    return R.PipelineHandle("never", False, _never)


def test_cli_main_writes_outputs_and_exit_code_reflects_failures(tmp_path, monkeypatch):
    sm = _scannetpp(tmp_path / "root")
    mdir = tmp_path / "m"
    mdir.mkdir()
    M.save_scene_manifest(sm, mdir / "s.json")
    monkeypatch.setattr(R, "production_pipeline", lambda: (_ for _ in ()).throw(R.PipelineUnavailable("none")))
    monkeypatch.setattr(R, "run_scene", lambda sm, root, pipeline, **kw: {"scene_id": sm.scene_id, "status": "BLOCKED", "reason": "x", "split": sm.split,
                                                                           "data_kind_key": "k", "gt_provenance_classes": []})
    assert R.main(["--manifests", str(mdir), "--root", str(tmp_path / "root"), "--out", str(tmp_path / "o")]) == 0
    assert json.loads((tmp_path / "o" / "aggregate.json").read_text())["status_counts"]["BLOCKED"] == 1
