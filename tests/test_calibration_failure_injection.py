"""Failure injection for the calibration / export / reload lifecycle, through the real HTTP routes, the real world store and the real
evidence-world stage. The engines under it are synthetic doubles; every persistence, versioning, gate and refusal path is production code.
For each case: the persisted state, adoption (HEAD), export eligibility and recovery are asserted, not only the status code."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("trimesh")

from tests.test_evidence_world_api import A, _PAIRS, _dist, _gauge_run, _url, client  # noqa: E402,F401

ROOT = Path(__file__).resolve().parents[1]


def _anchors():
    return [_dist(i, *p, held=(i == 3)) for i, p in enumerate(_PAIRS)]


def _store(tmp_path, wid, run):
    from worldstore.store import WorldStore

    return WorldStore(tmp_path / "ws" / "evidence-world" / "tenant-a" / wid / run / "worldstore")


def _head(client, wid, run):
    return client.get(_url(wid, run, "readiness"), headers=A).json()["world"]["version_id"]


def test_concurrent_identical_calibration_requests_produce_exactly_one_new_version(client, tmp_path):
    wid = _gauge_run(client, tmp_path, "run-f1")
    gauge = _head(client, wid, "run-f1")
    body = {"distance_anchors": _anchors(), "expected_version_id": gauge}
    with ThreadPoolExecutor(max_workers=6) as pool:
        resps = list(pool.map(lambda _: client.post(_url(wid, "run-f1", "calibrate"), json=body, headers=A), range(6)))
    codes = sorted(r.status_code for r in resps)
    assert set(codes) <= {200, 409} and codes.count(200) >= 1, codes          # a loser may be refused while the winner commits; nobody errors
    store = _store(tmp_path, wid, "run-f1")
    versions = [v.version_id for v in store.list_versions()]
    assert len(versions) == 2 and versions[0] == gauge                          # the gauge version plus exactly ONE calibrated version
    assert store.verify_version(versions[1]) == []
    rd = client.get(_url(wid, "run-f1", "readiness"), headers=A).json()
    assert rd["calibration"]["status"] == "validated" and rd["calibration"]["parent_version"] == gauge and rd["metric_export_ready"] is True
    assert client.post(_url(wid, "run-f1", "calibrate"), json=body, headers=A).json()["idempotent"] is True       # a late retry is a no-op


def test_a_failed_calibration_persist_is_retried_to_exactly_one_version_with_no_duplicate_entities(client, tmp_path, monkeypatch):
    from worldstore.store import WorldStore

    wid = _gauge_run(client, tmp_path, "run-f2")
    gauge = _head(client, wid, "run-f2")
    n_entities = len(_store(tmp_path, wid, "run-f2").load_version(gauge).entities)
    real, calls = WorldStore.save_version, {"n": 0}

    def flaky(self, *a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("injected: disk full while persisting the calibrated version")
        return real(self, *a, **k)

    monkeypatch.setattr(WorldStore, "save_version", flaky)
    first = client.post(_url(wid, "run-f2", "calibrate"), json={"distance_anchors": _anchors(), "expected_version_id": gauge}, headers=A)
    assert first.status_code == 503 and _head(client, wid, "run-f2") == gauge                         # nothing adopted
    rd = client.get(_url(wid, "run-f2", "readiness"), headers=A).json()
    assert rd["metric_export_ready"] is False and rd["calibration"] is None                                       # export stays gated
    for _ in range(3):                                                                                           # repeated retries after the partial failure
        assert client.post(_url(wid, "run-f2", "calibrate"), json={"distance_anchors": _anchors(), "expected_version_id": gauge}, headers=A).status_code == 200
    versions = _store(tmp_path, wid, "run-f2").list_versions()
    assert len(versions) == 2                                                                                     # exactly one calibrated version
    assert len(_store(tmp_path, wid, "run-f2").load_version(versions[-1].version_id).entities) == n_entities    # no duplicated entities
    assert client.get(_url(wid, "run-f2", "readiness"), headers=A).json()["metric_export_ready"] is True


def test_a_corrupted_world_artifact_is_caught_on_fresh_process_reload_and_blocks_export(client, tmp_path):
    wid = _gauge_run(client, tmp_path, "run-f3")
    gauge = _head(client, wid, "run-f3")
    assert client.post(_url(wid, "run-f3", "calibrate"), json={"distance_anchors": _anchors(), "expected_version_id": gauge}, headers=A).status_code == 200
    store = _store(tmp_path, wid, "run-f3")
    head = store.list_versions()[-1].version_id
    digest = store._record(head)["artifact_hash"]
    victims = [f for f in (tmp_path / "ws").rglob("*") if f.is_file() and digest in f.name]
    assert victims, "the stored world artifact must be found to be corrupted"
    victims[0].write_bytes(b'{"corrupted": true}')
    code = ("import json,sys;sys.path.insert(0,sys.argv[1]);from worldstore.store import WorldStore;s=WorldStore(sys.argv[2]);"
            "print(json.dumps({'problems':s.verify_version(sys.argv[3])}))")
    out = subprocess.run([sys.executable, "-c", code, str(ROOT), str(store._root), head], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0 and json.loads(out.stdout)["problems"], out.stdout + out.stderr      # a NEW process detects the tampering
    rd = client.get(_url(wid, "run-f3", "readiness"), headers=A).json()
    assert rd["world"]["status"] in ("failed_verification", "failed") and rd["export_ready"] is False
    blocked = client.post(_url(wid, "run-f3", "export"), json={"format": "glb", "acknowledge_unscaled": True}, headers=A)
    assert blocked.status_code == 409 and "not exportable" in blocked.json()["detail"]                 # a damaged world is never exported


def test_an_export_that_crashes_midway_surfaces_as_an_error_leaves_no_artifact_and_the_next_export_works(client, tmp_path, monkeypatch):
    wid = _gauge_run(client, tmp_path, "run-f4")
    from sdk import reality

    export_dir = tmp_path / "ws" / "evidence-world" / "tenant-a" / wid / "run-f4" / "export-artifacts"
    before = sorted(p.name for p in export_dir.rglob("*")) if export_dir.exists() else []
    real, calls = reality.export, {"n": 0}

    def dying(world, fmt, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("injected: exporter crashed after writing part of the output")
        return real(world, fmt, **k)

    monkeypatch.setattr(reality, "export", dying)
    with pytest.raises(RuntimeError, match="injected"):  # the crash reaches the caller as a server error, never as a success response
        client.post(_url(wid, "run-f4", "export"), json={"format": "glb", "acknowledge_unscaled": True}, headers=A)
    after = sorted(p.name for p in export_dir.rglob("*")) if export_dir.exists() else []
    assert after == before                                                                                 # nothing half-written was left behind
    ok = client.post(_url(wid, "run-f4", "export"), json={"format": "glb", "acknowledge_unscaled": True}, headers=A)
    assert ok.status_code == 200 and ok.content[:4] == b"glTF" and client.get("/healthz").status_code == 200      # recovery: the next export works


def test_a_fusion_failure_in_the_stage_is_recorded_the_gate_does_not_pass_and_no_fused_surface_is_claimed(tmp_path, monkeypatch):
    pytest.importorskip("open3d")
    from PIL import Image

    import engine.multiengine.depth_fusion as df
    from apps.api import evidence_world_stage as ew
    from tests.test_lift_stage_e2e import DX, H, W, _camera, _Depth, _inputs, _per_view, _points
    from worldstore.store import WorldStore

    def boom(*a, **k):
        raise RuntimeError("injected fusion failure")

    monkeypatch.setattr(df, "fuse_depth_to_surface", boom)
    photos = []
    for n in ("a.png", "b.png"):
        p = tmp_path / n
        Image.new("RGB", (W, H), (120, 120, 120)).save(p)
        photos.append(str(p))
    r = ew.run_for_job(team_id="t1", world_id="w1", job_id="job-1", sources=photos, base=tmp_path / "ws", flow_inputs=_inputs(_Depth(_per_view())),
                       depth_engines=["fake_depth"], cameras={p: _camera(DX[os.path.basename(p)]) for p in photos}, points3d=_points())
    assert r["lifting"]["fusion"] == {"status": "failed", "error": "RuntimeError: injected fusion failure"}      # recorded, not swallowed
    assert r["world_ir"]["status"] == "persisted"                                                              # the objects are still there
    root = tmp_path / "ws" / "evidence-world" / "t1" / "w1" / "job-1"
    rd = ew.readiness(root)
    gates = {g["name"]: g["status"] for g in rd["quality"]["gates"]}
    assert gates["depth_fusion"] != "pass" and rd["quality"]["verdict"] != "READY"
    w = WorldStore(root / "worldstore").load_version(ew.head_version(root))
    assert "surf-fused-surface" not in w.entities                                                              # no fused surface was claimed
