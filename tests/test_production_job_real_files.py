"""REAL production job: actual photographs, the production job queue and entry point, real COLMAP and real models -- no engine doubles
and no deterministic test backend. Opt-in (it takes tens of minutes and needs the South Building photos, COLMAP and cached model weights):

    RE_REAL_PRODUCTION_TEST=1 python -m pytest tests/test_production_job_real_files.py -s

Dataset: ``datasets/south_building/images`` (EXTERIOR photographs of a building facade). There is NO independent physical ground truth for
it, so physical accuracy is asserted ``NOT_MEASURED`` -- this test proves the pipeline runs, persists, reloads and reports honestly, not
that the reconstruction is accurate."""

from __future__ import annotations

import glob
import hashlib
import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = [pytest.mark.slow, pytest.mark.integration]

ROOT = Path(__file__).resolve().parents[1]
PHOTOS = sorted(glob.glob(str(ROOT / "datasets" / "south_building" / "images" / "*.JPG")))[:10]
ENABLED = os.environ.get("RE_REAL_PRODUCTION_TEST") == "1"

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")
pytest.importorskip("aiosqlite")
requires_real = pytest.mark.skipif(
    not (ENABLED and len(PHOTOS) >= 6),
    reason="opt-in real-model test: set RE_REAL_PRODUCTION_TEST=1 and provide datasets/south_building/images (not run in CI)",
)


@pytest.fixture(scope="module")
def job_run(tmp_path_factory):
    """Run ONE real job and share its result across the assertions below."""
    if not (ENABLED and len(PHOTOS) >= 6):
        pytest.skip("opt-in real-model test (RE_REAL_PRODUCTION_TEST=1 and the South Building photos)")
    from fastapi.testclient import TestClient

    tmp = tmp_path_factory.mktemp("realjob")
    env = {
        "DATABASE_URL": f"sqlite+aiosqlite:///{(tmp / 'app.db').as_posix()}", "STORAGE_ROOT": str(tmp / "art"),
        "WORLDSTORE_ROOT": str(tmp / "ws"), "RE_EVIDENCE_WORLD": "1", "RE_EVIDENCE_WORLD_DEPTH_ENGINES": "midas,depth_anything",
    }
    saved = {k: os.environ.get(k) for k in (*env, "REALITY_TEST_BACKEND")}
    os.environ.update(env)
    os.environ.pop("REALITY_TEST_BACKEND", None)  # the REAL backend: no deterministic test double
    try:
        import apps.api.db as db_mod

        importlib.reload(db_mod)
        import apps.api.main as main_mod

        importlib.reload(main_mod)
        import apps.api.worldstore_service as ws

        ws._store = None
        H = {"x-tenant-id": "tenant-real"}
        with TestClient(main_mod.app) as c:
            sid = c.post("/api/sessions", json={"name": "real"}, headers=H).json()["id"]
            wid = c.post("/api/worlds", json={"name": "real"}, headers=H).json()["id"]
            assert c.post(f"/api/worlds/{wid}/attach/{sid}", headers=H).status_code == 200
            hashes = {}
            for p in PHOTOS:
                data = Path(p).read_bytes()
                r = c.post(f"/api/uploads?session_id={sid}", headers=H, files={"file": (os.path.basename(p), data, "image/jpeg")})
                assert r.status_code == 201, r.text
                hashes[os.path.basename(p)] = hashlib.sha256(data).hexdigest()
            jid = c.post(f"/api/sessions/{sid}/reconstruct", headers=H).json()["job_id"]
            t0, job = time.time(), {}
            while time.time() - t0 < 2400:
                job = c.get(f"/api/jobs/{jid}", headers=H).json()
                if job["status"] in ("succeeded", "partial", "failed", "cancelled"):
                    break
                time.sleep(5)
            rd = c.get(f"/api/worlds/{wid}/evidence-world/{jid}/readiness", headers=H)
            yield {"job": job, "job_id": jid, "world_id": wid, "tmp": tmp, "hashes": hashes, "readiness": rd,
                   "evidence": tmp / "ws" / "evidence-world" / "tenant-real" / wid / jid, "seconds": round(time.time() - t0)}
    finally:
        for k, v in saved.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


@requires_real
def test_the_real_job_settles_in_an_honest_state_and_reports_the_evidence_stage(job_run):
    job = job_run["job"]
    assert job["status"] in ("succeeded", "partial"), (job["status"], job.get("error"))  # failed/cancelled/queued are NOT acceptable here
    ew = job["payload"]["evidence_world"]
    assert ew["status"] in ("complete", "partial") and ew["run_id"] == job_run["job_id"]
    print(f"\nREAL JOB: status={job['status']} outcome={job['payload'].get('outcome')} in {job_run['seconds']}s; stage={ew['status']}")


@requires_real
def test_model_outputs_are_persisted_and_the_metric3d_records_reload_in_a_fresh_process(job_run):
    from engine.multiengine.evidence_matrix import EvidenceMatrix

    ev = job_run["evidence"]
    report = json.loads((ev / "depth_report.json").read_text())
    assert {r["engine_id"] for r in report} == {"midas", "depth_anything"}
    for r in report:  # one at a time, each reported: success or an explicit failure, never skipped
        assert r["status"] == "succeeded", r
        assert r["accepted"] == len(job_run["hashes"]), r
        assert r["checkpoint_sha256"] is None or len(r["checkpoint_sha256"]) == 64
    code = (
        "import json,io,hashlib,numpy as np; from engine.multiengine.evidence_matrix import EvidenceMatrix; "
        "from engine.multiengine.metric3d import Metric3DObservation; "
        f"m=EvidenceMatrix(r'{ev}'); out=[]\n"
        "for o in m.store.query(task='metric3d'):\n"
        "    r=Metric3DObservation.from_observation(o); raw=m.artifacts.get(r.raw_artifact_ref); a=np.load(io.BytesIO(raw))\n"
        "    out.append({'state':r.scale_state.value,'units':r.units,'sha_ok':hashlib.sha256(raw).hexdigest()==r.raw_artifact_ref.removeprefix('artifact://'),"
        "'shape':list(a.shape),'summary_ok':(float(a.min()),float(np.median(a)),float(a.max()))==tuple(r.value),'src':r.source_id})\n"
        "print(json.dumps(out))"
    )
    done = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr[-500:]
    recs = json.loads(done.stdout.strip().splitlines()[-1])
    assert len(recs) == 2 * len(job_run["hashes"])
    assert all(r["sha_ok"] and r["summary_ok"] for r in recs)  # artifacts resolve after a fresh process; summaries match the pixels
    assert {r["state"] for r in recs} == {"relative_depth"} and {r["units"] for r in recs} == {"relative"}  # never metres from these models
    assert EvidenceMatrix(ev).rejected() is not None  # the rejected log is readable (it may be empty)


@requires_real
def test_originals_are_preserved_and_provenance_references_them(job_run):
    from engine.multiengine.evidence_matrix import EvidenceMatrix

    m = EvidenceMatrix(job_run["evidence"])
    sources = {os.path.basename(o.source_id) for o in m.store.query(task="metric3d")}
    assert sources == set(job_run["hashes"].values())  # every uploaded photo was consumed, identified by its content hash
    stored = {p.name: p for p in (job_run["tmp"] / "art").rglob("*") if p.is_file()}  # the upload store is content-addressed (name = sha256)
    for name, digest in job_run["hashes"].items():  # every original is still on disk and still hashes to what was uploaded
        assert digest in stored, name
        assert hashlib.sha256(stored[digest].read_bytes()).hexdigest() == digest


@requires_real
def test_the_persisted_world_state_is_reported_honestly_and_physical_accuracy_is_not_measured(job_run):
    ew = job_run["job"]["payload"]["evidence_world"]
    assert ew["world_ir"]["status"] in ("persisted", "not_persisted"), ew["world_ir"]  # 'failed' would be a real defect to investigate
    rd = job_run["readiness"].json()
    assert rd["physical_accuracy"] == "NOT_MEASURED" and rd["scale"]["status"] == "unanchored"
    if ew["world_ir"]["status"] == "persisted":
        assert rd["world"]["entities"] == ew["world_ir"]["entities"] and rd["world"]["status"] == "persisted"
    else:  # e.g. an exterior scene where no detector found interior objects: nothing persisted, and the reason is on record
        assert ew["world_ir"]["reason"] and rd["world"]["status"] == "not_persisted"
    assert rd["export_ready"] is False or rd["world"]["entities_with_position"] > 0
    print("\nREADINESS:", json.dumps({k: rd[k] for k in ("stage_status", "scale", "world", "export_ready", "physical_accuracy")}, default=str))
