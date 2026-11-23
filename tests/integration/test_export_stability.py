"""Repeated and LARGE exports must never terminate (or leak into) the API process.

A real API server (uvicorn, its own process, file-backed SQLite and world store) serves the evidence-world export route while this test seeds a
run through the production stage, adds a LARGE version (thousands of entities), then exports it repeatedly in several formats. Asserted from the
outside: the process is still running, answers /healthz after every request, and its resident memory does not grow with the number of exports.
Reproduces the PR #372 browser session (API exit -1 after an export download) without a browser; the measured cause is recorded in the PR."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
psutil = pytest.importorskip("psutil")

pytestmark = pytest.mark.slow
ROOT = Path(__file__).resolve().parents[2]
TENANT = {"x-tenant-id": "tenant-a"}
N_BIG = 1500  # entities in the large version


def _http(method: str, url: str, body: dict | None = None, headers: dict | None = None, timeout: float = 120.0) -> tuple[int, bytes, dict]:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method=method, headers={"content-type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read(), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read(), dict(e.headers)


def _seed(tmp_path: Path, world_id: str) -> str:
    """A persisted evidence-world run (production stage, synthetic engines) plus a LARGE head version; returns the run id."""
    from PIL import Image

    from apps.api import evidence_world_stage as ew
    from tests.test_lift_stage_e2e import DX, H, W, _camera, _Depth, _inputs, _per_view, _points
    from world_ir.schema_v1 import Entity, EntityType, Geometry, GeometryType, Vector3
    from worldstore.store import WorldStore

    photos = []
    for n in ("a.png", "b.png"):
        p = tmp_path / n
        Image.new("RGB", (W, H), (120, 120, 120)).save(p)
        photos.append(str(p))
    ew.run_for_job(team_id="tenant-a", world_id=world_id, job_id="run-big", sources=photos, base=tmp_path / "ws", flow_inputs=_inputs(_Depth(_per_view())),
                   depth_engines=["fake_depth"], cameras={p: _camera(DX[os.path.basename(p)]) for p in photos}, points3d=_points())
    root = tmp_path / "ws" / "evidence-world" / "tenant-a" / world_id / "run-big"
    store = WorldStore(root / "worldstore")
    head = ew.head_version(root)
    world = store.load_version(head)
    for i in range(N_BIG):
        x, y, z = (i % 20) * 0.5, ((i // 20) % 20) * 0.5, (i // 400) * 0.5
        gid = f"geom-big-{i}"
        world.geometries[gid] = Geometry(id=gid, type=GeometryType.BOX, bounds_min=Vector3(x, y, z), bounds_max=Vector3(x + 0.3, y + 0.3, z + 0.3),
                                         quality_metrics={"scale_state": "sfm_gauge"})
        world.entities[f"ent-big-{i}"] = Entity(id=f"ent-big-{i}", type=EntityType.UNKNOWN, geometry_ids=[gid],
                                                transform={"position": {"x": x + 0.15, "y": y + 0.15, "z": z + 0.15}, "frame_id": "world", "scale_state": "sfm_gauge"})
    store.save_version(world, parent=head, version_id=head + "-big")
    return "run-big"


def test_repeated_and_large_exports_do_not_terminate_or_grow_the_api_process(tmp_path):
    env = {**os.environ, "DATABASE_URL": f"sqlite+aiosqlite:///{(tmp_path / 'app.db').as_posix()}", "STORAGE_ROOT": str(tmp_path / "artifacts"),
           "WORLDSTORE_ROOT": str(tmp_path / "ws"), "PYTHONUNBUFFERED": "1", "PYTHONPATH": str(ROOT)}
    port = 8390 + (os.getpid() % 90)
    log = tmp_path / "api.log"
    with open(log, "wb") as lf:
        proc = subprocess.Popen([sys.executable, "-c", f"import uvicorn; uvicorn.run('apps.api.main:app', host='127.0.0.1', port={port}, log_level='warning')"],
                                env=env, stdout=lf, stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(180):
            try:
                if _http("GET", f"{base}/healthz", timeout=2)[0] == 200:
                    break
            except Exception:  # noqa: BLE001
                time.sleep(0.5)
        else:
            pytest.fail("the API did not start: " + log.read_text(errors="replace")[-600:])
        code, raw, _ = _http("POST", f"{base}/api/worlds", {"name": "big"}, TENANT)
        assert code in (200, 201), raw
        wid = json.loads(raw)["id"]
        run = _seed(tmp_path, wid)
        ps = psutil.Process(proc.pid)
        rss = [ps.memory_info().rss]
        sizes = []
        for rnd in range(3):
            for fmt in ("glb", "gltf", "obj", "ply"):
                code, body, hdr = _http("POST", f"{base}/api/worlds/{wid}/evidence-world/{run}/export", {"format": fmt, "acknowledge_unscaled": True}, TENANT)
                assert code == 200, (rnd, fmt, code, body[:300])
                sizes.append(len(body))
                assert proc.poll() is None, f"the API process died during round {rnd} format {fmt} with exit code {proc.returncode}: " + log.read_text(errors="replace")[-600:]
                assert _http("GET", f"{base}/healthz", timeout=10)[0] == 200
                rss.append(ps.memory_info().rss)
        assert max(sizes) > 200_000                                        # the exports really were large
        growth_mb = (rss[-1] - rss[4]) / 2**20                             # after the first round: no growth proportional to the number of exports
        assert growth_mb < 250, f"RSS grew {growth_mb:.0f} MB over 8 further exports"
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
