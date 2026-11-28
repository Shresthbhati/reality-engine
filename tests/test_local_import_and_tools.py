"""The offline "give it photos/video from this PC -> get a world" entry points, and honest COLMAP discovery.

Covers: COLMAP location order (env > saved setting > PATH > common dirs, with an exact 'not found' message), the
``/api/system/tools`` probe, and ``POST /api/reconstructions/from-path`` (folder, single video, refusals).
"""
from __future__ import annotations

import importlib
import os
from pathlib import Path

import numpy as np
import pytest

from reconstruction.backend import locate


@pytest.fixture()
def isolated(monkeypatch, tmp_path):
    """No COLMAP anywhere: empty PATH, no env, no common dirs, settings file in tmp."""
    for v in locate.ENV_VARS:
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    monkeypatch.setenv("REALITY_SETTINGS_FILE", str(tmp_path / "settings.json"))
    monkeypatch.setattr(locate, "_common_dirs", lambda: [])
    return tmp_path


def _fake_colmap(d: Path) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    f = d / locate._EXE
    f.write_text("not really colmap")
    return f


def test_not_found_names_every_place_it_looked(isolated):
    loc = locate.locate_colmap()
    assert not loc.found and loc.source == "not_found"
    text = locate.not_found_message()
    assert "PATH" in text and "common install" in text and "REALITY_COLMAP_PATH" in text


def test_order_env_then_settings_then_path_then_common(isolated, monkeypatch):
    common = _fake_colmap(isolated / "common")
    monkeypatch.setattr(locate, "_common_dirs", lambda: [str(isolated / "common")])
    assert locate.locate_colmap().source == "common_dir" and locate.locate_colmap().path == str(common)

    on_path = _fake_colmap(isolated / "onpath")
    monkeypatch.setenv("PATH", str(on_path.parent))
    assert locate.locate_colmap().source == "PATH"

    saved = _fake_colmap(isolated / "saved")
    locate.write_settings({"colmap_path": str(saved)})
    assert (locate.locate_colmap().source, locate.locate_colmap().path) == ("settings", str(saved))

    env = _fake_colmap(isolated / "env" / "bin")
    monkeypatch.setenv("REALITY_COLMAP_PATH", str(env.parent.parent))  # a folder containing bin/colmap
    got = locate.locate_colmap()
    assert got.source == "env:REALITY_COLMAP_PATH" and got.path == str(env)


def test_a_bad_configured_path_is_reported_not_silently_used(isolated, monkeypatch):
    monkeypatch.setenv("COLMAP_PATH", str(isolated / "nope"))
    loc = locate.locate_colmap()
    assert not loc.found and any("COLMAP_PATH" in w for w in loc.warnings)


def test_which_passes_other_binaries_through(isolated):
    assert locate.which("definitely-not-a-real-binary-xyz") is None
    assert locate.which(sys_exe()) is not None


def sys_exe() -> str:
    import sys
    return sys.executable


def test_backend_resolves_through_the_locator(isolated, monkeypatch):
    from reconstruction.backend.colmap_backend import ColmapReconstructionBackend

    ok, detail = ColmapReconstructionBackend().availability_probe()
    assert not ok and "COLMAP was not found" in detail
    fake = _fake_colmap(isolated / "x")
    monkeypatch.setenv("REALITY_COLMAP_PATH", str(fake))
    b = ColmapReconstructionBackend()
    assert b._colmap_binary == str(fake) and b.availability_probe()[0]


# ----------------------------------------------------------------------------- API

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")
pytest.importorskip("aiosqlite")
cv2 = pytest.importorskip("cv2")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture()
def api(isolated, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{(isolated / 'app.db').as_posix()}")
    monkeypatch.setenv("STORAGE_ROOT", str(isolated / "artifacts"))
    monkeypatch.setenv("WORLDSTORE_ROOT", str(isolated / "ws"))
    import apps.api.db as db_mod
    importlib.reload(db_mod)
    import apps.api.main as main_mod
    importlib.reload(main_mod)

    async def idle_worker(*a, **k):  # the reconstruction itself is covered by the real-engine journeys
        import asyncio
        await asyncio.Event().wait()

    monkeypatch.setattr(main_mod.jobrunner, "worker_loop", idle_worker)
    import apps.api.worldstore_service as ws
    ws._store = None
    with TestClient(main_mod.app) as c:
        yield c


def _textured(i: int, w=160, h=120) -> np.ndarray:
    rng = np.random.default_rng(i)
    img = cv2.GaussianBlur(rng.integers(0, 255, (h, w, 3), dtype=np.uint8), (0, 0), 1.2)
    return cv2.putText(img, str(i), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)


def _photos(folder: Path, n=4) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        cv2.imwrite(str(folder / f"p{i}.jpg"), _textured(i))
    return folder


def test_tools_probe_says_not_found_with_where_it_looked(api):
    t = api.get("/api/system/tools").json()
    assert t["colmap"]["found"] is False and t["colmap"]["source"] == "not_found"
    assert any("PATH" in x for x in t["colmap"]["tried"])
    assert "COLMAP was not found" in t["colmap"]["message"]
    assert t["video"]["available"] is True and "OpenCV" in t["video"]["backend"]


def test_saving_a_colmap_path_is_validated_and_takes_effect(api, isolated):
    assert api.put("/api/system/tools/colmap", json={"path": str(isolated / "nowhere")}).status_code == 422
    fake = _fake_colmap(isolated / "mycolmap")
    r = api.put("/api/system/tools/colmap", json={"path": str(fake)})
    assert r.status_code == 200 and r.json()["colmap"]["found"] and r.json()["colmap"]["source"] == "settings"
    cleared = api.put("/api/system/tools/colmap", json={"path": ""}).json()
    assert cleared["colmap"]["found"] is False


def test_import_a_folder_from_a_local_path(api, isolated):
    folder = _photos(isolated / "shots")
    r = api.post("/api/reconstructions/from-path", json={"path": str(folder), "name": "Kitchen"})
    assert r.status_code == 202, r.text
    body = r.json()
    assert body["imported_from"]["files"] == 4 and len(body["evidence"]) == 4 and body["created_world"]
    st = api.get(f"/api/worlds/{body['world_id']}/status").json()
    assert st["name"] == "Kitchen" and st["evidence_summary"]["count"] == 4
    # a second import joins the SAME world
    r2 = api.post("/api/reconstructions/from-path", json={"path": str(folder / "p0.jpg"), "world_id": body["world_id"]})
    assert r2.status_code == 202 and r2.json()["world_id"] == body["world_id"]


def test_import_a_video_from_a_local_path_samples_frames(api, isolated):
    vid = isolated / "walk.mp4"
    vw = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 120))
    assert vw.isOpened()
    for i in range(30):
        vw.write(_textured(i // 3))
    vw.release()
    r = api.post("/api/reconstructions/from-path", json={"path": str(vid)})
    assert r.status_code == 202, r.text
    types = [e["type"] for e in r.json()["evidence"]]
    assert types.count("video") == 1 and types.count("photo") >= 2  # the video is kept; its frames are photo evidence


def test_import_refusals_are_specific(api, isolated):
    assert api.post("/api/reconstructions/from-path", json={"path": str(isolated / "missing")}).status_code == 404
    empty = isolated / "empty"
    empty.mkdir()
    (empty / "notes.txt").write_text("x")
    r = api.post("/api/reconstructions/from-path", json={"path": str(empty)})
    assert r.status_code == 422 and "no photos or videos" in r.json()["detail"]


def test_only_this_machine_may_read_local_paths(api, isolated, monkeypatch):
    import apps.api.main as main_mod
    folder = _photos(isolated / "shots2")
    async def remote_app(scope, receive, send):  # the same app, but every HTTP peer is a LAN/internet address
        if scope["type"] == "http":
            scope = {**scope, "client": ("203.0.113.9", 5555)}
        await main_mod.app(scope, receive, send)

    with TestClient(remote_app) as remote:
        r = remote.post("/api/reconstructions/from-path", json={"path": str(folder)})
        assert r.status_code == 403
        assert remote.put("/api/system/tools/colmap", json={"path": ""}).status_code == 403


# ----------------------------------------------------------------------------- bugs found by the audit


def test_colmap_failure_excerpt_shows_the_fatal_line_not_the_progress_tail():
    from reconstruction.dense_pipeline import error_excerpt

    log = "F20261018 patch_match_cuda.cc:123] Check failed: cudaSuccess == cudaGetLastError() out of memory\n" + "\n".join(
        f"I20261018 cudacc.cc:52]  Sweep {i}: 1.0s" for i in range(80))
    out = error_excerpt(log)
    assert "out of memory" in out and "Sweep" not in out
    assert "Sweep 79" in error_excerpt("I only progress\nSweep 79")  # no error lines -> falls back to the tail


def test_missing_colmap_is_reported_as_the_cause_not_blamed_on_the_photos():
    from engine.pipeline.guidance import build_guidance
    from evidence.session import EvidenceItem, EvidenceKind
    from reconstruction.backend.colmap_backend import ColmapReconstructionBackend
    from reconstruction.orchestrator import ReconstructionOrchestrationError, ReconstructionOrchestrator

    attempts = [{"level": 2, "name": "sparse", "outcome": "failed",
                 "detail": "E: all candidate backends failed or declined: colmap declined: unavailable: COLMAP was not found. Looked in: PATH"}]
    kinds = [g["kind"] for g in build_guidance(
        level=0, entity_roles=[], names={}, input_ids=[], registered_ids=[], camera_poses=[], contribution_summary=None,
        bootstrap_facts={"fused_views": 6}, attempts=attempts)]
    assert "engine_unavailable" in kinds and "add_overlap" not in kinds and "engine_note" not in kinds
    # and the orchestrator error itself carries the decline reason instead of "no attempts made"
    orch = ReconstructionOrchestrator([ColmapReconstructionBackend(colmap_binary="no-such-colmap-binary-xyz")])
    items = [EvidenceItem(id=f"p{i}", kind=EvidenceKind.PHOTO, source_uri=f"file://p{i}.jpg") for i in range(3)]
    with pytest.raises(ReconstructionOrchestrationError) as exc:
        orch.run(items)
    assert "no attempts made" not in str(exc.value) and "declined" in str(exc.value) and "was not found" in str(exc.value)


def test_camera_artifact_records_the_real_image_size_or_nothing():
    import json
    from types import SimpleNamespace as NS

    from apps.api.jobs import _render_cameras_json, _uniform_image_size

    ev = lambda w, h: NS(metadata_json={"image_facts": {"width": w, "height": h}})  # noqa: E731
    assert _uniform_image_size([ev(960, 720), ev(960, 720)]) == (960, 720)
    assert _uniform_image_size([ev(960, 720), ev(1920, 1080)]) is None          # mixed -> unrecorded, not a guess
    assert _uniform_image_size([NS(metadata_json={}), ev(960, 720)]) is None    # one unknown -> unrecorded
    assert json.loads(_render_cameras_json([], "relative", None))["image_size"] is None
    assert json.loads(_render_cameras_json([], "relative", (960, 720)))["image_size"] == [960, 720]
