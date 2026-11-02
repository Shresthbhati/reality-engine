"""B-02 upload pre-flight, B-03 thumbnail, B-04 rename/delete, B-11 place summary on the world list.

Photographs are generated (noise images; one blurred copy) so decoding, sharpness and thumbnails are real.
"""

from __future__ import annotations

import io
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")
pytest.importorskip("aiosqlite")
PIL = pytest.importorskip("PIL")

from PIL import Image, ImageFilter  # noqa: E402

from tests.test_job_progress import _SlowBackend  # noqa: E402,F401
from tests.test_reconstruction_lifecycle_semantics import _wait, client  # noqa: E402,F401


def _noise_jpeg(seed: int, blur: float = 0.0, size: int = 256) -> bytes:
    import random

    rnd = random.Random(seed)
    img = Image.frombytes("RGB", (size, size), bytes(rnd.randrange(256) for _ in range(size * size * 3)))
    if blur:
        img = img.filter(ImageFilter.GaussianBlur(blur))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=92)
    return buf.getvalue()


def _tree_files(root: Path) -> set[str]:
    return {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()} if root.exists() else set()


def _make_place(client, tmp_path, monkeypatch, n=3, backend="tests.test_cli_compile:_TwoViewBackend"):
    monkeypatch.setenv("REALITY_TEST_BACKEND", backend)
    monkeypatch.setenv("WORLDSTORE_ROOT", str(tmp_path / "ws"))
    import apps.api.worldstore_service as ws_svc

    ws_svc._store = None
    sid = client.post("/api/sessions", json={"name": "S"}).json()["id"]
    wid = client.post("/api/worlds", json={"name": "Place"}).json()["id"]
    assert client.post(f"/api/worlds/{wid}/attach/{sid}").status_code == 200
    for i in range(n):
        r = client.post(f"/api/uploads?session_id={sid}",
                        files={"file": (f"p{i}.jpg", _noise_jpeg(i), "image/jpeg")})
        assert r.status_code == 201, r.text
    return sid, wid


# ------------------------------------------------------------------------------------------------- B-02


def test_check_flags_soft_duplicate_and_non_image_and_stores_nothing(client, tmp_path):
    storage = tmp_path / "artifacts"
    before = _tree_files(storage)
    files = [("files", (f"sharp{i}.jpg", _noise_jpeg(i), "image/jpeg")) for i in range(1, 4)]
    files.append(("files", ("soft.jpg", _noise_jpeg(9, blur=6), "image/jpeg")))
    files.append(("files", ("dup.jpg", _noise_jpeg(1), "image/jpeg")))
    files.append(("files", ("tiny.jpg", _noise_jpeg(5, size=32), "image/jpeg")))
    files.append(("files", ("notes.txt", b"hello", "text/plain")))
    r = client.post("/api/uploads/check", files=files)
    assert r.status_code == 200, r.text
    by = {f["name"]: f for f in r.json()["files"]}
    assert by["soft.jpg"]["relatively_soft"] is True
    assert by["sharp1.jpg"]["relatively_soft"] is False
    assert by["dup.jpg"]["duplicate_of"] == "sharp1.jpg"
    assert by["tiny.jpg"]["too_small"] is True
    assert by["notes.txt"]["kind"] != "photo"
    summary = r.json()["summary"]
    assert summary["verdict"] == "add_angles"  # 4 usable photographs < 8
    assert _tree_files(storage) == before, "a pre-flight check must not store anything"


def test_check_good_set_and_empty_request(client):
    files = [("files", (f"p{i}.jpg", _noise_jpeg(i), "image/jpeg")) for i in range(10)]
    r = client.post("/api/uploads/check", files=files)
    assert r.json()["summary"]["verdict"] == "good"
    assert client.post("/api/uploads/check", files=[]).status_code in (400, 422)


# ------------------------------------------------------------------------------------------- B-03 / B-11


def test_thumbnail_and_place_summary(client, tmp_path, monkeypatch):
    sid, wid = _make_place(client, tmp_path, monkeypatch, n=3)
    card = next(w for w in client.get("/api/worlds").json()["items"] if w["id"] == wid)
    assert card["photo_count"] == 3
    assert card["placed_count"] == 0 and card["quality_band"] == "unknown"
    assert card["thumbnail_url"] and card["thumbnail_url"].startswith(f"/api/worlds/{wid}/thumbnail")
    assert card["active_job"] is None
    assert card["updated_at"].endswith("+00:00"), "naive UTC must be labelled, or a browser reads it as local time"
    t = client.get(card["thumbnail_url"])
    assert t.status_code == 200 and t.headers["content-type"] == "image/jpeg"
    img = Image.open(io.BytesIO(t.content))
    assert max(img.size) <= 480

    empty = client.post("/api/worlds", json={"name": "Empty"}).json()["id"]
    assert client.get(f"/api/worlds/{empty}/thumbnail").status_code == 404
    ec = next(w for w in client.get("/api/worlds").json()["items"] if w["id"] == empty)
    assert ec["photo_count"] == 0 and ec["thumbnail_url"] is None

    job = _wait(client, client.post(f"/api/sessions/{sid}/reconstruct").json()["job_id"])
    assert job["status"] in ("succeeded", "partial")
    after = next(w for w in client.get("/api/worlds").json()["items"] if w["id"] == wid)
    assert after["quality_band"] in ("good", "fair", "weak")
    assert after["quality_band"] != "unknown"
    assert after["thumbnail_url"] != card["thumbnail_url"], "a new version must refresh the cached thumbnail"


def test_summary_shows_the_running_build(client, tmp_path, monkeypatch):
    sid, wid = _make_place(client, tmp_path, monkeypatch, backend="tests.test_job_progress:_SlowBackend")
    job_id = client.post(f"/api/sessions/{sid}/reconstruct").json()["job_id"]
    seen = None
    for _ in range(100):
        card = next(w for w in client.get("/api/worlds").json()["items"] if w["id"] == wid)
        if card["active_job"]:
            seen = card["active_job"]
            break
        time.sleep(0.05)
    assert seen and seen["id"] == job_id
    _wait(client, job_id)


# ------------------------------------------------------------------------------------------------- B-04


def test_rename_and_validation(client):
    wid = client.post("/api/worlds", json={"name": "Old"}).json()["id"]
    assert client.patch(f"/api/worlds/{wid}", json={"name": "New name"}).json()["name"] == "New name"
    assert client.get(f"/api/worlds/{wid}").json()["name"] == "New name"
    assert client.patch(f"/api/worlds/{wid}", json={"name": "   "}).status_code in (400, 422)
    assert client.patch("/api/worlds/wld_missing", json={"name": "x"}).status_code == 404


def test_delete_refused_while_building_then_removes_everything(client, tmp_path, monkeypatch):
    sid, wid = _make_place(client, tmp_path, monkeypatch, backend="tests.test_job_progress:_SlowBackend")
    job_id = client.post(f"/api/sessions/{sid}/reconstruct").json()["job_id"]
    assert client.delete(f"/api/worlds/{wid}").status_code == 409
    assert client.get(f"/api/worlds/{wid}").status_code == 200, "a refused delete must change nothing"
    _wait(client, job_id)

    assert client.delete(f"/api/worlds/{wid}").status_code == 204
    assert client.get(f"/api/worlds/{wid}").status_code == 404
    assert client.get(f"/api/sessions/{sid}").status_code == 404
    assert client.get(f"/api/jobs/{job_id}").status_code == 404
    assert all(w["id"] != wid for w in client.get("/api/worlds").json()["items"])
    assert client.delete(f"/api/worlds/{wid}").status_code == 404
    ws = tmp_path / "ws"
    assert not (ws / "colmap-sessions" / wid).exists()
    live_versions = [p for p in (ws / "versions").glob("v-*.json")] if (ws / "versions").exists() else []
    assert live_versions == [], "version records must be quarantined, not left in the lineage"


# ------------------------------------------------------------------------------------------------- B-07


def test_entity_band_rule():
    from apps.api.place_summary import entity_band

    assert entity_band(0.9, "OBSERVED") == "good"
    assert entity_band(0.75, "RECONSTRUCTED") == "good"
    assert entity_band(0.74, "RECONSTRUCTED") == "fair"
    assert entity_band(0.5, "INFERRED") == "fair"
    assert entity_band(0.49, "INFERRED") == "weak"
    assert entity_band(0.95, "CONFLICT") == "weak", "disagreeing sources are never good, whatever the score"
    assert entity_band(0.9, "UNKNOWN") == "unknown", "not measured is not a good score"
    assert entity_band(None, "OBSERVED") == "unknown"
    assert entity_band(float("nan"), "OBSERVED") == "unknown"
    assert entity_band(0.9, None) == "good"


def test_quality_endpoint(client, tmp_path, monkeypatch):
    sid, wid = _make_place(client, tmp_path, monkeypatch, n=3)
    assert client.get(f"/api/worlds/{wid}/quality").status_code in (404, 409), "no version yet"
    job = _wait(client, client.post(f"/api/sessions/{sid}/reconstruct").json()["job_id"])
    assert job["status"] in ("succeeded", "partial")
    q = client.get(f"/api/worlds/{wid}/quality").json()
    assert q["version_id"] and q["photos_total"] == 3
    assert q["band"] in ("good", "fair", "weak")
    assert set(q["entity_counts"]) == {"good", "fair", "weak", "unknown"}
    assert sum(q["entity_counts"].values()) == len(q["entities"])
    for e in q["entities"].values():
        assert e["band"] in ("good", "fair", "weak", "unknown")
        assert e["provenance"] == e["provenance"].upper()
    assert isinstance(q["open_conflicts"], int)


def test_band_reason_explains_the_band():
    from apps.api.place_summary import band_and_reason

    assert band_and_reason(None, 4, 0) == ("unknown", None)
    assert band_and_reason({"level": 3, "status": "SUCCESS"}, 10, 3) == ("weak", "few_placed")
    assert band_and_reason({"level": 1, "status": "SUCCESS"}, 1, 1) == ("weak", "rough_model")
    assert band_and_reason({"level": 3, "status": "SUCCESS"}, 10, 9) == ("good", None)
    assert band_and_reason({"level": 3, "status": "SUCCESS"}, 10, 6) == ("fair", "some_unplaced")
    assert band_and_reason({"level": 3, "status": "PARTIAL_SUCCESS"}, 10, 9) == ("fair", "partial_build")


# ------------------------------------------------------------------------------------- reconstruction diagnostics


def test_job_report_carries_reconstruction_diagnostics(client, tmp_path, monkeypatch):
    sid, wid = _make_place(client, tmp_path, monkeypatch, n=5)
    job = _wait(client, client.post(f"/api/sessions/{sid}/reconstruct").json()["job_id"])
    assert job["status"] in ("succeeded", "partial")
    stages = client.get(f"/api/worlds/{wid}/report").json()["stages"]
    assert "dense_gate" in stages and "dense_mvs" in stages, "the dense decision must be visible in the version report"
    rec = stages["reconstruction"]
    diag = rec["diagnostics"]
    assert diag and "error" not in diag, diag
    # the two-view fake registers 3 cameras: 2 of the 5 photographs are unplaced, and camera 3 sees no point
    assert diag["summary"]["registered"] == 3 and diag["summary"]["input"] == 5
    kinds = [d["type"] for d in diag["diagnostics"]]
    assert kinds.count("UNREGISTERED_IMAGE") == 2
    assert kinds.count("WEAKLY_SUPPORTED_CAMERA") >= 1
    assert all(d["severity"] in ("info", "warning", "error") for d in diag["diagnostics"])
