"""Upload lifecycle contract (release gate, Phase 3): what goes in is what is stored, and bad input is refused.

Original bytes stay byte-identical (sha256 == input), hostile filenames cannot escape STORAGE_ROOT, size and
count limits answer 413, and EXIF + the original filename stay retrievable from GET /api/evidence/{id}.
"""
from __future__ import annotations

import hashlib
import importlib
import io
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")
pytest.importorskip("aiosqlite")
PIL = pytest.importorskip("PIL.Image")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{(tmp_path / 'app.db').as_posix()}")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("WORLDSTORE_ROOT", str(tmp_path / "ws"))
    import apps.api.db as db_mod

    importlib.reload(db_mod)
    import apps.api.main as main_mod

    importlib.reload(main_mod)
    with TestClient(main_mod.app) as c:
        yield c


def _jpeg_with_exif(seed: int = 7) -> bytes:
    import random

    rnd = random.Random(seed)
    img = PIL.new("RGB", (320, 240))
    img.putdata([(rnd.randrange(256), rnd.randrange(256), rnd.randrange(256)) for _ in range(320 * 240)])
    exif = img.getexif()
    exif[0x010F] = "TestCam"
    exif[0x0110] = "Model-X"
    buf = io.BytesIO()
    img.save(buf, "JPEG", exif=exif, quality=92)
    return buf.getvalue()


def test_original_bytes_are_stored_unchanged(client):
    data = _jpeg_with_exif()
    r = client.post("/api/uploads", files={"file": ("a.jpg", data, "image/jpeg")})
    assert r.status_code == 201, r.text
    assert r.json()["checksum"] == hashlib.sha256(data).hexdigest()
    art = client.get(f"/api/evidence/{r.json()['evidence_id']}/artifact")
    assert art.status_code == 200 and art.content == data


def test_hostile_filename_cannot_escape_storage_root(client, tmp_path):
    data = _jpeg_with_exif(1)
    r = client.post("/api/uploads", files={"file": ("../../../outside/evil.jpg", data, "image/jpeg")})
    assert r.status_code == 201, r.text
    ev = client.get(f"/api/evidence/{r.json()['evidence_id']}").json()
    assert ev["name"] == "evil.jpg"
    assert not list(tmp_path.parent.rglob("evil.jpg")) and not (tmp_path / "outside").exists()
    root = (tmp_path / "artifacts").resolve()
    for p in (tmp_path / "artifacts").rglob("*"):
        assert root in p.resolve().parents


def test_oversize_upload_is_refused_with_413(client, monkeypatch):
    monkeypatch.setenv("UPLOAD_MAX_BYTES", "1000")
    data = _jpeg_with_exif(2)
    assert len(data) > 1000
    assert client.post("/api/uploads", files={"file": ("big.jpg", data, "image/jpeg")}).status_code == 413
    r = client.post("/api/reconstructions", files=[("files", ("big.jpg", data, "image/jpeg"))])
    assert r.status_code == 413, r.text


def test_too_many_files_is_refused_with_413(client):
    from apps.api.routes_reconstructions import MAX_FILES_PER_REQUEST

    files = [("files", (f"f{i}.jpg", b"x", "image/jpeg")) for i in range(MAX_FILES_PER_REQUEST + 1)]
    assert client.post("/api/reconstructions", files=files).status_code == 413


def test_exif_and_original_filename_are_retrievable(client):
    data = _jpeg_with_exif(3)
    r = client.post("/api/reconstructions", files=[("files", ("IMG_0042.JPG", data, "image/jpeg"))])
    assert r.status_code == 202, r.text
    ev_id = r.json()["evidence"][0]["id"]
    ev = client.get(f"/api/evidence/{ev_id}").json()
    assert ev["name"] == "IMG_0042.JPG"
    assert ev["checksum"] == hashlib.sha256(data).hexdigest()
    exif = ev["metadata"]["image_facts"]["exif"]
    assert exif["make"] == "TestCam" and exif["model"] == "Model-X"
