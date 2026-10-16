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


# --------------------------------------------------------------------------- content-type boundary (Phase 5)


def _png() -> bytes:
    buf = io.BytesIO()
    PIL.new("RGB", (8, 8), (10, 200, 30)).save(buf, "PNG")
    return buf.getvalue()


def _img(fmt: str) -> bytes:
    buf = io.BytesIO()
    PIL.new("RGB", (16, 16), (90, 120, 200)).save(buf, fmt)
    return buf.getvalue()


MP4_HEAD = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom"
HEIC_HEAD = b"\x00\x00\x00\x18ftypheic\x00\x00\x00\x00mif1heic"


@pytest.mark.parametrize(
    "name, mime, data, kind",
    [
        ("a.jpg", "image/jpeg", _jpeg_with_exif(11), "photo"),
        ("a.png", "image/png", _png(), "photo"),
        ("a.webp", "image/webp", _img("WEBP"), "photo"),
        ("a.tif", "image/tiff", _img("TIFF"), "photo"),
        ("a.heic", "image/heic", HEIC_HEAD, "photo"),
        ("clip.mp4", "video/mp4", MP4_HEAD, "video"),
        ("scan.ply", "application/octet-stream", b"ply\nformat ascii 1.0\n", "point_cloud"),
        ("track.gpx", "application/gpx+xml", b"<?xml version='1.0'?><gpx></gpx>", "gnss"),
        ("log.csv", "text/csv", b"t,x\n0,1\n", "sensor_log"),
        # bytes win over labels: a JPEG sent as an opaque blob is still a photograph
        ("renamed.bin", "application/octet-stream", _jpeg_with_exif(12), "photo"),
    ],
    ids=["jpeg", "png", "webp", "tiff", "heic", "mp4", "ply", "gpx", "csv", "jpeg-sent-as-opaque-blob"],
)
def test_supported_formats_are_accepted_and_classified_by_their_bytes(client, name, mime, data, kind):
    r = client.post("/api/uploads", files={"file": (name, data, mime)})
    assert r.status_code == 201, r.text
    ev = client.get(f"/api/evidence/{r.json()['evidence_id']}").json()
    assert ev["type"] == kind
    assert client.get(f"/api/evidence/{ev['id']}/artifact").content == data       # accepted bytes are untouched


@pytest.mark.parametrize(
    "name, mime, data",
    [
        ("evil.jpg", "image/jpeg", b"<html><script>alert(1)</script></html>"),       # markup labelled as a photo
        ("fake.png", "image/png", b"MZ\x90\x00 this is an executable"),               # executable labelled as a photo
        ("clip.mp4", "video/mp4", b"not a video at all"),
        ("photo.jpg", "video/mp4", _jpeg_with_exif(13)),                               # a photo labelled as a video
        ("page.svg", "image/svg+xml", b"<svg xmlns='http://www.w3.org/2000/svg'><script/></svg>"),
        ("tool.exe", "application/octet-stream", b"MZ\x90\x00"),                       # unknown type
        ("archive.zip", "application/zip", b"PK\x03\x04"),
        ("scan.ply", "application/octet-stream", b"MZ\x90\x00 renamed executable"),    # allowed name, executable bytes
        ("track.csv", "text/csv", b"#!/bin/sh\nrm -rf /\n"),
        ("log.json", "application/json", b"<!DOCTYPE html><html></html>"),
    ],
    ids=["html-as-jpeg", "exe-as-png", "garbage-as-mp4", "photo-as-video", "svg", "exe", "zip", "exe-as-ply",
         "script-as-csv", "html-as-json"],
)
def test_unsupported_or_mislabelled_content_is_refused_with_415(client, name, mime, data):
    r = client.post("/api/uploads", files={"file": (name, data, mime)})
    assert r.status_code == 415, (name, r.status_code, r.text)
    # nothing was stored for it
    assert client.get("/api/evidence").json()["items"] == []


def test_unknown_types_can_be_allowed_explicitly(client, monkeypatch):
    monkeypatch.setenv("UPLOAD_ALLOW_ANY_TYPE", "1")
    assert client.post("/api/uploads", files={"file": ("notes.dat", b"abc", "application/octet-stream")}).status_code == 201
    # ...but executable/markup content and mislabelled photos stay refused
    assert client.post("/api/uploads", files={"file": ("x.dat", b"MZ\x90", "application/octet-stream")}).status_code == 415
    assert client.post("/api/uploads", files={"file": ("x.jpg", b"nope", "image/jpeg")}).status_code == 415


def test_per_session_upload_count_is_limited(client, monkeypatch):
    monkeypatch.setenv("UPLOAD_MAX_FILES_PER_SESSION", "2")
    sid = client.post("/api/sessions", json={"name": "limit"}).json()["id"]
    for i in range(2):
        r = client.post(f"/api/uploads?session_id={sid}", files={"file": (f"{i}.jpg", _jpeg_with_exif(20 + i), "image/jpeg")})
        assert r.status_code == 201, r.text
    over = client.post(f"/api/uploads?session_id={sid}", files={"file": ("3.jpg", _jpeg_with_exif(22), "image/jpeg")})
    assert over.status_code == 413 and "UPLOAD_MAX_FILES_PER_SESSION" in over.text
    # another session is not affected
    other = client.post("/api/sessions", json={"name": "other"}).json()["id"]
    assert client.post(f"/api/uploads?session_id={other}", files={"file": ("o.jpg", _jpeg_with_exif(23), "image/jpeg")}).status_code == 201


def test_reconstruction_ingest_refuses_unusable_files_per_file_and_keeps_the_good_ones(client):
    good = _jpeg_with_exif(30)
    r = client.post("/api/reconstructions", files=[
        ("files", ("ok.jpg", good, "image/jpeg")),
        ("files", ("evil.jpg", b"<html></html>", "image/jpeg")),
        ("files", ("tool.exe", b"MZ\x90\x00", "application/octet-stream")),
    ])
    assert r.status_code == 202, r.text
    body = r.json()
    assert [e["name"] for e in body["evidence"]] == ["ok.jpg"]
    assert {x["name"] for x in body["rejected"]} == {"evil.jpg", "tool.exe"}
    assert all(x["reason"] for x in body["rejected"])
    # a request with nothing usable is a 422 that says why
    none = client.post("/api/reconstructions", files=[("files", ("evil.jpg", b"<html></html>", "image/jpeg"))])
    assert none.status_code == 422 and none.json()["detail"]["rejected"][0]["name"] == "evil.jpg"


def test_artifacts_are_served_as_attachments_without_sniffing(client):
    r = client.post("/api/uploads", files={"file": ("a.jpg", _jpeg_with_exif(31), "image/jpeg")})
    art = client.get(f"/api/evidence/{r.json()['evidence_id']}/artifact")
    assert art.headers["x-content-type-options"] == "nosniff"
    assert art.headers["content-disposition"].startswith("attachment")
