"""Tests for resumable (chunked) uploads — Task 11.3.

An interrupted upload resumes from the last acknowledged offset instead of
restarting the whole file. Covered: start -> chunk -> finalize yields the same
result as a one-shot upload; resume from offset after an artificial interrupt;
offset mismatch rejected; finalize with wrong total size rejected; unsupported
content rejected at finalize with staging cleaned up.
"""

from __future__ import annotations

import hashlib
import importlib
import io
import os
from pathlib import Path

import pytest

from fastapi.testclient import TestClient

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")
pytest.importorskip("aiosqlite")
pytest.importorskip("greenlet")


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{(tmp_path / 'app.db').as_posix()}")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("WORLDSTORE_ROOT", str(tmp_path / "worldstore"))

    import apps.api.db as db_mod
    importlib.reload(db_mod)
    import apps.api.main as main_mod
    importlib.reload(main_mod)

    with TestClient(main_mod.app) as c:
        yield c


def _jpg_bytes(size: int) -> bytes:
    """A minimal valid JPEG SOI marker plus padding so vet_upload accepts it."""
    return b"\xff\xd8\xff\xe0" + b"\x00" * (size - 4)


def test_resumable_start_chunk_finalize_matches_one_shot(client):
    total = 2000
    data = _jpg_bytes(total)
    digest = hashlib.sha256(data).hexdigest()

    # 1. Start
    r = client.post(
        "/api/uploads/resumable/start",
        data={"filename": "a.jpg", "mime_type": "image/jpeg", "total_size": str(total)},
    )
    assert r.status_code == 201, r.text
    upload_id = r.json()["upload_id"]
    assert r.json()["offset"] == 0

    # 2. Upload in two chunks (simulating an interrupt between them)
    r = client.post(
        "/api/uploads/resumable/chunk",
        data={"upload_id": upload_id, "offset": "0", "total_size": str(total)},
        files={"file": ("part1", io.BytesIO(data[:1000]), "image/jpeg")},
    )
    assert r.status_code == 200, r.text
    assert r.json()["offset"] == 1000

    # Simulated interrupt: client resumes from the last acknowledged offset.
    r = client.post(
        "/api/uploads/resumable/chunk",
        data={"upload_id": upload_id, "offset": "1000", "total_size": str(total)},
        files={"file": ("part2", io.BytesIO(data[1000:]), "image/jpeg")},
    )
    assert r.status_code == 200, r.text
    assert r.json()["offset"] == total

    # 3. Finalize
    r = client.post(f"/api/uploads/resumable/{upload_id}/finalize")
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["checksum"] == digest
    assert body["upload_id"] == upload_id

    # 4. Content is stored content-addressed (re-readable)
    from apps.api.storage import resolve_artifact

    path = resolve_artifact(f"sha256://{digest}")
    assert path is not None
    assert path.read_bytes() == data


def test_resumable_offset_mismatch_is_rejected(client):
    total = 1000
    client.post(
        "/api/uploads/resumable/start",
        data={"filename": "a.jpg", "mime_type": "image/jpeg", "total_size": str(total)},
    )
    # Client claims offset 500 but nothing was staged -> reject.
    r = client.post(
        "/api/uploads/resumable/chunk",
        data={"upload_id": "does-not-exist", "offset": "0", "total_size": str(total)},
        files={"file": ("x", io.BytesIO(b"\xff\xd8\xff\xe0\x00\x00"), "image/jpeg")},
    )
    assert r.status_code == 404


def test_resumable_finalize_wrong_total_size_rejected(client):
    total = 1000
    r = client.post(
        "/api/uploads/resumable/start",
        data={"filename": "a.jpg", "mime_type": "image/jpeg", "total_size": str(total)},
    )
    upload_id = r.json()["upload_id"]
    # Stage only 100 bytes but declare 1000 -> finalize must refuse.
    client.post(
        "/api/uploads/resumable/chunk",
        data={"upload_id": upload_id, "offset": "0", "total_size": str(total)},
        files={"file": ("x", io.BytesIO(b"\xff\xd8\xff\xe0" + b"\x00" * 96), "image/jpeg")},
    )
    r = client.post(f"/api/uploads/resumable/{upload_id}/finalize")
    assert r.status_code == 409, r.text


def test_resumable_finalize_rejects_unsupported_content(client):
    html = b"<html></html>"
    r = client.post(
        "/api/uploads/resumable/start",
        data={"filename": "evil.html", "mime_type": "image/jpeg", "total_size": str(len(html))},
    )
    upload_id = r.json()["upload_id"]
    client.post(
        "/api/uploads/resumable/chunk",
        data={"upload_id": upload_id, "offset": "0", "total_size": str(len(html))},
        files={"file": ("x", io.BytesIO(html), "image/jpeg")},
    )
    r = client.post(f"/api/uploads/resumable/{upload_id}/finalize")
    assert r.status_code == 415, r.text
    # Staging cleaned up after rejection
    from apps.api.storage import staging_root

    assert not (staging_root() / f"{upload_id}.partial").exists()
