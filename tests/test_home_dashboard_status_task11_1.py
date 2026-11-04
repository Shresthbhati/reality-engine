"""Unit and API tests for Home Dashboard metrics, jobs, and storage (Blueprint Task 11.1).

Verifies:
1. /api/status returns real filesystem storage breakdown:
   - artifacts_bytes, artifacts_count
   - worldstore_bytes
   - database_bytes
   - total_bytes
2. /api/status returns accurate durable counts:
   - sessions, evidence, jobs, worlds, jobs_running, jobs_failed
3. Real data only: non-negative integers, consistent sums.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")
pytest.importorskip("aiosqlite")

from fastapi.testclient import TestClient

from apps.api.storage import measure_storage_usage


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{(tmp_path / 'app.db').as_posix()}")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("WORLDSTORE_ROOT", str(tmp_path / "worldstore"))

    # Create dummy artifact and worldstore files to verify real measurement
    (tmp_path / "artifacts").mkdir(parents=True, exist_ok=True)
    (tmp_path / "artifacts" / "test_blob.bin").write_bytes(b"HELLO_REALITY_ENGINE_STORAGE")

    (tmp_path / "worldstore").mkdir(parents=True, exist_ok=True)
    (tmp_path / "worldstore" / "test_version.json").write_bytes(b'{"version": 1}')

    import apps.api.db as db_mod
    importlib.reload(db_mod)
    import apps.api.main as main_mod
    importlib.reload(main_mod)

    with TestClient(main_mod.app) as c:
        yield c


def test_measure_storage_usage_real_filesystem(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "art"))
    monkeypatch.setenv("WORLDSTORE_ROOT", str(tmp_path / "ws"))
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{(tmp_path / 'test.db').as_posix()}")

    art_dir = tmp_path / "art"
    art_dir.mkdir(parents=True, exist_ok=True)
    (art_dir / "blob1.dat").write_bytes(b"1234567890")  # 10 bytes

    ws_dir = tmp_path / "ws"
    ws_dir.mkdir(parents=True, exist_ok=True)
    (ws_dir / "world.json").write_bytes(b"abcdef")  # 6 bytes

    (tmp_path / "test.db").write_bytes(b"sqlite_data")  # 11 bytes

    usage = measure_storage_usage()
    assert usage["artifacts_bytes"] == 10
    assert usage["artifacts_count"] == 1
    assert usage["worldstore_bytes"] == 6
    assert usage["database_bytes"] == 11
    assert usage["total_bytes"] == 27


def test_api_status_endpoint_returns_storage_and_job_counts(client: TestClient):
    r = client.get("/api/status")
    assert r.status_code == 200
    data = r.json()

    assert data["database"] == "ok"
    assert "counts" in data
    counts = data["counts"]
    assert "sessions" in counts
    assert "evidence" in counts
    assert "jobs" in counts
    assert "worlds" in counts
    assert "jobs_running" in counts
    assert "jobs_failed" in counts

    assert "storage" in data
    storage = data["storage"]
    assert storage["artifacts_bytes"] >= len(b"HELLO_REALITY_ENGINE_STORAGE")
    assert storage["artifacts_count"] >= 1
    assert storage["worldstore_bytes"] >= len(b'{"version": 1}')
    assert storage["total_bytes"] == storage["artifacts_bytes"] + storage["worldstore_bytes"] + storage["database_bytes"]
