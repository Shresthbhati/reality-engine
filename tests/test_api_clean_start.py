"""The API starts on a machine with no local state.

Found by building a clean venv and starting the API from an empty directory: the default ``./data/app.db`` needs a
``./data`` that a clean checkout does not have (it is git-ignored), so startup died with 'unable to open database file'.
"""

from __future__ import annotations

import importlib
import os

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")
pytest.importorskip("aiosqlite")
pytest.importorskip("greenlet")      # the async engine needs it: declared via sqlalchemy[asyncio]
from fastapi.testclient import TestClient


def test_a_file_backed_sqlite_url_gets_its_directory(tmp_path):
    from apps.api.db import _ensure_sqlite_dir

    target = tmp_path / "does" / "not" / "exist" / "app.db"
    _ensure_sqlite_dir(f"sqlite+aiosqlite:///{target.as_posix()}")
    assert target.parent.is_dir()
    _ensure_sqlite_dir("sqlite+aiosqlite:///:memory:")              # untouched
    _ensure_sqlite_dir("postgresql+asyncpg://u:p@localhost/db")     # untouched


def test_the_app_boots_with_default_paths_in_an_empty_directory(tmp_path, monkeypatch):
    for k in ("DATABASE_URL", "STORAGE_ROOT", "WORLDSTORE_ROOT"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.chdir(tmp_path)
    import apps.api.db as db_mod

    importlib.reload(db_mod)
    import apps.api.main as main_mod

    importlib.reload(main_mod)
    with TestClient(main_mod.app) as c:
        assert c.get("/api/health").json()["status"] == "ok"
        assert c.get("/api/worlds").json()["items"] == []
    assert (tmp_path / "data" / "app.db").is_file() and os.listdir(tmp_path / "data")
