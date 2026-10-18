"""C2: room templates are data-only default dimensions served over the API."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from procedural.templates import ROOM_TEMPLATES

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")
pytest.importorskip("aiosqlite")

from fastapi.testclient import TestClient


def test_template_data_matches_spec():
    assert ROOM_TEMPLATES == {
        "studio": {"width": 4.0, "depth": 3.0, "height": 2.4},
        "bedroom": {"width": 3.5, "depth": 3.2, "height": 2.5},
        "bathroom": {"width": 2.4, "depth": 1.8, "height": 2.4},
        "kitchen": {"width": 3.2, "depth": 2.8, "height": 2.5},
        "living": {"width": 5.5, "depth": 4.0, "height": 2.6},
        "office": {"width": 3.0, "depth": 3.0, "height": 2.5},
    }


def test_templates_route(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{(tmp_path / 'a.db').as_posix()}")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "artifacts"))
    import apps.api.db as db_mod
    importlib.reload(db_mod)
    import apps.api.main as main_mod
    importlib.reload(main_mod)
    with TestClient(main_mod.app) as c:
        r = c.get("/api/procedural/templates")
    assert r.status_code == 200
    body = r.json()
    assert "default" in body["label"].lower()
    got = {t["name"]: {k: t[k] for k in ("width", "depth", "height")} for t in body["templates"]}
    assert got == ROOM_TEMPLATES
