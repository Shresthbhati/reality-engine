"""Phase 1: audit events carry their own owner, and legacy rows are owned only on evidence.

Synthetic verification (temp SQLite databases, signed test tokens); no real tenant data.
"""
from __future__ import annotations

import importlib
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from apps.api.auth import Role, issue_scoped_token, rate_limiter

SECRET = "ownership-test-secret"
INI = Path(__file__).resolve().parents[1] / "apps" / "api" / "alembic.ini"


@pytest.fixture()
def api(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{(tmp_path / 'app.db').as_posix()}")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("WORLDSTORE_ROOT", str(tmp_path / "ws"))
    monkeypatch.setenv("REALITY_SIGNING_KEY", SECRET)
    monkeypatch.setenv("REALITY_API_TOKEN", "master-token-for-tests")
    import apps.api.db as db_mod
    importlib.reload(db_mod)
    import apps.api.main as main_mod
    importlib.reload(main_mod)
    rate_limiter.reset()
    headers = {
        "a": {"Authorization": f"Bearer {issue_scoped_token('tenant-a', 'user-a', Role.EDITOR, secret=SECRET)}"},
        "b": {"Authorization": f"Bearer {issue_scoped_token('tenant-b', 'user-b', Role.EDITOR, secret=SECRET)}"},
        "tenant-admin-b": {"Authorization": f"Bearer {issue_scoped_token('tenant-b', 'admin-b', Role.ADMIN, secret=SECRET)}"},
        "root": {"Authorization": f"Bearer {issue_scoped_token('root', 'root-admin', Role.ADMIN, secret=SECRET)}"},
    }
    with TestClient(main_mod.app) as client:
        yield client, headers
    rate_limiter.reset()


def _events(client, headers) -> list[dict]:
    res = client.get("/api/activity", headers=headers)
    assert res.status_code == 200, res.text
    return res.json()["items"]


def _create_world(client, headers, name="Owned world") -> str:
    res = client.post("/api/worlds", json={"name": name}, headers=headers)
    assert res.status_code == 201, res.text
    return res.json()["id"]


def test_events_are_stamped_with_the_subjects_owner(api):
    client, h = api
    wid = _create_world(client, h["a"])
    assert any(e["entity_id"] == wid for e in _events(client, h["a"]))
    assert not any(e["entity_id"] == wid for e in _events(client, h["b"]))
    assert not any(e["entity_id"] == wid for e in _events(client, h["tenant-admin-b"]))  # tenant admin is not global


def test_events_about_a_purged_world_stay_with_its_owner_only(api):
    client, h = api
    wid = _create_world(client, h["a"])
    res = client.delete(f"/api/worlds/{wid}?permanent=true", headers=h["a"])
    assert res.status_code in (200, 204), res.text

    mine = [e for e in _events(client, h["a"]) if e["entity_id"] == wid]
    assert any(e["type"] == "world.purged" for e in mine), "the owner must still see the deletion audit event"
    assert not any(e["entity_id"] == wid for e in _events(client, h["b"]))
    assert not any(e["entity_id"] == wid for e in _events(client, h["tenant-admin-b"]))
    assert any(e["entity_id"] == wid for e in _events(client, h["root"]))  # only a global administrator crosses tenants


def _insert(con: sqlite3.Connection, table: str, **values) -> None:
    """Insert a row, filling every other NOT NULL column with a neutral value (the 0005 schema is what is tested)."""
    row = dict(values)
    for _cid, name, ctype, notnull, default, _pk in con.execute(f"PRAGMA table_info({table})").fetchall():
        if name in row or not notnull or default is not None:
            continue
        row[name] = 0 if any(t in ctype.upper() for t in ("INT", "FLOAT", "REAL", "BOOL")) else (
            "2026-01-01 00:00:00" if "DATE" in ctype.upper() else "x")
    cols = ", ".join(row)
    con.execute(f"INSERT INTO {table} ({cols}) VALUES ({', '.join('?' * len(row))})", list(row.values()))


def test_migration_assigns_owners_only_where_the_subject_proves_them(tmp_path):
    pytest.importorskip("alembic")
    from alembic import command
    from alembic.config import Config

    db = tmp_path / "legacy.db"
    cfg = Config(str(INI))
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db.as_posix()}")
    command.upgrade(cfg, "0005")

    con = sqlite3.connect(db)
    _insert(con, "worlds", id="w-owned", name="Owned", team_id="tenant-a")
    _insert(con, "worlds", id="w-legacy", name="Legacy", team_id=None)
    for eid, etype, entity in (
        ("ev-proven", "world", "w-owned"),      # subject exists and is owned -> owner proven
        ("ev-unowned", "world", "w-legacy"),    # subject exists but has no owner -> nothing to prove
        ("ev-gone", "world", "w-deleted"),      # subject no longer exists -> nothing to prove
        ("ev-bogus", "bogus", "w-owned"),       # unknown subject type -> never guessed
        ("ev-global", None, None),              # no subject at all
    ):
        _insert(con, "activity_events", id=eid, type="t", summary="s", entity_type=etype, entity_id=entity)
    con.commit()
    con.close()

    command.upgrade(cfg, "0006")
    con = sqlite3.connect(db)
    owners = dict(con.execute("SELECT id, team_id FROM activity_events").fetchall())
    con.close()
    assert owners == {"ev-proven": "tenant-a", "ev-unowned": None, "ev-gone": None, "ev-bogus": None, "ev-global": None}

    command.upgrade(cfg, "0006")  # idempotent
    command.downgrade(cfg, "0005")  # reversible
    con = sqlite3.connect(db)
    assert "team_id" not in {r[1] for r in con.execute("PRAGMA table_info(activity_events)").fetchall()}
    assert con.execute("SELECT count(*) FROM activity_events").fetchone()[0] == 5  # no event lost
    con.close()
