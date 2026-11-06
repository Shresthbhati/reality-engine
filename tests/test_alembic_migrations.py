"""Task 1.1: `alembic upgrade head` builds the schema on an empty DB and upgrades an old create_all DB losslessly."""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

import pytest

pytest.importorskip("alembic")
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect

from apps.api import routes_assistant  # noqa: F401  AssistantPlan lives there
from apps.api.models import Base

INI = Path(__file__).resolve().parents[1] / "apps" / "api" / "alembic.ini"
NEW_COLUMNS = {
    "worlds": {"archived_at", "deleted_at", "thumbnail_uri", "scale_state"},
    "world_versions": {"label", "created_by_job_id"},
    "jobs": {"stage_counts"},
    "uploads": {"team_id"},
    "evidence": {"team_id"},
    "sites": {"team_id"},
}


def _cfg(db: Path) -> Config:
    cfg = Config(str(INI))
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db.as_posix()}")
    return cfg


def _schema(db: Path) -> dict[str, set[str]]:
    eng = create_engine(f"sqlite:///{db.as_posix()}")
    insp = inspect(eng)
    out = {t: {c["name"] for c in insp.get_columns(t)} for t in insp.get_table_names()}
    eng.dispose()
    return out


def _old_style_db(db: Path) -> None:
    """What the pre-Alembic code left on disk: create_all, minus every column/table this task adds."""
    eng = create_engine(f"sqlite:///{db.as_posix()}")
    Base.metadata.create_all(eng)
    eng.dispose()
    con = sqlite3.connect(db)
    con.execute("DROP TABLE capture_requests")
    for table, cols in NEW_COLUMNS.items():
        for col in cols:
            con.execute(f"ALTER TABLE {table} DROP COLUMN {col}")
    con.execute("INSERT INTO worlds (id, name, description, status, owner_id, current_version_id, created_at, updated_at)"
                " VALUES ('wld_old1', 'Old house', 'keep me', 'active', 'local', 'v-abc', '2026-01-01', '2026-01-02')")
    con.execute("INSERT INTO world_versions (id, world_id, source_session_ids, changed_entity_ids, changed_geometry_ids,"
                " artifact_uri, created_at) VALUES ('v-abc', 'wld_old1', '[]', '[]', '[]', 'artifact://x', '2026-01-01')")
    con.execute("INSERT INTO jobs (id, type, entity_type, entity_id, status, stage, attempts, max_attempts, priority,"
                " cancel_requested, created_at) VALUES ('job_old1', 'reconstruct_session', 'world', 'wld_old1',"
                " 'succeeded', 'done', 1, 3, 0, 0, '2026-01-01')")
    con.commit()
    con.close()


def test_upgrade_head_on_empty_db_matches_models(tmp_path):
    db = tmp_path / "empty.db"
    command.upgrade(_cfg(db), "head")
    schema = _schema(db)
    expected = {t.name: {c.name for c in t.columns} for t in Base.metadata.sorted_tables}
    assert {k: v for k, v in schema.items() if k != "alembic_version"} == expected
    assert "capture_requests" in schema


def test_old_create_all_db_upgrades_without_data_loss(tmp_path):
    db = tmp_path / "old.db"
    _old_style_db(db)
    before = _schema(db)
    assert not (NEW_COLUMNS["worlds"] & before["worlds"]) and "capture_requests" not in before
    command.upgrade(_cfg(db), "head")
    after = _schema(db)
    for table, cols in NEW_COLUMNS.items():
        assert cols <= after[table]
    assert "capture_requests" in after
    con = sqlite3.connect(db)
    assert con.execute("SELECT name, description, status, current_version_id, archived_at FROM worlds").fetchall() == [
        ("Old house", "keep me", "active", "v-abc", None)]
    assert con.execute("SELECT id, world_id, label FROM world_versions").fetchall() == [("v-abc", "wld_old1", None)]
    assert con.execute("SELECT id, status, stage, stage_counts FROM jobs").fetchall() == [
        ("job_old1", "succeeded", "done", None)]
    con.close()


def test_upgrade_is_idempotent_and_init_db_path_still_works(tmp_path, monkeypatch):
    """init_db (create_all + add missing columns) on an old DB reaches the same schema; alembic then accepts it."""
    db = tmp_path / "init.db"
    _old_style_db(db)
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db.as_posix()}")
    from apps.api import db as apidb

    async def go():
        await apidb.dispose_db()
        await apidb.init_db()
        await apidb.init_db()          # idempotent
        await apidb.dispose_db()

    asyncio.run(go())
    for table, cols in NEW_COLUMNS.items():
        assert cols <= _schema(db)[table]
    command.upgrade(_cfg(db), "head")  # unstamped create_all-style DB: migrations skip what already exists
    command.upgrade(_cfg(db), "head")
    con = sqlite3.connect(db)
    assert con.execute("SELECT name FROM worlds").fetchall() == [("Old house",)]
    assert con.execute("SELECT version_num FROM alembic_version").fetchall() == [("0003",)]
    con.close()
