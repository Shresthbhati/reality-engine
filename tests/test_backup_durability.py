"""Phase 2: backups never skip a component silently, restores are all-or-nothing, and the restored data is checked.

Synthetic verification: temp files, SQLite, a real WorldStore in a temp directory, and fake ``pg_dump``/``pg_restore``
executables (no Postgres server is available here, so the Postgres path is verified at the process boundary only).
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import sqlite3
import sys
import tarfile
from pathlib import Path

import pytest

from apps import backup as bk
from world_ir.world_v1 import WorldIR
from worldstore.store import WorldStore


def _artifact(storage: Path, payload: bytes) -> str:
    digest = hashlib.sha256(payload).hexdigest()
    path = storage / digest[:2] / digest[2:4] / digest
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return f"sha256://{digest}"


def _deployment(root: Path, *, revision: str | None = "0006") -> dict:
    """A small but real deployment: a WorldStore with one version, a DB row pointing at it, an evidence artifact."""
    ws, st = root / "worldstore", root / "storage"
    stored = WorldStore(ws).save_version(WorldIR(), parent=None)
    uri = _artifact(st, b"\xff\xd8evidence-bytes")
    db = root / "app.db"
    conn = sqlite3.connect(db)
    conn.execute("create table world_versions (id text primary key, artifact_uri text, artifact_hash text,"
                 " points_artifact_uri text, cameras_artifact_uri text)")
    conn.execute("create table evidence (id text primary key, artifact_uri text)")
    conn.execute("create table uploads (id text primary key, artifact_uri text)")
    conn.execute("insert into world_versions values (?, ?, ?, null, null)", (stored.version_id, stored.artifact_uri, stored.artifact_hash))
    conn.execute("insert into evidence values ('e1', ?)", (uri,))
    if revision:
        conn.execute("create table alembic_version (version_num text)")
        conn.execute("insert into alembic_version values (?)", (revision,))
    conn.commit()
    conn.close()
    return {"db_path": db, "worldstore_root": ws, "storage_root": st, "_version": stored.version_id, "_uri": uri}


def _src(d: dict) -> dict:
    return {k: v for k, v in d.items() if not k.startswith("_")}


def _targets(root: Path) -> dict:
    return {"db_path": root / "app.db", "worldstore_root": root / "worldstore", "storage_root": root / "storage"}


def _leftovers(root: Path) -> list[str]:
    return sorted(p.name for p in root.glob("*.restoring")) if root.exists() else []


def test_a_skipped_component_is_recorded_and_never_dropped_silently(tmp_path):
    d = _deployment(tmp_path / "s")
    skipped = bk.create_backup(tmp_path / "no-storage.tar.gz", **{**_src(d), "storage_root": None})
    assert skipped["components"]["storage"]["status"] == "skipped"
    assert not any(n.startswith("storage/") for n in skipped["files"])

    full = tmp_path / "full.tar.gz"
    assert bk.create_backup(full, **_src(d))["components"]["storage"]["status"] == "included"
    out = tmp_path / "r"
    with pytest.raises(bk.BackupError, match="contains the storage component"):
        bk.restore_backup(full, db_path=out / "app.db", worldstore_root=out / "worldstore")   # no target, no explicit skip
    assert not out.exists()

    bk.restore_backup(full, db_path=out / "app.db", worldstore_root=out / "worldstore", skip=("storage",))
    assert (out / "app.db").is_file() and not (out / "storage").exists()


def test_a_configured_root_that_does_not_exist_is_an_error_not_an_empty_component(tmp_path):
    d = _deployment(tmp_path / "s")
    with pytest.raises(bk.BackupError, match="WorldStore root not found"):
        bk.create_backup(tmp_path / "a.tar.gz", **{**_src(d), "worldstore_root": tmp_path / "typo"})
    with pytest.raises(bk.BackupError, match="storage root not found"):
        bk.create_backup(tmp_path / "a.tar.gz", **{**_src(d), "storage_root": tmp_path / "typo"})
    assert not (tmp_path / "a.tar.gz").exists()


def test_the_cli_backs_up_the_default_artifact_store_when_storage_root_is_unset(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    d = _deployment(tmp_path / "data")
    os.replace(d["storage_root"], tmp_path / "data" / "artifacts")          # the app's default location (./data/artifacts)
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{d['db_path'].as_posix()}")
    monkeypatch.setenv("WORLDSTORE_ROOT", str(d["worldstore_root"]))
    monkeypatch.delenv("STORAGE_ROOT", raising=False)

    assert bk.main(["create", str(tmp_path / "cli.tar.gz")]) == 0
    manifest = bk.verify_backup(tmp_path / "cli.tar.gz")
    assert manifest["components"]["storage"]["status"] == "included"
    assert any(n.startswith("storage/") for n in manifest["files"]), "the default artifact store was silently left out"

    assert bk.main(["create", str(tmp_path / "skip.tar.gz"), "--skip-storage"]) == 0
    assert "skipped: storage" in capsys.readouterr().out


def test_restore_is_all_or_nothing_when_a_rename_fails_midway(tmp_path, monkeypatch):
    d = _deployment(tmp_path / "s")
    archive = tmp_path / "a.tar.gz"
    bk.create_backup(archive, **_src(d))
    out = tmp_path / "r"
    real, calls = os.replace, {"n": 0}

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] == 3:                                               # storage and worldstore moved; the database fails
            raise OSError("simulated disk failure")
        return real(src, dst)

    monkeypatch.setattr(bk.os, "replace", flaky)
    with pytest.raises(bk.BackupError, match="could not move"):
        bk.restore_backup(archive, **_targets(out))
    monkeypatch.undo()

    assert not (out / "app.db").exists() and not (out / "worldstore").exists() and not (out / "storage").exists()
    assert _leftovers(out) == []
    assert bk.restore_backup(archive, **_targets(out))["files_restored"] > 0   # and a retry then succeeds


def test_a_crash_before_the_commit_leaves_the_real_targets_untouched_and_the_next_restore_cleans_up(tmp_path, monkeypatch):
    d = _deployment(tmp_path / "s")
    archive = tmp_path / "a.tar.gz"
    bk.create_backup(archive, **_src(d))
    out = tmp_path / "r"

    def crash(**_kw):
        raise RuntimeError("power cut")

    monkeypatch.setattr(bk, "check_consistency", crash)
    with pytest.raises(RuntimeError):
        bk.restore_backup(archive, **_targets(out))
    monkeypatch.undo()
    assert not (out / "app.db").exists() and not (out / "worldstore").exists()
    assert _leftovers(out) == []

    (out / "worldstore.restoring").mkdir(parents=True)                      # what a hard kill would have left behind
    (out / "worldstore.restoring" / "half-written").write_bytes(b"x")
    bk.restore_backup(archive, **_targets(out))
    assert (out / "app.db").is_file() and not (out / "worldstore" / "half-written").exists()
    assert _leftovers(out) == []


def test_an_inconsistent_backup_is_refused_before_anything_is_placed(tmp_path):
    d = _deployment(tmp_path / "s")
    conn = sqlite3.connect(d["db_path"])
    conn.execute("insert into world_versions values ('v-ghost', 'sha256://x', 'x', null, null)")   # points at no stored version
    conn.commit()
    conn.close()
    archive = tmp_path / "bad.tar.gz"
    bk.create_backup(archive, **_src(d))                                    # the archive itself is intact; its CONTENT is not
    out = tmp_path / "r"
    with pytest.raises(bk.BackupError, match="inconsistent.*dangling pointer"):
        bk.restore_backup(archive, **_targets(out))
    assert not out.exists() or not any(out.iterdir())


def test_consistency_check_detects_corrupt_versions_missing_and_tampered_artifacts(tmp_path):
    d = _deployment(tmp_path / "s")
    ok = bk.check_consistency(db_path=d["db_path"], worldstore_root=d["worldstore_root"], storage_root=d["storage_root"], deep=True)
    assert ok["problems"] == [] and ok["checked"] == {"world_versions": 1, "artifacts": 1}

    art = bk._artifact_path(d["storage_root"], d["_uri"])
    art.write_bytes(b"tampered")
    deep = bk.check_consistency(db_path=d["db_path"], worldstore_root=d["worldstore_root"], storage_root=d["storage_root"], deep=True)
    assert any("content hash" in p for p in deep["problems"])
    assert bk.check_consistency(db_path=d["db_path"], worldstore_root=d["worldstore_root"], storage_root=d["storage_root"])["problems"] == []
    art.unlink()
    assert any("missing from the artifact store" in p for p in bk.check_consistency(
        db_path=d["db_path"], worldstore_root=d["worldstore_root"], storage_root=d["storage_root"])["problems"])

    store_artifacts = [p for p in (d["worldstore_root"] / "artifacts").rglob("*") if p.is_file()]
    assert store_artifacts
    store_artifacts[0].write_bytes(b"corrupt")
    assert any("hash mismatch" in p for p in bk.check_consistency(
        db_path=d["db_path"], worldstore_root=d["worldstore_root"], storage_root=d["storage_root"])["problems"])


def test_what_cannot_be_checked_is_reported_not_assumed_fine(tmp_path):
    rep = bk.check_consistency(db_path=None, worldstore_root=tmp_path, storage_root=None)
    assert rep["problems"] == [] and rep["not_checked"], "an unchecked database must be listed as not checked"
    toy = tmp_path / "toy.db"
    sqlite3.connect(toy).close()
    assert any("absent" in n for n in bk.check_consistency(db_path=toy, worldstore_root=tmp_path, storage_root=tmp_path)["not_checked"])


def test_schema_revision_is_recorded_and_returned_on_restore(tmp_path):
    d = _deployment(tmp_path / "s", revision="0006")
    archive = tmp_path / "a.tar.gz"
    assert bk.create_backup(archive, **_src(d))["components"]["db"]["schema_revision"] == "0006"
    res = bk.restore_backup(archive, **_targets(tmp_path / "r"))
    assert res["schema_revision"] == "0006" and res["format"] == bk.FORMAT


def _write_archive(path: Path, manifest: dict, files: dict[str, bytes]) -> None:
    with tarfile.open(path, "w:gz") as tar:
        for name, data in {bk.MANIFEST: json.dumps(manifest).encode(), **files}.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


def test_a_format_1_archive_still_restores_and_an_unknown_format_is_refused(tmp_path):
    src = sqlite3.connect(tmp_path / "old.db")
    src.execute("create table worlds (id text)")
    src.commit()
    src.close()
    db_bytes = (tmp_path / "old.db").read_bytes()
    sha = lambda b: {"sha256": hashlib.sha256(b).hexdigest(), "size": len(b)}  # noqa: E731
    files = {"db/app.db": db_bytes, "worldstore/versions/x.json": b"{}"}
    v1 = {"format": 1, "created_at": "2026-01-01T00:00:00+00:00", "has_db": True, "files": {n: sha(b) for n, b in files.items()}}
    _write_archive(tmp_path / "v1.tar.gz", v1, files)
    res = bk.restore_backup(tmp_path / "v1.tar.gz", db_path=tmp_path / "r" / "app.db",
                            worldstore_root=tmp_path / "r" / "worldstore", skip=("storage",), check=False)
    assert res["format"] == 1 and (tmp_path / "r" / "app.db").is_file()

    _write_archive(tmp_path / "v9.tar.gz", {**v1, "format": 9}, files)
    with pytest.raises(bk.BackupError, match="unsupported backup format 9"):
        bk.verify_backup(tmp_path / "v9.tar.gz")


def test_an_interrupted_backup_never_leaves_a_truncated_archive(tmp_path, monkeypatch):
    d = _deployment(tmp_path / "s")
    monkeypatch.setattr(bk.os, "replace", lambda *_a: (_ for _ in ()).throw(OSError("killed")))
    with pytest.raises(OSError):
        bk.create_backup(tmp_path / "a.tar.gz", **_src(d))
    assert not (tmp_path / "a.tar.gz").exists()


# ---- Postgres, verified at the process boundary with fake client tools --------------------------------------------------

FAKE_PG = r'''
import json, os, sys
tool, args = sys.argv[1], sys.argv[2:]
with open(os.environ["FAKE_PG_LOG"], "a") as log:
    log.write(json.dumps({"tool": tool, "args": args, "pgpassword": os.environ.get("PGPASSWORD")}) + "\n")
if os.environ.get("FAKE_PG_FAIL"):
    sys.stderr.write("connection refused"); sys.exit(1)
if tool == "pg_dump":
    open(args[args.index("--file") + 1], "wb").write(b"FAKE-PGDUMP")
else:
    sys.exit(0 if open(args[-1], "rb").read() == b"FAKE-PGDUMP" else 3)
'''


@pytest.fixture()
def fake_pg(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    script = bin_dir / "fake_pg.py"
    script.write_text(FAKE_PG)
    for tool in ("pg_dump", "pg_restore"):
        if sys.platform == "win32":
            (bin_dir / f"{tool}.bat").write_text(f'@echo off\r\n"{sys.executable}" "{script}" {tool} %*\r\n')
        else:
            exe = bin_dir / tool
            exe.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" {tool} "$@"\n')
            exe.chmod(0o755)
    log = tmp_path / "pg.log"
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_PG_LOG", str(log))
    return lambda: [json.loads(line) for line in log.read_text().splitlines()]


def test_postgres_backup_and_restore_use_pg_tools_and_keep_the_password_off_the_command_line(tmp_path, fake_pg):
    d = _deployment(tmp_path / "s")
    url = "postgresql+asyncpg://reality:s3cr3t%40pw@db.internal:5433/reality_prod"
    archive = tmp_path / "pg.tar.gz"
    manifest = bk.create_backup(archive, db_path=None, database_url=url, worldstore_root=d["worldstore_root"], storage_root=d["storage_root"])
    assert manifest["components"]["db"]["kind"] == "postgres" and bk.PG_NAME in manifest["files"]

    out = tmp_path / "r"
    res = bk.restore_backup(archive, worldstore_root=out / "worldstore", storage_root=out / "storage",
                            database_url="postgresql://reality:s3cr3t%40pw@db.internal/restored")
    assert any("Postgres" in n for n in res["consistency"]["not_checked"])

    calls = fake_pg()
    assert [c["tool"] for c in calls] == ["pg_dump", "pg_restore"]
    for call in calls:
        assert all("s3cr3t" not in a for a in call["args"]), "the password must never appear on a command line"
        assert call["pgpassword"] == "s3cr3t@pw"
        assert "--no-owner" in call["args"]
    assert calls[0]["args"][calls[0]["args"].index("--host") + 1] == "db.internal"
    assert calls[0]["args"][calls[0]["args"].index("--port") + 1] == "5433"
    assert (out / "worldstore").is_dir()


def test_postgres_failures_and_missing_targets_are_loud(tmp_path, fake_pg, monkeypatch):
    d = _deployment(tmp_path / "s")
    url = "postgresql://u:p@h/db"
    monkeypatch.setenv("FAKE_PG_FAIL", "1")
    with pytest.raises(bk.BackupError, match="pg_dump failed.*connection refused"):
        bk.create_backup(tmp_path / "a.tar.gz", db_path=None, database_url=url, worldstore_root=d["worldstore_root"])
    assert not (tmp_path / "a.tar.gz").exists()
    monkeypatch.delenv("FAKE_PG_FAIL")

    archive = tmp_path / "pg.tar.gz"
    bk.create_backup(archive, db_path=None, database_url=url, worldstore_root=d["worldstore_root"])
    out = tmp_path / "r"
    with pytest.raises(bk.BackupError, match="no --database-url"):
        bk.restore_backup(archive, worldstore_root=out / "worldstore", skip=("storage",))
    monkeypatch.setenv("FAKE_PG_FAIL", "1")
    with pytest.raises(bk.BackupError, match="pg_restore failed"):
        bk.restore_backup(archive, worldstore_root=out / "worldstore", skip=("storage",), database_url=url)
    assert not out.exists(), "a failed database restore must leave the filesystem untouched"


def test_missing_pg_tools_are_reported(tmp_path, monkeypatch):
    d = _deployment(tmp_path / "s")
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    with pytest.raises(bk.BackupError, match="pg_dump not found on PATH"):
        bk.create_backup(tmp_path / "a.tar.gz", db_path=None, database_url="postgresql://u:p@h/db", worldstore_root=d["worldstore_root"])
