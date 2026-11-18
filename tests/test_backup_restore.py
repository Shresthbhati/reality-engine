"""RE-DB-000005: a backup can actually be restored, tampering is caught, and restore never overwrites."""
import hashlib
import io
import json
import sqlite3
import tarfile
from pathlib import Path

import pytest

from apps import backup as bk


def _data(root: Path) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    db = root / "app.db"
    conn = sqlite3.connect(db)
    conn.execute("create table worlds (id text primary key, head text)")
    conn.execute("insert into worlds values ('w1', 'v-abc')")
    conn.commit()
    conn.close()
    ws, st = root / "worldstore", root / "storage"
    (ws / "ab").mkdir(parents=True)
    (ws / "ab" / "blob1").write_bytes(b"world-json")
    (ws / "versions.json").write_text('{"v-abc": "ab/blob1"}')
    (st / "uploads").mkdir(parents=True)
    (st / "uploads" / "photo.jpg").write_bytes(b"\xff\xd8photo")
    return {"db_path": db, "worldstore_root": ws, "storage_root": st}


def _hashes(root: Path) -> dict:
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob("*")) if p.is_file()}


def test_restore_drill_reproduces_every_file_and_the_database(tmp_path):
    src = _data(tmp_path / "src")
    archive = tmp_path / "b" / "snap.tar.gz"
    manifest = bk.create_backup(archive, **src, created_at="2026-11-18T00:00:00+00:00")
    assert manifest["has_db"] and set(manifest["files"]) == {
        "db/app.db", "worldstore/ab/blob1", "worldstore/versions.json", "storage/uploads/photo.jpg"}
    assert bk.verify_backup(archive)["created_at"] == "2026-11-18T00:00:00+00:00"

    out = tmp_path / "restored"
    res = bk.restore_backup(archive, db_path=out / "app.db", worldstore_root=out / "worldstore", storage_root=out / "storage")

    assert res["files_restored"] == 4
    assert _hashes(out / "worldstore") == _hashes(src["worldstore_root"]) and _hashes(out / "storage") == _hashes(src["storage_root"])
    assert sqlite3.connect(out / "app.db").execute("select head from worlds where id='w1'").fetchone() == ("v-abc",)


def test_backup_is_consistent_while_the_source_database_is_open_and_changing(tmp_path):
    src = _data(tmp_path / "s")
    live = sqlite3.connect(src["db_path"])
    live.execute("insert into worlds values ('w2', 'v-def')")          # an uncommitted write must not leak into the snapshot
    bk.create_backup(tmp_path / "snap.tar.gz", **src)
    live.rollback()
    live.close()
    out = tmp_path / "r"
    bk.restore_backup(tmp_path / "snap.tar.gz", db_path=out / "app.db", worldstore_root=out / "ws", skip=("storage",))
    assert sqlite3.connect(out / "app.db").execute("select count(*) from worlds").fetchone() == (1,)


def _rewrite(archive: Path, mutate) -> Path:
    members = {}
    with tarfile.open(archive, "r:gz") as tar:
        for m in tar.getmembers():
            members[m.name] = tar.extractfile(m).read()
    members = mutate(members)
    out = archive.with_name("tampered.tar.gz")
    with tarfile.open(out, "w:gz") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return out


def test_tampering_and_unsafe_archives_are_rejected_before_anything_is_written(tmp_path):
    archive = tmp_path / "snap.tar.gz"
    bk.create_backup(archive, **_data(tmp_path / "s"))

    flipped = _rewrite(archive, lambda m: {**m, "worldstore/ab/blob1": b"world-JSON"})
    with pytest.raises(bk.BackupError, match="checksum mismatch"):
        bk.verify_backup(flipped)

    extra = _rewrite(archive, lambda m: {**m, "worldstore/injected": b"x"})
    with pytest.raises(bk.BackupError, match="does not match its manifest"):
        bk.verify_backup(extra)

    def traversal(m):
        man = json.loads(m[bk.MANIFEST])
        man["files"]["../evil"] = {"sha256": hashlib.sha256(b"x").hexdigest(), "size": 1}
        return {**m, bk.MANIFEST: json.dumps(man).encode(), "../evil": b"x"}
    with pytest.raises(bk.BackupError, match="unsafe member"):
        bk.verify_backup(_rewrite(archive, traversal))

    out = tmp_path / "never"
    with pytest.raises(bk.BackupError):
        bk.restore_backup(flipped, db_path=out / "app.db", worldstore_root=out / "ws")
    assert not out.exists()                                            # nothing was written for a bad archive

    (tmp_path / "junk.tar.gz").write_bytes(b"not an archive")
    with pytest.raises(bk.BackupError, match="unreadable"):
        bk.verify_backup(tmp_path / "junk.tar.gz")


def test_restore_never_overwrites_existing_data(tmp_path):
    archive = tmp_path / "snap.tar.gz"
    bk.create_backup(archive, **_data(tmp_path / "s"))

    busy = tmp_path / "busy"
    (busy / "ws").mkdir(parents=True)
    (busy / "ws" / "keep.txt").write_text("precious")
    with pytest.raises(bk.BackupError, match="non-empty"):
        bk.restore_backup(archive, db_path=busy / "app.db", worldstore_root=busy / "ws", skip=("storage",))
    assert (busy / "ws" / "keep.txt").read_text() == "precious" and not (busy / "app.db").exists()

    (tmp_path / "live.db").write_bytes(b"x")
    with pytest.raises(bk.BackupError, match="existing database"):
        bk.restore_backup(archive, db_path=tmp_path / "live.db", worldstore_root=tmp_path / "fresh", skip=("storage",))


def test_cli_roundtrip_and_non_sqlite_is_refused_with_the_pg_dump_hint(tmp_path, monkeypatch, capsys):
    src = _data(tmp_path / "s")
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{src['db_path']}")
    monkeypatch.setenv("WORLDSTORE_ROOT", str(src["worldstore_root"]))
    monkeypatch.setenv("STORAGE_ROOT", str(src["storage_root"]))
    archive = str(tmp_path / "cli.tar.gz")

    assert bk.main(["create", archive]) == 0
    assert bk.main(["verify", archive]) == 0
    assert bk.main(["restore", archive, "--into", str(tmp_path / "r")]) == 0
    assert (tmp_path / "r" / "app.db").is_file()

    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://u:p@db/x")
    assert bk.main(["create", archive + "2"]) == 2
    assert "pg_dump" in capsys.readouterr().err
