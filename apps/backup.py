"""Backup, verify and restore of a Reality Engine data directory (stdlib only).

    python -m apps.backup create  <archive.tar.gz>
    python -m apps.backup verify  <archive.tar.gz>
    python -m apps.backup restore <archive.tar.gz> --into <empty-dir>

What is saved: the SQLite database (a consistent copy through SQLite's online backup API), the WorldStore tree and the uploads
tree, with a manifest of sha256 + size for every file.

Ordering is deliberate. ``commit_version`` writes the WorldStore FIRST and the DB mirror row SECOND, so the DB is copied FIRST
and the WorldStore SECOND: every version the copied DB points at already existed in the store when the DB was copied, and the
store is append-only (immutable, content-addressed), so a copy taken while the service runs can contain extra, not-yet-adopted
versions but never a dangling pointer.

Restore never overwrites: the target must be missing or empty, every file is hash-checked against the manifest BEFORE anything
is written, archive member names are validated (no absolute paths, no ``..``), and the restored database must pass
``PRAGMA integrity_check``. A non-SQLite DATABASE_URL (Postgres) is not handled here: back it up with ``pg_dump`` and pass
``--skip-db``.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sqlite3
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

FORMAT = 1
MANIFEST = "manifest.json"
DB_NAME = "db/app.db"


class BackupError(RuntimeError):
    """The archive is unusable (corrupt, tampered, unsafe) or the target is not safe to restore into."""


def sqlite_path_from_url(url: str) -> Path | None:
    """The file behind a ``sqlite[+driver]:///path`` URL, or None for any other database."""
    if not url.startswith("sqlite"):
        return None
    _, _, path = url.partition(":///")
    return Path(path) if path and path != ":memory:" else None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _tree(root: Path, prefix: str) -> list[tuple[str, Path]]:
    if not root.is_dir():
        return []
    return [(f"{prefix}/{p.relative_to(root).as_posix()}", p) for p in sorted(root.rglob("*")) if p.is_file()]


def create_backup(archive: Path, *, db_path: Path | None, worldstore_root: Path, storage_root: Path | None = None,
                  created_at: str | None = None) -> dict:
    """Write ``archive`` (a .tar.gz) and return its manifest. DB first, WorldStore second (see the module docstring)."""
    files: list[tuple[str, Path]] = []
    with tempfile.TemporaryDirectory() as tmp:
        if db_path is not None:
            if not db_path.is_file():
                raise BackupError(f"database file not found: {db_path}")
            snap = Path(tmp) / "app.db"
            src, dst = sqlite3.connect(str(db_path)), sqlite3.connect(str(snap))
            try:
                src.backup(dst)                       # consistent even while the API is writing
            finally:
                dst.close()
                src.close()
            files.append((DB_NAME, snap))
        files += _tree(worldstore_root, "worldstore")
        if storage_root is not None:
            files += _tree(storage_root, "storage")
        manifest = {
            "format": FORMAT,
            "created_at": created_at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "has_db": db_path is not None,
            "files": {name: {"sha256": _sha256(p), "size": p.stat().st_size} for name, p in files},
        }
        payload = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
        archive.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive, "w:gz") as tar:
            info = tarfile.TarInfo(MANIFEST)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
            for name, p in files:
                tar.add(str(p), arcname=name, recursive=False)
    return manifest


def _safe_name(name: str) -> bool:
    p = Path(name)
    return not (p.is_absolute() or ".." in p.parts or name.startswith(("/", "\\")) or ":" in name)


def verify_backup(archive: Path) -> dict:
    """Check the archive end to end (manifest present and supported, every member named safely, every hash and size matching,
    no member missing or extra). Returns the manifest; raises ``BackupError`` otherwise. Writes nothing."""
    try:
        with tarfile.open(archive, "r:gz") as tar:
            members = {m.name: m for m in tar.getmembers() if m.isfile()}
            if MANIFEST not in members:
                raise BackupError("archive has no manifest.json")
            manifest = json.loads(tar.extractfile(members[MANIFEST]).read().decode("utf-8"))
            if manifest.get("format") != FORMAT:
                raise BackupError(f"unsupported backup format {manifest.get('format')!r}")
            listed = manifest["files"]
            bad = [n for n in members if not _safe_name(n)]
            if bad:
                raise BackupError(f"unsafe member name(s): {bad[:3]}")
            extra, missing = set(members) - set(listed) - {MANIFEST}, set(listed) - set(members)
            if extra or missing:
                raise BackupError(f"archive does not match its manifest (extra {sorted(extra)[:3]}, missing {sorted(missing)[:3]})")
            for name, meta in listed.items():
                h, size = hashlib.sha256(), 0
                f = tar.extractfile(members[name])
                for chunk in iter(lambda f=f: f.read(1 << 20), b""):
                    h.update(chunk)
                    size += len(chunk)
                if h.hexdigest() != meta["sha256"] or size != meta["size"]:
                    raise BackupError(f"checksum mismatch: {name}")
    except (tarfile.TarError, OSError, EOFError, KeyError, ValueError) as exc:
        raise BackupError(f"unreadable archive: {exc}") from exc
    return manifest


def restore_backup(archive: Path, *, db_path: Path | None, worldstore_root: Path, storage_root: Path | None = None) -> dict:
    """Verify, then restore into targets that do not exist yet or are empty. Never overwrites. Returns a summary."""
    manifest = verify_backup(archive)
    for target in (worldstore_root, storage_root):
        if target is not None and target.exists() and any(target.iterdir()):
            raise BackupError(f"refusing to restore over a non-empty directory: {target}")
    if db_path is not None and db_path.exists():
        raise BackupError(f"refusing to restore over an existing database: {db_path}")
    if db_path is None and manifest["has_db"]:
        raise BackupError("the archive contains a database but no database target was given (pass --skip-db to ignore it)")

    dest = {"worldstore": worldstore_root, "storage": storage_root}
    written = 0
    with tarfile.open(archive, "r:gz") as tar:
        for name in manifest["files"]:
            if name == DB_NAME:
                if db_path is None:
                    continue
                out = db_path
            else:
                top, _, rel = name.partition("/")
                base = dest.get(top)
                if base is None:
                    continue
                out = base / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "wb") as f:
                f.write(tar.extractfile(name).read())
            written += 1
    if db_path is not None and db_path.exists():
        conn = sqlite3.connect(str(db_path))
        try:
            verdict = conn.execute("PRAGMA integrity_check").fetchone()[0]
        finally:
            conn.close()
        if verdict != "ok":
            raise BackupError(f"restored database failed integrity_check: {verdict}")
    return {"files_restored": written, "created_at": manifest["created_at"]}


def _env_targets(skip_db: bool) -> tuple[Path | None, Path, Path | None]:
    url = os.environ.get("DATABASE_URL", "sqlite+aiosqlite:///./data/app.db")
    db = None if skip_db else sqlite_path_from_url(url)
    if db is None and not skip_db:
        raise BackupError(f"DATABASE_URL is not SQLite ({url.split('@')[-1]}); back it up with pg_dump and pass --skip-db")
    storage = os.environ.get("STORAGE_ROOT")
    return db, Path(os.environ.get("WORLDSTORE_ROOT", "./data/worldstore")), Path(storage) if storage else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m apps.backup", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("create", "verify", "restore"):
        p = sub.add_parser(name)
        p.add_argument("archive", type=Path)
        if name != "verify":
            p.add_argument("--skip-db", action="store_true")
        if name == "restore":
            p.add_argument("--into", type=Path, required=True, help="empty directory to restore into (app.db, worldstore/, storage/)")
    args = ap.parse_args(argv)
    try:
        if args.cmd == "verify":
            m = verify_backup(args.archive)
            print(f"OK: {len(m['files'])} files, created {m['created_at']}")
        elif args.cmd == "create":
            db, ws, st = _env_targets(args.skip_db)
            m = create_backup(args.archive, db_path=db, worldstore_root=ws, storage_root=st)
            print(f"wrote {args.archive}: {len(m['files'])} files")
        else:
            root = args.into
            res = restore_backup(args.archive, db_path=None if args.skip_db else root / "app.db",
                                 worldstore_root=root / "worldstore", storage_root=root / "storage")
            print(f"restored {res['files_restored']} files into {root} (backup taken {res['created_at']})")
    except BackupError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
