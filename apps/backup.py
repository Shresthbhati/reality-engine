"""Backup, verify, check and restore of a Reality Engine deployment (stdlib only; Postgres needs pg_dump/pg_restore).

    python -m apps.backup create  <archive.tar.gz> [--skip-db] [--skip-storage]
    python -m apps.backup verify  <archive.tar.gz>
    python -m apps.backup restore <archive.tar.gz> --into <dir> [--skip-db] [--skip-storage] [--database-url URL]
    python -m apps.backup check                      (read-only: DB <-> WorldStore <-> artifact consistency of the live data)

What is saved: the database (SQLite: a consistent copy through SQLite's online backup API; Postgres: ``pg_dump`` custom
format), the WorldStore tree and the content-addressed artifact tree (``STORAGE_ROOT``: uploads, evidence, point clouds,
stage artifacts), with a manifest of sha256 + size for every file and an explicit status for every component. A component
is never skipped silently: ``skip`` is recorded in the manifest, and a restore that has no target for a component the
archive contains refuses instead of dropping it.

Ordering is deliberate. ``commit_version`` writes the WorldStore FIRST and the DB mirror row SECOND, so the DB is copied FIRST
and the WorldStore SECOND: every version the copied DB points at already existed in the store when the DB was copied, and the
store is append-only (immutable, content-addressed), so a copy taken while the service runs can contain extra, not-yet-adopted
versions but never a dangling pointer. Artifacts are copied LAST for the same reason.

Restore never overwrites and is all-or-nothing on disk: the archive is verified end to end first, everything is written to
``<target>.restoring`` siblings, the restored database must pass ``PRAGMA integrity_check`` and the DB<->WorldStore<->artifact
consistency check, and only then are the staged trees renamed into place (rolled back if a rename fails). A crash before the
renames leaves the real targets untouched; leftovers named ``*.restoring`` are removed by the next restore. For Postgres the
target database must be empty; it is restored first, so a failure leaves the filesystem untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urlsplit

FORMAT = 2
SUPPORTED_FORMATS = (1, 2)                    # a format-1 archive (pre-components) still verifies and restores
MANIFEST = "manifest.json"
DB_NAME = "db/app.db"                         # SQLite snapshot
PG_NAME = "db/app.pgdump"                     # pg_dump --format=custom
COMPONENTS = ("db", "worldstore", "storage")
DEFAULT_WORLDSTORE = "./data/worldstore"
DEFAULT_STORAGE = "./data/artifacts"


class BackupError(RuntimeError):
    """The archive is unusable (corrupt, tampered, unsafe), the target is not safe to restore into, or the data is inconsistent."""


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
    return [(f"{prefix}/{p.relative_to(root).as_posix()}", p) for p in sorted(root.rglob("*")) if p.is_file()]


# ----------------------------------------------------------------------------------------------------------------------
# Postgres (pg_dump / pg_restore): argv only, the password travels in PGPASSWORD and is never put on a command line.
# ----------------------------------------------------------------------------------------------------------------------

def _pg_connection(url: str) -> tuple[list[str], dict[str, str]]:
    """(connection arguments, environment) for a ``postgresql[+driver]://user:pass@host:port/db`` URL."""
    parts = urlsplit(url.replace("+asyncpg", "").replace("+psycopg2", "").replace("+psycopg", ""))
    if parts.scheme not in ("postgresql", "postgres") or not parts.path.strip("/"):
        raise BackupError("not a postgresql:// URL with a database name")
    args = ["--dbname", unquote(parts.path.strip("/"))]
    if parts.hostname:
        args += ["--host", parts.hostname]
    if parts.port:
        args += ["--port", str(parts.port)]
    if parts.username:
        args += ["--username", unquote(parts.username)]
    env = {**os.environ}
    if parts.password:
        env["PGPASSWORD"] = unquote(parts.password)
    return args, env


def _run_pg(tool: str, args: list[str], env: dict[str, str]) -> None:
    exe = shutil.which(tool)
    if exe is None:
        raise BackupError(f"{tool} not found on PATH (install the PostgreSQL client tools)")
    done = subprocess.run([exe, *args], env=env, capture_output=True, text=True, check=False)  # noqa: S603 - argv list, no shell
    if done.returncode != 0:
        raise BackupError(f"{tool} failed (exit {done.returncode}): {(done.stderr or '').strip()[-400:]}")


def _schema_revision(sqlite_file: Path) -> str | None:
    """The Alembic revision recorded in a SQLite database, or None (no alembic_version table)."""
    conn = sqlite3.connect(f"file:{sqlite_file.as_posix()}?mode=ro", uri=True)
    try:
        row = conn.execute("select version_num from alembic_version").fetchone()
        return row[0] if row else None
    except sqlite3.DatabaseError:
        return None
    finally:
        conn.close()


# ----------------------------------------------------------------------------------------------------------------------
# create
# ----------------------------------------------------------------------------------------------------------------------

def _provably_unused(db_path: Path | None, what: str) -> bool:
    """True only when the SQLite database itself proves a store has nothing in it: the WorldStore directory is created
    lazily at the first commit, so a deployment with no world version legitimately has none. Without a database to prove
    it (Postgres, skipped), a missing root is an error."""
    if db_path is None or not db_path.is_file():
        return False
    queries = {
        "world_versions": ["select count(*) from world_versions"],
        "artifacts": ["select count(*) from evidence where artifact_uri is not null",
                      "select count(*) from uploads where artifact_uri is not null",
                      "select count(*) from world_versions where points_artifact_uri is not null or cameras_artifact_uri is not null"],
    }[what]
    conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    try:
        return all(conn.execute(q).fetchone()[0] == 0 for q in queries)
    except sqlite3.DatabaseError:
        return False                                        # a table is missing: nothing is proven, so the root must exist
    finally:
        conn.close()


def create_backup(archive: Path, *, db_path: Path | None = None, worldstore_root: Path, storage_root: Path | None = None,
                  database_url: str | None = None, created_at: str | None = None) -> dict:
    """Write ``archive`` (a .tar.gz) and return its manifest. ``db_path`` (SQLite) or ``database_url`` (Postgres) selects
    the database; neither means the database is deliberately skipped (recorded). A root that is given but does not exist
    is an error, not an empty component. DB first, WorldStore second, artifacts last (see the module docstring)."""
    if db_path is not None and database_url is not None:
        raise BackupError("give either db_path (SQLite) or database_url (Postgres), not both")
    files: list[tuple[str, Path]] = []
    components: dict[str, dict] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for label, root, used in (("WorldStore", worldstore_root, "world_versions"), ("storage", storage_root, "artifacts")):
            if root is not None and not root.is_dir() and not _provably_unused(db_path, used):
                raise BackupError(f"{label} root not found: {root}")
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
            components["db"] = {"status": "included", "kind": "sqlite", "schema_revision": _schema_revision(snap)}
        elif database_url is not None:
            dump = Path(tmp) / "app.pgdump"
            conn_args, env = _pg_connection(database_url)
            _run_pg("pg_dump", ["--format=custom", "--no-owner", "--file", str(dump), *conn_args], env)
            files.append((PG_NAME, dump))
            components["db"] = {"status": "included", "kind": "postgres", "schema_revision": None}
        else:
            components["db"] = {"status": "skipped", "kind": None, "schema_revision": None}
        if worldstore_root.is_dir():
            files += _tree(worldstore_root, "worldstore")
        components["worldstore"] = {"status": "included"}     # a root that is absent but provably unused is an empty store
        if storage_root is not None:
            if storage_root.is_dir():
                files += _tree(storage_root, "storage")
            components["storage"] = {"status": "included"}
        else:
            components["storage"] = {"status": "skipped"}
        manifest = {
            "format": FORMAT,
            "created_at": created_at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "has_db": components["db"]["status"] == "included",
            "components": components,
            "files": {name: {"sha256": _sha256(p), "size": p.stat().st_size} for name, p in files},
        }
        payload = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
        archive.parent.mkdir(parents=True, exist_ok=True)
        partial = archive.with_name(archive.name + ".partial")
        with tarfile.open(partial, "w:gz") as tar:
            info = tarfile.TarInfo(MANIFEST)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
            for name, p in files:
                tar.add(str(p), arcname=name, recursive=False)
        os.replace(partial, archive)                  # an interrupted backup never leaves a truncated archive under the real name
    return manifest


# ----------------------------------------------------------------------------------------------------------------------
# verify
# ----------------------------------------------------------------------------------------------------------------------

def _safe_name(name: str) -> bool:
    p = Path(name)
    return not (p.is_absolute() or ".." in p.parts or name.startswith(("/", "\\")) or ":" in name)


def _components_of(manifest: dict) -> dict[str, dict]:
    """The component table of a manifest; a format-1 archive (no table) is derived from what it contains."""
    if "components" in manifest:
        return manifest["components"]
    files = manifest["files"]
    return {
        "db": {"status": "included", "kind": "sqlite", "schema_revision": None} if DB_NAME in files else {"status": "skipped"},
        "worldstore": {"status": "included"},
        "storage": {"status": "included"} if any(n.startswith("storage/") for n in files) else {"status": "skipped"},
    }


def verify_backup(archive: Path) -> dict:
    """Check the archive end to end (manifest present and supported, every member named safely, every hash and size matching,
    no member missing or extra). Returns the manifest; raises ``BackupError`` otherwise. Writes nothing."""
    try:
        with tarfile.open(archive, "r:gz") as tar:
            members = {m.name: m for m in tar.getmembers() if m.isfile()}
            if MANIFEST not in members:
                raise BackupError("archive has no manifest.json")
            manifest = json.loads(tar.extractfile(members[MANIFEST]).read().decode("utf-8"))
            if manifest.get("format") not in SUPPORTED_FORMATS:
                raise BackupError(
                    f"unsupported backup format {manifest.get('format')!r} (this version reads {SUPPORTED_FORMATS}); "
                    "restore it with the release that wrote it")
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


# ----------------------------------------------------------------------------------------------------------------------
# consistency: the database, the WorldStore and the artifact store must agree
# ----------------------------------------------------------------------------------------------------------------------

def _artifact_path(root: Path, uri: str) -> Path | None:
    """The file behind a ``sha256://<hex>`` artifact URI in the content-addressed layout, or None for another scheme."""
    if not uri or not uri.startswith("sha256://"):
        return None
    digest = uri[len("sha256://"):]
    return root / digest[:2] / digest[2:4] / digest if len(digest) == 64 else None


def check_consistency(*, db_path: Path | None, worldstore_root: Path | None, storage_root: Path | None, deep: bool = False) -> dict:
    """Read-only. Every ``world_versions`` row must point at a WorldStore version whose bytes verify (no dangling pointer,
    no corrupt version), and every ``sha256://`` artifact the database references (versions' points/cameras, evidence,
    uploads) must exist in the artifact store; ``deep`` re-hashes them. Returns ``{"checked": {...}, "problems": [...],
    "not_checked": [...]}``; tables a database does not have are skipped and listed, never assumed fine."""
    report: dict = {"checked": {"world_versions": 0, "artifacts": 0}, "problems": [], "not_checked": []}
    if db_path is None:
        report["not_checked"].append("database (Postgres or skipped): row <-> store consistency needs the database")
        return report
    conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    try:
        tables = {r[0] for r in conn.execute("select name from sqlite_master where type='table'")}
        if worldstore_root is not None and "world_versions" in tables:
            from worldstore.store import WorldStore, WorldStoreError

            store = WorldStore(worldstore_root)
            have: set[str] = set()
            try:
                have = {v.version_id for v in store.list_versions()}
            except WorldStoreError as exc:
                report["problems"].append(f"WorldStore history unreadable: {exc}")
            for (vid,) in conn.execute("select id from world_versions"):
                report["checked"]["world_versions"] += 1
                if vid not in have:
                    report["problems"].append(f"world_versions {vid}: no such version in the WorldStore (dangling pointer)")
                    continue
                for failure in store.verify_version(vid):
                    report["problems"].append(f"world_versions {vid}: {failure['reason']}")
        elif "world_versions" in tables:
            report["not_checked"].append("world_versions <-> WorldStore: no WorldStore root")

        refs: list[tuple[str, str]] = []
        for table, cols in (("world_versions", ("points_artifact_uri", "cameras_artifact_uri")),
                            ("evidence", ("artifact_uri",)), ("uploads", ("artifact_uri",))):
            if table not in tables:
                report["not_checked"].append(f"table {table}: absent")
                continue
            for col in cols:
                refs += [(f"{table}.{col}", r[0]) for r in conn.execute(f"select {col} from {table} where {col} is not null")]
        if storage_root is None:
            if refs:
                report["not_checked"].append(f"{len(refs)} artifact reference(s): no artifact store root")
        else:
            for where, uri in refs:
                path = _artifact_path(storage_root, uri)
                if path is None:
                    report["not_checked"].append(f"{where}: unrecognised artifact URI scheme {uri.split(':')[0]!r}")
                    continue
                report["checked"]["artifacts"] += 1
                if not path.is_file():
                    report["problems"].append(f"{where}: artifact {uri} is missing from the artifact store")
                elif deep and _sha256(path) != uri[len("sha256://"):]:
                    report["problems"].append(f"{where}: artifact {uri} does not match its content hash")
    finally:
        conn.close()
    return report


# ----------------------------------------------------------------------------------------------------------------------
# restore
# ----------------------------------------------------------------------------------------------------------------------

def _staging(target: Path) -> Path:
    return target.with_name(target.name + ".restoring")


def _discard(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    elif path.exists():
        path.unlink()


def restore_backup(archive: Path, *, db_path: Path | None = None, worldstore_root: Path | None = None,
                   storage_root: Path | None = None, database_url: str | None = None, skip: tuple[str, ...] = (),
                   check: bool = True) -> dict:
    """Verify, then restore into targets that do not exist yet or are empty. Never overwrites; all-or-nothing on disk.
    Every component the archive contains needs a target or an explicit entry in ``skip`` (``"db"``, ``"worldstore"``,
    ``"storage"``). Returns a summary including the consistency report."""
    manifest = verify_backup(archive)
    comps = _components_of(manifest)
    unknown = set(skip) - set(COMPONENTS)
    if unknown:
        raise BackupError(f"unknown component(s) in skip: {sorted(unknown)}")
    kind = comps["db"].get("kind") if comps["db"]["status"] == "included" else None
    targets = {"worldstore": worldstore_root, "storage": storage_root}
    for name in ("worldstore", "storage"):
        if comps[name]["status"] == "included" and name not in skip and targets[name] is None:
            raise BackupError(f"the archive contains the {name} component but no {name} target was given "
                              f"(pass skip=({name!r},) to drop it deliberately)")
    if kind == "sqlite" and "db" not in skip and db_path is None:
        raise BackupError("the archive contains a SQLite database but no database target was given (skip it explicitly with --skip-db)")
    if kind == "postgres" and "db" not in skip and database_url is None:
        raise BackupError("the archive contains a Postgres dump but no --database-url (an EMPTY database) was given (or skip it with --skip-db)")
    for name, target in targets.items():
        if target is not None and name not in skip and comps[name]["status"] == "included":
            if target.exists() and any(target.iterdir()):
                raise BackupError(f"refusing to restore over a non-empty directory: {target}")
    if kind == "sqlite" and "db" not in skip and db_path.exists():
        raise BackupError(f"refusing to restore over an existing database: {db_path}")

    planned = {n: t for n, t in targets.items() if t is not None and n not in skip and comps[n]["status"] == "included"}
    sqlite_target = db_path if kind == "sqlite" and "db" not in skip else None
    every_target = [*planned.values(), *([sqlite_target] if sqlite_target is not None else [])]
    # Leftovers of an interrupted earlier restore are ours (named *.restoring) and are discarded; real targets are untouched.
    for t in every_target:
        _discard(_staging(t))

    written = 0
    report: dict = {"checked": {}, "problems": [], "not_checked": ["consistency not run (check=False)"]}
    try:
        if kind == "postgres" and "db" not in skip:
            conn_args, env = _pg_connection(database_url)
            with tempfile.TemporaryDirectory() as tmp, tarfile.open(archive, "r:gz") as tar:
                dump = Path(tmp) / "app.pgdump"
                with open(dump, "wb") as out:
                    shutil.copyfileobj(tar.extractfile(PG_NAME), out)
                _run_pg("pg_restore", ["--no-owner", "--exit-on-error", *conn_args, str(dump)], env)
            written += 1
        with tarfile.open(archive, "r:gz") as tar:
            for name, meta in manifest["files"].items():
                top, _, rel = name.partition("/")
                if name == DB_NAME:
                    if sqlite_target is None:
                        continue
                    out_path = _staging(sqlite_target)
                elif top in planned:
                    out_path = _staging(planned[top]) / rel
                else:
                    continue                                   # a skipped (or Postgres-dump) member
                out_path.parent.mkdir(parents=True, exist_ok=True)
                h = hashlib.sha256()
                with open(out_path, "wb") as out:
                    src = tar.extractfile(name)
                    for chunk in iter(lambda src=src: src.read(1 << 20), b""):
                        h.update(chunk)
                        out.write(chunk)                       # streamed: artifacts can be far larger than memory
                if h.hexdigest() != meta["sha256"]:
                    raise BackupError(f"checksum mismatch while writing {name}")
                written += 1
        for t in planned.values():
            _staging(t).mkdir(parents=True, exist_ok=True)     # an included but empty component still becomes a directory
        if sqlite_target is not None:
            conn = sqlite3.connect(str(_staging(sqlite_target)))
            try:
                verdict = conn.execute("PRAGMA integrity_check").fetchone()[0]
            finally:
                conn.close()
            if verdict != "ok":
                raise BackupError(f"restored database failed integrity_check: {verdict}")
        if check:
            report = check_consistency(
                db_path=_staging(sqlite_target) if sqlite_target is not None else None,
                worldstore_root=_staging(planned["worldstore"]) if "worldstore" in planned else None,
                storage_root=_staging(planned["storage"]) if "storage" in planned else None)
            if report["problems"]:
                raise BackupError("restored data is inconsistent: " + "; ".join(report["problems"][:5]))
        # Commit: rename the staged trees into place; if one rename fails, undo the ones already done.
        done: list[Path] = []
        try:
            for t in every_target:
                t.parent.mkdir(parents=True, exist_ok=True)
                if t.is_dir():
                    t.rmdir()                                   # an existing EMPTY directory (checked above)
                os.replace(_staging(t), t)
                done.append(t)
        except OSError as exc:
            for t in reversed(done):
                os.replace(t, _staging(t))
            raise BackupError(f"could not move the restored data into place: {exc}") from exc
    finally:
        for t in every_target:
            _discard(_staging(t))
    return {"files_restored": written, "created_at": manifest["created_at"], "format": manifest["format"],
            "schema_revision": comps["db"].get("schema_revision"), "consistency": report}


# ----------------------------------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------------------------------

def _env_targets(skip_db: bool, skip_storage: bool) -> tuple[Path | None, str | None, Path, Path | None]:
    """(sqlite file, postgres URL, WorldStore root, storage root) from the environment, defaulting exactly like the API."""
    url = os.environ.get("DATABASE_URL", "sqlite+aiosqlite:///./data/app.db")
    sqlite_file, pg_url = None, None
    if not skip_db:
        sqlite_file = sqlite_path_from_url(url)
        if sqlite_file is None:
            if not url.startswith("postgres"):
                raise BackupError(f"unsupported DATABASE_URL scheme ({url.split(':')[0]}); use --skip-db and back the database up yourself")
            pg_url = url
    storage = None if skip_storage else Path(os.environ.get("STORAGE_ROOT") or DEFAULT_STORAGE)
    return sqlite_file, pg_url, Path(os.environ.get("WORLDSTORE_ROOT") or DEFAULT_WORLDSTORE), storage


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m apps.backup", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("create", "verify", "restore"):
        p = sub.add_parser(name)
        p.add_argument("archive", type=Path)
        if name != "verify":
            p.add_argument("--skip-db", action="store_true")
            p.add_argument("--skip-storage", action="store_true")
        if name == "restore":
            p.add_argument("--into", type=Path, required=True, help="directory to restore into (app.db, worldstore/, storage/)")
            p.add_argument("--database-url", help="an EMPTY Postgres database to restore a Postgres dump into")
    chk = sub.add_parser("check", help="read-only consistency check of the live data")
    chk.add_argument("--deep", action="store_true", help="re-hash every referenced artifact")
    args = ap.parse_args(argv)
    try:
        if args.cmd == "verify":
            m = verify_backup(args.archive)
            print(f"OK: {len(m['files'])} files, format {m['format']}, created {m['created_at']}")
        elif args.cmd == "create":
            db, pg, ws, st = _env_targets(args.skip_db, args.skip_storage)
            m = create_backup(args.archive, db_path=db, database_url=pg, worldstore_root=ws, storage_root=st)
            skipped = [n for n, c in m["components"].items() if c["status"] == "skipped"]
            print(f"wrote {args.archive}: {len(m['files'])} files" + (f" (skipped: {', '.join(skipped)})" if skipped else ""))
        elif args.cmd == "check":
            db, _pg, ws, st = _env_targets(False, False)
            rep = check_consistency(db_path=db, worldstore_root=ws, storage_root=st, deep=args.deep)
            print(json.dumps(rep, indent=2))
            return 2 if rep["problems"] else 0
        else:
            root = args.into
            skip = tuple(n for n, flag in (("db", args.skip_db), ("storage", args.skip_storage)) if flag)
            res = restore_backup(args.archive, db_path=root / "app.db", worldstore_root=root / "worldstore",
                                 storage_root=root / "storage", database_url=args.database_url, skip=skip)
            print(f"restored {res['files_restored']} files into {root} (backup taken {res['created_at']}, "
                  f"schema {res['schema_revision'] or 'unknown'}); consistency: {len(res['consistency']['problems'])} problem(s)")
    except BackupError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
