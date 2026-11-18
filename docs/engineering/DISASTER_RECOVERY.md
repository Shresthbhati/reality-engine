# Backup, restore and disaster recovery

Operational runbook for `python -m apps.backup`. Behaviour is defined by `apps/backup.py` and verified by
`tests/test_backup_restore.py`, `tests/test_backup_durability.py` and `tests/test_backup_lifecycle.py`.

## What is protected

| Component | What | How it is saved |
|---|---|---|
| Database | worlds, sessions, jobs, stage checkpoints, version mirror rows, activity, tenants | SQLite: SQLite online-backup copy (consistent while the API writes). Postgres: `pg_dump --format=custom` |
| WorldStore | immutable, content-addressed world versions (`WORLDSTORE_ROOT`) | file tree, every file hashed |
| Artifact store | uploads, evidence, point clouds, stage artifacts (`STORAGE_ROOT`, default `./data/artifacts`) | file tree, every file hashed |

Uploads are not a separate tree: they are content-addressed blobs in the artifact store. Job checkpoints live in the database.
A component is never skipped silently: `--skip-db` / `--skip-storage` are recorded in the archive's manifest, and a restore
refuses to drop a component the archive contains unless you skip it explicitly.

## Backup

```powershell
python -m apps.backup create backups/2026-11-18.tar.gz        # uses DATABASE_URL, WORLDSTORE_ROOT, STORAGE_ROOT
python -m apps.backup verify backups/2026-11-18.tar.gz        # re-hashes every file; run it after copying the archive off-box
python -m apps.backup check                                   # live data: DB <-> WorldStore <-> artifacts (add --deep to re-hash artifacts)
```

- The service can keep running. The database is copied first and the WorldStore second, so every version a copied row points
  at is already in the store; the store is append-only, so the copy can hold extra versions but never a dangling pointer.
- An archive is written under a `.partial` name and renamed on success: a crash never leaves a truncated archive.
- A missing store directory is an error, except when the SQLite database proves the store is empty (the WorldStore directory
  is created at the first commit, so a brand-new deployment has none).
- Postgres needs `pg_dump` on `PATH`; the password travels in `PGPASSWORD`, never on a command line.
- Take backups on a schedule and copy the archive to storage that does not share a failure domain with the data. This repo
  ships no scheduler.

## Restore

```powershell
python -m apps.backup verify backups/2026-11-18.tar.gz
python -m apps.backup restore backups/2026-11-18.tar.gz --into D:\reality\restored
# Postgres archive: restore into an EMPTY database
python -m apps.backup restore backups/pg.tar.gz --into D:\reality\restored --database-url postgresql://user:pass@host/emptydb
```

Restore never overwrites: the targets must not exist or must be empty. It is all-or-nothing on disk:

1. The archive is verified end to end before anything is written (hashes, sizes, member names, format).
2. Everything is written to `<target>.restoring` siblings and re-hashed while writing.
3. The database must pass `PRAGMA integrity_check`, and the DB <-> WorldStore <-> artifact consistency check must find no
   problem (no version row without its stored version, no corrupt version, no missing artifact).
4. Only then are the staged trees renamed into place; if a rename fails, the ones already done are undone.

A crash before step 4 leaves the real targets untouched. Leftovers named `*.restoring` are discarded by the next restore.
Point the service at the restored directory (`DATABASE_URL`, `WORLDSTORE_ROOT`, `STORAGE_ROOT`) and start it. Then run
`python -m apps.backup check` and open a world.

## Disaster-recovery procedure

1. Stop writers; if the old host is reachable, take one last `create` first.
2. Pick the newest archive that `verify` accepts.
3. Restore into a clean directory (or an empty Postgres database) as above. If restore reports the data is inconsistent, the
   archive was taken from a damaged deployment: try the previous archive and report the failure.
4. Start the service on the restored paths. Schema migrations run as usual (`alembic -c apps/api/alembic.ini upgrade head`);
   the restore summary shows the archive's recorded schema revision.
5. Verify: `check`, load a world in Reality Studio, confirm the version history is the expected length.

## Compatibility

- Archives carry `format`. This release reads formats 1 and 2 (format 1 predates the component table). An archive from a
  newer release is refused with the supported formats listed: restore it with the release that wrote it.
- Format 2 records the SQLite database's Alembic revision. Restoring an archive from an older schema is expected to work
  because the API upgrades the schema on start; restoring one from a NEWER schema into older code is not supported.

## What is verified and what is not

Verified (synthetic, local): round trip of every file and the database; tamper, traversal and non-overwrite refusal;
skipped-component refusal; all-or-nothing restore under a failing rename and under a crash; consistency failure blocks the
restore; Postgres argument/environment handling and failure paths against fake `pg_dump`/`pg_restore`; the full API
lifecycle (create world, upload, reconstruct, backup, restore into a clean directory, restart, reload, compare versions,
entities, geometry, relationships, provenance and evidence bytes, edit, new version, export).

NOT verified: a real PostgreSQL server (no server here: the Postgres path is checked at the process boundary only, and the
DB <-> store consistency check does not run for Postgres, it is reported under `not_checked`); restores across operating
systems; very large artifact stores (restore is streamed, but no size benchmark exists); scheduling and off-site copy.
