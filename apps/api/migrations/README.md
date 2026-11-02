# API database migrations (Alembic)

Run from the repo root (the DB is `DATABASE_URL`, same as the API; default `./data/app.db`):

    alembic -c apps/api/alembic.ini upgrade head      # build / upgrade
    alembic -c apps/api/alembic.ini current           # which revision
    alembic -c apps/api/alembic.ini revision --autogenerate -m "what changed" --rev-id 0003

Revisions: `0001` baseline = the schema `metadata.create_all` produced before Alembic; `0002` = doc 13 columns
(worlds `archived_at/deleted_at/thumbnail_uri/scale_state`, world_versions `label/created_by_job_id`,
jobs `stage_counts`, new `capture_requests`).

## Two paths, one schema

- `apps/api/db.py::init_db` (runs at API startup and in tests) still does `create_all` and then adds any
  *nullable* model column an existing table lacks. It never drops or rewrites anything.
- `alembic upgrade head` reaches the same schema. Every revision is **idempotent** (it checks the live schema before
  creating a table/column), so it also works on a database that `create_all`/`init_db` built and that has no
  `alembic_version` row yet. No stamping step is needed; the first `upgrade head` records the revision.
- Rule for new changes: add the column to `models.py` **and** write a new revision that checks before it acts.
  A NOT NULL column without a default cannot be auto-added by `init_db` (it raises) and needs a migration.
- Never hand-edit a user's DB; back up `data/app.db` before upgrading real data. SQLite: use
  `op.batch_alter_table` for anything beyond adding a nullable column. Postgres needs a sync driver for the CLI
  (`psycopg2`); the `+asyncpg` / `+aiosqlite` suffix is dropped automatically.
