# 13 — Backend, persistence and world lifecycle

## Stack (keep what exists)

- **API**: FastAPI (`apps/api`), async SQLAlchemy.
- **DB**: SQLite (`./data/app.db`) for dev; Postgres (+PostGIS later) via `DATABASE_URL`.
- **Migrations**: today `metadata.create_all`. **Add Alembic** before any column
  change ships (Phase 1). Never hand-edit a user's DB.
- **Object storage**: content-addressed files (`apps/api/storage.py`, `sha256://` URIs);
  local disk now, S3-compatible behind the same interface later.
- **Versioned worlds**: `worldstore/store.py` (immutable WorldIR versions);
  the `world_versions` table mirrors lineage for fast listing.
- **Jobs**: DB-backed queue in `apps/api/jobs.py` with claim, heartbeat,
  stale-job recovery, retries, cancel and subprocess ownership. Keep it. Do not
  add Celery/Redis for v1.

## Tables (current → target)

| Table | Exists | Change needed |
|-------|--------|---------------|
| `worlds` | yes | add `archived_at`, `deleted_at` (soft delete), `thumbnail_uri`, `scale_state`. `status` values: `active · archived · trashed`. |
| `world_versions` | yes | add `label`, `created_by_job_id`. |
| `sessions` | yes | keep; `world_id` stays nullable (a session can exist alone). |
| `uploads`, `evidence`, `evidence_decisions` | yes | keep; evidence rows reference sha256 URIs. |
| `jobs` | yes | add `stage` and `stage_counts` (JSON) if missing; keep heartbeat/attempts. |
| `world_measurements` | yes | keep the scale-state rule (P10). |
| `completion_proposals` | yes | keep; proposals are `PRIOR_GENERATED` until accepted. |
| `capture_requests` | **no** | new: mirror of WorldIR CaptureRequests for listing/status. |
| `activity_events`, `notifications` | yes | keep; every lifecycle operation writes an activity event (audit trail). |
| `users`, `workspaces`, `memberships` | **no** | **design only in v1**: every row already has `owner_id`/`team_id`; keep them filled with the local user ID so auth can be added without a data migration. |

## World lifecycle (P12) — required API

| Operation | Endpoint | Behaviour |
|-----------|----------|-----------|
| Create | `POST /worlds` | exists |
| List | `GET /worlds?status=active\|archived\|trashed` | add the status filter |
| Open | `GET /worlds/{id}` | exists |
| Rename / edit | `PATCH /worlds/{id}` `{name, description}` | **new**; identity unchanged |
| Duplicate | `POST /worlds/{id}/duplicate` | **new**; copies metadata and points the new world at the same immutable current version (copy-on-write; no file copies) |
| Archive | `POST /worlds/{id}/archive` | **new**; `status=archived`, hidden from the default list |
| Restore | `POST /worlds/{id}/restore` | **new**; archived or trashed → active |
| Delete | `DELETE /worlds/{id}` | **new**; soft delete: `status=trashed`, `deleted_at=now`. Running jobs for the world are cancelled first. |
| Purge | `DELETE /worlds/{id}?permanent=true`, plus a daily sweep after 30 days | **new**; deletes rows + WorldStore versions + unreferenced blobs (blob GC checks references first) |
| Versions | `GET /worlds/{id}/versions`, `POST /worlds/{id}/rollback` | exist |
| Export | `POST /export` + download | exists; fix request-path consistency and error surfacing (Phase 1) |
| Import | `POST /worlds/import` (world package zip) | Phase 11 |

Delete rules: the UI confirmation states what is deleted (sessions, photos,
versions, exports). Sessions attached only to this world go to trash with it;
evidence blobs are kept while any other row references them.

## Session merge rule (P1)

Attaching a session to a world (`POST /worlds/{id}/attach/{session_id}`, exists)
must run registration first. If the new photos do not register to the world,
the attach is **refused with the reason**, never silently merged.

## Reliability rules

- A failed job never leaves a world looking successful: the world's
  `current_version_id` moves only on a successful S14 commit (atomic).
- Every job ends in exactly one of `succeeded · failed · cancelled`, with `error` text on failure.
- A worker restart recovers stale `running` jobs (exists — keep its test).
- Uploads: size limits (exist), type check; resumable uploads later (Phase 11).

## Settings that must be real (P14)

| Area | v1 | Note |
|------|----|------|
| Appearance (theme, density, reduced motion) | real, localStorage | partly exists |
| Processing (GPU on/off, max image size, dense on/off) | real, stored in `config/` via the API | new |
| Data (storage used, trash retention, purge now) | real | new |
| Account, Workspace, Integrations | shown as "Not available in this version" | never fake controls |
