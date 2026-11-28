# Tenant Isolation and Evidence Ownership Remediation

**Branch:** `security/tenant-isolation-evidence-ownership`  
**Base (`origin/main`):** `34a073f25f57c47f23a5efc0e28c30a13bcbe1ba`  

---

## 1. Root-Cause Analysis (Phase 0)

### 1.1 Global Evidence Deduplication (`apps/api/routes_misc.py:create_upload`)
- `create_upload` queried `select(Evidence).where(Evidence.checksum == digest)` globally without scoping to `team_id` or `session_id`.
- When Tenant B (or a second session) uploaded byte-identical content to Tenant A, `create_upload` returned Tenant A's `evidence_id`, created **no** `Evidence` row for Tenant B's `session_id`, and enqueued `PROCESS_EVIDENCE` against Tenant A's `Evidence` row (`team_id=existing_ev.team_id`).
- **CI Regression Impact:** This single defect also caused both `tests/test_furniture_commands_and_undo.py::test_cross_world_undo_isolation` (`KeyError: 'entities'`) and `tests/test_reconstruction_reliability.py::test_same_world_concurrent_completion_single_head` (which also had a missing `def test_...` header merging two tests at line 619) to fail with `InsufficientEvidence: World has 0 usable photo evidence item(s)` when two sessions uploaded identical dummy bytes.

### 1.2 World & Export API Authorization Gaps (`apps/api/routes_worlds.py`, `apps/api/routes_export.py`)
- `worlds = APIRouter(prefix="/api/worlds", tags=["worlds"])` lacked router-level `dependencies=[Depends(world_scope)]`.
- Eight endpoints in `routes_worlds.py` (`world_report`, `world_review_coco`, `world_points`, `world_cameras`, `get_scale_suggestion`, `reject_scale_suggestion`, `world_quality`, `get_world_spatial_intelligence`) omitted `auth: AuthContext = Depends(get_auth_context)` entirely and called `_version_row(db, world_id, version)` without `auth`.
- `entity_provenance` accepted `auth` but omitted `auth=auth` in `_version_row(...)`.
- `get_world_diff`, `get_world_space_graph`, and `query_world_spaces` loaded `WorldVersion` / `WorldStore` artifacts by `version_id` without verifying `WorldVersion.world_id == world_id`, allowing a tenant to pass another tenant's `version_id` alongside their own `world_id`.
- `download_export` in `routes_export.py` accepted `(world_id, content_hash, format)` without checking `ExportRecord` ownership or `auth`, allowing any tenant with their own `world_id` to download another tenant's export `content_hash`.

### 1.3 Background Job Ownership (`apps/api/jobs.py`)
- `_run_process_evidence` and `_run_reconstruct_session_inner` (`seg_evidence_resolve`) did not re-verify that `job.team_id`, `session.team_id`, `world.team_id`, and `Evidence.team_id` match before processing.

---

## 2. Tenant-Isolation Invariants (Phase 3)

| Invariant | Guarantee | Enforcement Mechanism |
|---|---|---|
| **A. Tenant Isolation** | Tenant A cannot read, modify, delete, export, or enumerate Tenant B's resources. | `world_scope`, `session_scope`, `evidence_scope`, `upload_scope`, `job_scope` + `enforce_project_isolation` + `visible()` SQL filters. |
| **B. Evidence Isolation** | Identical bytes uploaded by Tenant B never expose or reference Tenant A's `evidence_id`. | Physical blobs (`sha256://`) are shared; logical `Evidence` dedup is scoped to `(team_id, session_id, checksum)` with partial unique index `uq_evidence_team_session_checksum`. |
| **C. World Isolation** | Knowing another tenant's `world_id`, `version_id`, `entity_id`, or `content_hash` grants no access. | `world_scope` on `worlds` router, `_version_row(..., auth=auth)` verifying `WorldVersion.world_id == world_id`, and `ExportRecord` check in `download_export`. |
| **D. Session Isolation** | A session cannot consume unauthorized evidence or attach to an unauthorized world. | `attach_session` checks both `World` and `Session`; `seg_evidence_resolve` enforces `session.team_id == world.team_id` and `Evidence.team_id == world.team_id`. |
| **E. Version Isolation** | Version lookup always establishes parent `World` ownership. | `_version_row(db, world_id, version_id, auth=auth)` enforces `enforce_project_isolation(w, auth)` and `row.world_id == world_id`. |
| **F. Background Job Isolation** | Workers independently validate ownership of jobs, sessions, worlds, and evidence at execution time. | `_run_process_evidence` and `seg_evidence_resolve` refuse cross-tenant `job.team_id`/`session.team_id`/`world.team_id`/`Evidence.team_id`. |
| **G. Export Isolation** | Export creation and download enforce world and `ExportRecord` ownership. | `export` router uses `world_scope`, `export_world`/`export_package` pass `auth=auth`, `download_export` verifies `ExportRecord.world_id == world_id`. |
| **H. Error Isolation** | Foreign IDs return 404 matching unknown IDs without leaking checksums, filenames, or existence. | `_check` / `enforce_project_isolation` / `download_export` return uniform 404s. |

---

## 3. World & Export API Endpoint Authorization Matrix (Phase 2)

All routes under `/api/worlds` are now protected at the router boundary by `Depends(world_scope)` (except `/api/worlds/shared/access/{token}`, which has no `{world_id}` path param and validates `sh1:` share tokens) AND at the handler/version level:

| Endpoint | Auth Dependency | Ownership / Version Check |
|---|---|---|
| `POST /api/worlds`, `POST /api/worlds/import/package`, `GET /api/worlds` | `get_auth_context` | Stamps `team_id=auth.tenant_id`; `list_worlds` filters `World.team_id == auth.tenant_id` |
| `GET/PATCH/DELETE /api/worlds/{world_id}`, `duplicate`, `archive`, `restore`, `versions`, `capture-requests`, `coverage`, `rollback`, `scale`, `camera`, `thumbnail`, `observations` | `world_scope` + `get_auth_context` | `_world_or_404(db, world_id, auth)` + `enforce_project_isolation(w, auth)` |
| `GET /api/worlds/{world_id}/find`, `/worldir`, `/entities/{eid}/provenance`, `/entities/{eid}/confidence` | `world_scope` + `get_auth_context` | `_version_row(db, world_id, version, auth=auth)` |
| `GET /api/worlds/{world_id}/report`, `/review/coco`, `/points`, `/cameras`, `/quality`, `/spatial-intelligence` | `world_scope` + `get_auth_context` | `_version_row(db, world_id, version, auth=auth)` |
| `GET /api/worlds/{world_id}/scale/suggestion`, `POST .../reject` | `world_scope` + `get_auth_context` | `_version_row(db, world_id, None, auth=auth)` |
| `GET /api/worlds/{world_id}/diff`, `/space-graph`, `/query/spaces` | `world_scope` + `get_auth_context` | `_version_row(db, world_id, ..., auth=auth)` verifies `WorldVersion.world_id == world_id` |
| `POST /api/worlds/{world_id}/commit`, `/corrections/*`, `/commands`, `/undo`, `/redo`, `/refine`, `/shares/*`, `/options/*`, `/views/*`, `/render` | `world_scope` + `get_auth_context` | `enforce_project_isolation(w, auth)` + `_version_row(..., auth=auth)` |
| `POST /api/worlds/{world_id}/export`, `/compare`, `GET /package` | `world_scope` + `get_auth_context` | `_version_row(db, world_id, ..., auth=auth)` + records `ExportRecord` |
| `GET /api/worlds/{world_id}/export/{content_hash}/download` | `world_scope` + `get_auth_context` | `enforce_project_isolation(w, auth)` + `ExportRecord(world_id, artifact_uri)` |

---

## 4. Database Migration & Compatibility Report (Phase 5)

- **Migration:** `apps/api/migrations/versions/0007_evidence_ownership_indexes.py` (revises `0006`).
- **Changes:**
  - `ix_uploads_team_session_checksum` on `uploads (team_id, session_id, checksum)`
  - `ix_evidence_team_session_checksum` on `evidence (team_id, session_id, checksum)`
  - `uq_evidence_team_session_checksum` partial unique index on `evidence (team_id, session_id, checksum)` where `session_id IS NOT NULL AND checksum IS NOT NULL`.
- **Compatibility & Safety:**
  - Checks existing index names via `sa.inspect(bind)` before creating/dropping (safe on `init_db()`-initialized dev DBs).
  - Reversible `downgrade()` drops all three indexes cleanly.
  - Legacy `team_id=NULL` rows remain unowned (`NULL` is only visible to `SYSTEM_ADMIN` via `visible()`), never silently reassigned.
  - Concurrent duplicate uploads within the same `(team_id, session_id)` are protected by `uq_evidence_team_session_checksum` + `IntegrityError` fallback in `create_upload`.

---

## 5. Remaining Security Risks

1. **Legacy `ActivityEvent` / `WorldVersion` rows with `team_id=NULL`:** Legacy records predating tenant columns remain visible to `SYSTEM_ADMIN` only; operators must backfill or purge them explicitly.
2. **Loopback unauthenticated fallback (`REALITY_API_TOKEN` unset):** When `REALITY_API_TOKEN` is unset, loopback clients (`127.0.0.1`, `::1`) can supply `X-Tenant-Id` and `X-Role` headers (expected for local dev, must never be deployed without `REALITY_API_TOKEN`).
3. **`download_export` for legacy exports predating `ExportRecord`:** Any export artifact created before migration `0004` (without an `ExportRecord` row) will return 404 on `/download` and must be re-exported.
4. **Unrelated CI failures on `main`:** `test_gltf_exporter.py::test_mesh_geometry_without_resolvable_artifact_falls_back_to_cube`, `test_matcher_policy.py` (missing `torch`/`kornia` in `.[dev]`), `test_ci_test_registry.py::test_doc_is_current`, and Ruff baseline regression remain outside the scope of this security PR.
