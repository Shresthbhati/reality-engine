# Stash `stash@{6}` — Handoff for OpenCode (Backend Recovery Owner)

**Author:** Buffy (perception/reconstruction engineering agent)
**Date:** 2026-11-26
**Branch context:** `feat/day4-real-geometry-measurements` @ `741d1b19` (PR #395 MERGED per gh)
**Audience:** OpenCode — backend recovery owner
**Purpose:** Hand off the precise findings about `stash@{6}` so OpenCode can decide what to port, without me applying the stash (mandate §9: "Do not apply stash@{6}. OpenCode owns backend recovery. Send OpenCode the precise findings about the reconstruction-job conflict and potentially reusable heartbeat-based recovery.")

---

## 1. What stash@{6} contains

Stashed on `main` by "other-agent-wip-backend" — a prior agent session's in-progress backend work that was never committed. Full diff in recovery report §4.1 / Appendix B.

**Files modified:**
- `apps/api/jobs.py` — major: `_run_reconstruct_session` fully implemented, `reclaim_abandoned_jobs` added, `worker_loop` updated with reclamation + `CancelledError` handling, `HEARTBEAT_STALE_SECONDS` added
- `apps/api/main.py` — minor: logging import + graceful-shutdown wait (30 s)
- `apps/api/routes_misc.py` — medium: upload dedup in `create_upload`, `verify_artifact` import, removed unused model imports (Job, Location, Notification)
- `apps/api/routes_worlds.py` — major: 4 new routes (worldir, cameras, points, diff) + `_load_worldir_from_store` helper
- `apps/api/storage.py` — minor: local `verify_artifact()` function (sha256 integrity check)
- `apps/cli/main.py` — medium: incremental-update CLI command (`cmd_incremental_update`), GNSS anchor loading, `apply_reconstruction_update` / `apply_reconstruction_update_tiled`
- `evidence/clocks.py` — minor: methods-registry text update
- `pyproject.toml` — trivial: colmap optional-dep comment
- `tests/test_application_api.py` — minor: `test_reconstruct_fails_with_honest_error_when_no_evidence` expanded to directly test `_run_reconstruct_session`
- `.obsidian/graph.json` + `.obsidian/workspace.json` — trivial: editor state (ignore)

---

## 2. What I ported already (done this session)

**Upload dedup** — `apps/api/routes_misc.py::create_upload` + test `test_upload_deduplicates_byte_identical_content` — committed as `2e15cd44`, pushed, test passes. This was the one genuinely-improving, non-conflicting piece.

Reasoning: the current `create_upload` had NO dedup — every upload, even byte-identical re-uploads, minted a new Upload + Evidence + processing job. The stash's checksum-based dedup (reuse existing Evidence row + artifact_uri when the sha256 matches) is a real improvement that touches the same function, so I ported it directly rather than applying the stash.

---

## 3. What I did NOT port, and why (the conflict)

### 3.1 `_run_reconstruct_session` (the big one) — CONFLICTING

**Stash version:** A single-function rewrite: load session + evidence from DB → convert to EvidenceItem → run `ReconstructionOrchestrator` with `ColmapReconstructionBackend` → `reality.compile_world_from_reconstruction` → `WorldStore(save_version)` → verify read-back → DB World record → link session → job result with world_id/version_id/entities/geometries.

**Current branch version (`_run_reconstruct_session` + `_run_reconstruct_session_inner`):** A stage-segmented vertical slice: `seg_evidence_resolve` → `seg_evidence_quality` → `seg_reconstruct` (via `run_progressive` + `vertical_slice`, with `VerticalSliceOptions`, `ColmapSession`, head-snapshot, head-frame, head-unit-k, dense-auto, dense-cpu, layout analyzer, furniture detector) → compile + validate → changeset commit → manifest grading (`_grade_result` with SUCCEEDED/PARTIAL/FAILED/CANCELLED, `degraded` reasons, integrity findings, stage telemetry, per-stage timing). WorldStore is accessed via `worldstore_service.commit_version` (service layer with CAS adoption, duplicate detection, quarantine). The COLMAP session lifecycle is wrapped with an `abandoned` threading.Event so a cancelled/timed-out handler discards staging.

**Why conflicting:** Same function name + signature (both `async def _run_reconstruct_session(db, job) -> str`), different body, different architecture. The stash's version is a *simpler, earlier* implementation (orchestrator + direct WorldStore save); the current branch's version is a *more complete, tested, graded* implementation (stage segments + manifest grading + service-layer commit + COLMAP-session lifecycle + comprehensive test suite in `test_application_api.py`, `test_job_lease_recovery.py`, `test_job_failure_injection.py`, `test_evidence_world_job.py`).

**Verdict:** Do NOT port the stash's `_run_reconstruct_session`. The current branch's implementation is superior and tested. The stash's version is a dead-end predecessor, not a complementary addition.

### 3.2 `routes_worlds.py` 4 new routes — SUPERSEDED

Stash adds: `GET /api/worlds/{id}/worldir`, `/cameras`, `/points`, `/diff` + `_load_worldir_from_store`.

Current branch already has: worldir/points/cameras routes + a full commit endpoint with CAS validation, version integrity, duplicate detection, four-way commit-storm tests, space-graph, correction-commits (5 architectural correction types), stale-parent rejection, failed-WorldStore-write leave-HEAD-intact, failed-DB-write rollback+quarantine, version-duplicate convergence, concurrent-commits-no-corruption. See `tests/test_application_api.py` (commit tests) + `tests/test_adoption_transaction.py`.

**Verdict:** The stash's routes are a subset of what the current branch already provides, and less complete (the stash's commit route is a basic `setattr` + `save_version` with no CAS, no validation, no duplicate detection). Do NOT port — current branch already has more.

### 3.3 `storage.py` `verify_artifact` — CONFLICTS

Stash adds a local `verify_artifact(uri)` function in `apps/api/storage.py`.

Current branch's `apps/api/storage.py` already re-exports `verify_artifact` from `engine.pipeline.stages.content_store` (the engine layer). The `evidence_artifact` route already calls `resolve_artifact` (and the stash's intent — verify content hash before serving — is partially covered by the engine-layer `verify_artifact` being available; the current route does not call it on every serve, which may be a separate gap).

**Verdict:** Do NOT port the stash's local reimplementation — it would conflict with the current re-export. If the goal is "verify artifact integrity before serving," the right fix is to call the engine-layer `verify_artifact` in `evidence_artifact` (if that's desired), not to add a second local copy.

### 3.4 `main.py` graceful shutdown — SUBSET

Stash: `await asyncio.wait_for(task, timeout=30.0)` in `_shutdown`, with a warning log on timeout.

Current branch: `await asyncio.wait_for(asyncio.shield(task), timeout=10.0)` + warning log + `dispose_db()` with its own 10 s timeout + shield.

**Verdict:** The stash's version is a less-defensive subset (no shield, longer timeout, no DB-dispose timeout). The current branch's version is more defensive. Do NOT port — current is better. (If anything, the stash's 30 s timeout is worth comparing against the current 10 s — but that's a tuning decision for the backend owner, not a port.)

### 3.5 `reclaim_abandoned_jobs` (heartbeat-based) — ALTERNATIVE (potentially reusable)

Stash adds `reclaim_abandoned_jobs(db) -> int`:
- Reclaims jobs in `running` status where `job.worker_id != WORKER_ID` (different process) OR `job.heartbeat_at` is older than `HEARTBEAT_STALE_SECONDS` (default 60 s, configurable via `REALITY_JOB_HEARTBEAT_TIMEOUT`).
- Resets reclaimed jobs to `queued`, clears `worker_id`, increments `attempts`, clears `stage` + `error`.
- Called at the top of `worker_loop` on startup.

Current branch: `worker_loop` exists; startup in `main.py` calls `recover_orphaned_jobs(db)` from `reconstruction.proc` (proc-based: looks at dead workers, not heartbeats). There is NO heartbeat column or heartbeat-based reclamation in the current branch's `jobs.py`.

**Verdict:** This is NOT conflicting per se — it's an *alternative* reclamation strategy (heartbeat-based vs proc-based). The stash's approach could complement the current one: heartbeat-based reclamation catches the case where a worker dies mid-job without updating its worker-row state (proc-based recovery handles the "worker registration present but process dead" case; heartbeat-based handles "no heartbeat for N seconds"). The two are complementary, not competing.

**What's needed to port it (OpenCode's call):**
- The `Job` model would need a `heartbeat_at` column (the stash assumes it exists — check the current model). If absent, add it (migration).
- The job handlers would need to set `job.heartbeat_at = utcnow()` periodically (the stash's `_run_reconstruct_session` sets it at stage transitions; the current branch's stage segments already set `job.heartbeat_at` at stage transitions — so the heartbeat-update behavior partially EXISTS; what's missing is the reclamation side that consumes it).
- Decide: replace proc-based `recover_orphaned_jobs` with heartbeat-based, add heartbeat-based as a complement, or keep proc-based and add heartbeat as a second safety net. My recommendation: add heartbeat-based as a complement (it catches a different failure mode), but that's OpenCode's call.

**Bottom line for OpenCode:** The heartbeat reclamation is the one piece of the stash that is potentially reusable AND not conflicting. It requires a `heartbeat_at` column (verify the model) and a decision on how it relates to the existing proc-based `recover_orphaned_jobs`. The heartbeat-update behavior already exists in the stage segments; only the reclamation consumer is missing.

### 3.6 `apps/cli/main.py` incremental update — INDEPENDENT (separate concern)

Stash adds `cmd_incremental_update` (CLI) with `apply_reconstruction_update` / `apply_reconstruction_update_tiled`, GNSS anchor loading, initial-transform loading. This is a CLI concern, not the API backend. It may be independently valuable but is outside the API-backend recovery scope. Leave for OpenCode/CLI owner to evaluate separately.

### 3.7 `evidence/clocks.py`, `pyproject.toml`, `test_application_api.py` — OBSOLETE/TRIVIAL

- `evidence/clocks.py`: text update to the methods registry — trivial, not worth porting.
- `pyproject.toml`: comment update on the colmap optional dep — trivial, not worth porting (and the current `pyproject.toml` may already differ).
- `test_application_api.py`: the stash's test expansion predates the current test suite; the current `test_reconstruct_fails_with_honest_error_when_no_evidence` is more complete (full chain test). Not worth porting.

### 3.8 `.obsidian/*` — IGNORE (editor state)

---

## 4. Summary for OpenCode

| Stash piece | Verdict | Action for OpenCode |
|---|---|---|
| `apps/api/jobs.py` `_run_reconstruct_session` | CONFLICTING | Do NOT port. Current branch's stage-segmented + graded implementation is superior + tested. |
| `apps/api/jobs.py` `reclaim_abandoned_jobs` (heartbeat) | ALTERNATIVE (reusable) | Evaluate porting as a complement to proc-based `recover_orphaned_jobs`. Needs `heartbeat_at` column (verify model). Heartbeat-update behavior already exists in stage segments; only the reclamation consumer is missing. |
| `apps/api/main.py` graceful shutdown | SUBSET | Do NOT port. Current (shield + 10 s + DB-dispose timeout) is more defensive. |
| `apps/api/routes_misc.py` upload dedup | PORTED (done) | Already committed as `2e15cd44` with test. |
| `apps/api/routes_misc.py` verify_artifact import | CONFLICTING | Do NOT port local reimplementation; current re-exports from engine layer. |
| `apps/api/routes_worlds.py` 4 new routes | SUPERSEDED | Do NOT port. Current branch has more complete worldir/points/cameras + full commit endpoint + CAS + tests. |
| `apps/api/storage.py` verify_artifact | CONFLICTS | Do NOT port local reimplementation. |
| `apps/cli/main.py` incremental update | INDEPENDENT | Evaluate separately (CLI concern, outside API-backend scope). |
| `evidence/clocks.py`, `pyproject.toml`, `test_application_api.py` | OBSOLETE/TRIVIAL | Not worth porting. |
| `.obsidian/*` | IGNORE | Editor state. |

---

## 5. What NOT to do

- **Do NOT apply `stash@{6}`** to any branch. It conflicts with the current `apps/api/jobs.py` and `routes_worlds.py` and is superseded/conflicting on most files.
- **Do NOT drop `stash@{6}`.** It contains real work; the findings should be preserved (this document + recovery report §4.1).
- **Do NOT port the `_run_reconstruct_session` rewrite** under any circumstances — the current branch's implementation is the tested production path.

---

*End of handoff.*
