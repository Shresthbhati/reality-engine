# Release Gate Plan: Real Photos -> Persisted World -> Correction -> V2 Diff

**Executor:** Claude Sonnet (single agent, sequential). Optional parallel split in section 8.
**Written:** 2026-10-16. **Base:** `main` @ 3e452a6.
**Goal:** one end-to-end workflow that a user can test and repeat. Do not add features. Do not polish the UI. Interior semantics and extra automation come only after this gate passes.

---

## 0. Rules for the executor (read first)

1. **Evidence only.** A step is done only when a command you ran shows it passing. Never write "should work". Never report a percentage complete.
2. **No fake success.** A missing or malformed reconstruction, an empty world, or a UI showing data that is not persisted counts as a FAILURE. Never make one look like success.
3. **Smallest diff.** Fix root causes in shared functions (grep every caller first). Do not refactor unrelated code.
4. **One branch per phase:** `claude/gate-p<N>-<slug>`. One reviewable commit per fix. Never push or merge without the user's go-ahead.
5. **Stop and ask** only when a product or architecture decision is needed (section 51 of CLAUDE.md: record BLOCKED with OPTIONS / TRADEOFFS / RECOMMENDATION). Decide everything else yourself.
6. **Ledger discipline:** after each phase, update `.agent/EXECUTION_STATE.md` and `.agent/TASKS.yaml` with the exact commands run and their results.
7. Every phase ends with the **Phase Report** (section 7). If the report is missing, the phase is not accepted.

### Known facts (verified 2026-10-16, re-check before relying on them)

| Thing | Where |
|---|---|
| API entrypoint | `uvicorn apps.api.main:app --port 8100` (`QUICKSTART.md:36`); storage root env var `STORAGE_ROOT` (`apps/api/storage.py:19`) |
| Upload | `POST /uploads` (`apps/api/routes_misc.py`) |
| Reconstruct | `POST /reconstructions`, `POST /sessions/{id}/reconstruct`; job status `GET /jobs/{id}` |
| World read | `GET /worlds/{id}`, `/worldir`, `/versions`, `/points`, `/cameras`, `/status` |
| Correction | `POST /worlds/{id}/commit` (`apps/api/routes_worlds.py:745`; documented contract: 422/413/409/404, never moves HEAD on failure, an identical retry returns `duplicate: true`) |
| Diff / rollback | `GET /worlds/{id}/diff`, `POST /worlds/{id}/rollback` |
| Real dataset | `datasets/south_building` (128 real photos, `MANIFEST.json`) |
| Existing golden journeys | `tests/integration/test_progressive_product_journey.py` (A-H), `tests/integration/test_reliability_journey.py` |
| Frontend | `frontend/` (Next 16): `npm run build`, `npm run lint`, `npm run test:contract`; Studio runs on :3000 |

---

## Phase 1: Fix the `world.geometries` regression (highest priority)

**Why first:** every later step reads `world.geometries` (points route `routes_worlds.py:398`, exporters, `frontend/src/lib/viewport/three-scene.ts`, `WorldNavPanel.tsx`, `AdaptiveInspector.tsx`).

The repo does not document the regression's symptom. **Reproduce it before you change anything.**

1. Run these and collect every failure that mentions `geometries`:
   ```bash
   python -m pytest tests/test_canonical_interior.py tests/integration/test_golden_world_flow.py tests/integration/test_contract_compatibility.py tests/integration/test_reconstruction_to_worldos_e2e.py -q -x
   python -m pytest tests -q -m "not slow" -k geometr
   ```
2. If nothing fails, find the regression by reading. Grep the writers and readers of `.geometries` (list: `grep -rln "\.geometries" --include=*.py --include=*.ts --include=*.tsx . | grep -v node_modules`). Check these invariants:
   - every `entity.geometry_ref` resolves to a key in `world.geometries`;
   - the WorldStore save/load round-trip keeps `geometries` (same count, same digests);
   - incremental updates keep untouched geometries by identity (`is`) and never drop them;
   - `GET /worlds/{id}/worldir` and `/points` return geometries for a reconstructed world.
3. If you still can't reproduce it: **stop and ask the user for the symptom** (traceback, command, world id). Do not guess-fix.
4. Write a **minimal failing regression test** first, in `tests/test_world_geometries_regression.py`. It must fail on the current code, then fix the root cause, then show it passing.
5. Gate: the new test passes, and the non-slow suite shows no new failures (compare against the baseline you ran in step 1).

**Done when:** you can name the root cause in one sentence, a regression test exists and passes, and the non-slow suite matches the baseline.

---

## Phase 2: Reconstruction output validation (prove it is real and measurable)

Goal: a reconstruction only counts as success when its output is valid. Finishing the process is not enough.

1. Find where a reconstruction is turned into a success (job terminal-state code in `apps/api/jobs.py`, `apps/api/reconstruction_state.py`, `evidence/promote_reconstruction.py`, `reconstruction/diagnostics.py`).
2. Check that an output validator rejects each of the following. Add a test for any check that is missing:
   - zero registered cameras, or fewer than 2 registered cameras;
   - an empty or missing sparse model or point cloud;
   - NaN or Inf in camera poses or points;
   - a world with entities whose geometry refs dangle.
   On any of these, the job must end `FAILED` (or an honestly labelled degraded state) with a useful error message, and **no new WorldStore version** may be created.
3. Run the real controlled experiment with the SIX window (`[15:21]`) from the product-journey test:
   ```bash
   python -m pytest tests/integration/test_progressive_product_journey.py -q -k "test_B or test_D" -m slow
   ```
   Record these measured numbers: registered images out of submitted, point count, mean reprojection error, and wall time. Put them in the Phase Report.
4. Gate: malformed-output tests pass, and test_B and test_D pass with measured values recorded.

---

## Phase 3: Lifecycle, persistence and API contract baseline (read and test first, change little)

Goal: establish what is actually true today. Do not change architecture. Any change to WorldStore or WorldIR persistence needs a regression test and must be kept small.

1. Run and record:
   ```bash
   python -m pytest tests/integration/test_reliability_journey.py -q
   python -m pytest tests/integration/test_progressive_product_journey.py -q -k "test_E or test_F" -m slow
   cd frontend && npm run test:contract && npx tsc --noEmit
   ```
2. Write a contract table into this file's Appendix A for every route in "Known facts": the request shape, the success shape, and each error code. Check it against the frontend callers (`frontend/src/lib/**`) and flag every mismatch (a field the UI reads that the API doesn't send, or the reverse).
3. Security and lifecycle checks (fix only real defects, each with a test):
   - Upload: limits on size, type and count; filenames can't escape `STORAGE_ROOT` (path traversal); original bytes are stored unchanged (sha256 matches the input).
   - Input metadata (EXIF, original filename) is preserved and retrievable via `GET /evidence/{id}`.
   - Job cancel moves the job to a terminal state and creates no version.
4. Gate: every listed test passes, Appendix A is filled in, and each mismatch is either fixed with a test or listed as a handoff.

---

## Phase 4: The release-gate test (the definition of "prototype complete")

Create **one** test: `tests/integration/test_release_gate.py` (marked `slow`). It drives the real API in-process, the same way `test_progressive_product_journey.py` does, and asserts **all eight acceptance criteria in order**:

| # | Step | Assertion |
|---|---|---|
| 1 | Upload SIX real photos | each evidence sha256 equals the input file's sha256; EXIF and filename are retrievable |
| 2 | Reconstruct | the job reaches a terminal `SUCCEEDED`; registered cameras >= 4; point count > 0; poses are finite |
| 3 | Canonical WorldIR | `validate()` passes; every geometry ref resolves; every entity carries uncertainty and provenance |
| 4 | Persistent V1 | **restart**: rebuild the app or a new process against the same `STORAGE_ROOT`; `GET /worlds/{id}` returns the same version id and WorldIR digest |
| 5 | Studio data source | `/worldir`, `/points` and `/cameras` return exactly the persisted data (digest match). No synthetic or placeholder fields |
| 6 | Correction | `POST /commit` with a valid change creates V2. An invalid change gets 422 and HEAD stays put. A stale parent gets 409 |
| 7 | V2 and diff | V2's parent is V1; `GET /diff?from=V1&to=V2` lists **exactly** the corrected entity/field and nothing else |
| 8 | Failure path | the HARD window (`[0:6]`) ends FAILED or degraded, has a useful error message, creates **no** successful world version, and leaves the HEAD of an existing world untouched |

Plus a **reproducibility** check: run steps 1-2 twice from scratch with the same input and config. Registered-camera counts must match exactly, and point counts must be within the documented tolerance (if needed, set the tolerance after measuring Phase 2's run-to-run variance and write it down here).

Write `docs/implementation/RELEASE_GATE_RUNBOOK.md` with the exact commands, the environment (Python version, COLMAP version, `pip freeze` hash), and the expected output, so a human can rerun the gate.

**Gate:** `python -m pytest tests/integration/test_release_gate.py -q -m slow` passes **twice in a row** on a clean `STORAGE_ROOT`.

---

## Phase 5: Integrated validation (the KiloCode role, done by you)

1. Merge the phase branches into `claude/gate-integration` locally (no push).
2. Run everything:
   ```bash
   python -m pytest tests -q -m "not slow"
   python -m pytest tests/integration -q -m slow
   cd frontend && npx tsc --noEmit && npm run build && npm run test:contract
   ```
3. Compare against the 2026-10-03 baseline in `.agent/EXECUTION_STATE.md` (2776 passed, 0 failed, 28 skipped non-slow). Any new failure blocks the gate: fix it or report it as a regression.

---

## Phase 6: Reality Studio wired to real persisted data (the Antigravity role)

Goal: the UI shows only persisted data. No optimistic or fabricated state.

1. Start the API (`STORAGE_ROOT=<tmp> uvicorn apps.api.main:app --port 8100`) and the Studio (`cd frontend && npm run dev`). Use the browser preview tools.
2. Walk through steps 1-8 of the gate **in the UI**: upload photos, watch the job, open the world, inspect the points and cameras, make a correction, see V2 and the diff, then trigger a failing upload.
3. For each screen, check against the network responses (`read_network_requests`) that every displayed number or entity comes from an API response. Grep the components you touch for hard-coded or mock data, and for optimistic updates that show a change before the server confirms it. Remove them. While waiting, show an honest pending state.
4. Restart the API, reload the Studio, and confirm the world, the versions and the diff all reappear.
5. Gate: screenshots of each step plus a network-trace check. `npm run build` and `npx tsc --noEmit` pass. Lint shows no **new** errors (the 10 existing ones are known).

---

## 7. Phase Report (required at the end of every phase)

```
### Phase N report
Changed:      <file:line -- behavior change>, one line each
Verified:     <exact command> -> <actual result line, e.g. "14 passed, 0 failed in 312s">
Unverified:   <anything not run or not confirmed, and why>
Regressions:  <existing behavior affected, or "none observed vs baseline <cmd>">
Handoffs:     <interfaces or assumptions a later phase must accommodate>
Commit:       <branch> @ <sha>, "<message>"
```

Release criterion = the Phase 4 test passes twice, plus a clean Phase 5 run, plus the Phase 6 walkthrough evidence. Nothing else counts.

---

## 8. Optional parallel split (only if the user asks for multiple sessions)

Use one designated **integration owner** (the main session). Agents with no shared files can run at the same time:

| Track | Phases | Owns files | Must coordinate on |
|---|---|---|---|
| A (Claude/Sonnet) | 1, then 4 | `world_ir/`, `worldstore/`, `engine/compiler/`, new tests | none (owner of WorldIR and WorldStore) |
| B (FreeBuff role) | 2 | `reconstruction/`, `evidence/promote_reconstruction.py`, `apps/api/jobs.py` | A, before touching persistence |
| C (OpenCode + Cline role) | 3 | `apps/api/routes_misc.py`, `validation.py`, Appendix A, `frontend/tests/contract` | A for any WorldStore change; read-only on `routes_worlds.py` |
| D (KiloCode role) | 5 | none (runs tests only) | starts after A, B and C have merged into integration |
| E (Antigravity role) | 6 | `frontend/src/**` | starts after D is green |

Each track works in its own worktree and branch. Only the integration owner merges.

---

## Appendix A: API contract baseline (filled in during Phase 3, 2026-10-16)

Shapes are observed from a live in-process run (`REALITY_TEST_BACKEND` seam), error codes from the handlers. All routes are under `/api`.

| Route | Request | 2xx shape | Error codes | Frontend caller | Mismatch |
|---|---|---|---|---|---|
| `POST /uploads` | multipart `file`; `?session_id=` | 201 `{upload_id, evidence_id, job_id, checksum}` (checksum = sha256 of the original bytes) | 400 empty; 413 over `UPLOAD_MAX_BYTES` (default 1 GiB) | `lib/api` `UploadResult` | none |
| `POST /reconstructions` | multipart `files[]`; form `world_id?, name?, evidence_class?, latitude?, longitude?` | 202 `{world_id, session_id, job_id, evidence[], rejected[], coalesced, created_world}` | 422 no files / bad class / lat-lon pair / no usable photo (body has `rejected[]`); 413 >300 files or oversize; 404 unknown `world_id` | `lib/api/reconstructions.ts` | none found |
| `POST /sessions/{id}/reconstruct` | none | 200 `{job_id, type, status}` | 404; 409 already in progress | `lib/api/sessions.ts` | none |
| `GET /jobs/{id}` | none | `{id, type, status, stage, attempts, error, worker_id, cancel_requested, entity_type, entity_id, payload{world_id, version_id, registration_status, level, model_state, outcome, degraded[], validated, adopted, verdict, changes[], points, cameras_registered, skipped_evidence[]}}` | 404 | `JobDto` | none (`created_at/started_at/completed_at` optional in UI) |
| `POST /jobs/{id}/cancel` | none | 200/202 | 404; 409 already terminal | jobs client | none |
| `GET /worlds/{id}` | none | `{id, name, description, status, latitude, longitude, current_version_id, session_count, created_at}` | 404 | `WorldDto` | none |
| `GET /worlds/{id}/status` | none | `{world_id, state, state_label, in_progress, job, has_model, model{...}, last_run, evidence[], evidence_summary, guidance[], versions[], failure}` | 404 | `lib/api/reconstructions.ts` | none found |
| `GET /worlds/{id}/versions` | none | `{items:[{id, world_id, parent_version_id, artifact_uri, artifact_hash, source_session_ids, changed_entity_ids, created_at, is_current}]}` newest first | 404 | `WorldVersionDto` | none |
| `GET /worlds/{id}/worldir` | `?version=` | full `WorldIR.to_dict()` (adds `environment`, `physics`; entities carry `custom_properties`, `relationships`, `uncertainty`) | 404; 409 artifact unreadable | `WorldIRDto`, `types/worldir.ts` | UI DTO omits `environment`/`physics` (UI never reads them: harmless) |
| `GET /worlds/{id}/points` | `?version=` | PLY stream (`application/octet-stream`) | 404 no renderable points (honest empty); 409 | viewport | none |
| `GET /worlds/{id}/cameras` | `?version=` | `{frame, rotation_convention, image_size, version_id, cameras:[{evidence_id, position_m[3], rotation_wxyz[4]}]}` | 404 | `types/worldir.ts` `CamerasPayload` | **FIXED**: `lib/api/types.ts` held a second `CamerasPayload` (`entity_id`, `{x,y,z}`) that the API never sent; `getWorldCameras` now returns the real type and the wrong DTO is deleted |
| `POST /worlds/{id}/commit` | `{entity_id, changes{}, parent_version_id?, commit_message?}` | 200 `{version_id, world_id, entity_id, changed_fields[], duplicate}` | 404 world/entity/version; 409 stale parent or HEAD moved; 413/422 invalid change (confidence out of [0,1], non-finite, bad relationship, unknown id); 503 DB failure. HEAD never moves on failure | `commitWorldCorrection` | **FIXED**: `CommitResponse` omitted `duplicate`. Handoff (Phase 6): `SpatialWorkstation` still shows a duplicate retry like a new version; surface `duplicate` honestly |
| `GET /worlds/{id}/diff` | **`?base=<version>&head=<version>`** (not from/to); default head = HEAD, base = its parent | `{from_world_id, to_world_id, summary{entities_*, geometries_*}, categories{...}, entity_diffs[{entity_id, kind, changes[{field, old, new}]}], geometry_diffs[]}` | 404 no HEAD / no base / version missing | `getWorldDiff` | UI `WorldDiffDto` omits `geometries_*` and `categories` (extra fields only). **Phase 4 must call with `base`/`head`** |
| `POST /worlds/{id}/rollback` | `{version_id}` (required) | 200 `{world_id, version_id, previous_head, note}`: moves HEAD back, history unchanged, no new version | 404; 409 reconstruction in progress or concurrent change; 422 unknown version | rollback caller | none |
| `GET /evidence/{id}` | none | `{id, name, type, session_id, processing_state, mime_type, size, checksum, created_at, processed_at, metadata{verified, image_facts{width, height, evidence_class, exif{make, model, ...}, gps, geometry_eligible}, registration{...}}, location}` | 404 | `EvidenceDto` | none |

### Phase 3 lifecycle and security results
- Upload bytes are stored content-addressed (`sha256://`), unchanged: `tests/test_upload_integrity_contract.py` (5 passed). Filenames are sanitised to a basename and never form a path, so traversal cannot leave `STORAGE_ROOT`. Size limit answers 413, file-count limit (300) answers 413, EXIF make/model and the original filename are retrievable.
- Not enforced (recorded, not changed): `/uploads` has no content-type allowlist (unknown types are classified `dataset`; only `/reconstructions` rejects non-photo input) and no per-session count limit. Decision for the owner if untrusted users are expected.
- Job cancel: existing tests (`test_cancel_queued_job`, `test_cancelled_run_adopts_nothing`, `test_repeat_cancel_is_idempotent`, `test_cancel_terminal_job_409`) cover terminal state and no version.


## Appendix B: Reconstruction job and world state transitions (as implemented and exercised by the gate)

Source: `apps/api/jobs.py` (grading), `apps/api/reconstruction_state.py` (`derive_state`), `engine/pipeline/progressive.py`
(the level ladder), `engine/pipeline/world_delta.py` (`decide`).

**Job status:** `queued` -> `running` (stages `resolving_evidence`, `analyzing_evidence`, `building_rough_model` | `reconstructing` | `refining`,
`committing_version`) -> exactly one of `succeeded | partial | failed | cancelled`.

| Outcome | Exact condition | Version created? | World state shown |
|---|---|---|---|
| `succeeded` | Clean end to end: registration `success`, level >= 2, no skipped evidence, no degraded optional stage, no validation warning, no "changed with uncertainty" note. Also: an identical rerun (`unchanged`) | yes (or none for `unchanged`; HEAD stands) | `READY_TO_INSPECT` |
| `partial`, adopted | A real valid version with recorded degradation: level <= 1 (a rough model is partial by construction), partial registration, skipped evidence, a degraded stage, or an uncertainty note | yes, labelled (`model_state` ROUGH/PARTIAL, `degraded[]` lists why) | `PARTIALLY_COMPLETE` |
| `partial`, not adopted | The candidate was valid but `world_delta.decide` returned REJECT against HEAD (photos lost with no net gain; level fell with no photo gained; structure not reproduced). `payload.adopted=false`, `kept_version_id` names HEAD | no; HEAD is untouched | `PARTIALLY_COMPLETE` (the Studio now says "not adopted") |
| `failed` | Insufficient evidence, no usable level, WorldIR failed validation, untraced entity, read-back verification failed, timeout, crash | no | `FAILED` / `NEEDS_MORE_EVIDENCE` |
| `cancelled` | Operator cancel before adoption. Late cancel (after adoption): the version stands, `adopted_before_cancel` is recorded, the job still grades `cancelled` | only in the late case | `CANCELLED` |

**Malformed multi-view output** (fewer than 2 registered cameras, empty cloud, NaN/Inf pose or point): `vertical_slice` raises, the ladder records
`L2 failed` with the reason, and the next level is tried. On a fresh world the single-image fallback yields an adopted, labelled `partial`/ROUGH
version (never `succeeded`). On a world that already has a HEAD the fallback is arbitrated against it and REJECTed (verified by the gate).

**Precondition for arbitration (fixed in this release):** the HEAD report is the nearest ancestor's report, because a correction commit has none of its own.

**Cameras (fixed in this release):** `/cameras` serves only photographs the version recorded as registered.

**Rollback** moves HEAD to an earlier version; it creates no version and rewrites no history.


## Appendix C: Results (2026-10-16). Final tree: branch `claude/gate-p4-release-gate`

Environment: Windows 11, Python 3.14.7, COLMAP (CUDA build) at `C:\Users\shres\tools\colmap-extracted\bin\colmap`, MiDaS from the local torch hub cache,
dataset `datasets/south_building` (128 real 1024px photographs; gate uses indices `[15:21]` SIX and `[0:6]` HARD plus one more). `httpx 0.27.2` via PYTHONPATH
because the installed httpx 0.28.1 breaks starlette 0.35.1's TestClient (the repo now pins `httpx>=0.27,<0.28`).

| Gate / suite | Command | Result |
|---|---|---|
| Release gate run 1 | `python -m pytest -q tests/integration/test_release_gate.py` (commit 994af9c, clean tree) | **3 passed**, exit 0, 8m05s |
| Release gate run 2 (consecutive, alone) | same command and commit | **3 passed**, exit 0, 7m54s |
| Earlier gate attempts (not counted) | commit c0f3837 / b2074cb | FAILED: found the defects listed below; fixed before the counted runs |
| Backend, non-slow, full | `python -m pytest tests -q -m "not slow"` | **2976 passed, 28 skipped, 5 xfailed, 0 failed**, exit 0, 39m39s (baseline 2776 / 0 / 28) |
| Slow journeys | `pytest tests/integration/test_progressive_product_journey.py tests/integration/test_reliability_journey.py -m slow` | 27 passed, 1 failed (test_H: stale rejection-message assertion), 1h57m; test_H then re-run alone: **1 passed**, exit 0 |
| Frontend | `npx tsc --noEmit` / `npm run test:contract` / `npm run build` / `npm run lint` | exit 0 / 17 of 17 / exit 0 / 10 errors + 40 warnings (the known baseline) |
| System runtime proof | `tests/integration/test_system_runtime_proof.py` | **NOT RUN**: skips with an explicit reason; its gitignored fixtures (`datasets/real_room_capture*`, `datasets/room_capture/pipeline_out`) are absent. Regenerate with `scripts/generate_room_dataset.py` |

Real reconstruction (SIX window, real COLMAP): 6 of 6 photographs registered, 1,132 sparse points, mean reprojection error 0.295 px, 101 s wall time
(clean machine); the served `/points` cloud held 85,782 points including dense. Same input twice: identical registered count, point counts within the 5% tolerance
(measured spread 0.9%).

### Defects found by the gate or its audits, and fixed (each with a regression test that failed first)
1. Correction commits carried no reconstruction report, so a degraded fallback was accepted over a corrected HEAD and `/status` reported 0 registered photos. `worldstore_service.nearest_report`.
2. Malformed backend output (1 camera, NaN/Inf pose or point) compiled into a world. `reconstruction.validation.validate_reconstruction_output` in `vertical_slice`.
3. Correction commits accepted a NaN/garbage `transform` (stored `NaN` is invalid JSON for the browser) and could erase the pipeline-owned `reconstruction` trace.
4. `/cameras` served cameras for photographs that never registered (level-0 fusion display layout).
5. Uploads were not vetted by content or capped per session; artifacts had no `nosniff`.
6. Frontend: wrong `/cameras` DTO; `CommitResponse` lacked `duplicate`; the Studio showed a duplicate retry as a new version; provenance failures rendered as "no evidence"; list proxies turned a broken 200 body into an empty list; the job poll swallowed a lost API forever; a rejected rebuild was announced as adopted.
7. Environment: httpx 0.28 broke every TestClient test.

### Remaining risks and blockers (not fixed)
- **No authentication, authorization or rate limiting anywhere in the API** (CORS defaults to `localhost:3000`). Blocker for any multi-user or public deployment.
- The `world.geometries` regression the user reported **could not be reproduced**; no symptom was supplied. Entity-to-geometry references, persistence round-trips, restart and the viewport's bounds contract all verified on real data.
- The malformed-output fallback over many photos runs single-image depth (MiDaS) on CPU for 10 to 25 minutes before the rebuild is rejected.
- `/uploads` per-session cap defaults to 2000 files and unknown types are refused unless `UPLOAD_ALLOW_ANY_TYPE=1`; limits are conservative defaults, not tuned.
- `frontend/tests/contract/sessions-route.test.mts` passes without running its tests; the real assertions live in `tests/test_frontend_sessions_proxy_contract.py`.
- `AdaptiveInspector.tsx` was changed after the counted gate runs (frontend only; verified by `tsc`, build, lint and contract tests, not by a browser walkthrough).
- Phase 6 (a real browser walkthrough of the Studio) was **not performed**.
- Git-ignored test byproducts exist in the worktree (`data/`, `MagicMock/`, `.ruff_cache/`); not removed.
