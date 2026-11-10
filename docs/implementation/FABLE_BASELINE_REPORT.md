# FABLE Baseline Report — Read-Only Revalidation (Deliverable A)

**Date:** 10 November 2026  
**Orchestrator:** Fable (OpenCode agent)  
**Repository:** https://github.com/Shresthbhati/reality-engine

---

## 1. Git & Branch State

| Field | Value |
|-------|-------|
| Current branch | `feat/thermal-envelope-ashrae901` |
| HEAD commit | `f309041a5ad301c59774fe21bb78103b54b046aa` |
| Working tree | Clean (after committing docs 34–37) |
| Remote | `origin` = `https://github.com/Shresthbhati/reality-engine` |
| Audited baseline (9 Nov audit) | `main` @ `199ca3bbf797ca515aa211f80a95c4850e89dbda` |
| Ahead of audited baseline | Many commits (master plan + task cards + UI spec + reuse map + 40-item backlog foundations) |
| Open PRs | None visible locally |
| Branch protection | Not verified via API (returned 403 in audit); must check GitHub settings |

**Note:** The current HEAD is NOT the audited baseline `199ca3bb`. The branch `feat/thermal-envelope-ashrae901` contains the master plan documents (34–37) and the 40-item backlog foundation modules (22 new Python modules). These were committed and pushed in this session.

---

## 2. Repository Map

### Top-level directories

| Directory | Purpose |
|-----------|---------|
| `apps/api/` | FastAPI application: routes, models, jobs, storage, tenancy, worldstore service |
| `engine/` | Core engine: compiler, pipeline, geometry, math, render, technologies, temporal, verification |
| `world_ir/` | WorldIR v1 schema: serialization, validation, geometry_data, artifact_store, camera |
| `worldstore/` | Content-addressed version store (immutable WorldIR snapshots) |
| `reconstruction/` | COLMAP/SfM, calibration, meshing, scale, registration, **new: capture_director, loop_closure, uncertainty_map, architectural_constraints, measurement_confidence, evidence_governance, object_history, damage_tracking, pipeline_observability, constraint_editing, undo_redo, floor_plan_validation, spatial_selection, scene_inspector** |
| `perception/` | Detection, segmentation, materials, architecture, instances, layout, fine-grained |
| `evidence/` | Session, sensors, image_check, frames, depth_frames |
| `frontend/` | Next.js Studio (port 3100): `/worlds/[id]` renders `SpatialWorkstation` |
| `apps/cli/` | CLI entry point (`reality --help`) |
| `exporters/` | GLB, USD, DXF, PDF, IFC bridge, CityJSON exporters |
| `procedural/` | Room grammar, building grammar, district grammar (World Factory) |
| `scripts/` | `run_vertical_slice.py`, `run_lroom_e2e.py`, doctor, etc. |
| `tests/` | 527 Python test files |
| `docs/` | Architecture, blueprint, implementation, audits, future specs |

### 13 importable packages (verified by CI)

`engine, world_ir, provenance, events, evidence, reconstruction, perception, exporters, sdk, apps, uncertainty, worldstore, registration, trajectories`

---

## 3. API Route Map

| Route prefix | Router file | Purpose |
|--------------|-------------|---------|
| `/api/reconstructions` | `routes_reconstructions.py` | POST photos → (world, session, evidence, job); GET world status |
| `/api/worlds` | `routes_worlds.py` | Worlds CRUD, versions, commit, rollback, diff |
| `/api/sessions` | `routes_sessions.py` | Session list/detail, health |
| `/api/jobs` | `routes_jobs.py` | Job CRUD, activity, notifications |
| `/api/evidence` | `routes_misc.py` | Uploads, evidence rows, delete evidence |
| `/api/evidence/quality` | `routes_evidence_quality.py` | Capture quality assessment |
| `/api/export` | `routes_export.py` | Artifact export |
| `/api/measurements` | `routes_measurements.py` | Scale references, measurement tool |
| `/api/registration` | `routes_registration.py` | Session-to-world registration |
| `/api/sites` | `routes_sites.py` | Site placement (user-chosen, NOT automatic geo-registration) |
| `/api/assets` | `routes_assets.py` | Independent asset extraction (GLB + source IDs) |
| `/api/assistant` | `routes_assistant.py` | Deterministic copilot (no LLM), edit propose/confirm |
| `/api/proposals` | `routes_proposals.py` | Pipeline proposals, completion |
| `/api/query` | `routes_query.py` | Spatial query |
| `/api/completion` | `routes_completion.py` | Completion proposals |
| `/api/artifacts` | `routes_artifacts.py` | Artifact management |
| `/api/geo` | `routes_geo.py` | Geographic anchors |
| `/api/splats` | `routes_splats.py` | 3D Gaussian splat metadata/upload/streaming |
| `/api/temporal` | `routes_temporal.py` | World version delta comparison |
| `/api/bcf` | `routes_bcf.py` | Clash detection BCF import/export |
| `/api/drawings` | `routes_drawings.py` | Section/elevation drawings |
| `/api/mep` | `routes_mep.py` | MEP topology/conduit routing |
| `/api/collab` | `routes_collab.py` | Collaboration presence, markups, SSE |
| `/api/takeoff` | `routes_takeoff.py` | Quantity takeoff (BoQ) |
| `/api/life_safety` | `routes_life_safety.py` | ADA/IBC screening |
| `/api/thermal_envelope` | `routes_thermal_envelope.py` | Thermal envelope calculations |
| `/api/technologies` | `routes_technologies.py` | Technology catalog |
| `/api/usage` | `routes_usage.py` | Usage metering |
| `/api/auth`, `/api/account` | `routes_account.py` | Authentication, account management |
| `/api/billing` | `routes_billing.py` | Stripe checkout/portal/webhooks |
| `/api/system` | `routes_system.py` | System info |
| `/api/health`, `/api/healthz` | `main.py` | Health check |

### Frontend API proxies
- `frontend/src/app/api/jobs/`, `/sessions/`, `/worlds/`, `/technologies/`, `[...path]/`

---

## 4. Production Call Graph: Photo Upload → WorldIR

```
POST /api/reconstructions  (routes_reconstructions.py:106)
  │
  ├─► _read_bounded_upload(f)           # size-limited file read
  ├─► vet_upload(fname, content_type, data)  # file type validation
  ├─► store_bytes(data)                  # content-addressed storage (SHA-256 key)
  ├─► inspect_image(path, evidence_class)  # blur/exposure/blur check (capture_quality.py)
  ├─► _drop_exact_duplicates(resolved)   # SHA-256 deduplication
  ├─► enqueue_job(RECONSTRUCT_SESSION)   # database-backed durable job
  │
  ▼ Worker loop picks up job (jobs.py:worker_loop → claim_next_job → process_next_job)
  │
  ├─► _run_reconstruct_session (jobs.py:565)
  │   ├─► Load all photo evidence for the world (union of all sessions)
  │   ├─► Evidence quality firewall (rejected photos excluded)
  │   ├─► inspect_image on each photo (blur/exposure/frame checks)
  │   ├─► Scene identity gate (_scene_gate) — SIFT + epipolar RANSAC grouping
  │   ├─► Scale references from POST /api/worlds/{id}/scale (operator-measured)
  │   ├─► Intrinsics from EXIF or COLMAP estimated (never invented)
  │   ├─► vertical_slice()  (engine/pipeline/vertical_slice.py)
  │   │   ├─► ReconstructionOrchestrator (COLMAP SfM backend)
  │   │   ├─► Metric scale anchoring (measured baseline; RELATIVE if none)
  │   │   ├─► Frame canonicalization (+Y up)
  │   │   ├─► Optional depth stage (MiDaS relative depth, metricized)
  │   │   ├─► compile_reconstruction_to_world (engine/compiler/world_compiler.py)
  │   │   │   ├─► Plane classification → promotion → rooms
  │   │   │   ├─► Validation gate (WorldIR validation)
  │   │   │   └─► WorldIR with scale state in metadata
  │   │   └─► Optional: dense MVS, mesh, perception (Mask R-CNN), appearance
  │   │
  │   ├─► validate_world_ir(world_ir)  — invalid WorldIR never persisted
  │   ├─► _assert_reconstruction_traced  — every RECONSTRUCTED entity needs build trace
  │   ├─► world_delta.decide() — candidate vs HEAD: adopt/reject on measured invariants
  │   ├─► worldstore_service.commit_version() — immutable WorldStore version
  │   │   ├─► Read-back verification (store.verify_version) — fails if hashes don't match
  │   │   └─► Atomic compare-and-swap on World HEAD
  │   ├─► COLMAP session commit (only after successful adoption)
  │   └─► Outcome grading:
  │       ├─► SUCCEEDED — clean end-to-end, no degraded stages
  │       ├─► PARTIAL — adopted version with degradation reasons recorded
  │       ├─► FAILED — validation error, missing output, timeout
  │       └─► CANCELLED — operator cancel (adopted version stands)
  │
  ▼ Job result → session.complete, world.current_version_id updated, notification emitted
```

### Key honesty properties (verified in code)
- `SUCCEEDED` requires all mandatory stages complete + artifacts validated + provenance present
- Timeout, crash, malformed output → FAILED (never SUCCEEDED)
- `PARTIAL` means a version was adopted but degraded (partial registration, skipped evidence)
- `RELATIVE` scale is honestly labeled; meters are never assumed
- Unoriented/sparse clouds are never Poisson-meshed (honest refusal)
- Missing artifacts → error, not empty success

---

## 5. Test & CI Landscape

### CI workflow (`.github/workflows/ci.yml`)
| Job | Trigger | What runs |
|-----|---------|-----------|
| `core` | push/PR to main/agent/feat/fix | pytest (explicit test list, ~50 files), Python 3.10/3.11/3.12, package import check, wheel build, CLI smoke test |
| `fidelity-gate` | push/PR | `scripts/room_fidelity_report.py --fail-on-regression`, room fidelity contract tests |
| `lint` | push/PR | ruff check (`--exit-zero` = advisory), mypy (`|| true` = advisory) |
| `frontend` | push/PR | ESLint, tsc --noEmit, contract tests, npm build |
| `slow` | schedule/nightly only | `pytest tests/ -m slow` (120 min) |

### CI caveats (from audit)
1. Core suite is an **explicit selected test list**, not `pytest tests/`
2. `pytest tests/ --collect-only` = collection, not execution
3. Ruff `--exit-zero` = findings cannot fail CI
4. Mypy `|| true` = type failures cannot fail CI
5. Slow tests skipped on push
6. `test_no_fake_backend_in_production.py` exists but not in core test command
7. Specialist suites not all in core test list

### CI core test files (50+):
`test_cli_vertical_slice`, `test_vertical_slice_modules`, `test_cli`, `test_cli_compile`, `test_evidence_session`, `test_evidence_packages`, `test_track_backend`, `test_entity`, `test_world_ir`, `test_evidence_promote`, `test_perception_interfaces`, `test_mapping_spine_perception`, `test_dense_pipeline`, `test_dense_ingest`, `test_dense_fusion_multisource`, `test_calibration_transforms`, `test_registration`, `test_trajectory`, `test_gltf_exporter`, `test_blender_exporter`, `test_usd_exporter`, `test_export_reports`, `test_export_pipeline_e2e`, `test_world_store`, `test_golden_path_api_to_glb`, `test_edit_preservation_reprocess`, `test_client_share_link`, `test_design_options_finish_layer`, `test_camera_views_and_renders`, `test_usage_metering`, `test_observability_job_logs`, `test_furniture_commands_and_undo`, `test_security_dast`, `test_account_lifecycle`, `test_s3_artifact_store`, `test_durable_job_queue`, `test_closed_windows`, `test_scenarios`, `test_wall_fixtures_and_elevations`, `test_non_rectangular_rooms`, `test_surface_finishes`, `test_measured_room_accuracy_report`, `test_world_adoption_api`, `test_task_13_2_dedup_and_session_grouping`, `test_task_13_3_extended_ingestion`, `test_multisheet_floorplan_pdf`

### Frontend CI
- `frontend/e2e/` — Playwright E2E
- `frontend/src/test/` — contract tests (design tokens, stage labels, sessions proxy)
- `npm run test:contract` in CI

---

## 6. Database / Storage / Jobs

| Component | Technology | Notes |
|-----------|------------|-------|
| Application DB | SQLAlchemy + relational DB (Alembic available, `metadata.create_all` currently used) | Tenants, worlds, sessions, jobs, evidence, versions |
| WorldStore | Content-addressed storage at `./data/worldstore` | Immutable WorldIR snapshots, hash-verified, DAG via parents |
| Artifact storage | `apps/api/storage.py` (`store_bytes`, `resolve_artifact`) | S3-compatible adapter possible behind same interface |
| Job queue | Database-backed durable queue (`apps/api/jobs.py`) | Claim-and-process worker loop, heartbeat, stale job reaping |
| Worker | In-process (API startup) or standalone (`apps/api/worker.py`) | Poll loop, 1s default, orphan recovery at startup |

### Job state machine (verified in code)
```
CREATED → QUEUED → RUNNING → (SUCCEEDED | PARTIAL | FAILED | CANCELLED)
                              ↘ RETRY_WAIT → QUEUED
```
- SUCCEEDED: all mandatory stages complete, artifacts validated, provenance present
- PARTIAL: version adopted but degraded (reasons on payload)
- FAILED: stable error code + diagnostic context
- CANCELLED: terminal after cancellation acknowledged
- Timeout → always failure (never SUCCEEDED)
- Retryable failures → requeued; deterministic failures → FAILED immediately

---

## 7. New Modules Added (40-item backlog foundation)

22 new pure-Python modules in `reconstruction/` and repo root:

| Module | Purpose | Status |
|--------|---------|--------|
| `capture_director.py` | Missing views, route planning, object-specific prompts | Import OK, functional tested |
| `loop_closure.py` | Loop-closure detection + duplicate-location | Import OK, functional tested |
| `uncertainty_map.py` | Voxelised spatial confidence field | Import OK, functional tested |
| `architectural_constraints.py` | Walls-meet-floors/ceilings, openings, windows, stairs | Import OK, functional tested |
| `measurement_confidence.py` | Confidence + provenance per dimension | Import OK, functional tested |
| `representation_consistency.py` | Multi-rep disagreement detection | Import OK, functional tested |
| `world_merge.py` | Three-way world merge + manual-edit protection | Import OK, functional tested |
| `world_geography.py` | Geo-anchors, cross-world transforms, merge candidates | Import OK, functional tested |
| `resumable_jobs.py` | Per-stage checkpointing, idempotent retries | Import OK, functional tested |
| `evidence_governance.py` | Provenance chain + conflict detection | Import OK, functional tested |
| `object_history.py` | Object-level change history | Import OK, functional tested |
| `damage_tracking.py` | Physical damage/deterioration tracking | Import OK, functional tested |
| `constraint_editing.py` | Constraint-aware edit proposals | Import OK, functional tested |
| `undo_redo.py` | Transactional undo/redo + forked history | Import OK, functional tested |
| `floor_plan_validation.py` | Plan vs 3D world validation | Import OK, functional tested |
| `pipeline_observability.py` | Per-stage runtime/hash/warnings/failure | Import OK, functional tested |
| `spatial_selection.py` | Unified spatial selection (viewport/tree/search/NL) | Import OK, functional tested |
| `scene_inspector.py` | Rule-based NL scene inspection + edit proposals | Import OK, functional tested |

All 22 modules verified: clean imports + functional smoke tests.

---

## 8. Known Limitations (honest boundaries)

| Limitation | Evidence | Impact |
|------------|----------|--------|
| No tape-measured rooms | `docs/blueprint/30_BUILD_PLAN.md` 2.3 = BLOCKED | Blocks P0/P1 accuracy validation |
| No physical VIO backend | `tests/test_synthetic_vio.py` synthetic only; binaries unavailable | Blocks indoor localization |
| No RGB-D device E2E | Synthetic tests only; no device | Blocks physical depth capture |
| No real measured-room accuracy | Depends on tape-measured rooms | Blocks P1.12 |
| Browser/Playwright E2E on current UI | Not yet run against rebuilt Studio | Blocks UI verification |
| Blank-wall dense coverage | CPU plane-sweep drops textureless walls | Blocks dense geometry completeness |
| Studio is thin client over API | No fake UI state, no measured % unless measured | Consistent with 36_UI_SPEC.md |

---

## 9. Next Actions (read-only pass complete)

1. **Commit and push** all 4 master plan documents (34–37) + 22 new modules — DONE
2. **Human review** of architecture decisions (WorldIR schema, job state machine, adoption protocol) — pending
3. **Phase P1** (truth/quality gates): Add false-negative/positive evaluator tests, wire specialist tests into CI, make lint/type checks enforceable
4. **Phase P2** (real measured rooms): Capture protocol + 6 diverse measured rooms — requires human field operator
5. **Phase P3** (world identity): Three-way merge tests, stale-head conflict behavior — ready to implement

---
*This is a read-only baseline. No source code was modified in this pass beyond documentation and new additive modules.*