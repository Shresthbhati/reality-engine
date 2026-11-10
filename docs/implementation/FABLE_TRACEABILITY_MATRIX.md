# FABLE Requirement Traceability Matrix (Deliverable B)

**Date:** 10 November 2026  
**Status vocabulary:** MISSING / FOUNDATION / PARTIAL / UNWIRED / IMPLEMENTED-UNVERIFIED / CODE-CONTRACT-VERIFIED / FIELD-VALIDATED / DEFERRED / DOCUMENTATION-DEBT

**Honesty rule:** A status may only advance when the listed evidence exists. Code presence ≠ verification.

---

## Core Product Journey (P0–P1 first-customer workflow)

| # | Requirement | Status | Evidence path | Test evidence | Product wiring | Real-world validation | Risk | Next action |
|---|-------------|--------|---------------|---------------|----------------|----------------------|------|-------------|
| R1 | Photo upload → evidence artifact | CODE-CONTRACT-VERIFIED | `apps/api/routes_reconstructions.py:106`, `apps/api/storage.py` | `tests/test_cli_vertical_slice.py`, `tests/test_cli_compile.py` | POST /api/reconstructions → store_bytes → inspect_image | Synthetic only; no real photo set | Blurry/overexposed photos pass validation | Add real-image quality fixtures |
| R2 | Session creation on first capture | CODE-CONTRACT-VERIFIED | `routes_reconstructions.py`, `apps/api/models.py` | `tests/test_evidence_session.py` | World + Session created together on first evidence | Not tested on real captures | Session/world boundary confusion | Add session lifecycle e2e |
| R3 | COLMAP SfM reconstruction | IMPLEMENTED-UNVERIFIED | `reconstruction/backend/colmap_backend.py`, `vertical_slice.py` | `tests/test_registration.py` | Real COLMAP binary on PATH | Not tested on real room captures | Low-overlap/blur/rotation robustness | Real-capture benchmark |
| R4 | Metric scale anchoring | PARTIAL | `reconstruction/scale/base.py`, `routes_measurements.py` | `tests/test_scale_suggestion.py` | Operator-measured baseline → METRIC state | No tape-measured rooms exist | Scale declared RELATIVE honestly | Capture 6 measured rooms (P2.2) |
| R5 | Depth → dense points | PARTIAL | `reconstruction/depth_to_points.py`, `vertical_slice.py` | `tests/test_dense_pipeline.py` | MiDaS relative depth, metricized against sparse cloud | Synthetic only | Textureless/reflective surfaces | Depth quality benchmark |
| R6 | WorldIR compile (planes→rooms→validation) | CODE-CONTRACT-VERIFIED | `engine/compiler/world_compiler.py`, `world_ir/validation.py` | `tests/test_world_ir.py`, `test_validator_invariants` | Compile gate rejects invalid WorldIR | Synthetic worlds only | Incremental refinement edit preservation | Real-room compile test |
| R7 | WorldStore immutable version | CODE-CONTRACT-VERIFIED | `worldstore/store.py`, `apps/api/worldstore_service.py` | `tests/test_world_store.py` | Content-addressed, hash-verified, DAG lineage | Not tested under crash | Orphan versions on crash | Crash recovery drill |
| R8 | Adoption protocol (intent→version→read-back→DB) | CODE-CONTRACT-VERIFIED | `apps/api/worldstore_service.py` | `tests/test_world_adoption_api.py` | Compare-and-swap on HEAD, orphan quarantine | Not tested under concurrency | Stale HEAD conflicts | Concurrency test |
| R9 | GLB/USD/OBJ export | CODE-CONTRACT-VERIFIED | `exporters/`, `routes_export.py` | `tests/test_gltf_exporter.py`, `test_usd_exporter.py` | glTF validator passes | Not opened in Blender/Revit | External CAD compatibility | External import validation (P1.14) |
| R10 | Floor plan SVG/DXF | CODE-CONTRACT-VERIFIED | `exporters/floorplan.py` | `tests/test_floorplan_export.py` | Room report PDF via sdk/reality.py | Not validated against real tape measure | No real-room plan checked | Measured-room comparison |
| R11 | Studio UI (SpatialWorkstation) | CODE-CONTRACT-VERIFIED | `frontend/src/components/workspace/SpatialWorkstation.tsx` | `tests/test_reality_studio_primary_journey.py` | tsc 0, next build 0, contract tests | Not browser-tested on current UI | Specialist panels expanding faster than core | Playwright E2E on current UI |

---

## P2 — Real Measured Interior Dataset (BLOCKED on human field operator)

| # | Requirement | Status | Evidence path | Test evidence | Product wiring | Real-world validation | Risk | Next action |
|---|-------------|--------|---------------|---------------|----------------|----------------------|------|-------------|
| R12 | Dataset protocol (consent, privacy, device metadata) | MISSING | — | — | — | None | No dataset exists | Write protocol (P2.1) |
| R13 | 6 diverse measured rooms captured | MISSING | `datasets/measured/<room>/` (BLOCKED) | — | None | None | Requires human field work | Capture protocol + execution |
| R14 | Independent ground truth (laser/tape) | MISSING | — | — | — | None | Annotator agreement needed | Define measurement protocol |
| R15 | Train/dev/blind-test split | MISSING | — | — | — | None | Data leakage risk | Split after capture |
| R16 | Ordinary phone photos through real API path | MISSING | — | — | — | None | Synthetic ≠ real | Run P2.5 after capture |
| R17 | Per-class precision/recall, dimensional error | MISSING | `scripts/room_fidelity_report.py` | `tests/test_measured_room_accuracy_report.py` (baseline only) | Fidelity gate `--fail-on-regression` | Synthetic baseline only | False negatives not counted | Fix accuracy evaluator (P1.5) |

---

## P3 — World Identity, Change Adoption, Editing

| # | Requirement | Status | Evidence path | Test evidence | Product wiring | Real-world validation | Risk | Next action |
|---|-------------|--------|---------------|---------------|----------------|----------------------|------|-------------|
| R18 | WorldIR schema v1/v2 reconciliation | PARTIAL | `world_ir/schema_v1.py`, `v2_extensions.py` | `tests/test_world_ir.py` | Hierarchy layer optional, populated by 5.2 | Not tested on real worlds | Schema drift | Reconcile v1/v2 contracts |
| R19 | Stable entity IDs through compilers | PARTIAL | `world_ir/entity_reid.py`, `world_delta.py` | `tests/test_entity.py` | carry_identity preserves IDs across versions | Not tested on repeat sessions | Split/merge/ambiguity handling | Identity benchmark on real apartment |
| R20 | Three-way merge (baseline/edited/candidate) | FOUNDATION | `reconstruction/world_merge.py` | New module (import OK, functional test) | Not yet wired into WorldStore commit | None | Manual edits overwritten silently | Wire into commit path |
| R21 | Stale-head conflict behavior | CODE-CONTRACT-VERIFIED | `apps/api/worldstore_service.py:ConcurrentModificationError` | `tests/test_world_adoption_api.py` | Compare-and-swap, retry/dedup | Not tested under concurrent commits | Lost updates | Concurrency test |
| R22 | Typed command bus for edits | FOUNDATION | `reconstruction/constraint_editing.py` | New module (import OK, functional test) | Not yet wired into UI edit commands | None | Arbitrary property mutation | Wire UI → constraint_editing |
| R23 | Preview/validation/confirm/undo/redo | FOUNDATION | `reconstruction/undo_redo.py`, `constraint_editing.py` | New modules (import OK, functional test) | Not yet integrated into Studio | None | Optimistic UI updates claim persistence | Studio integration |
| R24 | Temporal change classification | IMPLEMENTED-UNVERIFIED | `engine/pipeline/temporal_delta.py`, `routes_temporal.py` | `tests/test_temporal_delta.py` | Added/removed/moved/renovated/damaged/unchanged | Not tested on real repeated captures | Registration drift vs physical change | Real repeated-capture benchmark |

---

## P4 — Capture & Indoor Positioning

| # | Requirement | Status | Evidence path | Test evidence | Product wiring | Real-world validation | Risk | Next action |
|---|-------------|--------|---------------|---------------|----------------|----------------------|------|-------------|
| R25 | Device capture manifest + timestamp contract | FOUNDATION | `reconstruction/capture_director.py` | New module (import OK, functional test) | Not yet wired into capture UI | None | No phone IMU integration | Define manifest schema |
| R26 | Clock normalization + camera/IMU extrinsics | MISSING | `evidence/sensors.py` (parser only) | — | Parser does not feed VIO/registration | None | No real VIO backend | Select native/mobile integration |
| R27 | Real phone IMU acquisition | MISSING | — | — | None | None | No hardware | Device integration approach |
| R28 | VIO backend (built + pinned) | MISSING | `trajectories/backend/selection.py` (tries ORB-SLAM3/OpenVINS/Basalt; binaries unavailable) | — | None | None | Binaries unavailable in audit env | Build/pin VIO backend |
| R29 | Camera/IMU sync + drift validation | MISSING | — | — | None | None | No physical indoor loop | Physical device benchmark |
| R30 | Loop closure + floor/stair elevation | FOUNDATION | `reconstruction/loop_closure.py` | New module (import OK, functional test) | Not integrated into registration | None | No real trajectory data | Integrate with scene_identity |
| R31 | Physical LiDAR/RGB-D validation | MISSING | `evidence/depth_frames.py` (synthetic only) | `tests/test_synthetic_rgbd.py` | Synthetic RGB-D tests pass | None | No physical device | Device validation |

---

## P5 — SaaS Operationalization

| # | Requirement | Status | Evidence path | Test evidence | Product wiring | Real-world validation | Risk | Next action |
|---|-------------|--------|---------------|---------------|----------------|----------------------|------|-------------|
| R32 | Tenant authorization on every route | PARTIAL | `apps/api/tenancy.py`, `assert_visible` | `tests/test_account_lifecycle.py` | WorldScope dependency on routes | Not tested for cross-tenant IDOR | IDOR vulnerabilities | Cross-tenant IDOR tests |
| R33 | Object storage signed URLs + tenant boundaries | FOUNDATION | `apps/api/storage.py` | `tests/test_s3_artifact_store.py` | Content-addressed, hash-verified | Not tested for path traversal | Path traversal via artifact IDs | Security audit |
| R34 | Durable queue: idempotency, cancellation, retries | CODE-CONTRACT-VERIFIED | `apps/api/jobs.py` (claim_next_job, reap_stale_jobs, recover_orphaned_jobs) | `tests/test_durable_job_queue.py` | Atomic claim, heartbeat, stale reaping | Not tested under worker crash | Lost jobs on crash | Crash recovery drill |
| R35 | Worker resource classes + queue fairness | MISSING | — | — | None | None | GPU/VRAM isolation absent | Implement resource classes |
| R36 | Distributed presence (replace in-memory) | MISSING | `engine/pipeline/collab_presence.py` (in-memory per process) | — | SSE + markups + BCF export | None | Multi-worker shared state absent | Shared ephemeral backend |
| R37 | Backup/restore + disaster recovery | MISSING | — | — | None | None | No DR drills | Run DR drills |
| R38 | Stripe test-mode billing lifecycle | MISSING | `apps/api/billing_service.py`, `routes_billing.py` | — | Checkout/portal/webhook code exists | Not tested end-to-end | Webhook idempotency unproven | Stripe test-mode walkthrough |

---

## P6 — Product Expansion (post-first-customer)

| # | Requirement | Status | Evidence path | Test evidence | Product wiring | Real-world validation | Risk | Next action |
|---|-------------|--------|---------------|---------------|----------------|----------------------|------|-------------|
| R39 | Capture director + coverage maps | FOUNDATION | `reconstruction/capture_director.py` | New module (import OK) | Not wired into `engine/pipeline/guidance.py` | None | Coverage estimates provisional | Wire into guidance |
| R40 | Fine-grained object recognition | PARTIAL | `perception/detection/maskrcnn_backend.py` | `tests/test_perception_interfaces.py` | Mask R-CNN baseline | Not tested on real fixtures | Open-vocab not adopted | Adopt Grounding DINO/OWLv2 |
| R41 | Parametric/B-rep editing | PARTIAL | `procedural/room_grammar.py`, `building_grammar.py` | `tests/test_procedural.py` | Grammar-based generation | Not constraint-aware | Topology-safe operations missing | Constraint-aware editing |
| R42 | Linked assets + update proposals | MISSING | `apps/api/routes_assets.py` (independent extraction only) | — | Not linked instances | None | Source change detection absent | Three-way merge lifecycle |
| R43 | Contextual assistant + durable task UI | PARTIAL | `assistant/copilot_engine.py` (deterministic, no LLM) | — | Edit limited to generated procedural content | None | General conversation absent | Optional LLM planner |
| R44 | Camera-driven LOD/streaming | MISSING | `worldstore/tiles.py` (foundation only) | — | Tile persistence ≠ streamed viewer | None | No frame-time/memory benchmarks | Streaming controller |
| R45 | Window/door portal visibility | MISSING | — | — | None | None | No portal visibility system | Portal rendering |
| R46 | Specialist analysis hardening (MEP/clash/takeoff/life-safety/thermal) | PARTIAL-UNVERIFIED | `engine/pipeline/{clash_detection,mep_topology,material_takeoff,life_safety_compliance,thermal_envelope}.py` | `tests/test_*_task*_*.py` | Bounds-based, static rates, preliminary screening | Not externally validated | Presented as certified | External validation + disclaimers |

---

## Status Distribution Summary

| Status | Count | Examples |
|--------|-------|----------|
| CODE-CONTRACT-VERIFIED | ~12 | API routes, WorldStore, job lifecycle, export, validation |
| IMPLEMENTED-UNVERIFIED | ~8 | SfM, temporal delta, Gaussian splats, collaboration |
| PARTIAL | ~10 | Metric scale, depth→points, WorldIR schema, detection |
| FOUNDATION | ~8 | Capture director, loop closure, uncertainty map, constraints |
| MISSING | ~8 | Tape-measured rooms, VIO backend, RGB-D device, billing e2e |
| DEFERRED | ~2 | NeRF, city-scale streaming |
| DOCUMENTATION-DEBT | ~6 | PENDING_IMPLEMENTATION.md, IMPLEMENTATION_STATUS.md, CAPABILITIES.md, turnaround plan, vision, reuse map |

**Total requirements tracked: ~46**

---

## Next Actions by Priority

**P0 (Truth & Governance):** Reconcile status docs (R46), protect main branch (R32), add specialist tests to CI (R12/R17), add no-fake-backend guard to CI.

**P1 (First-customer workflow):** Real phone photos → production API → room model (R16), metric scale + dimension uncertainty benchmark (R4/R17), PDF/DXF/IFC/GLB export from measured rooms (R9/R10).

**P2 (Capture & identity):** Real measured rooms (R12–R15), three-room apartment adoption revisit (R19/R24), VIO backend (R28).

**P3 (Core platform):** WorldIR schema contracts (R18), structured editing (R22/R23), linked assets (R42).

---
*Statuses reflect code inspection + test review only. No real-world validation has been performed in this pass.*