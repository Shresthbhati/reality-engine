# FABLE Architecture Delta Report (Deliverable C)

**Date:** 10 November 2026  
**Audited baseline:** `main` @ `199ca3bbf797ca515aa211f80a95c4850e89dbda` (9 Nov 2026)  
**Current HEAD:** `feat/thermal-envelope-ashrae901` @ `f309041a`  
**Target architecture reference:** `docs/blueprint/34_MASTER_PLAN.md`, `docs/blueprint/36_UI_SPEC.md`, `10_ORIGINAL_OPTIMIZED_SYSTEM_DESIGN.md`

---

## 1. Executive Summary

The current implementation **substantially matches** the target architecture's core design decisions (one canonical World model, modular monolith + isolated workers, immutable evidence/versions, explicit adoption, local-first processing). The primary deltas are in **integration proof** (the pipeline is not yet proven end-to-end on real photos), **SaaS operationalization** (tenant isolation, billing, distributed presence not validated), and **real-world validation** (no measured-room accuracy data).

No migration is required to the core schema; the delta is primarily about **wiring, verification, and operational hardening**.

---

## 2. Five Architectural Decisions — Match vs. Delta

| Decision | Current implementation | Match? | Delta / Action |
|----------|----------------------|--------|----------------|
| **D1: One canonical World model** | WorldIR v1 is the single schema; meshes/points/splats/images are representations; WorldStore is immutable version store | **YES** | WorldIR v1/v2 schema reconciliation needed (not yet done) |
| **D2: Persistent World ≠ time-bounded Session** | World and Session are separate models; Session produces evidence + ChangeSet; World versions immutable | **YES** | None — architecture correctly separates them |
| **D3: Sessions never silently overwrite World** | Adoption protocol: intent file → WorldStore version → read-back → DB → clear; orphan quarantine | **YES** | None — correctly implemented |
| **D4: World versions immutable; HEAD advances transactionally** | WorldStore DAG via parents; compare-and-swap on HEAD; `worldstore_service.commit_version` | **YES** | Concurrent commit race handling verified |
| **D5: Evidence states distinct (observed/reconstructed/inferred/generated/unknown/conflict)** | `Provenance` enum + `EvidenceState` — split across two locations, not unified | **PARTIAL** | Unify into one `EvidenceState` enum (name mapping in 37_REUSE_MAP.md) |
| **D6: Not-observed ≠ removed/empty** | Coverage fractions (seen/weak/not_seen) in `wall_coverage()`; `unobserved_gap` recorded explicitly | **YES** | None |
| **D7: Manual and AI edits use same typed command service** | `constraint_editing.py` + `undo_redo.py` + `scene_inspector.py` exist but not yet wired into UI command bus | **PARTIAL** | Wire UI edit commands through constraint_editing |
| **D8: Modular monolith + isolated CPU/GPU workers** | FastAPI monolith + in-process worker daemon; standalone `apps/api/worker.py` | **YES** | GPU worker resource classes not yet implemented |
| **D9: Local-first + optional cloud acceleration** | Same job contract targets local or hosted workers; `REALITY_DISABLE_IN_PROCESS_WORKER` env | **YES** | None |
| **D10: Existing Next.js frontend is the main UI** | `/worlds/[id]` renders `SpatialWorkstation`; `apps/studio/` is Vite skeleton (not used) | **YES** | None — do not build parallel Studio |
| **D11: Specialist modules consume same versioned World** | MEP/clash/takeoff/life-safety/thermal derive from WorldIR entities | **YES** | None — but outputs not externally validated |

---

## 3. What Must Be Preserved (Do Not Change Without Approval)

| Contract | Why | Owner |
|----------|-----|-------|
| WorldIR v1 schema entities/geometry/relationships | Canonical World model; all subsystems depend on it | WorldIR owner |
| Job state machine (QUEUED→RUNNING→{SUCCEEDED|PARTIAL|FAILED|CANCELLED}) | Failure honesty; timeout never becomes success | Job orchestration owner |
| Adoption protocol (intent→version→read-back→DB→clear) | Prevents orphan versions and duplicate commits | WorldStore owner |
| Edit command schema (preview/validate/confirm/audit/undo) | Manual and AI edits must share one path | Editing owner |
| API response schemas (worlds/sessions/jobs/evidence) | Frontend consumers depend on exact shapes | API contract owner |
| Artifact content-addressing (SHA-256 keys) | Integrity and deduplication | Storage owner |
| Evidence states (OBSERVED/RECONSTRUCTED/INFERRED/GENERATED/UNKNOWN/CONFLICT) | Truth policy; never inferred as observed | Provenance owner |

---

## 4. What Needs Migration

| Migration | Scope | Risk | Owner |
|-----------|-------|------|-------|
| Alembic migrations (replace `metadata.create_all`) | `apps/api/migrations/`, `apps/api/db.py` | Medium — existing dev DBs must upgrade without data loss | Backend owner |
| WorldIR v1→v2 schema reconciliation | `world_ir/schema_v1.py`, `v2_extensions.py` | High — stable entity IDs must survive | WorldIR owner |
| Unify EvidenceState enum | `world_ir/quality_state.py` + `Provenance` enum | Medium — all consumers must be updated | Provenance owner |
| Worker resource classes (CPU/GPU/high-memory/export) | `apps/api/jobs.py` worker_loop | Medium — resource isolation not yet implemented | DevOps owner |
| Distributed presence backend | `engine/pipeline/collab_presence.py` (in-memory) | High — multi-worker shared state absent | Collaboration owner |

---

## 5. What Needs Human Decision

| Decision | Why human | Options | Recommendation |
|----------|-----------|---------|----------------|
| Alembic migration strategy | Existing dev DBs have data; migration must not lose it | Alembic vs manual SQL vs keep `create_all` | Adopt Alembic with tested upgrade paths |
| WorldIR v2 schema fields | Breaking change to canonical model | Additive-only vs breaking restructure | Additive-only (preserve v1 compatibility) |
| AuthN/Z implementation | Currently absent (prototype trust boundary) | Add full auth vs document as production blocker | Document as blocker; add minimal auth for SaaS |
| VIO backend selection | ORB-SLAM3/OpenVINS/Basalt binaries unavailable | Build own vs adopt existing vs defer | Defer until hardware available |
| Billing production readiness | Stripe code exists but unverified | Add real Stripe test-mode vs keep as code-only | Document as production blocker |

---

## 6. Architecture Consistency Checks (verified in code)

| Check | Result | Evidence |
|-------|--------|----------|
| Frontend ↔ backend route consistency | PARTIAL | `frontend/src/app/api/*` proxies exist; some routes may have stale shapes (Cline audit needed) |
| Job lifecycle ↔ UI status mapping | PARTIAL | `routes_reconstructions.py` `world_status` router; UI shows stage labels (honest, no fake %) |
| Artifact storage ↔ WorldStore | YES | `worldstore_service.commit_version` stores points/cameras/report; artifact store for mesh/plane payloads |
| Evidence → WorldIR provenance | PARTIAL | `_stamp_reconstruction_provenance` stamps entities; `_assert_reconstruction_traced` gates persistence |
| Scale state ↔ dimension behavior | YES | `dimension_state_for_basis()`; ESTIMATED vs MEASURED honestly labeled |
| Collision between modular monolith and microservices | YES | No microservice split; worker daemon is process-isolated but same codebase |

---

## 7. Delta Summary by Priority

| Priority | Delta items | Effort | Blocking |
|----------|-------------|--------|----------|
| P0 (Truth/Governance) | Reconcile status docs, protect main, add specialist tests to CI, no-fake-backend guard | Low–Medium | No |
| P1 (First-customer workflow) | Real-capture pipeline proof, measured-room accuracy, external export validation | High | Yes (needs measured rooms) |
| P2 (Capture/identity) | VIO backend, indoor localization, three-way merge tests | High | Yes (needs hardware) |
| P3 (Core platform) | WorldIR schema contracts, structured editing, linked assets | Medium | No |
| P4–P6 (Rendering/SaaS/specialist) | Streaming, portals, billing e2e, distributed presence | High | Yes (needs infrastructure) |

---

## 8. Migration Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Alembic migration loses dev DB data | Medium | High | Test upgrade on copy of existing DB before production |
| WorldIR v2 breaks existing worlds | Medium | High | Additive-only changes; version-pinned reads |
| AuthN/Z addition breaks unauthenticated prototype | Low | Medium | Document as production blocker; add incrementally |
| VIO backend integration breaks SfM pipeline | Low | Medium | VIO is optional; SfM works without it |

---
*This is a read-only analysis. No source code was modified.*