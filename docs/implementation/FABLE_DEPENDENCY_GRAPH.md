# FABLE Dependency Graph & Staged Roadmap (Deliverable D)

**Date:** 10 November 2026  
**Orchestrator:** Fable (OpenCode agent)

---

## 1. Dependency Principles

- **Dependency before breadth:** If P1.10 (registration) blocks P1.11 (fusion), registration goes first.
- **Shared contracts frozen before parallel work:** Two agents must not edit WorldIR schemas, job state machines, or API contracts simultaneously.
- **One owner per shared contract:** Consumer migration must be in the same PR or coordinated versioned rollout.
- **No code changes in Phase P0:** Read-only validation only.
- **Hardware/real-data items cannot start until assets exist.**

---

## 2. Phase P0 — Fresh Baseline & Project Truth (Read-Only)

**Goal:** Ensure all agents work from the same actual repository state. No code implementation required.

| Task | ID | Deps | Owner | Deliverable | Exit gate |
|------|-----|------|-------|-------------|-----------|
| Revalidate branch/SHA/worktree | P0.1 | — | Fable | git status, SHA, open PRs, recent commits | Source-backed repo map |
| Repository map (frontend/API/engine/WorldIR/storage/jobs/tests/docs) | P0.2 | P0.1 | Fable + code model | Module map, API route map, reconstruction call graph | Every claim has file path/symbol |
| Trace real production path (upload → worker → WorldStore) | P0.3 | P0.2 | Fable + code-analysis model | Call graph, stage contracts, failure branches | Sequence diagram + source refs |
| Trace UniversalSpatialPipeline usage | P0.4 | P0.3 | Code-analysis model | Is it in the production path? | Yes/no with evidence |
| Map API routes, schemas, frontend proxies, consumers | P0.5 | P0.2 | Cline | Route → schema → service → consumer map | Contract mismatches listed |
| Map DB migrations, tenant scope, storage, job persistence | P0.6 | P0.2 | OpenCode | DB schema, migration strategy, tenant scoping | Storage/database map |
| Reconcile CI (executed vs collect-only vs advisory) | P0.7 | P0.2 | Qwen/Nemotron | CI test registry (required/scheduled/manual/advisory) | Exact commands documented |
| Create status ledger + ADR index | P0.8 | P0.1–P0.7 | Fable | `IMPLEMENTATION_LEDGER.md`, `ARCHITECTURE_DECISIONS.md` | Verified requirement matrix |

**Exit gate P0:** Source-backed repository map + verified requirement matrix. Zero source code changes.

---

## 3. Phase P1 — Truth, Quality Gates & Regression Protection

**Goal:** Stop false success; make missing geometry visible.

| Task | ID | Deps | Owner | Deliverable | Exit gate |
|------|-----|------|-------|-------------|-----------|
| Tests for API failure vs successful empty responses | P1.1 | P0.5 | Cline + Qwen | Regression tests for sessions/worlds/jobs routes | Backend 500 → error state, not empty list |
| Verify every job state (QUEUED/RUNNING/PARTIAL/SUCCEEDED/FAILED/CANCELLED) | P1.2 | P0.6 | OpenCode + KiloCode | State transition tests, stale job reaping | Timeout/missing output never becomes success |
| No timeout/malformed/5xx → successful empty data | P1.3 | P1.2 | OpenCode | Error-state tests across all API routes | Empty-on-error eliminated |
| Audit fake/demo data; classify as test/dev/docs/production | P1.4 | P0.2 | Cline + Qwen | Inventory of hard-coded/fake data; classification | Only production fakes flagged |
| Fix fidelity evaluator: count false negatives + false positives | P1.5 | P0.7 | KiloCode + strong model | Adversarial fixtures (missing wall, extra wall, wrong dimensions, duplicate, wrong class, offset, occluded) | Missing/extra entities fail the gate |
| Per-class precision/recall, geometry/dimension error, coverage, uncertainty | P1.6 | P1.5 | Strong model | Evaluator output with component metrics | No opaque "accuracy" score |
| Wire specialist + no-fake-backend tests into required CI | P1.7 | P1.6 | Qwen + OpenCode | CI workflow updated; runtime measured | CI reports what it truly executes |
| Decide which lint/type checks become required | P1.8 | P0.7 | OpenCode | Ruff/mypy enforceability decision | Baseline cleanup before enabling |
| Artifact manifest validation + WorldIR schema/relationship validation | P1.9 | P0.6 | OpenCode | Validation tests for artifacts + WorldIR relationships | Invalid WorldIR rejected before persistence |

**Exit gate P1:** Deterministic failures caught; incomplete/failed processing cannot masquerade as valid World; CI reports what it truly executes.

---

## 4. Phase P2 — Real Measured Interior Dataset (HUMAN BLOCKED)

**Goal:** Prove the first paid workflow on physical spaces. **Blocked on human field operator + physical captures.**

| Task | ID | Deps | Owner | Deliverable | Exit gate |
|------|-----|------|-------|-------------|-----------|
| Define dataset protocol (consent, privacy, device metadata, measurements, coordinate frame, annotations) | P2.1 | P0.1 | Fable + human field operator + QA | Dataset protocol document | Consent + annotation schema defined |
| Capture 6+ diverse measured rooms (plain walls, windows/glass, reflective surfaces, doors, fixtures, furniture, irregular geometry, difficult lighting) | P2.2 | P2.1 | Human field operator | `datasets/measured/<room>/` with raw media + measurements | 6 rooms with tape/laser measurements |
| Record independent ground truth + annotator agreement | P2.3 | P2.2 | Human field operator | Ground truth file per room | Laser/tape measurements for L×W×H, openings, floor/ceiling heights |
| Train/dev/blind-test split (no leakage) | P2.4 | P2.3 | Qwen/Nemotron | Split definition document | Blind holdout preserved |
| Run ordinary phone photos through real API path (not hand-built results) | P2.5 | P2.1, P2.4 | Qwen/Nemotron | Reconstruction runs via POST /api/reconstructions | Real photos processed through production path |
| Evaluate wall/opening completeness, object recall, dimensional error, registration residual, geometry validity, runtime/memory | P2.6 | P2.5 | Strong model | Evaluation report with per-class metrics | False negatives reported; dimensional error table |
| Generate floor plan/section + PDF/DXF/IFC/GLB from same World version | P2.7 | P2.6 | Qwen/Nemotron | Exported artifacts from real World | Artifacts generated from real data |
| Validate exports in external target applications (record versions/results) | P2.8 | P2.7 | Human field operator | External validation report | Blender/Revit/AutoCAD import results |
| Three-room apartment repeat-capture trial; revisit one room; adopt second session without duplicate World | P2.9 | P2.8 | Human field operator | Repeat-capture report | No duplicate World; no overwritten edits |

**Exit gate P2:** Documented real-data report with raw input, ground truth, reproducible commands, per-class metrics, false negatives, external export evidence.

**BLOCKER:** P2.2–P2.3 require human field operator + physical captures. Cannot be done autonomously.

---

## 5. Phase P3 — World Identity, Change Adoption & Editing Correctness

**Goal:** Persistent Worlds behave safely over time.

| Task | ID | Deps | Owner | Deliverable | Exit gate |
|------|-----|------|-------|-------------|-----------|
| Reconcile WorldIR schema versions + stable ID rules | P3.1 | P0.6 | Claude/WorldIR owner | Schema reconciliation proposal | Stable ID rules documented |
| Define transform graph (session-local, World-local, building/site, geographic frames) | P3.2 | P3.1 | Claude | Transform graph documentation | Coordinate frames explicit |
| Define matching score inputs + ambiguity thresholds for cross-session identity | P3.3 | P3.2 | Claude | Matching score specification | Ambiguity thresholds explicit |
| Separate not-observed, occluded, removed, moved, changed states | P3.4 | P3.3 | Claude | State separation specification | Visibility/coverage-based absence |
| Implement/verify three-way merge (baseline/user-edited/new-session candidate) | P3.5 | P3.4 | Claude + OpenCode | Three-way merge implementation + tests | Baseline World, edited World, candidate merge correctly |
| Enforce base-version preconditions + stale-head conflict behavior | P3.6 | P3.5 | OpenCode | Conflict behavior tests | Stale HEAD returns conflict, never silent overwrite |
| Route manual + AI edits through one typed command bus | P3.7 | P3.6 | OpenCode | Command bus integration | All edits go through typed commands |
| Add preview/validation/commit/undo/redo/audit trail tests | P3.8 | P3.7 | Qwen/Nemotron | Edit operation tests | Undo/redo preserves entity IDs + geometry |
| Validate temporal change classification on repeated real captures | P3.9 | P2.9, P3.8 | Strong model | Temporal change test on real data | Change classification validated on real captures |

**Exit gate P3:** Second-session update preserves user edits, avoids duplicate entities, records conflicts/uncertainty for review.

---

## 6. Phase P4 — Capture & Indoor Positioning

**Goal:** Upgrade capture from photo + outdoor GPS to reliable multi-sensor capture.

| Task | ID | Deps | Owner | Deliverable | Exit gate |
|------|-----|------|-------|-------------|-----------|
| Define device capture manifest + sensor timestamp contract | P4.1 | P0.6 | Capture/ingestion specialist | Manifest contract document | Required/optional fields, timestamp uncertainty |
| Implement clock normalization + camera/IMU extrinsics metadata | P4.2 | P4.1 | Qwen/Nemotron | Clock normalization code | Timestamps aligned across sources |
| Add real phone IMU acquisition (after selecting native/mobile integration approach) | P4.3 | P4.2 | Capture specialist | IMU acquisition integration | Phone IMU data captured |
| Build + pin one supported VIO backend (license, binary provenance, build flags, platforms) | P4.4 | P4.3 | FreeBuff/strong model | VIO backend integrated | Backend pinned; license verified |
| Validate camera/IMU synchronization + trajectory drift on physical indoor loop | P4.5 | P4.4 | FreeBuff + human operator | Drift benchmark report | Reproducible physical-device benchmark |
| Add loop closure + floor/stair elevation handling | P4.6 | P4.5 | FreeBuff | Loop closure + floor tracking code | Loop closure corrects drift; floors consistent |
| Validate physical LiDAR/RGB-D on specific devices before claiming support | P4.7 | P4.6 | Human operator | Device validation report | Physical LiDAR/RGB-D tested |

**Exit gate P4:** Reproducible physical-device benchmark with drift, scale, timing, failure metrics. Parser tests alone do not pass.

**BLOCKER:** P4.3–P4.7 require physical hardware.

---

## 7. Phase P5 — SaaS Operationalization

**Goal:** Make hosted operation safe, measurable, commercially usable.

| Task | ID | Deps | Owner | Deliverable | Exit gate |
|------|-----|------|-------|-------------|-----------|
| Verify tenant authorization on every Project/World/Session/Artifact/Job route | P5.1 | P0.6 | OpenCode security owner | Tenant isolation tests | Cross-tenant IDOR fails closed |
| Audit object storage signed URLs, object keys, tenant boundaries, retention | P5.2 | P5.1 | OpenCode | Storage security audit | No public object keys; no unscoped paths |
| Verify durable queue: idempotency, cancellation, retries, checkpointing, crash recovery | P5.3 | P0.6 | OpenCode | Queue reliability tests | Idempotent stage writes; no duplicate World commit |
| Implement worker resource classes (CPU, GPU, high-memory, export) + queue fairness/backpressure | P5.4 | P5.3 | OpenCode | Resource class implementation | GPU/VRAM isolation between tenants |
| Add per-tenant concurrency + VRAM/resource budgets | P5.5 | P5.4 | OpenCode | Tenant budget enforcement | Concurrency limits enforced |
| Replace process-local presence with shared ephemeral backend + durable audit | P5.6 | P5.5 | Collaboration owner | Distributed presence implementation | Multi-worker shared state |
| Run backup/restore + disaster-recovery drills | P5.7 | P5.5 | OpenCode | DR drill results | Backup/restore verified |
| Run Stripe test-mode lifecycle (checkout, webhooks, entitlements, cancellation, quota races) | P5.8 | P5.5 | OpenCode | Billing test results | Real Stripe test-mode flow |
| Hosted pilot without developer intervention; measure cost per successful room | P5.9 | P5.8 | Human operator | Hosted pilot report | Design partner completes workflow solo |

**Exit gate P5:** Tenant-isolation tests, crash recovery, billing lifecycle, hosted design-partner evidence.

---

## 8. Phase P6 — Product Expansion (Post-First-Customer)

Prioritize based on customer evidence. Each capability needs: defined customer workflow, canonical WorldIR contract, product entry point, real validation, documented limitations.

| Workstream | Items | Priority |
|------------|-------|----------|
| Evidence & reliability | R6–R10, R18–R20, R36–R40 | P0 |
| Capture & perception | R1–R5, R11–R15 | P1 |
| Persistent World | R14, R21–R25 | P1 |
| Editing & product | R16–R17, R26–R30 | P2 |
| Rendering & scale | R31–R35 | P3 |

---

## 9. Parallelism Rules

**Safe to parallelize (after shared contracts frozen):**
- Agent A: Add tests for session proxy error semantics
- Agent B: Inventory test coverage for specialist endpoints (report-only)
- Agent C: Write dataset capture protocol from approved requirements
- Agent D: Add observability to one worker stage using existing telemetry interface

**Unsafe to parallelize:**
- Two agents editing WorldIR schemas simultaneously
- One agent changing job states while another rewrites frontend status mapping
- Multiple agents independently implementing entity identity
- One agent migrating storage while another changes artifact keys
- One agent altering API response shape while another changes consumers

---

## 10. Task Sizing Guide

A good low-cost model task should:
- Touch 1–5 files (occasionally more for mechanical test sweeps)
- Have one principal behavior change
- Take less than one focused working session
- Have an existing or easily written reproduction
- Have a deterministic test command
- Avoid schema/API migrations unless the contract is already approved

If the task requires more than one subsystem or an architectural decision, split it.

---
*This is a dependency-ordered plan. Execution order follows `30_BUILD_PLAN.md`; this document maps the FABLE deliverables onto that order.*