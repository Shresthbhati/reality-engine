# FABLE Model Allocation Plan (Deliverable F)

**Date:** 10 November 2026  
**Orchestrator:** Fable

---

## 1. Tiered Model Strategy

The exact provider/model can change; task characteristics determine assignment. Do not delegate final authority on architectural decisions to several low-cost models in parallel.

---

## 2. Tier A — High-Reasoning Model / Human Architecture Review

**Use for:** Canonical data model changes, migration decisions, reconstruction algorithm design, coordinate-frame architecture, World identity/merge semantics, security/tenancy/authorization design, job durability/transaction/consistency changes, cross-module contract decisions, reviewing complex PRs.

**Owner:** Claude (strongest reasoning coder) + human approval for material changes.

**Tasks assigned:**
- WorldIR schema v1/v2 reconciliation (P3.1)
- Transform graph definition (P3.2)
- Three-way merge architecture (P3.5)
- Stale-head conflict behavior (P3.6)
- VIO backend selection (P4.4)
- AuthN/Z implementation decision (P5.1)
- Alembic migration strategy
- EvidenceState enum unification

**Constraint:** One owner proposes; one reviewer challenges; human approves material architecture changes. Never let multiple low-cost models make conflicting architecture decisions in parallel.

---

## 3. Tier B — Capable Low-Cost Coding Model (Qwen/Nemotron-class)

**Good assignments:**
- Add focused unit and regression tests around existing contracts
- Trace API route → schema → service → frontend consumer
- Implement small adapter against already-approved interface
- Add validation for defined schema
- Add structured logging/metrics to one module
- Fix deterministic bugs with reproduction
- Update one documentation section from verified source
- Add CI wiring for known test command
- Implement mechanical refactors with explicit before/after behavior

**Tasks assigned (from FABLE_TASK_BACKLOG.md):** TB-001 through TB-010

**Constraint:** Never assign open-ended instructions like "build indoor localization", "make reconstruction accurate", "implement all WorldIR", or "finish SaaS". These require research, hardware/data, architectural choices, cross-domain validation.

---

## 4. Tier C — Small/Fast Model Tasks

**Good assignments:**
- Generate test matrices from stable API schema
- Identify duplicate/unused code candidates (report only)
- Write straightforward fixtures and serialization tests
- Add typed error messages and UI empty/error states to bounded component
- Convert approved checklist into docs or CI configuration
- Classify repository paths into module inventory for human review

**Tasks assigned (from FABLE_TASK_BACKLOG.md):** TC-001 through TC-005

**Constraint:** Never let small model invent acceptance criteria, silently modify core contracts, remove fixtures without classification, or claim tests ran when it only inspected code.

---

## 5. Role-to-Task Mapping (from 06_AGENT_OWNERSHIP_AND_TASK_CARDS.md)

| Owner | Primary responsibility | Owns | Must not independently own |
|-------|----------------------|------|---------------------------|
| **Fable** | Architecture/programme orchestration | Backlog, dependency graph, contracts, task routing, integration review | Ad hoc rewrite of every module |
| **Claude** | WorldIR + interior geometry/compiler seam | Canonical schema decisions, geometry invariants, compiler integration | SaaS billing, unrelated UI redesign |
| **FreeBuff** | Real reconstruction/perception | Production capture-to-reconstruction, detector/segmenter/identity integration | WorldStore commit semantics without contract approval |
| **Antigravity** | Desktop Reality Studio | Existing `frontend/` Next.js UI, interaction design, viewport workflow | A competing `apps/studio/` product |
| **Cline** | Frontend/API truth | API proxy/consumer consistency, honest error/empty/loading states | Redesigning backend schemas without owner approval |
| **OpenCode** | Backend hardening | Job lifecycle, persistence, auth, resource limits, security and recovery | Replacing product workflows or canonical geometry schema |
| **KiloCode** | Integration and failure testing | Negative tests, race/failure injection, golden-loop QA | Owning production implementation of shared subsystems |
| **Qwen/Nemotron-class** | Bounded implementation | Focused tests, adapters, mechanical fixes, docs, inventories | Architectural authority or unbounded tasks |

---

## 6. Interface Ownership (Single Owner Per Contract)

Before implementation, nominate a single owner for each contract:

| Contract | Owner |
|----------|-------|
| WorldIR schema/versioning | Claude (WorldIR owner) |
| Job lifecycle and stage schema | OpenCode (orchestration owner) |
| Session/evidence manifest | Capture/ingestion owner |
| ChangeSet and World commit | WorldStore owner |
| Edit command schema | Editing owner |
| API response schemas | API contract owner (Cline) |
| Frontend status mapping | UI contract owner (Antigravity) |
| Artifact manifest/key convention | Storage owner |

**Rule:** Any cross-boundary change requires a short contract proposal and explicit approval before parallel coding.

---

## 7. Tasks Requiring Human Approval (Before Code Changes)

| Decision | Why human approval required |
|----------|----------------------------|
| Alembic migration strategy | Existing dev DBs have data; migration must not lose data |
| WorldIR v2 schema fields | Breaking change to canonical model; all consumers affected |
| AuthN/Z implementation | Currently absent (prototype trust boundary); security decision |
| VIO backend selection | Hardware dependency; license/cost implications |
| Billing production readiness | Stripe code exists but unverified; financial decision |
| Distributed presence backend | Multi-worker shared state; architectural change |

---

## 8. Tasks That Cannot Be Done Autonomously (Hardware/Real Data Required)

| Task | ID | Blocker |
|------|-----|---------|
| Capture 6+ diverse measured rooms | P2.2 | Requires human field operator + physical captures |
| Record independent ground truth | P2.3 | Requires laser/tape measurements + annotator agreement |
| Run ordinary phone photos through real API path | P2.5 | Requires real photo dataset first |
| Evaluate on real measured rooms | P2.6 | Requires P2.2–P2.5 first |
| Validate exports in external CAD/BIM apps | P2.8 | Requires P2.7 first |
| Three-room apartment repeat-capture trial | P2.9 | Requires human field operator |
| Real phone IMU acquisition | P4.3 | Requires device integration approach decision first |
| Build + pin VIO backend | P4.4 | Requires P4.3 first |
| Validate camera/IMU sync + drift on physical indoor loop | P4.5 | Requires physical device |
| Add loop closure + floor/stair elevation | P4.6 | Requires P4.5 first |
| Validate physical LiDAR/RGB-D | P4.7 | Requires physical device |
| Run backup/restore + DR drills | P5.7 | Requires infrastructure |
| Stripe test-mode billing lifecycle | P5.8 | Requires Stripe test-mode credentials |
| Hosted pilot without developer intervention | P5.9 | Requires P5.8 first |

---

## 9. Model Allocation Summary

| Tier | Model class | Task count | Primary owners |
|------|------------|------------|----------------|
| Tier A | High-reasoning + human | ~8 tasks | Claude + human |
| Tier B | Capable low-cost (Qwen/Nemotron) | 10 tasks | Qwen/Nemotron-class models |
| Tier C | Small/fast | 5 tasks | Small models |
| **Blocked** | Hardware/real data required | ~12 tasks | Human field operator + hardware |

**Rule:** Tier B and C tasks must include verified context, exact paths, tests, and stop conditions. A model must stop and report if it discovers the task requires a shared contract change, cross-boundary refactor, missing dependency, unavailable hardware, unclear semantics, or conflicting implementation. It must not improvise a new architecture.