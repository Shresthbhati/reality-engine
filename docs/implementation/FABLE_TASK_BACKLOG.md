# FABLE Task Backlog for Cheaper Models (Deliverable E)

**Date:** 10 November 2026  
**Orchestrator:** Fable  
**Task packet format:** Every task includes ID, objective, baseline SHA, allowed files, verified context, expected behavior, non-goals, acceptance criteria, required tests, failure cases, deliverable report.

**Baseline SHA:** `f309041a5ad301c59774fe21bb78103b54b046aa` (branch `feat/thermal-envelope-ashrae901`)

---

## Tier B — Capable Low-Cost Coding Model Tasks (Qwen/Nemotron-class)

These tasks touch 1–5 files, have one principal behavior change, and deterministic test commands.

---

### TB-001 — Session contract: error vs empty state

**Objective:** Verify that backend failure on the sessions endpoint is never presented as an empty sessions list.

**Allowed files:** `frontend/src/app/api/sessions/route.ts` (verify existence first), `apps/api/routes_sessions.py`, `tests/test_sessions_proxy_contract.py` (create if needed)

**Verified context:**
- `frontend/src/app/api/sessions/` directory exists
- Prior audit found fallback returning `{error: "...", sessions: [], count: 0}` — dangerous because backend failure looks like "zero sessions"
- Need to verify current file contents and git state

**Expected behavior:** Backend failure → error state with retry; empty successful response = genuinely zero sessions; never convert 500/503/timeout/malformed → empty sessions.

**Non-goals:** Redesign session endpoint; change API shape without approval.

**Acceptance criteria:**
1. Current `sessions/route.ts` contents verified and recorded
2. All consumers of sessions endpoint use correct response shape (`items` vs `sessions`/`count`)
3. Regression test added: backend 500 → error state; 200 empty → empty state
4. No production fake-success fallback remains

**Required tests:** `pytest tests/test_sessions_proxy_contract.py` (create); `npm run test:contract` in frontend

**Failure cases:** 500, 503, timeout, malformed JSON, network failure

**Deliverable:** Changed files, tests run, exact output, remaining risks

---

### TB-002 — API contract sweep: worlds route

**Objective:** Audit the worlds API surface for request → validation → backend response → frontend expectation → error handling mismatches.

**Allowed files:** `apps/api/routes_worlds.py`, `frontend/src/app/api/worlds/`

**Verified context:** `routes_worlds.py` is 147KB (largest route file); handles CRUD, versions, commit, rollback, diff

**Expected behavior:** Correct property names; no stale response shapes; no success-on-error; no empty-on-error; no hard-coded IDs; no fake progress/counts/versions/reports; no swallowed errors; no HTTP errors converted to 200.

**Non-goals:** Redesign endpoints unless actual contract bug found.

**Acceptance criteria:**
1. Route map documented (request → validation → backend response → frontend expectation → error handling)
2. Mismatches listed with file:line references
3. No production fake state found in worlds route

**Required tests:** Frontend contract tests (`npm run test:contract`); relevant API tests

**Failure cases:** Unauthorized, not found, stale version, malformed body, timeout, internal failure

---

### TB-003 — API contract sweep: sessions route

**Objective:** Same as TB-002 for sessions endpoint.

**Allowed files:** `apps/api/routes_sessions.py`, `frontend/src/app/api/sessions/`

---

### TB-004 — API contract sweep: evidence route

**Objective:** Same as TB-002 for evidence endpoint.

**Allowed files:** `apps/api/routes_misc.py` (evidence uploads), `frontend/src/app/api/`

---

### TB-005 — API contract sweep: reconstructions route

**Objective:** Same as TB-002 for reconstructions endpoint (the production photo upload path).

**Allowed files:** `apps/api/routes_reconstructions.py`

---

### TB-006 — API contract sweep: jobs route

**Objective:** Same as TB-002 for jobs endpoint.

**Allowed files:** `apps/api/routes_jobs.py`

---

### TB-007 — Fake state hunt: search for production fakes

**Objective:** Search frontend and API for hard-coded/fake data; classify each occurrence as test/dev/docs/production fake; only production fakes are problems.

**Allowed files:** Full repo (read-only search); create `tests/test_no_fake_backend_in_production.py` if not exists

**Verified context:** Search terms: `world-compiled-seed42`, `v1-canonical-baseline`, `hard-coded room data`, `fake mesh`, `fake points`, `fake cameras`, `fake evidence`, `fake confidence`, `fake reports`, `static versions`, `fake commits`, `artificial delays`, `fallback-success`

**Expected behavior:** Every occurrence classified; only #4 (production fake) requires action; never delete deterministic test fixtures without classification.

**Non-goals:** Remove test fixtures; change test behavior.

**Acceptance criteria:**
1. Inventory of all fake-state occurrences with classification
2. Production fakes listed with file:line
3. No test fixtures deleted

**Required tests:** None (inventory task)

**Failure cases:** None (report-only)

---

### TB-008 — CI test registry: document exact test execution

**Objective:** Document which test files execute on PR vs schedule vs manual vs not at all. Identify specialist suites absent from required CI.

**Allowed files:** `.github/workflows/ci.yml`, `.github/workflows/e2e.yml`, `tests/` (read-only inventory)

**Verified context:** CI core test list is explicit (~50 files); `pytest tests/ --collect-only` is collection not execution; ruff `--exit-zero`; mypy `|| true`; slow tests skipped on push

**Expected behavior:** Exact command and frequency for every test group; specialist suites identified as absent from required CI

**Non-goals:** Add tests; change CI configuration without approval.

**Acceptance criteria:**
1. CI test registry document with exact commands
2. Specialist suites classified (required/scheduled/manual/advisory/collection-only/not wired)
3. No collect-only command described as execution

**Required tests:** None (documentation task)

---

### TB-009 — Frontend product simplification: route inventory

**Objective:** Map current Next.js user flows and routes; identify redundant panels; propose simplified Project → Capture → Review → World workflow.

**Allowed files:** `frontend/src/app/` (read-only inventory)

**Verified context:** 40 routes → target 12 routes per `36_UI_SPEC.md`; SpatialWorkstation renders `/worlds/[id]`

**Expected behavior:** Route inventory, user journey, redundant panels, empty/error/partial states, responsive/mobile behavior, accessibility gaps

**Non-goals:** Build a parallel Studio app (`apps/studio/` is Vite skeleton, not used)

**Acceptance criteria:**
1. Route inventory with keep/replace per responsibility
2. Simplified workflow proposal
3. No parallel Studio created

**Required tests:** None (planning task)

---

### TB-010 — Specialist feature truth labels

**Objective:** Ensure MEP, clash, takeoff, life safety, thermal outputs expose assumptions, input version, observed-vs-inferred status, validation limitations.

**Allowed files:** `engine/pipeline/{clash_detection,mep_topology,material_takeoff,life_safety_compliance,thermal_envelope}.py`, `apps/api/routes_{clash,mep,takeoff,life_safety,thermal_envelope}.py`

**Verified context:** Bounds-based checks, static example USD rates, preliminary screening, default "Reality Engine Certified Auditor" label

**Expected behavior:** No unsupported certification claims; no hidden-service detection claims; PDFs/UI labels consistent; tests cover default assumptions + disclaimers

**Non-goals:** Add real regional pricing; add jurisdiction-specific code profiles

**Acceptance criteria:**
1. All specialist outputs declare assumptions + input version + observed-vs-inferred + validation limitations
2. No "certified" claims without external validation
3. Tests cover disclaimers

**Required tests:** Existing specialist test suites

---

## Tier C — Small/Fast Model Tasks

### TC-001 — Test matrix generation from API schema

**Objective:** Generate test matrices from stable API schema (worlds, sessions, jobs).

**Allowed files:** `tests/` (create new test files)

**Expected behavior:** Test matrices for unauthorized, not found, stale version, malformed body, timeout, internal failure per endpoint

---

### TC-002 — Duplicate/unused code identification (report only)

**Objective:** Identify duplicate/unused code candidates; report only, do not delete.

**Allowed files:** Full repo (read-only)

**Expected behavior:** List of candidate duplicates with file:line; no deletions

---

### TC-003 — Fixture and serialization tests

**Objective:** Write straightforward fixtures and serialization tests for WorldIR entities.

**Allowed files:** `tests/test_world_ir_serialization.py` (create)

**Expected behavior:** WorldIR entities serialize/deserialize correctly; geometry_data round-trip

---

### TC-004 — Typed error messages + UI empty/error states

**Objective:** Add typed error messages and UI empty/error states to a bounded component.

**Allowed files:** `frontend/src/components/` (one bounded component)

**Expected behavior:** Component shows honest empty/error states; no fake loading

---

### TC-005 — Convert approved checklist into docs/CI configuration

**Objective:** Convert the FABLE deliverables checklist into CI configuration or docs.

**Allowed files:** `.github/workflows/`, `docs/implementation/`

**Expected behavior:** CI runs the documented test groups; docs reflect current state

---

## Task Assignment Summary

| Tier | Task count | Estimated effort | Blockers |
|------|-----------|------------------|----------|
| Tier B (Qwen/Nemotron) | 10 | 1–2 sessions each | None (read-only + bounded writes) |
| Tier C (small/fast) | 5 | <1 session each | None |

**Tasks requiring strong model / human:** P2.2–P2.3 (measured rooms), P4.3–P4.7 (VIO/hardware), P5.8–P5.9 (billing e2e, hosted pilot), P3.1 (WorldIR schema), P3.5 (three-way merge architecture)

---
*Task packets follow the mandatory format from `05_EXECUTION_PLAN_CHEAP_MODELS.md`. Each task includes verified context, exact paths, tests, and stop conditions.*