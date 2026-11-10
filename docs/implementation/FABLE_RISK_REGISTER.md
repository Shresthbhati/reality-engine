# FABLE Risk Register (Deliverable G)

**Date:** 10 November 2026  
**Orchestrator:** Fable  
**Status:** Initial register — risks to be reviewed and scored at each phase gate

---

## 1. Risk Classification Scale

| Likelihood | Impact |
|------------|--------|
| Low | Unlikely to occur; minor effect if it does |
| Medium | Could occur; noticeable effect |
| High | Likely to occur; significant effect |
| Critical | Near-certain; severe effect |

**Risk score** = Likelihood × Impact (qualitative: Low/Medium/High/Critical)

---

## 2. Risk Register

| # | Risk | Category | Likelihood | Impact | Score | Mitigation / Owner | Status |
|---|------|----------|------------|--------|-------|--------------------|--------|
| RK-01 | No tape-measured rooms → accuracy claims unverified | Data | High | High | Critical | Block P1.12; capture 6 measured rooms (human field operator); do not ship accuracy numbers without them | Open |
| RK-02 | No physical VIO backend → indoor localization unverified | Technical | High | Medium | High | Keep synthetic tests as gates; label as synthetic; defer until hardware available | Open |
| RK-03 | No RGB-D device E2E → physical depth capture unverified | Technical | High | Medium | High | Synthetic RGB-D tests only; label clearly; defer device validation | Open |
| RK-04 | Blank-wall dense coverage poor | Technical | Medium | Medium | Medium | CPU plane-sweep drops textureless walls (honest refusal); do not claim coverage on blank walls | Open |
| RK-05 | Poisson meshing hangs on building-scale clouds | Technical | Medium | High | High | Depth capped at 8; threads capped at 4 (implemented); monitor benchmark runtime | Mitigated |
| RK-06 | Studio UI drift from backend state | Product | Medium | Medium | Medium | Thin-client contract (36_UI_SPEC.md); contract tests + tsc/build in CI | Open |
| RK-07 | Import-boundary regressions | Technical | Low | High | Medium | `tests/test_import_boundaries.py` in CI; facade ownership | Open |
| RK-08 | Synthetic fixtures mistaken for real data | Data | Low | High | Medium | `datasets/room_capture/` manifest + 1.0 m baseline; synthetic asserted out-of-distribution | Mitigated |
| RK-09 | Tenant authorization gaps → data leakage | Security | Medium | Critical | Critical | Cross-tenant IDOR tests for all routes; server-side authorization on every object/action | Open |
| RK-10 | Job queue lost on worker crash | Reliability | Medium | High | Medium | Stale job reaping at startup; orphan recovery; heartbeat; max_attempts | Mitigated |
| RK-11 | Concurrent commit → duplicate or lost versions | Reliability | Low | Critical | Medium | Compare-and-swap on HEAD; orphan quarantine; read-back verification | Mitigated |
| RK-12 | Billing code exists but unverified → revenue risk | SaaS | Medium | Medium | Medium | Document as production blocker; real Stripe test-mode walkthrough needed | Open |
| RK-13 | Collaboration presence in-memory only → multi-worker failure | Reliability | Medium | Medium | Medium | Document as limitation; replace with shared ephemeral backend | Open |
| RK-14 | Specialist modules presented as certified → compliance risk | Product | Medium | High | Medium | Life-safety/thermal are preliminary screening; add disclaimers; external validation needed | Open |
| RK-15 | CI not executing all tests → false confidence | Process | High | Medium | Medium | Core suite is explicit list not `pytest tests/`; ruff/mypy advisory; add specialist suites to required CI | Open |
| RK-16 | Documentation drift from code → wrong decisions | Process | High | Medium | Medium | Reconcile PENDING_IMPLEMENTATION.md, IMPLEMENTATION_STATUS.md, CAPABILITIES.md; one authoritative ledger | Open |
| RK-17 | No real measured-room accuracy → cannot prove first-customer workflow | Data | Critical | Critical | Critical | Block P1.12; requires human field captures; highest-priority unblocker | Open |
| RK-18 | Session/world boundary confusion → overwritten edits | Product | Low | High | Medium | World and Session are separate models; adoption protocol prevents silent overwrite | Mitigated |
| RK-19 | External CAD/BIM export incompatibility → customer rejection | Product | Medium | Medium | Medium | Exporters exist; external validation (Blender/Revit/AutoCAD) pending | Open |
| RK-20 | Gaussian splats presented as trained → false capability claim | Product | Low | Medium | Low | Format/conversion/streaming exist; training not established; label honestly | Mitigated |

---

## 3. Top 5 Risks (Priority for Mitigation)

| Rank | Risk | Score | Why top 5 | Immediate action |
|------|------|-------|-----------|------------------|
| 1 | RK-17: No real measured-room accuracy | Critical | Blocks first-customer workflow; cannot prove accuracy | Capture 6 measured rooms; human field operator required |
| 2 | RK-01: Same as RK-17 (duplicate) | Critical | Same | Same |
| 2 | RK-09: Tenant authorization gaps | Critical | Security breach; data leakage | Cross-tenant IDOR tests; server-side authorization |
| 3 | RK-05: Poisson meshing hangs | High | Blocks building-scale reconstruction | Already mitigated (depth/thread caps); monitor |
| 4 | RK-02: No VIO backend | High | Blocks indoor localization | Defer until hardware; synthetic gates remain |
| 5 | RK-15: CI not executing all tests | Medium | False confidence in code quality | Add specialist suites to required CI |

---

## 4. Risk Acceptance Criteria

| Risk | Acceptable when | Owner |
|------|-----------------|-------|
| RK-17 (measured rooms) | 6 diverse rooms captured with ground truth; evaluation report published | Human field operator + Fable |
| RK-09 (tenant auth) | Cross-tenant IDOR tests pass; all routes enforce authorization | OpenCode security owner |
| RK-05 (meshing hangs) | Depth ≤ 8; threads ≤ 4; benchmark shows < 5 min for building-scale | OpenCode |
| RK-02 (VIO) | Physical device tested; drift benchmark published OR explicitly deferred | FreeBuff + human operator |
| RK-15 (CI coverage) | All specialist suites either in required CI or explicitly scheduled/manual | Qwen/Nemotron |

---

## 5. Risk Review Cadence

| Review | Frequency | Owner | Inputs |
|--------|-----------|-------|--------|
| Risk register review | Weekly | Fable | New risks; mitigations completed; score changes |
| Phase gate review | At each phase exit (P0–P6) | Fable | Exit gate evidence; risk scores |
| Security review | Monthly | OpenCode | New auth findings; IDOR test results |
| Data validation review | Per capture batch | Human field operator | Measured room results; ground truth quality |

---

## 6. Risks Retired in This Session

| Risk | Reason retired | Evidence |
|------|----------------|----------|
| Poisson meshing hangs (was unmitigated) | Depth capped at 8; threads capped at 4; meshing tests pass | `tests/test_meshing_pipeline.py` 46 passed; vertical_slice.py depth=8 default |
| Import-boundary violations (48 → 4 resolved) | `engine/pipeline` facade; 47 sibling-coupling pairs removed; 7 upward pairs removed | `tests/test_import_boundaries.py` passes |
| 48 of 52 import-boundary violations resolved | Same as above | Boundary test passes |

---
*This is a living register. Scores and mitigations should be updated at each phase gate.*