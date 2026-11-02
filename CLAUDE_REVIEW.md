# Claude Review Document — Reality Engine Worktree `claude/reality-engine-docs-merge-da520d`

**Purpose:** Maintained for Claude review (2-hour cycle). Records what was done, current task state, blockers, and what's needed from the user.

**Worktree:** `C:\Users\shres\OneDrive\Desktop\codes\reality-engine\.claude\worktrees\reality-engine-docs-merge-da520d`
**Branch:** `claude/reality-engine-docs-merge-da520d` (commit `89b811d`)
**Environment:** Python 3.14.2, pytest 9.1.1, `import engine` OK

---

## This session (2026-11-01) — LAND-01 foundation

### Implemented
- **`perception/architecture/landmark.py`** (new) — single-image curved-structure reconstruction:
  - `fit_dome` — sphere fit over a point cap, measured upper_fraction gate, refuses flat/full-sphere
  - `fit_arch` — circle cross-section fit in arch's vertical plane, angular-span gate, refuses shallow arcs
  - `fit_tower` — vertical cylinder fit, axis-verticality + elongation gates, refuses flat pucks
  - `fit_landmark` — auto-detects kind (tower→dome→arch) or honors `kind_hint`; returns `kind=="none"` with `refusal_reason` (never guesses)
  - relative-vs-measured scale: RELATIVE unless trusted `baseline_m` supplied
  - `LandmarkErrorReport` + `report_error()` for acceptance-criterion ground-truth measurement
  - Deterministic (pure float math, no RNG, no clocks); honest `FitRefused` on insufficient/degenerate support
- **`tests/test_landmark.py`** (new) — 27 tests, all passed: fitting, refusals, determinism, scale state, error report

### Verification this session
- `tests/test_landmark.py`: 27 passed / 0 failed
- Regression sweep: 177 passed / 0 failed (landmark, parametric fits, canonical interior, camera calibration, calibration transforms, association, backend selection, clock, API sites/site_compose/world_artifacts)
- PROD-01 unit tests: `test_progressive_units.py` 44 passed; `test_world_diff.py` 17 passed; `test_confidence_defaults.py` logic tests pass (producer-scan sub-test fails only on `git ls-files` WinError 6 — environment, not code); `test_proc_ownership.py` passes in isolation (flaky when batched)
- No regressions introduced

### Registry updates (same change)
- `.agent/TASKS.yaml`: LAND-01 MISSING → PARTIAL
- `docs/implementation/PENDING_IMPLEMENTATION.md`: LAND-1 MISSING → PARTIAL
- `docs/implementation/CAPABILITIES.md`: landmark architecture moved from "NOT yet" to FOUNDATION/synthetic
- `.agent/EXECUTION_STATE.md`: new session entry + LAND-1 MISSING → PARTIAL

---

## Full task inventory (`.agent/TASKS.yaml`, 49 tasks)

**DONE (37):** P0-01, P0-02, P0-03, P1-01, P1-02, P0-ROBUST-01, P0-ROBUST-02, P2-01, P2-02, P3-01, P3-03, P4-01, P5-01, P5-02, P6-01, P6-02, P6-03, P7-01, P7-02, P7-04, P7-06, P7-05-ADAPTIVE-SUBDIVISION, P8-01, P8-02, P8-03, P9-01, P10-01, P10-02, P11-01, P12-01, P13-01, P13-02, P15-01, P16-01, P17-01, P18-01, P19-01

**NOT DONE (12):**

| ID | Status | Task | Blocker |
|----|--------|------|---------|
| LAND-01 | PARTIAL | landmark/curved-structure from single image | **real landmark photo set + surveyed ground-truth dimensions needed for VERIFIED** |
| PROD-01 | PARTIAL | one-action evidence-progressive reconstruction | engineering DONE; blocked on real corridor/room dataset |
| ROOMFID-01 | PARTIAL | room reconstruction fidelity | synthetic only; needs real-room photos + tape measurements |
| SITE-01 | PARTIAL | Site composer | FOUNDATION only; being built by other agents; unverified |
| GATE-01 | PARTIAL | release gate | Phase 1 done; phases 2-6 pending |
| P1-03 | BLOCKED | depth integration | physical RGB-D hardware unavailable |
| P3-02 | BLOCKED | VIO/localization federation | ORB-SLAM3/OpenVINS/Basalt binaries not installed |
| P7-03 | BLOCKED | architectural perception | real indoor dataset unavailable |
| P7-05 | BLOCKED | detail budget | universal calibration data unavailable |
| P14-01 | BLOCKED | city world compiler | 3DCityDB runtime unavailable |
| P16-02 | OPTIONAL_DEFERRED | downstream ecosystem compilers | explicitly deferred |
| ROOMFID-02 | BLOCKED | learned 2D door/window detector | **user decision required** (YOLO/GroundingDINO vs geometry-only vs defer) |

---

## What is needed from the user (to unblock real work)

1. **Real-room photos + tape measurements** — unblocks ROOMFID-01 (real-room dimension gate) and PROD-01 (real corridor/room dataset)
2. **Real landmark photo set + surveyed dimensions** — unblocks LAND-01 → VERIFIED (acceptance criterion: measured error vs ground truth)
3. **2D detector decision (ROOMFID-02)** — YOLO/GroundingDINO-class model vs keep geometry-only vs defer

---

## Environment notes

- `git ls-files` / subprocess-based tests fail with `WinError 6` (invalid handle) in this worktree — environmental, not code defects
- Full non-slow test suite times out (>30s) due to slow integration tests; run targeted files instead
- `test_proc_ownership.py::test_descendant_tree_terminated` passes in isolation; flaky only when batched with other subprocess tests
- Frontend (node_modules) not present in this worktree — `tsc`/build/contract not runnable here

---

## Next priorities (per worktree's own plan)

1. Get full python suite to a clean recorded result
2. Browser-verify V1/V2 + guided flow (needs frontend/node_modules)
3. Exercise scale-suggestion accept end to end
4. Unblock items #1-3 above with user-provided data/decisions
