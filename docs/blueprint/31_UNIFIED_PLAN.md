# 31 — Unified plan (crosswalk of every plan into one queue)

Written 2026-11-06 at `b6a59514`. **[30_BUILD_PLAN.md](30_BUILD_PLAN.md) remains the only work queue.**
This file maps the other plans onto it, so no item exists in two places.
New work found here was added to 30 as tasks 8.6–8.8, U.1–U.8 and Phase 13.

| Source plan | Date / baseline | What became of it |
|---|---|---|
| Master Development Plan (Phases 0–9, sprints S0–S12) | 2026-10-28 @ `1d872fe` | Superseded by 30 for order. Its uncovered capabilities are now **Phase 13** |
| ML Reconstruction, Overlap & Export plan | 2026-11-01 | Became 30 Phases 1–10 almost one-to-one (observations, association, matcher, depth, coverage, capture requests, completion) |
| 30 Build Plan (download copy) | 2026-11-01..04 | Older copy of the repo file. Its 6.2–6.4 DONE rows are confirmed by merged PRs #220–#222 |
| Scene-complete reconstruction ([spec](../future/perception/SCENE_COMPLETE_RECONSTRUCTION.md)) | 2026-11-06 | **Tasks 8.6–8.8** (wire detection into the product path) |
| UI/UX research + redesign guidelines (pasted 2026-11-06) | — | **Tasks U.1–U.8**; existing 1.4 and 11.4 are part of this track |

## 1. Crosswalk

### Master plan → build plan

| Master plan item | Where it lives now | State |
|---|---|---|
| 0.1–0.8 baseline, branch audit, CODEOWNERS | 30: 0.1–0.7 | done except branch audit / CODEOWNERS → 13.1 |
| 1.x technology acquisition, licence registry | 30: 4.1 (registry + `.agent/LICENSES.yaml`) | DONE |
| 2.2/2.3 video keyframes | 30: 9.2 | DONE |
| 2.6 pHash dedup, 2.7 session grouping | lifecycle doc: code exists, **no caller** | → 13.2 |
| 2.1 HEIC, 2.4 360°, 2.8 point-cloud, 2.9 mesh ingestion | not in 30 | → 13.3 |
| 3.x LightGlue matching, scene grouping | 30: 4.3 | DONE |
| 4.2 Depth Anything V2, 4.3 scale reconciliation | 30: 4.4; P0.4 | DONE / IMPLEMENTED |
| 4.4–4.8 dense cloud, Open3D, meshing | P0.11, P1.11, P1.12/13 | IMPLEMENTED (Open3D not installed; trimesh used) |
| 5.1 PolyLayout, 5.3 DeepLSD, 5.8 non-Manhattan | not in 30 | → 13.4 |
| 5.5 doors/windows, 5.6/5.7 topology + storeys | P2.16 (BLOCKED on real indoor data), 30: 5.2/5.4 | partial |
| 5.10 / 9.4 floor plan SVG/PDF | not in 30 | → 13.5 |
| 6.1 SAM 2 | not installed (`segment_anything` v1 only) | → 13.6 |
| 6.2/6.3 fixtures + furniture detection | 30: 8.2, 9.1 — **components DONE, not called by pipeline/API** | → **8.6–8.8** |
| 6.4–6.6 identity, 3D bounding, attachment | 30: 3.x, 8.3, 8.4 | DONE except 3.6 |
| 6.7–6.9 outdoor, terrain, facades | not in 30 | → 13.7 |
| 6.10/6.11 materials, textures | 30: 10.1, 10.2; P2.17 MISSING | 10.x DONE; P2.17 spec open |
| 7.x WorldIR, validation, jobs, security | 30: 3.4, 5.1, 1.6; P0.17 | DONE |
| 8.x Reality Studio | 30: 3.5, 7.2, 7.3, 8.5, 6.4 | DONE; redesign → U track |
| 9.1–9.6 exports | 30: 1.5, 11.2; P0 exporters | DONE (USDA, not binary USD) |
| 9.7 golden datasets, 9.12 acceptance | 30: 2.3 (BLOCKED), 12.1–12.3 | open |

### ML overlap plan → build plan

| ML plan item | Build plan | State |
|---|---|---|
| §1 export fix | 1.5 | landed `13b9a533` |
| Phase 1 ledger, Hungarian association, no transitive merges | 3.1–3.4 | DONE |
| Adversarial scenarios A1–A20 | 2.2 skeleton DONE; **3.6 open** | open |
| Phase 2 matcher / metric depth | 4.3, 4.4 | DONE |
| Phase 3 cross-session identity | 6.2, 6.3, PROD.2 | landed |
| Phase 4 coverage + capture requests | 5.3, 6.1, 6.4 | DONE / landed |
| Phase 5 constrained completion | 10.3 | DONE |
| Phase 6 appearance (ΔE colour) | 10.1 partly | colour accuracy → 13.8 |
| MASt3R (NC-gated), DINOv2 re-ID | not adopted | → 13.9 |
| §6 "show, don't tell" UI | 7.2, 6.4 + U track | partly |

## 2. Remaining work — counts (2026-11-06)

**Build plan (59 tasks before this change):** 43 DONE in the file. 10 TODO / IN PROGRESS rows already have
merged commits but were never marked (0.1, 0.6a, 0.6c, 1.5, 1.6, 1.7, 6.2, 6.3, 6.4, 11.4); they are now
marked `LANDED (<commit>), not re-verified`. **Truly open: 1.4, 3.6, 12.1, 12.2, 12.3, plus 2.3 BLOCKED on your room captures.**
This change adds 3 + 8 + 9 = 20 tasks (8.6–8.8, U.1–U.8, 13.1–13.9).

**PENDING_IMPLEMENTATION.md (48 headed items):**

| Status | Count | Meaning for remaining work |
|---|---|---|
| VERIFIED / DONE | 18 | finished |
| IMPLEMENTED | 16 | works and is tested; still needs real-data verification to reach VERIFIED |
| PARTIAL | 6 | P0.15 relationships, P0.18 end-to-end (mesh leg), P1.5 IMU/GNSS, P4.21 incremental, P7.29 CI, PROD.1 one-action reconstruction |
| BLOCKED | 3 | P1.4 RGB-D (hardware), P1.8 VIO (no backend), P2.16 doors/windows/stairs (real indoor data) |
| MISSING | 5 | P2.20 scene-complete (new), P2.17 materials, P7.25 GIS, P7.26 robotics, P8.31 simulation (deliberately deferred) |

Its tables hold ~40 more rows: 8 IMPLEMENTED, 23 PARTIAL (room-fidelity S/D/M/U rows, P9.1–P9.9 "API only, no UI", CE-01..04),
3 EXPERIMENTAL, 1 FOUNDATION (SITE-1), 1 BLOCKED (O1/O2 detector decision), 1 not implemented (CE-11), PT-02 deferred, PT-06 open.
Housekeeping: P4.21 and P7.29 each appear twice with different statuses.

**Bottom line:** ~38 % of headed backlog items are finished, ~33 % work but are unverified on real data, ~29 % are partial, blocked or missing.
The biggest single gap is that **real indoor data has never gone through the full path**: 2.3, P2.16, P0.18 and 12.1 all wait on it.

## 3. Unified order of work

1. **Reconcile** (docs only): re-verify the 10 LANDED rows; de-duplicate P4.21 / P7.29.
2. **8.6–8.8 Wire perception into the product path.** This is the furniture fix: the eval set, then single-image M1, then open-vocab + crop pass in the API path.
3. **3.6** identity scenarios.
4. **2.3** your 3 tape-measured rooms (only you can do this) → unblocks 12.2 and P2.16.
5. **U.1–U.8** UI track (U.1 audit first; 1.4 and 11.4 fold in).
6. **12.1–12.3** v1 acceptance.
7. **Phase 13** expansions, after v1.
