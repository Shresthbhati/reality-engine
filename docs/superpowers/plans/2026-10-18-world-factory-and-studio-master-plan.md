# Reality Engine — World Factory + Studio Master Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use `- [ ]`.
> This is a *master* plan. Phases A and B are task-level. Phases C–G are outlined and each gets its own task-level plan when its predecessor's gate is green (superpowers:writing-plans scope rule: one plan per independent subsystem).

**Goal:** Finish Reality Engine as one product: photos (or a description) in → an editable, evidence-labelled 3D world (room → apartment → building → district → city) out, inside a Studio that looks and behaves like the reference images.

**Spec:** the five vision documents supplied 2026-10-18 (Product Vision, Spatial UI/UX, World Factory, Photo-to-World, Complete Vision) + reference images in `C:\Users\shres\Downloads\reference` (~125 screens). They agree on one core; this plan implements that core and defers the rest explicitly.

## 0. Where the project really is (observed, not assumed)

| Area | State | Source |
|---|---|---|
| Reconstruction engine (COLMAP, scale, depth, fusion, mesh, WorldIR, WorldStore, exporters incl. glTF/USD/IFC/CityGML/Godot/Unreal/ROS/SUMO/Habitat) | **Built and verified** on real photos (South Building) | `docs/implementation/IMPLEMENTATION_STATUS.md`, `.agent/EXECUTION_STATE.md` (2776 tests passing 2026-10-03) |
| Task ledger | DONE 36 / PARTIAL 1 / BLOCKED 6 / MISSING 0 | `.agent/EXECUTION_STATE.md` |
| Launcher | Was broken (no API, browser opened early); **fixed + verified** this session | `launch_ui.bat`, commit 3145271 |
| Creation ("World Factory") | Room: API + grammar existed but **no UI**. Corridor + city grammar, API, Build dialog: **added this session**, 74 Python tests pass, browser-verified | `procedural/`, `BuildModal.tsx` |
| Studio UI | Desktop (33 pages) + mobile exist; **not yet compared against the reference images** | `frontend/src/app` |
| Blockers outside code | No real corridor/room photo dataset; no GPU for dense MVS by default; VIO backend; 3DCityDB runtime | `docs/KNOWN_LIMITATIONS.md` |

**Diagnosis of "it's not finishing":** the engine is far along. What is missing is (1) a usable creation path, (2) a Studio that matches the target and wires every capability to a screen, (3) one rehearsed end-to-end journey, and (4) a single source of truth for scope — five overlapping vision documents plus `ROADMAP.md`. This plan fixes those four, in that order of leverage.

## Global Constraints
- `docs/implementation/ROADMAP.md` stays the authoritative WHAT (project rule). Everything here is **additive**: Phase C/D (World Factory) is not in the roadmap today and is registered as **P9 — World Factory** in `PENDING_IMPLEMENTATION.md` (Task A5) instead of silently reordering P0–P8.
- Honesty rules from the vision docs are acceptance criteria, not style: never show success without verified success; generated content is labelled `GENERATED/PROCEDURAL`, never `OBSERVED`; unknown stays unknown; no fabricated geometry/progress/thumbnails.
- Status moves only with evidence (MISSING→FOUNDATION→PARTIAL→EXPERIMENTAL→IMPLEMENTED→VERIFIED). Update `.agent/TASKS.yaml` + `EXECUTION_STATE.md` in the same change.
- Reuse before creating: `procedural/room_grammar.py`, `procedural/urban_grammar.py`, `apps/api/routes_procedural.py::_generate_and_commit`, `worldstore_service.commit_version`, existing exporters, existing `frontend` design tokens.
- Frontend gate each task: `npx tsc --noEmit`, eslint on touched files, `npm run test:contract`. Backend gate: `pytest` on touched suites + `tests/test_application_api.py`.

## Review Focus (likely to bite, not yet tested)
1. Windows first-run: no `.venv`, Python 3.14, path with spaces/OneDrive — launcher must give an actionable error (Task A2).
2. Procedural worlds have no point cloud: any viewport filter tuned for photo clouds can hide real geometry (the oversized-filter bug found this session; add a regression test, Task A3).
3. Z-up data vs Y-up viewport: rooms/streets render rotated (pre-existing). Decide the axis convention once (Task B1) or every new generator inherits the confusion.
4. Re-running a generator with the same name overwrites ids silently; need collision handling + scoped regeneration (Task C1).
5. A 100-lot city is ~700 entities; city-scale needs LOD/streaming before anything bigger (Phase C gate).

---

## Phase A — Make it solid (days)  *Gate: fresh clone → double-click → working app*
- [ ] **A1** Commit the launcher + builder work (done: 3145271). Push branch, open PR. *Verify:* CI green.
- [ ] **A2** Launcher hardening: detect missing `.venv`/packages/Node<20 and print the exact fix; add `launch_ui.bat` section to `QUICKSTART.md`. *Test:* run on a machine/dir without `.venv` → readable error, exit non-zero.
- [ ] **A3** Regression tests: (a) viewport shows 20 m entities when no point cloud (extract the `isOversized` rule to `frontend/src/lib/viewport/oversized.ts` + node test), (b) `/corridors` and `/cities` collision behaviour.
- [ ] **A4** Fix the 10 pre-existing frontend lint errors (EXECUTION_STATE) so the gate can be strict.
- [ ] **A5** Scope reconciliation: add **P9 World Factory** and **P6b Studio parity** rows to `PENDING_IMPLEMENTATION.md`/`ROADMAP.md`; write `docs/product/VISION.md` = the one-page merge of the five documents (core promise, three modes Reconstruct/Complete/Create, honesty rules, non-goals) and mark the originals as background.

## Phase B — Studio parity with the reference images (the biggest visible gap)
Map each reference screen to a route and close the gap, one slice at a time (screens observed; others in the folder follow the same patterns):

| Reference screen | Target route/component | What it must do |
|---|---|---|
| Home ("A Clearer Tomorrow") | `(desktop)/page.tsx` | recent worlds from real data, quick start, activity; no fake stats |
| **World Intent wizard** (scale → capabilities → resource profile → Run) | `worlds/new` + `sessions/new` | pick Interior/Building/Street/…/City; capability checkboxes; resource profile **computed from real dataset size**, never invented |
| Asset inspector (frame, quality, spatial value, ACCEPTED/REJECTED + reason) | `evidence/[id]/view` | uses real `evidence/contribution.py` + `image_check.py` output |
| Measurement mode (distance/height/area/volume, ± uncertainty, evidence list) | tool inside `SpatialWorkstation` | scale-aware: refuse metres when scale is unknown |
| Plot / map 3D view (layers, legend, evidence strip, plot info) | `sessions/[id]/map` + world map layer | GIS from `exporters`/`city_osm` data only |
| Dense 3D city + properties/insights/AI tabs | `worlds/[id]` | LOD + layer panel; AI tab disabled until Phase F |
| Pipeline graph (stage nodes, status, confidence, WORLD CONSTRUCTED) | bottom diagnostics / `analysis/[id]` | driven by real job stages (`apps/api/jobs.py`), failed ≠ succeeded |

- [ ] **B0 Gap audit (1 task):** screenshot each live route at 1536×1024, put beside its reference in `docs/screenshots/parity/`, list concrete deltas. *Output:* a checked table; drives B2+.
- [ ] **B1 Axis + units decision:** one convention (recommend ENU z-up: convert in the viewport loader, not in data) so rooms/streets render upright; update `three-scene.ts`, add a render test with a known box. Record in `DESIGN_DECISIONS.md`.
- [ ] **B2 App shell:** top mode bar (Explore/Inspect/Edit/Capture/Build/Review), left world navigator that lists only real entities, right context inspector, bottom pipeline inspector; spatial-scale breadcrumb (Room→…→City).
- [ ] **B3 Intent wizard**, **B4 Asset inspector**, **B5 Measurement + uncertainty**, **B6 Map/plot**, **B7 Pipeline view** — each: failing UI/contract test → implement → screenshot vs reference → commit.
- [ ] **B8 Mobile capture** parity (guided capture screens in the reference set).
*Gate B:* every reference screen has a live equivalent with real data or an honest empty/unavailable state; Playwright smoke over all routes passes.

## Phase C — World Factory: creation (rooms → city)
*Needs its own plan; outline and exit gates:*
1. **C1 Scoped regeneration:** generator specs stored on entities (`custom_properties.spec`), dependency tracking, edit parameters → regenerate only dependents. (Fixes Review Focus #4.)
2. **C2 Room Factory v2:** openings UI (doors/windows), room templates, variants. *Gate:* create→edit→save→reload→export.
3. **C3 Apartment Factory:** spatial program → adjacency graph → layout generator (constraint solver) → shared walls/doors; alternatives side by side.
4. **C4 Building Factory:** floor reuse, stairs/lift, facade rules, roof; scoped regen across floors.
5. **C5 District/City:** roads, parcels, building families, districts, detail policies, spatial partitioning + LOD (extends today's grid city). *Gate:* 500-building district generates and navigates within a measured frame/memory budget.
6. **C6 Game-ready export profile:** pick one target (recommend Godot — writer exists), add collision + LOD + validation report.

## Phase D — Reconstruction ↔ creation meet
Completion mode: reconstructed shell + labelled `GENERATED` interior proposals, accept/reject UI, existing-vs-proposed diff. Uses WorldIR provenance enum already present.

## Phase E — Real data & benchmarks (needs you)
Capture 2–3 real room/corridor/apartment photo sets with rights; register in `datasets/`; unblocks P7-03/P7-05; first real accuracy numbers.

## Phase F — Intelligence
Tool-based assistant (query world, explain evidence, propose edits through validated operations), semantic search, capture guidance (builds on `engine/pipeline/guidance.py`), controlled self-correction loop (already partly built).

## Phase G — Release gate
Auth/limits/recovery per `Appendix A`, acceptance journey automated: photos → reconstruct → WorldIR → commit → restart → correct → diff → export.

## Decisions I need from you (everything else I decide)
1. **First user/workflow** (recommend: "photos of a building/room → editable model + export"; creation tools second).
2. **Fidelity to the reference UI:** exact visual match, or same layout/behaviour with the existing design tokens (recommend the latter for speed).
3. **Game target:** Godot (writer exists) vs Unreal vs Unity.
4. **GPU:** may dense reconstruction be on by default on your machine?
5. **Real datasets:** can you capture the room/corridor sets in Phase E?

## Self-review
- Spec coverage: reconstruction (built), evidence/uncertainty (B4/B5), Room/Apartment/Building/City Factory (C), completion mode (D), game export (C6), assistant (F), release gates (G), UI parity (B). Not covered on purpose: multi-user cloud, pricing, BIM-certified output, structural analysis (vision docs themselves say do not promise).
- Not yet verified: reference-vs-live deltas (B0 will measure them); claims about the 33 pages' content.
- Proportion: Phase A/B are task-level; C–G intentionally outline-only.
