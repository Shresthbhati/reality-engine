# Room Reconstruction Fidelity — Implementation Plan

Date: 2026-10-30
Source: the user's observation register for the first testing round (geometry, visual, measurement and usability defects).
Executor: Sonnet. Reviewer: the main session plus the user. **Nothing is "fixed" until the running app shows it.**

---

## 0. Read this first (rules for the executor)

1. **Do not rebuild what exists.** Most of the pipeline is already in the repo. The defects are gaps, dropped data and
   missing wiring, not missing architecture. Before writing code for any task, find the existing module listed for it
   and extend that module.
2. **One canonical world.** Every output (wireframe, coloured view, confidence view, measurements, export) is read from
   the same compiled `WorldIR`. No view may generate its own geometry. This is the user's hard requirement.
3. **Never invent scale or data.** Decision 023 (`docs/engineering/DESIGN_DECISIONS.md`): metric scale comes only from
   evidence. Colours come from observed pixels or points. Anything estimated is labelled as estimated. If evidence is
   missing, show an honest "not measured" state. Never fill in a plausible-looking default.
4. **Evidence-based status only.** Follow CLAUDE.md §51: update `docs/implementation/PENDING_IMPLEMENTATION.md`,
   `IMPLEMENTATION_STATUS.md` and `.agent/TASKS.yaml` in the same change. Use MISSING → PARTIAL → IMPLEMENTED → VERIFIED.
5. **Verify in the browser** (`launch_ui.bat`; UI on port 3100, API on 8100) with a screenshot for every user-visible
   task. A passing build is not verification.
6. Work phase by phase. Commit per task on a feature branch. Do not merge without review.

---

## 1. What already exists (verified by code audit on 2026-10-30)

| Area | Existing code | Known state |
|---|---|---|
| Canonical pipeline | `engine/compiler/world_compiler.py`, documented in `docs/engineering/INTERIOR_COMPILER_CONVERGENCE.md` | planes → walls/floors/ceilings → rooms → windows/stairs → room semantics → WorldIR |
| Planes / walls | `perception/geometry/planes.py`, `orientation.py`, `evidence/promote_planes.py` | Walls are plane entities, not solid parametric walls |
| Rooms | `evidence/promote_rooms.py`, `perception/architecture/room_graph.py`, `topology.py` | Axis-aligned box rooms. **Known bug: `topology.py` has a second room-promotion path that creates duplicate room entities** (needs a `_find_existing_room_entity` dedupe) |
| Interior semantics | `perception/architecture/room_semantics.py` (`refine_interior_semantics`) | True footprint polygon, ceiling layers, furniture candidates, per-entity `mean_color_rgb`, `wall_color_rgb`, `floor_color_rgb` |
| Openings | `perception/architecture/openings.py`, `windows.py`, `parametric.py`, `engine/pipeline/openings.py` | Gap-in-points detection only. Thresholds are PROVISIONAL (synthetic L-room). A blank, poorly covered wall also reads as a gap. Backlog: doors/windows "exist only as room-boundary-derived openings" |
| Objects | `perception/instances/measurement.py`, room semantics furniture clustering | AABB / oriented plan box. Type `UNKNOWN` with label `furniture_candidate` |
| Scale | `reconstruction/scale.py`, `POST /api/worlds/{id}/scale`, `apps/api/jobs.py:526` | METRIC only when the operator enters a camera-to-camera baseline. Otherwise RELATIVE, so "dimensions" are in model units |
| Dense geometry | `reconstruction/dense_pipeline.py` | Per-view MiDaS mono-depth aligned to SfM. Not true MVS (backlog P0.11) |
| Viewer | `frontend/src/components/viewport/World3DViewport.tsx`, `lib/viewport/room-shell-model.ts`, `three-scene.ts` | Coloured / Wireframe / Both toggle, **shown only when a measured room shell (footprint polygon) exists**. Colours fall back to hard-coded `TYPE_FALLBACK` greys when no colour was measured. **No confidence mode.** |
| Measurement UI | `MeasurementPanel.tsx`, `measure-model.ts`, `apps/api/routes_measurements.py` | Exists |
| Backlog items | `PENDING_IMPLEMENTATION.md` P0.9, P0.11, P0.15, P0.16, P0.18, P2.17 (Material perception: MISSING), P9.7 | See each section |

**Likely root causes of the reported defects** (hypotheses that Phase 0 must confirm or reject):

- *"Walls / complete geometry not reconstructed"*: walls are rendered as thin plane patches or legacy bounding boxes. When `room_semantics` produces no footprint polygon, the viewer silently drops to boxes. Planes on sparsely covered walls are never detected.
- *"Wall colours not extracted"*: `mean_color` needs per-point colour. If the points that reach `refine_interior_semantics` carry no colour (dense mono-depth points, or a path that drops `ReconstructedPoint.color`), every surface falls back to `TYPE_FALLBACK`. That looks like "no colour extracted".
- *"Dimensions inaccurate"*: either no scale baseline was entered (the world is RELATIVE and units are not metres), or the dimensions come from axis-aligned boxes or the lowest ceiling instead of the footprint, or the outline is not regularised (walls not parallel or orthogonal).
- *"Wireframe not reliably available"*: the toggle is hidden whenever no footprint polygon exists.
- *"Confidence visualization missing"*: it is not implemented.

---

## 2. Target architecture (the single pipeline)

```
Input photos / video frames (multiple viewpoints)
  -> SfM + dense depth                      reconstruction/*
  -> metric scale (evidence only)           reconstruction/scale.py
  -> structure: planes -> walls/floor/ceil  perception/geometry/*, evidence/promote_planes.py
  -> room layout solve (NEW: regularised)   perception/architecture/room_layout.py
  -> openings (doors/windows) per wall      perception/architecture/openings.py (+ image evidence)
  -> objects (furniture/fixtures)           room_semantics.py + perception/instances/*
  -> appearance (colour/texture per surface) perception/materials/* (NEW)
  -> CANONICAL WorldIR (one compile)        engine/compiler/world_compiler.py
        every entity: geometry + dimensions{value, unit, state} + appearance + confidence + provenance
  -> ONE shell model in the viewer           lib/viewport/room-shell-model.ts -> ShellPart[]
        ├─ Wireframe  = edges of ShellPart[]
        ├─ Coloured   = ShellPart[] with measured colour/texture
        └─ Confidence = ShellPart[] coloured by confidence/state
```

Contract: the three modes differ **only in material**. They share the same `ShellPart.id` set, built once per world
version. A contract test enforces this (Task V1).

---

## 3. Phases and tasks

Each task lists: **Files**, **Do**, and **Done when** (acceptance). Run the targeted tests first, then the
broader suite.

### Phase 0 — Baseline and diagnosis (do first; no feature code)

**T0.1 Ground-truth test fixtures**
- Files: `synthetic/lroom.py`, `synthetic/indoor.py`, `scripts/generate_room_dataset.py`, `datasets/`.
- Do: create two fixtures. (a) A synthetic room with known dimensions, one door, one window, two furniture boxes and
  distinctly coloured walls. Extend the existing L-room generator; do not write a new one. (b) One real room the user
  photographs (ask the user for the photo set used in testing, plus tape-measured length, width, height, the
  door/window sizes and one camera-to-camera baseline). Store the ground truth as `ground_truth.json` next to the
  images.
- Done when: both fixtures exist, and the ground-truth schema is documented in `datasets/README` (or the existing dataset docs).

**T0.2 Fidelity scorecard script**
- Files: new `scripts/room_fidelity_report.py` (reuse `reconstruction/evaluation.py` and `world_ir/measure.py`).
- Do: compile a fixture and compare the WorldIR against ground truth. Report: wall count, and wall coverage (% of the
  ground-truth perimeter represented by a wall entity); room L/W/H error in % and cm; door/window recall/precision and
  size error; furniture recall; per-surface colour ΔE (CIE76) against ground-truth colours; scale state; % of dimensions
  that carry a measurement state. Output JSON plus a short table.
- Done when: it runs on both fixtures. The **baseline numbers are committed** to
  `docs/reconstruction/ROOM_FIDELITY_BASELINE.md`. All later phases quote before and after numbers from this script.

**T0.3 Per-stage drop-off trace**
- Do: for the user's real capture, log the counts and properties surviving each stage: points (and how many have
  colour), planes per orientation, walls promoted, rooms (and duplicates), footprint polygon present yes/no, openings,
  furniture, `*_color_rgb` present, scale state. Write it into the same baseline doc.
- Done when: every hypothesis in §1 is marked CONFIRMED or REJECTED with the evidence line. Re-order the tasks below if
  the trace shows a different main cause. Record the reason in the doc.

### Phase 1 — Structural reconstruction (P0)

**S1 Remove the duplicate room-promotion path**
- Files: `perception/architecture/topology.py`, `evidence/promote_rooms.py`, `tests/test_interior_compiler.py`.
- Do: make one room-promotion path canonical. Topology must reuse the existing room entity (match by footprint
  overlap) and never create a second one. Tighten the loose `>=` assertions in the tests to exact counts and ID
  consistency.
- Done when: room count equals the ground-truth room count on both fixtures. Tests assert no duplicate room IDs.

**S2 Room layout solver (regularised footprint)**
- Files: new `perception/architecture/room_layout.py`, called from `world_compiler.py` after room semantics.
- Do: from the footprint polygon and the wall planes, fit a closed, regularised layout. Snap edges to the dominant
  Manhattan directions when within a tolerance (keep non-Manhattan edges when the evidence disagrees). Merge collinear
  segments. Infer occluded or unobserved wall segments **only to close the loop**, and label those segments
  `provenance=INFERRED` with lower confidence. Output one ordered wall list: start, end, height, thickness (thickness:
  measured if both faces are seen, otherwise a labelled default).
- Done when: on the synthetic fixture, the layout is closed, has the correct wall count, and has ≤2 % L/W error
  relative to the scaled model. Inferred segments are flagged. Unit tests cover the rectangular, L-shaped and
  partially observed cases.

**S3 Solid wall / floor / ceiling entities in WorldIR**
- Files: `evidence/promote_planes.py` or a new promotion step, `world_ir` geometry types, `room-shell-model.ts`.
- Do: each wall from S2 becomes a WALL entity with real geometry: an extruded slab (a MESH or parametric box with
  thickness). The floor and ceiling become polygon slabs from the footprint and ceiling layers. These entities replace
  the thin plane patches as the primary representation (keep the plane entities as evidence, linked by a relationship).
  The viewer shell model renders walls from these entities, so it no longer depends on the presence of
  `footprint_polygon_m`.
- Done when: the browser shows a closed room with solid walls on both fixtures (screenshot). WorldIR validation passes,
  and the glTF export contains the wall meshes.

**S4 Openings carved into walls**
- Files: `perception/architecture/openings.py`, `windows.py`, the shell model.
- Do: attach each detected door/window to its host wall (by plane distance and in-plane bounds). Store its in-wall
  rectangle (offset along the wall, sill height, width, height). The shell model emits the wall as a polygon with
  holes (or split quads) plus frame and glass parts. Re-measure `MIN_OPENING_RING_PER_M` on the real fixture and
  record the value with its basis. Distinguish "gap = opening" from "gap = no coverage" using camera coverage: a wall
  region that no camera viewed is UNOBSERVED, not an opening.
- Done when: the synthetic door and window are recovered with ≤5 cm size error (scaled), and appear as holes in the
  walls in the browser. On the real fixture, report recall/precision honestly in the baseline doc.

### Phase 2 — Dimension accuracy (P0)

**D1 Measurement state on every dimension**
- Files: `world_ir` (dimension/property schema), `room_semantics.py`, `room_layout.py`, `perception/instances/measurement.py`, `MeasurementPanel.tsx`.
- Do: every dimension becomes `{value, unit, state, basis, uncertainty_m}` with
  `state ∈ {MEASURED (direct from metric evidence), RECONSTRUCTED (derived from geometry), ESTIMATED (inferred or
  closed by the solver), VERIFIED (user-confirmed against a tape measurement)}`. `unit` is `m` only when the world is
  METRIC, otherwise `model_units`. Reuse the existing measured/reconstructed/inferred distinction (backlog P0.16) and
  extend it; do not fork it.
- Done when: the UI shows the state badge and units next to every dimension. A RELATIVE world never shows "m".

**D2 Scale capture made unavoidable in the workflow**
- Files: `apps/api/routes_*` scale route (exists), the reconstruct / world workspace UI.
- Do: when a world is RELATIVE, show a clear call to action: "Dimensions are not in metres. Enter one known distance."
  Accept (a) the existing camera-baseline method and (b) a **new** "known length in the scene" method: the user picks
  two points or one edge in the viewer and types its real length. Method (b) feeds `reconstruction/scale.py` as a new
  `ScaleReference` method with the same median and spread gate.
- Done when: entering one real distance on the real fixture turns the world METRIC. The room L/W/H error is reported
  in the baseline doc.
- **DECIDED by the user on 2026-10-30: option 2, the engine suggests a scale and the user must accept it.**
  **D2b Scale suggestion from priors.** When a world is RELATIVE and a door, a ceiling or a standard object has been
  detected, offer a suggestion, for example: "The door is 0.93 model units tall. Typical doors are 2.0–2.1 m.
  Use 2.05 m as scale?" Show the range and which object it came from. Nothing changes until the user clicks
  Accept. An accepted suggestion becomes a `ScaleReference` with `method="prior_accepted_by_user"`, and the world's
  dimensions show state ESTIMATED (not MEASURED) until a real measurement replaces it. Rejecting it leaves the
  world RELATIVE. Record this as an amendment to Decision 023 in `DESIGN_DECISIONS.md`: scale is never applied
  without user consent, and prior-based scale is always labelled ESTIMATED.
  Done when: the suggestion appears on the real fixture, Accept turns the world METRIC with ESTIMATED dimensions,
  Reject changes nothing, and tests cover both paths plus "no door detected = no suggestion".

**D3 Dimensional consistency**
- Files: `room_layout.py`, `room_semantics.py`.
- Do: compute room L/W from the regularised layout (not the axis-aligned box). Compute room height per ceiling layer
  (not the lowest ceiling only; the old box value stays as a separate property). Opposite parallel walls must agree
  within tolerance, or the room is flagged `inconsistent` with the mismatch value. A door/window must fit inside its
  host wall.
- Done when: the consistency checks are unit-tested, and the scorecard reports them. On the synthetic fixture, L/W/H
  error is ≤2 %.

**D4 Ground-truth gate test**
- Files: `tests/` (closes backlog P0.16 "remaining" and part of P0.18).
- Do: add a committed test that compiles the synthetic fixture and asserts the dimension errors against ground truth,
  plus the existing known 0.44 m baseline assertion. Mark the real-data variant as opt-in (it needs COLMAP).
- Done when: the synthetic gate runs in the normal test suite and passes.

### Phase 3 — Objects, doors, windows, fixtures (P1)

**O1 Image-evidence opening detection**
- Files: `perception/segmentation/`, `perception/detection/`, `openings.py`.
- Do: add a second, independent evidence source: a 2D detector or segmenter labels door/window regions in the photos.
  Project them onto the host wall plane using the known camera poses. Fuse them with the point-gap evidence:
  agreement raises confidence; image-only gives a lower-confidence OBSERVED_2D hypothesis; gap-only on an observed wall
  stays as is. If no model is installed, report unavailability honestly (Decision 022). Never fall back to fake labels.
- Done when: real-fixture door/window recall improves over the Phase 1 baseline, measured by the scorecard.

**O2 Furniture categories and oriented geometry**
- Files: `room_semantics.py` (furniture clustering), `perception/instances/*`, `perception/detection/*`.
- Do: label each furniture cluster with a category from 2D detections projected into 3D (bed, sofa, table, chair,
  cabinet, …), with confidence. Unmatched clusters stay `furniture_candidate`. Replace AABB geometry with an oriented
  box fitted to the mask-constrained points (backlog P0.9 "remaining"). Add the ontology categories needed. Fixtures
  (lights, radiators, sockets) are in scope only if the detector supports them; otherwise list them as not done.
- Done when: synthetic furniture recall is 100 % with a ≤10 cm size error. Real-fixture numbers are reported. Labels
  appear in the inspector.

### Phase 4 — Material and colour extraction (P1) — spec: `docs/future/perception/MATERIAL_PERCEPTION.md`

**M1 Fix colour loss (likely the main cause of "no wall colours")**
- Do: based on the T0.3 trace, make sure point colours survive every stage (SfM points, dense points from
  `depth_to_points.py` sampled from the source image, merge/fusion). If points still lack colour, compute surface colour
  directly from the images (M2). Never leave a silent `TYPE_FALLBACK`. When a fallback is shown, the inspector says
  "colour not measured".
- Done when: the real fixture shows measured wall/floor/ceiling colours, and the ΔE against ground truth is reported.

**M2 Image-projected surface appearance**
- Files: new `perception/materials/surface_appearance.py`.
- Do: for each wall, floor and ceiling entity, project the polygon into every camera that sees it (pose, visibility and
  angle-weighted). Sample the pixels and compute a robust colour (a median in a perceptual space, with specular and
  shadow outliers rejected). Then build a **planar texture**: rectify each surface into a texture atlas from its best
  views, blend them, and mark unobserved texels. Store it as an artifact (hash-addressed, like the meshes), referenced
  from the entity's appearance property.
- Done when: the coloured view shows real wall textures on the real fixture (screenshot). Unobserved regions are
  visibly marked, not hallucinated.

**M3 Material hypotheses (minimal)**
- Do: per surface, give an evidence-backed material class (paint, wood, tile, glass, fabric, …) with confidence, only
  when a classifier is available. Otherwise state "unknown". Feed the result to the existing physics-material mapping
  (`engine/materials/`).
- Done when: P2.17 moves from MISSING to PARTIAL, with tests and registry updates.

### Phase 5 — Visualization modes (P1)

**V1 One shell, three modes (contract)**
- Files: `room-shell-model.ts`, `three-scene.ts`, `World3DViewport.tsx`, `frontend/tests/contract/room-shell-model.test.mts`.
- Do: build `ShellPart[]` once per world version. Wireframe = an edges/line material over the same parts. Coloured =
  measured colour or texture (M2). **Confidence** (new) = per-part colour ramp from entity confidence, with a hatched
  overlay for INFERRED/ESTIMATED parts and an "unobserved" colour. Always show the toggle. When no reconstructed shell
  exists, explain why (reuse `NoGeometryExplanation.tsx`) instead of hiding the control.
- Done when: a contract test asserts that the same `ShellPart.id` set and positions are used by all three modes. All
  three modes are screenshotted on both fixtures. A legend explains the confidence colours.

**V2 Inspector shows the evidence**
- Do: clicking a wall, door or window shows its dimensions with state badges, colour/material with basis, confidence,
  provenance, and which photos support it (thumbnails via the existing evidence links).
- Done when: browser-verified on the real fixture.

### Phase 6 — Usability (P2)

**U1 Guided reconstruction flow**
- Files: `frontend/src/app/(desktop)/reconstruct/page.tsx`, `ReconstructPanel.tsx`, `ReconstructionStatusBar.tsx`.
- Do: a linear, visible flow: 1 Add photos (with capture tips: cover every wall, overlap 60 %+) → 2 Reconstruct (real
  stage progress from the job) → 3 Set scale (D2) → 4 Review the room (opens the viewer with the mode switcher) →
  5 Measure / export. Each step shows done / needs action / not available, with the reason.
- Done when: a first-time user can go from photos to a measured room without docs. Verify by walking the flow in the
  browser and capturing screenshots of each step.

**U2 Results summary**
- Do: a results card per world with the room dimensions (with state), wall/door/window/furniture counts, the scale
  state, a coverage warning (unobserved walls) and a "what to capture next" hint derived from the unobserved regions.
- Done when: browser-verified. The numbers match the WorldIR and the scorecard.

**U3 Navigation clarity pass**
- Do: run the `design:design-critique` / `ui-ux-pro-max` skills on the reconstruct → world → viewer path only. Fix
  labelling, naming and dead ends there. Do not redesign unrelated screens.
- Done when: the before/after screenshots are attached to the PR.

---

## 4. Order and dependencies

```
Phase 0 (T0.1 -> T0.2 -> T0.3)   mandatory, blocks everything
  -> S1 -> S2 -> S3 -> S4         (structure)
  -> D1 -> D2 -> D3 -> D4         (D1 can start in parallel with S2; D3 needs S2)
  -> M1 (can start right after T0.3; independent of S*) -> M2 (needs S3 surfaces) -> M3
  -> O1 (needs S4), O2
  -> V1 (needs S3; texture part needs M2), V2
  -> U1 (needs D2, V1), U2, U3
```

A sensible PR split: (1) Phase 0, (2) S1–S2, (3) S3–S4, (4) D1–D4, (5) M1–M2, (6) V1–V2, (7) O1–O2, (8) M3,
(9) U1–U3.

## 5. Definition of done (whole plan)

- The scorecard on both fixtures is committed, with before and after numbers for every metric in T0.2.
- Synthetic: closed room, all walls, door and window carved, furniture boxed and labelled, L/W/H ≤2 %, colours ΔE ≤10.
- Real fixture: numbers reported honestly, whatever they are. No defect is claimed fixed without a browser screenshot
  of the running app.
- Wireframe, Coloured and Confidence are always available, read the same canonical WorldIR, and are enforced by a test.
- Every dimension shows its units and state. No RELATIVE world displays metres.
- The registry (`PENDING_IMPLEMENTATION.md`, `IMPLEMENTATION_STATUS.md`, `.agent/TASKS.yaml`) is updated in the same
  PRs. The D2b scale-suggestion decision and the Decision 023 amendment are recorded.

## 6. Open items for the user (do not guess)

1. ~~Scale from priors~~: decided, option 2 (suggest, user accepts). See D2b.
2. Provide the exact photo set used in this test round, plus tape measurements (T0.1b).
3. Which 2D detector/segmenter is acceptable to install for O1/O2 (offline model size, GPU or CPU)?

---

## 7. Further improvements (Phase 7; start only after Phases 0–5 are done)

Ordered by expected impact on the observation register. Each one uses existing code or specs where they exist.

**F1 Capture guidance (highest leverage: most defects start with bad capture)**
- Before reconstructing: check the photos for blur (Laplacian variance), exposure and overlap. Warn about photos that
  will be dropped and explain why. For video input (`routes_reconstructions.py::_video_frames`), pick sharp,
  well-spaced keyframes instead of evenly spaced ones.
- After reconstructing: a top-down coverage map that shows which walls were seen, seen weakly, or not seen, with
  "take 3 more photos of the north wall" style hints. Uploading the extra photos refines the same world (the job
  already supports `refining`).
- Done when: the scorecard shows better wall coverage on a re-capture guided by the hints.

**F2 Phone LiDAR / depth ingestion (biggest accuracy win for dimensions)**
- Spec: `docs/future/sensor-ingestion/DEPTH_INGESTION.md`; existing `evidence/depth_frames.py`, `depth_exr.py`.
- Accept iPhone/iPad LiDAR depth (or any metric depth sensor). That gives true metric scale without a tape measure
  (state MEASURED) and dense walls even where the surface is blank, which photo-only matching cannot handle.
- Done when: a LiDAR capture of the real room shows L/W/H error ≤2 % with no manual scale entry.

**F3 Better dense geometry for blank walls (backlog P0.11)**
- Plain painted walls have no texture, so photo matching finds few points there. That is the root cause of missing
  walls and false openings. Add a dense multi-view stereo backend that runs without CUDA (for example OpenMVS on CPU),
  or a learned MVS model. Keep it behind the existing backend registry (`reconstruction/backend/registry.py`), and
  add the cross-view depth consistency check.
- Done when: wall coverage on the real fixture improves measurably over mono-depth.

**F4 Learned room-layout evidence**
- Add a single-image room-layout estimator (HorizonNet/LayoutNet style: wall-floor and wall-ceiling boundary lines)
  as an extra, lower-confidence evidence source for the S2 layout solver. It helps exactly where points are sparse.
  Fuse it, don't replace: the points still win where they exist.

**F5 Lighting-independent colour**
- Photos taken under different lights give different wall colours. Normalise white balance and exposure across the
  photo set before sampling (M2). Optionally separate shading from base colour (intrinsic decomposition), so a wall
  in shadow is still reported as the same paint colour. Report the colour as "approximate under capture lighting"
  unless a colour reference card was photographed.

**F6 User corrections as evidence**
- In the viewer, let the user drag a wall edge, mark or delete a door/window, or type a real dimension. Each
  correction is stored as VERIFIED evidence with provenance `user`. It survives re-reconstruction (reuse
  `world_ir/entity_reid.py` identity carry-over) and is never overwritten by a lower-confidence automatic result.
- This turns "the engine got it wrong" into a 10-second fix and gives labelled data for tuning thresholds.

**F7 Multi-room and whole-home**
- Join rooms through their shared doorways (`room_graph.py` / BuildingGraph already model this), so a walk-through
  capture produces a connected apartment with consistent shared walls. Extends P9.7 completion mode to fill
  unobserved rooms with clearly labelled GENERATED proposals.

**F8 Useful outputs**
- A 2D floor plan export (SVG and DXF) with the dimension lines and their states. A room report (PDF) with
  area, wall areas net of openings (useful for paint and flooring estimates) and a confidence summary. Keep the
  existing glTF/Godot export in sync with the solid walls and textures.

**F9 Regression guard in CI**
- Run the synthetic fidelity scorecard in CI and fail the build if any metric gets worse than the committed
  baseline. This stops future changes from quietly undoing this work.

**F10 Speed and progressive preview**
- Show a rough room within seconds (sparse points + layout), then refine it. Measure and report the end-to-end time
  per stage in the scorecard, and set a target once the baseline is known.

Suggested order inside Phase 7: F1 → F6 → F9 → F2 → F3 → F5 → F4 → F8 → F7 → F10. F1, F6 and F9 are cheap and protect
everything else. F2 and F3 are the largest accuracy gains, but they depend on hardware or new backends.

---

## 8. Implementation status (2026-10-31)

See `docs/implementation/PENDING_IMPLEMENTATION.md` ("Room reconstruction fidelity") for the authoritative per-item
status. Summary: F1, F5, F6, F7, F8, F9 implemented; F2 implemented and verified on synthetic RGB-D (physical device
verification pending); F3 and F10 partial (CPU plane sweep and per-stage timings in place, real-fixture coverage and a
performance target still need a real capture); F4 **BLOCKED on a model decision** — the sidecar format, validation,
floor-plane projection and lower-confidence fusion are implemented, but no learned layout weights are shipped.

### F4 decision required

- **DECISION REQUIRED:** which single-image room-layout estimator to install (HorizonNet/LayoutNet weights, or another
  offline CPU-runnable model).
- **OPTIONS:** (a) vendor HorizonNet weights + inference (offline, ~100 MB, CPU); (b) LayoutNet reimplementation;
  (c) a hosted inference API (needs a service + key); (d) merge image-layout labels from an external tool as sidecars
  only (no runtime model).
- **TRADEOFFS:** local weights add repository/binary weight and a Python/torch dependency for a lower-confidence
  signal; a hosted API adds a network dependency and cost; sidecar-only keeps the engine lean and is already supported.
- **RECOMMENDATION:** sidecar-only at first (already implemented, zero new dependency); vendor a CPU model only if a
  real capture shows sparse point support that layout evidence would fix.
- **DEPENDENCIES:** the real-room photo set + tape measurements (open item 2) to measure whether F4 changes any outcome.
