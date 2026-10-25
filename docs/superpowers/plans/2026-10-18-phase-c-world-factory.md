# Phase C — World Factory (task-level plan)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use `- [ ]`.
> Parent: `2026-10-18-world-factory-and-studio-master-plan.md` (Phase C). Backlog rows: P9.1–P9.8 in `docs/implementation/PENDING_IMPLEMENTATION.md`.

**Goal:** From room → apartment → building → district, every generated thing is *parametric, re-editable and regenerated only where its inputs changed*, labelled `PROCEDURAL/GENERATED`, and exportable.

**Starting point (verified):** `procedural/room_grammar.py` (room, floor layout), `procedural/urban_grammar.py` (corridor, grid city), routes `POST /api/worlds/{id}/rooms|corridors|cities` sharing `_generate_and_commit` in `apps/api/routes_procedural.py`, Studio **Build** dialog (`BuildModal.tsx`), grammars emit the canonical Y-up frame via `_entity` (authoring is local Z-up). A generator call today is fire-and-forget: the spec is **not stored**, a name clash silently overwrites ids, and nothing can be edited afterwards.

## Global Constraints
- Same honesty rules as the master plan: everything produced here is `StatementState.PROCEDURAL` / `Provenance.GENERATED`; never presented as observed.
- Determinism: same spec + seed → byte-identical WorldIR (existing tests assert this; keep them).
- One write path: `worldstore_service.commit_version` through `_generate_and_commit`. New version per edit; stale-parent rejection preserved.
- Every task ends with: backend `pytest` for touched suites, `tests/test_application_api.py`, frontend `tsc` + `eslint` (0 errors) + `npm run test:contract`, and a browser check by the parent.

## Review Focus
1. Name collision: generating `room-1` twice must not silently merge/overwrite (C1: 409 on POST, explicit PUT to edit).
2. A dependent (corridor attached to a room) must follow when the room is resized; an *unrelated* room must keep identical bytes (scoped regeneration test).
3. Deleting a room that a corridor `connects` must be refused or cascade explicitly (never leave a dangling ADJACENT_TO).
4. Apartment layouts with infeasible constraints must return a structured "cannot satisfy: …" result, not a distorted plan.
5. 500-building districts: generation time and entity count must stay inside a measured budget (C5 gate) before any LOD claim.

---

## C1 — Stored specs + scoped regeneration (prerequisite for everything below)
**Files:** `procedural/spec_store.py` (new), `procedural/room_grammar.py`, `procedural/urban_grammar.py` (stamp specs), `apps/api/routes_procedural.py` (PUT/DELETE/GET), `tests/test_procedural_regeneration.py` (new), `tests/test_application_api.py`.
**Interfaces:**
- Produces: `GeneratedSpec{kind:"room"|"corridor"|"city", name, params:dict, seed:int, origin:[x,y,z], attach:{to:str, side:str}|None, generator:str}` stored as `Entity.custom_properties["spec"]` on the semantic entity (`{name}-room|-corridor|-city`); every geometry/child entity gets `custom_properties["generated_by"] = <semantic entity id>`.
- `regenerate(world, entity_id, new_params) -> (WorldIR, changed_ids:set[str], removed_ids:set[str])`: removes exactly the entities whose `generated_by == entity_id`, regenerates them, then re-runs dependents (specs whose `attach.to == entity_id`), nothing else.
- Routes: `GET /api/worlds/{id}/generated` (list specs), `PUT /api/worlds/{id}/rooms|corridors|cities/{name}` (edit → new version, returns `changed`/`removed`/`unchanged` counts), `DELETE …/{name}` (409 when dependents exist unless `?cascade=true`), `POST` of an existing name → 409.
- [ ] Failing tests: (a) resize room A → A's entities change, room B's geometry JSON byte-identical; (b) corridor `attach` room A east side follows the resized edge; (c) POST duplicate name → 409; (d) DELETE with dependent → 409; cascade deletes both; (e) determinism preserved.
- [ ] Implement spec stamping, `regenerate`, routes. Update `BuildModal` later in C2.

## C2 — Room Factory v2 (UI: edit, openings, templates)
**Files:** `frontend/src/components/workspace/BuildModal.tsx` (+ new `GeneratedList.tsx`), `frontend/src/lib/api/procedural.ts`, `procedural/templates.py` (new, data-only: studio, bedroom, bathroom, kitchen, living, office with default dims), tests.
- [ ] Generated-objects list in the workspace (from `GET …/generated`), select → edit form → PUT → refetch; show "N changed / M unchanged".
- [ ] Openings editor (door/window: wall, offset, width, height) writing `RoomIn.openings` (backend already validates).
- [ ] Templates dropdown fills the form (never fixed geometry).
- [ ] Browser check: create → edit width → only that room changes; reload; export glTF (existing exporter) opens.

## C3 — Apartment Factory
**Files:** `procedural/apartment_grammar.py` (new), `procedural/layout_solver.py` (new), `apps/api/routes_procedural.py` (`POST …/apartments`), `tests/test_apartment_grammar.py`, UI step in `BuildModal`.
**Interfaces:** `ApartmentSpec{name, footprint:(w,d)|None, rooms:[{id, kind, min_area, max_area, aspect_max}], adjacency:[(a,b,required|preferred)], entrance:str, seed}`; `solve_layout(spec) -> LayoutResult{ok, rects|None, violations:[str], alternatives:int}` (deterministic treemap/BSP + constraint scoring; **no ML**). `generate_apartment(spec, world)` builds rooms via `generate_floor_layout`'s shared-wall logic, cuts doors where adjacency is required, adds a CORRIDOR when a room has no direct door path to the entrance.
- [ ] Tests: all required adjacencies satisfied or `ok=False` with the unmet pairs named; every room reachable from the entrance (graph test); shared walls not duplicated; same seed → same plan; different seeds → alternatives list (≥3 when feasible).
- [ ] UI: pick rooms + adjacency, show 3 candidate plans (2D) before committing one.

## C4 — Building Factory
**Files:** `procedural/building_grammar.py`, `procedural/facade_rules.py`, tests, UI step.
- [ ] `BuildingSpec{footprint, storeys:[{height, layout:apartment_ref|template}], core:{stairs,lift}, facade:{bays, window_family, balconies}, roof}`; repeated floors reference ONE stored floor spec (C1) so editing it regenerates every floor that uses it and no other.
- [ ] Stairs: `EntityType.STAIRS` box-run between storey elevations with landing; validate headroom ≥ 2.0 m and that storeys connect (graph test).
- [ ] Facade rules → window entities as real openings in exterior walls (not decorative).
- [ ] Gate: change one floor height → only that floor + stairs + facade bays regenerate (assert unchanged entity ids by byte hash).

## C5 — District/City
**Files:** `procedural/district_grammar.py`, `procedural/road_network.py`, tests, perf benchmark in `benchmarks/`.
- [ ] Road graph (grid + arterials), parcels via block subdivision, building families with seeded variation, parks; extends `generate_city` (keep its API working).
- [ ] Detail policy per building (`footprint_only | shell | accessible_interior | hero`) stored in its spec; upgrade a building in place without touching neighbours.
- [ ] **Perf gate (measured, not assumed):** 500 buildings generate + commit in a budget recorded in `docs/BENCHMARKS.md`; viewport frame time with LOD/instancing recorded from the browser; only then claim "district scale".

## C6 — Game-ready export profile (one target)
**Files:** `exporters/godot/` (existing writer) + `exporters/profiles.py` (new), tests.
- [ ] Profile `godot_basic`: per-building collision boxes, LOD proxy (box) for `footprint_only`, scene hierarchy City→District→Building→Room, validation report listing what was dropped and why.
- [ ] Round-trip test: export → parse `.tscn` → counts and bounds match WorldIR. Target-runtime acceptance (opening in Godot) stays EXTERNAL.

## Order and gates
C1 → C2 → C3 → C4 → C5 → C6. C1 is the only hard prerequisite; C2 and C3 can run in parallel worktrees after it. Each task updates P9 rows only with evidence (tests + browser check), per the status rules.

## Self-review
Spec coverage: scoped regeneration (C1), room edit/openings/templates (C2), apartment layout with constraints and alternatives (C3), building floors/stairs/facade (C4), districts/LOD/perf (C5), game export (C6). Not covered on purpose: AI text-to-layout (Phase F), completion mode (Phase D), terrain.
