# Room fidelity baseline (plan T0.1 - T0.3, S1)

Date: 2026-10-31. Plan: `docs/superpowers/plans/2026-10-30-room-reconstruction-fidelity.md`.
Machine-readable baseline: `docs/reconstruction/room_fidelity_baseline.json` (gate: `--fail-on-regression`).

## What this is, and what it is not

Every number below comes from ONE synthetic fixture (`fidelity_room`: 5.0 x 4.0 x 2.6 m, one door, one window, a table and a
wardrobe, six distinctly coloured surfaces) pushed through the real canonical compiler
(`engine.compiler.compile_reconstruction_to_world`, production `CompileOptions`). The ground truth is exact because the
generator made it. **No real-room number exists yet**: that needs the user's photos and tape measurements
(`datasets/README.md`, "real_room"). Nothing here proves anything about a real capture, and the user's reported defects
are therefore mostly **NOT TESTABLE** below rather than confirmed or rejected.

Reproduce (headless, no COLMAP, about 10-40 s per compile):

```
python scripts/room_fidelity_report.py                                   # table, baseline config
python scripts/room_fidelity_report.py --json out.json                   # full machine-readable report incl. stage trace
python scripts/room_fidelity_report.py --baseline docs/reconstruction/room_fidelity_baseline.json --fail-on-regression
python scripts/room_fidelity_report.py --sweep 0.04,0.06,0.08,0.10,0.12  # density sensitivity
python scripts/room_fidelity_report.py --cameras one | --colors rendered | --colors none | --promote-building
python scripts/room_fidelity_report.py --fixture real                    # every metric "not available" (no fixture)
```

Baseline config: `colors=albedo, voxel_m=0.08, cameras=all (3 standing positions, 72 views), promote_building=False, seed=42`.
Why 0.08: it keeps the compile to about 10 s for CI. **The result depends on the density** (sweep below), so this baseline
describes this density only.

## Baseline numbers (`room_fidelity_baseline.json`)

| metric | value | ground truth / note |
|---|---|---|
| ROOM entities | 1 (error 0), duplicate ids 0, duplicate-room pairs 0 | 1 room |
| walls | 4, perimeter coverage 100 % | 4 walls |
| room length / width / height error | 0.003 % / 0.001 % / 0.004 % (0.01 / 0.006 / 0.009 cm) | 5.0 / 4.0 / 2.6 m, from `floor_dimension_u/v_m`, `height_m` |
| footprint polygon area error | 1.79 % (19.64 m2) | 20.0 m2 - **unstable, see sweep** |
| door recall / precision / size error | 1.0 / 1.0 / width 5.0 cm, height 5.0 cm | 1 door 0.9 x 2.05 m - **lost at voxel 0.04** |
| window recall / precision / size error | 1.0 / 1.0 / width 0.0 cm, height 5.0 cm | 1 window 1.2 x 1.1 m |
| furniture recall / precision | 1.0 / 1.0 | 2 items |
| furniture size error (max over axes) | 12.0 cm (table 120 x 80 x 65 cm vs 120 x 80 x 75; wardrobe 57 x 57 x 168 vs 60 x 60 x 180) | heights are short by 10-12 cm: the base is taken 0.12 m above the floor |
| colour dE76, albedo points | floor 0.49, ceiling 0.0, walls S 0.44 / E 0.0 / N 0.59 / W 0.0 | per-surface `mean_color_rgb` vs ground-truth colour |
| room `floor_color_rgb` / `wall_color_rgb` present | yes / yes (wall colour absent at voxel 0.04) | |
| scale state | **not available** | `compile_reconstruction_to_world` records none (only `engine/pipeline/vertical_slice` sets `world.metadata['scale']`); treat as RELATIVE (Decision 023). The cm figures above are meaningful only because the generator is metric by construction |
| dimensions that carry a measurement state | **0 %** | all dimension properties are bare floats (plan D1 not started) |

`reconstruction/evaluation.py` (pose/point-cloud evaluation) and `world_ir/measure.py` (saved-measurement maths) were
reviewed and do not fit this comparison (neither compares a WorldIR with a room ground truth); the script computes its own
CIE76 and plan-distance matching. `world_ir.measure.unit_for` is the hook to use once a scale state exists.

### Colour source matters

| point colour | dE76 floor / ceiling / S / E / N / W | entities with `mean_color_rgb` |
|---|---|---|
| `albedo` (surface base colour) | 0.5 / 0.0 / 0.4 / 0.0 / 0.6 / 0.0 | 8 |
| `rendered` (lambert-shaded pixel) | 13.2 / 26.6 / 20.7 / 8.3 / 8.0 / 17.4 | 8 |
| `none` (no per-point colour) | not available x6 | **0** |

The pipeline averages whatever colour the points carry. Shaded pixels therefore give a lit colour, not the paint colour
(dE 8-27). Recovering true surface colour from lit photos is material perception (backlog P2.17), not a compile bug.

## Density sensitivity sweep (same scene, same cameras; only the cloud voxel changes)

| voxel m | points | floors | doors | windows | door recall | footprint area err | wall colour present |
|---|---|---|---|---|---|---|---|
| 0.04 | 97 015 | 1 | **0** | 1 | **0** | **16.4 %** | **no** |
| 0.06 | 35 226 | **2** | **0** | 1 | **0** | 7.1 % | yes |
| 0.08 (baseline) | 22 063 | 1 | 1 | 1 | 1 | 1.8 % | yes |
| 0.10 | 17 068 | 1 | 1 | 1 | 1 | 0.4 % | yes |
| 0.12 | 8 952 | 1 | 1 | 1 | 1 | 1.2 % | yes |

Walls (4), rooms (1), room L/W/H error (< 0.05 %), window and furniture recall are stable across all five densities.
A single standing position (`--cameras one`, 17 971 points) still gives 4 walls, 100 % perimeter coverage, a footprint
polygon, door and window, footprint error 0.6 %, furniture size error 16.8 cm.

## T0.3 stage trace (baseline config; surviving counts per stage)

| stage | count / property |
|---|---|
| points / with colour | 22 063 / 22 063 (albedo); `none` run: 22 063 / 0 |
| points in planes | 22 063 (100 %) |
| planes by role | floor 2, wall 10, ceiling 1, unknown 1 (13) |
| promoted to entities | walls 4, floor 1, ceiling 1; 8 planes unpromoted (furniture tops / sub-room sheets, correctly refused) |
| rooms / duplicates | 1 / 0 |
| footprint polygon | present, 4 vertices |
| openings | door 1, window 1, generic 0 |
| furniture candidates | 2 |
| entities with `mean_color_rgb` | 8 (floor, ceiling, 4 walls, 2 furniture) |
| room `floor_color_rgb` / `wall_color_rgb` | present / present |
| scale state | not recorded by the compile |
| interior warnings | none (but see D2 with `promote_building`) |

## Caveat: the tree changed under this measurement

Other agents edited `engine/compiler/world_compiler.py`, `perception/architecture/openings.py`, added `wall_slabs.py` /
`room_layout.py` etc. in the same working tree while this was being measured. The sweep rows and the baseline JSON were
taken on the code as it stood at the time (HEAD plus those edits as they landed); I could not make a clean-HEAD run (a
detached worktree was denied). Evidence of drift: a later 0.06 run showed 8 wall and 3 floor entities (a `floor-slab-*`
entity now exists), unlike the 4/2 recorded in the sweep. The baseline config (0.08) was re-run on the final tree and still
passes the gate (`regression gate: pass`). Re-take the sweep after those changes merge.

## Hypotheses from plan section 1 ("Likely root causes")

Verdicts are against the synthetic fixture only. "NOT TESTABLE" means the failing condition cannot be produced or observed
without the user's real capture.

| # | hypothesis | verdict | evidence |
|---|---|---|---|
| H1a | Walls / complete geometry not reconstructed (sparse walls never get a plane) | **REJECTED on synthetic; real sparse walls NOT TESTABLE** | 4/4 walls, 100 % perimeter coverage at every density 0.04-0.12 and from a single standing position. No real wall with sparse coverage was available |
| H1b | Walls are thin plane patches, not solid walls | **CONFIRMED** (code + data) | wall entity geometry is a ~2 cm slab of plane inliers (e.g. y-extent -0.003..0.019 m); no thickness model exists (plan S2/S3 not done) |
| H1c | Viewer silently falls to legacy boxes when no footprint polygon | **NOT TESTABLE** | the polygon is present in every synthetic run; the gate is real in code (`room-shell-model.ts` returns no parts without `footprint_polygon_xyz_m`, at HEAD 1d872fe) but the missing-polygon condition was not reproduced |
| H2 | Wall colours not extracted because no per-point colour reaches `refine_interior_semantics` | **CONFIRMED as mechanism; cause in the real capture NOT TESTABLE** | with colour: all 6 surfaces dE < 1.1. With none: 0 coloured entities and no room colours (colourless in, colourless out, never an invented colour). Code audit: `.color` is propagated by colmap_backend, dense_output, depth_to_points, frame, merge, orchestrator, scale; it is **dropped by `engine/pipeline/single_image.py`** (monocular blob). Whether the user's points carry colour is unknown |
| H3a | Dimensions inaccurate because the world is RELATIVE | **NOT TESTABLE** for the user's world; mechanism **CONFIRMED** | compile output has no scale state and 0 % of dimensions carry a state or unit. The user's actual scale state needs their capture |
| H3b | Dimensions come from axis-aligned boxes / the lowest ceiling | **REJECTED for a single-layer box room; false ceiling NOT TESTABLE here** | L/W/H error < 0.01 %. The two-layer L-room (`synthetic/lroom.py`) needs COLMAP and was not run |
| H3c | Outline not regularised | **PARTLY CONFIRMED** | `floor_dimension_u/v_m` (ring-based) are exact, but the footprint polygon area is wrong by 0.4-16 % depending only on cloud density (above): the polygon, not the ring, is the unstable quantity |
| H4 | Wireframe not reliably available (toggle hidden without a footprint polygon) | **NOT TESTABLE** (code gate confirmed, trigger not reproduced) | `World3DViewport.tsx` at HEAD shows the Coloured/Wireframe/Both toggle only when a measured room shell exists; polygon was present in all synthetic runs |
| H5 | Confidence visualisation missing | **CONFIRMED at HEAD 1d872fe** | no confidence render mode in `room-shell-model.ts` / `World3DViewport.tsx` at that commit (another change is adding one in the working tree; not evaluated here) |

Task order: the trace does not change the plan's order for the user-visible items, but it adds one thing the plan does not
contain: the same scene gives different doors, floors and footprints depending only on point density (D1 below). Plan S2
(room layout solver, fed by the footprint polygon) and S4 (openings) should be built and gated against this sweep, not
against one density. This is a recommendation for the reviewer; I did not reorder anything.

## S1: duplicate room promotion

The plan says `topology.py` lacks room dedupe. **That claim is stale.** `perception/architecture/topology.py` already has
`_find_existing_room_entity` (Jaccard overlap >= 0.5 of boundary parts) and, when evidence-side rooms exist, refuses to mint a
second ROOM. Verified, not assumed:

* fixture compiled with and without `promote_building`: exactly 1 ROOM entity, unique id, 0 duplicate pairs, every
  wall/floor/ceiling contained once (`tests/test_room_fidelity_fixture.py::test_S1_*`, 2 parametrised cases pass);
* the canonical two-room, two-storey scene: exactly 2 ROOM ids (`room-plane-000`, `room-plane-001`), 8 walls, 2 floors,
  2 ceilings, 2 storeys, 1 building, space graph 2 levels / 2 rooms; the loose `>=` assertions in
  `tests/test_interior_compiler.py` and `tests/test_interior_invariants.py` are now exact (`==`), plus unique-id and
  room-to-storey ownership checks. A stale comment there claimed only one storey survived; the measured result is two.

## Defects found (evidence only; none fixed: the code is outside this task's file set)

* **D1. Result depends on cloud density.** Table above. At voxel 0.04: door lost, footprint polygon 5.35 x 4.35 m (about
  0.17 m outside each wall; area +16.4 %), room `wall_color_rgb` absent. At 0.06: a second FLOOR entity (a furniture top
  promoted as floor) and the door lost. Suspected: `room_semantics.py` footprint raster/cell choice shifting the wall scan
  lines used for openings and the wall colour band (not proven). Files: `perception/architecture/room_semantics.py`
  (footprint at ~line 300, wall band ~line 320, wall colour ~line 430).
* **D2. Building promotion silently refuses.** With `promote_building=True` (any density tested: 0.04, 0.08) the compile
  warns "building topology promotion: room room-001 boundary parts missing from world: ['struct-plane-0xx', ...]". The RoomGraph
  lists planes that were not promoted (furniture tops), so no BUILDING/STOREY is created for a single room. Pinned as a
  strict xfail in `tests/test_room_fidelity_fixture.py`. Files: `perception/architecture/room_graph.py`, `topology.py`.
* **D3. Furniture heights are short by 10-12 cm** (table 65 vs 75 cm, wardrobe 168 vs 180 cm): `base_height_above_floor_m` is
  about 0.12 m, so the lowest band is excluded from the height. Plan sizes are within 3 cm. The cause given is a reading of `base_height_above_floor_m`, not proven.
* **D4. The canonical compile records no scale state and no dimension state** (plan D1/D2). `dimensions_with_state_pct` = 0.
* **D5. `single_image.py` drops point colour** (see H2).
* **D6. Opening sizes are quantised by the cloud voxel** (door and window height 5 cm off, door width 5 cm off at 0.08).
  Plan S4 asks for <= 5 cm, which this just meets at 0.08 and does not at coarser clouds.

## Not done / needs the user

* Real-room fixture and every real-capture number (needs the photo set, tape L/W/H, door/window sizes and one
  camera-to-camera baseline). All real metrics print `not available`.
* The L-shaped, false-ceiling room (`synthetic/lroom.py`) is not in the headless gate (COLMAP, minutes).
* H1c/H4 need a real room (or a deliberately partial real capture) to reproduce the missing-polygon fallback.
