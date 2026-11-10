# 35 — Task cards (work queue)

> Each card is one agent-session-sized unit of work. Tags indicate who should
> carry it: **Nemotron** = this model (implementation-heavy coding), **strong
> model** = complex reasoning/architecture needing a more capable model, **you**
> = the OpenCode agent (general execution, verification, glue). Take the first
> card whose dependencies are DONE. Status: `TODO · IN PROGRESS · DONE
> (<commit>) · BLOCKED: <reason>`. Update this file and
> `docs/implementation/IMPLEMENTATION_STATUS.md` in the same commit.

---

## Tag key
- `Nemotron` — code implementation this model can ship end-to-end.
- `strong model` — architecture/data-model decisions needing stronger reasoning.
- `you` — general execution, verification, test wiring, repo hygiene.

---

## Phase 0 — Consolidate and baseline (all DONE — kept as the executed baseline)
| ID | Card | Tag | Deps | Done when | Verify | Status |
|----|------|-----|------|-----------|--------|--------|
| 0.1 | Merge `integration/all-worktrees` into `main` | you | — | `main` contains `world_ir/observations.py`, `perception/instances/association.py`, `reconstruction/matching/`, `perception/device.py` | full pytest before merging; paste pass/fail/skip counts | DONE |
| 0.2 | Environment doctor script | Nemotron | 0.1 | `scripts/doctor.py` prints Python/venv/torch CUDA/GPU VRAM/colmap/node/npm/DB writable | run it; paste output | DONE |
| 0.3 | CUDA torch in project venv | Nemotron | 0.2 | doctor shows CUDA true on RTX machine | doctor output | DONE (not needed) |
| 0.4 | One-command launcher (API + worker + frontend) | Nemotron | 0.1 | `start.bat` brings all three up; `GET /health` OK; UI loads | curl `/health`; browser screenshot | DONE |
| 0.5 | Record the baseline | Nemotron | 0.4 | `IMPLEMENTATION_STATUS.md` has pytest counts + frontend build + e2e, dated + commit | the commands themselves | DONE |
| 0.6 | Import-boundary test | Nemotron | 0.1 | `tests/test_import_boundaries.py` encodes the layer table; passes today or each violation is a follow-up card | pytest output | DONE |
| 0.6a–0.6c | Boundary cleanup (shared types, pipeline facade, upward imports) | Nemotron | 0.6 | allow-list shrinks; 47 sibling-coupling pairs + 7 upward pairs removed | boundary test passes | DONE |
| 0.7 | Fix inherited baseline failures | Nemotron | 0.1 | listed tests pass (`test_colmap_coverage`, `test_compile_command`, `test_confidence_defaults`, `test_packaging_smoke`, `test_frontend_sessions_proxy_contract`) | pytest on those files | DONE |

---

## Phase 1 — Product spine (all DONE — kept as baseline)
| ID | Card | Tag | Deps | Done when | Verify | Status |
|----|------|-----|------|-----------|--------|--------|
| 1.1 | Alembic migrations | Nemotron | 0.5 | `alembic upgrade head` creates schema on empty DB; existing dev DBs upgrade with no data loss | run on empty DB + copy of existing | DONE |
| 1.2 | World lifecycle API | Nemotron | 1.1 | PATCH/duplicate/archive/restore/delete/soft-purge + `?status=` filter; delete cancels jobs; one activity event per op | pytest; curl each endpoint | DONE |
| 1.3 | Trash purge + blob GC | Nemotron | 1.2 | permanent delete removes rows+versions+unreferenced blobs only; shared blobs survive | test with two worlds sharing a blob | DONE |
| 1.4 | World card menu + Trash/Archived views + delete confirmation | Nemotron | 1.2 | Open/Rename/Duplicate/Export/Archive/Delete work against real API; confirmation lists what will be deleted | Playwright worlds-lifecycle.spec; screenshots 1280/1920 | DONE |
| 1.5 | Export fix | Nemotron | 0.4 | one request-path convention; errors shown to user; GLB passes `gltf-validator` and opens in Blender with correct transforms | validator output + Blender screenshot | DONE |
| 1.6 | Honest job progress | Nemotron | 0.4 | jobs expose `stage` + `stage_counts`; UI shows labels; failed jobs show reason; no % unless measured | run a reconstruction; screenshot each state | DONE |
| 1.7 | Settings honesty | Nemotron | 0.4 | Appearance real; Processing/Data real (or a task opened); Account/Workspace/Integrations say "Not available in this version" | screenshot | DONE |

---

## Phase 2 — Baseline benchmark
| ID | Card | Tag | Deps | Done when | Verify | Status |
|----|------|-----|------|-----------|--------|--------|
| 2.1 | Benchmark runner | Nemotron | 0.5 | `benchmarks/run.py` runs pipeline on `south_building` + living-room set; writes `benchmarks/results/<date>-<commit>.json` with registration ratio, reprojection error, entity counts, duplicates, runtime, peak RAM/VRAM | run it; commit the JSON | DONE |
| 2.2 | Scenario suite skeleton A1–A20 | Nemotron | 0.5 | `tests/scenarios/` one test per scenario; synthetic ones implemented; rest `skip("not covered: needs <data>")` | pytest shows pass/skip per scenario | DONE |
| 2.3 | Tape-measured rooms | you | — | `datasets/measured/<room>/`: 3 rooms photographed + measured (L×W×H, 5 object dimensions) | dataset README with measurements | **BLOCKED** — needs real captures by the user |

---

## Phase 3 — Identity (all DONE — kept as baseline)
| ID | Card | Tag | Deps | Done when | Verify | Status |
|----|------|-----|------|-----------|--------|--------|
| 3.1 | Complete the Observation record | Nemotron | 0.1 | fields per `11_WORLDIR_CONTRACT.md` (mask/bbox, visibility, embedding_ref, session_id, created_at); ledger saved inside each WorldIR version | round-trip serialization test | DONE |
| 3.2 | Association is the only resolver path | Nemotron | 3.1 | Hungarian default everywhere; union-find removed or behind explicit non-default flag with deprecation note; every decision stored | grep shows no default union-find; A4 test | DONE |
| 3.3 | Scale gate in lifting | Nemotron | 3.1 | relative depth yields `depth_source=relative` + wide sigma; no metric dimensions unless scale_state ≠ relative | unit test with relative depth | DONE |
| 3.4 | Validator invariants 1–7 | Nemotron | 3.1 | commit blocked on invariant failure with readable report | one unit test per invariant | DONE |
| 3.5 | Entity evidence in the inspector | Nemotron | 3.2 | selecting an entity shows its source photos with masks + association decisions | Playwright + screenshot | DONE |
| 3.6 | Identity scenarios pass | Nemotron | 3.2–3.4 | A1–A6 pass on real or synthetic data | pytest + benchmark JSON diff vs Phase 2 | DONE |

---

## Phase 4 — Better matching and depth (all DONE — kept as baseline)
| ID | Card | Tag | Deps | Done when | Verify | Status |
|----|------|-----|------|-----------|--------|--------|
| 4.1 | Registry entries for all current models | Nemotron | 0.1 | every backend has licence, commercial_ok, measured VRAM, status | registry dump | DONE |
| 4.2 | One-model-at-a-time scheduler | Nemotron | 4.1 | peak VRAM ≤ budget on living-room set; CPU fallback works | benchmark JSON peak VRAM | DONE |
| 4.3 | ALIKED+LightGlue behind `IMatcher` | Nemotron | 2.1, 4.1 | registered ratio/inlier ratio vs SIFT recorded; default switched only if better | benchmark diff | DONE |
| 4.4 | Metric depth candidate | Nemotron | 2.1, 4.1 | one metric model behind the depth interface; `scale_state="estimated"` when used; benchmarked vs MiDaS | benchmark diff + measured-room error (if 2.3 done) | DONE |

---

## Phase 5 — Architecture and coverage (all DONE — kept as baseline)
| ID | Card | Tag | Deps | Done when | Verify | Status |
|----|------|-----|------|-----------|--------|--------|
| 5.1 | Contract hierarchy in WorldIR | Nemotron | 3.4 | World→Building→Floor→Space→Surface/Opening/Entity stored; old versions still load | round-trip + old-fixture load test | DONE |
| 5.2 | Architecture → contract adapters | Nemotron | 5.1 | planes/rooms/openings become Space/Surface/Opening with evidence states | A1, A6, A14 tests | DONE |
| 5.3 | Ray-cast coverage | Nemotron | 5.2 | each Surface gets coverage fraction + OBSERVED/OCCLUDED/UNOBSERVED | test on synthetic room with known hidden region | DONE |
| 5.4 | Topology revalidation after merges | Nemotron | 5.2 | merged topology rechecked geometrically; failures flagged, not hidden | unit test with a bad merge | DONE |

---

## Phase 6 — Guided capture and incremental updates (all DONE — kept as baseline)
| ID | Card | Tag | Deps | Done when | Verify | Status |
|----|------|-----|------|-----------|--------|--------|
| 6.1 | World-space CaptureRequests | Nemotron | 5.3 | requests carry target region, suggested poses, instruction text, expected gain | API response on living room | DONE |
| 6.2 | Register new photos into an existing world | Nemotron | 3.6 | new images register against existing model; non-matching photos refused with reason (A11, A19) | scenario tests | DONE |
| 6.3 | Incremental update on ingest | Nemotron | 6.2 | only affected closure recomputes; new version + diff; untouched entities keep IDs + geometry | A12 test + diff output | DONE |
| 6.4 | Capture UI | Nemotron | 6.1, 6.3 | user sees a request, adds a photo, sees what changed | Playwright + screenshots | DONE |

---

## Phase 7 — Evidence, confidence and the three views (all DONE — kept as baseline)
| ID | Card | Tag | Deps | Done when | Verify | Status |
|----|------|-----|------|-----------|--------|--------|
| 7.1 | ConfidenceVector populated | Nemotron | 3.6 | every entity has the vector; `null` where unmeasured | unit test; API sample | DONE |
| 7.2 | Real / Wireframe / Evidence views | Nemotron | 5.3, 7.1 | toggles switch modes on same scene graph; legend for evidence colours | screenshots of all three | DONE |
| 7.3 | Measurement with honest scale | Nemotron | 7.2 | shows value/unit/uncertainty/scale state; "Set scale" turns relative into calibrated | e2e + screenshot | DONE |

---

## Phase 8 — Object and furniture reconstruction (all DONE — kept as baseline)
| ID | Card | Tag | Deps | Done when | Verify | Status |
|----|------|-----|------|-----------|--------|--------|
| 8.1 | Ontology file + loader | Nemotron | 3.2 | `config/ontology.yaml` + loader; v1 class set; unknown → `unknown_object` | loader test | DONE |
| 8.2 | Open-vocab detector via the registry | Nemotron | 4.1, 8.1 | prompts from ontology; output = Observations only | per-class precision/recall on living room | DONE |
| 8.3 | Parametric object templates | Nemotron | 8.1 | table, cabinet, sofa, chair, bed, TV/media console fitted from observations with components | geometry error vs measured objects (if 2.3 done) | DONE |
| 8.4 | Relationships | Nemotron | 8.3 | supported_by/mounted_on/part_of/inside computed geometrically | unit tests | DONE |
| 8.5 | User corrections | Nemotron | 8.3 | reclassify/split/merge → user-basis decision + new version | e2e | DONE |

---

## Phase 8b — Wire perception into the product path
| ID | Card | Tag | Deps | Done when | Verify | Status |
|----|------|-----|------|-----------|--------|--------|
| 8.6 | Scene-complete eval set | strong model | 8.1 | `benchmarks/scene_complete/` with ≥20 AI + ≥20 real room photos labeled (class, box, support surface) + scorer | scorer + runner + label checker + tests done; 40 labeled photos present | **PARTIAL** — scorer/runner done; 40 labeled photos missing (needs a person) |
| 8.7 | Single-image path: DA-V2 depth, planes on wall/floor/ceiling masks, detector → `furniture_from_detections` | Nemotron | 8.6 | spec M1 acceptance (COCO-class furniture recall ≥70%, floor contact <5% room height) | spec M1 acceptance test | PARTIAL — layout+furniture wired, acceptance unmeasured |
| 8.8 | Open-vocab detector + crop pass from API path | Nemotron | 8.7 | spec M2 acceptance; fixtures on wall planes | tests/test_task_8_8_fixtures_and_open_vocab.py pass | DONE |

---

## Phase U — UI/UX redesign (all DONE locally — kept as baseline)
| ID | Card | Tag | Deps | Done when | Verify | Status |
|----|------|-----|------|-----------|--------|--------|
| U.1 | Frontend audit: routes→APIs, libs, duplicates, tokens | Nemotron | — | audit table committed; `36_UI_SPEC.md` updated with keep/replace per responsibility | audit committed | DONE |
| U.2 | Library compatibility + licence matrix | Nemotron | U.1 | matrix committed; nothing installed without a row | matrix committed | DONE |
| U.3 | Shared design tokens (colour, type, spacing, radius) light + dark | Nemotron | U.1 | every component uses tokens; contrast checks pass | design-tokens.test.mts; WCAG AA | DONE |
| U.4 | Home: New Session · Open World + recent + needs-attention | Nemotron | U.3, 11.1 | Playwright + screenshots at 1280/1920/375 | e2e/home.spec.ts | DONE |
| U.5 | Session: one guided flow with explicit processing states | Nemotron | U.3, 1.6 | states match backend `stage`; no fake % | e2e/session-flow.spec.ts; stage-labels.test.mts | DONE |
| U.6 | World workspace: viewport-first, hierarchy left, inspector right | Nemotron | U.3 | inspector varies by selection type; screenshots | e2e/world.spec.ts | DONE |
| U.7 | Session adoption review | Nemotron | U.6, 6.3 | adopting a second session matches rather than duplicates | e2e | DONE |
| U.8 | Mobile priorities + WCAG 2.2 AA pass | Nemotron | U.4–U.7 | keyboard + touch walkthrough recorded; axe clean | mobile.spec.ts | DONE |

---

## Phase 9 — Fine detail, video, geo (all DONE — kept as baseline)
| ID | Card | Tag | Deps | Done when | Verify | Status |
|----|------|-----|------|-----------|--------|--------|
| 9.1 | High-res crop pass (sockets/switches) | Nemotron | 8.2 | sockets/switches detected when resolvable; none claimed below `min_pixels` | — | DONE |
| 9.2 | Video keyframe ingestion | Nemotron | 6.3 | walk-around video reconstructs like a photo set; keyframe count reported | — | DONE |
| 9.3 | Geo anchor (EXIF GPS → `World.geo_anchor`) | Nemotron | 5.1 | world shows on a map; local geometry unchanged | — | DONE |

---

## Phase 10 — Appearance and constrained completion (all DONE — kept as baseline)
| ID | Card | Tag | Deps | Done when | Verify | Status |
|----|------|-----|------|-----------|--------|--------|
| 10.1 | Texture projection from photos onto meshes | Nemotron | 7.2 | Real view uses image-derived textures; GLB export carries them | — | DONE |
| 10.2 | Material classification as attributes | Nemotron | 8.2 | material + confidence on entities | — | DONE |
| 10.3 | Completion proposals as PRIOR_GENERATED | Nemotron | 7.2 | hidden parts proposed, visibly marked, never OBSERVED | — | DONE |

---

## Phase 11 — Product polish (all DONE — kept as baseline)
| ID | Card | Tag | Deps | Done when | Verify | Status |
|----|------|-----|------|-----------|--------|--------|
| 11.1 | Home dashboard | Nemotron | 1.4 | real data only | — | DONE |
| 11.2 | World package export/import (versioned zip) | Nemotron | 1.2 | export → import yields identical world | — | DONE |
| 11.3 | Resumable uploads | Nemotron | 1.2 | interrupted upload resumes | — | DONE |
| 11.4 | Navigation consolidation | Nemotron | 1.4 | Capture and Reports are single merged screens | — | DONE |

---

## Phase 12 — v1 acceptance
| ID | Card | Tag | Deps | Done when | Verify | Status |
|----|------|-----|------|-----------|--------|--------|
| 12.1 | Living-room demo end to end | Nemotron | Phases 1–8 | all 12 steps in `00_VISION.md` pass in the browser; screenshots + benchmark JSON committed | — | DONE |
| 12.2 | Measured-room accuracy report | Nemotron | 2.3, 4.4, 8.3 | dimension error table per room/object committed | — | **BLOCKED** — depends on 2.3 |
| 12.3 | Mark capabilities VERIFIED | Nemotron | 12.1 | `IMPLEMENTATION_STATUS.md` updated with evidence links | — | DONE |

---

## Phase 13 — After v1 (all DONE — kept as baseline)
| ID | Card | Tag | Deps | Done when | Verify | Status |
|----|------|-----|------|-----------|--------|--------|
| 13.1 | Branch audit + CODEOWNERS | Nemotron | — | stale remote branches deleted; CODEOWNERS committed | — | DONE |
| 13.2 | pHash dedup + time/GPS session grouping | Nemotron | — | production caller exists | tests/test_task_13_2_*.py pass | DONE |
| 13.3 | HEIC, 360°, PLY/LAS, OBJ/glTF ingestion | Nemotron | — | formats ingest correctly | — | DONE |
| 13.4 | Learned room layout, non-Manhattan rooms | Nemotron | — | `perception/architecture/non_rectangular.py` | tests/test_non_rectangular_rooms.py | DONE |
| 13.5 | Floor plan SVG/PDF export | Nemotron | — | exports valid SVG/PDF | — | DONE |
| 13.6 | SAM 2 behind segmentation interface | Nemotron | — | benchmark vs SAM 1 | tests/test_sam2_backend.py | DONE |
| 13.7 | Outdoor perception | Nemotron | — | vegetation, roads, terrain, facades, roofs | tests/test_outdoor_perception_task13_7.py | DONE |
| 13.8 | Colour accuracy (ΔE) | Nemotron | — | controlled capture ΔE | tests/test_colour_accuracy_task13_8.py | DONE |
| 13.9 | MASt3R (NC flag) + DINOv2 re-ID | Nemotron | — | adopt only if benchmarks beat current | tests/test_task_13_9_*.py | DONE |

---

## New capabilities (40-item improvement backlog) — not yet implemented

These are the cards from the improvement backlog that are architecturally intended but not implemented. Tagged per the work-allocation note.

| ID | Card | Tag | Deps | Done when | Verify | Status |
|----|------|-----|------|-----------|--------|--------|
| N-1 | Live capture coverage map — per-surface seen/weak/not_seen, observed vs estimated | Nemotron | 5.3 | `reconstruction/capture_quality.py` `wall_coverage()` consumed by UI; provisional label | unit test on synthetic room; UI shows coverage | **TODO** |
| N-2 | Automatic image-quality inspection — blur/over/under-exposure/obstruction/reflection/framing flagged with reason | Nemotron | — | poor images flagged with specific reason + suggested correction before acceptance | `assess_capture()` verdicts on real photos | **TODO** |
| N-3 | Multi-sensor timestamp sync — photos/video/GPS/IMU on common time base | strong model | evidence/sensors.py | captured frame associable with correct sensor measurements + timestamps | unit test syncing mixed streams | **TODO** |
| N-4 | Visual-inertial localization — real phone IMU → VIO backend with fallback when tracking fails | strong model | N-3, P3-02 | phone capture uses visual + inertial together, not GPS alone | `tests/test_synthetic_vio.py` passes on real device data | **BLOCKED** — no VIO backend / no device |
| N-5 | Indoor floor/elevation tracking — stairs/ramps/lifts/multi-floor without vertical displacement confusion | Nemotron | N-4 | multi-storey capture maintains consistent floor assignments + vertical positioning | multi-storey scenario test | **TODO** |
| N-6 | Automatic loop-closure detection — return to observed location corrects drift, no duplicate location | Nemotron | scene_identity.py | revisiting a room improves alignment instead of duplicating | loop-closure test on revisit sequence | **TODO** |
| N-7 | Reconstruction uncertainty map — spatial confidence: strong/weak/conflicting/no evidence | Nemotron | — | users can inspect where model is trustworthy vs needs capture | uncertainty map unit test | **TODO** |
| N-8 | Fine-grained architectural components — door leaf/frame/hinges/handle/glazing, window frame/sill/lintel, socket/switch faceplate | Nemotron | 8.1 | components independently identified when image evidence supports it | per-component detection test | **TODO** |
| N-9 | Glass/transparent-surface reasoning — windows/transparent doors/mirrors not reconstructed as opaque walls | Nemotron | 8.2 | closed glazing not auto-reconstructed as opaque wall | glass scene test | **TODO** |
| N-10 | Occlusion-aware reconstruction — hidden surfaces labeled unknown/inferred, not empty or solid | Nemotron | — | occluded geometry labeled unknown/inferred | occlusion scenario test | **TODO** |
| N-11 | Persistent object identity — same door/window/tree/fixture matched across viewpoints, no duplicates | Nemotron | 3.2, 8.2 | stable identities across images/videos/sessions | cross-session identity test | **TODO** |
| N-12 | Material/finish recognition — wood/concrete/tile/stone/metal/glass/paint with evidence+confidence+unknown | Nemotron | 10.2 | each material label has evidence, confidence, explicit unknown state | material classification test | **TODO** |
| N-13 | Architectural constraint solver — walls not meeting floors, doors detached, floating windows, unconnected stairs flagged | Nemotron | 5.1 | invalid relationships detected and reported before model accepted | constraint solver unit tests | **TODO** |
| N-14 | Parametric architectural geometry — editable wall thickness/height/opening/floor elevation preserving relationships | Nemotron | N-13 | user can change a parameter preserving relationships | parametric edit test | **TODO** |
| N-15 | Geometry closure/watertightness validation — holes, non-manifold edges, disconnected surfaces, inverted normals flagged | Nemotron | — | exports include geometry-quality report | mesh validation test | **TODO** |
| N-16 | Measurement confidence/tolerances — CI + provenance per dimension; measured vs image-derived vs heuristic distinguished | Nemotron | 7.3 | displayed measurement communicates uncertainty + evidence source | dimension confidence test | **TODO** |
| N-17 | Multi-representation consistency — point cloud/mesh/splat/entity disagreement flagged, not silently resolved | Nemotron | — | consistency report surfaces conflicts | representation conflict test | **TODO** |
| N-18 | Automatic cross-session world registration — new capture proposed as update to existing world, not new world | Nemotron | 6.2 | revisit proposed as update to existing world | cross-session registration test | **TODO** |
| N-19 | Three-way world merge — original/new capture/edited; physical changes separated from user edits | Nemotron | N-18 | new capture cannot silently overwrite manual corrections | three-way merge test | **TODO** |
| N-20 | Object-level change history — what changed/when/evidence per wall/door/window/fixture/furniture/site object | Nemotron | — | user can inspect change + evidence + support | change history test | **TODO** |
| N-21 | Physical-damage/deterioration tracking — cracks/stains/deformation distinguished from existing condition across captures | Nemotron | — | new defect distinguished from existing, accounting for lighting/viewpoint | damage tracking test | **TODO** |
| N-22 | Cross-world geographic identity — consistent CRS for buildings/floors/rooms/corridors/outdoor/sites | Nemotron | 9.3 | multiple captured properties in coherent geographic context without manual transforms | geo-anchor multi-world test | **TODO** |
| N-23 | Unified spatial selection — viewport/object tree/search/NL query, synchronized everywhere | Nemotron | — | selecting a window anywhere highlights same object throughout app | selection sync test | **TODO** |
| N-24 | Constraint-aware editing — editing wall/opening/floor preserves connections, detects downstream consequences | Nemotron | N-13 | common architectural edits cannot leave disconnected geometry without warning | constraint-aware edit test | **TODO** |
| N-25 | Global undo/redo — transactional across manual edits + confirmed AI ops, version-aware recovery | Nemotron | — | reverse multi-object edit without corrupting world | undo/redo test | **TODO** |
| N-26 | Natural-language scene inspection/editing — "show me all windows wider than one metre" grounded; unsupported edits rejected | strong model | — | grounded copilot returns world-based results; unsupported edits rejected honestly | NL query test | **TODO** |
| N-27 | Automatic floor-plan/room-layout validation — plan vs 3D world, discrepancies reported | Nemotron | 5.1 | generated plan compared against 3D world; discrepancies reported | floor-plan validation test | **TODO** |
| N-28 | Unified rendering-quality manager — splats/mesh/solids chosen by task; switchable | Nemotron | — | users switch visual quality vs geometric precision without managing separate models | renderer test | **TODO** |
| N-29 | Streaming + memory-budget controller — load/unload by view distance/importance/memory | Nemotron | — | large worlds usable on constrained hardware; measurable memory/frame-time limits | memory budget test | **TODO** |
| N-30 | Seamless navigation transitions — orbit/aerial/third-person/first-person one continuous camera | Nemotron | — | move exterior→interior without manual mode switches | camera transition test | **TODO** |
| N-31 | Realistic window/portal visibility — openings as spatial connections showing correct exterior/adjacent space | Nemotron | N-9 | user inside room sees correct surrounding world through openings | portal visibility test | **TODO** |
| N-32 | Lighting/exposure consistency — no visible seams between differently-exposed rooms | Nemotron | — | world does not develop lighting seams from different exposures | exposure consistency test | **TODO** |
| N-33 | Real-building benchmark programme — measured rooms/apartments/corridors/stairs/buildings/outdoor with ground truth | strong model | 2.3 | every major release has results against real physical measurements | benchmark report against measured set | **BLOCKED** — depends on 2.3 |
| N-34 | False-negative/completeness scoring — missed walls/openings/objects measured, not just detected accuracy | strong model | N-33 | high completeness score requires detecting difficult objects | completeness scoring test | **TODO** |
| N-35 | Stage-by-stage observability — input/output/runtime/memory/warnings/failure per stage | Nemotron | — | engineer can trace why a wall/object was missing | stage observability test | **TODO** |
| N-36 | Crash-safe resumable reconstruction jobs — stage checkpoints, idempotent retries, restart without duplicating | Nemotron | — | interrupting a job does not restart successful stages | crash/resume test | **TODO** |
| N-37 | Evidence/provenance governance — every spatial claim traceable to source/sensor/stage/edit/inference | Nemotron | — | user can inspect why entity/measurement exists; observed vs inferred distinguished | provenance audit test | **TODO** |
