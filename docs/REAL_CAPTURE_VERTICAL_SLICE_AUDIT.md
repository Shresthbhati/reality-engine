# Real Capture → WorldIR Vertical Slice Audit

**Date:** 2026-09-13. Direct repository inspection.

Scoped against the convergence campaign's Phase 0 ask: classify the actual state of the EVIDENCE → PERCEPTION → RECONSTRUCTION → SEMANTICS → GEOMETRY → EVIDENCE FUSION → WORLD COMPILER → WORLDIR pipeline as it exists today, not as previously documented.

| Stage | Status | Evidence |
|---|---|---|
| Evidence ingestion (disk photos/video/LAS) | IMPLEMENTED | `evidence/importers.py` — real EXIF/GPS decode, corruption gates, deterministic ids, tested end-to-end into `to_evidence_items()`. |
| Camera reconstruction (SfM) | IMPLEMENTED (one real backend) | `reconstruction/backend/colmap_backend.py` — real COLMAP subprocess pipeline, previously run against 4 real photos (3/4 registered, 228 points, per `docs/CAPABILITY_MATRIX.md` row C). COLMAP 4.2.0 confirmed installed in this environment (`C:\Users\shres\tools\colmap-extracted`). **No persistent real-photo fixture exists in this repo** (`datasets/` is a README only) — the prior real run is documented but not re-runnable from a committed fixture. |
| Backend selection/orchestration | IMPLEMENTED, now INTEGRATED into Studio | `reconstruction/orchestrator.py` (real availability probing, fallback, attempt-log diagnostics — landed on `main`, merged into this branch). **Previously unwired**: nothing connected the orchestrator to a Studio session. This session added `StudioSession.reconstruct_and_compile(evidence, orchestrator)` closing that gap — see below. |
| Depth | PARTIAL | A MiDaS depth backend exists (`perception/depth/`, another agent's work, in-flight per prior CURRENT_STATE.md notes) implementing `IDepthBackend`. Not verified in this session — not exercised by the pipeline wiring done here. |
| Segmentation | PARTIAL | A SAM segmentation backend exists (`perception/segmentation/`, per commit history: "Add SAM segmentation backend"). Not verified in this session. |
| Point cloud / mesh generation | MISSING | No TSDF/Poisson/volumetric surface reconstruction exists — `Geometry` still stores only `vertex_count`, never real vertex buffers (unchanged from every prior audit). |
| 2D→3D lifting / object hypotheses | MISSING | No code converts a 2D detection/segmentation mask into a 3D object hypothesis anywhere in the repo. |
| Room/structural geometry | IMPLEMENTED | `perception/geometry/planes.py` + `orientation.py` + `evidence/promote_planes.py` + `evidence/promote_rooms.py` — real RANSAC plane detection, wall/floor/ceiling classification, room-ring closure, real bugs found and fixed in prior sessions, tested against synthetic and (once) real COLMAP output. |
| Entity resolution across views | PARTIAL | Room/wall/floor promotion already deduplicates within one compile (wall-pairing for thickness). Cross-session (two separate captures of the same room) resolution is new this session: `world_ir/entity_reid.py` (deterministic geometric+type matcher). |
| Measurement | IMPLEMENTED | Extent/thickness/area/height measurements with real ESTIMATED provenance, precision from fit RMS (`evidence/promote_planes.py`, `evidence/promote_rooms.py`). |
| Scene graph | IMPLEMENTED | `engine/scene_graph/graph.py` — CONTAINS/PART_OF/SUPPORTS queries over real promoted relationships. |
| Evidence fusion (scalar conflict resolution) | IMPLEMENTED, narrow | `reconstruction/fusion/fusion.py` — real inverse-variance fusion with CONFLICT provenance. No caller yet feeds it competing plane fits automatically. |
| World Compiler | IMPLEMENTED, INTEGRATED | `engine/compiler/world_compiler.py` — real multi-stage pipeline (planes → classify → promote → rooms → validate), consumes real `ReconstructionResult`, refuses invalid/empty input, gates on validation. Now reachable from Studio via one call (`StudioSession.reconstruct_and_compile`, this session). |
| WorldIR validation | IMPLEMENTED | `world_ir/validation.py` — geometry/transform/measurement/provenance checks, composed into the compiler's gate. |
| End-to-end CLI (`reality reconstruct-room`) | **MISSING** | `apps/cli/` is a README placeholder only — no CLI exists anywhere in this repo. |
| Reality Studio inspection UI | **MISSING** | `apps/studio/` is a README placeholder. `engine/studio/session.py` is a real headless session object now capable of running the whole pipeline via `reconstruct_and_compile()`, but there is no viewport/inspector UI a person can look at. |

## This session's change

**`StudioSession.reconstruct_and_compile(evidence, orchestrator, compile_options=None)`** (`engine/studio/session.py`) — the literal missing glue this audit's own prior session flagged as the next task: evidence → real orchestrator-selected backend → compiled, validated WorldIR, in one call, returning both the raw `ReconstructionRun` diagnostics (which backend ran, attempt log) and the `CommandResult` of the compile. Raises `ReconstructionOrchestrationError` when every candidate backend declines/fails — never silently returns an empty or partial world (verified: a total-failure test asserts `len(session.world.entities) == 0` afterward).

3 tests (`tests/test_studio_reconstruct_and_compile.py`): success path (produces >=2 structural entities from a synthetic room, using the real `ReconstructionOrchestrator` class with `FakeReconstructionBackend` since no real photo fixture exists in-repo — the orchestrator itself is real and identical to what a COLMAP backend would flow through), total-failure path leaves the world untouched, diagnostics correctly name the backend that ran.

## Honest gap against Prompt 1's Definition of Done

Prompt 1 asks for `REAL IMAGES → CAMERA RECONSTRUCTION → DEPTH → POINT CLOUD → SEMANTICS → 3D OBJECT HYPOTHESES → ROOM GEOMETRY → MEASUREMENTS → SCENE GRAPH → EVIDENCE FUSION → WORLD COMPILER → VALIDATED WORLDIR` as one executable workflow, run against a real ~20-50 photo room dataset.

**Not achieved this session, named honestly:**
- No real photo fixture exists in this repository to re-run the COLMAP path end-to-end right now (the prior real run is documented, not reproducible from a committed asset). Acquiring/committing or documenting acquisition instructions for a real 20-50 image room dataset is unstarted.
- Point cloud → mesh generation, depth integration into the compiler, and 2D→3D object lifting are all still missing entirely — the compiler only consumes camera-reconstruction points + planes today, never depth or segmentation output.
- No CLI and no Studio UI exist to run or inspect the pipeline as a single user-facing workflow; today it is a sequence of Python calls a test or script makes directly (now one call shorter, via `reconstruct_and_compile`).

This session closed one real, concrete, previously-flagged integration gap (orchestrator → Studio → compiler) rather than attempting the full vertical slice, which remains multi-session scope requiring a real dataset and at minimum a depth/mesh integration this session did not attempt.
