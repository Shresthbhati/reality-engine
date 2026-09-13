# Reality Studio / Production Integration Audit

**Date:** 2026-09-13
**Method:** direct inspection of the repository (file reads, `wc -l`, `ls`, test runs) — not inference from prior docs. Prior docs (`.agent/CURRENT_STATE.md`, `docs/CAPABILITY_MATRIX.md`) were cross-checked against actual files, not trusted blindly.

This is a scoped, honest snapshot against the 33-phase production campaign brief. It intentionally does **not** claim broad completion — most phases below are MISSING. One concrete gap (WorldDiff, Phase 6) was closed and verified in this same session; see "This session's change" at the end.

Status vocabulary used below, applied strictly:

- **IMPLEMENTED** — real code exists and does what it claims.
- **INTEGRATED** — implemented *and* wired into the actual pipeline other code calls.
- **PARTIAL** — real but narrower than the phase's stated scope.
- **STUB** — interface/shape exists, no real logic.
- **MOCK** — fake data standing in for a real backend.
- **MISSING** — nothing exists.
- **UNVERIFIED** — code exists, not exercised by a test or a real run in this session.

## Phase-by-phase state

| Phase | Area | Status | Evidence |
|---|---|---|---|
| 2 | WorldIR schema | IMPLEMENTED, PARTIAL | `world_ir/schema_v1.py` (569 lines) has entities/geometry/materials/provenance/uncertainty/temporal events/causal relations/coordinate frames/branches. Deterministic `to_dict()`/`from_dict()` round-trips exist and are tested. No content-hash/fingerprint function exists anywhere (`grep` for "fingerprint\|content_hash" in world_ir/ finds nothing) — Phase 2's "deterministic hashing" item is MISSING. |
| 2 | WorldIR validation | IMPLEMENTED, INTEGRATED | `world_ir/validation.py` (249 lines) — geometry bounds (missing/non-finite/inverted), transform (non-finite/non-affine/singular), duplicate-identity heuristic, measurement (non-finite/negative-dimension), provenance/confidence consistency, composed with `WorldIR.validate()`'s own structural self-check (dangling refs, temporal/branch consistency). Wired into `engine/compiler/world_compiler.py`'s gate — a corrupted world is refused, not silently accepted. This already covers most of Phase 2's checklist; NOT covered: coordinate-frame-consistency-across-transforms, impossible-hierarchy-cycle detection (no entity hierarchy/parent field exists in the schema to cycle), stale-evidence-reference detection. |
| 3 | World Compiler | IMPLEMENTED, INTEGRATED, PARTIAL | `engine/compiler/world_compiler.py` runs planes → classification → promotion → rooms → validation gate as one deterministic call, with `CompileDiagnostics`. It is a real multi-stage pipeline, but far narrower than Phase 3's 16-stage list: no depth/segmentation/material fusion stage exists (there is nothing to fuse — no concrete perception backend produces those observations yet), no explicit "entity resolution" or "completeness analysis" stage as named subsystems (completeness is implicit in what compiles vs. what raises). |
| 4/5 | Reality Studio (UI) | **MISSING** | `apps/studio/` contains only a 254-byte `README.md` — no application code. What exists under `engine/studio/` (`session.py` 160 lines, `outliner.py` 77 lines, `selection.py` 52 lines) is a *headless Python session object*: create/select/move/delete entities through the command pipeline, list an outliner tree. There is no 3D viewport, no entity inspector panel, no evidence inspector, no measurement tool UI, no conflict inspector, no timeline UI, no simulation controls UI. Everything in Phase 4's spec describing a viewport/panels/overlays is aspirational relative to the current repo. |
| 5 | Studio state architecture (overrides, undo/redo) | **MISSING** | No override layer exists between compiled WorldIR and "what the user changed." `WorldCommandProcessor` mutates the session's live WorldIR directly (validated, permissioned, event-logged) — there's no separate "evidence truth vs. override" layering, and no undo/redo stack. |
| 6 | World differencing | **IMPLEMENTED, UNVERIFIED-IN-PRODUCTION** (this session) | `world_ir/diff.py` (new) — deterministic `diff_worlds(a, b)` → `WorldDiff` (added/removed/modified entities and geometries, field-level `FieldChange` for transform/type/name/provenance/confidence/custom_properties/geometry bounds). 12 tests in `tests/test_world_diff.py` pass (determinism, order-independence, no-mutation, serialization shape). Not yet wired into any caller (no exporter/regression test/branch-comparison uses it yet — that wiring is the natural next step, not done here). |
| 7 | Export system | PARTIAL, INTEGRATED (narrow) | `exporters/gltf/exporter.py`, `exporters/usd/exporter.py`, `exporters/blender/exporter.py` (glTF added earlier, Blender added last session) all real and tested (structural/byte-level assertions, not just "no exception"). All three: BOX+PLANE geometry only (AABB, not real mesh data — `Geometry` has no vertex buffer), unit-cube-shaped placeholder for gltf/usd, real-AABB-sized cube for Blender. None of the three produce a structured "export report" (entities exported/skipped, warnings, deterministic hash) — that's Phase 7's explicit ask and is MISSING. No OBJ or GIS exporter exists. |
| 8 | Blender integration | PARTIAL | Script-generation exists and is tested (`exporters/blender/exporter.py`, 8 tests). No actual Blender binary is installed in this environment (`which blender` → not found) — the generated script's *syntax* is verified (`ast.parse`) but it has never been run inside real Blender, so "collections/objects/materials/hierarchy actually created" is UNVERIFIED beyond source-level correctness. No round-trip validation path (WorldIR → Blender → export → compare → WorldDiff) exists — Phase 6's new `diff_worlds` makes that buildable next, but it isn't built. |
| 9 | USD/glTF round-trip validation | **MISSING** | No parser-based reload/compare step exists for either format — `docs/CAPABILITY_MATRIX.md` already flags this honestly ("never round-tripped through the real `pxr` library"). |
| 10 | Golden scenes | **MISSING** | No `tests/golden_scenes/` or equivalent directory exists. The geometric-reasoning/room-inference test suites (`tests/test_geometric_reasoning.py`, `tests/test_room_inference.py`) use hand-computed synthetic fixtures with the same spirit (deterministic, hand-checkable invariants) but are not packaged as reusable named "golden scenes" with expected-invariant contracts. |
| 11 | Reconstruction benchmark contract | PARTIAL | `docs/BENCHMARKS.md` exists with some real measured numbers (per `.agent/CURRENT_STATE.md`'s prior-session notes); no structured per-metric contract (pose error, depth abs/rel error, IoU, etc.) as a machine-checkable framework exists in code. |
| 12/13 | Performance / caching | **MISSING** | No profiling instrumentation, no cache layer for decoded media/features/reconstruction/compiled WorldIR found in a repo-wide search for "cache" in perception/reconstruction/evidence modules beyond ad-hoc in-memory dicts. |
| 14 | Error handling taxonomy | PARTIAL | Individual modules raise specific exceptions with real messages (`PlanePromotionError`, `RoomInferenceError`, `FusionError`, `ImporterCapabilityError`) rather than swallowing failures — good hygiene already present. No repo-wide `SUCCESS/PARTIAL_SUCCESS/FAILED/INVALID_INPUT/...` status enum exists as Phase 14 asks. |
| 15 | Security/trust boundaries | UNVERIFIED | Not audited this session; `evidence/importers.py` (media ingestion) was not re-reviewed specifically for path traversal / unsafe archive handling in this pass. |
| 16 | Observability | **MISSING** | No `run_id`/structured trace system exists; logging (where present) is ad-hoc. |
| 17 | CLI / headless workflow | **MISSING** | `apps/cli/` is a 250-byte `README.md` only — no CLI entry point exists (`ingest`/`inspect`/`reconstruct`/`compile`/`validate`/`measure`/`simulate`/`diff`/`export`/`benchmark` commands from Phase 17 all absent). Everything usable today is library-level Python, imported and called directly (as this session's tests do), not a `--help`-discoverable CLI. |
| 18/19 | Project config / technology registry | PARTIAL | `docs/TECHNOLOGY_REGISTRY.md` exists with real license/capability research (per prior session) for depth/segmentation candidates; no project-config file format (coordinate system, units, backend selection, cache dir, seed, etc.) exists as a loadable config schema. |
| 20 | Adapter boundaries | PARTIAL | `IDepthBackend`, `ISegmentationBackend`, `ITrackBackend`, `IReconstructionBackend`, `IPhysicsBackend` interfaces exist (per prior CURRENT_STATE.md) with zero-to-one concrete implementations each; export adapters (gltf/usd/blender) are functions, not a common `IExporter` interface with declared capabilities/versions — no shared adapter contract across exporters. |
| 21 | Test pyramid | IMPLEMENTED, PARTIAL | 902 tests pass as of this session (`python -m pytest -q --ignore=tests/test_midas_backend.py`). Strong unit coverage for WorldIR/geometry/measurement/fusion/compiler/diff. No dedicated end-to-end test that starts from raw evidence *files* (photos on disk) through Studio through export through readback in one test — `tests/test_export_pipeline_e2e.py` (added last session) is the closest thing and only covers reconstruction→promotion→export, not the full evidence-ingestion-to-export chain. |
| 22–27 | Studio UI validation, large-scene behavior, temporal/branching UI, causal debugging, confidence visualization, completeness | **MISSING** | All depend on Phase 4's UI, which does not exist. |
| 28 | Production project format | **MISSING** | No project container/manifest format exists. |
| 30 | Final integration test (evidence → export) | PARTIAL | The closest real chain executed and verified in this repo: disk photo folder → `evidence/importers.py` → `DeterministicPackageBuilder` → COLMAP reconstruction (real binary, per CURRENT_STATE.md) → RANSAC planes → orientation → promotion (now with real transform) → world compiler → validation gate → gltf/usd/blender export. This chain is real but has never been run as a *single* test in one file from raw files on disk all the way to an exported file — it exists in pieces across several test modules that were each verified independently. |

## This session's change (implemented and verified)

**`world_ir/diff.py`** — Phase 6, World Differencing. Previously **MISSING entirely** (repo-wide search for "worlddiff"/"world_diff" found nothing before this change). Now:

- `diff_worlds(a, b) -> WorldDiff`: pure, deterministic function comparing entity and geometry collections between two `WorldIR` instances.
- Reports added/removed/modified entities and geometries; modifications carry explicit per-field `FieldChange(field, old, new)` — type, name, transform, provenance, confidence, geometry_id/material_id set membership, `custom_properties` key-by-key (where measurements currently live), and geometry type/bounds/vertex_count/triangle_count/provenance/confidence.
- Deterministic: sorted id iteration means dict-insertion order in either input world never affects the result (tested explicitly).
- Never mutates its inputs (tested explicitly).
- `to_dict()` produces a plain-data, JSON-serializable structure with a `summary()` count block.

**Verification performed:**
```
python -m pytest -q tests/test_world_diff.py -v   # 12 passed
python -m pytest -q --ignore=tests/test_midas_backend.py  # 902 passed, 1 skipped (890 baseline + 12, zero regressions)
```

**Not done in this pass** (explicitly out of scope for one increment, listed here rather than silently deferred): wiring `diff_worlds` into an actual caller (export-fidelity check, reconstruction-vs-reconstruction comparison, simulation branch comparison — Phase 6 explicitly lists these as consumers, none exist yet), Material/Surface-level diffing, relationship-edge diffing.

## Largest remaining gap (unchanged from before this session)

**Reality Studio as a UI does not exist.** `apps/studio/` is a placeholder. Everything Phases 4/5/22–27 ask for (viewport, inspector panels, measurement tools, conflict UI, timeline, causal debugging surfaces) depends on an application layer that has not been started — only a headless session/command-pipeline backend exists underneath where that UI would sit. This is the single biggest blocker to the campaign's stated product vision and the most honest "next highest-value" target, but building even a minimal viewport is multi-session scope, not something completed in this pass.
