# Capability Matrix — Reality Engine (world-compiler spec)

Tracks only the **Exit Goals (A-L)** from the autonomous-implementation-loop
directive — that's what decides whether to keep looping, not the full
40-section spec. Statuses: NOT_STARTED / SKELETON / PARTIAL / FUNCTIONAL /
INTEGRATED / VALIDATED.

| Goal | What it needs | Status | Evidence |
|---|---|---|---|
| A — Evidence system | Dataset, Session, evidence persists/reopens | **FUNCTIONAL** | `evidence/session.py`, `evidence/dataset.py`. Create/reopen round-trip tested (`test_serialize_deserialize_roundtrip_preserves_evidence_and_history`, `test_dataset_roundtrip_preserves_sessions_and_evidence`). Not yet wired to a real file importer (no photo/video ingestion path) — evidence items reference a `source_uri`, nothing reads the actual bytes yet. |
| B — Session merging | Select, merge, preserve provenance, detect conflicts | **PARTIAL** | `Dataset.merge()` produces a real `MergedContext`; source sessions verified untouched (`test_merge_preserves_source_session_provenance`); coordinate-frame conflict detection is real (`test_merge_flags_coordinate_frame_conflict`). Geometric registration/feature-matching/scale reconciliation is explicitly **not implemented** — `registration_status: "not_performed"`, honestly reported rather than faked. |
| C — Real reconstruction | End-to-end evidence → geometry | PARTIAL | `reconstruction/backend/fake.py` exercises the pipeline shape (6 tests). `reconstruction/backend/colmap_backend.py` is a real implementation — shells out to `feature_extractor`/`exhaustive_matcher`/`mapper`/`model_converter`, parses COLMAP's documented text output (`images.txt`, `points3D.txt`) into typed poses/points. **COLMAP 4.2.0 (no-CUDA Windows build) is now installed and the real subprocess pipeline has been run end-to-end** against 4 real photos of a public test scene: 3/4 images registered, 228 real 3D points produced, ~23s wall time. Two real bugs found and fixed by that run: (1) COLMAP's Windows CLI constructs a QApplication and blocks on a GUI dialog unless `QT_QPA_PLATFORM=offscreen`/`QT_PLUGIN_PATH` are set; (2) this build has no GPU support, so `--FeatureExtraction.use_gpu 0`/`--FeatureMatching.use_gpu 0` must be passed or every step fails. Both fixed unconditionally in `colmap_backend.py` (`_subprocess_env()`, `use_gpu` constructor flag, default `False`). Still PARTIAL, honestly: only one 4-image scene has been exercised (not a stress test, not a large building), and no automated test in `tests/` invokes the real binary — committed tests still only cover the pure parsing functions against fixture text. |
| D — WorldIR from reconstruction | Reconstructed result becomes typed entities+relationships+provenance+confidence | PARTIAL | Two real write paths now: `evidence/promote.py` (`MANUAL_MEASUREMENT` → `Entity`+`Material`+`Measurement`) and `evidence/promote_reconstruction.py` (`ReconstructionResult` → `Entity`+point-cloud `Geometry`, provenance forced to `RECONSTRUCTED`, refuses to promote a failed/empty result via `EmptyReconstructionError`). Both round-trip through `WorldIR.to_dict()`. Photo/video evidence still can't be promoted directly — it must go through a real `IReconstructionBackend` first, and only the fake one exists. |
| E — Incremental world growth | New Session added without destroying prior evidence | **FUNCTIONAL** | `Dataset.add_session()` only appends; `Session.add_evidence()` only appends; nothing in this layer supports deletion. Not yet exercised with a real multi-round "add evidence, re-derive" cycle since there's no derivation step (C) yet. |
| F — Validation loop | Reconstruction checked against source evidence, disagreement surfaced | PARTIAL | `reconstruction/validation.py`'s `validate_reconstructions()` nearest-point-matches two independent `ReconstructionResult`s within a distance threshold and reports agreements/disagreements/unmatched honestly -- no averaging, no silently picking a winner. Refuses to validate a `failed` reconstruction. **Now wired into a real workflow:** `evidence/promote_reconstruction.py`'s `promote_validated_reconstruction_to_entity()` runs validation as part of promotion, not as a dead-end function -- any disagreement downgrades the promoted Entity's and Geometry's provenance from RECONSTRUCTED to CONFLICT (not canonical, per `Provenanced.is_canonical()`), caps confidence at the agreement rate, and records the disagreement counts as a real Observation on the geometry. 9 tests (5 validation + 4 promotion-with-validation). Still not automatic on every new Session evidence (nothing currently triggers it without a caller supplying a second, independent reconstruction), and only compares two reconstructions to each other, not a reconstruction to an independent manual measurement -- that comparison still needs unit/scale reconciliation to be meaningful and remains a real follow-on. |
| G — Studio foundation | Viewport, outliner, inspector, selection | PARTIAL | `engine/render/viewport.py` (camera+frustum culling, REQ-029) and `engine/inspector/` (REQ-030) existed already. **New:** `engine/studio/` -- `Outliner` (hierarchy from `Entity.component_ids`, cycle-guarded, plus a grouped-by-type fallback and name search), `Selection` (ordered multi-select, replace/additive/toggle, `active` id), and `StudioSession` tying viewport+inspector+outliner+selection together, with `provenance_panel()` aggregating exactly what a provenance/evidence UI panel needs in one call and `visible_entities()` culling entities that carry a transform. 15 tests, 504/504 passing overall. Still no transform/edit tools -- deliberately: mutating WorldIR needs the validated command pipeline from spec §7 (intent -> parsing -> validation -> permission -> WorldAPI), which doesn't exist yet; adding editing here would be building it un-validated. That pipeline is separate, real, undone work. The Three.js artifact built in an earlier session is a one-off visualization, still not part of the repo. |
| H — World compilation foundation | WorldIR → render / physics / navigation representations, separately | PARTIAL | Physics representation exists and is mature (`engine/physics/*`, REQ-010-013). Render representation is viewport-only (no mesh/material compile step). Navigation representation: **not started** (no navmesh anywhere in the repo). |
| I — Large-world foundation | Spatial partitioning, local/world coordinate hierarchy, streamable cells | NOT_STARTED | `world_ir/coordinates.py` has multi-frame transforms (WGS84/UTM/ENU/local, REQ-005) but no cell/tile/streaming concept at all. |
| J — Engine extensibility | Reconstruction/AI/geometry backends replaceable via adapters | N/A yet | Nothing to make replaceable until C exists. Physics backend itself (`engine/physics/backend/interface.py`) already follows this pattern (`IPhysicsBackend`), so the precedent is established for when it's needed. |
| K — Trust (observed/inferred/uncertain/validated/generated) | Distinguished, never fabricated | **FUNCTIONAL** (pre-existing + extended) | `provenance.Provenance` enum already covers this (REQ-002/§7). Session/evidence layer defaults to `OBSERVED` and carries `Uncertainty` per item — consistent, not a new taxonomy. |
| L — Tested baseline | Evidence → Session → Merge → Reconstruction → WorldIR → Validation → Compilation, deterministic | PARTIAL | Evidence → Session → Merge is real and tested (25 tests, this pass). Reconstruction → WorldIR → Validation → Compilation don't exist yet, so the full chain isn't there — the chain is correct as far as it currently goes, not stubbed further.

## Termination test (§35), honest answers today

Can a dataset enter the system as structured Sessions? **Yes.** Can Sessions be
merged? **Yes, with honest limits.** Can real reconstruction occur? **No.** Can
results become WorldIR? **No** (nothing produces them yet). Provenance traced?
**Yes.** Uncertainty represented? **Yes.** World compiled into multiple runtime
representations? **Partially** (physics only). New evidence incrementally
improve an existing world? **The data model supports it; nothing yet re-derives
on new evidence, since nothing derives at all.**

**Loop verdict: keep going — do not stop.** Reconstruction (Goal C) is the
next foundational blocker; everything from D onward is downstream of it.

## Next highest-leverage blocker

Reconstruction backend evaluated and decided: **COLMAP** (BSD license,
subprocess-boundary integration, sparse output maps directly onto existing
`Provenance.RECONSTRUCTED` + `Observation`/`Measurement` records). Full
rationale, alternatives considered (ODM, Meshroom/AliceVision, OpenSfM), and
why each was rejected: `docs/RECONSTRUCTION_BACKEND_DECISION.md`.

Real COLMAP integration code now exists (`ColmapReconstructionBackend`):
real subprocess pipeline, real parsing of COLMAP's documented text format,
verified against literal COLMAP-format fixtures (481/481 tests, 6 new).

**Update (2026-09-12):** COLMAP 4.2.0 (no-CUDA Windows build) was installed
and `ColmapReconstructionBackend.reconstruct()` was run for real against
4 photos of a public test scene (`kicker` from the
`alexmkwizu/colmap-testing-dataset` HF dataset). Result: 3/4 images
registered, 228 real 3D points, ~23s. This was not a no-op -- the run
surfaced two real bugs in `colmap_backend.py`, now fixed: (1) COLMAP's
Windows CLI needs `QT_QPA_PLATFORM=offscreen` or it blocks on a GUI
dialog; (2) this build has no GPU, so the feature extractor/matcher need
`use_gpu=0` explicitly or they hard-fail. Both fixes are unconditional
(`_subprocess_env()`, `use_gpu` constructor flag), not environment-specific
hacks.

Goal F (validation) no longer requires C to be fully done -- it only needs
two `ReconstructionResult`s to compare, and the fake backend already
produces those. `validate_reconstructions()` now exists (486/486 tests,
5 new) and surfaces point-level disagreement without fabricating agreement.

**Update (2026-09-12):** Validation wired into promotion.
`promote_validated_reconstruction_to_entity()` in
`evidence/promote_reconstruction.py` now calls `validate_reconstructions()`
as part of promoting a reconstruction to WorldIR, and its result changes
real state rather than being discarded: disagreement flips the promoted
Entity/Geometry's provenance to CONFLICT and records the disagreement
counts as an Observation. 9 tests, 489/489 passing overall.

Remaining open item, not a blocker but explicitly deferred:
comparing a reconstruction against an independent manual measurement
(rather than reconstruction-vs-reconstruction) needs a real scale/unit
reconciliation step first -- COLMAP's monocular sparse output has no
absolute scale without ground-control points, so a raw distance
comparison against a manually measured real-world distance would be
comparing incompatible units. That reconciliation is real, undone work,
not a small wiring step.

Remaining honest gaps on reconstruction itself (not blockers, but real
limits): only one small 4-image scene has been exercised -- no large
building, no stress test, no automated CI coverage of the real binary
(only the pure parsing functions are unit-tested against fixture text).
