# Capability Matrix — Reality Engine (world-compiler spec)

Tracks only the **Exit Goals (A-L)** from the autonomous-implementation-loop
directive — that's what decides whether to keep looping, not the full
40-section spec. Statuses: NOT_STARTED / SKELETON / PARTIAL / FUNCTIONAL /
INTEGRATED / VALIDATED.

| Goal | What it needs | Status | Evidence |
|---|---|---|---|
| A — Evidence system | Dataset, Session, evidence persists/reopens | **FUNCTIONAL** | `evidence/session.py`, `evidence/dataset.py`. Create/reopen round-trip tested (`test_serialize_deserialize_roundtrip_preserves_evidence_and_history`, `test_dataset_roundtrip_preserves_sessions_and_evidence`). Not yet wired to a real file importer (no photo/video ingestion path) — evidence items reference a `source_uri`, nothing reads the actual bytes yet. |
| B — Session merging | Select, merge, preserve provenance, detect conflicts | **PARTIAL** | `Dataset.merge()` produces a real `MergedContext`; source sessions verified untouched (`test_merge_preserves_source_session_provenance`); coordinate-frame conflict detection is real (`test_merge_flags_coordinate_frame_conflict`). Geometric registration/feature-matching/scale reconciliation is explicitly **not implemented** — `registration_status: "not_performed"`, honestly reported rather than faked. |
| C — Real reconstruction | End-to-end evidence → geometry | PARTIAL | Backend decided (COLMAP, `docs/RECONSTRUCTION_BACKEND_DECISION.md`). `reconstruction/backend/interface.py` defines `IReconstructionBackend`; `reconstruction/backend/fake.py` is a real, deterministic (canned-data) implementation that lets the full pipeline be exercised (6 tests, `test_reconstruction_pipeline.py`) without a CV dependency. No COLMAP call exists anywhere — the fake backend invents nothing itself, it returns exactly what tests supply. |
| D — WorldIR from reconstruction | Reconstructed result becomes typed entities+relationships+provenance+confidence | PARTIAL | Two real write paths now: `evidence/promote.py` (`MANUAL_MEASUREMENT` → `Entity`+`Material`+`Measurement`) and `evidence/promote_reconstruction.py` (`ReconstructionResult` → `Entity`+point-cloud `Geometry`, provenance forced to `RECONSTRUCTED`, refuses to promote a failed/empty result via `EmptyReconstructionError`). Both round-trip through `WorldIR.to_dict()`. Photo/video evidence still can't be promoted directly — it must go through a real `IReconstructionBackend` first, and only the fake one exists. |
| E — Incremental world growth | New Session added without destroying prior evidence | **FUNCTIONAL** | `Dataset.add_session()` only appends; `Session.add_evidence()` only appends; nothing in this layer supports deletion. Not yet exercised with a real multi-round "add evidence, re-derive" cycle since there's no derivation step (C) yet. |
| F — Validation loop | Reconstruction checked against source evidence, disagreement surfaced | NOT_STARTED | Depends on C existing first. |
| G — Studio foundation | Viewport, outliner, inspector, selection | SKELETON | `engine/render/viewport.py` (camera+frustum culling, REQ-029), `engine/inspector/` (REQ-030) exist. No transform tools, no evidence/provenance visualization panel, no editing. The Three.js artifact built earlier this session is a one-off visualization, not part of the repo. |
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

The full chain evidence → reconstruction → WorldIR now runs end-to-end
against a deterministic fake backend (475/475 tests passing, 6 new). Real
COLMAP is still not wired in anywhere -- this only proves the *pipeline
shape* is right, not that reconstruction works on real imagery.

Next highest-leverage blocker: a real `IReconstructionBackend` implementation
that shells out to COLMAP (subprocess invocation + parsing its sparse
output into `ReconstructionResult`). That's the first point in this whole
loop where an actual external dependency and actual CV computation enters
the codebase -- worth its own careful pass (subprocess error handling,
COLMAP binary availability, output-format parsing) rather than folding into
a larger step.
