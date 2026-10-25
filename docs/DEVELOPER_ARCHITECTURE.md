# Developer architecture -- how a photo becomes a versioned world

For users, see [`../QUICKSTART.md`](../QUICKSTART.md). This document is for people changing the engine. Settled
decisions and their reasons: [`engineering/DESIGN_DECISIONS.md`](engineering/DESIGN_DECISIONS.md). What is verified, and
against what: `reality verification` / [`../.agent/CAPABILITIES.yaml`](../.agent/CAPABILITIES.yaml).

## The product path in one picture

```
POST /api/reconstructions (photos, optional world_id)
  -> World row (created on first upload) + Session + Evidence rows         apps/api/routes_reconstructions.py
  -> Job (queued)  ->  in-process worker loop                               apps/api/jobs.py
       evidence_staged   photos checked/classified, staged to disk
       colmap_staged     persistent COLMAP workspace for THIS world updated   reconstruction/colmap_session.py
       (candidates)      incremental candidate AND full re-solve, judged as WORLDS
       worldir_created   candidate compiled to a WorldIR                      engine/pipeline/vertical_slice.py
       candidate_evaluated  compared with HEAD                                engine/pipeline/world_delta.py
       before_adoption / during_adoption   adoption protocol                  apps/api/worldstore_service.py
       version_persisted -> colmap_current_updated
  -> GET /api/worlds/{id}/status  (what the Studio polls)                     apps/api/routes_reconstructions.py
```

There is **no separate worker service**: the API process starts the worker loop (`apps/api/main.py` startup) and recovers
jobs and half-adopted versions left by a previous process.

## WorldIR

The canonical world: entities (walls, floors, rooms, objects ...) with geometry, transform, provenance
(OBSERVED / RECONSTRUCTED / INFERRED / GENERATED / UNKNOWN), confidence and uncertainty, plus relationships and
observations. `world_ir/`. Nothing downstream (exporters, Studio, diff) reads anything but WorldIR. Provenance travels
with every statement; unknown stays unknown (no default confidence of 1.0 for measured things).

## WorldStore (immutable versions) and the DB mirror

`worldstore/store.py` writes **content-addressed, immutable versions** (`versions/<id>.json` plus artifacts) and an ordered
`sequence.json`. The database mirrors them (`WorldVersion` rows) and holds one mutable thing: `World.current_version_id`,
the **HEAD pointer**. Rollback moves only that pointer; history is never rewritten and evidence is never removed.

## COLMAP, and the staging/current workspaces

COLMAP is an external binary (`colmap` on `PATH`). Each world has a persistent workspace
(`<worldstore>/colmap-sessions/<world>/`) with `current/` (the model HEAD was built from) and `staging/` (this run's
working copy). A run edits **staging only**; `current/` advances strictly *after* the version is adopted and read back, so
a crash leaves it at most one version behind, which converges. `incremental` registers new photos into the existing sparse
model (keeps COLMAP's coordinate frame); `full` re-solves everything (new arbitrary frame).

## Candidate arbitration (`engine/pipeline/candidate_selection.py`)

Both candidates become world snapshots and are compared against HEAD by an **ordered hierarchy, never a scalar score and
never by registered-camera count**: photos lost, surfaces lost, unsupported moves/conflicts, camera stability, recorded
uncertainty, **coordinate frame preserved**, useful evidence placed, new understanding. A candidate that keeps HEAD's frame
beats one that re-expresses the world in a new frame unless the latter places >= `FRAME_CHANGE_MIN_EXTRA_PHOTOS` (2, PROVISIONAL)
more photos. If both candidates fail world-level acceptance, HEAD is kept (`world_delta.decide`).

**Frame continuity.** Frame canonicalisation (dominant plane / camera-up -> +Y) used to be re-estimated per candidate,
so versions of one world drifted by degrees. `reconstruction.frame.canonicalize_like_head` reuses HEAD's recorded rotation
(`report["frame"]`) when the candidate's shared cameras demonstrably sit where HEAD's are; otherwise it estimates. The frame
shift is *measured* (`candidate_selection.frame_shift`) and reported to the Studio.

## The adoption protocol (failure recovery)

Order: intent file -> WorldStore version -> **read-back verification** (before the pointer moves) -> DB HEAD + mirror row in
**one transaction** -> intent cleared. A version written but never adopted is **quarantined** (never deleted): immediately on
an in-process fault, or by `reconcile_adoptions` at startup after a hard kill. A rerun with unchanged evidence mints no
version (`world_delta.is_unchanged`). The eight failure boundaries above are exercised by real process kills synchronised to
the stage (`tests/integration/stage_server.py`, `test_reliability_journey.py`), not by timers.

## Depth, dense, detail

Dense MVS is attempted only when the measured sparse model justifies it (`engine/pipeline/dense_gate.py`) and is kept only
if judged better than the sparse world (`dense_judge.py`); otherwise the sparse world stands. The detail budget is
scale-aware: metric worlds get GSD bands, relative worlds withhold every metric statement and derive voxels from the robust
scene extent (`perception/detail/voxel.py`; thresholds and their calibration class in `engineering/DETAIL_CALIBRATION.md`).
Depth/perception need the optional `perception` extra and skip *visibly* without it.

## Verification model and synthetic data

`engine/core/verification.py` states what each capability has been verified against: IMPLEMENTED < SYNTHETICALLY VERIFIED
< REAL-WORLD VERIFIED, with EXTERNAL VERIFICATION PENDING listed beside it. `synthetic/` holds deterministic RGB-D, VIO and
indoor fixtures with ground truth; they verify contracts, never devices or real buildings.

## Where things are

| concern | location |
|---|---|
| API routes | `apps/api/routes_*.py`, `apps/api/jobs.py`, `apps/api/worldstore_service.py` |
| pipeline stages | `engine/pipeline/` (vertical_slice, progressive, world_delta, candidate_selection, dense_*) |
| reconstruction backends | `reconstruction/` (colmap_backend, colmap_session, frame, scale, depth) |
| exporters | `exporters/` (gltf, usd, blender, cityjson, citygml, ifc_bridge; optional ecosystem writers) |
| Studio | `frontend/src/` (status panel: `components/reconstruct/ReconstructionStatusBar.tsx`) |
| tests | `tests/` (unit, API), `tests/integration/` (real COLMAP, slow) |
