# Capabilities Manifest

What the Reality Engine can do **today**, verified 2026-09-23.
Companion to `IMPLEMENTATION_STATUS.md` (component detail) and
`PENDING_IMPLEMENTATION.md` (what is missing). A capability listed here
has been executed against real inputs unless marked otherwise.

> **Note:** This file is subsidiary to REALITY_ENGINE_CURRENT_STATUS.md.
> Where they disagree, the latter is authoritative.

## Compile a capture into a world

```
reality compile ./capture            # one command, full spine
```

Input: a capture directory (images/video + manifest). Output: WorldIR
JSON + persisted artifacts + report. Backend injection via
`REALITY_TEST_BACKEND` supports deterministic offline runs.

- SfM camera poses (COLMAP), metric scale from measurement priors
- TRUE dense MVS (COLMAP patch_match_stereo → stereo_fusion): wired
  as a backend continuation (`dense_mvs=True`) — runs in the same
  workspace as sparse SfM, probes the binary's dense capability and
  per-stage CLI options, parses fused.ply into canonical points, and
  ingests the cloud through the content-addressed ArtifactStore into
  WorldIR geometry with dense provenance. Real GPU dense run on the
  committed real-photo dataset recorded 2026-09-21.
- Metric depth sidecars (iPhone/iPad LiDAR or any metric sensor, 16-bit PNG + declared scale) → metric scale with no tape measure + dense unprojection; a synthetic RGB-D capture is verified, a physical device is not yet.
- CPU plane-sweep dense geometry (opt-in, no CUDA): admits a pixel only when one depth clearly wins the photometric match, and refuses blank walls instead of filling them.
- Solid walls / floors / ceilings from the solved room layout, with openings carved; rooms can be joined through shared doorways.
- 2D floor plan export (SVG + DXF) and a room report with wall areas net of openings.
- User corrections (wall drag, door/window mark, typed dimension) stored as VERIFIED evidence and carried across rebuilds.
- Dimensions carry a state (MEASURED / RECONSTRUCTED / ESTIMATED / VERIFIED); a relative world can be scaled by one known length, or by an accepted door-height suggestion that is always labelled ESTIMATED. Checked only against synthetic ground truth.
- Room colours (lighting-normalised) and a texture atlas on the room shell; material labels are ESTIMATED hypotheses, never asserted, and are not used by physics. Viewer and inspector support is unit-tested, not browser-verified.
- Monocular depth (MiDaS) → fused point cloud (provenance per view)
- Poisson surface mesh from oriented fused points (CPU COLMAP
  mesher; auto depth/trim adaptation; camera-envelope outlier
  filtering from sparse SfM points)
- WorldIR world: entities (planes from measurements, objects from
  detections), measurements, metadata (scale state, frame)
- glTF export incl. real triangle mesh + point primitives

Verified on a real room capture: 25/28 cameras registered, 15.8 m
honest extent, 240,735-vertex mesh, 71 entities.

## Artifact store

Content-addressed (sha256), integrity-checked storage for meshes,
clouds, reports, exports; artifacts referenced from WorldIR via
`data_uri` + `data_hash`. Deterministic compilation: identical inputs
produce identical artifact hashes.

## Deterministic testing path

Fake reconstruction backend + synthetic fixtures run the full spine
offline (no COLMAP, no torch) — the backbone of the test suite.

## WorldStore (versioned world persistence)

SQLite-backed world versioning with:
- Immutable world versions with lineage tracking
- Integrity verification (hash checks)
- CLI save/load/list/verify commands
- Integration with WorldIR and exporters

## Registration & Trajectories

- **GNSS registration**: WGS84 anchors with covariance
- **ICP registration**: Iterative Closest Point alignment
- **Point-to-plane registration**: Refined pose estimation
- **Trajectory model**: TUM format, VIO federation, sync, diagnostics

## Reality Studio (Browser UI)

Next.js 16 + Three.js viewer with:
- **Desktop layout**: 33 page components for worlds/sessions/evidence/reports/analysis
- **Mobile layout**: camera/evidence/sessions/worlds views
- **12 API proxy routes**: jobs (compile/ingest/reconstruct/jobId), sessions (list/ingest-mobile/mobile-tasks), worlds (list/id/cameras/points/report/worldir)
- **Three.js viewer**: offline single-file with selection + provenance

## API Backend (FastAPI)

- **10 backend modules**: db, jobs, main, models, routes_jobs, routes_misc, routes_sessions, routes_worlds, storage
- **REST endpoints**: health, sessions (CRUD + location/trajectory/reconstruct), worlds (CRUD + versions/attach/coverage), evidence (CRUD + artifact), jobs, notifications, activity
- **SQLite + async worker**: durable job execution

## Explicitly NOT capabilities yet

- Physical LiDAR device verification — the RGB-D path is verified on synthetic data only
- Accuracy on a REAL room: every dimension/fidelity check runs on synthetic fixtures; no real photos or tape measurements exist yet
- Learned door/window/furniture detector (blocked on a user decision; openings come from geometry only)
- Curved or ornamented landmark architecture from a single image: FOUNDATION landed (`perception/architecture/landmark.py`: dome/arch/tower fits from single-image 3D points with measured residuals, honest refusals, relative-vs-measured scale); REAL-DATA VERIFICATION BLOCKED — no real landmark photo set or surveyed ground-truth dimensions exist yet; all verification synthetic (`tests/test_landmark.py`, 27 passed)
- Automatic cross-source or geo-registration: the Site composer (FOUNDATION) is manual placement of worlds on a metric grid
- Learned single-image room-layout model — the fusion/projection path exists, no model weights are shipped (choice open)
- Time sync / VIO / cross-source registration — specs written
- Co-observation landmark resolver — identified gap
- Uncertainty propagation — schema exists, propagation not
- Provenance graph (queryable) — lineage exists, graph not
- Material perception — spec written
- Multi-material contact resolution — 4-material database only
- GIS, robotics, large-world partitioning — specs written

Each "not yet" is specified under `docs/future/` with model, failure
modes, and acceptance criteria.

## Hardware & Environment

- COLMAP 4.2.0 with CUDA (RTX 4050, driver 616.56) — GPU feature extraction works
- CPU-only fallback available
- MiDaS via torch.hub — relative depth (metricized per-view)
- Mask R-CNN via torchvision — real COCO detections
- SAM via torch.hub — env-dependent (torch-hub cache fragility)
