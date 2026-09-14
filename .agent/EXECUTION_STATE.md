# Reality Engine — Execution State

**Session end:** 2026-09-15 (RE-006 / P2-01 time synchronization)
**Queue:** `.agent/TASKS.yaml` (RE-2026-CORE-V1) — this file records
where execution actually stands, nothing else defines that.

## Verified baseline

- Suite: **1,392 passed, 1 skipped** on the time-sync branch (main
  `54b7574` at 1,359; +33 clock/sync tests). Main includes PR #22
  (depth sidecar ingestion + RGB-D unprojection + metric-scale fix)
  and PR #23 (P0-01 canonical state).
- Known environment failures (deselected, never counted): 20 SAM tests
  (`tests/test_perception_sam.py`, `tests/test_sam_backend.py`) —
  torch-hub cache failure on this machine, pre-existing, unrelated to
  code changes. Never reclassify; fix the environment or record it.

## Completed (most recent first, with evidence)

- **2026-09-15 — P2-01 (PARTIAL)** time synchronization:
  `evidence/clocks.py` — ClockModel/Timestamp/TimeAlignment/
  SynchronizedSample/SynchronizationDiagnostics with
  t_global = a*t_sensor + b; backend seam (ISynchronizationBackend +
  DEFAULT_BACKENDS in spec preference order); first backend
  metadata_alignment (shared_clock + known offset/drift from capture
  metadata); SensorStream.synchronized(clock_id, metadata) wiring
  (stream never mutated). 33 tests: pure-b, drift a≠1, cross-stream,
  rejected samples (non-finite per-sample guard), backend fallthrough,
  honest UNSYNCHRONIZED degradation, uncertainty honesty (unknown ≠
  zero; zero only by construction for shared clock), roundtrip.
  Suite 1,392. NOT claimed: spec methods 3-6 (GNSS/PPS, trigger,
  signal correlation, optimization); real multi-device capture
  evidence.
- **2026-09-15 — P0-01** canonical state (PR #23, merged `54b7574`):
  ENGINEERING_CONSTITUTION.md, REALITY_ENGINE_MISSION.md, TASKS.yaml
  (RE-2026-CORE-V1), this file, CAPABILITIES.yaml, LICENSES.yaml;
  legacy state archived with banners. Suite green at merge time.
- **2026-09-15 — P1-03 (PR #22, merged `f48c733`)**: DepthFrame +
  16-bit PNG sidecar parser (explicit depth_scale, Decision 020),
  MultiSourceSession.depth_frames(), pipeline sidecar unprojection
  with scale-coherence gate, deterministic RGB-D E2E (33 wall points
  at exactly z=2.0 m), metric-scale `result` reassignment bugfix.
  Suite 1,363. Status: PARTIAL — real-device RGB-D capture run is the
  named evidence for VERIFIED.
- **2026-09-14 — PR #21 (merged)**: canonical docs architecture
  (docs/{architecture,implementation,engineering,future}), CLAUDE.md
  §51–54. Suite 1,320 at the time.
- **2026-09-14 — PR #19 (merged)**: `reality compile` one-command CLI
  with deterministic fake-backend E2E.
- **2026-09-14 — PR #16/#17/#18 (merged)**: meshing stack (MeshData,
  scipy preprocessing, COLMAP poisson backend, camera-envelope
  filter), glTF uint32 mesh export, viewer mesh layer.

## In progress

- None open. P0-01 landed this session; next queue item not started.

## Blockers

- **P6-01 dense MVS**: CUDA unavailable (COLMAP 4.2.0 CPU-only) —
  backend slot is the deliverable; real-MVS verification gated on
  hardware. Decision required: none (recorded, not blocking the chain).
- **SAM perception tests**: torch-hub cache failure (environment).
- **open3d/trimesh/skimage absent**: meshing deliberately
  scipy/subprocess-based; revisit only if a capability demands it.

## Next (exact next step, per queue)

1. **P0-02** platform/application separation scoping: inventory core
   for disaster/application-specific logic; produce the extraction
   plan (files, dependency cuts, what stays generic).
2. **P0-03** verify CAPABILITIES.yaml against code; add the
   time_synchronization entry update (now PARTIAL) when landed.
3. **P3-01** canonical trajectory model (the critical chain's next
   link; consumes P2-01's ClockModel for monotonic global-time
   trajectories).

## Rules reminder

Never mark work complete without execution evidence. Never erase
incomplete work. Update this file at session end (constitution
Article II/IV).
