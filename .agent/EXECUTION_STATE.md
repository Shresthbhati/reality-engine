# Reality Engine — Execution State

**Session end:** 2026-09-15 (P0-03 registry verification + P18-01 closure, PR #25)
**Queue:** `.agent/TASKS.yaml` (RE-2026-CORE-V1) — this file records
where execution actually stands, nothing else defines that.

## Verified baseline

- Suite: **1,400 passed, 1 skipped** on the trajectory branch (main
  `5235960` at 1,392; +8 platform-boundary tests), re-verified after
  ledger closures. Main includes PR #22
  (depth sidecars + RGB-D unprojection), PR #23 (P0-01 canonical
  state), PR #24 (P2-01 clock model + first sync backend).
- Known environment failures (deselected, never counted): 20 SAM tests
  (`tests/test_perception_sam.py`, `tests/test_sam_backend.py`) —
  torch-hub cache failure on this machine, pre-existing, unrelated to
  code changes. Never reclassify; fix the environment or record it.

## Completed (most recent first, with evidence)

- **2026-09-15 — P0-03 (DONE)** backend capability registry verified:
  mechanical cross-check of .agent/CAPABILITIES.yaml — 19/19 entries,
  every backend/fallback .py path resolves to real code, all required
  fields present, honest maturity labels (13 IMPLEMENTED / 2 PARTIAL /
  3 MISSING / 1 BROKEN; BROKEN image_segmentation = pre-existing SAM
  torch-hub env failure, recorded not hidden). Satisfies P0-03's two
  verification clauses; P5-02 (backend selection) dependency now met.
- **2026-09-15 — P18-01 (DONE)** application extraction closed as
  premise-falsified: P0-02's verified inventory found NO application
  workflows in core to extract ('engine/disasters/' is a README-only
  scaffold; fire/destruction/environment are generic primitives).
  Preventive boundary adopted instead — R1 enforced by
  tests/test_platform_boundary.py (its 'core imports no application
  modules' verification item). Remaining item ('application package
  installs/tests independently') recorded as CONDITIONAL until an
  application package exists.
- **2026-09-15 — P0-02 (DONE)** platform/application separation
  scoping: verified inventory (the disaster-application layer does NOT
  exist in code — `engine/disasters/` is a README-only scaffold; no
  evacuation/hazard-scenario/emergency-response logic anywhere;
  fire/destruction/environment are generic physics primitives, STAY).
  Preventive rules R1–R3 adopted; **R1 enforced structurally** by
  `tests/test_platform_boundary.py` (platform chain must not import
  simulation domains; physics-compiler bridge attaches only via the
  SDK facade; 4 basic-utility couplings allowlisted with follow-ups
  F1–F3 registered). Plan:
  `docs/implementation/PLATFORM_APPLICATION_SEPARATION.md`. Suite
  1,400. No runtime code changed beyond the new guard tests.
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

1. **P0-03** capability-registry hardening: fold the P2-01
   time_synchronization entry to PARTIAL with the landed facts
   (clocks.py, backend seam, metadata_alignment backend); probe what
   else is cheaply verifiable.
2. **P3-01** canonical trajectory model (the critical chain's next
   link; consumes P2-01's ClockModel for monotonic global-time
   trajectories; backend-neutral per the no-custom-VIO rule).

## Rules reminder

Never mark work complete without execution evidence. Never erase
incomplete work. Update this file at session end (constitution
Article II/IV).
