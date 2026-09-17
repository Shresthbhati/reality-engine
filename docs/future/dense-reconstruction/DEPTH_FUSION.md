# Depth Fusion

Status: IMPLEMENTED (monocular path) — spec records the contract for
all depth sources (2026-09-14)

## Purpose

Turn per-image depth information (sensor depth maps now, monocular
estimates today) into a single fused 3D point cloud in the session's
world frame, preserving per-point provenance and honest uncertainty.

## Current implementation

`reconstruction/depth_to_points.py` unprojects MiDaS depth maps using
each camera's K and pose; `reconstruction/fusion/` merges per-view
clouds. Fused points feed `reconstruction/meshing/` (preprocess →
Poisson). Scale comes from the session scale state (metric prior);
relative-only depth is explicitly not treated as metric
(see `docs/engineering/DESIGN_DECISIONS.md` Decision 023).

## Contract for any depth source

A depth source must provide, per image:

- `depth`: HxW float32 **meters** (already scaled) or INVALID
- `valid_mask`: bool HxW
- `scale_basis`: how meters were derived (device calibration /
  metric prior / unknown) — UNKNOWN is allowed and propagates
- intrinsics K and pose T_cam_world at that image's timestamp
- timestamp (for sync-aware fusion once time sync lands, P1)

Raw integer depth must never become meters without an explicit scale
factor (see Decision 020).

## Fusion rules

- Unproject only `valid_mask` pixels; reject non-finite depth.
- Depth-consistency across views (reprojection agreement) gates
  whether a point is kept when two or more views observe it.
- Per-point provenance: contributing image ids kept alongside the
  cloud (sidecar, not in the PLY unless colors stay standard).
- Downstream consumers (meshing) receive the fused cloud only;
  envelope filtering belongs to meshing preprocessing, not here.

## Failure modes

- Misaligned depth/RGB time base → blurred surfaces at edges; time
  sync (P1) must precede multi-sensor fusion.
- Wrong scale basis → silently wrong mesh; the scale basis is recorded
  in world metadata and surfaced in the compile report.
- Monocular depth edge bleeding → the camera-envelope filter plus
  Poisson depth limiting contain it; report residual stats.

## Acceptance criteria

- Deterministic: synthetic plane at known depth → unprojected cloud
  lies on the plane within tolerance (existing tests).
- Round-trip: fused cloud persists as an artifact with hash; reload
  reproduces identical points.

## Priority

P1 (sensor depth ingestion extends this contract; fusion core exists).
