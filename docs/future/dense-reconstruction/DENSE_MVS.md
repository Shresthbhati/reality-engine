# Dense MVS (Multi-View Stereo)

Status: PARTIAL → target IMPLEMENTED via depth-fusion + optional MVS
backend (2026-09-14)

## Purpose

Produce a dense, accurate point cloud (and, via meshing, a mesh) from
posed cameras, as the primary geometry source when depth sensors are
absent and the highest-fidelity source when they are not.

## Current state

- Depth-map fusion from MiDaS monocular depth exists
  (`reconstruction/depth_to_points.py`, `reconstruction/fusion/`) and
  feeds meshing (P0.11–P0.13, PR #16).
- COLMAP 4.2.0 CPU-only is available: `patch_match_stereo` (the real
  MVS densifier) requires CUDA and is **dependency-blocked** on this
  machine; `poisson_mesher`/`delaunay_mesher` are used for surface
  reconstruction (`reconstruction/meshing/surface.py`).
- Therefore: today's dense geometry = monocular depth fused into a
  cloud + SfM metric scale. Real MVS is a backend slot that is empty.

## Backend contract

A dense-MVS backend consumes an image set + posed cameras + intrinsics
and produces a per-image depth/normal map set (COLMAP workspace format
is the lingua franca). It must:

- expose `available()` honestly (CUDA present? binary present?)
- return BACKEND_UNAVAILABLE rather than empty-but-success
- write outputs as artifacts with provenance (binary version, config,
  input dataset hash)

## Candidate backends (mature, do not vendor)

| Backend | Requirement | Notes |
|---|---|---|
| COLMAP patch_match_stereo | CUDA GPU | reference quality |
| OpenMVS | CPU possible | densify + reconstruct + refine |
| AliceVision Meshroom | CPU heavy | pipeline wrapper |

## Fusion

Per-image depth maps → fused cloud must:

- depth-consistency-check across views before adding a point
- keep per-point provenance (which images/depth maps contributed)
- record fusion parameters in world metadata
- hand the fused cloud to `meshing/preprocess.py` (downsample →
  outlier filter → camera-oriented normals → `surface.py` Poisson)

The camera-envelope filter (sparse-SfM points bound plausible scene
extent) stays in front of Poisson — see
`docs/architecture/PIPELINE.md` and the mesh-stage notes; MVS clouds
should be much tighter than the current 43k-point depth-outlier tail
the filter removes.

## Failure modes

- Textureless surfaces → MVS holes; honest holes, no inpainting.
- CUDA absent → BACKEND_UNAVAILABLE; monocular-depth path continues to
  serve as the (clearly labeled, lower-accuracy) fallback.
- Scale: MVS depths are metric-per-SfM-scale; apply the same scale
  state as everything else and record it.

## Acceptance criteria

- With a GPU: room capture → patch_match_stereo → fused cloud → mesh;
  compare vertex count/extent against the monocular-depth mesh;
  document both.
- Without a GPU: suite passes with honest skip; no regression.

## Priority

P1-P2 (backend slot is the deliverable; quality depends on hardware).
