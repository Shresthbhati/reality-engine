# VIO Trajectory Estimation

Status: MISSING (2026-09-14)

## Purpose

Estimate a continuous, metric, time-stamped camera/imu trajectory from a
video or image+IMU source when the COLMAP SfM path is unavailable, fails
(global/panoramic motion, textureless scenes), or must run in real time.
VIO is also the bridge to cross-source registration: any trajectory that
is continuous in time can be aligned to other sensors.

## Current state

The pipeline reconstructs per-session camera poses with COLMAP SfM
(`backend.py`) and metricizes them from a scale prior
(`scale_estimator.py`). There is no VIO backend, no IMU ingestion
(frontend manifest accepts IMU metadata only), no odometry interface,
and no trajectory representation in WorldIR beyond per-image poses.

## Approach

Two-tier, both mature backends, no novel research:

1. **Visual odometry (V-only fallback).** Feature tracking + PnP frame
   to frame over the video stream, scale from a prior (scene height,
   user input, or depth sensor). Use OpenCV. Good enough for walk-through
   captures; drift must be reported, not hidden.
2. **VIO (IMU present).** Delegate to a mature optimizer — ORB-SLAM3 or
   VINS-Fusion as subprocess backends behind the standard backend
   interface. Never vendor their internals.

## Canonical representation

`Trajectory` (new, in `world_ir/` or a `trajectories/` module):

- `frames`: list of `(timestamp_ns, T_cam_body_4x4, pose_covariance?)`
- `frame_source`: `vio | svo | sfm | ekf` plus provenance
- `drift_estimate`: scalar or 6x6 covariance, UNKNOWN if not estimable
- monotonic, strictly increasing timestamps (checked at construction)

Camera poses in WorldIR may then be *sampled or derived from* a
trajectory instead of independent SfM points, with provenance recorded.

## Coordinate frames

- VIO returns body-frame (IMU) poses; conversion to camera frame uses
  the recorded `T_cam_imu` extrinsic. UNKNOWN extrinsic = BLOCKED, never
  assume identity.

## Failure modes

- IMU/camera time offset: estimate in the backend (VINS does); record it.
- Scale drift in V-only: mandatory `drift_estimate`; gates METRIC claims.
- Loop closure unavailable in small rooms: acceptable; drift is small and
  reported.
- Backend crash: BACKEND_UNAVAILABLE status, honest skip; no fake path.

## Acceptance criteria

- Deterministic fixture: synthetic 2D-euclidean video with known
  trajectory → VO pose error reported, monotonic timestamps verified.
- Real capture: VIO backend runs if binary present; report statuses.
- All integration tests pass without the backend (offline, honest skip).

## Priority

P1 (after depth ingestion, with time synchronization).
