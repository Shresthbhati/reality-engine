# Reality Engine — Coordinate Frames

## Frames in play

| Frame | Handedness | Up | Where defined |
|---|---|---|---|
| Camera (COLMAP) | right-handed, +Z forward (optical axis), +Y down | — | `reconstruction/backend/interface.py` output |
| World (raw SfM) | backend-dependent | arbitrary | COLMAP model space, up to scale+rotation |
| World (canonicalized) | right-handed, +Y up | floor plane | `reconstruction/frame.py` — dominant plane → floor |
| Depth camera (unprojection) | same as camera | — | `reconstruction/depth_to_points.py` conversion |

## Rules

1. **Every point cloud, mesh, and transform declares its frame.** No
   silently mixed camera-space and world-space coordinates. `MeshData`
   and `Geometry` carry `coordinate_frame` metadata.
2. **Unprojection boundary.** Depth maps are camera-space; conversion to
   world points happens only through trusted intrinsics + registered
   pose, in one place (`depth_to_points.py`). With no trusted intrinsics
   the pipeline refuses (see "no trusted intrinsics" skip in report.json)
   rather than guessing a camera model.
3. **Canonical world frame.** After stage 2.5 the world frame is
   floor-dominant, +Y up, room-scale. Frame canonicalization records the
   applied transform; the pre-canonicalization frame is recoverable.
4. **Units.** Metric geometry is meters. `ScaleState` is
   `metric` (with `meters_per_unit` + anchoring provenance) or
   `relative` — relative quantities are never silently treated as
   meters. Mono-depth metricization is labeled `metric-by-alignment`
   (approximate, derived from relative depth + SfM scale), never
   sensor-grade.
5. **Quaternions** are `(w, x, y, z)` throughout
   (`ReconstructedCameraPose.rotation`, `math3.Quat`).
6. **Future frames** (NED/ENU for drones, ECEF/ENU for GIS, ROS TF) are
   P1/P7 concerns — explicit conversion layers at ingestion, never
   reinterpretation in place:
   [`../future/registration/CROSS_SOURCE_REGISTRATION.md`](../future/registration/CROSS_SOURCE_REGISTRATION.md).
