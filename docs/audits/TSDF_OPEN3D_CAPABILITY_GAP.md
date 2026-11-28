# §4 TSDF Dense Reconstruction — Open3D Capability Gap (honest finding)

**Date:** 2026-11-26
**Context:** Directive §4 — "Connect Open3D TSDF to the actual production reconstruction execution path" with validations: depth consistency, pose validity, voxel resolution, truncation distance, memory limits, partial scans, invalid depth, missing views, mesh extraction.

## Finding: BLOCKED by the installed Open3D build, not merely "unconnected"

The prior audits (RCEAP Cycle 0, DAY6) reported "TSDF absent from the job path." On inspection, the gap is deeper than "unconnected" — the installed Open3D build does not expose a TSDF API at all.

**Open3D version installed:** 0.20.0 (verified: `import open3d; open3d.__version__` → `0.20.0`)

**TSDF API — NOT AVAILABLE in this build:**
- `open3d.integration` module: **NOT PRESENT** (ModuleNotFoundError). The classic `open3d.integration.ScalableTSDFVolume` (the well-known RGB-D volumetric integration class from older Open3D) is absent.
- `open3d.t.geometry.TSDFVolume`: **NOT PRESENT** (`hasattr` → False). The tensor/PyTorch-based TSDF volume is absent.
- `open3d.t.geometry.RGBADepth`: **NOT PRESENT**.

**Mesh extraction from point cloud — NOT AVAILABLE:**
- `TriangleMesh.create_from_point_cloud`: **NOT PRESENT** (`hasattr` → False).
- `TriangleMesh.create_from_point_cloud_alpha_shapening`: **NOT PRESENT**.
- So even if points were produced, there is no Open3D built-in Poisson/alpha-shape mesh extraction from a point cloud in this build.

**What Open3D 0.20.0 DOES provide (per-view, not volumetric fusion):**
- `RGBDImage.create_from_color_and_depth` ✓
- `PointCloud.create_from_rgbd_image` ✓ (per-view RGBD → point cloud)
- `PointCloud.create_from_depth_image` ✓ (per-view depth → point cloud)
- `RGBDImage`, `PointCloud`, `TriangleMesh`, `PinholeCameraIntrinsic` (but NOT PinholeCameraExtrinsics — extrinsics are represented as 4x4 matrices passed directly to integration methods in older APIs; in 0.20 the integrate path itself is absent)

## What this means for §4

The directive asks for a production TSDF integration with: voxel resolution, truncation distance, depth integration across views, mesh extraction, partial-scan/invalid-depth/missing-view handling, memory limits, and an actual reconstruction job executing it.

With this Open3D build, none of the following is available:
- **Volumetric TSDF fusion** (accumulate depth maps into a TSDF grid across views) — no TSDFVolume API.
- **Mesh extraction from the fused representation** — no point-cloud→mesh extraction API.
- So there is no way to implement "TSDF → mesh" with this Open3D build, because neither the TSDF integration nor the mesh extraction exists in the installed library.

The existing dense path (`engine/pipeline/stages/s07_dense.py` → `run_dense` → `_dense_mvs_stage` [COLMAP dense MVS] + `_plane_sweep_stage` [CPU plane-sweep MVS] + fusion) is **photogrammetric point-cloud-based**, not volumetric TSDF. That path already consumes metric depth maps + camera poses + intrinsics and produces dense points + a mesh (via the mesh stage, `run_mesh` → `_mesh_stage`). So the repository already has a dense reconstruction path — it just isn't TSDF.

## Honest options (not a recommendation to manufacture)

1. **Declare §4 BLOCKED** with this finding: the installed Open3D 0.20.0 build lacks TSDF + mesh-extraction APIs, so TSDF production integration cannot be implemented as specified. The "actual reconstruction job executes it" bar cannot be met because the library capability is absent. This is a real, verifiable blocker, not a delay tactic.
2. **Re-scope §4 to use the available Open3D APIs** (per-view RGBD→point cloud) — but that duplicates the existing `depth_map_to_points` + fusion path (already photogrammetric point cloud), so it would not be a new TSDF capability; it would be redundant with committed code.
3. **Find a TSDF-capable library** (different Open3D build/version that exposes TSDF, or another volumetric fusion library) — this is a dependency decision requiring validation (installability on the RTX 4050 / 6 GB / Windows environment, VRAM/RAM impact, licence). Not started; would need its own evaluation.
4. **Decide that TSDF is not the right dense representation** for this pipeline and use the existing photogrammetric dense path (COLMAP dense MVS / CPU plane sweep / fusion / mesh) as the dense output — but that's a product/architecture decision about whether volumetric TSDF is wanted at all.

## What was verified (evidence)

- `import open3d; open3d.__version__` → `0.20.0`
- `hasattr(open3d.integration, 'ScalableTSDFVolume')` → ModuleNotFoundError (no `open3d.integration` module)
- `hasattr(open3d.t.geometry, 'TSDFVolume')` → False
- `hasattr(open3d.t.geometry, 'RGBADepth')` → False
- `hasattr(open3d.geometry.TriangleMesh, 'create_from_point_cloud')` → False
- `hasattr(open3d.geometry.RGBDImage, 'create_from_color_and_depth')` → True
- `hasattr(open3d.geometry.PointCloud, 'create_from_rgbd_image')` → True
- `hasattr(open3d.geometry.PointCloud, 'create_from_depth_image')` → True

## Decision requested / ownership

This is a blocker report, not an implementation. Per the directive, §4 says "Coordinate canonical geometry interfaces with Claude." TSDF would consume camera poses + depth (geometry interfaces) — so Claude (geometry owner) + the Open3D capability finding together determine whether §4 can proceed. Buffy (perception/execution) owns the integration layer but cannot implement TSDF without a TSDF-capable library.

**Recommendation (honest, not a decision):** Mark §4 BLOCKED pending either (a) a TSDF-capable Open3D build/library being made available and validated on the target environment, or (b) a product decision to drop the TSDF requirement and use the existing photogrammetric dense path. Do not start a TSDF implementation that imports a nonexistent API.
