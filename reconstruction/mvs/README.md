# mvs

Multi-view stereo status (2026-09-14): the dense-geometry route currently
runs through per-view metricized monocular depth (MiDaS aligned to the
sparse SfM cloud, `reconstruction/depth_to_points.py`) fused with the
sparse cloud itself — not classic MVS.

The user's COLMAP 4.2.0 build is CPU-only ("without GPU support");
`patch_match_stereo` requires CUDA, so classic COLMAP dense MVS is
dependency-blocked in this environment and is NOT silently emulated.
When a CUDA COLMAP (or OpenMVS) is available, this directory is the
intended home for a dense-MVS backend behind the same
capability-probe / honest-failure contract as
`reconstruction/meshing/surface.py`.
