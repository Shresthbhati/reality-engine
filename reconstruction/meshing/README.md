# meshing

Implemented (2026-09-14): point cloud -> triangle mesh -> WorldIR geometry artifact.

- `mesh.py` — `MeshData` (vertices / triangle faces / per-vertex normals),
  deterministic binary encoding (artifact-hashable), minimal binary-PLY
  writer/reader (reads COLMAP's `list int int` face headers too).
- `preprocess.py` — deterministic voxel downsample, statistical outlier
  filter, and camera-oriented PCA normals (scipy cKDTree; oriented toward
  the observing camera centers).
- `surface.py` — mature backend: the configured COLMAP binary's CPU
  `poisson_mesher`. Capability-probed (never assumed); adapts to
  density-bounded solver depth and trim-vs-sparsity behavior; honest
  `MeshingUnavailableError` / `MeshingError`, never a fabricated
  fallback mesh.

Pipeline seam: `engine/pipeline/vertical_slice.py::_mesh_stage` (stage 3.6)
meshes the fused metric cloud and persists the mesh as a content-addressed
artifact referenced by `Geometry(type=MESH, data_uri, data_hash)`.
The glTF exporter emits real TRIANGLES primitives for resolvable MESH
payloads (placeholder cube only when no real artifact exists).

Verification: `tests/test_meshing_pipeline.py` (26 deterministic tests)
plus a dependency-gated real-COLMAP test (`REALITY_COLMAP`).
