"""Surface reconstruction (P0.11-P0.13): oriented point cloud -> MeshData.

Mature backend first: the configured COLMAP binary's CPU poisson_mesher
(surface.py). Preprocessing (downsample, outlier filter, camera-oriented
normals) is deterministic scipy code (preprocess.py) so the same input
cloud always yields the same oriented cloud -- the mesher itself is
allowed nondeterministic thread scheduling, but its output is hashed as
a real artifact either way.
"""

from .mesh import MeshData, mesh_summary

__all__ = ["MeshData", "mesh_summary"]
