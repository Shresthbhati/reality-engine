"""Real geometry payloads (P0.10/P0.11: "Geometry currently stores only
vertex_count, no actual vertex buffer").

`Geometry.data_uri`/`data_hash` (world_ir/schema_v1.py) were already
designed for this -- a reference to a real geometry artifact -- but
nothing ever wrote one. This module is the payload format
(`PointCloudData` and `MeshData`, deterministic binary encodings) plus
`world_ir.artifact_store.ArtifactStore`, the thing `data_uri` points
into.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Sequence, Tuple

Point = Tuple[float, float, float]
Triangle = Tuple[int, int, int]

_MAGIC = b"RESPC001"  # Reality Engine Sparse Point Cloud, format v1
_MESH_MAGIC = b"RESMSH01"  # Reality Engine Surface Mesh, format v1


@dataclass(frozen=True)
class PointCloudData:
    """An ordered, real set of 3D points. Order is preserved (not
    resorted) so `len(points) == vertex_count` stays meaningful and a
    caller's own point identity/order survives a round trip."""

    points: Tuple[Point, ...]

    def __post_init__(self):
        if not isinstance(self.points, tuple):
            object.__setattr__(self, "points", tuple(self.points))

    def to_bytes(self) -> bytes:
        """Deterministic binary encoding: magic, point count (uint32 LE),
        then count * 3 float64 LE. float64 (not float32) because these
        are real reconstructed coordinates, not a rendering approximation
        -- precision loss here would silently corrupt downstream
        measurements."""
        body = struct.pack(f"<{len(self.points) * 3}d", *(c for p in self.points for c in p))
        return _MAGIC + struct.pack("<I", len(self.points)) + body

    @staticmethod
    def from_bytes(data: bytes) -> "PointCloudData":
        if data[:8] != _MAGIC:
            raise ValueError(f"not a PointCloudData payload (bad magic {data[:8]!r})")
        (count,) = struct.unpack_from("<I", data, 8)
        flat = struct.unpack_from(f"<{count * 3}d", data, 12)
        points = tuple((flat[i], flat[i + 1], flat[i + 2]) for i in range(0, len(flat), 3))
        return PointCloudData(points=points)

    @staticmethod
    def from_positions(positions: Sequence[Point]) -> "PointCloudData":
        return PointCloudData(points=tuple(positions))


@dataclass(frozen=True)
class MeshData:
    """Deterministic binary encoding of a 3D triangulated surface mesh.

    Vertices are stored as float64 LE coordinates for true metric precision.
    Triangles are stored as uint32 LE indices.
    Order is strictly preserved so vertex and face numbering matches B-Rep topology.
    """

    vertices: Tuple[Point, ...]
    triangles: Tuple[Triangle, ...]

    def __post_init__(self):
        if not isinstance(self.vertices, tuple):
            object.__setattr__(self, "vertices", tuple(self.vertices))
        if not isinstance(self.triangles, tuple):
            object.__setattr__(self, "triangles", tuple(self.triangles))

    def to_bytes(self) -> bytes:
        """Binary encoding: magic (8B), vertex count (uint32 LE), triangle count (uint32 LE),
        vertex coordinates (V * 3 float64 LE), triangle indices (F * 3 uint32 LE)."""
        v_body = struct.pack(f"<{len(self.vertices) * 3}d", *(c for v in self.vertices for c in v))
        f_body = struct.pack(f"<{len(self.triangles) * 3}I", *(idx for f in self.triangles for idx in f))
        header = _MESH_MAGIC + struct.pack("<II", len(self.vertices), len(self.triangles))
        return header + v_body + f_body

    @staticmethod
    def from_bytes(data: bytes) -> "MeshData":
        if data[:8] != _MESH_MAGIC:
            raise ValueError(f"not a MeshData payload (bad magic {data[:8]!r})")
        v_count, f_count = struct.unpack_from("<II", data, 8)
        offset = 16
        v_flat = struct.unpack_from(f"<{v_count * 3}d", data, offset)
        offset += v_count * 3 * 8
        f_flat = struct.unpack_from(f"<{f_count * 3}I", data, offset)
        vertices = tuple((v_flat[i], v_flat[i + 1], v_flat[i + 2]) for i in range(0, len(v_flat), 3))
        triangles = tuple((int(f_flat[i]), int(f_flat[i + 1]), int(f_flat[i + 2])) for i in range(0, len(f_flat), 3))
        return MeshData(vertices=vertices, triangles=triangles)

    @staticmethod
    def from_arrays(vertices: Sequence[Point], triangles: Sequence[Triangle]) -> "MeshData":
        return MeshData(vertices=tuple(vertices), triangles=tuple(triangles))

    def to_ply(self) -> bytes:
        """Export as standard ASCII Stanford PLY format."""
        lines = [
            "ply",
            "format ascii 1.0",
            f"element vertex {len(self.vertices)}",
            "property float x",
            "property float y",
            "property float z",
            f"element face {len(self.triangles)}",
            "property list uchar int vertex_indices",
            "end_header",
        ]
        for v in self.vertices:
            lines.append(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
        for f in self.triangles:
            lines.append(f"3 {f[0]} {f[1]} {f[2]}")
        return "\n".join(lines).encode("utf-8")
