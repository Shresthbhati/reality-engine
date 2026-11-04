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

    Optional normals (per-vertex) and UVs (per-vertex) may be present;
    `units` and `frame` record the coordinate system for provenance.
    """

    vertices: Tuple[Point, ...]
    triangles: Tuple[Triangle, ...]
    normals: Optional[Tuple[Point, ...]] = None
    uvs: Optional[Tuple[Tuple[float, float], ...]] = None
    units: str = "meter"
    frame: str = "world"

    def __post_init__(self):
        if not isinstance(self.vertices, tuple):
            object.__setattr__(self, "vertices", tuple(self.vertices))
        if not isinstance(self.triangles, tuple):
            object.__setattr__(self, "triangles", tuple(self.triangles))
        if self.normals is not None:
            object.__setattr__(self, "normals", tuple(self.normals))
        if self.uvs is not None:
            object.__setattr__(self, "uvs", tuple(self.uvs))
        if len(self.normals or ()) != len(self.vertices):
            raise ValueError(
                f"normals ({len(self.normals or ())}) must be per-vertex "
                f"({len(self.vertices)})"
            )
        if len(self.uvs or ()) != len(self.vertices):
            raise ValueError(
                f"uvs ({len(self.uvs or ())}) must be per-vertex ({len(self.vertices)})"
            )
        n = len(self.vertices)
        for fi, (a, b, c) in enumerate(self.triangles):
            for idx in (a, b, c):
                if not 0 <= idx < n:
                    raise ValueError(f"face {fi} references vertex {idx} of {n}")

    # -- deterministic artifact encoding ------------------------------------

    def to_bytes(self) -> bytes:
        """Binary encoding: magic (8B), vertex count (uint32 LE),
        triangle count (uint32 LE), vertex coordinates (V * 3 float64 LE),
        triangle indices (F * 3 uint32 LE), optional normals (V * 3
        float64 LE), optional UVs (V * 2 float64 LE)."""
        flags = (1 if self.normals is not None else 0) | (2 if self.uvs is not None else 0)
        v_body = struct.pack(f"<{len(self.vertices) * 3}d", *(c for v in self.vertices for c in v))
        f_body = struct.pack(f"<{len(self.triangles) * 3}I", *(idx for f in self.triangles for idx in f))
        header = _MESH_MAGIC + struct.pack("<IIB", len(self.vertices), len(self.triangles), flags)
        body = v_body + f_body
        if self.normals is not None:
            body += struct.pack(f"<{len(self.normals) * 3}d", *(c for n in self.normals for c in n))
        if self.uvs is not None:
            body += struct.pack(f"<{len(self.uvs) * 2}d", *(c for uv in self.uvs for c in uv))
        return header + body

    @staticmethod
    def from_bytes(data: bytes) -> "MeshData":
        if data[:8] != _MESH_MAGIC:
            raise ValueError(f"not a MeshData payload (bad magic {data[:8]!r})")
        v_count, f_count, flags = struct.unpack_from("<IIB", data, 8)
        has_normals = bool(flags & 1)
        has_uvs = bool(flags & 2)
        offset = 9
        v_flat = struct.unpack_from(f"<{v_count * 3}d", data, offset)
        offset += v_count * 3 * 8
        f_flat = struct.unpack_from(f"<{f_count * 3}I", data, offset)
        offset += f_count * 3 * 4
        vertices = tuple((v_flat[i], v_flat[i + 1], v_flat[i + 2]) for i in range(0, len(v_flat), 3))
        triangles = tuple((int(f_flat[i]), int(f_flat[i + 1]), int(f_flat[i + 2])) for i in range(0, len(f_flat), 3))
        normals = None
        if has_normals:
            n_flat = struct.unpack_from(f"<{v_count * 3}d", data, offset)
            offset += v_count * 3 * 8
            normals = tuple((n_flat[i], n_flat[i + 1], n_flat[i + 2]) for i in range(0, len(n_flat), 3))
        uvs = None
        if has_uvs:
            u_flat = struct.unpack_from(f"<{v_count * 2}d", data, offset)
            uvs = tuple((u_flat[i], u_flat[i + 1]) for i in range(0, len(u_flat), 2))
        return MeshData(vertices=vertices, triangles=triangles, normals=normals, uvs=uvs)

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
        ]
        if self.normals is not None:
            lines += [
                "property float nx",
                "property float ny",
                "property float nz",
            ]
        if self.uvs is not None:
            lines += ["property float u", "property float v"]
        lines += [
            f"element face {len(self.triangles)}",
            "property list uchar int vertex_indices",
            "end_header",
        ]
        for v in self.vertices:
            lines.append(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
            if self.normals is not None:
                i = self.vertices.index(v)
                lines.append(f" {self.normals[i][0]:.6f} {self.normals[i][1]:.6f} {self.normals[i][2]:.6f}")
            if self.uvs is not None:
                i = self.vertices.index(v)
                lines.append(f" {self.uvs[i][0]:.6f} {self.uvs[i][1]:.6f}")
        for f in self.triangles:
            lines.append(f"3 {f[0]} {f[1]} {f[2]}")
        return "\n".join(lines).encode("utf-8")

    # -- integrity ----------------------------------------------------------

    def is_watertight_ish(self) -> bool:
        """Cheap structural check (every edge shared by exactly 2 faces in
        a closed manifold). Informational only -- reconstructed meshes are
        usually NOT closed after trimming; this reports, never enforces."""
        from collections import Counter

        edges = Counter()
        for a, b, c in self.triangles:
            for u, v in ((a, b), (b, c), (c, a)):
                edges[(min(u, v), max(u, v))] += 1
        return bool(self.triangles) and all(k == 2 for k in edges.values())

    def bounds(self) -> Tuple[Point, Point]:
        if not self.vertices:
            raise ValueError("empty mesh has no bounds")
        xs, ys, zs = zip(*self.vertices)
        return (
            (min(xs), min(ys), min(zs)),
            (max(xs), max(ys), max(zs)),
        )
