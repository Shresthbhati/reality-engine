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
from typing import List, Optional, Sequence, Tuple

Point = Tuple[float, float, float]
Triangle = Tuple[int, int, int]
Vert = Point
Tri = Triangle
UV = Tuple[float, float]

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


@dataclass(frozen=True, init=False)
class MeshData:
    """A real triangle mesh. `faces` indices are into `vertices`;
    `normals` (when present) is per-vertex, same length as vertices."""

    vertices: Tuple[Vert, ...]
    faces: Tuple[Tri, ...]
    normals: Optional[Tuple[Vert, ...]] = None
    uvs: Optional[Tuple[UV, ...]] = None
    units: str = "meter"
    frame: str = "world"

    def __init__(self, vertices, faces=None, normals=None, uvs=None,
                 units: str = "meter", frame: str = "world", *, triangles=None):
        # `triangles` is the Rule-7 spelling of `faces`; accept either.
        if faces is None:
            faces = triangles if triangles is not None else ()
        elif triangles is not None:
            raise TypeError("pass either faces or triangles, not both")
        for k, v in (("vertices", vertices), ("faces", faces), ("normals", normals),
                     ("uvs", uvs), ("units", units), ("frame", frame)):
            object.__setattr__(self, k, v if v is None or isinstance(v, str) or isinstance(v, tuple) else tuple(v))
        self.__post_init__()

    @property
    def triangles(self) -> Tuple[Tri, ...]:
        return self.faces

    @staticmethod
    def from_arrays(vertices: Sequence[Vert], triangles: Sequence[Tri]) -> "MeshData":
        return MeshData(vertices=tuple(vertices), faces=tuple(triangles))

    def to_ply(self) -> bytes:
        return self.to_ply_bytes()

    def __post_init__(self) -> None:
        if self.normals is not None and len(self.normals) != len(self.vertices):
            raise ValueError(
                f"normals ({len(self.normals)}) must be per-vertex "
                f"({len(self.vertices)})"
            )
        if self.uvs is not None and len(self.uvs) != len(self.vertices):
            raise ValueError(
                f"uvs ({len(self.uvs)}) must be per-vertex "
                f"({len(self.vertices)})"
            )
        n = len(self.vertices)
        for fi, (a, b, c) in enumerate(self.faces):
            for idx in (a, b, c):
                if not 0 <= idx < n:
                    raise ValueError(
                        f"face {fi} references vertex {idx} of {n}"
                    )

    # -- integrity ---------------------------------------------------------

    def is_watertight_ish(self) -> bool:
        """Cheap structural check (every edge shared by exactly 2 faces in
        a closed manifold). Informational only -- reconstructed meshes are
        usually NOT closed after trimming; this reports, never enforces."""
        from collections import Counter

        edges = Counter()
        for a, b, c in self.faces:
            for u, v in ((a, b), (b, c), (c, a)):
                edges[(min(u, v), max(u, v))] += 1
        return bool(self.faces) and all(k == 2 for k in edges.values())

    def bounds(self) -> Tuple[Vert, Vert]:
        if not self.vertices:
            raise ValueError("empty mesh has no bounds")
        xs, ys, zs = zip(*self.vertices)
        return (
            (min(xs), min(ys), min(zs)),
            (max(xs), max(ys), max(zs)),
        )

    # -- deterministic artifact encoding ------------------------------------

    def to_bytes(self) -> bytes:
        """Deterministic encoding: magic, counts (uint32 LE), float64 LE
        vertices [then normals], then uint32 LE face indices. float64
        because these are reconstructed coordinates that must survive a
        hash-identical round trip (same reasoning as PointCloudData)."""
        flags = (1 if self.normals else 0) | (2 if self.uvs else 0)
        parts = [
            _MESH_MAGIC,
            struct.pack("<III", len(self.vertices), len(self.faces), flags),
        ]
        for v in self.vertices:
            parts.append(struct.pack("<3d", *v))
        if self.normals is not None:
            for n in self.normals:
                parts.append(struct.pack("<3d", *n))
        if self.uvs is not None:
            for uv in self.uvs:
                parts.append(struct.pack("<2d", float(uv[0]), float(uv[1])))
        for f in self.faces:
            parts.append(struct.pack("<3I", *f))
        return b"".join(parts)

    @staticmethod
    def from_bytes(data: bytes) -> "MeshData":
        if data[:8] != _MESH_MAGIC:
            raise ValueError(f"not a MeshData payload (bad magic {data[:8]!r})")
        nv, nf, flags = struct.unpack_from("<III", data, 8)
        has_normals = bool(flags & 1)
        has_uvs = bool(flags & 2)
        off = 20
        vertices: List[Vert] = []
        for _ in range(nv):
            vertices.append(struct.unpack_from("<3d", data, off))
            off += 24
        normals: Optional[List[Vert]] = None
        if has_normals:
            normals = []
            for _ in range(nv):
                normals.append(struct.unpack_from("<3d", data, off))
                off += 24
        uvs: Optional[List[UV]] = None
        if has_uvs:
            uvs = []
            for _ in range(nv):
                uvs.append(struct.unpack_from("<2d", data, off))
                off += 16
        faces: List[Tri] = []
        for _ in range(nf):
            faces.append(struct.unpack_from("<3I", data, off))
            off += 12
        return MeshData(
            vertices=tuple(vertices),
            faces=tuple(faces),
            normals=tuple(normals) if normals is not None else None,
            uvs=tuple(uvs) if uvs is not None else None,
        )

    # -- PLY (binary little endian) ----------------------------------------

    def to_ply_bytes(self, with_colors: Optional[Sequence[Vert]] = None) -> bytes:
        """Minimal binary_little_endian PLY. `with_colors` (0..1 float or
        0..255 int per vertex) is written as uchar rgb when given."""
        if with_colors is not None and len(with_colors) != len(self.vertices):
            raise ValueError("with_colors must be per-vertex")
        header_lines = [
            "ply",
            "format binary_little_endian 1.0",
            "comment reality-engine MeshData export",
            f"element vertex {len(self.vertices)}",
            "property float x", "property float y", "property float z",
        ]
        if self.normals is not None:
            header_lines += ["property float nx", "property float ny", "property float nz"]
        if with_colors is not None:
            header_lines += ["property uchar red", "property uchar green", "property uchar blue"]
        header_lines += [
            f"element face {len(self.faces)}",
            "property list uchar uint vertex_indices",
            "end_header",
        ]
        header = ("\n".join(header_lines) + "\n").encode("ascii")

        body = bytearray()
        for i, v in enumerate(self.vertices):
            body += struct.pack("<3f", *v)
            if self.normals is not None:
                body += struct.pack("<3f", *self.normals[i])
            if with_colors is not None:
                c = with_colors[i]
                rgb = tuple(
                    int(round(255 * ch)) if 0.0 <= ch <= 1.0 else max(0, min(255, int(ch)))
                    for ch in c
                )
                body += struct.pack("<3B", *rgb)
        for a, b, c in self.faces:
            body += struct.pack("<B3I", 3, a, b, c)
        return header + bytes(body)

    @staticmethod
    def from_ply_bytes(data: bytes) -> "MeshData":
        """Read the subset of binary_little_endian PLY that our own
        writer (and COLMAP's meshers) emit: x/y/z[/nx/ny/nz] vertices and
        triangle faces. Raises ValueError on anything else -- a silently
        half-parsed mesh would corrupt downstream measurement."""
        text, _, body = data.partition(b"end_header\n")
        if not _:
            raise ValueError("PLY header has no 'end_header' terminator")
        header = text.decode("ascii", errors="replace").splitlines()
        fmt = next((l for l in header if l.startswith("format")), "")
        if "binary_little_endian" not in fmt:
            raise ValueError(f"only binary_little_endian PLY supported, got {fmt!r}")

        # (type, name) pairs for the vertex element; stride computed from
        # actual types so uchar color properties do not corrupt offsets.
        _PLY_SIZES = {"float": 4, "double": 8, "uchar": 1, "char": 1,
                      "ushort": 2, "short": 2, "uint": 4, "int": 4}
        vprops: List[tuple] = []  # (type, name)
        nv = 0
        in_vertex = False
        for line in header:
            if line.startswith("element"):
                in_vertex = line.split()[1] == "vertex"
                if in_vertex:
                    nv = int(line.split()[2])
                continue
            if in_vertex and line.startswith("property"):
                parts = line.split()
                if len(parts) != 3 or parts[1] == "list":
                    raise ValueError(f"unsupported vertex property: {line!r}")
                vprops.append((parts[1], parts[2]))
        names = {name for _, name in vprops}
        if not {"x", "y", "z"} <= names:
            raise ValueError(f"PLY vertex element lacks xyz: {names}")

        stride = sum(_PLY_SIZES[t] for t, _ in vprops)
        expected = nv * stride
        if len(body) < expected:
            raise ValueError(f"PLY body truncated: {len(body)} < {expected} bytes")

        names_lower = {name: t for t, name in vprops if name in ("x", "y", "z", "nx", "ny", "nz")}
        has_normals = {"nx", "ny", "nz"} <= set(names_lower)
        normals: Optional[List[Vert]] = [] if has_normals else None
        vertices: List[Vert] = []
        for i in range(nv):
            off = i * stride
            rec = {}
            for t, name in vprops:
                if t == "float":
                    (val,) = struct.unpack_from("<f", body, off)
                elif t == "uchar":
                    (val,) = struct.unpack_from("<B", body, off)
                elif t == "double":
                    (val,) = struct.unpack_from("<d", body, off)
                else:
                    raise ValueError(f"unsupported property type {t!r} for {name!r}")
                rec[name] = val
                off += _PLY_SIZES[t]
            vertices.append((rec["x"], rec["y"], rec["z"]))
            if normals is not None:
                normals.append((rec["nx"], rec["ny"], rec["nz"]))

        # faces: list property -- count type read from the header (COLMAP
        # writes `property list int int vertex_indices`; our writer uses
        # uchar). Indices accepted as uint/int, triangles only.
        face_count_fmt = "<B"  # default uchar
        in_face = False
        for line in header:
            if line.startswith("element"):
                in_face = line.split()[1] == "face"
                continue
            if in_face and line.startswith("property list"):
                parts = line.split()
                # parts: ['property', 'list', <count_type>, <index_type>, name]
                if len(parts) >= 5:
                    face_count_fmt = {"uchar": "<B", "char": "<b", "ushort": "<H",
                                      "short": "<h", "uint": "<I", "int": "<i"}.get(
                        parts[2], "<B")
        count_size = struct.calcsize(face_count_fmt)
        off = nv * stride
        faces: List[Tri] = []
        while off + count_size <= len(body):
            (count,) = struct.unpack_from(face_count_fmt, body, off)
            off += count_size
            if count == 0:
                break  # declared empty face element
            if count != 3:
                raise ValueError(f"non-triangle face (count={count}) -- refusing to guess")
            if off + 12 > len(body):
                break
            faces.append(struct.unpack_from("<3I", body, off))
            off += 12
        return MeshData(
            vertices=tuple(vertices),
            faces=tuple(faces),
            normals=tuple(normals) if normals else None,
        )
