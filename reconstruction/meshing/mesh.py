"""MeshData: the real triangle-mesh artifact representation (P0.12).

`world_ir.schema_v1.Geometry` was designed for this (type=MESH with
`data_uri`/`data_hash` referencing a stored artifact) but nothing ever
produced one -- `reconstruction/meshing/` was an empty scaffold and
"vertex_count" numbers described shapes nobody stored. This module is
the payload: vertices, triangle indices, optional per-vertex normals,
plus a deterministic binary encoding (so the same mesh always hashes to
the same artifact id) and a minimal binary-PLY writer/reader (so the
COLMAP mesher output round-trips without adding a dependency).

Coordinates: vertices are WORLD-frame meters (the fused reconstruction
cloud's frame). Units and frame travel inside `to_dict()` so the
artifact is never silently frame-ambiguous.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

Vert = Tuple[float, float, float]
Tri = Tuple[int, int, int]

_MAGIC = b"RESMESH1"  # Reality Engine Mesh, format v1


@dataclass(frozen=True)
class MeshData:
    """A real triangle mesh. `faces` indices are into `vertices`;
    `normals` (when present) is per-vertex, same length as vertices."""

    vertices: Tuple[Vert, ...]
    faces: Tuple[Tri, ...]
    normals: Optional[Tuple[Vert, ...]] = None
    units: str = "meter"
    frame: str = "world"

    def __post_init__(self) -> None:
        if self.normals is not None and len(self.normals) != len(self.vertices):
            raise ValueError(
                f"normals ({len(self.normals)}) must be per-vertex "
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
        parts = [
            _MAGIC,
            struct.pack("<III", len(self.vertices), len(self.faces), 1 if self.normals else 0),
        ]
        for v in self.vertices:
            parts.append(struct.pack("<3d", *v))
        if self.normals is not None:
            for n in self.normals:
                parts.append(struct.pack("<3d", *n))
        for f in self.faces:
            parts.append(struct.pack("<3I", *f))
        return b"".join(parts)

    @staticmethod
    def from_bytes(data: bytes) -> "MeshData":
        if data[:8] != _MAGIC:
            raise ValueError(f"not a MeshData payload (bad magic {data[:8]!r})")
        nv, nf, has_normals = struct.unpack_from("<III", data, 8)
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
        faces: List[Tri] = []
        for _ in range(nf):
            faces.append(struct.unpack_from("<3I", data, off))
            off += 12
        return MeshData(
            vertices=tuple(vertices),
            faces=tuple(faces),
            normals=tuple(normals) if normals is not None else None,
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


def mesh_summary(mesh: MeshData) -> dict:
    """Observed facts for provenance/reporting. watertight is reported,
    never enforced -- trimmed Poisson output is typically open."""
    (mnx, mny, mnz), (mxx, mxy, mxz) = mesh.bounds()
    extent = (mxx - mnx, mxy - mny, mxz - mnz)
    return {
        "vertices": len(mesh.vertices),
        "faces": len(mesh.faces),
        "bounds_min": [mnx, mny, mnz],
        "bounds_max": [mxx, mxy, mxz],
        "extent_m": [round(e, 4) for e in extent],
        "watertight": mesh.is_watertight_ish(),
        "has_normals": mesh.normals is not None,
        "units": mesh.units,
        "frame": mesh.frame,
    }
