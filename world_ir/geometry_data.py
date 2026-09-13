"""Real geometry payloads (P0.10/P0.11: "Geometry currently stores only
vertex_count, no actual vertex buffer").

`Geometry.data_uri`/`data_hash` (world_ir/schema_v1.py) were already
designed for this -- a reference to a real geometry artifact -- but
nothing ever wrote one. This module is the payload format
(`PointCloudData`, a deterministic binary encoding) plus
`world_ir.artifact_store.ArtifactStore`, the thing `data_uri` points
into.

Point clouds first, not meshes: `evidence/promote_planes.py` already has
real reconstructed inlier positions in hand at promotion time (used
today only for their bounding box) -- that is real geometry sitting
unused, not a new capability to build from scratch. A triangulated mesh
representation is a real, separate, larger follow-on (needs actual
surface reconstruction, not just point storage).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Sequence, Tuple

Point = Tuple[float, float, float]

_MAGIC = b"RESPC001"  # Reality Engine Sparse Point Cloud, format v1


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
