"""Pure 2-D polygon operations shared by the layers that may not import each other.

``reconstruction`` meshes room shells and ``perception`` lays rooms out; both need to triangulate a plan polygon, and
neither may import the other (tests/test_import_boundaries.py), so the geometry lives here, in the shared ``world_ir``
layer. No WorldIR types, no I/O: just arithmetic on coordinate sequences.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple


def signed_area(poly: Sequence[Sequence[float]]) -> float:
    """Shoelace area of a plan polygon; positive when the vertices run counter-clockwise."""
    return 0.5 * sum(poly[i][0] * poly[(i + 1) % len(poly)][1] - poly[(i + 1) % len(poly)][0] * poly[i][1]
                     for i in range(len(poly)))


def ear_clip(poly: Sequence[Sequence[float]]) -> List[Tuple[int, int, int]]:
    """Counter-clockwise triangles (vertex index triples) of a simple polygon; [] when it is not simple."""
    n = len(poly)
    if n < 3:
        return []
    idx = list(range(n))
    if signed_area(poly) < 0:
        idx.reverse()

    def orient(o, a, b):
        return (poly[a][0] - poly[o][0]) * (poly[b][1] - poly[o][1]) - (poly[a][1] - poly[o][1]) * (poly[b][0] - poly[o][0])

    out: List[Tuple[int, int, int]] = []
    guard = 0
    while len(idx) > 3 and guard < n * n:
        guard += 1
        for k in range(len(idx)):
            a, b, c = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
            if orient(a, b, c) <= 1e-12:
                continue
            if any(p not in (a, b, c) and orient(a, b, p) >= -1e-12 and orient(b, c, p) >= -1e-12 and orient(c, a, p) >= -1e-12
                   for p in idx):
                continue
            out.append((a, b, c))
            idx.pop(k)
            break
        else:
            return []
    if len(idx) == 3:
        out.append((idx[0], idx[1], idx[2]))
    return out
