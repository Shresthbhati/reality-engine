"""Mesh quality validation (P6-03): honest measurement, not a pass/fail gate.

`MeshData` existing (mesh.py) is not the same claim as the mesh being
GOOD -- degenerate triangles, non-manifold edges, bad normals, holes,
disconnected junk, and self-intersections all produce a structurally
valid `MeshData` that is still garbage for downstream use. This module
computes real counts for each of those checks and returns them in a
frozen `MeshQualityReport`. It never collapses them into a boolean --
callers decide what counts matter for their use case.

Explicitly OUT OF SCOPE (see task ledger P6-03):
  - Mesh REPAIR (removing degenerate triangles, closing holes, etc.) --
    this module only measures.
  - Exact/robust self-intersection testing (e.g. Moller-Trumbore over
    all triangle pairs, or a BVH-accelerated exact test). We do a
    scoped-down, documented approximation: bounding-box overlap between
    non-adjacent triangles, which is a cheap necessary-but-not-sufficient
    proxy (flags candidate overlapping regions, will have false
    positives, will miss thin/edge-case exact intersections). Good
    enough to catch gross overlap (e.g. two blobs pushed into each
    other); not a substitute for a real intersection test.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

from .mesh import MeshData, Vert

_Edge = Tuple[int, int]


def _edge_key(a: int, b: int) -> _Edge:
    return (a, b) if a < b else (b, a)


def _sub(u: Vert, v: Vert) -> Vert:
    return (u[0] - v[0], u[1] - v[1], u[2] - v[2])


def _cross(u: Vert, v: Vert) -> Vert:
    return (
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    )


def _norm(u: Vert) -> float:
    return (u[0] ** 2 + u[1] ** 2 + u[2] ** 2) ** 0.5


@dataclass(frozen=True)
class MeshQualityReport:
    """Measured facts about a mesh's structural quality. Every field is a
    real count/list, never a fabricated pass/fail. `degenerate_area_eps`
    records the threshold actually used, so a report is reproducible."""

    n_triangles: int
    n_vertices: int
    degenerate_area_eps: float
    degenerate_triangle_indices: Tuple[int, ...]
    non_manifold_edges: Tuple[_Edge, ...]
    boundary_edges: Tuple[_Edge, ...]
    bad_normal_vertex_indices: Tuple[int, ...]
    n_connected_components: int
    candidate_self_intersection_pairs: Tuple[Tuple[int, int], ...]
    extent_m: Tuple[float, float, float]

    @property
    def n_degenerate_triangles(self) -> int:
        return len(self.degenerate_triangle_indices)

    @property
    def n_non_manifold_edges(self) -> int:
        return len(self.non_manifold_edges)

    @property
    def n_boundary_edges(self) -> int:
        return len(self.boundary_edges)

    @property
    def n_bad_normals(self) -> int:
        return len(self.bad_normal_vertex_indices)

    @property
    def has_holes(self) -> bool:
        return self.n_boundary_edges > 0

    def to_dict(self) -> dict:
        return {
            "n_triangles": self.n_triangles,
            "n_vertices": self.n_vertices,
            "degenerate_area_eps": self.degenerate_area_eps,
            "n_degenerate_triangles": self.n_degenerate_triangles,
            "degenerate_triangle_indices": list(self.degenerate_triangle_indices),
            "n_non_manifold_edges": self.n_non_manifold_edges,
            "non_manifold_edges": [list(e) for e in self.non_manifold_edges],
            "n_boundary_edges": self.n_boundary_edges,
            "boundary_edges": [list(e) for e in self.boundary_edges],
            "has_holes": self.has_holes,
            "n_bad_normals": self.n_bad_normals,
            "bad_normal_vertex_indices": list(self.bad_normal_vertex_indices),
            "n_connected_components": self.n_connected_components,
            "n_candidate_self_intersection_pairs": len(self.candidate_self_intersection_pairs),
            "candidate_self_intersection_pairs": [
                list(p) for p in self.candidate_self_intersection_pairs
            ],
            "extent_m": list(self.extent_m),
        }


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def validate_mesh(mesh: MeshData, degenerate_area_eps: float = 1e-12) -> MeshQualityReport:
    """Compute a `MeshQualityReport` for `mesh`. `degenerate_area_eps` is
    the minimum triangle area (in mesh units^2) below which a triangle is
    counted degenerate; the default is a near-zero floor (catches
    coincident/collinear vertices, not "small but real" triangles)."""
    verts = mesh.vertices
    faces = mesh.faces
    n_tri = len(faces)
    n_vert = len(verts)

    # -- degenerate triangles (zero/near-zero area) -------------------------
    degenerate: List[int] = []
    tri_normals: List[Vert] = []  # face normal, unnormalized cross product
    for fi, (a, b, c) in enumerate(faces):
        e1 = _sub(verts[b], verts[a])
        e2 = _sub(verts[c], verts[a])
        cr = _cross(e1, e2)
        area = _norm(cr) * 0.5
        tri_normals.append(cr)
        if area < degenerate_area_eps:
            degenerate.append(fi)

    # -- edge -> triangle count (non-manifold / boundary) --------------------
    edge_faces: Dict[_Edge, List[int]] = defaultdict(list)
    for fi, (a, b, c) in enumerate(faces):
        for u, v in ((a, b), (b, c), (c, a)):
            edge_faces[_edge_key(u, v)].append(fi)

    non_manifold = tuple(sorted(e for e, fs in edge_faces.items() if len(fs) > 2))
    boundary = tuple(sorted(e for e, fs in edge_faces.items() if len(fs) == 1))

    # -- bad/missing normals --------------------------------------------------
    bad_normals: List[int] = []
    if mesh.normals is not None:
        for vi, n in enumerate(mesh.normals):
            length = _norm(n)
            if not (length == length) or length < 1e-9:  # NaN check + near-zero
                bad_normals.append(vi)

    # -- connected components (triangle adjacency via shared vertices) -------
    uf = _UnionFind(n_vert)
    for a, b, c in faces:
        uf.union(a, b)
        uf.union(b, c)
    touched_verts = {v for tri in faces for v in tri}
    n_components = len({uf.find(v) for v in touched_verts}) if touched_verts else 0

    # -- candidate self-intersections: AABB overlap of non-adjacent tris -----
    # ponytail: O(n^2) triangle-pair scan, fine for validation-sized meshes;
    # switch to a BVH/grid broadphase if this becomes a bottleneck on large
    # production meshes.
    bboxes: List[Tuple[Vert, Vert]] = []
    for a, b, c in faces:
        xs = (verts[a][0], verts[b][0], verts[c][0])
        ys = (verts[a][1], verts[b][1], verts[c][1])
        zs = (verts[a][2], verts[b][2], verts[c][2])
        bboxes.append(((min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))))

    def _overlap(lo1, hi1, lo2, hi2) -> bool:
        return all(lo1[k] <= hi2[k] and lo2[k] <= hi1[k] for k in range(3))

    candidates: List[Tuple[int, int]] = []
    for i in range(n_tri):
        vi = set(faces[i])
        for j in range(i + 1, n_tri):
            if vi & set(faces[j]):
                continue  # adjacent triangles sharing a vertex/edge -- not an intersection
            if _overlap(bboxes[i][0], bboxes[i][1], bboxes[j][0], bboxes[j][1]):
                candidates.append((i, j))

    (mnx, mny, mnz), (mxx, mxy, mxz) = mesh.bounds() if verts else ((0.0,) * 3, (0.0,) * 3)
    extent = (mxx - mnx, mxy - mny, mxz - mnz)

    return MeshQualityReport(
        n_triangles=n_tri,
        n_vertices=n_vert,
        degenerate_area_eps=degenerate_area_eps,
        degenerate_triangle_indices=tuple(degenerate),
        non_manifold_edges=non_manifold,
        boundary_edges=boundary,
        bad_normal_vertex_indices=tuple(bad_normals),
        n_connected_components=n_components,
        candidate_self_intersection_pairs=tuple(candidates),
        extent_m=extent,
    )
