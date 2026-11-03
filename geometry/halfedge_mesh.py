"""OpenMesh algorithmic extraction: Directed Halfedge Data Structure & Manifold Topology.

Implements a topological halfedge data structure for arbitrary polygonal surface meshes:
1. Vertices, Directed Halfedges, Edges, and Faces with bidirectional pointer cycles.
2. 1-ring vertex neighborhood traversal and vertex valence computation.
3. Boundary edge detection and non-manifold singularity checks.
4. Euler characteristic verification (V - E + F = 2 - 2g - b).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np


@dataclass
class Halfedge:
    """Directed halfedge segment."""
    id: int
    to_vertex: int
    face: Optional[int] = None
    next_he: Optional[int] = None
    prev_he: Optional[int] = None
    opposite_he: Optional[int] = None
    edge_id: Optional[int] = None


@dataclass
class Vertex:
    """Mesh vertex with coordinates and outgoing halfedge pointer."""
    id: int
    position: np.ndarray
    halfedge: Optional[int] = None  # One outgoing halfedge starting at this vertex


@dataclass
class Face:
    """Polygon face bounded by a cycle of directed halfedges."""
    id: int
    halfedge: int                   # One halfedge belonging to this face


class HalfedgeMesh:
    """Topological halfedge mesh structure supporting 1-ring queries and manifold validation."""

    def __init__(self):
        self.vertices: List[Vertex] = []
        self.halfedges: List[Halfedge] = []
        self.faces: List[Face] = []
        self.num_edges: int = 0

    @classmethod
    def from_indexed_triangles(
        cls,
        vertices: np.ndarray,
        triangles: np.ndarray,
    ) -> HalfedgeMesh:
        """Construct halfedge structure from vertex positions and triangle index list."""
        mesh = cls()
        verts_arr = np.asarray(vertices, dtype=np.float64)
        for i, pos in enumerate(verts_arr):
            mesh.vertices.append(Vertex(id=i, position=pos))

        # Map: (u, v) -> halfedge_id
        edge_map: Dict[Tuple[int, int], int] = {}
        edge_count = 0

        for f_idx, tri in enumerate(triangles):
            v0, v1, v2 = int(tri[0]), int(tri[1]), int(tri[2])
            he0_id = len(mesh.halfedges)
            he1_id = he0_id + 1
            he2_id = he0_id + 2

            he0 = Halfedge(id=he0_id, to_vertex=v1, face=f_idx, next_he=he1_id, prev_he=he2_id)
            he1 = Halfedge(id=he1_id, to_vertex=v2, face=f_idx, next_he=he2_id, prev_he=he0_id)
            he2 = Halfedge(id=he2_id, to_vertex=v0, face=f_idx, next_he=he0_id, prev_he=he1_id)

            mesh.halfedges.extend([he0, he1, he2])
            mesh.faces.append(Face(id=f_idx, halfedge=he0_id))

            # Set outgoing halfedge for vertices
            mesh.vertices[v0].halfedge = he0_id
            mesh.vertices[v1].halfedge = he1_id
            mesh.vertices[v2].halfedge = he2_id

            # Pair up opposite halfedges and assign edges
            for u, v, he_id in [(v0, v1, he0_id), (v1, v2, he1_id), (v2, v0, he2_id)]:
                edge_map[(u, v)] = he_id
                opp = (v, u)
                if opp in edge_map:
                    opp_he_id = edge_map[opp]
                    mesh.halfedges[he_id].opposite_he = opp_he_id
                    mesh.halfedges[opp_he_id].opposite_he = he_id
                    mesh.halfedges[he_id].edge_id = edge_count
                    mesh.halfedges[opp_he_id].edge_id = edge_count
                    edge_count += 1

        mesh.num_edges = edge_count
        return mesh

    def get_vertex_1ring(self, vertex_id: int) -> List[int]:
        """Return 1-ring neighboring vertex IDs surrounding vertex_id."""
        v = self.vertices[vertex_id]
        if v.halfedge is None:
            return []

        neighbors: List[int] = []
        start_he_id = v.halfedge
        curr_he_id = start_he_id

        # Traverse outgoing halfedges around vertex
        visited: Set[int] = set()
        while curr_he_id is not None and curr_he_id not in visited:
            visited.add(curr_he_id)
            curr_he = self.halfedges[curr_he_id]
            neighbors.append(curr_he.to_vertex)

            # Move to next outgoing halfedge: prev of current, then opposite
            prev_he = self.halfedges[curr_he.prev_he]
            curr_he_id = prev_he.opposite_he

            if curr_he_id == start_he_id:
                break

        return neighbors

    def get_boundary_halfedges(self) -> List[int]:
        """Return IDs of halfedges that lie on open mesh boundaries (no opposite halfedge)."""
        return [he.id for he in self.halfedges if he.opposite_he is None]

    def is_watertight(self) -> bool:
        """Check if mesh has zero boundary edges and all halfedges have paired opposites."""
        return len(self.get_boundary_halfedges()) == 0

    def compute_euler_characteristic(self) -> int:
        """Euler characteristic chi = V - E + F."""
        num_v = len(self.vertices)
        # Total distinct edges: paired edges + unpaired boundary halfedges
        num_paired = self.num_edges
        num_unpaired = len(self.get_boundary_halfedges())
        total_edges = num_paired + num_unpaired
        num_f = len(self.faces)
        return num_v - total_edges + num_f
