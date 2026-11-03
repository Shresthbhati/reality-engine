"""Meshoptimizer algorithmic extraction: Quadric Decimation & Vertex Cache Optimization.

Provides high-performance mesh optimization primitives inspired by meshoptimizer:
1. Quadric Error Metric (QEM) edge collapse decimation (Garland-Heckbert formulation).
2. Forsyth-style LRU vertex cache reordering to minimize Average Cache Miss Ratio (ACMR).
3. Vertex fetch optimization to reorder vertex buffers for linear memory access.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np


@dataclass(frozen=True)
class MeshoptStats:
    """Statistics comparing mesh before and after optimization."""
    original_vertices: int
    original_triangles: int
    optimized_vertices: int
    optimized_triangles: int
    acmr_before: float
    acmr_after: float
    reduction_ratio: float


def compute_acmr(triangles: np.ndarray, cache_size: int = 16) -> float:
    """Compute the Average Cache Miss Ratio (ACMR) of a triangle index buffer.
    
    ACMR = (Total Cache Misses) / (Number of Triangles).
    A theoretical optimal strip yields ACMR ~ 0.5; unoptimized meshes typically have ACMR > 1.5.
    """
    if len(triangles) == 0:
        return 0.0

    cache: List[int] = []
    misses = 0

    for tri in triangles:
        for vert_idx in tri:
            if vert_idx in cache:
                # Cache hit: move to front (LRU)
                cache.remove(vert_idx)
                cache.insert(0, vert_idx)
            else:
                # Cache miss: insert at front, evict oldest if exceeding capacity
                misses += 1
                cache.insert(0, vert_idx)
                if len(cache) > cache_size:
                    cache.pop()

    return float(misses / len(triangles))


def optimize_vertex_cache(triangles: np.ndarray, cache_size: int = 16) -> np.ndarray:
    """Reorder triangles to optimize for GPU vertex cache locality using Forsyth's algorithm.
    
    Assigns dynamic scores to vertices based on their position in the simulated LRU cache
    and their valence (remaining active triangle count), iteratively selecting triangles
    with highest combined vertex score.
    """
    if len(triangles) <= 1:
        return triangles.copy()

    num_triangles = len(triangles)
    num_verts = int(np.max(triangles)) + 1 if num_triangles > 0 else 0

    # Build vertex to triangle adjacency
    vert_triangles: List[List[int]] = [[] for _ in range(num_verts)]
    for tri_idx, tri in enumerate(triangles):
        for v in tri:
            vert_triangles[v].append(tri_idx)

    active_tri_count = np.array([len(tris) for tris in vert_triangles], dtype=np.int32)
    emitted_triangles = np.zeros(num_triangles, dtype=bool)
    reordered_triangles = np.zeros_like(triangles)
    output_idx = 0

    # LRU cache simulation
    cache: List[int] = []

    def score_vertex(v_idx: int) -> float:
        num_active = active_tri_count[v_idx]
        if num_active <= 0:
            return -1.0

        score = 0.0
        # Cache score based on position
        if v_idx in cache:
            pos = cache.index(v_idx)
            if pos < 3:
                # Top of cache
                score = 0.75
            else:
                score = math.pow(1.0 - float(pos - 3) / float(cache_size - 3), 1.5)

        # Valence boost: vertices with few remaining triangles get priority
        valence_boost = math.pow(float(num_active), -0.5)
        return score + 2.0 * valence_boost

    # Initial vertex scores
    vertex_scores = np.array([score_vertex(v) for v in range(num_verts)], dtype=np.float64)

    # Pick seed triangle with highest sum of scores
    tri_scores = np.sum(vertex_scores[triangles], axis=1)
    current_tri = int(np.argmax(tri_scores))

    while current_tri != -1:
        emitted_triangles[current_tri] = True
        tri = triangles[current_tri]
        reordered_triangles[output_idx] = tri
        output_idx += 1

        # Decrement valence for vertices of emitted triangle
        for v in tri:
            active_tri_count[v] -= 1
            if current_tri in vert_triangles[v]:
                vert_triangles[v].remove(current_tri)

            # Update cache: move to front
            if v in cache:
                cache.remove(v)
            cache.insert(0, v)
            if len(cache) > cache_size:
                cache.pop()

        # Update scores of all vertices in cache and neighbors
        candidates: Set[int] = set()
        for v in cache:
            vertex_scores[v] = score_vertex(v)
            for t in vert_triangles[v]:
                if not emitted_triangles[t]:
                    candidates.add(t)

        # Pick best next candidate triangle from cache neighbors
        best_cand = -1
        best_score = -1e9
        for cand in candidates:
            cand_score = float(np.sum(vertex_scores[triangles[cand]]))
            if cand_score > best_score:
                best_score = cand_score
                best_cand = cand

        if best_cand != -1:
            current_tri = best_cand
        else:
            # Cache depleted: search unvisited triangles
            unvisited = np.where(~emitted_triangles)[0]
            if len(unvisited) > 0:
                # Pick triangle with best score among unvisited
                scores = np.sum(vertex_scores[triangles[unvisited]], axis=1)
                current_tri = int(unvisited[np.argmax(scores)])
            else:
                current_tri = -1

    return reordered_triangles


def optimize_vertex_fetch(
    vertices: np.ndarray,
    triangles: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reorder vertex buffer according to triangle index order for cache prefetch efficiency.
    
    Returns (optimized_vertices, remapped_triangles).
    """
    if len(triangles) == 0:
        return vertices.copy(), triangles.copy()

    new_vert_map: Dict[int, int] = {}
    new_vertices: List[np.ndarray] = []
    remapped_triangles = np.zeros_like(triangles)

    for i, tri in enumerate(triangles):
        for j, v in enumerate(tri):
            if v not in new_vert_map:
                new_idx = len(new_vertices)
                new_vert_map[v] = new_idx
                new_vertices.append(vertices[v])
            remapped_triangles[i, j] = new_vert_map[v]

    return np.array(new_vertices, dtype=vertices.dtype), remapped_triangles


class QuadricMeshSimplifier:
    """Garland-Heckbert Quadric Error Metric (QEM) edge collapse simplifier.
    
    Computes 4x4 quadric fundamental error matrices for each vertex:
    Q = sum(p * p^T) where p = [a, b, c, d]^T is the normalized plane equation ax + by + cz + d = 0.
    Contracts edges (v_i, v_j) that minimize v^T (Q_i + Q_j) v.
    """

    def __init__(self, vertices: np.ndarray, triangles: np.ndarray):
        self.vertices = np.array(vertices, dtype=np.float64)
        self.triangles = np.array(triangles, dtype=np.int64)

    def simplify(
        self,
        target_triangles: int,
        max_error: float = 1e5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simplify mesh to target triangle count using quadric edge collapses."""
        verts = self.vertices.copy()
        tris = self.triangles.copy()
        num_triangles = len(tris)

        if num_triangles <= target_triangles or len(verts) <= 4:
            return verts, tris

        # Compute initial vertex quadrics
        num_verts = len(verts)
        quadrics = [np.zeros((4, 4), dtype=np.float64) for _ in range(num_verts)]

        for tri in tris:
            p0, p1, p2 = verts[tri[0]], verts[tri[1]], verts[tri[2]]
            normal = np.cross(p1 - p0, p2 - p0)
            norm = float(np.linalg.norm(normal))
            if norm < 1e-12:
                continue
            normal = normal / norm
            d = -float(np.dot(normal, p0))
            plane = np.array([normal[0], normal[1], normal[2], d], dtype=np.float64)
            q = np.outer(plane, plane)
            for v_idx in tri:
                quadrics[v_idx] += q

        # Build initial edges
        edges_set: Set[Tuple[int, int]] = set()
        for tri in tris:
            for i in range(3):
                u, v = int(tri[i]), int(tri[(i + 1) % 3])
                if u > v:
                    u, v = v, u
                edges_set.add((u, v))

        # Vertex union-find mapping for collapsed vertices
        parent = list(range(num_verts))

        def find(i: int) -> int:
            path = []
            while parent[i] != i:
                path.append(i)
                i = parent[i]
            for node in path:
                parent[node] = i
            return i

        # Calculate edge costs
        def eval_edge(u: int, v: int) -> Tuple[float, np.ndarray]:
            q_sum = quadrics[u] + quadrics[v]
            # Candidate optimal vertex: midpoint
            v_mid = 0.5 * (verts[u] + verts[v])
            v_homo = np.array([v_mid[0], v_mid[1], v_mid[2], 1.0], dtype=np.float64)
            cost = float(v_homo.T @ q_sum @ v_homo)
            return cost, v_mid

        # Edge list with costs
        edge_list: List[Tuple[float, int, int, np.ndarray]] = []
        for u, v in edges_set:
            cost, v_mid = eval_edge(u, v)
            edge_list.append((cost, u, v, v_mid))

        edge_list.sort(key=lambda x: x[0])

        current_tri_count = num_triangles
        active_tris = [list(tri) for tri in tris]

        for cost, u, v, v_mid in edge_list:
            if current_tri_count <= target_triangles:
                break
            if cost > max_error:
                break

            ru, rv = find(u), find(v)
            if ru == rv:
                continue

            # Collapse rv into ru
            verts[ru] = v_mid
            quadrics[ru] += quadrics[rv]
            parent[rv] = ru

            # Update active triangles and remove degenerate ones
            surviving_tris: List[List[int]] = []
            for tri in active_tris:
                t0, t1, t2 = find(tri[0]), find(tri[1]), find(tri[2])
                if t0 != t1 and t1 != t2 and t2 != t0:
                    surviving_tris.append([t0, t1, t2])

            current_tri_count = len(surviving_tris)
            active_tris = surviving_tris

        # Remap and compact vertices
        unique_verts: Dict[int, int] = {}
        compact_verts: List[np.ndarray] = []
        final_tris: List[List[int]] = []

        for tri in active_tris:
            remapped_tri = []
            for v_idx in tri:
                root = find(v_idx)
                if root not in unique_verts:
                    unique_verts[root] = len(compact_verts)
                    compact_verts.append(verts[root])
                remapped_tri.append(unique_verts[root])
            final_tris.append(remapped_tri)

        res_verts = np.array(compact_verts, dtype=np.float64) if len(compact_verts) > 0 else np.zeros((0, 3))
        res_tris = np.array(final_tris, dtype=np.int64) if len(final_tris) > 0 else np.zeros((0, 3), dtype=np.int64)

        return res_verts, res_tris


def optimize_mesh(
    vertices: np.ndarray,
    triangles: np.ndarray,
    target_reduction_ratio: float = 1.0,
    optimize_cache: bool = True,
    optimize_fetch: bool = True,
) -> Tuple[np.ndarray, np.ndarray, MeshoptStats]:
    """Execute full meshoptimizer pipeline: optional QEM decimation, vertex cache, and fetch optimization."""
    orig_v_count = len(vertices)
    orig_t_count = len(triangles)
    acmr_before = compute_acmr(triangles)

    out_verts = vertices.copy()
    out_tris = triangles.copy()

    # 1. Simplification if reduction requested
    if target_reduction_ratio < 0.99 and orig_t_count > 4:
        target_triangles = max(4, int(orig_t_count * target_reduction_ratio))
        simplifier = QuadricMeshSimplifier(out_verts, out_tris)
        out_verts, out_tris = simplifier.simplify(target_triangles)

    # 2. Vertex Cache Optimization (reorder triangles)
    if optimize_cache and len(out_tris) > 0:
        out_tris = optimize_vertex_cache(out_tris)

    # 3. Vertex Fetch Optimization (reorder vertices)
    if optimize_fetch and len(out_tris) > 0:
        out_verts, out_tris = optimize_vertex_fetch(out_verts, out_tris)

    acmr_after = compute_acmr(out_tris)
    stats = MeshoptStats(
        original_vertices=orig_v_count,
        original_triangles=orig_t_count,
        optimized_vertices=len(out_verts),
        optimized_triangles=len(out_tris),
        acmr_before=acmr_before,
        acmr_after=acmr_after,
        reduction_ratio=float(len(out_tris) / max(1, orig_t_count)),
    )

    return out_verts, out_tris, stats
