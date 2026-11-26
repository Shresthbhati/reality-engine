"""Pairwise-rotation comparison against a reference model, and pose-graph optimisation.

Two things live here, and they are deliberately separate:

``compare_pair_rotations``
    Scores every measured two-view relative rotation against the SAME relative
    rotation computed from a reference reconstruction (here: the COLMAP model
    shipped with the dataset). That is the only way to tell a bad pairwise
    estimate from a correct one without a survey -- it needs no gauge choice,
    because a relative rotation between two cameras is invariant to the world
    frame.

``solve_pose_graph``
    Robust rotation averaging over the verified pairs, then translation
    averaging from the unit translation directions, then an optional similarity
    alignment to a reference for per-camera error measurement. Nothing is
    averaged blindly: an edge that disagrees with the consensus is kept in the
    output with its residual and a rejection reason.

Conventions (fixed, and asserted by the tests):

* A camera pose is world->camera: ``x_cam = R @ X_world + t``.
* An edge ``a -> b`` carries ``R_ab`` with ``x_b ~ R_ab @ x_a + t`` (OpenCV
  ``recoverPose``), so ``R_ab = R_b @ R_a.T`` in world->camera terms.
* Rotation closure around a triangle ``a -> b -> c`` is therefore
  ``R_bc @ R_ab @ R_ac.T == I``; the product ORDER matters and the tests pin it.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# --------------------------------------------------------------------------- rotations


def rotation_angle_deg(R: Any) -> float:
    """Geodesic angle of a rotation matrix, in degrees."""
    R = np.asarray(R, dtype=np.float64)
    cos = max(-1.0, min(1.0, (float(np.trace(R)) - 1.0) / 2.0))
    return math.degrees(math.acos(cos))


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]], dtype=np.float64)


def _rotation_defect(R: np.ndarray) -> str | None:
    """Why a 3x3 matrix is not a proper rotation, or None if it is one.

    A reflection (det = -1) and a non-orthogonal matrix both satisfy nothing that the
    averaging assumes; averaged silently they produce a "solution" that no camera can
    have. Input edges are screened with this instead of being orthonormalised.
    """
    if R.shape != (3, 3) or not np.all(np.isfinite(R)):
        return f"not a finite 3x3 matrix ({R.shape})"
    det = float(np.linalg.det(R))
    # A reflection is det = -1. The tolerance is 1e-3, not 1e-6: measured rotations arrive
    # rounded to 6 decimals, which alone puts det at 0.999999 (five real edges tripped a
    # 1e-6 test and were wrongly discarded).
    if det < 0.0:
        return f"determinant {det:.6f} < 0, so it is a reflection, not a rotation"
    if abs(det - 1.0) > 1e-3:
        return f"determinant {det:.6f} != 1, so it is not a rotation"
    if float(np.max(np.abs(R @ R.T - np.eye(3)))) > 1e-4:
        return "not orthonormal (R R^T != I)"
    return None


def rotation_log(R: Any) -> np.ndarray:
    """Axis-angle vector of a rotation matrix (angle in [0, pi])."""
    R = np.asarray(R, dtype=np.float64)
    cos = max(-1.0, min(1.0, (float(np.trace(R)) - 1.0) / 2.0))
    angle = math.acos(cos)
    if angle < 1e-9:
        return np.zeros(3)
    if abs(math.pi - angle) < 1e-5:
        # Near pi the skew part vanishes; recover the axis from R + I.
        A = (R + np.eye(3)) / 2.0
        axis = np.sqrt(np.clip(np.diag(A), 0.0, None))
        i = int(np.argmax(axis))
        if axis[i] < 1e-12:
            return np.zeros(3)
        axis = A[:, i] / axis[i]
        return axis / max(float(np.linalg.norm(axis)), 1e-12) * angle
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2.0 * math.sin(angle))
    return w * angle


def rotation_exp(w: Any) -> np.ndarray:
    """Rotation matrix of an axis-angle vector."""
    w = np.asarray(w, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(w))
    if theta < 1e-12:
        return np.eye(3) + _skew(w)
    k = w / theta
    K = _skew(k)
    return np.eye(3) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)


def relative_rotation(R_a: Any, R_b: Any) -> np.ndarray:
    """``R_ab`` with ``x_b ~ R_ab @ x_a`` from two world->camera rotations."""
    return np.asarray(R_b, dtype=np.float64) @ np.asarray(R_a, dtype=np.float64).T


def cycle_residuals(nodes: Sequence[str],
                    edges: Iterable[tuple[str, str, Any]]) -> list[dict[str, Any]]:
    """Rotation closure residual for every triangle with three measured edges.

    ``edges`` yields ``(a, b, R_ab)`` with ``x_b ~ R_ab @ x_a``. The residual is
    ``angle(R_bc @ R_ab @ R_ac.T)``: composing a->b then b->c must equal a->c.
    Returns one row per triangle, sorted by residual, each with the three names
    so a caller can inspect which triangle disagreed.
    """
    R: dict[tuple[str, str], np.ndarray] = {}
    for a, b, Rab in edges:
        if a == b or Rab is None:
            continue
        R[(a, b)] = np.asarray(Rab, dtype=np.float64)

    def _get(a: str, b: str) -> np.ndarray | None:
        if (a, b) in R:
            return R[(a, b)]
        if (b, a) in R:
            return R[(b, a)].T
        return None

    rows: list[dict[str, Any]] = []
    order = list(dict.fromkeys(nodes))
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            for k in range(j + 1, len(order)):
                a, b, c = order[i], order[j], order[k]
                Rab, Rbc, Rac = _get(a, b), _get(b, c), _get(a, c)
                if Rab is None or Rbc is None or Rac is None:
                    continue
                rows.append({"a": a, "b": b, "c": c,
                             "residual_deg": round(rotation_angle_deg(Rbc @ Rab @ Rac.T), 4)})
    rows.sort(key=lambda r: r["residual_deg"], reverse=True)
    return rows


def _percentile(values: Sequence[float], q: float) -> float | None:
    return round(float(np.percentile(values, q)), 4) if len(values) else None


# --------------------------------------------------------------------------- reference


@dataclass(frozen=True)
class ReferencePose:
    """A camera pose from a reference reconstruction, world->camera."""

    image: str
    R: np.ndarray
    t: np.ndarray
    center: np.ndarray


def _quat_to_rot(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """COLMAP/Hamilton quaternion (w first) to rotation matrix."""
    n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if n < 1e-12:
        return np.eye(3)
    w, x, y, z = qw / n, qx / n, qy / n, qz / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def read_colmap_text_model(sparse_dir: Path | str) -> dict[str, ReferencePose]:
    """Per-image reference poses from a COLMAP TEXT model (``sparse/images.txt``).

    Only the text format is parsed. A binary model is reported by raising
    ``FileNotFoundError`` naming what is missing: silently returning no
    reference would turn "not compared" into "compared and fine".
    """
    sparse_dir = Path(sparse_dir)
    images_txt = sparse_dir / "images.txt"
    if not images_txt.is_file():
        raise FileNotFoundError(f"no text model at {images_txt} (binary models are not parsed)")
    poses: dict[str, ReferencePose] = {}
    lines = images_txt.read_text(encoding="utf-8").splitlines()
    # Every image takes two lines (pose row, then its 2D observations); blank and
    # '#' lines are skipped, so advance by two per pose row.
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 10 or not parts[0].isdigit():
            continue
        name = parts[9]
        R = _quat_to_rot(*(float(v) for v in parts[1:5]))
        t = np.array([float(v) for v in parts[5:8]], dtype=np.float64)
        poses[Path(name).name] = ReferencePose(image=name, R=R, t=t, center=-R.T @ t)
        i += 1  # the 2D-observation row of the same image
    return poses


def _pair_key(a: str, b: str) -> tuple[str, str]:
    na, nb = Path(str(a)).name, Path(str(b)).name
    return (na, nb) if na <= nb else (nb, na)


def compare_pair_rotations(pairs: Iterable[Mapping[str, Any]],
                           reference: Mapping[str, ReferencePose],
                           *, near_deg: float = 2.0, far_deg: float = 30.0) -> dict[str, Any]:
    """Score measured pairwise relative rotations against the reference model.

    Each pair row must carry ``a``, ``b`` and ``relative_pose`` (with ``R`` and
    optionally ``t_unit``); extra keys (``engine``, ``inliers``) are passed
    through. A pair whose images are not both in the reference model is counted
    as skipped with that reason -- never as agreeing.
    """
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for pair in pairs:
        pose = pair.get("relative_pose") or {}
        if pose.get("R") is None:
            skipped.append({"a": pair.get("a"), "b": pair.get("b"), "reason": "no relative pose measured"})
            continue
        na, nb = Path(str(pair["a"])).name, Path(str(pair["b"])).name
        ra, rb = reference.get(na), reference.get(nb)
        if ra is None or rb is None:
            missing = [n for n, r in ((na, ra), (nb, rb)) if r is None]
            skipped.append({"a": na, "b": nb, "reason": f"not in reference model: {missing}"})
            continue
        R_est = np.asarray(pose["R"], dtype=np.float64)
        R_ref = relative_rotation(ra.R, rb.R)
        err = rotation_angle_deg(R_est @ R_ref.T)
        row: dict[str, Any] = {
            "engine": pair.get("engine"),
            "a": na,
            "b": nb,
            "inliers": pair.get("inliers"),
            "rotation_error_deg": round(err, 4),
        }
        t_unit = pose.get("t_unit")
        if t_unit is not None and len(np.asarray(t_unit).reshape(-1)) == 3:
            # Translation directions are compared in camera a's frame, which is
            # gauge-free: the direction of (c_b - c_a) expressed in a.
            t = np.asarray(t_unit, dtype=np.float64).reshape(3)
            d_est = -R_est.T @ t
            d_ref = ra.R @ (rb.center - ra.center)
            n_ref = float(np.linalg.norm(d_ref))
            if n_ref > 1e-12:
                d_ref = d_ref / n_ref
                cos = max(-1.0, min(1.0, float(np.dot(d_est / max(float(np.linalg.norm(d_est)), 1e-12), d_ref))))
                row["translation_direction_error_deg"] = round(math.degrees(math.acos(cos)), 4)
        rows.append(row)

    errs = [r["rotation_error_deg"] for r in rows]
    trans = [r["translation_direction_error_deg"] for r in rows if "translation_direction_error_deg" in r]
    by_engine: dict[str, dict[str, Any]] = {}
    for engine in sorted({str(r["engine"]) for r in rows}):
        sub = [r for r in rows if str(r["engine"]) == engine]
        e = [r["rotation_error_deg"] for r in sub]
        by_engine[engine] = {
            "pairs": len(sub),
            "median_rotation_error_deg": _percentile(e, 50),
            "p90_rotation_error_deg": _percentile(e, 90),
            "max_rotation_error_deg": round(max(e), 4) if e else None,
            "pairs_within_near_deg": int(sum(1 for v in e if v <= near_deg)),
            "pairs_beyond_far_deg": int(sum(1 for v in e if v > far_deg)),
        }
    return {
        "status": "measured" if rows else "unavailable",
        "reference": "relative rotation from the reference model (gauge-free: a pair rotation does not depend on the world frame)",
        "near_deg": near_deg,
        "far_deg": far_deg,
        "pairs_compared": len(rows),
        "pairs_skipped": len(skipped),
        "skipped": skipped[:20],
        "median_rotation_error_deg": _percentile(errs, 50),
        "mean_rotation_error_deg": round(float(np.mean(errs)), 4) if errs else None,
        "p90_rotation_error_deg": _percentile(errs, 90),
        "p95_rotation_error_deg": _percentile(errs, 95),
        "max_rotation_error_deg": round(max(errs), 4) if errs else None,
        "pairs_within_near_deg": int(sum(1 for v in errs if v <= near_deg)),
        "pairs_beyond_far_deg": int(sum(1 for v in errs if v > far_deg)),
        "median_translation_direction_error_deg": _percentile(trans, 50),
        "mean_translation_direction_error_deg": round(float(np.mean(trans)), 4) if trans else None,
        "p95_translation_direction_error_deg": _percentile(trans, 95),
        "by_engine": by_engine,
        "worst_pairs": rows and sorted(rows, key=lambda r: r["rotation_error_deg"], reverse=True)[:10],
        "pairs": rows,
    }


# --------------------------------------------------------------------------- pose graph


def _bfs_init(nodes: Sequence[str], edges: Sequence[Mapping[str, Any]]
              ) -> tuple[dict[str, np.ndarray], list[str], int, list[list[str]]]:
    """Maximum-weight spanning forest: one root per connected group, R_root = I.

    Kruskal by edge weight (RANSAC inliers), NOT breadth-first discovery order. A
    breadth-first walk let a weak wrong edge win over a much stronger correct edge
    that would have reached the same camera a hop later: on the south_building
    capture a 52-inlier wrong pair re-oriented a camera before its 598-inlier correct
    pair was considered, and the wrong orientation then rejected the correct edge.

    A disconnected pair graph also has one gauge per component, so each component
    needs its own root; sharing one root would silently produce a wrong solution.
    """
    parent = {n: n for n in nodes}

    def find(a: str) -> str:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    tree: dict[str, list[tuple[str, np.ndarray, bool]]] = {n: [] for n in nodes}
    n_edges_used = 0
    for e in sorted(edges, key=lambda x: -float(x.get("weight") or 1.0)):
        ra, rb = find(e["a"]), find(e["b"])
        if ra == rb:
            continue
        parent[rb] = ra
        R_ab = np.asarray(e["R"], dtype=np.float64)
        tree[e["a"]].append((e["b"], R_ab, True))
        tree[e["b"]].append((e["a"], R_ab, False))
        n_edges_used += 1

    strength = {n: max((float(x.get("weight") or 1.0) for x in edges if n in (x["a"], x["b"])),
                       default=-1.0) for n in nodes}
    R: dict[str, np.ndarray] = {}
    roots: list[str] = []
    components: list[list[str]] = []
    for start in sorted(nodes, key=lambda n: -strength[n]):
        if start in R:
            continue
        R[start] = np.eye(3)
        roots.append(start)
        component = [start]
        queue = [start]
        while queue:
            u = queue.pop(0)
            for v, R_uv, forward in tree[u]:
                if v in R:
                    continue
                # R_uv maps u->v: R_v = R_uv R_u ; read backwards: R_u = R_uv R_v
                R[v] = R_uv @ R[u] if forward else R_uv.T @ R[u]
                queue.append(v)
                component.append(v)
        components.append(sorted(component))
    return R, roots, n_edges_used, components


def _huber_weight(residual_norm: float, delta: float) -> float:
    if residual_norm <= delta:
        return 1.0
    return delta / max(residual_norm, 1e-12)


def _bridges(nodes: Sequence[str], edges: Sequence[Mapping[str, Any]]) -> set:
    """Edges whose removal disconnects the graph (Tarjan), keyed by frozenset of endpoints.

    A bridge has NO cycle through it, so nothing can verify it and the sub-graph on one
    side can be rotated rigidly about it while every accepted edge stays satisfied. That
    is an unobservable degree of freedom, not an accuracy result -- measured here as a
    10-camera group sitting ~168 deg away after being oriented by one wrong pair.
    """
    adj: dict[str, list[tuple[str, int]]] = {n: [] for n in nodes}
    for i, e in enumerate(edges):
        adj[e["a"]].append((e["b"], i))
        adj[e["b"]].append((e["a"], i))
    disc: dict[str, int] = {}
    low: dict[str, int] = {}
    found: set = set()
    timer = [0]

    def dfs(u: str, parent_edge: int) -> None:
        disc[u] = low[u] = timer[0]
        timer[0] += 1
        for v, ei in adj[u]:
            if ei == parent_edge:
                continue
            if v in disc:
                low[u] = min(low[u], disc[v])
                continue
            dfs(v, ei)
            low[u] = min(low[u], low[v])
            if low[v] > disc[u]:
                found.add(frozenset((u, v)))

    for n in nodes:
        if n not in disc:
            dfs(n, -1)
    return found


def _fit_rotations(nodes: Sequence[str], edges: Sequence[Mapping[str, Any]],
                   R: dict[str, np.ndarray], roots: Sequence[str],
                   iterations: int, huber_deg: float) -> tuple[dict[str, np.ndarray], list[float]]:
    """IRLS on SO(3): right perturbations, Huber weights, one pinned root per component."""
    index = {n: i for i, n in enumerate(nodes)}
    delta_thresh = math.radians(huber_deg)
    history: list[float] = []
    n_pin = 3 * len(roots)
    for _ in range(iterations):
        n_var = 3 * len(nodes)
        A = np.zeros((3 * len(edges) + n_pin, n_var))
        b = np.zeros(3 * len(edges) + n_pin)
        for row, e in enumerate(edges):
            Ra, Rb = R[e["a"]], R[e["b"]]
            phi = rotation_log(e["R"] @ Ra @ Rb.T)
            w = _huber_weight(float(np.linalg.norm(phi)), delta_thresh) * math.sqrt(max(e["weight"], 1e-6))
            ia, ib = 3 * index[e["a"]], 3 * index[e["b"]]
            A[3 * row:3 * row + 3, ia:ia + 3] = w * Rb
            A[3 * row:3 * row + 3, ib:ib + 3] = -w * Rb
            b[3 * row:3 * row + 3] = -w * phi
        for k, root in enumerate(roots):
            ir = 3 * index[root]
            A[3 * len(edges) + 3 * k:3 * len(edges) + 3 * k + 3, ir:ir + 3] = 1e6 * np.eye(3)
        step, *_ = np.linalg.lstsq(A, b, rcond=None)
        for n in nodes:
            i = 3 * index[n]
            if np.any(step[i:i + 3]):
                R[n] = R[n] @ rotation_exp(step[i:i + 3])
        history.append(round(float(np.max(np.abs(step))), 8))
        if history[-1] < 1e-10:
            break
    return R, history


def solve_pose_graph(nodes: Sequence[str],
                     edges: Sequence[Mapping[str, Any]],
                     *, iterations: int = 40, huber_deg: float = 1.0,
                     reject_deg: float = 5.0,
                     reference: Mapping[str, ReferencePose] | None = None,
                     points3d: Mapping[str, Sequence[tuple[str, np.ndarray]]] | None = None) -> dict[str, Any]:
    """Robust rotation averaging, translation averaging, and error measurement.

    ``edges`` entries are ``{a, b, R, t_unit, inliers}`` with ``x_b ~ R @ x_a + t``.
    Rotation averaging is IRLS on SO(3) (right perturbations, Huber on the edge
    residual); translation averaging solves for camera centres and one scale per
    edge from the unit directions, with one edge's scale fixed and the root
    centre pinned, because two-view geometry carries no scale at all. The result
    is a reconstruction whose scale is that fixed baseline's -- it is reported as
    a gauge, never as metres.

    When ``points3d`` is provided (camera_id -> list of (point_id, world_position)),
    cheirality constraints are added to the translation solve: each observed
    3D point must lie in front of its observing camera
    (``(R_c @ P + t_c).z > 0``). This pins the layout's handedness and scale
    instead of leaving them ambiguous under direction-only constraints.
    """
    nodes = list(dict.fromkeys(nodes))
    usable: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    for e in edges:
        if e.get("a") is None or e.get("b") is None or e.get("R") is None:
            continue
        if e["a"] not in nodes or e["b"] not in nodes or e["a"] == e["b"]:
            continue
        defect = _rotation_defect(np.asarray(e["R"], dtype=np.float64))
        if defect is not None:
            invalid.append({"a": str(e["a"]), "b": str(e["b"]), "inliers": e.get("inliers"),
                            "reason": f"edge rotation rejected: {defect}"})
            continue
        usable.append({"a": str(e["a"]), "b": str(e["b"]),
                       "R": np.asarray(e["R"], dtype=np.float64),
                       "t_unit": (np.asarray(e["t_unit"], dtype=np.float64).reshape(3)
                                  if e.get("t_unit") is not None else None),
                       "inliers": e.get("inliers"),
                       "weight": float(e.get("inliers") or 1.0)})
    if not usable:
        return {"status": "unavailable", "reason": "no edges with a measured relative rotation"}

    R, roots, n_tree, _components_given = _bfs_init(nodes, usable)
    R, history = _fit_rotations(nodes, usable, R, roots, iterations, huber_deg)

    # per-edge residual at the solution; an edge that disagrees with the consensus is
    # kept and labelled, never dropped silently
    for e in usable:
        E = e["R"] @ R[e["a"]] @ R[e["b"]].T
        e["residual_deg"] = round(rotation_angle_deg(E), 4)
        e["status"] = "rejected" if e["residual_deg"] > reject_deg else "accepted"
        if e["status"] == "rejected":
            e["reason"] = f"disagrees with the averaged rotations by {e['residual_deg']} deg (> {reject_deg})"
    accepted = [e for e in usable if e["status"] == "accepted"]

    # Re-fit on the accepted edges only. Dropping edges can disconnect the graph, and a
    # camera in a second component has its own gauge: keeping the first fit would report
    # that gauge as an accuracy error (measured before this was fixed: a whole trailing
    # block of cameras sat at ~33 deg with no edge left to constrain it).
    passes = 1
    if accepted and len(accepted) < len(usable):
        passes = 2
        R, roots, n_tree, components = _bfs_init(nodes, accepted)
        R, history = _fit_rotations(nodes, accepted, R, roots, iterations, huber_deg)
        for e in accepted:
            e["residual_deg"] = round(rotation_angle_deg(e["R"] @ R[e["a"]] @ R[e["b"]].T), 4)
    else:
        components = _components_given
    unreached = [n for n in nodes if n not in R]
    for n in unreached:  # isolated cameras stay at identity and are reported as such
        R[n] = np.eye(3)
    main_component = sorted(max(components, key=len)) if components else []

    out: dict[str, Any] = {
        "status": "optimised" if accepted else "failed",
        "method": ("IRLS rotation averaging on SO(3) (right perturbations, Huber "
                   f"{huber_deg} deg, {len(history)} iterations, one root per component "
                   f"pinned to identity), {passes} fitting pass(es) over the accepted edges"),
        "passes": passes,
        "components_before_rejection": _components_given,
        "nodes": len(nodes),
        "edges_given": len(usable),
        "invalid_edges": invalid,
        "n_invalid_edges": len(invalid),
        "edges_accepted": len(accepted),
        "edges_rejected": len(usable) - len(accepted),
        "spanning_tree_edges": n_tree,
        "components": components,
        "n_components": len(components),
        "main_component_cameras": len(main_component),
        "unreached_cameras": sorted(unreached),
        "isolated_cameras": sorted(n for n in nodes
                                   if not any(e["a"] == n or e["b"] == n for e in accepted)),
        "reject_deg": reject_deg,
        "final_step_norm": history[-1] if history else None,
        "edge_residual_deg": {
            "median": _percentile([e["residual_deg"] for e in accepted], 50),
            "p90": _percentile([e["residual_deg"] for e in accepted], 90),
            "max": round(max((e["residual_deg"] for e in accepted), default=0.0), 4),
        },
        "rejected_edges": sorted([{"a": e["a"], "b": e["b"], "residual_deg": e["residual_deg"],
                                   "inliers": e["inliers"], "reason": e["reason"]}
                                  for e in usable if e["status"] == "rejected"],
                                 key=lambda r: r["residual_deg"], reverse=True),
        "rotations": {n: [[round(float(v), 6) for v in row] for row in R[n]] for n in nodes},
    }

    # Edges with no cycle through them cannot be verified by the graph itself.
    bridges = _bridges(nodes, accepted)
    out["bridges"] = sorted(
        [{"a": Path(e["a"]).name, "b": Path(e["b"]).name, "residual_deg": e["residual_deg"],
          "inliers": e["inliers"]}
         for e in accepted if frozenset((e["a"], e["b"])) in bridges],
        key=lambda r: r["a"])
    out["n_bridges"] = len(bridges)
    out["bridge_note"] = ("a bridge has no cycle through it: the part it connects can be rotated about it "
                          "without changing any accepted residual, so that placement is unverified")

    # ---------------- observability, and rotation uncertainty kept separate
    bridge_pairs = {frozenset((e["a"], e["b"])) for e in accepted
                    if frozenset((e["a"], e["b"])) in bridges}
    observable: list[str] = []
    unobservable: list[dict[str, str]] = []
    rot_uncertainty: dict[str, float] = {}
    for n in nodes:
        inc = [e for e in accepted if n in (e["a"], e["b"])]
        verified = [e for e in inc if frozenset((e["a"], e["b"])) not in bridge_pairs]
        if not inc:
            unobservable.append({"camera": n, "reason": "no accepted edge"})
        elif not verified:
            unobservable.append({"camera": n, "reason":
                                 "every incident edge is a bridge, so nothing constrains its rotation "
                                 "against the rest of the graph (a rigid rotation about the bridge is free)"})
        else:
            observable.append(n)
        if inc:
            res = np.array([e["residual_deg"] for e in inc], dtype=np.float64)
            rot_uncertainty[n] = round(float(np.sqrt(np.mean(res ** 2)) / math.sqrt(max(1, len(inc) - 1))), 4)
    out["observability"] = {
        "observable_cameras": len(observable),
        "unobservable_cameras": len(unobservable),
        "unobservable": unobservable,
        "definition": ("a camera is rotation-observable when it has at least one accepted edge that lies on a "
                       "cycle; an edge with no cycle (bridge) cannot verify anything"),
    }
    out["rotation_uncertainty_deg"] = {
        "per_camera": {Path(k).name: v for k, v in rot_uncertainty.items()},
        "note": ("RMS of a camera's incident accepted-edge residuals divided by sqrt(deg-1): the spread of the "
                 "fit around it, NOT a confidence interval, and not comparable to the per-camera reference error"),
    }

    # ---------------- translation averaging, once per connected component
    centres: dict[str, np.ndarray] = {}
    per_component: list[dict[str, Any]] = []
    for ci, comp in enumerate(components):
        comp_edges = [e for e in accepted if e["a"] in comp and e["b"] in comp]
        comp_points = {n: points3d.get(n, ()) for n in comp} if points3d else None
        c, info = _translation_average(comp, comp_edges, R, comp[0], points3d=comp_points)
        per_component.append({"component": ci, "cameras": len(comp), **(info or {})})
        if c:
            centres.update(c)
    if per_component:
        out["translation"] = (per_component[0] if len(per_component) == 1
                              else {"components": per_component})
        out["centres"] = {n: [round(float(v), 6) for v in centres[n]] for n in nodes if n in centres}

    # ---------------- per-camera error against the reference model
    centres_usable = bool(per_component) and all(c.get("status") == "estimated" for c in per_component)
    if reference:
        out["reference_comparison"] = _per_camera_reference_error(
            components, R, centres, accepted, reference, centres_usable=centres_usable,
            observable=set(observable), unobservable=unobservable)
        out["reference_comparison"]["cameras_outside_main_component"] = sorted(
            Path(str(n)).name for n in nodes if n not in main_component)
        out["reference_comparison"]["bridge_edges"] = [f"{b['a']}|{b['b']}" for b in out["bridges"]]
        out["reference_comparison"]["bridge_note"] = out["bridge_note"]
    return out


def _translation_average(nodes: Sequence[str], accepted: Sequence[Mapping[str, Any]],
                         R: Mapping[str, np.ndarray], root: str,
                         points3d: Mapping[str, Sequence[tuple[str, np.ndarray]]] | None = None):
    """Camera centres + one scale per edge from the unit direction constraints.

    When ``points3d`` is provided (camera_id -> list of (point_id, world_position)),
    cheirality residuals are added: each observed 3D point must lie in front of its
    observing camera (``(R_c @ P + t_c).z > 0``). This pins the layout's handedness
    and scale instead of leaving them ambiguous under direction-only constraints.
    """
    if not accepted:
        return None, {"status": "unavailable", "reason": "no accepted edges"}
    with_dir = [e for e in accepted if e.get("t_unit") is not None]
    if len(with_dir) < 3:
        return None, {"status": "unavailable",
                      "reason": f"only {len(with_dir)} accepted edges carry a translation direction"}

    free = [n for n in nodes if n != root]  # the root centre is pinned to the origin
    col = {n: 3 * i for i, n in enumerate(free)}
    n_e = len(with_dir)
    # c_b - c_a = s_e * d_e ; all edges share one global (arbitrary) scale, so the
    # scale vector is constrained to unit norm. Pinning one edge's scale instead
    # lets every other edge collapse towards zero (measured: 1e-4 baselines with a
    # near-zero residual), which is why this is solved as an eigenproblem.
    M = np.zeros((3 * n_e, 3 * len(free)))
    P = np.zeros((3 * n_e, n_e))
    for k, e in enumerate(with_dir):
        # direction of c_b - c_a in the world frame fixed by the averaging above
        d = -R[e["b"]].T @ e["t_unit"]
        for node, sign in ((e["a"], -1.0), (e["b"], 1.0)):
            j = col.get(node)
            if j is not None:
                M[3 * k:3 * k + 3, j:j + 3] += sign * np.eye(3)
        P[3 * k:3 * k + 3, k] = d

    # Cheirality observations: (camera free-col index, world point, optical axis row, weight)
    cheirality: list[tuple[int, np.ndarray, np.ndarray, float]] = []
    if points3d:
        for cam_id, obs_list in points3d.items():
            if cam_id == root or cam_id not in col:
                continue  # root centre is pinned; no cheirality variable for it
            R_c = np.asarray(R[cam_id], dtype=float)
            r_c = R_c[2, :]  # third row of R: optical axis expressed in world frame
            for _point_id, P in obs_list:
                P_w = np.asarray(P, dtype=float)
                if P_w.shape != (3,) or not np.all(np.isfinite(P_w)):
                    continue
                cheirality.append((col[cam_id], P_w, r_c, 1.0))
    n_cheirality = len(cheirality)

    from scipy.optimize import least_squares

    strongest = max(range(n_e), key=lambda i: float(with_dir[i].get("weight") or 1.0))
    keep = [i for i in range(n_e) if i != strongest]
    # Map edge index -> scale variable position (strongest edge is fixed to 1, no variable)
    scale_pos = {k: j for j, k in enumerate(keep)}
    L, *_ = np.linalg.lstsq(M, P, rcond=None)  # centres as a linear function of the scales
    Q = P - M @ L
    _w, vecs = np.linalg.eigh(Q.T @ Q)
    s0 = vecs[:, 0]
    s0 = s0 / max(float(np.linalg.norm(s0)), 1e-12)
    if s0[int(np.argmax(np.abs(s0)))] < 0:
        s0 = -s0
    s0 = np.abs(s0) / max(float(np.abs(s0[strongest])), 1e-12)
    x0 = np.concatenate([L @ s0, s0[keep]])

    n_cam_vars = M.shape[1]
    n_scale_vars = len(keep)

    def _residual(x: np.ndarray) -> np.ndarray:
        c = x[:n_cam_vars]
        s = np.ones(n_e)
        s[keep] = x[n_cam_vars:n_cam_vars + n_scale_vars]
        direction_res = (M @ c - P @ s).ravel()
        if n_cheirality == 0:
            return direction_res
        cheirality_res = np.zeros(n_cheirality)
        for i, (cam_col, P_w, r_c, weight) in enumerate(cheirality):
            cam_center = c[cam_col:cam_col + 3]
            z = float(r_c @ (P_w - cam_center))
            cheirality_res[i] = weight * min(0.0, z)
        return np.concatenate([direction_res, cheirality_res])

    def _jac(x: np.ndarray) -> np.ndarray:
        # Direction Jacobian: sparse pattern, dense values
        n_dir_rows = 3 * n_e
        n_vars = n_cam_vars + n_scale_vars
        J_dir = np.zeros((n_dir_rows, n_vars))
        for k, e in enumerate(with_dir):
            d = -R[e["b"]].T @ e["t_unit"]
            for node, sign in ((e["a"], -1.0), (e["b"], 1.0)):
                j = col.get(node)
                if j is not None:
                    J_dir[3 * k:3 * k + 3, j:j + 3] += sign * np.eye(3)
            sp = scale_pos.get(k)
            if sp is not None:
                J_dir[3 * k:3 * k + 3, n_cam_vars + sp] = -d  # d/ds_e of (M c - P s): -d
        # Cheirality Jacobian
        if n_cheirality == 0:
            return J_dir
        J_cheirality = np.zeros((n_cheirality, n_vars))
        # Need current x to evaluate z sign; recompute from x
        c = x[:n_cam_vars]
        for i, (cam_col, P_w, r_c, weight) in enumerate(cheirality):
            cam_center = c[cam_col:cam_col + 3]
            z = float(r_c @ (P_w - cam_center))
            if z < 0.0:
                J_cheirality[i, cam_col:cam_col + 3] = -weight * r_c
        return np.vstack([J_dir, J_cheirality])

    bounds = (np.concatenate([np.full(n_cam_vars, -np.inf), np.zeros(n_scale_vars)]),
              np.full(n_cam_vars + n_scale_vars, np.inf))
    sol = least_squares(_residual, x0, jac=_jac, bounds=bounds, method="trf", max_nfev=2000)
    c = sol.x[:n_cam_vars]
    s = np.ones(n_e)
    s[keep] = sol.x[n_cam_vars:n_cam_vars + n_scale_vars]
    centres = {root: np.zeros(3)}
    for n in free:
        centres[n] = c[col[n]:col[n] + 3]
    scales = {f"{e['a']}|{e['b']}": round(float(s[i]), 6) for i, e in enumerate(with_dir)}
    direction_residual = (M @ c - P @ s).reshape(-1, 3)
    direction_norms = np.linalg.norm(direction_residual, axis=1)

    # Cheirality residual at solution
    cheirality_norms: list[float] = []
    if n_cheirality > 0:
        for cam_col, P_w, r_c, weight in cheirality:
            cam_center = c[cam_col:cam_col + 3]
            z = float(r_c @ (P_w - cam_center))
            cheirality_norms.append(round(min(0.0, z), 6))
    cheirality_violations = sum(1 for z in cheirality_norms if z < -1e-6)

    # Self-check on the solved layout: two cameras cannot occupy one point.
    collapsed = 0
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if float(np.linalg.norm(centres[nodes[i]] - centres[nodes[j]])) < 1e-6:
                collapsed += 1

    # Degeneracy check: if centres span fewer than 3 dimensions, the translation
    # problem is underdetermined in the missing direction. Cheirality constraints
    # can resolve handedness but not rank deficiency -- if all cameras and points
    # are coplanar, the missing direction is still unconstrained.
    C = np.array([centres[n] for n in nodes], dtype=np.float64)
    sv = np.linalg.svd(C - C.mean(axis=0), compute_uv=False)
    layout_rank = int(np.sum(sv > max(float(sv[0]), 1e-12) * 1e-6)) if sv.size else 0
    zero_baseline = [f"{with_dir[i]['a']}|{with_dir[i]['b']}" for i in keep if s[i] <= 1e-9]

    # With cheirality, a layout that passes direction constraints but violates
    # cheirality (points behind cameras) is still ambiguous: the mirror layout
    # satisfies directions but puts points behind cameras. Cheirality rules it out.
    ambiguous = collapsed > 0 or layout_rank < 3 or bool(zero_baseline)
    if n_cheirality > 0 and cheirality_violations > 0:
        ambiguous = True
    reasons: list[str] = []
    if collapsed:
        reasons.append(f"{collapsed} camera pairs sit at the same point")
    if layout_rank < 3:
        reasons.append(f"the solved centres span rank {layout_rank} (a {len(nodes)}-camera layout must span 3)")
    if zero_baseline:
        reasons.append(f"{len(zero_baseline)} baseline(s) are pinned at zero, which folds two cameras onto one point")
    if n_cheirality > 0 and cheirality_violations > 0:
        reasons.append(f"{cheirality_violations} cheirality violation(s): 3D point behind its observing camera")
    extra_note = (
        ". Cheirality constraints from observed 3D points were applied but the layout still failed a physicality check. "
        if n_cheirality > 0 else
        " Unit two-view directions with one unknown scale per edge admit layouts that fit every edge and are not "
        "camera configurations (reproduced with EXACT reference directions), "
    )
    return centres, {
        "status": "ambiguous" if ambiguous else "estimated",
        **({} if not ambiguous else {
            "reason": ("the solved layout is not physical: " + "; ".join(reasons) + extra_note +
                       "so this result is not used for any centre accuracy claim"),
            "collapsed_camera_pairs": collapsed,
            "zero_baseline_edges": zero_baseline,
        }),
        "layout_rank": layout_rank,
        "camera_degree": {Path(n).name: sum(1 for e in with_dir if n in (e["a"], e["b"])) for n in nodes},
        "translation_observability": {
            "cameras_with_two_or_more_directions": sum(
                1 for n in nodes if sum(1 for e in with_dir if n in (e["a"], e["b"])) >= 2),
            "note": ("a camera with one incident direction has its position fixed only along a ray; its "
                     "distance is set by the edge's unknown baseline, so its placement is not observable"),
        },
        "scale": (f"gauge: baseline of edge {with_dir[strongest]['a']}|{with_dir[strongest]['b']} "
                  "fixed to 1, every other baseline bounded >= 0"
                  + (f"; cheirality from {n_cheirality} observed 3D points pins handedness"
                     if n_cheirality > 0 else "; no metric basis exists for this capture")),
        "edges_used": n_e,
        "baselines_non_negative": True,
        "baselines_at_zero": int(sum(1 for i in keep if s[i] <= 1e-9)),
        "solver": f"bounded least squares, {sol.nfev} evaluations, optimality {sol.optimality:.3g}",
        "direction_residual": {
            "median": round(float(np.median(direction_norms)), 6),
            "p90": round(float(np.percentile(direction_norms, 90)), 6),
            "max": round(float(np.max(direction_norms)), 6),
        },
        "cheirality": {
            "n_observations": n_cheirality,
            "n_violations": cheirality_violations,
            "median_residual": round(float(np.median(cheirality_norms)), 6) if cheirality_norms else None,
            "max_residual": round(float(max(cheirality_norms)), 6) if cheirality_norms else None,
            "note": ("residual = min(0, (R_c @ P + t_c).z); zero means the point is in front of the camera, "
                     "negative means it is behind (a violation)") if n_cheirality > 0 else None,
        } if n_cheirality > 0 else {"n_observations": 0, "note": "no 3D points supplied; direction-only solve"},
        "per_edge_scale": scales,
    }


def _umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Similarity (scale, rotation, translation) mapping src onto dst, least squares."""
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    Sc = src - mu_s
    Dc = dst - mu_d
    cov = Dc.T @ Sc / len(src)
    U, S, Vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1.0, 1.0, d])
    R = U @ D @ Vt
    var_s = float((Sc ** 2).sum() / len(src))
    scale = float(np.trace(np.diag(S) @ D) / var_s) if var_s > 0 else 0.0
    t = mu_d - scale * (R @ mu_s)
    return scale, R, t


def _per_camera_reference_error(components: Sequence[Sequence[str]], R: Mapping[str, np.ndarray],
                                centres: Mapping[str, np.ndarray] | None,
                                accepted: Sequence[Mapping[str, Any]],
                                reference: Mapping[str, ReferencePose],
                                *, centres_usable: bool = True,
                                observable: set | None = None,
                                unobservable: Sequence[Mapping[str, Any]] = ()) -> dict[str, Any]:
    """Rotation/centre error per camera, with ONE GAUGE PER CONNECTED COMPONENT.

    Cameras that are no longer connected after edge rejection have an independent
    gauge; scoring them in the main component's gauge reports a gauge difference as an
    accuracy error (measured: a disconnected trailing block showed a flat ~33 deg).
    A component with fewer than 3 reference cameras cannot be scored -- a rotation
    Procrustes on 1-2 cameras can absorb the error -- so it is listed, not scored.

    Rotation and centre are fitted with separate gauges: deriving the rotation gauge
    from the centres would charge a translation error as a rotation error (measured as
    a spurious ~80 deg median on this capture).
    """
    nodes = [n for comp in components for n in comp]
    degree = {n: 0 for n in nodes}
    for e in accepted:
        if e["a"] in degree and e["b"] in degree:
            degree[e["a"]] += 1
            degree[e["b"]] += 1

    rows: list[dict[str, Any]] = []
    gauges: list[dict[str, Any]] = []
    for ci, comp in enumerate(components):
        common = [n for n in comp if Path(str(n)).name in reference]
        entry: dict[str, Any] = {"component": ci, "cameras": len(comp), "reference_cameras": len(common)}
        if len(common) < 3:
            entry["status"] = "not_scored"
            entry["reason"] = f"{len(common)} reference cameras cannot determine a gauge"
            gauges.append(entry)
            continue

        # rotation-only Procrustes: for a consistent pair R_est = R_ref R_a^T, so R_a is
        # the right polar factor of sum(R_est^T R_ref)
        H = np.zeros((3, 3))
        for n in common:
            H += np.asarray(R[n]).T @ reference[Path(str(n)).name].R
        U, _S, Vt = np.linalg.svd(H)
        D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
        R_a = U @ D @ Vt

        scale = t_g = None
        to_est = None
        if centres is not None and all(n in centres for n in comp) and centres_usable:
            src = np.array([centres[n] for n in common])
            dst = np.array([reference[Path(str(n)).name].center for n in common])
            # est -> ref: c_ref = scale * R_g c_est + t_g, so ref -> est is R_g^T
            scale, R_g, t_g = _umeyama(src, dst)
            to_est = R_g.T
        entry.update({"status": "measured",
                      "scale_to_reference": None if scale is None else round(scale, 6),
                      "gauge": "rotation: rotation-only Procrustes; centre: similarity fitted on centres"})
        gauges.append(entry)

        for n in comp:
            ref = reference.get(Path(str(n)).name)
            if ref is None:
                continue
            err = rotation_angle_deg(np.asarray(R[n]) @ R_a @ ref.R.T)
            c_err = None
            if scale and to_est is not None:
                predicted = to_est @ (ref.center - t_g) / scale  # this camera's centre in the est frame
                c_err = float(np.linalg.norm(centres[n] - predicted) * scale)
            rows.append({"image": Path(str(n)).name, "component": ci,
                         "rotation_error_deg": round(err, 4),
                         "centre_error_model_units": None if c_err is None else round(c_err, 6),
                         "observable": True if observable is None else n in observable,
                         "graph_degree": degree[n]})

    scored = [r for r in rows if any(g.get("status") == "measured" and g["component"] == r["component"]
                                     for g in gauges)]
    if not scored:
        return {"status": "unavailable",
                "reason": "no connected component has the 3 reference cameras a gauge needs",
                "components": gauges}
    rot = [r["rotation_error_deg"] for r in scored]
    cen = [r["centre_error_model_units"] for r in scored if r["centre_error_model_units"] is not None]
    main_ci = max((g["component"] for g in gauges if g.get("status") == "measured"),
                  key=lambda ci: len(components[ci]))
    main_rot = [r["rotation_error_deg"] for r in scored if r["component"] == main_ci]
    obs_rot = [r["rotation_error_deg"] for r in scored if r["observable"]]
    return {
        "status": "measured",
        "cameras_compared": len(scored),
        "observable_cameras": (None if observable is None
                               else sum(1 for r in scored if r["observable"])),
        "unobservable_cameras": ([{"camera": Path(str(u["camera"])).name, "reason": u["reason"]}
                                  for u in unobservable] if unobservable else []),
        "cameras_not_in_reference": sorted(Path(str(n)).name for n in nodes
                                           if Path(str(n)).name not in reference),
        "components": gauges,
        "main_component": main_ci,
        "rotation_error_deg": {
            "cameras": len(rot),
            "median": _percentile(rot, 50),
            "mean": round(float(np.mean(rot)), 4) if rot else None,
            "p90": _percentile(rot, 90),
            "p95": _percentile(rot, 95),
            "max": round(max(rot), 4) if rot else None,
            "n_within_2_deg": int(sum(1 for v in rot if v <= 2.0)),
            "n_within_5_deg": int(sum(1 for v in rot if v <= 5.0)),
        },
        "rotation_error_deg_observable_only": {
            "cameras": len(obs_rot),
            "median": _percentile(obs_rot, 50),
            "mean": round(float(np.mean(obs_rot)), 4) if obs_rot else None,
            "p95": _percentile(obs_rot, 95),
            "max": round(max(obs_rot), 4) if obs_rot else None,
        },
        "rotation_error_deg_main_component": {
            "cameras": len(main_rot),
            "median": _percentile(main_rot, 50),
            "mean": round(float(np.mean(main_rot)), 4) if main_rot else None,
            "p90": _percentile(main_rot, 90),
            "p95": _percentile(main_rot, 95),
            "max": round(max(main_rot), 4) if main_rot else None,
        },
        "centre_error_model_units": {
            "median": _percentile(cen, 50), "p90": _percentile(cen, 90),
            "max": round(max(cen), 6) if cen else None,
        } if cen else {"status": "unavailable"},
        "per_camera": sorted(scored, key=lambda r: r["rotation_error_deg"], reverse=True),
        "note": ("error against a reconstruction shipped with the dataset, after a gauge fit per "
                 "connected component; it is a consistency measure against COLMAP's model, not "
                 "survey accuracy"),
    }
