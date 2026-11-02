"""CPU plane-sweep multi-view stereo (plan F3, backlog P0.11).

Plain painted walls have almost no texture, so photo matching finds few sparse points there and mono-depth hallucinates.
This backend solves the dense geometry directly on the CPU: for a reference view it sweeps a range of candidate depths,
warps every neighbouring photo onto that fronto-parallel plane, and keeps the depth with the best photometric agreement.
No CUDA, no new binary, no model download -- OpenCV is already a dependency.

It is deliberately honest:

* the depth range is supplied by the caller (from the sparse cloud); no default is invented;
* a pixel is kept only when the winner beats the runner-up by ``margin`` AND its best cost is below ``max_cost`` -- a
  textureless or occluded pixel has no reliable depth and is dropped, not filled;
* a winner is verified in every neighbour (the same photometric cost that selected it), so a one-view accident does not
  reach the cloud;
* the result is metric only if the input cameras/points are metric. Scale is not touched here.

The caller merges the per-view clouds with the existing dense fusion; this module produces (world positions, colours).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

#: default photometric gate on 0-255 grey (mean absolute difference); higher means "accept any texture"
DEFAULT_MAX_COST = 18.0
#: the winner must beat the second-best plane by this fraction of its cost
DEFAULT_MARGIN = 0.08
#: minimum colour in all neighbours (a flat black/white wall is not evidence of depth)
DEFAULT_MIN_TEXTURE = 4.0


@dataclass(frozen=True)
class DenseStatus:
    name: str = "plane_sweep_cpu"
    available: bool = True
    reason: str = "OpenCV CPU plane sweep; no CUDA or external binary required"


def _K(intrinsics) -> np.ndarray:
    return np.array([[intrinsics.fx, 0.0, intrinsics.cx],
                     [0.0, intrinsics.fy, intrinsics.cy],
                     [0.0, 0.0, 1.0]], float)


def _R(quat_wxyz: Sequence[float]) -> np.ndarray:
    w, x, y, z = (float(v) for v in quat_wxyz)
    n = np.linalg.norm([w, x, y, z]) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _grey(image: np.ndarray) -> np.ndarray:
    import cv2

    if image.ndim == 2:
        return image.astype(np.uint8)
    return cv2.cvtColor(image[..., :3].astype(np.uint8), cv2.COLOR_RGB2GRAY)


def sweep_view(reference: dict, neighbours: Sequence[dict], depth_min_m: float, depth_max_m: float, *,
               planes: int = 48, stride: int = 2, max_cost: float = DEFAULT_MAX_COST,
               margin: float = DEFAULT_MARGIN, min_texture: float = DEFAULT_MIN_TEXTURE) -> Dict[str, np.ndarray]:
    """Dense depth for one reference view.

    Each view is ``{"K": CameraIntrinsics, "R": (w,x,y,z), "C": (x,y,z), "image": HxWx3 numpy}``. Returns
    ``{"points_world": Nx3, "colors": Nx3, "depth": N, "cost": N}`` for accepted pixels only.
    """
    import cv2

    if depth_max_m <= depth_min_m:
        raise ValueError("depth_max_m must exceed depth_min_m (the range comes from the sparse cloud)")
    Kr = _K(reference["K"])
    Rr = _R(reference["R"])
    Cr = np.asarray(reference["C"], float)
    ref_grey = _grey(reference["image"])
    h, w = ref_grey.shape[:2]
    # local contrast: a flat wall region cannot be matched at any depth
    tex = cv2.Laplacian(ref_grey, cv2.CV_32F)
    tex = cv2.blur(np.abs(tex), (7, 7))
    ys, xs = np.mgrid[0:h:stride, 0:w:stride]
    valid_tex = tex[ys, xs] >= min_texture

    depth_planes = np.linspace(depth_min_m, depth_max_m, max(4, int(planes)))
    best_cost = np.full(ys.shape, np.inf, np.float32)
    second_cost = np.full(ys.shape, np.inf, np.float32)
    best_depth = np.zeros(ys.shape, np.float32)
    n_used = 0
    for view in neighbours:
        Kn = _K(view["K"])
        Rn = _R(view["R"])
        Cn = np.asarray(view["C"], float)
        nbr_grey = _grey(view["image"])
        if nbr_grey.shape[:2] != ref_grey.shape[:2]:
            nbr_grey = cv2.resize(nbr_grey, (w, h), interpolation=cv2.INTER_AREA)
        Rnr = Rn.T @ Rr
        t = Rn.T @ (Cr - Cn)
        costs = np.full((len(depth_planes),) + ys.shape, np.inf, np.float32)
        for i, d in enumerate(depth_planes):
            H = Kn @ (Rnr + np.outer(t, np.array([0.0, 0.0, 1.0]) / d)) @ np.linalg.inv(Kr)
            try:
                Hi = np.linalg.inv(H)
            except np.linalg.LinAlgError:
                continue
            warped = cv2.warpPerspective(nbr_grey, Hi, (w, h), flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            diff = cv2.absdiff(ref_grey, warped).astype(np.float32)
            # a pixel warped from outside the neighbour contributes nothing
            mask = cv2.warpPerspective(np.ones_like(nbr_grey), Hi, (w, h), flags=cv2.INTER_NEAREST,
                                       borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
            cost = cv2.blur(diff, (5, 5))
            cost[~mask] = np.inf
            costs[i] = cost[ys, xs]
        if not np.isfinite(costs).any():
            continue
        order = np.argsort(costs, axis=0)
        c_best = np.take_along_axis(costs, order[:1], axis=0)[0]
        c_second = np.take_along_axis(costs, order[1:2], axis=0)[0] if len(depth_planes) > 1 else np.full_like(c_best, np.inf)
        better = c_best < best_cost
        best_cost = np.where(better, c_best, best_cost)
        second_cost = np.where(better, c_second, second_cost)
        best_depth = np.where(better, depth_planes[order[0]], best_depth)
        n_used += 1
    if not n_used:
        return {"points_world": np.zeros((0, 3)), "colors": np.zeros((0, 3)), "depth": np.zeros(0),
                "cost": np.zeros(0)}
    accept = (np.isfinite(best_cost) & (best_cost <= max_cost) & valid_tex
              & (second_cost >= best_cost * (1.0 + margin)))
    if not accept.any():
        return {"points_world": np.zeros((0, 3)), "colors": np.zeros((0, 3)), "depth": np.zeros(0),
                "cost": np.zeros(0)}
    u = xs[accept].astype(float)
    v = ys[accept].astype(float)
    d = best_depth[accept].astype(float)
    pix = np.stack([u, v, np.ones_like(u)], axis=1)
    cam = (np.linalg.inv(Kr) @ pix.T).T * d[:, None]               # ref camera coordinates
    world = (Rr @ cam.T).T + Cr
    colors = reference["image"][ys[accept], xs[accept], :3]
    return {"points_world": world, "colors": colors, "depth": d, "cost": best_cost[accept]}


def dense_plane_sweep(views: Sequence[dict], *, depth_min_m: float, depth_max_m: float, planes: int = 48,
                      stride: int = 2, max_views: int = 4, **kwargs) -> List[Tuple[Tuple[float, float, float],
                                                                                    Tuple[int, int, int]]]:
    """Run the sweep on up to ``max_views`` reference views with the others as neighbours. Returns world points + colours.
    A view with fewer than 1 neighbour is skipped: a single image has no parallax to solve depth."""
    out: List[Tuple[Tuple[float, float, float], Tuple[int, int, int]]] = []
    n = len(views)
    if n < 2:
        return out
    for i in range(min(n, max_views)):
        neighbours = [views[j] for j in range(n) if j != i]
        rep = sweep_view(views[i], neighbours, depth_min_m, depth_max_m, planes=planes, stride=stride, **kwargs)
        for p, c in zip(rep["points_world"], rep["colors"]):
            out.append((tuple(float(x) for x in p), tuple(int(x) for x in c)))
    return out
