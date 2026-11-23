"""Depth-to-geometry fusion: aligned per-image depth + calibrated cameras -> one fused surface (Open3D uniform TSDF + marching cubes).

(``ScalableTSDFVolume`` was tried first and extracts zero triangles in the installed Open3D 0.20 Windows build for any parameters, even a
single flat wall; ``UniformTSDFVolume`` works, with bounds taken from the data.)

What makes this production-honest rather than a demo:

* Scale discipline. The input depth is already aligned to the SfM gauge, so every length here (voxel size, truncation, tolerances) is a
  FRACTION of the scene's own median depth: the surface is in SfM gauge units, never metres. ``scale_state`` is returned as
  ``SFM_GAUGE`` and nothing converts it.
* Camera conventions are the repository's: OpenCV camera frame (x right, y down, z forward, depth = z), ``rotation`` is camera-to-world.
  Open3D wants world-to-camera, so the extrinsic is the inverse. The reprojection test in ``tests/test_depth_fusion.py`` pins this.
* Inputs are CHECKED before anything is integrated, and a bad input is an explicit, typed outcome (``error_code``), never a quiet mesh:
  missing / corrupt depth or colour, non-finite or non-positive depth, an invalid camera, a camera whose transform is wrong (reversed
  view direction, or world-to-camera mixed up with camera-to-world: found by re-testing the view under the alternative conventions),
  a view whose depth scale disagrees with the others (found by searching for the scale that makes it agree), and views that share no
  verified overlap with the rest (a disconnected view graph). Offending views are EXCLUDED and listed; the result is then ``partial``.
* Occlusion boundaries: a depth discontinuity mixes foreground and background in the pixel footprint; those pixels are masked out
  before consistency and integration (counted per view).
* Inconsistent depth is REJECTED before integration. Every depth pixel is reprojected into every other view: agreement within a
  relative tolerance is *support*, a surface that lies behind the other view's observed surface is *occluded* (neutral), and a surface
  that lies in front of what the other view saw (a free-space violation) is a *conflict*. A pixel with more conflicts than support is not
  integrated. Pixels seen by one view only are kept and counted as ``unsupported`` -- they are not evidence of consistency.
* Confidence: an optional per-pixel confidence map gates integration (``min_confidence``); without one it is DERIVED from the
  consistency statistics. Open3D's ``integrate`` has no per-pixel weight, so confidence gates and is recorded; it does not weight the TSDF.
* Contribution is MEASURED, not assumed, and TRACEABLE: per view, and per mesh vertex (which views' integrated depth the vertex lies
  on, with the confidence and the signed depth residual), aggregated per connected component. ``provenance_json`` / ``provenance_npz``
  are the sidecar; GRANULARITY IS PER VERTEX (and per component), not per voxel and not per triangle patch.
* The mesh is validated and diagnosed (degenerate / duplicate / coincident triangles, non-finite vertices, manifoldness, boundary loops
  and approximate hole area, winding consistency, orientation against the viewing rays, components with their supporting views) and
  persisted by the caller (``persist_fusion`` writes temp + atomic rename; a partial result is flagged ``partial`` and never complete).
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_VOXEL_FRACTION = 0.01     # voxel edge = 1% of the median scene depth (gauge units)
DEFAULT_TRUNC_VOXELS = 4.0        # SDF truncation = 4 voxels
DEFAULT_DEPTH_TRUNC_FACTOR = 4.0  # ignore depth beyond 4x the median depth
DEFAULT_CONSISTENCY_REL_TOL = 0.03  # two views agree on a surface when their depths differ by < 3% of that depth
DEFAULT_EDGE_REL_TOL = 0.10       # adjacent pixels whose depths differ by > 10% of the nearer one straddle an occlusion boundary
DEFAULT_MIN_CONFIDENCE = 0.2      # pixels below this confidence are not integrated (derived confidence of a conflict-free pixel is >= 0.25)
MIN_CLUSTER_TRIANGLES = 50        # connected pieces smaller than this are reported and removed (floating specks)
MIN_RESOLUTION, MAX_RESOLUTION = 64, 256  # voxels per axis of the (cubic) volume: 256^3 keeps memory ~0.5 GB; a cap makes the voxel coarser, and is reported
MAX_INVALID_FRACTION = 0.6        # a depth map with more than this share of non-finite / non-positive pixels is corrupt, not "sparse"
PROBE_POINTS = 3000               # pixels sampled per view by the input probes
PROBE_LOW_SUPPORT = 0.35          # a view supported by less than this share is probed for a wrong transform / scale
PROBE_ALT_MIN, PROBE_ALT_GAIN = 0.40, 0.30   # an alternative explanation must reach this support AND beat the as-given support by this much
PROBE_GLOBAL_ALT_MIN = 0.60       # a convention applied to ALL views must make them agree about as well as a clean set does (0.40 is reachable by chance with 2 views)
SCALE_GRID = np.exp(np.linspace(np.log(0.25), np.log(4.0), 49))  # candidate depth-scale factors (ratio 1.059 between neighbours; contains 1.0)
MIN_SCALE_DEVIATION = 1.12        # a best-fitting scale within 12% of 1 is not called a mismatch
MIN_PAIR_PIXELS = 30              # two views are connected when this many pixels of one are supported by the other
PROVENANCE_SCHEMA = "fusion_provenance/v1"

# typed outcomes (``error_code`` / per-view ``code``)
MISSING_ARTIFACT, CORRUPT_DEPTH, CORRUPT_COLOR, SHAPE_MISMATCH = "MISSING_ARTIFACT", "CORRUPT_DEPTH", "CORRUPT_COLOR", "SHAPE_MISMATCH"
INVALID_CAMERA, WRONG_CAMERA_TRANSFORM, DEPTH_SCALE_MISMATCH = "INVALID_CAMERA", "WRONG_CAMERA_TRANSFORM", "DEPTH_SCALE_MISMATCH"
SINGLE_VIEW, DISCONNECTED_VIEW_GRAPH, INTERRUPTED = "SINGLE_VIEW", "DISCONNECTED_VIEW_GRAPH", "INTERRUPTED"


class FusionInputError(ValueError):
    """A fusion input that cannot be used. ``code`` is one of the module constants; ``source`` names the offending view (or None for a global fault)."""

    def __init__(self, code: str, detail: str, source: str | None = None):
        super().__init__(f"{code}: {detail}" + (f" [{source}]" if source else ""))
        self.code, self.detail, self.source = code, detail, source


@dataclass
class FusionView:
    source_id: str
    color: np.ndarray | None   # H x W x 3 uint8 at the DEPTH resolution
    depth: np.ndarray | None   # H x W float, SfM-gauge depth along the camera z axis (<=0 or non-finite = no measurement)
    camera: Any                # world_ir.camera.PinholeCamera (camera-to-world extrinsics), intrinsics sized to the ORIGINAL photograph
    confidence: np.ndarray | None = None   # optional H x W in [0, 1]; default is derived from the cross-view consistency statistics
    image_id: str | None = None            # id of the source photograph (provenance); defaults to ``source_id``
    depth_record_id: str | None = None     # id of the depth record / artifact this map came from (provenance)
    depth_sha256: str | None = None        # hash of the stored depth artifact when known; otherwise hashed from the array


@dataclass
class FusionResult:
    status: str                       # "ran" (complete) | "partial" (a surface exists but views are missing or the job stopped) | "skipped"
    reason: str | None = None
    report: dict[str, Any] = field(default_factory=dict)
    mesh_ply: bytes | None = None     # binary PLY, vertices in the SAME frame as the input cameras
    vertices: np.ndarray | None = None
    faces: np.ndarray | None = None
    error_code: str | None = None     # typed reason when status is "skipped" (or why a partial result is partial)
    complete: bool = False            # True only when every offered view was integrated and nothing interrupted the job
    provenance_json: bytes | None = None   # sidecar: views / components / hashes (canonical JSON)
    provenance_npz: bytes | None = None    # sidecar: per-vertex contributing-view table, weights, residuals, confidence


def _intrinsics_at(cam: Any, shape: tuple[int, int]) -> tuple[float, float, float, float]:
    """fx, fy, cx, cy of ``cam`` rescaled to a depth map of ``shape`` (h, w)."""
    K = cam.intrinsics
    sx, sy = shape[1] / K.width, shape[0] / K.height
    return K.fx * sx, K.fy * sy, K.cx * sx, K.cy * sy


def _quat_to_R(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    n = np.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _pose(cam: Any) -> tuple[np.ndarray, np.ndarray]:
    """(R camera-to-world 3x3, camera centre in world)."""
    e = cam.extrinsics
    return _quat_to_R((e.rotation.w, e.rotation.x, e.rotation.y, e.rotation.z)), np.array([e.position.x, e.position.y, e.position.z], float)


def _backproject(depth: np.ndarray, fx: float, fy: float, cx: float, cy: float, R: np.ndarray, t: np.ndarray, max_points: int | None = None):
    """All valid pixels (or an evenly spaced ``max_points`` of them) -> world points. Returns (points N x 3, flat pixel indices N)."""
    h, w = depth.shape
    valid = np.isfinite(depth) & (depth > 0)
    idx = np.flatnonzero(valid.ravel())
    if max_points is not None and len(idx) > max_points:
        idx = idx[np.linspace(0, len(idx) - 1, max_points).astype(np.int64)]
    v, u = np.divmod(idx, w)
    z = depth.ravel()[idx].astype(float)
    pc = np.stack([(u - cx) / fx * z, (v - cy) / fy * z, z], axis=1)
    return pc @ R.T + t, idx


def _project(points: np.ndarray, fx: float, fy: float, cx: float, cy: float, R: np.ndarray, t: np.ndarray, shape: tuple[int, int]):
    """World points -> (u, v ints, z, in_image mask) for a camera with camera-to-world (R, t)."""
    pc = (points - t) @ R           # R^T (X - t) in row-vector form
    z = pc[:, 2]
    front = z > 1e-9
    zs = np.where(front, z, 1.0)
    u = np.round(pc[:, 0] / zs * fx + cx).astype(np.int64)
    v = np.round(pc[:, 1] / zs * fy + cy).astype(np.int64)
    h, w = shape
    inside = front & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    return u, v, z, inside


def cross_view_consistency(views: Sequence[FusionView], tol: float) -> list[dict[str, Any]]:
    """Per view: which valid depth pixels are supported / occluded / in conflict with the OTHER views. ``keep`` is the integration mask.
    ``pair_support[j]`` = how many of this view's pixels view ``j`` supports (the edges of the view graph)."""
    cams = []
    for v in views:
        fx, fy, cx, cy = _intrinsics_at(v.camera, v.depth.shape)
        R, t = _pose(v.camera)
        cams.append((fx, fy, cx, cy, R, t))
    out = []
    for i, vi in enumerate(views):
        fx, fy, cx, cy, R, t = cams[i]
        pts, idx = _backproject(vi.depth, fx, fy, cx, cy, R, t)
        support = np.zeros(len(idx), np.int32)
        conflict = np.zeros(len(idx), np.int32)
        pair_support: dict[int, int] = {}
        for j, vj in enumerate(views):
            if i == j:
                continue
            fxj, fyj, cxj, cyj, Rj, tj = cams[j]
            u, v, z, inside = _project(pts, fxj, fyj, cxj, cyj, Rj, tj, vj.depth.shape)
            dj = np.zeros(len(idx))
            dj[inside] = vj.depth[v[inside], u[inside]]
            seen = inside & np.isfinite(dj) & (dj > 0)
            rel = np.where(seen, (z - dj) / np.where(seen, dj, 1.0), 0.0)
            sup_j = seen & (np.abs(rel) <= tol)
            support += sup_j.astype(np.int32)
            pair_support[j] = int(sup_j.sum())
            conflict += (seen & (rel < -tol)).astype(np.int32)      # in front of what view j saw along that ray: free-space violation
            # rel > tol: behind j's observed surface = occluded in j = neutral
        keep = conflict <= support
        out.append({"idx": idx, "keep": keep, "support": support, "conflict": conflict, "pair_support": pair_support,
                    "valid": int(len(idx)), "supported": int((support > 0).sum()), "conflicting": int((conflict > support).sum()),
                    "unsupported": int(((support == 0) & (conflict == 0)).sum())})
    return out


# ---------------------------------------------------------------------------------------------------------------- input screening
def _array_sha256(a: np.ndarray) -> str:
    a = np.ascontiguousarray(a, dtype=np.float32)
    return hashlib.sha256(str(a.shape).encode() + a.tobytes()).hexdigest()


def load_view(source_id: str, color_path: str | Path, depth_path: str | Path, camera: Any, *, confidence_path: str | Path | None = None,
              image_id: str | None = None, depth_record_id: str | None = None) -> FusionView:
    """Load a view from artifact files (colour: any PIL format; depth / confidence: ``.npy``). A missing or unreadable file raises
    ``FusionInputError`` with a typed code; the view carries the sha256 of the depth FILE so the provenance names the exact artifact."""
    checks = [(color_path, "colour image"), (depth_path, "depth map")] + ([(confidence_path, "confidence map")] if confidence_path else [])
    for p, what in checks:
        if not Path(p).is_file():
            raise FusionInputError(MISSING_ARTIFACT, f"{what} file does not exist: {p}", source_id)
    try:
        depth = np.load(depth_path, allow_pickle=False)
    except Exception as exc:  # noqa: BLE001
        raise FusionInputError(CORRUPT_DEPTH, f"depth file unreadable ({type(exc).__name__}: {exc})", source_id) from exc
    try:
        from PIL import Image

        color = np.asarray(Image.open(color_path).convert("RGB"))
    except Exception as exc:  # noqa: BLE001
        raise FusionInputError(CORRUPT_COLOR, f"colour file unreadable ({type(exc).__name__}: {exc})", source_id) from exc
    conf = None
    if confidence_path:
        try:
            conf = np.load(confidence_path, allow_pickle=False)
        except Exception as exc:  # noqa: BLE001
            raise FusionInputError(CORRUPT_DEPTH, f"confidence file unreadable ({type(exc).__name__}: {exc})", source_id) from exc
    return FusionView(source_id, color, depth, camera, conf, image_id or source_id, depth_record_id,
                      hashlib.sha256(Path(depth_path).read_bytes()).hexdigest())


def _discontinuity_mask(depth: np.ndarray, rel_tol: float) -> np.ndarray:
    """Pixels with a 4-neighbour whose depth differs by more than ``rel_tol`` of the nearer of the two (an occlusion boundary)."""
    m = np.zeros(depth.shape, bool)
    for a, b, sa, sb in ((depth[:, 1:], depth[:, :-1], (slice(None), slice(1, None)), (slice(None), slice(None, -1))),
                         (depth[1:, :], depth[:-1, :], (slice(1, None), slice(None)), (slice(None, -1), slice(None)))):
        both = (a > 0) & (b > 0)
        jump = both & (np.abs(a - b) > rel_tol * np.minimum(a, b))
        m[sa] |= jump
        m[sb] |= jump
    return m


def _screen_view(v: FusionView, edge_rel_tol: float) -> tuple[FusionView | None, dict[str, Any]]:
    """Validate one view and return (cleaned view or None, record). The cleaned depth has non-finite / non-positive pixels and occlusion
    boundaries zeroed; the record says what was removed and, when the view is unusable, the typed code."""
    rec: dict[str, Any] = {"source": v.source_id}

    def bad(code: str, detail: str):
        rec.update(code=code, detail=detail)
        return None, rec

    if v.depth is None or v.color is None:
        return bad(MISSING_ARTIFACT, ("depth map" if v.depth is None else "colour image") + " is missing")
    try:
        depth = np.asarray(v.depth)
        color = np.asarray(v.color)
        if depth.ndim != 2 or not np.issubdtype(depth.dtype, np.number) or depth.size == 0:
            return bad(CORRUPT_DEPTH, f"depth must be a non-empty 2-D numeric array, got shape {depth.shape} dtype {depth.dtype}")
    except Exception as exc:  # noqa: BLE001
        return bad(CORRUPT_DEPTH, f"depth unreadable ({type(exc).__name__}: {exc})")
    if color.ndim != 3 or color.shape[:2] != depth.shape or color.shape[2] != 3:
        return bad(SHAPE_MISMATCH, f"colour {color.shape} does not match depth {depth.shape}")
    try:
        K = v.camera.intrinsics
        R, t = _pose(v.camera)
        if not (np.isfinite([K.fx, K.fy, K.cx, K.cy]).all() and K.fx > 0 and K.fy > 0 and K.width > 0 and K.height > 0
                and np.isfinite(R).all() and np.isfinite(t).all()):
            return bad(INVALID_CAMERA, "non-finite or non-positive intrinsics / non-finite pose")
    except Exception as exc:  # noqa: BLE001
        return bad(INVALID_CAMERA, f"camera unreadable ({type(exc).__name__}: {exc})")
    d = depth.astype(np.float64)
    nonfinite = ~np.isfinite(d)
    nonpos = ~nonfinite & (d <= 0)
    frac_invalid = float((nonfinite | nonpos).mean())
    rec.update(shape=list(d.shape), pixels=int(d.size), invalid_nonfinite=int(nonfinite.sum()), invalid_nonpositive=int(nonpos.sum()),
               invalid_fraction=round(frac_invalid, 4))
    if frac_invalid > MAX_INVALID_FRACTION:
        return bad(CORRUPT_DEPTH, f"{frac_invalid:.0%} of the depth pixels are non-finite or non-positive (limit {MAX_INVALID_FRACTION:.0%})")
    conf = None
    if v.confidence is not None:
        conf = np.asarray(v.confidence, float)
        if conf.shape != d.shape:
            return bad(SHAPE_MISMATCH, f"confidence {conf.shape} does not match depth {d.shape}")
        conf = np.clip(np.nan_to_num(conf, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)
    d = np.where(nonfinite | nonpos, 0.0, d)
    edge = _discontinuity_mask(d, edge_rel_tol) & (d > 0)
    rec["occlusion_boundary_pixels"] = int(edge.sum())
    d = np.where(edge, 0.0, d)
    if not (d > 0).any():
        return bad(CORRUPT_DEPTH, "no valid depth pixel remains")
    rec["depth_sha256"] = v.depth_sha256 or _array_sha256(depth)
    return replace(v, depth=d.astype(np.float32), color=color, confidence=conf), rec


# ---------------------------------------------------------------------------------------------------------------- input probes
def _cam_tuple(v: FusionView, R: np.ndarray | None = None, t: np.ndarray | None = None):
    fx, fy, cx, cy = _intrinsics_at(v.camera, v.depth.shape)
    R0, t0 = _pose(v.camera)
    return fx, fy, cx, cy, R0 if R is None else R, t0 if t is None else t


def _support_of_points(pts: np.ndarray, others: Sequence[tuple[Any, ...]], tol: float) -> tuple[float, float]:
    """(share of points supported by >=1 other view, share seen by >=1 other view). ``others``: (fx, fy, cx, cy, R, t, depth)."""
    n = len(pts)
    if n == 0:
        return 0.0, 0.0
    sup = np.zeros(n, bool)
    seen_any = np.zeros(n, bool)
    for fx, fy, cx, cy, R, t, depth in others:
        u, v, z, inside = _project(pts, fx, fy, cx, cy, R, t, depth.shape)
        dj = np.zeros(n)
        dj[inside] = depth[v[inside], u[inside]]
        seen = inside & (dj > 0)
        rel = np.where(seen, (z - dj) / np.where(seen, dj, 1.0), 0.0)
        sup |= seen & (np.abs(rel) <= tol)
        seen_any |= seen
    return float(sup.mean()), float(seen_any.mean())


def _alt_poses(R: np.ndarray, t: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {"inverse_world_to_camera_given_as_camera_to_world": (R.T, -R.T @ t),
            "reversed_view_direction": (R @ np.diag([-1.0, 1.0, -1.0]), t)}


def probe_inputs(views: Sequence[FusionView], tol: float) -> dict[str, Any]:
    """Consistency probe on screened views. Does each view agree with the others as given, and if not, does it agree under a different
    camera convention (world-to-camera mixed up with camera-to-world; the camera looking the opposite way) or a different depth scale?
    Returns ``{"global_wrong_transform": name | None, "views": [per-view probe]}``. This DETECTS a wrong input; it does not repair it."""
    n = len(views)
    cams = [_cam_tuple(v) for v in views]

    def others(i: int, use: Sequence[tuple[Any, ...]]):
        return [(*use[j], views[j].depth) for j in range(n) if j != i]

    def pts_of(i: int, R: np.ndarray, t: np.ndarray):
        fx, fy, cx, cy = cams[i][:4]
        return _backproject(views[i].depth, fx, fy, cx, cy, R, t, PROBE_POINTS)[0]

    given = [_support_of_points(pts_of(i, cams[i][4], cams[i][5]), others(i, cams), tol) for i in range(n)]
    out: dict[str, Any] = {"global_wrong_transform": None, "views": []}
    mean_given = float(np.mean([g[0] for g in given]))
    if n >= 2 and mean_given < PROBE_LOW_SUPPORT:  # EVERY view disagrees: the same wrong convention may be applied to all of them
        for name in _alt_poses(np.eye(3), np.zeros(3)):
            use = []
            for i in range(n):
                R, t = _alt_poses(cams[i][4], cams[i][5])[name]
                use.append((*cams[i][:4], R, t))
            mean_alt = float(np.mean([_support_of_points(pts_of(i, use[i][4], use[i][5]), others(i, use), tol)[0] for i in range(n)]))
            if mean_alt >= PROBE_GLOBAL_ALT_MIN and mean_alt >= mean_given + PROBE_ALT_GAIN:
                out["global_wrong_transform"] = name
                out["global_support"] = {"as_given": round(mean_given, 4), name: round(mean_alt, 4)}
                break
    for i, v in enumerate(views):
        sup, overlap = given[i]
        rec: dict[str, Any] = {"source": v.source_id, "support_as_given": round(sup, 4), "overlap_as_given": round(overlap, 4), "verdict": "ok"}
        if sup < PROBE_LOW_SUPPORT and n >= 2:
            best_pose = max(((name, _support_of_points(pts_of(i, *pose), others(i, cams), tol)[0])
                             for name, pose in _alt_poses(cams[i][4], cams[i][5]).items()), key=lambda kv: kv[1])
            base_pts = pts_of(i, cams[i][4], cams[i][5])
            c = cams[i][5]
            scale_sup = [_support_of_points(c + s * (base_pts - c), others(i, cams), tol)[0] for s in SCALE_GRID]
            k = int(np.argmax(scale_sup))
            best_scale, best_scale_sup = float(SCALE_GRID[k]), float(scale_sup[k])
            pose_ok = best_pose[1] >= PROBE_ALT_MIN and best_pose[1] >= sup + PROBE_ALT_GAIN
            scale_ok = (best_scale_sup >= PROBE_ALT_MIN and best_scale_sup >= sup + PROBE_ALT_GAIN
                        and max(best_scale, 1 / best_scale) >= MIN_SCALE_DEVIATION)
            rec.update(best_alternative_pose={"convention": best_pose[0], "support": round(best_pose[1], 4)},
                       best_depth_scale={"factor": round(best_scale, 4), "support": round(best_scale_sup, 4)})
            if pose_ok and (not scale_ok or best_pose[1] >= best_scale_sup):
                rec.update(verdict=WRONG_CAMERA_TRANSFORM, suggested_convention=best_pose[0])
            elif scale_ok:
                rec.update(verdict=DEPTH_SCALE_MISMATCH, estimated_depth_scale_factor=round(best_scale, 4))
        out["views"].append(rec)
    return out


def _view_graph(cons: Sequence[dict[str, Any]], names: Sequence[str]) -> dict[str, Any]:
    """Connected components of the 'verified overlap' graph: i ~ j when at least MIN_PAIR_PIXELS of one's pixels agree with the other."""
    n = len(names)
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    edges = []
    for i, c in enumerate(cons):
        for j, k in c["pair_support"].items():
            if k >= max(MIN_PAIR_PIXELS, 0.002 * c["valid"]):
                edges.append((i, j, k))
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    comps = sorted(groups.values(), key=lambda g: (-len(g), -sum(cons[i]["valid"] for i in g), g[0]))
    return {"components": [[names[i] for i in g] for g in comps], "component_indices": comps, "connected": len(comps) == 1,
            "edges": [{"a": names[i], "b": names[j], "pixels_of_a_supported_by_b": k} for i, j, k in edges]}


# ---------------------------------------------------------------------------------------------------------------- mesh diagnostics
def _boundary_loops(vertices: np.ndarray, uniq: np.ndarray, counts: np.ndarray) -> list[dict[str, Any]]:
    """Loops of boundary edges (edges used by exactly one triangle) with perimeter and approximate (Newell) enclosed area."""
    b = uniq[counts == 1]
    if len(b) == 0:
        return []
    adj: dict[int, list[int]] = {}
    for a, c in b.tolist():
        adj.setdefault(a, []).append(c)
        adj.setdefault(c, []).append(a)
    used: set[tuple[int, int]] = set()
    loops = []
    for a0, c0 in b.tolist():
        if (a0, c0) in used:
            continue
        path = [a0]
        prev, cur = a0, c0
        used.add((a0, c0))
        closed = False
        while True:
            path.append(cur)
            if cur == a0:
                closed = True
                path.pop()
                break
            nxt = [x for x in adj[cur] if x != prev and (min(cur, x), max(cur, x)) not in used]
            if not nxt:
                break
            x = nxt[0]
            used.add((min(cur, x), max(cur, x)))
            prev, cur = cur, x
        pts = vertices[np.asarray(path)]
        per = float(np.linalg.norm(np.diff(np.vstack([pts, pts[:1]]), axis=0), axis=1).sum())
        area = float(0.5 * np.linalg.norm(np.cross(pts, np.roll(pts, -1, axis=0)).sum(axis=0))) if len(pts) >= 3 else 0.0
        loops.append({"vertices": int(len(pts)), "perimeter_gauge": round(per, 6), "approx_area_gauge2": round(area, 6), "closed": bool(closed)})
    return loops


def _coincident_triangles(vertices: np.ndarray, faces: np.ndarray, radius: float) -> dict[str, Any]:
    """Triangles whose centroids lie within ``radius`` of a NON-adjacent triangle (no shared vertex) and whose normals are parallel (same
    direction = a duplicated sheet; opposite = a back-to-back thin shell). ``remove`` marks the later triangle of each same-direction pair."""
    out: dict[str, Any] = {"same_direction_pairs": 0, "back_to_back_pairs": 0, "suppressed_triangles": 0, "remove": np.zeros(len(faces), bool)}
    if len(faces) < 2 or radius <= 0:
        return out
    from scipy.spatial import cKDTree

    tri = vertices[faces]
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ln = np.linalg.norm(nrm, axis=1)
    nrm = nrm / np.where(ln > 0, ln, 1.0)[:, None]
    pairs = cKDTree(tri.mean(axis=1)).query_pairs(radius, output_type="ndarray")
    if len(pairs) == 0:
        return out
    a, b = pairs[:, 0], pairs[:, 1]
    share = (faces[a][:, :, None] == faces[b][:, None, :]).any(axis=(1, 2))
    dot = np.einsum("ij,ij->i", nrm[a], nrm[b])
    same = ~share & (dot > 0.95)
    back = ~share & (dot < -0.95)
    out.update(same_direction_pairs=int(same.sum()), back_to_back_pairs=int(back.sum()))
    out["remove"][np.maximum(a[same], b[same])] = True
    out["suppressed_triangles"] = int(out["remove"].sum())
    return out


def orientation_vs_views(vertices: np.ndarray, faces: np.ndarray, contrib: np.ndarray, centres: np.ndarray) -> dict[str, Any]:
    """Does each triangle's normal (from its winding) face the cameras that supported it? A triangle is *flipped* when more of its supporting
    views see it from behind than from the front. Triangles seen at grazing angles (|cos| < 0.05) abstain. Expected: normals toward the cameras
    (outward from the solid)."""
    rep: dict[str, Any] = {"expected": "normal toward the cameras that saw the surface (outward)", "triangles_evaluated": 0, "flipped_triangles": 0,
                           "flipped_fraction": None, "triangles_without_supporting_view": 0, "globally_inverted": False}
    if len(faces) == 0 or contrib.shape[1] == 0:
        return rep
    tri = vertices[faces]
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nrm = nrm / np.maximum(np.linalg.norm(nrm, axis=1), 1e-30)[:, None]
    cen = tri.mean(axis=1)
    sup = contrib[faces[:, 0]].astype(bool) & contrib[faces[:, 1]].astype(bool) & contrib[faces[:, 2]].astype(bool)
    d = cen[:, None, :] - centres[None, :, :]
    d = d / np.maximum(np.linalg.norm(d, axis=2), 1e-30)[:, :, None]
    cos = np.einsum("fc,fvc->fv", nrm, d)            # < 0: the normal points back toward the camera
    toward = (sup & (cos < -0.05)).sum(axis=1)
    away = (sup & (cos > 0.05)).sum(axis=1)
    has = sup.any(axis=1) & ((toward + away) > 0)
    flipped = has & (away > toward)
    rep.update(triangles_evaluated=int(has.sum()), flipped_triangles=int(flipped.sum()), triangles_without_supporting_view=int((~sup.any(axis=1)).sum()),
               flipped_fraction=round(float(flipped.sum() / has.sum()), 4) if has.any() else None,
               globally_inverted=bool(has.any() and flipped.sum() / has.sum() > 0.5))
    return rep


def _validate_mesh(vertices: np.ndarray, faces: np.ndarray, voxel: float | None = None) -> dict[str, Any]:
    """Measured validity of the fused mesh (nothing here is a threshold; the numbers are reported and gated structurally by the caller)."""
    rep: dict[str, Any] = {"vertices": int(len(vertices)), "triangles": int(len(faces))}
    if len(faces) == 0:
        return {**rep, "valid": False, "problems": ["no triangles were extracted"]}
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    problems: list[str] = []
    finite = bool(np.isfinite(vertices).all())
    if not finite:
        problems.append("non-finite vertex coordinates")
    tri = vertices[faces]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    degenerate = int((area <= 1e-12).sum())
    if degenerate:
        problems.append(f"{degenerate} degenerate triangle(s)")
    dup = int(len(faces) - len(np.unique(np.sort(faces, axis=1), axis=0)))
    if dup:
        problems.append(f"{dup} duplicate triangle(s)")
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    uniq, counts = np.unique(edges, axis=0, return_counts=True)
    nonmanifold, boundary = int((counts > 2).sum()), int((counts == 1).sum())
    if nonmanifold:
        problems.append(f"{nonmanifold} non-manifold edge(s)")
    directed = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    _, dcount = np.unique(directed, axis=0, return_counts=True)
    inconsistent_winding = int((dcount > 1).sum())   # an edge traversed twice in the same direction: neighbouring triangles wound oppositely
    if inconsistent_winding:
        problems.append(f"{inconsistent_winding} edge(s) with inconsistent triangle winding")
    loops = _boundary_loops(vertices, uniq, counts)
    loops.sort(key=lambda lp: -lp["approx_area_gauge2"])
    n = len(vertices)
    graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    face_label = labels[faces[:, 0]]
    comp_sizes = np.sort(np.unique(face_label, return_counts=True)[1])[::-1]
    co = _coincident_triangles(vertices, faces, 0.1 * voxel) if voxel else None
    return {**rep, "valid": not problems, "problems": problems, "finite": finite, "degenerate_triangles": degenerate, "duplicate_triangles": dup,
            "non_manifold_edges": nonmanifold, "boundary_edges": boundary, "inconsistent_winding_edges": inconsistent_winding,
            "boundary_loops": {"count": len(loops), "total_approx_area_gauge2": round(float(sum(lp["approx_area_gauge2"] for lp in loops)), 6),
                               "largest": loops[:10],
                               "note": "a loop is a hole OR the open border of the observed surface: the two cannot be told apart from the mesh alone"},
            "coincident_sheets": None if co is None else {k: v for k, v in co.items() if k != "remove"},
            "components": int(len(comp_sizes)), "largest_component_triangles": int(comp_sizes[0]), "surface_area_gauge2": round(float(area.sum()), 6),
            "bounds_gauge": [np.min(vertices, axis=0).round(4).tolist(), np.max(vertices, axis=0).round(4).tolist()]}


def vertex_provenance(vertices: np.ndarray, views: Sequence[FusionView], kept_depth: Sequence[np.ndarray], conf: Sequence[np.ndarray],
                      voxel: float, rel_tol: float) -> dict[str, np.ndarray]:
    """Which views' INTEGRATED depth each vertex lies on. contrib (nv x nview uint8), weight (confidence at that pixel, float16), residual
    (signed z - depth at that pixel, NaN when the view did not support the vertex), plus the derived vertex confidence and uncertainty."""
    nv, nw = len(vertices), len(views)
    contrib = np.zeros((nv, nw), np.uint8)
    weight = np.zeros((nv, nw), np.float16)
    resid = np.full((nv, nw), np.nan, np.float32)
    tol_abs = 2.0 * voxel
    for k, v in enumerate(views):
        fx, fy, cx, cy, R, t = _cam_tuple(v)
        u, vv, z, inside = _project(vertices, fx, fy, cx, cy, R, t, v.depth.shape)
        d = np.zeros(nv)
        d[inside] = kept_depth[k][vv[inside], u[inside]]
        on = inside & (d > 0) & (np.abs(z - d) <= tol_abs + rel_tol * d)
        contrib[:, k] = on
        weight[on, k] = conf[k][vv[on], u[on]]
        resid[on, k] = (z - d)[on]
    wsum = weight.astype(np.float32).sum(axis=1)
    nview = contrib.sum(axis=1)
    sq = np.where(np.isnan(resid), 0.0, resid) ** 2
    rms = np.where(nview >= 2, np.sqrt(sq.sum(axis=1) / np.maximum(nview, 1)), np.nan).astype(np.float32)
    return {"contrib": contrib, "weight": weight, "residual": resid, "vertex_confidence": (wsum / (wsum + 1.0)).astype(np.float32),
            "vertex_view_count": nview.astype(np.int16), "vertex_residual_rms": rms}


def _components_with_views(vertices: np.ndarray, faces: np.ndarray, contrib: np.ndarray, names: Sequence[str], weight: np.ndarray) -> list[dict[str, Any]]:
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    n = len(vertices)
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    _, labels = connected_components(coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)), directed=False)
    tri_label = labels[faces[:, 0]]
    nview = contrib.sum(axis=1)
    out = []
    for lab, ntri in zip(*np.unique(tri_label, return_counts=True), strict=True):
        vs = np.unique(faces[tri_label == lab].ravel())
        sub = contrib[vs].astype(bool)
        views = [{"source": names[k], "vertices_supported": int(sub[:, k].sum()), "fraction": round(float(sub[:, k].mean()), 4),
                  "mean_weight": round(float(weight[vs][sub[:, k], k].astype(np.float32).mean()), 4)} for k in range(len(names)) if sub[:, k].any()]
        out.append({"id": int(lab), "triangles": int(ntri), "vertices": int(len(vs)), "views": views,
                    "views_with_ge_5pct_of_vertices": [x["source"] for x in views if x["fraction"] >= 0.05],
                    "vertices_unsupported": int((nview[vs] == 0).sum()), "vertices_single_view": int((nview[vs] == 1).sum()),
                    "bounds_gauge": [vertices[vs].min(axis=0).round(4).tolist(), vertices[vs].max(axis=0).round(4).tolist()]})
    out.sort(key=lambda c: -c["triangles"])
    return out


# ---------------------------------------------------------------------------------------------------------------- integration (Open3D)
class _LibraryUnavailable(RuntimeError):
    pass


def _integrate_and_extract(views: Sequence[FusionView], kept_depth: Sequence[np.ndarray], *, z_med: float, origin: np.ndarray, side: float,
                           resolution: int, trunc: float, depth_trunc_factor: float, min_cluster_triangles: int,
                           should_stop: Callable[[int], bool] | None) -> dict[str, Any]:
    """TSDF integration of the already-screened, already-consistency-filtered depth, then marching cubes and cleanup. The ONLY Open3D code.
    Returns the mesh arrays and what happened: ``integrated`` (view indices actually integrated), ``interrupted`` (reason or None)."""
    try:
        import open3d as o3d
    except Exception as exc:  # noqa: BLE001
        raise _LibraryUnavailable(f"{type(exc).__name__}: {exc}") from exc
    volume = o3d.pipelines.integration.UniformTSDFVolume(length=side, resolution=resolution, sdf_trunc=trunc,
                                                         color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8, origin=origin)
    integrated: list[int] = []
    interrupted: str | None = None
    for k, v in enumerate(views):
        if should_stop is not None and should_stop(k):
            interrupted = f"stop requested before view {k} ({v.source_id})"
            break
        try:
            h, w = v.depth.shape
            fx, fy, cx, cy, R, t = _cam_tuple(v)
            T_cw = np.eye(4)
            T_cw[:3, :3], T_cw[:3, 3] = R, t
            extrinsic = np.linalg.inv(T_cw)  # Open3D integrates with WORLD-TO-CAMERA
            color = o3d.geometry.Image(np.ascontiguousarray(v.color.astype(np.uint8)))
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, o3d.geometry.Image(np.ascontiguousarray(kept_depth[k], dtype=np.float32)),
                                                                      depth_scale=1.0, depth_trunc=depth_trunc_factor * z_med, convert_rgb_to_intensity=False)
            volume.integrate(rgbd, o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy), extrinsic)
        except Exception as exc:  # noqa: BLE001 -- the views integrated so far are kept; the result is then PARTIAL, never complete
            interrupted = f"integration failed at view {k} ({v.source_id}): {type(exc).__name__}: {exc}"
            break
        integrated.append(k)
    out: dict[str, Any] = {"integrated": integrated, "interrupted": interrupted, "vertices": np.zeros((0, 3)), "faces": np.zeros((0, 3), np.int64),
                           "colors": None, "raw_triangles": 0, "removed_floating_specks_triangles": 0, "removed_zero_area_triangles": 0,
                           "suppressed_coincident_triangles": 0}
    if not integrated:
        return out
    mesh = volume.extract_triangle_mesh()
    out["raw_triangles"] = int(len(mesh.triangles))
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_unreferenced_vertices()
    if len(mesh.triangles):
        tri_clusters, cluster_n, _ = mesh.cluster_connected_triangles()
        tri_clusters, cluster_n = np.asarray(tri_clusters), np.asarray(cluster_n)
        small = np.flatnonzero(cluster_n < min_cluster_triangles)
        if len(small):
            out["removed_floating_specks_triangles"] = int(cluster_n[small].sum())
            mesh.remove_triangles_by_mask(np.isin(tri_clusters, small))
            mesh.remove_unreferenced_vertices()
    if len(mesh.triangles):  # marching cubes can leave numerically zero-area slivers that Open3D's own cleanup keeps: remove them explicitly
        tv, tf = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
        tri = tv[tf]
        zero = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) <= 1e-12
        out["removed_zero_area_triangles"] = int(zero.sum())
        if zero.any():
            mesh.remove_triangles_by_mask(zero)
            mesh.remove_unreferenced_vertices()
    if len(mesh.triangles):  # coincident duplicate sheets: keep one of each same-direction pair
        co = _coincident_triangles(np.asarray(mesh.vertices), np.asarray(mesh.triangles), 0.1 * side / resolution)
        if co["suppressed_triangles"]:
            mesh.remove_triangles_by_mask(co["remove"])
            mesh.remove_unreferenced_vertices()
        out["suppressed_coincident_triangles"] = co["suppressed_triangles"]
    out.update(vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.triangles),
               colors=(np.clip(np.asarray(mesh.vertex_colors), 0, 1) * 255).astype(np.uint8) if mesh.has_vertex_colors() else None)
    return out


def _export_ply(vertices: np.ndarray, faces: np.ndarray, colors: np.ndarray | None) -> bytes:
    import trimesh

    tm = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    if colors is not None and len(colors) == len(vertices):
        tm.visual.vertex_colors = colors
    buf = io.BytesIO()
    tm.export(buf, file_type="ply")
    return buf.getvalue()


# ---------------------------------------------------------------------------------------------------------------- orchestration
def fuse_depth_to_surface(views: Sequence[FusionView], *, voxel_fraction: float = DEFAULT_VOXEL_FRACTION,
                          trunc_voxels: float = DEFAULT_TRUNC_VOXELS, depth_trunc_factor: float = DEFAULT_DEPTH_TRUNC_FACTOR,
                          consistency_rel_tol: float = DEFAULT_CONSISTENCY_REL_TOL, min_cluster_triangles: int = MIN_CLUSTER_TRIANGLES,
                          edge_rel_tol: float = DEFAULT_EDGE_REL_TOL, min_confidence: float = DEFAULT_MIN_CONFIDENCE,
                          should_stop: Callable[[int], bool] | None = None, strict: bool = False) -> FusionResult:
    """Fuse ``views`` into one surface. Nothing is returned for fewer than two usable views: a single depth map cannot be checked for
    consistency and a one-view "fusion" would only re-wrap that depth map.

    Bad inputs never produce a silent mesh: a view with a missing/corrupt artifact, a wrong camera transform, a mismatched depth scale or no
    verified overlap is EXCLUDED and listed in ``report["views_excluded"]`` (status ``partial``); when fewer than two views remain the result
    is ``skipped`` with ``error_code``. ``strict=True`` raises ``FusionInputError`` on the first such fault instead. ``should_stop(k)`` is polled
    before each view is integrated: a stop (or an integration failure) yields a ``partial`` result from the views done so far."""
    base: dict[str, Any] = {"method": "open3d.UniformTSDFVolume (integrate + marching cubes)", "scale_state": "SFM_GAUGE", "scale_units": "sfm_units",
                            "physically_metric": False, "views_offered": len(views),
                            "parameters": {"voxel_fraction_of_median_depth": voxel_fraction, "sdf_trunc_voxels": trunc_voxels,
                                           "depth_trunc_factor": depth_trunc_factor, "consistency_rel_tol": consistency_rel_tol,
                                           "edge_rel_tol": edge_rel_tol, "min_confidence": min_confidence}}
    excluded: list[dict[str, Any]] = []
    iv: dict[str, Any] = {"views_excluded": excluded}
    base["input_validation"] = iv
    usable: list[FusionView] = []

    def exclude(source: str, code: str, detail: str):
        if strict:
            raise FusionInputError(code, detail, source)
        excluded.append({"source": source, "code": code, "detail": detail})

    def skip(reason: str, code: str) -> FusionResult:
        base.update(views_with_valid_depth=len(usable), complete=False, error_code=code, views_excluded=excluded)
        return FusionResult("skipped", reason, base, error_code=code)

    # 1. screening: each view is validated and cleaned on its own
    pre: dict[str, dict[str, Any]] = {}
    for v in views:
        cv, rec = _screen_view(v, edge_rel_tol)
        pre[v.source_id] = rec
        if cv is None:
            exclude(v.source_id, rec["code"], rec["detail"])
        else:
            usable.append(cv)
    iv["screening"] = list(pre.values())
    if len(usable) < 2:
        code = SINGLE_VIEW if not excluded else excluded[0]["code"]
        return skip("fewer than two views with valid aligned depth: no cross-view consistency can be established"
                    + (f" ({len(excluded)} view(s) excluded: {sorted({e['code'] for e in excluded})})" if excluded else ""), code)
    z_med = float(np.median(np.concatenate([v.depth[v.depth > 0] for v in usable])))
    for v in usable:  # depth beyond the truncation range is not integrated (Open3D would ignore it silently): count it instead
        far = v.depth > depth_trunc_factor * z_med
        pre[v.source_id]["beyond_depth_trunc"] = int(far.sum())
        v.depth[far] = 0.0
    voxel = voxel_fraction * z_med
    trunc = trunc_voxels * voxel
    base.update({"median_depth_gauge": round(z_med, 6), "voxel_length_gauge": round(voxel, 6), "sdf_trunc_gauge": round(trunc, 6)})

    # 2. probes: wrong transform / depth scale
    probe = probe_inputs(usable, consistency_rel_tol)
    iv["probe"] = probe
    if probe["global_wrong_transform"]:
        detail = (f"every view disagrees with every other as given, and all agree when the camera transform is read as "
                  f"'{probe['global_wrong_transform']}': the whole pose set uses a different convention")
        if strict:
            raise FusionInputError(WRONG_CAMERA_TRANSFORM, detail)
        iv["wrong_convention_for_all_views"] = probe["global_wrong_transform"]
        return skip(detail, WRONG_CAMERA_TRANSFORM)
    flagged = {p["source"]: p for p in probe["views"] if p["verdict"] != "ok"}
    scale_flagged = [s for s, p in flagged.items() if p["verdict"] == DEPTH_SCALE_MISMATCH]
    if scale_flagged and 2 * len(scale_flagged) >= len(usable):
        iv["scale_attribution_ambiguous"] = True   # with two views (or half the set) the pair disagree by a factor and the data cannot say which is wrong
    for s, p in flagged.items():
        if p["verdict"] == WRONG_CAMERA_TRANSFORM:
            detail = (f"camera transform inconsistent with the other views; agrees under '{p['suggested_convention']}' "
                      f"(support {p['best_alternative_pose']['support']:.2f} vs {p['support_as_given']:.2f} as given)")
        else:
            detail = (f"depth scale disagrees with the other views by an estimated factor of {p['estimated_depth_scale_factor']:.3f} "
                      f"(support {p['best_depth_scale']['support']:.2f} vs {p['support_as_given']:.2f} as given)")
        exclude(s, p["verdict"], detail)
    usable = [v for v in usable if v.source_id not in flagged]
    if len(usable) < 2:
        first = next(iter(flagged.values()))["verdict"] if flagged else SINGLE_VIEW
        return skip("fewer than two views remain after excluding inconsistent inputs: " + "; ".join(f"{e['source']}: {e['code']}" for e in excluded), first)

    # 3. cross-view consistency + the verified-overlap graph (disconnected views are excluded, never fused on trust)
    cons = cross_view_consistency(usable, consistency_rel_tol)
    graph = _view_graph(cons, [v.source_id for v in usable])
    iv["view_graph"] = {k: graph[k] for k in ("components", "connected", "edges")}
    if not graph["connected"]:
        keep_i = set(graph["component_indices"][0])
        for i, v in enumerate(usable):
            if i not in keep_i:
                exclude(v.source_id, DISCONNECTED_VIEW_GRAPH, "no verified overlap (agreeing depth) with the main group of views: its relative pose and scale cannot be checked")
        usable = [v for i, v in enumerate(usable) if i in keep_i]
        if len(usable) < 2:
            return skip("the view graph is disconnected and no group of two or more views shares verified overlap", DISCONNECTED_VIEW_GRAPH)
        cons = cross_view_consistency(usable, consistency_rel_tol)
    base["views_with_valid_depth"] = len(usable)

    # 4. per-view integration masks: consistency-kept AND confident; recorded accepted / rejected counts
    kept_depth: list[np.ndarray] = []
    conf_full: list[np.ndarray] = []
    per_view: list[dict[str, Any]] = []
    kept_pts = []
    for v, c in zip(usable, cons, strict=True):
        h, w = v.depth.shape
        derived = ((c["support"] + 0.25) / (c["support"] + c["conflict"] + 1.0)).astype(np.float32)   # unsupported 0.25; k supporters -> (k+.25)/(k+1)
        cf = np.zeros(h * w, np.float32)
        cf[c["idx"]] = derived if v.confidence is None else np.asarray(v.confidence, np.float32).ravel()[c["idx"]]
        low = c["keep"] & (cf[c["idx"]] < min_confidence)
        keep = c["keep"] & ~low
        depth = np.zeros(h * w, np.float32)
        kept = c["idx"][keep]
        depth[kept] = v.depth.ravel()[kept]
        kept_depth.append(depth.reshape(h, w))
        conf_full.append(cf.reshape(h, w))
        fx, fy, cx, cy, R, t = _cam_tuple(v)
        pts, _ = _backproject(v.depth, fx, fy, cx, cy, R, t)
        kept_pts.append(pts[keep][::7])
        rec = pre[v.source_id]
        rows = np.array_split(kept_depth[-1] > 0, 4, axis=0)
        per_view.append({"source": v.source_id, "image_id": v.image_id or v.source_id, "depth_record_id": v.depth_record_id,
                         "depth_sha256": rec["depth_sha256"],
                         "depth_pixels": rec["pixels"] - rec["invalid_nonfinite"] - rec["invalid_nonpositive"],
                         "integrated_pixels": int(keep.sum()), "accepted_pixels": int(keep.sum()),
                         "rejected_as_inconsistent": int((~c["keep"]).sum()), "rejected_low_confidence": int(low.sum()),
                         "rejected_occlusion_boundary": rec["occlusion_boundary_pixels"], "rejected_beyond_depth_range": rec.get("beyond_depth_trunc", 0),
                         "invalid_nonfinite": rec["invalid_nonfinite"], "invalid_nonpositive": rec["invalid_nonpositive"],
                         "supported_by_another_view": c["supported"], "unsupported_single_view": c["unsupported"],
                         "confidence_source": "caller" if v.confidence is not None else "derived_from_cross_view_consistency",
                         "mean_confidence_integrated": round(float(cf[kept].mean()), 4) if len(kept) else 0.0,
                         "coverage_fraction": round(float(keep.sum() / (h * w)), 4),
                         "coverage_grid_4x4": [[round(float(cell.mean()), 3) for cell in np.array_split(r, 4, axis=1)] for r in rows]})
    allp = np.concatenate(kept_pts)
    if len(allp) < 100:
        base.update(per_view=per_view, complete=False, views_excluded=excluded)
        return FusionResult("ran", "too few consistent depth pixels to define a volume", base)
    # Volume bounds from the KEPT depth itself (1st-99th percentile per axis, so a few stray pixels cannot inflate the volume), in gauge units.
    lo, hi = np.percentile(allp, 1, axis=0), np.percentile(allp, 99, axis=0)
    side = float(np.max(hi - lo)) + 2.0 * (trunc_voxels * voxel)
    resolution = int(np.clip(np.ceil(side / voxel), MIN_RESOLUTION, MAX_RESOLUTION))
    voxel = side / resolution                      # the EFFECTIVE voxel (gauge units); capped resolution makes it coarser than requested
    trunc = trunc_voxels * voxel
    origin = (lo + hi) / 2.0 - side / 2.0
    base.update({"voxel_length_gauge": round(voxel, 6), "sdf_trunc_gauge": round(trunc, 6), "volume_resolution": resolution,
                 "volume_side_gauge": round(side, 6), "resolution_capped": bool(resolution == MAX_RESOLUTION)})

    # 5. integrate + extract
    try:
        got = _integrate_and_extract(usable, kept_depth, z_med=z_med, origin=origin, side=side, resolution=resolution, trunc=trunc,
                                     depth_trunc_factor=depth_trunc_factor, min_cluster_triangles=min_cluster_triangles, should_stop=should_stop)
    except _LibraryUnavailable as exc:
        return skip(f"fusion library unavailable ({exc})", "LIBRARY_UNAVAILABLE")
    integrated = got["integrated"]
    interrupted = got["interrupted"]
    names = [v.source_id for v in usable]
    not_integrated = [names[k] for k in range(len(usable)) if k not in integrated]
    for k, pv in enumerate(per_view):
        pv["integrated_into_volume"] = k in integrated
    base["raw_triangles_before_cleanup"] = got["raw_triangles"]
    complete = not excluded and interrupted is None and not not_integrated
    degradations: list[str] = [f"view {e['source']} excluded: {e['code']}" for e in excluded]
    if interrupted:
        degradations.append(f"interrupted: {interrupted}")
    first_code = excluded[0]["code"] if excluded else INTERRUPTED
    base.update(views_integrated=[names[k] for k in integrated], views_not_integrated=not_integrated, complete=complete, degradations=degradations,
                views_excluded=excluded)
    if not integrated:
        base["per_view"] = per_view
        return FusionResult("skipped", f"no view was integrated before the job stopped ({interrupted})", base, error_code=INTERRUPTED)
    vertices, faces = got["vertices"], got["faces"]
    validity = _validate_mesh(vertices, faces, voxel)
    validity["removed_floating_specks_triangles"] = got["removed_floating_specks_triangles"]
    validity["removed_zero_area_triangles"] = got["removed_zero_area_triangles"]
    validity["suppressed_coincident_triangles"] = got["suppressed_coincident_triangles"]
    report = {**base, "per_view": per_view, "mesh": validity}
    status = "ran" if complete else "partial"
    reason = None if complete else "PARTIAL: " + "; ".join(degradations)
    if len(faces) == 0:
        return FusionResult(status, "fusion executed but the fused volume contains no surface", report, error_code=None if complete else first_code)

    # 6. contribution, provenance, coverage, confidence (per vertex, aggregated per component)
    integ_views = [usable[k] for k in integrated]
    vp = vertex_provenance(vertices, integ_views, [kept_depth[k] for k in integrated], [conf_full[k] for k in integrated], voxel, consistency_rel_tol)
    contrib, nview = vp["contrib"], vp["vertex_view_count"]
    for k in range(len(usable)):
        per_view[k].update(mesh_vertices_on_this_depth=0, contributed_to_surface=False)
    for j, k in enumerate(integrated):
        per_view[k]["mesh_vertices_on_this_depth"] = int(contrib[:, j].sum())
        per_view[k]["contributed_to_surface"] = bool(contrib[:, j].any() and per_view[k]["integrated_pixels"] > 0)
    report["mesh_support"] = {"vertices_supported_by_ge1_view": round(float((nview >= 1).mean()), 4),
                              "vertices_supported_by_ge2_views": round(float((nview >= 2).mean()), 4),
                              "views_that_did_not_contribute": [p["source"] for p in per_view if not p["contributed_to_surface"]]}
    comps = _components_with_views(vertices, faces, contrib, [names[k] for k in integrated], vp["weight"])
    validity["components_detail"] = comps[:20]
    centres = np.stack([_pose(v.camera)[1] for v in integ_views])
    orient = orientation_vs_views(vertices, faces, contrib, centres)
    orient["note"] = ("winding convention of Open3D's marching cubes not verified on the development machine (no Open3D wheel for its Python); "
                      "`globally_inverted` is reported here instead of being folded into `valid`")
    validity["orientation_vs_viewing_rays"] = orient
    if orient["globally_inverted"]:
        degradations.append("mesh normals point away from the cameras that saw them (see mesh.orientation_vs_viewing_rays)")
    unsup, single = nview == 0, nview == 1
    rms = vp["vertex_residual_rms"]
    meas = rms[~np.isnan(rms)]
    weak = unsup | single
    report["unsupported_regions"] = {
        "vertices_not_on_any_integrated_depth": int(unsup.sum()), "vertices_on_exactly_one_view": int(single.sum()),
        "unsupported_fraction": round(float(unsup.mean()), 4), "single_view_fraction": round(float(single.mean()), 4),
        "bounds_of_single_or_unsupported_vertices_gauge": (None if not weak.any() else
                                                           [vertices[weak].min(axis=0).round(4).tolist(), vertices[weak].max(axis=0).round(4).tolist()]),
        "views_missing": [e["source"] for e in excluded] + not_integrated,
        "statement": "the surface only exists where views contributed; vertices on one view are unverified by any other view, excluded views are listed in views_excluded"}
    report["geometry_confidence"] = {
        "mean_vertex_confidence": round(float(vp["vertex_confidence"].mean()), 4),
        "vertex_confidence_definition": "sum of supporting-view weights w / (w + 1); weight = confidence at the supporting pixel (heuristic, uncalibrated)",
        "uncertainty_measured_gauge": {"vertices_with_measured_spread": int(len(meas)),
                                       "rms_residual_gauge": round(float(np.sqrt((meas ** 2).mean())), 6) if len(meas) else None,
                                       "p95_residual_gauge": round(float(np.percentile(meas, 95)), 6) if len(meas) else None,
                                       "single_view_vertices_unmeasured": int(single.sum()),
                                       "quantisation_floor_gauge": round(voxel / 2.0, 6),
                                       "note": "RMS of the signed depth residuals of the views a vertex lies on (needs >= 2 views); gauge units, not metres"}}
    report["degradations"] = degradations

    ply = _export_ply(vertices, faces, got["colors"])
    mesh_sha = hashlib.sha256(ply).hexdigest()
    views_meta = []
    for j, k in enumerate(integrated):
        v, pv = usable[k], per_view[k]
        _, t = _pose(v.camera)
        e = v.camera.extrinsics
        views_meta.append({"index": j, "source_id": v.source_id, "image_id": pv["image_id"], "depth_record_id": pv["depth_record_id"],
                           "depth_sha256": pv["depth_sha256"],
                           "pose": {"convention": "camera-to-world, OpenCV camera frame (x right, y down, z forward)", "position": t.tolist(),
                                    "quaternion_wxyz": [float(e.rotation.w), float(e.rotation.x), float(e.rotation.y), float(e.rotation.z)]},
                           "intrinsics_at_depth_resolution": dict(zip(("fx", "fy", "cx", "cy"), _intrinsics_at(v.camera, v.depth.shape), strict=True)),
                           "depth_shape": list(v.depth.shape), "integrated_pixels": pv["integrated_pixels"],
                           "weight": pv["mean_confidence_integrated"], "confidence_source": pv["confidence_source"],
                           "mesh_vertices_on_this_depth": pv["mesh_vertices_on_this_depth"]})
    npz = io.BytesIO()
    np.savez_compressed(npz, **vp)
    npz_bytes = npz.getvalue()
    prov = {"schema": PROVENANCE_SCHEMA,
            "granularity": "per vertex (contributing-view table in the npz) aggregated per connected component; NOT per voxel, NOT per triangle patch",
            "weight_meaning": "confidence at the supporting pixel (caller's map or derived from cross-view consistency); the TSDF itself integrates with unit weight",
            "mesh_sha256": mesh_sha, "mesh_vertices": int(len(vertices)), "mesh_triangles": int(len(faces)), "complete": complete, "status": status,
            "vertex_table": {"sha256": hashlib.sha256(npz_bytes).hexdigest(), "format": "npz",
                             "arrays": {"contrib": f"uint8 [{len(vertices)}, {len(integrated)}] 1 = vertex lies on that view's integrated depth (column = views[].index)",
                                        "weight": "float16, same shape: confidence at the supporting pixel (0 where not supporting)",
                                        "residual": "float32, same shape: signed z - depth at the supporting pixel (NaN where not supporting)",
                                        "vertex_confidence": "float32 [n]", "vertex_view_count": "int16 [n]",
                                        "vertex_residual_rms": "float32 [n], NaN with fewer than 2 supporting views"}},
            "views": views_meta, "views_excluded": excluded, "views_not_integrated": not_integrated, "components": comps}
    pj = json.dumps(prov, sort_keys=True, separators=(",", ":"), default=str).encode()
    report["provenance"] = {"schema": PROVENANCE_SCHEMA, "granularity": prov["granularity"], "json_sha256": hashlib.sha256(pj).hexdigest(),
                            "vertex_table_sha256": prov["vertex_table"]["sha256"], "mesh_sha256": mesh_sha, "components": len(comps),
                            "views": len(views_meta)}
    return FusionResult(status, reason, report, ply, vertices, faces, None if complete else first_code, complete, pj, npz_bytes)


# ---------------------------------------------------------------------------------------------------------------- persistence
def persist_fusion(result: FusionResult, directory: str | Path, stem: str) -> dict[str, Any]:
    """Write mesh + provenance sidecars next to each other, ATOMICALLY: every file goes to a temp name in the target directory first and is
    renamed into place (sidecars before the mesh, so a visible mesh always has its provenance). Any failure removes whatever this call
    placed and leaves nothing half-written. A partial result is written as ``<stem>.partial.ply`` and the manifest says ``complete: false``."""
    if result.mesh_ply is None or result.provenance_json is None or result.provenance_npz is None:
        raise FusionInputError("NO_SURFACE", f"nothing to persist (status {result.status}: {result.reason})")
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    suffix = "" if result.complete else ".partial"
    finals = {"provenance_vertex_table": d / f"{stem}{suffix}.provenance.npz", "provenance": d / f"{stem}{suffix}.provenance.json", "mesh": d / f"{stem}{suffix}.ply"}
    payload = {"provenance_vertex_table": result.provenance_npz, "provenance": result.provenance_json, "mesh": result.mesh_ply}
    tmps: dict[str, Path] = {}
    placed: list[Path] = []
    try:
        for k, data in payload.items():
            tmp = d / f".{finals[k].name}.{os.getpid()}.tmp"
            tmps[k] = tmp
            with open(tmp, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
        for k in ("provenance_vertex_table", "provenance", "mesh"):
            os.replace(tmps[k], finals[k])
            placed.append(finals[k])
    except BaseException:
        for p in [*tmps.values(), *placed]:
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass
        raise
    return {"complete": result.complete, "status": result.status,
            **{k: {"path": str(finals[k]), "sha256": hashlib.sha256(payload[k]).hexdigest(), "bytes": len(payload[k])} for k in finals}}
