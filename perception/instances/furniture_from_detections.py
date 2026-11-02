"""2D furniture detections (one or more photos) -> accurately PLACED 3D furniture entities.

Classical and testable, no learned depth. Per detection the ray through the bottom-centre of its box (or the
lowest mask contact point) is cast from the registered camera and intersected with the FLOOR plane. That gives the
floor contact point independent of depth quality. The contact is the *front* edge of the object, so the footprint
centre is pushed back along the horizontal viewing direction by half a class-prior depth. Footprint extents come
from the box width back-projected at that depth, clamped to the class prior; height is the class prior. Detections
of the same class whose floor points agree across views are fused (score weighted); confidence rises with the
number of agreeing views and falls with their spread. Single-view furniture is capped at ``SINGLE_VIEW_CAP``.

Every dimension is labelled by state (footprint ESTIMATED from image, height ESTIMATED class prior); yaw is
``wall_aligned`` only when it truly snaps to a wall direction (<= 20 deg), otherwise ``unknown`` -- an orientation is
never faked. Nothing is invented: unknown labels go to ``rejected`` with a reason.

Integration point (NOT wired yet): the room compiler should call ``furniture_from_detections`` after the room layout
and registered camera poses are solved, passing ``RoomFrame`` from the solved floor plane / wall slabs, and append
``result.entities`` to the world's furniture entities; ``result.rejected`` is diagnostics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from world_ir.measure import make_dimension

SINGLE_VIEW_CAP = 0.5
WALL_SNAP_DEG = 20.0
CLUSTER_RADIUS_M = 0.6   # brief said 0.5; measured oblique single-view scatter on the fidelity room reaches ~0.55
CLAMP_FLAG_M = 0.3
SIZE_TOLERANCE = (0.75, 1.33)      # measured footprint scale vs the class prior is clamped to this band

#: class -> (length, width, height) metres; length is the longer footprint side.
SIZE_PRIORS: Dict[str, Tuple[float, float, float]] = {
    "sofa": (2.0, 0.9, 0.85), "armchair": (0.9, 0.9, 0.9), "chair": (0.5, 0.5, 0.9),
    "dining_table": (1.4, 0.9, 0.75), "coffee_table": (1.0, 0.6, 0.45), "desk": (1.4, 0.7, 0.75),
    "bed": (2.0, 1.6, 0.5), "wardrobe": (1.0, 0.6, 1.9), "bookshelf": (0.9, 0.3, 1.8),
    "dresser": (1.0, 0.5, 0.85), "nightstand": (0.45, 0.4, 0.5), "tv_stand": (1.4, 0.4, 0.5),
    "tv": (1.0, 0.1, 0.6), "lamp": (0.35, 0.35, 1.5), "stool": (0.4, 0.4, 0.5), "cabinet": (0.8, 0.4, 0.9),
}
ALIASES = {"couch": "sofa", "settee": "sofa", "table": "dining_table", "dining table": "dining_table",
           "coffee table": "coffee_table", "closet": "wardrobe", "armoire": "wardrobe", "bookcase": "bookshelf",
           "shelf": "bookshelf", "television": "tv", "monitor": "tv", "floor lamp": "lamp", "tv stand": "tv_stand",
           "side table": "nightstand", "chest of drawers": "dresser", "table lamp": "lamp"}
#: classes that cannot be a floor-standing footprint with the prior depth (wall/ceiling items) are not supported.


@dataclass(frozen=True)
class FurnitureDetection:
    label: str
    score: float
    box_xyxy: Tuple[float, float, float, float]
    view_id: str
    mask: Optional[Any] = None           # optional HxW bool array in the same image as the box


@dataclass(frozen=True)
class FurnitureCamera:
    """Camera-to-world pose (``R`` columns = right, down, forward in world) + pinhole intrinsics."""
    view_id: str
    R: Tuple[Tuple[float, float, float], ...]
    t: Tuple[float, float, float]
    fx: float
    fy: float
    cx: float
    cy: float
    width: Optional[int] = None      # image size: lets a box clipped by the frame edge be down-weighted
    height: Optional[int] = None

    @staticmethod
    def from_pose(view_id: str, pose: Any, intr: Any) -> "FurnitureCamera":
        return FurnitureCamera(view_id, tuple(tuple(r) for r in pose.R), tuple(pose.t),
                               intr.fx, intr.fy, intr.cx, intr.cy, intr.width, intr.height)


@dataclass(frozen=True)
class RoomFrame:
    """Solved room: floor height and an (optionally wall-rotated) rectangular footprint.
    ``min_xy``/``max_xy`` are in the frame rotated by ``wall_yaw`` (rad) about the world origin."""
    floor_z: float
    min_xy: Tuple[float, float]
    max_xy: Tuple[float, float]
    wall_yaw: float = 0.0


@dataclass
class FurnitureResult:
    entities: List[dict] = field(default_factory=list)
    rejected: List[dict] = field(default_factory=list)


def _canon(label: str) -> Optional[str]:
    k = label.strip().lower().replace("_", " ")
    k = ALIASES.get(k, k).replace(" ", "_")
    return k if k in SIZE_PRIORS else None


def _rot(a: float) -> np.ndarray:
    return np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])


def _floor_contact(cam: FurnitureCamera, det: FurnitureDetection, floor_z: float):
    """Floor point under the object's lowest visible edge + its forward depth, or (None, reason)."""
    x0, y0, x1, y1 = det.box_xyxy
    u, v = (x0 + x1) / 2.0, y1
    if det.mask is not None:
        m = np.asarray(det.mask, dtype=bool)
        ys, xs = np.nonzero(m)
        if len(ys):
            vmax = ys.max()
            low = xs[ys >= vmax - 1]
            u, v = float(low.mean()), float(vmax + 1)
    R = np.array(cam.R, float)
    o = np.array(cam.t, float)
    d = R @ np.array([(u - cam.cx) / cam.fx, (v - cam.cy) / cam.fy, 1.0])
    if d[2] >= -1e-6:
        return None, "bottom edge is at or above the horizon: it cannot touch the floor"
    s = (floor_z - o[2]) / d[2]
    if s <= 0:
        return None, "camera is below the floor plane"
    p = o + s * d
    fwd = float(R[:, 2] @ (p - o))
    return (p[:2], fwd, o[:2]), None


def _yaw_snap(yaw: float, wall_yaw: float):
    rel = (yaw - wall_yaw) % (math.pi / 2)
    off = min(rel, math.pi / 2 - rel)
    if math.degrees(off) <= WALL_SNAP_DEG:
        snapped = yaw - (rel if rel < math.pi / 4 else rel - math.pi / 2)
        return snapped, "wall_aligned"
    return yaw, "unknown"


def furniture_from_detections(detections: Sequence[FurnitureDetection], cameras: Sequence[FurnitureCamera],
                              room: RoomFrame, *, scale_state: str = "metric") -> FurnitureResult:
    cams = {c.view_id: c for c in cameras}
    res = FurnitureResult()
    obs: List[dict] = []
    for det in detections:
        cls = _canon(det.label)
        if cls is None:
            res.rejected.append({"label": det.label, "view_id": det.view_id, "reason": "unknown furniture label"})
            continue
        cam = cams.get(det.view_id)
        if cam is None:
            res.rejected.append({"label": det.label, "view_id": det.view_id, "reason": "no registered camera"})
            continue
        hit, why = _floor_contact(cam, det, room.floor_z)
        if hit is None:
            res.rejected.append({"label": det.label, "view_id": det.view_id, "reason": why})
            continue
        contact, depth, eye = hit
        L, W, H = SIZE_PRIORS[cls]
        view = contact - eye
        dist = float(np.linalg.norm(view))
        vdir = view / dist if dist > 1e-9 else np.array([1.0, 0.0])
        width_m = (det.box_xyxy[2] - det.box_xyxy[0]) * depth / cam.fx
        # the contact is the front edge: move to the centre by half the prior mean footprint side along the view direction
        centre = contact + vdir * (0.25 * (L + W))
        # an observation can only be inside the room: pull it in by the smallest possible half-footprint so views
        # whose contact overshoots a wall still agree with the other views (the final clamp uses the true footprint)
        rr = _rot(-room.wall_yaw)
        m = 0.5 * W
        cr = rr @ centre
        cr = np.minimum(np.maximum(cr, np.array(room.min_xy) + m), np.maximum(np.array(room.max_xy) - m,
                                                                               np.array(room.min_xy) + m))
        pre_moved = float(np.linalg.norm(cr - rr @ centre))
        centre = _rot(room.wall_yaw) @ cr
        scale = float(np.clip(width_m / ((L + W) / 2.0), *SIZE_TOLERANCE))
        # a box clipped by the image side has a wrong width and a shifted bottom-centre: trust it far less
        trunc = cam.width is not None and (det.box_xyxy[0] <= 1.0 or det.box_xyxy[2] >= cam.width - 1.0)
        if trunc:
            scale = 1.0
        obs.append(dict(cls=cls, centre=centre, view=det.view_id, score=float(det.score), scale=scale,
                        eye=eye, vdir=vdir, label=det.label, pre_moved=pre_moved, w=float(det.score) * (0.2 if trunc else 1.0), trunc=trunc))

    # ---- multi-view fusion: per class greedy clustering on the floor point -------------------------------
    clusters: List[List[dict]] = []
    for o in sorted(obs, key=lambda o: -o["w"]):
        for cl in clusters:
            if cl[0]["cls"] != o["cls"] or any(x["view"] == o["view"] for x in cl):
                continue
            # single linkage: a view joins when it is within the radius of ANY member (single-view scatter is ~0.3 m)
            if min(np.linalg.norm(x["centre"] - o["centre"]) for x in cl) <= CLUSTER_RADIUS_M:
                cl.append(o)
                break
        else:
            clusters.append([o])

    counts: Dict[str, int] = {}
    rot = _rot(-room.wall_yaw)
    for cl in sorted(clusters, key=lambda c: (c[0]["cls"], tuple(c[0]["centre"]))):
        cls = cl[0]["cls"]
        w = np.array([o["w"] for o in cl])
        centre = np.average([o["centre"] for o in cl], axis=0, weights=w)
        spread = float(max(np.linalg.norm(o["centre"] - centre) for o in cl))
        scale = float(np.average([o["scale"] for o in cl], weights=w))
        L, W, H = SIZE_PRIORS[cls]
        L, W = L * scale, W * scale
        # yaw hypothesis: long axis perpendicular to the (mean) viewing direction, then wall snap
        vmean = np.mean([o["vdir"] for o in cl], axis=0)
        yaw0 = math.atan2(vmean[1], vmean[0]) + math.pi / 2
        yaw, basis = _yaw_snap(yaw0, room.wall_yaw)
        if len(cl) > 1 and basis == "unknown":
            basis = "unknown"
        # keep the footprint inside the room (axis-aligned in the room frame)
        c_room = rot @ centre
        ca, sa = abs(math.cos(yaw - room.wall_yaw)), abs(math.sin(yaw - room.wall_yaw))
        hx, hy = 0.5 * (L * ca + W * sa), 0.5 * (L * sa + W * ca)
        lo = np.array([room.min_xy[0] + hx, room.min_xy[1] + hy])
        hi = np.array([room.max_xy[0] - hx, room.max_xy[1] - hy])
        clamped = np.minimum(np.maximum(c_room, lo), np.maximum(lo, hi))
        moved = float(np.linalg.norm(clamped - c_room))
        flags = []
        if moved > 1e-6:
            centre = _rot(room.wall_yaw) @ clamped
        moved = max(moved, max(o["pre_moved"] for o in cl))
        if moved > CLAMP_FLAG_M:
            flags.append(f"clamped_into_room_{moved:.2f}m")
        n = len(cl)
        mean_score = float(np.mean([o["score"] for o in cl]))
        conf = mean_score * min(1.0, 0.5 + 0.2 * (n - 1)) * math.exp(-spread / 0.5)
        if n == 1:
            conf = min(conf, SINGLE_VIEW_CAP)
            flags.append("single_view")
        conf = round(max(0.0, min(0.95, conf)), 3)
        counts[cls] = counts.get(cls, 0) + 1
        name = f"{cls.replace('_', ' ').title()} {counts[cls]}"
        dim = lambda v, basis_txt: make_dimension(v, state="ESTIMATED", basis=basis_txt, scale_state=scale_state,
                                                   uncertainty_m=max(spread, 0.1))
        res.entities.append({
            "type": "furniture", "name": name,
            "custom_properties": {
                "category": cls,
                "position_m": [float(centre[0]), float(centre[1]), float(room.floor_z)],
                "dimensions": {
                    "length_m": dim(L, "footprint estimated from image box width, clamped to class prior"),
                    "width_m": dim(W, "footprint estimated from image box width, clamped to class prior"),
                    "height_m": dim(H, "class prior"),
                },
                "yaw_rad": float(yaw), "yaw_basis": basis,
                "n_views": n, "position_spread_m": round(spread, 3),
            },
            "evidence_ids": sorted({o["view"] for o in cl}),
            "provenance": "ESTIMATED", "confidence": conf, "flags": flags,
        })
    return res
