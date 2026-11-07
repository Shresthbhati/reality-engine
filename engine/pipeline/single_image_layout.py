"""Bridge between the single-image layout solver and the level-0 bootstrap.

``perception.layout.surface_layout.solve_layout`` works in the camera frame (x right, y DOWN, z forward). The bootstrap works
in x right, y UP, z forward, so every vector is converted by negating y. Planes come out as ordinary ``DetectedPlane`` objects
whose inliers are the bootstrap's own grid points, so promotion, furniture placement and everything downstream is unchanged.

Nothing here invents geometry: a surface that the solver refused (no floor, a non-planar ceiling, an unsupported wall) simply
has no plane, and the reason is carried in ``facts``.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from perception.geometry.planes import DetectedPlane, PlaneDetection
from perception.layout.surface_layout import SurfaceLayout, solve_layout
from provenance import Provenance, Uncertainty

MIN_PLANE_INLIERS = 10
OPENING_CONFIDENCE_CEILING = 0.3


def _flip(v: Sequence[float]) -> Tuple[float, float, float]:
    return (float(v[0]), -float(v[1]), float(v[2]))


def layout_planes(analysis, pts: Sequence[Tuple[float, float, float]], pix: Sequence[Tuple[int, int]], w: int, h: int, f: float):
    """Solve the layout and express it as planes over the bootstrap's grid points.

    Returns ``(layout, detection_or_None, up_vector_in_bootstrap_frame, facts)``. ``detection`` is None when the solver could
    not establish a floor, so the caller falls back to RANSAC over the (metric) points rather than getting nothing.
    """
    hfov = math.degrees(2.0 * math.atan(w / (2.0 * f)))
    layout = solve_layout(np.asarray(analysis.depth_m, dtype=np.float64), analysis.masks, hfov_deg=hfov)
    facts: Dict[str, object] = {
        "attempted": True, "method": "metric depth + ADE20K surface labels -> floor/ceiling/wall planes",
        "hfov_deg": round(hfov, 1), "refused": list(layout.refused), "notes": list(layout.notes),
    }
    if layout.camera_height_m is None:
        facts["used"] = False
        facts["reason"] = "; ".join(layout.refused) or "no floor"
        return layout, None, (0.0, 1.0, 0.0), facts

    up_b = _flip(layout.up)
    P = np.asarray(pts, dtype=np.float64)
    ui = np.asarray([p[0] for p in pix])
    vi = np.asarray([p[1] for p in pix])
    z = P[:, 2]
    band = 0.05 + 0.02 * np.abs(z)
    ids = np.asarray([f"bs:{i:05d}" for i in range(len(pts))], dtype=object)

    def plane(plane_id: str, normal_b, d: float, mask_key: str, note: str) -> Optional[DetectedPlane]:
        n = np.asarray(normal_b, dtype=np.float64)
        on_label = analysis.masks[mask_key][vi, ui]
        dist = P @ n + d
        sel = on_label & (np.abs(dist) <= band)
        if int(sel.sum()) < MIN_PLANE_INLIERS:
            return None
        rms = float(np.sqrt(np.mean(dist[sel] ** 2)))
        cnt = int(sel.sum())
        return DetectedPlane(
            plane_id=plane_id, normal=tuple(float(c) for c in n), d=float(d), inlier_ids=sorted(ids[sel].tolist()),
            inlier_rms_distance_m=rms,
            uncertainty=Uncertainty(confidence=min(1.0, cnt / (cnt + 10.0)), note=note),
        )

    planes: List[DetectedPlane] = []
    floor = plane("layout-floor", up_b, layout.camera_height_m, "floor",
                  f"floor from {int(analysis.masks['floor'].sum())} labelled pixels; camera height {layout.camera_height_m:.2f} m "
                  f"(floor rms {layout.floor_rms_m:.3f} m)")
    if floor is not None:
        planes.append(floor)
    if layout.room_height_m is not None:
        ceil = plane("layout-ceiling", tuple(-c for c in up_b), layout.room_height_m - layout.camera_height_m, "ceiling",
                     f"ceiling {layout.room_height_m:.2f} m above the floor")
        if ceil is not None:
            planes.append(ceil)
    for i, wl in enumerate(layout.walls):
        wp = plane(f"layout-wall-{i}", _flip(wl.plane.normal), wl.plane.d, "wall",
                   f"wall {wl.plane.d:.2f} m from the camera, {wl.support_px} supporting pixels")
        if wp is not None:
            planes.append(wp)
    facts.update({
        "used": bool(planes), "camera_height_m": round(layout.camera_height_m, 3),
        "room_height_m": None if layout.room_height_m is None else round(layout.room_height_m, 3),
        "floor_rms_m": round(layout.floor_rms_m or 0.0, 4), "walls": len(layout.walls),
        "openings": [{"kind": o.kind, "wall": o.wall_index, "width_m": round(o.s_max - o.s_min, 2),
                      "sill_m": round(o.h_min, 2), "head_m": round(o.h_max, 2)} for o in layout.openings],
    })
    if not planes:
        return layout, None, up_b, facts
    assigned = {pid for p in planes for pid in p.inlier_ids}
    detection = PlaneDetection(planes=planes, unassigned_point_ids=sorted(set(ids.tolist()) - assigned), points_total=len(pts))
    return layout, detection, up_b, facts


def add_layout_openings(world, obs, layout: SurfaceLayout, eid_of_plane: Dict[str, str], *, pre: str,
                        confidence_ceiling: float) -> Dict[str, object]:
    """Window/door entities from the solved openings (geometry from pixel rays meeting the wall plane, in metres)."""
    from world_ir import Entity, EntityType, Geometry, GeometryType, Vector3

    facts: Dict[str, object] = {"attempted": True, "method": "ADE20K window/door labels projected onto the solved wall plane",
                                "candidates": 0, "skipped": []}
    up = np.asarray(layout.up)
    cam_h = float(layout.camera_height_m)
    cap = min(confidence_ceiling, OPENING_CONFIDENCE_CEILING)
    k = 0
    for o in layout.openings:
        host = eid_of_plane.get(f"layout-wall-{o.wall_index}")
        if host is None:
            facts["skipped"].append(f"{o.kind} on wall {o.wall_index}: that wall was not promoted")
            continue
        wl = layout.walls[o.wall_index]
        n, t = np.asarray(wl.plane.normal), np.asarray(wl.tangent)
        corners = []
        for s_, h_ in ((o.s_min, o.h_min), (o.s_max, o.h_min), (o.s_max, o.h_max), (o.s_min, o.h_max)):
            p = s_ * t + (h_ - cam_h) * up + (-wl.plane.d) * n
            corners.append(_flip(p))
        xs, ys, zs = zip(*corners)
        k += 1
        oid = f"{pre}-opening-{k:02d}"
        geom = Geometry(
            id=f"geom-{oid}", type=GeometryType.BOX, vertex_count=4,
            bounds_min=Vector3(min(xs), min(ys), min(zs)), bounds_max=Vector3(max(xs), max(ys), max(zs)),
            provenance=Provenance.ESTIMATED, confidence=cap, observations=[obs],
        )
        world.geometries[geom.id] = geom
        width, height = o.s_max - o.s_min, o.h_max - o.h_min
        world.entities[oid] = Entity(
            id=oid, type=EntityType.WINDOW if o.kind == "window" else EntityType.DOOR,
            name=f"{o.kind.capitalize()} on {host} (from surface labels)",
            transform={"position": {"x": sum(xs) / 4, "y": sum(ys) / 4, "z": sum(zs) / 4}},
            geometry_ids=[geom.id], provenance=Provenance.INFERRED, confidence=cap,
            uncertainty=Uncertainty(
                confidence=cap,
                note=(f"{o.kind} segmented in the photograph and projected onto its wall plane: "
                      f"{width:.2f} m wide, sill {o.h_min:.2f} m, head {o.h_max:.2f} m (monocular-depth estimate)")),
            observations=[obs],
            custom_properties={"bootstrap": {
                "role": "opening_from_labels", "source": "single_image_layout", "host_entity_id": host,
                "resolved_kind": True, "kind_basis": "ADE20K semantic label",
                "dimensions": {"width_m": round(width, 3), "height_m": round(height, 3), "sill_m": round(o.h_min, 3),
                               "units": "metres, estimated from monocular metric depth"},
            }},
        )
    facts["candidates"] = k
    return facts


def _hull(points):
    """Convex hull (Andrew's monotone chain), counter-clockwise, no repeated end point."""
    pts = sorted(set((round(x, 6), round(y, 6)) for x, y in points))
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def add_layout_room(world, layout: SurfaceLayout, eid_of_plane: Dict[str, str], *, pre: str,
                    confidence_ceiling: float) -> Dict[str, object]:
    """A ROOM entity for the part of the room one photograph saw, so the floor plan / report exporters have something to draw.

    The footprint is the convex hull of the OBSERVED wall segments and the camera position (the camera stands inside the room).
    Everything outside that hull (the sides and rear nobody photographed) is not claimed: the closure is marked inferred and the
    area is a lower bound of the visible part. Walls get ``wall_slab`` data (observed extent, openings with offsets and sizes).
    """
    from world_ir import Entity, EntityType

    from provenance import Provenance as Prov

    facts: Dict[str, object] = {"attempted": True, "created": False}
    up = np.asarray(_flip(layout.up))
    x = np.array([1.0, 0.0, 0.0])
    e1 = x - (x @ up) * up
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(e1, up)  # forward (+z) for a level camera
    height = layout.room_height_m or 0.0
    cap = min(confidence_ceiling, OPENING_CONFIDENCE_CEILING)
    cam_h = float(layout.camera_height_m)

    seg_pts: List[Tuple[float, float]] = [(0.0, 0.0)]
    walls_used = []
    for i, wl in enumerate(layout.walls):
        eid = eid_of_plane.get(f"layout-wall-{i}")
        if eid is None or eid not in world.entities:
            continue
        n, t = np.asarray(_flip(wl.plane.normal)), np.asarray(_flip(wl.tangent))
        p0 = -wl.plane.d * n
        a, b = p0 + wl.s_min * t, p0 + wl.s_max * t
        seg_pts += [(float(a @ e1), float(a @ e2)), (float(b @ e1), float(b @ e2))]
        walls_used.append((i, eid, a, b, wl))
    if len(walls_used) < 1:
        facts["reason"] = "no promoted wall to bound a room"
        return facts
    poly = _hull(seg_pts)
    if len(poly) < 3:
        facts["reason"] = "walls and camera are collinear: no footprint"
        return facts
    area = 0.5 * abs(sum(poly[k][0] * poly[(k + 1) % len(poly)][1] - poly[(k + 1) % len(poly)][0] * poly[k][1] for k in range(len(poly))))
    room_id = f"{pre}-room-01"
    world.entities[room_id] = Entity(
        id=room_id, type=EntityType.ROOM, name="Room (visible part)",
        provenance=Prov.INFERRED, confidence=cap,
        uncertainty=Uncertainty(confidence=cap, note="one photograph: the walls that were seen plus the camera position; sides and rear are not claimed"),
        custom_properties={
            "footprint_polygon_m": [[round(px, 4), round(py, 4)] for px, py in poly],
            "footprint_basis": {"e1": [float(c) for c in e1], "e2": [float(c) for c in e2]},
            "footprint_closure": "inferred: convex hull of the observed wall segments and the camera position",
            "floor_area_m2": round(area, 3), "height_m": round(height, 3) if height else None, "dimensions": {},
            "visible_part_only": True,
        },
    )
    for i, eid, a, b, wl in walls_used:
        ent = world.entities[eid]
        openings = []
        for o in layout.openings:
            if o.wall_index != i:
                continue
            openings.append({"kind": o.kind, "offset_m": round(max(0.0, o.s_min - wl.s_min), 3),
                             "width_m": round(o.s_max - o.s_min, 3), "height_m": round(o.h_max - o.h_min, 3),
                             "sill_m": round(o.h_min, 3)})
        ent.custom_properties.update({
            "room_id": room_id, "length_m": round(wl.s_max - wl.s_min, 3), "height_m": round(height, 3),
            "thickness_m": 0.1, "thickness_basis": "default_unmeasured", "observed_fraction": 1.0,
            "wall_slab": {"start_xyz_m": [float(c) for c in a], "end_xyz_m": [float(c) for c in b], "openings": openings},
        })
    facts.update({"created": True, "room_id": room_id, "area_m2_visible_part": round(area, 2), "walls": len(walls_used)})
    return facts


def make_layout_analyzer():
    """``(analyze, unload)`` for the progressive ladder, or ``(None, noop)`` when REALITY_LAYOUT_MODELS=0.

    ``analyze(item)`` runs metric depth + surface segmentation on the item's photograph; it raises
    ``LayoutModelsUnavailable`` when the weights are not cached, which the ladder records before falling back to the classic path.
    """
    import os
    from pathlib import Path
    from urllib.parse import unquote, urlparse
    from urllib.request import url2pathname

    if os.environ.get("REALITY_LAYOUT_MODELS", "1") == "0":
        return None, (lambda: None)
    from perception.layout.learned_models import LayoutModels, analyze_path

    models = LayoutModels()

    def analyze(item):
        u = urlparse(item.source_uri)
        path = url2pathname(unquote(u.path)) if u.scheme == "file" else item.source_uri
        return analyze_path(models, Path(path))

    return analyze, models.unload
