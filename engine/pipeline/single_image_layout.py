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


SURFACE_KEYS = ("floor", "ceiling", "wall", "window", "door")


def view_input(ev, hfov_deg: float):
    """A registered extra photo (ImageAnalysis-like + pose) as a solver ViewInput."""
    from perception.layout.surface_layout import ViewInput

    return ViewInput(np.asarray(ev.depth_m, dtype=np.float64), ev.masks, hfov_deg, R=np.asarray(ev.R, float), t=np.asarray(ev.t, float),
                     name=str(getattr(ev, "name", "view")))


def extra_view_points(ev, hfov_deg: float, target_points: int):
    """A registered extra photo's depth as grid points in the bootstrap frame (x right, y UP, z forward, reference camera at 0).

    Returns (points (N,3), labels {surface: bool (N,)}, colours [(r,g,b) | None] (N)). Points with no valid depth are dropped.
    """
    d = np.asarray(ev.depth_m, dtype=np.float64)
    h2, w2 = d.shape
    f2 = (w2 / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    step = max(1, int(math.sqrt(w2 * h2 / float(target_points))))
    uu, vv = np.meshgrid(np.arange(step // 2, w2, step), np.arange(step // 2, h2, step))
    z = d[vv, uu]
    ok = np.isfinite(z) & (z > 0.05)
    uu, vv, z = uu[ok], vv[ok], z[ok]
    cam = np.stack([(uu - w2 / 2.0) / f2 * z, (vv - h2 / 2.0) / f2 * z, z], axis=-1)  # solver frame: y DOWN
    ref = cam @ np.asarray(ev.R, float).T + np.asarray(ev.t, float)
    boot = ref * np.array([1.0, -1.0, 1.0])
    labels = {k: np.asarray(ev.masks[k])[vv, uu] if k in ev.masks else np.zeros(len(z), bool) for k in SURFACE_KEYS}
    rgb = getattr(ev, "rgb", None)
    cols = [tuple(int(c) for c in rgb[v, u]) for u, v in zip(uu, vv)] if rgb is not None else [None] * len(z)
    return boot, labels, cols


def layout_planes(analysis, pts: Sequence[Tuple[float, float, float]], labels, w: int, h: int, f: float, extra_views=None):
    """Solve the layout (the reference photo plus any registered extra photos) and express it as planes over the points.

    ``labels``: {surface: bool array over ``pts``} (reference grid points first, then the extra photos' points). Returns
    ``(layout, detection_or_None, up_vector_in_bootstrap_frame, facts)``. ``detection`` is None when the solver could not
    establish a floor, so the caller falls back to RANSAC over the (metric) points rather than getting nothing.
    """
    from perception.layout.surface_layout import ViewInput, solve_layout_views

    hfov = math.degrees(2.0 * math.atan(w / (2.0 * f)))
    views = [ViewInput(np.asarray(analysis.depth_m, dtype=np.float64), analysis.masks, hfov, name="reference")]
    views += [view_input(ev, hfov) for ev in (extra_views or ())]
    layout = solve_layout_views(views)
    facts: Dict[str, object] = {
        "attempted": True, "method": "metric depth + ADE20K surface labels -> floor/ceiling/wall planes",
        "hfov_deg": round(hfov, 1), "refused": list(layout.refused), "notes": list(layout.notes),
    }
    if len(views) > 1:
        left_out = [n for n in layout.notes if "left out" in n]
        facts["fusion"] = {"photos_used": len(views) - len(left_out), "photos_offered": len(views), "refused": {}, "notes": left_out}
    if layout.camera_height_m is None:
        facts["used"] = False
        facts["reason"] = "; ".join(layout.refused) or "no floor"
        return layout, None, (0.0, 1.0, 0.0), facts

    up_b = _flip(layout.up)
    P = np.asarray(pts, dtype=np.float64)
    z = P[:, 2]
    band = 0.05 + 0.02 * np.abs(z)
    ids = np.asarray([f"bs:{i:05d}" for i in range(len(pts))], dtype=object)

    def plane(plane_id: str, normal_b, d: float, mask_key: str, note: str) -> Optional[DetectedPlane]:
        n = np.asarray(normal_b, dtype=np.float64)
        on_label = np.asarray(labels[mask_key], dtype=bool)
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
                  f"floor from {int(np.asarray(labels['floor']).sum())} labelled points; camera height {layout.camera_height_m:.2f} m "
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
    facts["by_opening"] = {}
    for oi, o in enumerate(layout.openings):
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
        facts["by_opening"][oi] = oid
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
            custom_properties={"host_wall_id": host, "dimensions": {
                "width_m": _dim(world, width), "height_m": _dim(world, height), "sill_m": _dim(world, o.h_min)}, "bootstrap": {
                "role": "opening_from_labels", "source": "single_image_layout", "host_entity_id": host,
                "resolved_kind": True, "kind_basis": "ADE20K semantic label",
                "dimensions": {"width_m": round(width, 3), "height_m": round(height, 3), "sill_m": round(o.h_min, 3),
                               "units": "metres, estimated from monocular metric depth"},
            }},
        )
    facts["candidates"] = k
    return facts


FIXTURE_CLASS = {"socket": "electrical.socket", "switch": "electrical.switch", "ceiling_light": "electrical.light",
                 "radiator": "hvac.radiator"}
FIXTURE_SIZE_M = {"socket": 0.08, "switch": 0.08, "ceiling_light": 0.3, "radiator": 0.6}
FIXTURE_DUPLICATE_M = 0.15


def add_layout_fixtures(world, obs, layout: SurfaceLayout, detections, eid_of_plane: Dict[str, str], *, pre: str,
                        hfov_deg: float, image_size: Tuple[int, int], confidence_ceiling: float) -> Dict[str, object]:
    """Sockets, switches, radiators and ceiling lights as entities on their wall/ceiling, with height above the floor."""
    from perception.layout.fixtures import place_fixtures
    from world_ir import Entity, EntityType, Geometry, GeometryType, Vector3

    placed, refused = place_fixtures(detections, layout, hfov_deg=hfov_deg, image_size=image_size)
    cap = min(confidence_ceiling, OPENING_CONFIDENCE_CEILING)
    kept = []
    for fx in sorted(placed, key=lambda f: -f.score):
        if any(k.kind == fx.kind and np.linalg.norm(np.subtract(k.position, fx.position)) < FIXTURE_DUPLICATE_M for k in kept):
            continue  # the same fixture seen twice
        host = eid_of_plane.get(f"layout-wall-{fx.wall_index}") if fx.wall_index is not None else None
        if fx.wall_index is not None and host is None:
            refused.append({"kind": fx.kind, "reason": "its wall was not promoted"})
            continue
        kept.append(fx)
        x, y, z = _flip(fx.position)
        half = FIXTURE_SIZE_M[fx.kind] / 2
        eid = f"{pre}-fixture-{len(kept):02d}"
        geom = Geometry(id=f"geom-{eid}", type=GeometryType.BOX, vertex_count=8,
                        bounds_min=Vector3(x - half, y - half, z - half), bounds_max=Vector3(x + half, y + half, z + half),
                        provenance=Provenance.ESTIMATED, confidence=cap, observations=[obs])
        world.geometries[geom.id] = geom
        props = {"entity_class": FIXTURE_CLASS[fx.kind], "host_surface": "wall" if host else "ceiling",
                 "dimensions": {"height_above_floor_m": _dim(world, fx.h)},
                 "bootstrap": {"role": "fixture", "source": "single_image_layout", "detector_score": round(fx.score, 3)}}
        if host:
            props["host_wall_id"] = host
        world.entities[eid] = Entity(
            id=eid, type=EntityType.INFRASTRUCTURE, name=f"{fx.kind.replace('_', ' ').capitalize()} ({props['host_surface']})",
            transform={"position": {"x": x, "y": y, "z": z}}, geometry_ids=[geom.id], provenance=Provenance.INFERRED,
            confidence=cap, uncertainty=Uncertainty(confidence=cap, note=f"{fx.kind} detected in the photograph and placed where "
                                                    f"its ray meets the solved {props['host_surface']}, {fx.h:.2f} m above the floor"),
            observations=[obs], custom_properties=props)
    return {"placed": len(kept), "refused": refused}


#: monocular metric depth is typically within 10-20 % indoors; the stated +/- is 15 % of the value, labelled as such
DIM_UNCERTAINTY_FRACTION = 0.15
DIM_BASIS = "monocular metric depth + surface labels (indoor model, typical error 10-20 %); not a measurement"


def _dim(world, value: float, *, kind: str = "distance", basis: str = DIM_BASIS) -> dict:
    from world_ir.measure import dimension_from_metadata

    return dimension_from_metadata(float(value), world.metadata, state="ESTIMATED", basis=basis,
                                   uncertainty_m=DIM_UNCERTAINTY_FRACTION * abs(float(value)), kind=kind)


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


RING_CORNER_TOL_M = 1.0  # a wall's observed extent may stop this far short of (or overshoot) the corner it meets its neighbour at


def _seg_dist(p, a, b) -> float:
    a, b, p = np.asarray(a, float), np.asarray(b, float), np.asarray(p, float)
    ab = b - a
    t = float(np.clip((p - a) @ ab / max(1e-12, float(ab @ ab)), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def _segments_cross(p1, p2, p3, p4) -> bool:
    def o(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    return o(p1, p2, p3) * o(p1, p2, p4) < 0 and o(p3, p4, p1) * o(p3, p4, p2) < 0


def _wall_ring(segs) -> Optional[List[Tuple[float, float]]]:
    """Closed outline from the walls themselves: corners are where neighbouring wall lines meet (L-shapes, notches).

    ``segs`` = [(a, b)] observed wall segments in the plan, camera at the origin. Walls are ordered by their angle around the
    camera. The outline is accepted only when every neighbouring pair meets at a corner near both segments, the polygon does not
    cross itself and the camera is inside it; otherwise None (the caller falls back to the visible-part hull).
    """
    if len(segs) < 3:
        return None
    order = sorted(range(len(segs)), key=lambda i: math.atan2(segs[i][0][1] + segs[i][1][1], segs[i][0][0] + segs[i][1][0]))
    corners: List[Tuple[float, float]] = []
    for k, i in enumerate(order):
        j = order[(k + 1) % len(order)]
        (a1, b1), (a2, b2) = segs[i], segs[j]
        d1, d2 = np.subtract(b1, a1), np.subtract(b2, a2)
        den = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(den) < 0.17 * np.linalg.norm(d1) * np.linalg.norm(d2):  # near-parallel neighbours do not form a corner
            return None
        s = ((a2[0] - a1[0]) * d2[1] - (a2[1] - a1[1]) * d2[0]) / den
        p = (float(a1[0] + s * d1[0]), float(a1[1] + s * d1[1]))
        if _seg_dist(p, a1, b1) > RING_CORNER_TOL_M or _seg_dist(p, a2, b2) > RING_CORNER_TOL_M:
            return None
        corners.append(p)
    n = len(corners)
    area2 = sum(corners[k][0] * corners[(k + 1) % n][1] - corners[(k + 1) % n][0] * corners[k][1] for k in range(n))
    if abs(area2) < 1e-6:
        return None
    for k in range(n):
        for m in range(k + 2, n):
            if k == 0 and m == n - 1:
                continue
            if _segments_cross(corners[k], corners[(k + 1) % n], corners[m], corners[(m + 1) % n]):
                return None
    inside = False
    for k in range(n):
        (x1, y1), (x2, y2) = corners[k], corners[(k + 1) % n]
        if (y1 > 0) != (y2 > 0) and 0 < x1 + (0 - y1) / (y2 - y1) * (x2 - x1):
            inside = not inside
    return corners if inside else None


def add_layout_room(world, layout: SurfaceLayout, eid_of_plane: Dict[str, str], *, pre: str,
                    confidence_ceiling: float, opening_ids: Optional[Dict[int, str]] = None) -> Dict[str, object]:
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
    ring = _wall_ring([((float(a @ e1), float(a @ e2)), (float(b @ e1), float(b @ e2))) for _, _, a, b, _ in walls_used])
    poly = ring or _hull(seg_pts)
    if len(poly) < 3:
        facts["reason"] = "walls and camera are collinear: no footprint"
        return facts
    area = 0.5 * abs(sum(poly[k][0] * poly[(k + 1) % len(poly)][1] - poly[(k + 1) % len(poly)][0] * poly[k][1] for k in range(len(poly))))
    room_id = f"{pre}-room-01"
    # a space needs bounds to be exported as IfcSpace: the footprint's corners at the floor and at the ceiling height
    up_h = np.asarray(up)
    floor_pts = [px * e1 + py * e2 - cam_h * up_h for px, py in poly]
    corners = floor_pts + [q + (height or 2.4) * up_h for q in floor_pts]
    from world_ir import Geometry, GeometryType, Vector3

    lo = [min(float(c[i]) for c in corners) for i in range(3)]
    hi = [max(float(c[i]) for c in corners) for i in range(3)]
    geom = Geometry(id=f"geom-{room_id}", type=GeometryType.BOX, vertex_count=8, bounds_min=Vector3(*lo), bounds_max=Vector3(*hi),
                    provenance=Prov.ESTIMATED, confidence=cap)
    world.geometries[geom.id] = geom
    world.entities[room_id] = Entity(
        id=room_id, type=EntityType.ROOM, name="Room (visible part)",
        geometry_ids=[geom.id], provenance=Prov.INFERRED, confidence=cap,
        uncertainty=Uncertainty(confidence=cap, note="one photograph: the walls that were seen plus the camera position; sides and rear are not claimed"),
        custom_properties={
            "footprint_polygon_m": [[round(px, 4), round(py, 4)] for px, py in poly],
            "footprint_basis": {"e1": [float(c) for c in e1], "e2": [float(c) for c in e2]},
            "footprint_closure": ("walls meet at their corners: closed outline of the observed walls (non-convex rooms keep their notches)"
                                  if ring else "inferred: convex hull of the observed wall segments and the camera position"),
            "floor_area_m2": round(area, 3), "height_m": round(height, 3) if height else None,
            # one photo sees one ceiling plane: min == max (the summary card and snapping read these keys)
            **({"ceiling_height_min_m": round(height, 3), "ceiling_height_max_m": round(height, 3)} if height else {}),
            "dimensions": {
                "floor_area_m2": _dim(world, area, kind="area", basis=DIM_BASIS + "; visible part only (lower bound)"),
                **({"height_m": _dim(world, height)} if height else {}),
                "layout_dimension_u_m": _dim(world, max(float(q @ e1) for q in floor_pts) - min(float(q @ e1) for q in floor_pts)),
                "layout_dimension_v_m": _dim(world, max(float(q @ e2) for q in floor_pts) - min(float(q @ e2) for q in floor_pts)),
            },
            "visible_part_only": ring is None,
        },
    )
    for i, eid, a, b, wl in walls_used:
        ent = world.entities[eid]
        openings = []
        for oi, o in enumerate(layout.openings):
            if o.wall_index != i or oi not in (opening_ids or {}):
                continue
            openings.append({"entity_id": (opening_ids or {})[oi], "kind": o.kind, "offset_m": round(max(0.0, o.s_min - wl.s_min), 3),
                             "width_m": round(o.s_max - o.s_min, 3), "height_m": round(o.h_max - o.h_min, 3),
                             "sill_m": round(o.h_min, 3)})
        ent.custom_properties.update({
            "room_id": room_id, "length_m": round(wl.s_max - wl.s_min, 3), "height_m": round(height, 3),
            "dimensions": {"length_m": _dim(world, wl.s_max - wl.s_min), **({"height_m": _dim(world, height)} if height else {})},
            "thickness_m": 0.1, "thickness_basis": "default_unmeasured", "observed_fraction": 1.0,
            # start/end ON THE FLOOR (the solver's wall line is at camera height), with the wall's direction and up
            "wall_slab": {"start_xyz_m": [float(c) for c in (a - cam_h * up_h)], "end_xyz_m": [float(c) for c in (b - cam_h * up_h)],
                          "direction": [float(c) for c in (b - a) / max(1e-9, float(np.linalg.norm(b - a)))],
                          # the wall normal faces the camera (the room's inside); outward is the other way
                          "outward": [float(c) for c in -np.asarray(_flip(wl.plane.normal))],
                          "up": [float(c) for c in up_h], "height_m": round(height, 3),
                          "length_m": round(float(np.linalg.norm(b - a)), 3), "thickness_m": 0.1, "openings": openings},
        })
    # floor and ceiling belong to the room too (the validator requires every surface to name its space)
    boundary = [eid for _, eid, _, _, _ in walls_used]
    for key in ("layout-floor", "layout-ceiling"):
        eid = eid_of_plane.get(key)
        if eid in world.entities:
            world.entities[eid].custom_properties["room_id"] = room_id
            boundary.append(eid)
    world.entities[room_id].custom_properties["boundary_element_ids"] = boundary
    # floor / ceiling polygons for Studio's solid shell (same hull as the footprint, at floor level and at ceiling height)
    for key, lift in (("layout-floor", 0.0), ("layout-ceiling", height or 2.4)):
        eid = eid_of_plane.get(key)
        if eid in world.entities:
            world.entities[eid].custom_properties["slab"] = {
                "up": [float(c) for c in up_h],
                "polygon_xyz_m": [[float(c) for c in (q + lift * up_h)] for q in floor_pts],
            }
    facts.update({"created": True, "room_id": room_id, "area_m2_visible_part": round(area, 2), "walls": len(walls_used)})
    return facts


#: a known-length reference implying a factor outside this range means the wrong points were picked, not a real correction
SCALE_FACTOR_RANGE = (0.2, 5.0)


def scale_anchor_from_references(refs):
    """The depth scale factor a user's known length(s) imply, as ``({factor, basis, distance_m, model_distance} | None, notes)``.

    The layout's depth is already metric-ish (monocular, typically 10-20 % off); a real length the user gives (distance_m, picked
    between two points whose model length is model_distance) corrects the whole thing by ``distance_m / model_distance``. Several
    lengths are median-gated by ``consolidate_references`` (they must agree); an absurd factor is refused with a note.
    """
    from reconstruction.scale import KNOWN_LENGTH, consolidate_references

    known = [r for r in refs if getattr(r, "method", None) == KNOWN_LENGTH and getattr(r, "model_distance", None)]
    if not known:
        return None, []
    chosen, notes = consolidate_references(known)
    chosen = [r for r in chosen if r.method == KNOWN_LENGTH]
    if not chosen:
        return None, list(notes)
    r = chosen[0]
    factor = r.distance_m / r.model_distance
    if not (SCALE_FACTOR_RANGE[0] <= factor <= SCALE_FACTOR_RANGE[1]):
        return None, list(notes) + [f"a scale factor of {factor:.2f} is implausible (limit {SCALE_FACTOR_RANGE[0]}-{SCALE_FACTOR_RANGE[1]}): "
                                    "check the two points and the length you entered; no scale was applied"]
    return {"factor": float(factor), "basis": KNOWN_LENGTH, "distance_m": float(r.distance_m),
            "model_distance": float(r.model_distance)}, list(notes)


def apply_scale_anchor(analysis, anchor):
    """A copy of ``analysis`` with its metric depth multiplied by the anchor's factor and the anchor attached."""
    import copy

    out = copy.copy(analysis)
    out.depth_m = np.asarray(analysis.depth_m, dtype=np.float64) * float(anchor["factor"])
    out.scale_anchor = anchor
    return out


def make_layout_analyzer(scale_anchor=None):
    """``(analyze, unload)`` for the progressive ladder, or ``(None, noop)`` when REALITY_LAYOUT_MODELS=0.
    A ``scale_anchor`` (see ``scale_anchor_from_references``) multiplies every photo's depth by the user-given factor.

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
        an = analyze_path(models, Path(path))
        return apply_scale_anchor(an, scale_anchor) if scale_anchor else an

    return analyze, models.unload
