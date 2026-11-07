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
