"""Room surfaces -> image-projected appearance (plan M2 integration).

``surface_appearance`` measures colour and builds a planar texture atlas for a list of posed polygons, but nothing in the
pipeline built those polygons or loaded the photos. This module is that bridge:

* ``surfaces_from_world`` reads the SOLID surfaces the room compiler produced (wall slabs, floor/ceiling slabs) and
  turns each into a ``PlanarSurface`` with the room interior as the side photos must be taken from;
* ``views_from_photos`` builds ``SurfaceView``s from the job's camera poses and the stored photos. Intrinsics come from
  the photograph's own metadata when the capture wrote it; otherwise the same 60-degree horizontal field of view the
  viewer documents is used, and that assumption is recorded on every appearance.
* ``measure_world_appearance`` runs the measurement, attaches the appearance reference (and the measured colour) to
  each entity, and records a summary on the world.

Never invents appearance: a surface no photo saw gets ``color_state="not_measured"`` and no colour; texels no camera saw
stay transparent. The atlas is stored in the same content-addressed artifact store as meshes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from world_ir import EntityType

from perception.materials.material_hypotheses import attach_material_hypotheses
from perception.materials.surface_appearance import (
    LIGHTING_NOTE, PlanarSurface, SurfaceView, attach_appearance, measure_surface_appearance,
)

#: documented assumption when a photo carries no intrinsics (the viewer's NoGeometryExplanation uses the same one)
ASSUMED_HFOV_DEG = 60.0
INTRINSICS_SOURCE_METADATA = "metadata.intrinsics"
INTRINSICS_SOURCE_ASSUMED = f"assumed {ASSUMED_HFOV_DEG:g} degree horizontal FOV (no intrinsics recorded)"


def _unit(v) -> np.ndarray:
    a = np.asarray(v, float)
    n = float(np.linalg.norm(a))
    return a / n if n > 1e-12 else np.array([0.0, 0.0, 1.0])


def _room_basis(cp: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    b = cp.get("footprint_basis") or {}
    e1 = np.asarray(b.get("e1", [1.0, 0.0, 0.0]), float)
    e2 = np.asarray(b.get("e2", [0.0, 1.0, 0.0]), float)
    u = np.asarray(b.get("up", [0.0, 0.0, 1.0]), float)
    return e1, e2, u


def _interior_point(cp: dict) -> Optional[Tuple[float, float, float]]:
    poly = cp.get("footprint_polygon_xyz_m")
    if not poly:
        return None
    arr = np.asarray(poly, float)
    if len(arr) < 3:
        return None
    _, _, u = _room_basis(cp)
    h = float(cp.get("height_m") or 2.5)
    return tuple(float(v) for v in (arr.mean(axis=0) + u * (h / 2.0)))


def surfaces_from_world(world) -> Tuple[List[PlanarSurface], Dict[str, object], List[str]]:
    """(surfaces, surface_id -> entity, warnings). Solid slabs only; an older world without them yields no surfaces
    (the caller reports that honestly rather than using thin plane patches with no interior side)."""
    surfaces: List[PlanarSurface] = []
    by_id: Dict[str, object] = {}
    warnings: List[str] = []
    rooms = [e for e in world.entities.values() if e.type == EntityType.ROOM]
    interior_by_room: Dict[str, Tuple[float, float, float]] = {}
    for room in rooms:
        ip = _interior_point(room.custom_properties or {})
        if ip is not None:
            interior_by_room[room.id] = ip
    for ent in world.entities.values():
        cp = ent.custom_properties or {}
        if ent.type == EntityType.WALL and cp.get("wall_slab"):
            s = cp["wall_slab"]
            start = np.asarray(s.get("start_xyz_m"), float)
            end = np.asarray(s.get("end_xyz_m"), float)
            up = _unit(s.get("up", [0.0, 0.0, 1.0]))
            h = float(s.get("height_m") or 0.0)
            if len(start) != 3 or h <= 0:
                warnings.append(f"{ent.id}: wall slab has no usable geometry")
                continue
            poly = np.array([start, end, end + up * h, start + up * h], float)
            surfaces.append(PlanarSurface(surface_id=ent.id, polygon=poly, kind="wall",
                                          interior_point=interior_by_room.get(cp.get("room_id")), up=tuple(up)))
            by_id[ent.id] = ent
        elif ent.type in (EntityType.FLOOR, EntityType.CEILING) and "polygon_slab" in ent.semantic_labels:
            s = cp.get("slab") or {}
            poly = s.get("polygon_xyz_m") or cp.get("slab_polygon_xyz_m")
            if not poly or len(poly) < 3:
                warnings.append(f"{ent.id}: slab has no polygon")
                continue
            ip = interior_by_room.get(cp.get("room_id"))
            up = _unit(s.get("up", [0.0, 0.0, 1.0]))
            if ip is None:
                arr = np.asarray(poly, float)
                ip = tuple(float(v) for v in (arr.mean(axis=0) + up * (1.0 if ent.type == EntityType.FLOOR else -1.0)))
            # photos stand above a floor and below a ceiling; pointing at the interior centre is ambiguous exactly on
            # the plane, so move the test point off it
            centre = np.asarray(ip, float)
            offset = up * (1.0 if ent.type == EntityType.FLOOR else -1.0)
            surfaces.append(PlanarSurface(surface_id=ent.id, polygon=np.asarray(poly, float),
                                          kind=ent.type.value, interior_point=tuple(centre + offset), up=tuple(up)))
            by_id[ent.id] = ent
    return surfaces, by_id, warnings


def _intrinsics_for(item, image: np.ndarray):
    from world_ir.camera import CameraIntrinsics

    h, w = image.shape[:2]
    meta = (getattr(item, "metadata", None) or {})
    intr = meta.get("intrinsics") if isinstance(meta.get("intrinsics"), dict) else None
    if intr and all(k in intr for k in ("fx", "fy")):
        return CameraIntrinsics(fx=float(intr["fx"]), fy=float(intr["fy"]),
                                cx=float(intr.get("cx", w / 2.0)), cy=float(intr.get("cy", h / 2.0)),
                                width=w, height=h), INTRINSICS_SOURCE_METADATA
    f = (w / 2.0) / math.tan(math.radians(ASSUMED_HFOV_DEG / 2.0))
    return CameraIntrinsics(fx=f, fy=f, cx=w / 2.0, cy=h / 2.0, width=w, height=h), INTRINSICS_SOURCE_ASSUMED


def _load_rgb(item) -> Optional[np.ndarray]:
    try:
        from urllib.parse import unquote, urlparse
        from urllib.request import url2pathname

        import cv2

        uri = getattr(item, "source_uri", "")
        u = urlparse(uri)
        if u.scheme == "file":
            path = url2pathname(unquote(u.path))
        elif u.scheme and len(u.scheme) > 1:      # s3://, https:// ... not a local file
            return None
        else:                                      # bare path, or a Windows drive letter parsed as a scheme
            path = uri
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        return None if bgr is None else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except Exception:  # noqa: BLE001 -- an unreadable photo is skipped, never guessed
        return None


@dataclass(frozen=True)
class PhotoView:
    view: SurfaceView
    intrinsics_source: str


def views_from_photos(poses: Sequence, items_by_id: Dict[str, object],
                      depth_by_id: Optional[Dict[str, np.ndarray]] = None) -> Tuple[List[SurfaceView], List[str], List[str]]:
    """(views, intrinsics sources, skipped reasons). Poses without a readable photo are skipped -- never guessed."""
    from engine.pipeline.facade import camera_from_pose_for_world as camera_from_pose

    views: List[SurfaceView] = []
    sources: List[str] = []
    skipped: List[str] = []
    for pose in poses:
        eid = pose[0] if isinstance(pose, tuple) else getattr(pose, "evidence_id", None)
        position = pose[1] if isinstance(pose, tuple) else getattr(pose, "position", None)
        rotation = pose[2] if isinstance(pose, tuple) else getattr(pose, "rotation", None)
        if not eid or position is None or rotation is None:
            continue
        item = items_by_id.get(eid)
        if item is None:
            skipped.append(f"{eid}: no evidence record")
            continue
        image = _load_rgb(item)
        if image is None:
            skipped.append(f"{eid}: photo is not locally readable")
            continue
        intrinsics, source = _intrinsics_for(item, image)
        shim = type("_Pose", (), {"evidence_id": eid, "position": tuple(position), "rotation": tuple(rotation)})()
        views.append(SurfaceView(eid, camera_from_pose(intrinsics, shim), image,
                                 (depth_by_id or {}).get(eid)))
        sources.append(source)
    return views, sources, skipped


def measure_world_appearance(world, views: Sequence[SurfaceView], *, artifact_store=None,
                             texel_m: float = 0.05, max_texels: int = 256,
                             intrinsics_sources: Sequence[str] = (), skipped: Sequence[str] = ()) -> dict:
    """Measure + attach appearance for every solid room surface. Mutates the world; returns the summary."""
    surfaces, by_id, warnings = surfaces_from_world(world)
    summary: dict = {"available": False, "surfaces": len(surfaces), "measured": 0, "textured": 0,
                     "lighting": LIGHTING_NOTE, "warnings": list(warnings) + list(skipped),
                     "intrinsics": sorted(set(intrinsics_sources))}
    if not surfaces:
        summary["reason"] = "no solid room surface: a room layout with wall/floor/ceiling slabs is required"
        world.metadata["surface_appearance"] = summary
        return summary
    if not views:
        summary["reason"] = "no posed photo could be loaded"
        world.metadata["surface_appearance"] = summary
        return summary
    report = measure_surface_appearance(surfaces, views, artifact_store=artifact_store,
                                        texel_m=texel_m, max_texels=max_texels)
    for sid, appearance in report.appearances.items():
        ent = by_id.get(sid)
        if ent is None:
            continue
        attach_appearance(ent, appearance)
        ent.custom_properties["color_state"] = appearance.color_state
        if appearance.color_rgb is not None and appearance.color_state != "not_measured":
            ent.custom_properties["mean_color_rgb"] = list(appearance.color_rgb)
            ent.custom_properties["color_source"] = "image_projected"
            attach_material_hypotheses(ent)          # M3: ESTIMATED hypotheses beside the colour, never a fact
            summary["measured"] += 1
        if appearance.atlas_uri:
            summary["textured"] += 1
    summary["available"] = True
    summary["views_used"] = len(views)
    summary["notes"] = list(report.notes)
    world.metadata["surface_appearance"] = summary
    return summary
