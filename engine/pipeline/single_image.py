"""Single-image bootstrap: the honest LEVEL 0 spatial hypothesis.

One photograph cannot give complete 3D geometry, and this module never
pretends it can. What it does produce, all explicitly marked as
estimated/inferred and carrying the assumptions behind it:

    image -> monocular relative depth (MiDaS) -> visible-surface points
          -> RANSAC planes -> wall/floor/ceiling classification
          -> WorldIR (visible surface + inferred planes + an explicit
             UNKNOWN entity for everything not seen)

What is NOT produced (and why):
  * no hidden/rear/side geometry -- unseen surfaces are recorded as UNKNOWN
  * no metric dimensions -- monocular depth is relative; scale_state=RELATIVE
  * no windows/doors/openings -- no image-space opening detector is wired
    into this path; the result says so under ``not_attempted``
  * no certainty -- entity confidence is capped at CONFIDENCE_CEILING
    regardless of how clean a plane fit looks, because the depth itself is
    a learned prior, not a measurement

Assumptions (all recorded on the world under metadata["bootstrap"]):
  * inverse-depth shift: MiDaS returns affine-invariant inverse depth
    normalised to [0,1]; depth = 1 / (INVERSE_DEPTH_OFFSET + d).
  * camera: EXIF 35mm-equivalent focal length when present, otherwise an
    ASSUMED 60 degree horizontal field of view.
  * gravity: the camera is assumed roughly level (+Y up), so a horizontal
    plane with an upward normal is read as ground.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from engine.pipeline.openings import detect_opening_candidates
from evidence.image_check import ImageFacts
from evidence.session import EvidenceItem
from provenance import Provenance, Uncertainty
from world_ir import (
    Entity,
    EntityType,
    Geometry,
    GeometryType,
    Observation,
    Vector3,
    WorldIR,
)
from world_ir.artifact_store import ArtifactStore
from world_ir.geometry_data import PointCloudData

#: monocular depth is a learned prior; no entity may claim more than this.
CONFIDENCE_CEILING = 0.35
INVERSE_DEPTH_OFFSET = 0.25
ASSUMED_HFOV_DEG = 60.0
#: share of the frame border that counts as "cut off by the frame edge".
EDGE_BAND = 0.04

#: openings are attempted (engine.pipeline.openings) but only as image-space CANDIDATES.
NOT_ATTEMPTED = [
    "object detection and instance segmentation (openings are rectangle candidates, not recognised objects)",
    "metric scale: monocular depth is relative",
]
#: confidence ceiling for an opening candidate: a contrast rectangle is weaker evidence than a plane fit
OPENING_CONFIDENCE_CEILING = 0.2
UNOBSERVED = [
    "rear and side surfaces (out of view)",
    "surfaces hidden behind foreground objects",
    "anything outside the photograph's frame",
    "true depth and dimensions (only relative depth was estimated)",
]


class BootstrapUnavailable(RuntimeError):
    """The depth model / dependencies are unavailable; nothing was estimated."""


@dataclass(frozen=True)
class BootstrapResult:
    world: WorldIR
    #: visible-surface points in RELATIVE units, camera at the origin
    points: Tuple[Tuple[float, float, float], ...]
    camera_poses: Tuple[tuple, ...]
    facts: Dict[str, object] = field(default_factory=dict)


def _intrinsics(width: int, height: int, facts: ImageFacts) -> Tuple[float, str]:
    f35 = (facts.exif or {}).get("focal_length_35mm")
    if f35:
        # 35mm-equivalent focal length is defined against the 36mm frame width
        return float(f35) / 36.0 * width, f"EXIF 35mm-equivalent focal length {f35:g}mm"
    f = (width / 2.0) / math.tan(math.radians(ASSUMED_HFOV_DEG / 2.0))
    return f, f"ASSUMED {ASSUMED_HFOV_DEG:g} degree horizontal field of view (no EXIF focal length)"


def _load_gray(item: EvidenceItem, w: int, h: int):
    """The photograph as uint8 grey at the depth map's size, or None if not locally readable."""
    try:
        from urllib.parse import unquote, urlparse
        from urllib.request import url2pathname

        import cv2

        u = urlparse(item.source_uri)
        if u.scheme not in ("file", ""):
            return None
        path = url2pathname(unquote(u.path)) if u.scheme == "file" else item.source_uri
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return None if img is None else cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    except Exception:  # noqa: BLE001 -- an unreadable photo just means no opening search
        return None


def _estimate_depth(item: EvidenceItem, model: str):
    try:
        from perception.depth.midAS_backend import MiDaSDepthBackend

        maps = MiDaSDepthBackend(model_type=model).estimate_depth([item])
    except Exception as exc:
        raise BootstrapUnavailable(f"monocular depth unavailable: {exc}") from exc
    if not maps:
        raise BootstrapUnavailable("depth backend returned no depth map for the image")
    return maps[0]


def bootstrap_single_image(
    item: EvidenceItem,
    facts: ImageFacts,
    *,
    artifact_store: Optional[ArtifactStore] = None,
    depth_model: str = "DPT_Hybrid",
    seed: int = 42,
    target_points: int = 4000,
    depth_map=None,
    tag: str = "",
) -> BootstrapResult:
    """Build the LEVEL 0 hypothesis for one photograph.

    ``depth_map`` lets tests inject a synthetic DepthMap; production leaves
    it None so the real MiDaS backend runs. Raises BootstrapUnavailable if
    depth cannot be estimated -- the caller decides what an honest
    fallback looks like; nothing is fabricated here.

    ``tag`` namespaces every entity/geometry id ("boot<tag>-...") so several
    single-view hypotheses can live in one world without collisions.
    """
    import numpy as np
    pre = f"boot{tag}"

    from evidence.promote_planes import (
        PlanePromotionError,
        positions_by_plane,
        promote_plane_to_entity,
    )
    from perception.geometry.orientation import classify_planes
    from perception.geometry.planes import detect_planes
    from reconstruction.backend.interface import (
        ReconstructedCameraPose,
        ReconstructedPoint,
        ReconstructionResult,
    )

    dm = depth_map or _estimate_depth(item, depth_model)
    depth = np.asarray(dm.values, dtype=np.float64)
    if depth.ndim != 2 or depth.size == 0 or not np.isfinite(depth).all():
        raise BootstrapUnavailable("depth map is empty or non-finite")
    if float(depth.max() - depth.min()) < 1e-6:
        raise BootstrapUnavailable("depth map is constant: no depth structure to recover")
    h, w = depth.shape

    f, intrinsics_basis = _intrinsics(w, h, facts)
    cx, cy = w / 2.0, h / 2.0
    step = max(1, int(math.sqrt(w * h / float(target_points))))
    pts: List[Tuple[float, float, float]] = []
    pix: List[Tuple[int, int]] = []
    for v in range(step // 2, h, step):
        for u in range(step // 2, w, step):
            z = 1.0 / (INVERSE_DEPTH_OFFSET + float(depth[v, u]))
            pts.append(((u - cx) / f * z, -(v - cy) / f * z, z))
            pix.append((u, v))

    point_unc = Uncertainty(
        confidence=CONFIDENCE_CEILING,
        note="monocular relative depth (learned prior), assumed camera; not a measurement",
    )
    rec_points = [
        ReconstructedPoint(position=p, track_id=f"bs:{i:05d}", source_evidence_ids=[item.id],
                           uncertainty=point_unc)
        for i, p in enumerate(pts)
    ]
    pose = ReconstructedCameraPose(
        evidence_id=item.id, position=(0.0, 0.0, 0.0), rotation=(1.0, 0.0, 0.0, 0.0),
        uncertainty=Uncertainty(confidence=CONFIDENCE_CEILING,
                                note="assumed pose: origin, level, looking +Z"),
    )
    result = ReconstructionResult(points=rec_points, camera_poses=[pose],
                                  registration_status="partial")

    # ---- planes in relative units (tolerance scaled to the depth range) ----
    tol = 0.02 * float(np.median([p[2] for p in pts])) + 0.02
    min_inl = max(40, int(0.02 * len(pts)))
    detection = detect_planes(result, seed=seed, distance_tolerance_m=tol, min_inliers=min_inl)
    oriented = classify_planes(detection.planes, [pose.position], (0.0, 1.0, 0.0))

    obs = Observation(
        id=f"obs-image-{item.id}", sensor_type="camera", data_uri=item.source_uri,
        data_hash=item.sha256 or "",
        metadata={
            "evidence_id": item.id, "width": w, "height": h,
            "evidence_class": facts.evidence_class, "class_basis": facts.class_basis,
            "intrinsics_basis": intrinsics_basis,
        },
        confidence=1.0,  # the pixels are an observation; what is INFERRED from them is capped below
        uncertainty=Uncertainty(confidence=1.0, note="raw pixels as received"),
    )
    world = WorldIR(id="bootstrap-pending", name="Single-image bootstrap",
                    main_branch_id="branch-main-bootstrap")
    world.observations[obs.id] = obs

    plane_pos = positions_by_plane(result, oriented)
    index_of = {f"bs:{i:05d}": i for i in range(len(pts))}
    promoted: List[str] = []
    unpromoted: List[dict] = []
    truncated = {"left": 0.0, "right": 0.0, "top": 0.0, "bottom": 0.0}
    wall_pixels: Dict[str, List[Tuple[int, int]]] = {}
    n = 0
    for op in oriented:
        if op.role == "unknown":
            unpromoted.append({"plane_id": op.plane.plane_id, "note": op.uncertainty.note or "unclassified"})
            continue
        eid = f"{pre}-plane-{n + 1:02d}"
        try:
            prom = promote_plane_to_entity(
                op, result, world, eid, entity_name=f"Inferred {op.role} (single view)",
                artifact_store=artifact_store, plane_positions=plane_pos,
            )
        except PlanePromotionError as exc:
            unpromoted.append({"plane_id": op.plane.plane_id, "note": f"promotion refused: {exc}"})
            continue
        n += 1
        ent, geom = prom.entity, prom.geometry
        cap = min(CONFIDENCE_CEILING, ent.confidence)
        geom.provenance = Provenance.ESTIMATED
        geom.confidence = min(CONFIDENCE_CEILING, geom.confidence)
        ent.provenance = Provenance.INFERRED
        ent.confidence = cap
        ent.uncertainty = Uncertainty(
            confidence=cap,
            note=(f"single-view {op.role}: RANSAC plane over monocular depth; relative scale, "
                  "assumed camera, orientation assumes a level camera"),
        )
        ent.observations.append(obs)
        ent.custom_properties["bootstrap"] = {
            "role": op.role, "scale": "relative", "source": "single_image",
            "inlier_count": op.plane.inlier_count, "normal": [round(c, 4) for c in op.normal],
        }
        # frame-edge truncation, measured from the plane's own supporting pixels
        idxs = [index_of[t] for t in op.plane.inlier_ids]
        if op.role == "wall":
            wall_pixels[eid] = [pix[i] for i in idxs]
        for side, test in (
            ("left", lambda u, v: u < EDGE_BAND * w), ("right", lambda u, v: u > (1 - EDGE_BAND) * w),
            ("top", lambda u, v: v < EDGE_BAND * h), ("bottom", lambda u, v: v > (1 - EDGE_BAND) * h),
        ):
            share = sum(1 for i in idxs if test(*pix[i])) / max(1, len(idxs))
            truncated[side] = max(truncated[side], share)
        promoted.append(eid)

    # ---- opening CANDIDATES on wall planes: image-space rectangles, never recognised objects ----
    openings_facts: Dict[str, object] = {
        "attempted": False, "candidates": 0, "ceiling": OPENING_CONFIDENCE_CEILING,
        "method": "engine.pipeline.openings: contrast rectangles inside wall-plane pixels",
    }
    gray = _load_gray(item, w, h) if wall_pixels else None
    if wall_pixels and gray is None:
        openings_facts["reason"] = "photograph not readable from its source_uri"
    elif not wall_pixels:
        openings_facts["reason"] = "no wall plane was found to search for openings on"
    else:
        openings_facts["attempted"] = True
        k = 0
        for host_eid, wpix in wall_pixels.items():
            for cand in detect_opening_candidates(gray, wpix, step):
                k += 1
                corners = []
                for cu, cv_ in cand.quad_px:
                    iu, iv = min(w - 1, max(0, int(round(cu)))), min(h - 1, max(0, int(round(cv_))))
                    z = 1.0 / (INVERSE_DEPTH_OFFSET + float(depth[iv, iu]))
                    corners.append(((iu - cx) / f * z, -(iv - cy) / f * z, z))
                cxs, cys, czs = zip(*corners)
                oid = f"{pre}-opening-{k:02d}"
                og = Geometry(
                    id=f"geom-{oid}", type=GeometryType.BOX, vertex_count=4,
                    bounds_min=Vector3(min(cxs), min(cys), min(czs)),
                    bounds_max=Vector3(max(cxs), max(cys), max(czs)),
                    provenance=Provenance.ESTIMATED, confidence=OPENING_CONFIDENCE_CEILING, observations=[obs],
                )
                world.geometries[og.id] = og
                world.entities[oid] = Entity(
                    id=oid, type=EntityType.DOOR if cand.kind == "door" else EntityType.WINDOW,
                    name=f"Opening candidate ({cand.kind}-like) on {host_eid}",
                    transform={"position": {"x": sum(cxs) / 4, "y": sum(cys) / 4, "z": sum(czs) / 4}},
                    geometry_ids=[og.id], provenance=Provenance.INFERRED,
                    confidence=OPENING_CONFIDENCE_CEILING,
                    uncertainty=Uncertainty(
                        confidence=OPENING_CONFIDENCE_CEILING,
                        note=(f"image-space rectangle on a single-view wall: contrast {cand.contrast:g}/255, "
                              f"{cand.area_frac:.1%} of the wall; the {cand.kind} label is a shape/position "
                              "heuristic, not recognition"),
                    ),
                    observations=[obs],
                    custom_properties={"bootstrap": {
                        "role": "opening_candidate", "source": "single_image", "host_entity_id": host_eid,
                        "kind_basis": cand.kind_basis, "contrast": cand.contrast,
                        "rectangularity": cand.rectangularity, "area_frac_of_wall": cand.area_frac,
                        "image_quad_px": [[round(a, 1), round(b, 1)] for a, b in cand.quad_px],
                        "extent": "axis-aligned box around 4 depth-estimated corners; relative units",
                    }},
                )
        openings_facts["candidates"] = k

    # ---- corridor direction: only when two distinct parallel walls and a floor were all found ----
    corridor = None
    walls_o = [o for o in oriented if o.role == "wall"]
    floors_o = [o for o in oriented if o.role == "floor"]
    if floors_o and len(walls_o) >= 2:
        med_z = float(np.median([p[2] for p in pts]))
        for i, a in enumerate(walls_o):
            for b in walls_o[i + 1:]:
                if abs(sum(x * y for x, y in zip(a.normal, b.normal))) < 0.85:
                    continue
                ca = np.mean([pts[index_of[t]] for t in a.plane.inlier_ids], axis=0)
                cb = np.mean([pts[index_of[t]] for t in b.plane.inlier_ids], axis=0)
                na = np.asarray(a.normal)
                if abs(float(np.dot(cb - ca, na))) < 0.15 * med_z:
                    continue          # same wall split in two, not two walls
                # the camera (origin) must sit BETWEEN the walls, else this is one
                # building face broken into parallel fragments, not a corridor
                if float(np.dot(ca, na)) * float(np.dot(cb, na)) >= 0:
                    continue
                axis = np.cross(np.asarray(a.normal), np.asarray(floors_o[0].normal))
                norm = float(np.linalg.norm(axis))
                if norm < 1e-6:
                    continue
                axis = axis / norm
                axis = axis if axis[2] >= 0 else -axis
                corridor = {
                    "axis": [round(float(c), 4) for c in axis],
                    "wall_planes": [a.plane.plane_id, b.plane.plane_id],
                    "floor_plane": floors_o[0].plane.plane_id,
                    "basis": ("two distinct near-parallel walls on opposite sides of the camera plus a floor; "
                              "axis = wall-normal x floor-normal"),
                    "provenance": "INFERRED", "confidence": OPENING_CONFIDENCE_CEILING,
                }
                break
            if corridor:
                break

    # ---- the depth-estimated visible surface itself ----
    xs, ys, zs = zip(*pts)
    d_uri, d_hash = "", ""
    if artifact_store is not None:
        d_uri, d_hash = artifact_store.put(PointCloudData.from_positions(pts).to_bytes())
    vgeom = Geometry(
        id=f"geom-{pre}-visible-surface", type=GeometryType.POINTCLOUD, vertex_count=len(pts),
        data_uri=d_uri, data_hash=d_hash,
        bounds_min=Vector3(min(xs), min(ys), min(zs)), bounds_max=Vector3(max(xs), max(ys), max(zs)),
        provenance=Provenance.ESTIMATED, confidence=CONFIDENCE_CEILING, observations=[obs],
    )
    world.geometries[vgeom.id] = vgeom
    world.entities[f"{pre}-visible-surface"] = Entity(
        id=f"{pre}-visible-surface", type=EntityType.UNKNOWN,
        name="Visible surface (estimated from one photograph)",
        transform={"position": {"x": (min(xs) + max(xs)) / 2, "y": (min(ys) + max(ys)) / 2,
                                "z": (min(zs) + max(zs)) / 2}},
        geometry_ids=[vgeom.id], provenance=Provenance.ESTIMATED, confidence=CONFIDENCE_CEILING,
        uncertainty=Uncertainty(confidence=CONFIDENCE_CEILING,
                                note="dense points from monocular depth: relative scale, not measured"),
        observations=[obs],
        custom_properties={"bootstrap": {"role": "visible_surface", "scale": "relative",
                                         "source": "single_image"}},
    )
    # the (assumed) viewpoint: Studio's camera layer reads camera_pose
    # observations off entities; this is the only entity that carries one, so
    # exactly one camera is drawn, and it says it is assumed.
    cam_obs = Observation(
        id=f"obs-camera-{item.id}", sensor_type="camera_pose", data_uri=item.source_uri,
        data_hash=item.sha256 or "",
        metadata={
            "evidence_id": item.id,
            "camera_pose": {"position": [0.0, 0.0, 0.0], "rotation": [1.0, 0.0, 0.0, 0.0]},
            "image_size": [w, h], "assumed": True,
            "note": "assumed pose: origin, level, looking +Z (single view; nothing to register against)",
        },
        confidence=CONFIDENCE_CEILING,
        uncertainty=Uncertainty(confidence=CONFIDENCE_CEILING, note="assumed, not registered"),
    )
    world.entities[f"{pre}-camera"] = Entity(
        id=f"{pre}-camera", type=EntityType.SENSOR, name=f"Camera (assumed pose) for {item.id}",
        transform={"position": {"x": 0.0, "y": 0.0, "z": 0.0}},
        provenance=Provenance.ESTIMATED, confidence=CONFIDENCE_CEILING,
        uncertainty=Uncertainty(confidence=CONFIDENCE_CEILING, note=cam_obs.metadata["note"]),
        observations=[cam_obs],
    )
    # everything not seen is UNKNOWN, on the record
    world.entities[f"{pre}-unobserved"] = Entity(
        id=f"{pre}-unobserved", type=EntityType.UNKNOWN,
        name="Unobserved (not visible in the photograph)",
        provenance=Provenance.UNKNOWN, confidence=0.0,
        uncertainty=Uncertainty(confidence=0.0, note="; ".join(UNOBSERVED)),
        custom_properties={"unobserved": list(UNOBSERVED)},
    )

    world.metadata["scale"] = {
        "state": "relative", "meters_per_unit": None,
        "note": "single-image monocular depth: relative units only; no metric scale",
    }
    world.metadata["bootstrap"] = {
        "level": 0, "source_evidence_id": item.id, "depth_model": depth_model,
        "intrinsics_basis": intrinsics_basis, "inverse_depth_offset": INVERSE_DEPTH_OFFSET,
        "confidence_ceiling": CONFIDENCE_CEILING, "gravity_assumption": "camera roughly level (+Y up)",
        "not_attempted": list(NOT_ATTEMPTED), "unobserved": list(UNOBSERVED),
        "openings": openings_facts, "corridor": corridor,
    }
    world.metadata["reconstruction"] = {
        "backend": "single_image_bootstrap", "registration_status": "partial",
        "cameras_registered": 1, "cameras_input": 1, "points": len(pts),
    }

    out_facts = {
        "level": 0, "points": len(pts), "planes_promoted": promoted, "planes_unpromoted": unpromoted,
        "intrinsics_basis": intrinsics_basis,
        "edge_truncation": {k: round(v, 3) for k, v in truncated.items()},
        "plane_roles": sorted({o.role for o in oriented if o.role != "unknown"}),
        "not_attempted": list(NOT_ATTEMPTED), "unobserved": list(UNOBSERVED),
        "depth_model": depth_model, "grid_step": step,
        "openings": openings_facts, "corridor": corridor,
    }
    return BootstrapResult(
        world=world, points=tuple(pts),
        camera_poses=((item.id, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),), facts=out_facts,
    )


def fuse_single_views(results: List[Tuple[EvidenceItem, BootstrapResult]], *, gap: float = 0.25) -> BootstrapResult:
    """Combine independent single-view hypotheses into ONE world, honestly.

    This is the floor under multi-view reconstruction: when no image pair can
    be registered, every usable photograph still contributes what it shows.
    The hypotheses are NOT spatially related -- nothing measured how the
    cameras sit relative to each other -- so each keeps its own frame. To keep
    them from stacking at the origin they are laid out side by side along X,
    and that layout is recorded as DISPLAY-ONLY (world.metadata["fusion"],
    per-entity custom_properties["fusion"]); it carries no spatial claim.
    Geometry artifacts stay in each image's own frame.
    """
    if not results:
        raise BootstrapUnavailable("no single-view hypotheses to fuse")
    world = WorldIR(id="bootstrap-pending", name="Fused single-view hypotheses",
                    main_branch_id="branch-main-bootstrap")
    points: List[Tuple[float, float, float]] = []
    poses: List[tuple] = []
    layout: Dict[str, List[float]] = {}
    cursor = 0.0
    sources: List[str] = []
    for item, res in results:
        xs = [p[0] for p in res.points]
        width = (max(xs) - min(xs)) if xs else 1.0
        dx = cursor - (min(xs) if xs else 0.0)
        cursor += width * (1.0 + gap)
        layout[item.id] = [round(dx, 4), 0.0, 0.0]
        sources.append(item.id)
        w = res.world
        world.observations.update(w.observations)
        world.geometries.update(w.geometries)
        for eid, ent in w.entities.items():
            pos = (ent.transform or {}).get("position")
            if pos is not None:
                pos["x"] = pos.get("x", 0.0) + dx
            ent.custom_properties["fusion"] = {
                "frame_of": item.id, "display_offset": [round(dx, 4), 0.0, 0.0],
                "note": "display layout only; relative placement between photographs is UNKNOWN",
            }
            world.entities[eid] = ent
        points.extend((x + dx, y, z) for x, y, z in res.points)
        poses.extend((eid, (px + dx, py, pz), rot) for eid, (px, py, pz), rot in res.camera_poses)
    world.metadata["scale"] = {
        "state": "relative", "meters_per_unit": None,
        "note": "independent single-image monocular depth: relative units, separate frames",
    }
    world.metadata["bootstrap"] = {
        "level": 0, "source_evidence_ids": sources, "fused": True,
        "not_attempted": list(NOT_ATTEMPTED), "unobserved": list(UNOBSERVED),
    }
    world.metadata["fusion"] = {
        "mode": "unregistered_single_views", "layout": "display_only", "display_offsets": layout,
        "note": ("no multi-view registration was possible; each photograph is an independent "
                 "single-view hypothesis, laid out side by side for display only"),
    }
    world.metadata["reconstruction"] = {
        "backend": "single_image_bootstrap_fused", "registration_status": "partial",
        "cameras_registered": 0, "cameras_input": len(results), "points": len(points),
    }
    facts = {
        "level": 0, "fused_views": len(results), "points": len(points),
        "per_view": {item.id: res.facts for item, res in results},
        "not_attempted": list(NOT_ATTEMPTED), "unobserved": list(UNOBSERVED),
        "depth_model": results[0][1].facts.get("depth_model"),
    }
    return BootstrapResult(world=world, points=tuple(points), camera_poses=tuple(poses), facts=facts)
