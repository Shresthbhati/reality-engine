"""Furniture for the single-image path (build-plan task 8.7, scene-complete spec M1).

The single-image bootstrap used to run no object detection, so a room photo produced shell planes and no
furniture. This module detects COCO-class furniture (Mask R-CNN) and places it with the existing
``furniture_from_detections`` floor-contact placer.

Scale is the honest problem: that placer works in metres, the bootstrap frame is RELATIVE. The camera height
above the solved floor plane is known in relative units, so we ASSUME a typical camera height
(``ASSUMED_CAMERA_HEIGHT_M``) to get one metres-per-unit factor, place in that assumed-metric frame, and convert back.
Every entity records the assumption; nothing here is a measurement, and dimensions stay in ``model_units``.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Sequence

#: assumed eye height above the floor, used only to give the placer's metre priors a relative-unit scale.
ASSUMED_CAMERA_HEIGHT_M = 1.5
#: the placer assumes a horizontal floor; refuse when the fitted floor normal is further than this from "up".
MAX_FLOOR_TILT_DEG = 20.0
MIN_DETECTION_SCORE = 0.5

Detector = Callable[[object, int, int], Sequence]

_MODEL = None


def maskrcnn_furniture_detector(item, w: int, h: int) -> list:
    """Mask R-CNN detections for ``item``, boxes rescaled to the ``w`` x ``h`` depth grid.

    Raises ``DetectionBackendUnavailableError`` (never returns an empty list for "could not run")."""
    global _MODEL
    from PIL import Image

    from perception.detection.maskrcnn_backend import MaskRCNNDetector, _uri_to_path
    from perception.instances.furniture_from_detections import FurnitureDetection

    if _MODEL is None:
        _MODEL = MaskRCNNDetector(score_threshold=MIN_DETECTION_SCORE, device="cpu")
    results = _MODEL.detect([item])
    if not results:
        raise RuntimeError("detector could not decode the photograph")
    with Image.open(_uri_to_path(item.source_uri)) as im:
        ow, oh = im.size
    sx, sy = w / float(ow), h / float(oh)
    return [
        FurnitureDetection(
            label=d.label, score=float(d.score),
            box_xyxy=(d.box[0] * sx, d.box[1] * sy, d.box[2] * sx, d.box[3] * sy), view_id=item.id,
        )
        for d in results[0].detections
    ]


def place_furniture(world, obs, item, detections: Sequence, *, floor_op, pts, f: float, cx: float, cy: float,
                    w: int, h: int, pre: str, confidence_ceiling: float) -> Dict[str, object]:
    """Add furniture entities to ``world``; return a facts dict (what ran, what was rejected and why)."""
    from perception.instances.furniture_from_detections import (
        FurnitureCamera, RoomFrame, furniture_from_detections,
    )
    from provenance import Provenance, Uncertainty
    from world_ir import Entity, EntityType, Geometry, GeometryType, Vector3
    from world_ir.measure import make_dimension

    facts: Dict[str, object] = {"attempted": True, "detections": len(detections), "placed": 0,
                                "method": "Mask R-CNN (COCO) + furniture_from_detections floor-contact placer",
                                "rejected": []}
    if floor_op is None:
        facts["reason"] = "no floor plane was found: furniture cannot be placed without a floor"
        return facts
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, floor_op.normal[1]))))
    if tilt > MAX_FLOOR_TILT_DEG:
        facts["reason"] = (f"floor normal is {tilt:.0f} degrees from up (limit {MAX_FLOOR_TILT_DEG:g}): "
                           "the level-camera assumption does not hold")
        return facts
    cam_h_rel = abs(float(floor_op.d))
    if cam_h_rel < 1e-6:
        facts["reason"] = "camera sits on the floor plane: no height to scale from"
        return facts
    s = ASSUMED_CAMERA_HEIGHT_M / cam_h_rel          # metres per relative unit (ASSUMED)

    # bootstrap frame (X right, Y up, Z forward) -> placer frame (x right, y forward, z up), assumed metres
    xs = [p[0] * s for p in pts]
    ys = [p[2] * s for p in pts]
    room = RoomFrame(floor_z=-ASSUMED_CAMERA_HEIGHT_M, min_xy=(min(xs), min(ys)), max_xy=(max(xs), max(ys)))
    cam = FurnitureCamera(
        view_id=item.id, R=((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0)), t=(0.0, 0.0, 0.0),
        fx=f, fy=f, cx=cx, cy=cy, width=w, height=h,
    )
    res = furniture_from_detections(list(detections), [cam], room, scale_state="unknown")
    facts["rejected"] = res.rejected
    facts["scale_assumption"] = {
        "assumed_camera_height_m": ASSUMED_CAMERA_HEIGHT_M, "camera_height_relative_units": round(cam_h_rel, 4),
        "meters_per_unit_assumed": round(s, 4),
        "note": "class-prior sizes (metres) are applied through an ASSUMED camera height; not a measurement",
    }
    for k, e in enumerate(res.entities, start=1):
        cp = e["custom_properties"]
        px, py, pz = cp["position_m"]
        pos = {"x": px / s, "y": pz / s, "z": py / s}
        dims = {key: make_dimension(v["value"] / s, state="ESTIMATED", basis=v["basis"], scale_state="unknown")
                for key, v in cp["dimensions"].items()}
        L, W, H = (cp["dimensions"][name]["value"] / s for name in ("length_m", "width_m", "height_m"))
        r = 0.5 * math.hypot(L, W)
        conf = min(confidence_ceiling, float(e["confidence"]))
        eid = f"{pre}-furniture-{k:02d}"
        geom = Geometry(
            id=f"geom-{eid}", type=GeometryType.BOX, vertex_count=8,
            bounds_min=Vector3(pos["x"] - r, pos["y"], pos["z"] - r),
            bounds_max=Vector3(pos["x"] + r, pos["y"] + H, pos["z"] + r),
            provenance=Provenance.ESTIMATED, confidence=conf, observations=[obs],
        )
        world.geometries[geom.id] = geom
        world.entities[eid] = Entity(
            id=eid, type=EntityType.FURNITURE, name=e["name"], transform={"position": pos},
            geometry_ids=[geom.id], provenance=Provenance.INFERRED, confidence=conf,
            uncertainty=Uncertainty(
                confidence=conf,
                note=("single-view furniture: COCO detection, floor-contact placement, class-prior size via an "
                      f"ASSUMED {ASSUMED_CAMERA_HEIGHT_M:g} m camera height; relative scale, not measured"),
            ),
            observations=[obs],
            custom_properties={
                "category": cp["category"], "dimensions": dims, "yaw_rad": cp["yaw_rad"], "yaw_basis": cp["yaw_basis"],
                "flags": e["flags"],
                "bootstrap": {"role": "furniture", "source": "single_image", "evidence_id": item.id,
                              "scale": "relative", "meters_per_unit_assumed": round(s, 4)},
            },
        )
    facts["placed"] = len(res.entities)
    return facts
