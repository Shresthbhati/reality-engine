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


def unload_furniture_model() -> None:
    global _MODEL
    if _MODEL is not None:
        _MODEL.unload()
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


#: items overlapping this much (intersection over the SMALLER volume, so containment counts) are one object seen twice
DUPLICATE_OVERLAP = 0.6


def _box_overlap(a, b) -> float:
    inter = 1.0
    for i in range(3):
        inter *= max(0.0, min(a[1][i], b[1][i]) - max(a[0][i], b[0][i]))
    vol = lambda x: math.prod(max(0.0, x[1][i] - x[0][i]) for i in range(3))  # noqa: E731
    union = vol(a) + vol(b) - inter
    smaller = min(vol(a), vol(b))
    return max(inter / union if union > 0 else 0.0, inter / smaller if smaller > 0 else 0.0)


MASK_BASIS = "footprint and height measured from the segmentation mask + metric depth (visible surfaces only: hidden sides are not seen)"


def _mask_measure(analysis, box, hint, floor_op, f, cx, cy):
    """``(class, footprint dict, mask)`` for one detection from the learned segmentation, or None (the box placement stays)."""
    if analysis is None or box is None or getattr(analysis, "label_map", None) is None:
        return None
    import numpy as np

    from perception.layout.furniture_masks import class_and_mask, footprint_from_mask

    got = class_and_mask(analysis.label_map, analysis.id2label, box, hint)
    if got is None:
        return None
    n = np.asarray(floor_op.normal, float)
    n = n if n[1] >= 0 else -n
    from perception.instances.furniture_from_detections import SIZE_PRIORS

    fp = footprint_from_mask(got[1], np.asarray(analysis.depth_m, float), f, cx, cy, up=(n[0], -n[1], n[2]),
                             camera_height_m=abs(float(floor_op.d)), depth_hint_m=SIZE_PRIORS.get(got[0], (0, 0.9, 0))[1])
    return None if fp is None else (got[0], fp, got[1])


def _measured_colour(rgb, box, mask=None) -> Dict[str, object]:
    """Median colour of the inner half of the detection box in the photograph: the piece's observed colour (not a texture)."""
    if rgb is None or box is None:
        return {}
    import numpy as np

    img = np.asarray(rgb)
    h, w = img.shape[:2]
    if mask is not None and int(mask.sum()) >= 60:
        col = np.median(img[mask][:, :3], axis=0)
        return {"mean_color_rgb": [int(round(float(v))) for v in col],
                "color_basis": "median colour of the piece's segmentation mask in the photograph"}
    x0, y0, x1, y1 = box
    dx, dy = (x1 - x0) / 4.0, (y1 - y0) / 4.0
    a, b, c, d = int(max(0, x0 + dx)), int(max(0, y0 + dy)), int(min(w, x1 - dx)), int(min(h, y1 - dy))
    if c - a < 2 or d - b < 2:
        return {}
    col = np.median(img[b:d, a:c].reshape(-1, img.shape[-1])[:, :3], axis=0)
    return {"mean_color_rgb": [int(round(float(v))) for v in col],
            "color_basis": "median colour of the inner half of the detection box in the photograph"}


def place_furniture(world, obs, item, detections: Sequence, *, floor_op, pts, f: float, cx: float, cy: float,
                    w: int, h: int, pre: str, confidence_ceiling: float,
                    metric: bool = False, rgb=None, analysis=None) -> Dict[str, object]:
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
    s = 1.0 if metric else ASSUMED_CAMERA_HEIGHT_M / cam_h_rel   # metres per unit: 1 in a metric world, else ASSUMED

    # bootstrap frame (X right, Y up, Z forward) -> placer frame (x right, y forward, z up), assumed metres
    xs = [p[0] * s for p in pts]
    ys = [p[2] * s for p in pts]
    room = RoomFrame(floor_z=-cam_h_rel * s, min_xy=(min(xs), min(ys)), max_xy=(max(xs), max(ys)))
    cam = FurnitureCamera(
        view_id=item.id, R=((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0)), t=(0.0, 0.0, 0.0),
        fx=f, fy=f, cx=cx, cy=cy, width=w, height=h,
    )
    # rgb given = the learned-layout path: its depth is metric-style, so heights and widths are trusted for class checks
    res = furniture_from_detections(list(detections), [cam], room, scale_state="metric" if metric else "unknown",
                                    trust_heights=metric or rgb is not None)
    facts["rejected"] = res.rejected
    facts["scale_assumption"] = {
        "assumed_camera_height_m": ASSUMED_CAMERA_HEIGHT_M, "camera_height_relative_units": round(cam_h_rel, 4),
        "meters_per_unit_assumed": round(s, 4),
        "note": "class-prior sizes (metres) are applied through an ASSUMED camera height; not a measurement",
    }
    kept_boxes: list = []
    k = 0
    for e in sorted(res.entities, key=lambda ent: -float(ent["confidence"])):
        cp = e["custom_properties"]
        px, py, pz = cp["position_m"]
        pos = {"x": px / s, "y": pz / s, "z": py / s}
        dims = {key: make_dimension(v["value"] / s, state="ESTIMATED", basis=v["basis"], scale_state="metric" if metric else "unknown")
                for key, v in cp["dimensions"].items()}
        L, W, H = (cp["dimensions"][name]["value"] / s for name in ("length_m", "width_m", "height_m"))
        category, yaw, yaw_basis, flags, mask = cp["category"], cp["yaw_rad"], cp["yaw_basis"], list(e["flags"]), None
        variants = [dict(pos=pos, L=L, W=W, H=H, yaw=yaw, dims=dims, suffix="")]
        meas = _mask_measure(analysis, e.get("box_xyxy"), category, floor_op, f, cx, cy)
        if meas is not None:
            category, fp, mask = meas
            yaw_basis = "measured_from_mask"
            flags.append("footprint_from_mask")
            from perception.instances.furniture_from_detections import SIZE_PRIORS

            kw = dict(state="ESTIMATED", basis=MASK_BASIS, scale_state="metric" if metric else "unknown")
            parts = fp["parts"]
            variants = []
            for pi_, q in enumerate(parts):
                fx, fy, fz = q["floor_point_cam"]
                qh = min(2.6, max(0.2, q["height"]))
                variants.append(dict(
                    pos={"x": fx, "y": -fy, "z": fz},  # camera frame (y down) -> world (y up); the depth map's units
                    L=q["length"], W=q["width"], H=qh, yaw=q["yaw_rad"], suffix=chr(ord("a") + pi_) if len(parts) > 1 else "",
                    dims={"length_m": make_dimension(q["length"], **kw), "width_m": make_dimension(q["width"], **kw),
                          "height_m": make_dimension(qh, **kw)}))
            if len(parts) > 1:
                flags.append(f"l_shaped_split_into_{len(parts)}_parts")
            elif category in SIZE_PRIORS and max(variants[0]["L"], variants[0]["W"]) > 1.8 * SIZE_PRIORS[category][0]:
                flags.append("larger_than_class_prior: bounding rectangle of the visible mask, possibly merged pieces")
        for v in variants:
            pos, L, W, H, yaw, dims = v["pos"], v["L"], v["W"], v["H"], v["yaw"], v["dims"]
            r = 0.5 * math.hypot(L, W)
            box = ((pos["x"] - r, pos["y"], pos["z"] - r), (pos["x"] + r, pos["y"] + H, pos["z"] + r))
            # mirror of the world validator's duplicate rule (same type, bounds IoU > 0.5), with margin: two detectors can
            # place one object twice, and the validator would then fail the whole job (parts of one piece do not overlap)
            if any(_box_overlap(box, kb) > DUPLICATE_OVERLAP for kb in kept_boxes):
                facts["rejected"].append({"label": cp["category"], "view_id": item.id,
                                          "reason": "duplicate: overlaps an already placed item of higher confidence"})
                continue
            kept_boxes.append(box)
            k += 1
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
                id=eid, type=EntityType.FURNITURE, name=e["name"] + (f" ({v['suffix']})" if v["suffix"] else ""),
                transform={"position": pos},
                geometry_ids=[geom.id], semantic_labels=["furniture_candidate"], provenance=Provenance.INFERRED, confidence=conf,
                uncertainty=Uncertainty(
                    confidence=conf,
                    note=("single-view furniture: " + ("segmentation-mask footprint over metric depth; visible surfaces only"
                                                       if meas is not None else
                                                       "COCO detection, floor-contact placement, class-prior size via an "
                                                       f"ASSUMED {ASSUMED_CAMERA_HEIGHT_M:g} m camera height; relative scale, not measured")),
                ),
                observations=[obs],
                custom_properties={
                    "category": category, "entity_class": f"furniture.{category}", "dimensions": dims, "yaw_rad": yaw,
                    "yaw_basis": yaw_basis, "flags": flags,
                    **_measured_colour(rgb, e.get("box_xyxy"), mask),
                    "bootstrap": {"role": "furniture", "source": "single_image", "evidence_id": item.id,
                                  "scale": "relative", "meters_per_unit_assumed": round(s, 4),
                                  **({"part": v["suffix"]} if v["suffix"] else {})},
                },
            )
    facts["placed"] = k
    return facts
