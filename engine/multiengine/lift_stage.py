"""Lift stage: detection boxes + relative-depth artifacts + registered cameras + sparse SfM points -> ``position`` / ``dimensions``
observations that join the detection's entity.

This is the missing link between the evidence matrix's two halves: before it, detections (2D boxes) and depth records (raw ``.npy``)
both lived in the matrix but nothing connected them, so every canonical entity had a class and NO geometry.

Honesty rules (enforced by the components it calls, restated here because this is where they meet):

* relative depth is only turned into lengths by ``align_relative_depth_to_metric`` against the job's own sparse SfM points, and only
  when its held-out and leave-region-out gates pass. The result is SFM GAUGE (``sfm_units``), never metres: nothing here is physically
  metric without an independent anchor, and no anchor is applied here.
* a region that cannot be lifted (no camera, depth not alignable, too few valid pixels, ...) becomes a REJECTED record with the reason
  in the rejected-prediction log. Nothing is defaulted and no geometry is invented.
* boxes only (no mask): the lifted extent can include background pixels inside the box. That is stated in the report limitations.
* positions/dimensions are linked to their detection by the EXACT (source image, box) pair, so the join is deterministic and
  auditable; ``lifted_members`` is the single place that defines it.
"""

from __future__ import annotations

import io
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from engine.multiengine.evidence_matrix import EvidenceMatrix
from engine.multiengine.metric3d import Metric3DObservation, ScaleState, ingest
from engine.multiengine.observation import EngineObservation

LIFT_ENGINE_PREFIX = "lift:"
LIFT_QUANTITIES = ("position", "dimensions")
ARCH = "depth-aligned-box-lifter"
LIMITATIONS = (
    "Positions and dimensions come from a monocular RELATIVE depth map aligned to this job's sparse SfM points: they are in SfM "
    "gauge units (arbitrary global scale), NOT metres, and no physical anchor was applied.",
    "Lifting uses the detection BOX only (no segmentation mask), so a lifted extent can include background pixels inside the box; "
    "dimensions are extents along the world axes and the object's own orientation is not claimed.",
)


def _load_depth(matrix: EvidenceMatrix, rec: Metric3DObservation) -> np.ndarray:
    return np.load(io.BytesIO(matrix.artifacts.get(rec.raw_artifact_ref)), allow_pickle=False)


def _resize(depth: np.ndarray, width: int, height: int) -> np.ndarray:
    if depth.shape == (height, width):
        return depth.astype(np.float32)
    from PIL import Image

    return np.asarray(Image.fromarray(depth.astype(np.float32), mode="F").resize((width, height), Image.BILINEAR), dtype=np.float32)


def _rejected(src: str, region: Sequence[float], engine: str, reason: str, *, ref: str | None, frame: str, quantity: str = "position") -> Metric3DObservation:
    return Metric3DObservation(
        engine_id=engine, architecture_id=ARCH, checkpoint_id="none:geometric-algorithm", source_id=src, quantity=quantity,
        value=(), units="sfm_units", scale_state=ScaleState.SFM_GAUGE, frame=frame, camera_id=src,
        image_region=tuple(float(v) for v in region), raw_artifact_ref=ref, validation_status="rejected", rejection_reason=reason,
    )


def run_lift_stage(
    matrix: EvidenceMatrix,
    *,
    cameras: Mapping[str, Any],
    points3d: Sequence[Sequence[float]],
    run_id: str,
    target_frame: str = "world",
    sources: Sequence[str] | None = None,
    observations: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Lift every non-rejected detection box whose image has a registered camera and an alignable relative depth map.

    ``cameras`` is keyed by the same source path the engines saw; ``points3d`` are the job's sparse SfM points in the cameras' own
    world frame. Returns a report; a per-region failure is recorded as a rejection and a missing prerequisite is reported as
    ``skipped`` with the reason (never silently ignored)."""
    from engine.multiengine.depth_scale_alignment import align_relative_depth_to_metric
    from engine.multiengine.metric_lifting import Metric3DLifter

    report: dict[str, Any] = {
        "status": "ran", "limitations": list(LIMITATIONS), "images": [], "lifted": 0, "rejected": 0, "rejection_reasons": {},
        "scale": "sfm_gauge (COLMAP_SFM_UNITS; not metres)", "target_frame": target_frame,
    }
    pts = np.asarray(points3d, dtype=float).reshape(-1, 3) if len(points3d) else np.zeros((0, 3))
    if len(pts) == 0:
        return {**report, "status": "skipped", "reason": "no sparse SfM points: relative depth cannot be aligned to any scale"}
    if not cameras:
        return {**report, "status": "skipped", "reason": "no registered cameras with intrinsics: nothing can be unprojected"}

    depth_recs: dict[str, list[tuple[str, Metric3DObservation]]] = {}
    for o in matrix.store.query(quantity="depth_relative"):
        if matrix.status(o.observation_id) == "rejected":
            continue
        try:
            rec = Metric3DObservation.from_observation(o)
        except Exception:  # noqa: BLE001 -- a record that does not parse is not usable evidence for lifting
            continue
        if rec.validation_status == "valid" and rec.raw_artifact_ref:
            depth_recs.setdefault(rec.source_id, []).append((o.observation_id, rec))
    if not depth_recs:
        return {**report, "status": "skipped", "reason": "no accepted relative-depth records in the evidence matrix"}

    boxes: dict[str, list[EngineObservation]] = {}
    for o in matrix.store.query(quantity="bbox"):
        if isinstance(o.predicted.get("label"), str) and o.image_region is not None and matrix.status(o.observation_id) != "rejected":
            boxes.setdefault(o.source_id, []).append(o)

    lifter = Metric3DLifter()
    reasons: Counter[str] = Counter()
    pts_list = [tuple(p) for p in pts]
    for src in sorted(set(sources) if sources is not None else set(boxes)):
        cam, bxs = cameras.get(src), boxes.get(src, [])
        img: dict[str, Any] = {"source": src, "boxes": len(bxs), "depth_engines": [], "lifted": 0, "rejected": 0}
        report["images"].append(img)
        if not bxs:
            continue
        if cam is None:
            img["note"] = "no registered camera for this image: its detections stay without geometry"
            continue
        if src not in depth_recs:
            img["note"] = "no relative-depth record for this image"
            continue
        K = cam.intrinsics
        try:
            from PIL import Image

            with Image.open(src) as im:
                iw, ih = im.size
        except Exception as exc:  # noqa: BLE001
            img["note"] = f"original image unreadable: {type(exc).__name__}: {exc}"
            continue
        if abs(K.width / K.height - iw / ih) > 0.01 * (iw / ih):
            img["note"] = (f"camera image size {K.width}x{K.height} and the photograph {iw}x{ih} have different aspect ratios; "
                           "pixel boxes cannot be mapped onto the camera (coordinate-frame mismatch)")
            continue
        sx, sy = K.width / iw, K.height / ih
        for depth_oid, drec in depth_recs[src]:
            engine = LIFT_ENGINE_PREFIX + drec.engine_id
            d_info: dict[str, Any] = {"depth_observation": depth_oid, "depth_engine": drec.engine_id}
            img["depth_engines"].append(d_info)
            try:
                depth = _resize(_load_depth(matrix, drec), K.width, K.height)
            except Exception as exc:  # noqa: BLE001
                d_info["status"] = f"depth artifact unreadable: {type(exc).__name__}: {exc}"
                continue
            own = (observations or {}).get(src)
            if own is not None and len(own["xy"]) >= 4:
                # the image's OWN observed keypoints (pixel <-> 3D point): occlusion-free reference for the depth alignment
                al = align_relative_depth_to_metric(depth, cam, [tuple(p) for p in own["xyz"]], [tuple(p) for p in own["xy"]],
                                                    scale_source="colmap_observed_keypoints")
                d_info["reference"] = f"{len(own['xy'])} observed keypoints of this image"
            else:
                # fallback: every sparse point projected through the camera -- includes points hidden from this view, so the gates
                # usually refuse it on real scenes; kept so a model without observations still gets an honest verdict
                al = align_relative_depth_to_metric(depth, cam, pts_list, scale_source="colmap_sparse_points")
                d_info["reference"] = "all sparse points projected (occlusion-unaware)"
            d_info["alignment"] = {k: v for k, v in al.to_dict().items() if k != "metric_anchor"}
            if not al.is_aligned:
                d_info["status"] = "not_aligned"
                recs = [_rejected(src, b.image_region, engine, f"depth not alignable to the SfM points: {al.rejection_reason}",
                                  ref=drec.raw_artifact_ref, frame=target_frame) for b in bxs]  # each detection records WHY it has no geometry
                res = ingest(matrix, recs, run_id=run_id, model_version="lift-stage/1")
                img["rejected"] += len(res["rejected"])
                reasons[str(al.rejection_reason).split(":")[0]] += len(res["rejected"])
                continue
            d_info["status"] = "aligned"
            for b in bxs:
                box_cam = (b.image_region[0] * sx, b.image_region[1] * sy, b.image_region[2] * sx, b.image_region[3] * sy)
                lifted = lifter.lift_detection(
                    entity_id=b.observation_id, class_name=str(b.predicted["label"]), box_2d=box_cam, depth_map=al.aligned_depth_map,
                    camera=cam, target_frame=target_frame, scale_state="SFM_ALIGNED_UNMETRIC", scale_units="COLMAP_SFM_UNITS",
                    is_metric_depth=False, depth_source=f"{drec.engine_id}+sfm_affine_alignment", calibration_source="colmap",
                    scale_source="colmap_sparse_points", model_id=drec.architecture_id, checkpoint_id=drec.checkpoint_id,
                    source_id=src, axis_aligned=True,
                )
                support = {"detection_observation": b.observation_id, "depth_observation": depth_oid, "box_px": list(b.image_region),
                           "box_on_camera": list(box_cam), "alignment": d_info["alignment"],
                           "lift": None if lifted is None else lifted.to_dict(), "rejection": lifter.last_rejection_reason}
                ref, _ = matrix.artifacts.put(json.dumps(support, sort_keys=True, default=float).encode("utf-8"))
                if lifted is None:
                    recs = [_rejected(src, b.image_region, engine, str(lifter.last_rejection_reason), ref=ref, frame=target_frame)]
                    reasons[str(lifter.last_rejection_reason).split(":")[0]] += 1
                else:
                    common: dict[str, Any] = dict(
                        engine_id=engine, architecture_id=ARCH, checkpoint_id=drec.checkpoint_id or "none:geometric-algorithm",
                        source_id=src, units="sfm_units", scale_state=ScaleState.SFM_GAUGE, frame=target_frame, camera_id=src,
                        image_size=(iw, ih), image_region=tuple(float(v) for v in b.image_region), raw_artifact_ref=ref,
                    )
                    recs = [Metric3DObservation(quantity="position", value=tuple(lifted.position), covariance=lifted.covariance_position, **common),
                            Metric3DObservation(quantity="dimensions", value=tuple(lifted.dimensions), covariance=lifted.covariance_dimensions, **common)]
                res = ingest(matrix, recs, run_id=run_id, model_version="lift-stage/1")
                img["lifted"] += 1 if lifted is not None and res["accepted"] else 0
                img["rejected"] += len(res["rejected"])
                for r in res["rejected"]:
                    reasons["contract: " + r[:80]] += 1
    report["lifted"] = sum(i["lifted"] for i in report["images"])
    report["rejected"] = sum(i["rejected"] for i in report["images"])
    report["rejection_reasons"] = dict(reasons)
    return report


def lifted_members(matrix: EvidenceMatrix, observations: Sequence[EngineObservation]) -> dict[str, tuple[str, ...]]:
    """detection observation id -> ids of the lifted ``position``/``dimensions`` observations of THE SAME box in THE SAME image.
    Only accepted (non-rejected) lift observations count."""
    want = {(o.source_id, tuple(o.image_region)): o.observation_id for o in observations if o.image_region is not None}
    out: dict[str, list[str]] = {}
    for q in LIFT_QUANTITIES:
        for lo in matrix.store.query(quantity=q):
            if not lo.engine_id.startswith(LIFT_ENGINE_PREFIX) or lo.image_region is None or matrix.status(lo.observation_id) == "rejected":
                continue
            oid = want.get((lo.source_id, tuple(lo.image_region)))
            if oid is not None:
                out.setdefault(oid, []).append(lo.observation_id)
    return {k: tuple(sorted(v)) for k, v in out.items()}
