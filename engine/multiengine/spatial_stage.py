"""Production adapter for ``spatial_intelligence``: real detections + registered cameras + aligned depth -> physical entities.

This is the integration, not a wrapper that discards results. It

1. aligns each image's relative depth to the job's geometry (``lift_stage.align_depth_record``: held-out + leave-region-out gates, the
   image's own keypoint observations, SfM-gauge units -- never metres);
2. builds ``DetectionInput`` from the Evidence Matrix's detections and the original photographs;
3. runs ``reconstruct_spatial_scene`` in a Z-up WORKING frame (the package assumes Z up; the job frame is Y up), then maps every result
   back into the job frame;
4. writes the results BACK INTO THE EVIDENCE MATRIX so the existing evidence / identity / WorldIR persistence path consumes them:
   per-detection ``position`` + ``dimensions`` (axis-aligned extents, covariance propagated), a ``material`` attribute claim, the full
   lifted record (orientation, surface normal, planarity, oriented extents) as the raw artifact; the package's multi-view association
   becomes the pair scorer; its spatial relationships become bridge relations.

Honest limits (also returned in the report): the package's thresholds are metre-calibrated while the job frame is SfM gauge; no mask is
available in production yet (boxes only); relation kinds that need a vertical axis rely on the job frame really being Y-up.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from engine.multiengine.association import Candidate
from engine.multiengine.association import PairScore as EvidencePairScore
from engine.multiengine.evidence_matrix import EvidenceContext, EvidenceMatrix
from engine.multiengine.lift_stage import LIFT_ENGINE_PREFIX, align_depth_record
from engine.multiengine.lift_stage import LIMITATIONS as LIFT_LIMITATIONS
from engine.multiengine.metric3d import Metric3DObservation, ScaleState, ingest
from engine.multiengine.observation import ObservationDraft, make_observation

ENGINE_ID = LIFT_ENGINE_PREFIX + "spatial_intelligence"
ARCH = "spatial_intelligence/lift_region_to_spatial_entity"
MODEL_VERSION = "spatial-stage/1"
#: job frame (+Y up) -> package working frame (+Z up): a rotation of +90 degrees about X, (x, y, z) -> (x, -z, y)
R_UP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
RAY_FRACTION = 0.1  # max cross-view ray distance as a fraction of the median scene depth (untuned; gauge-relative, not metres)
LIMITATIONS = (
    LIFT_LIMITATIONS[0],  # gauge units, NOT metres, no physical anchor
    "spatial_intelligence thresholds (ray distance, wall/support distances) were calibrated for metres; the job frame is SfM gauge, "
    "so the ray threshold is scaled to the scene's depth and relation thresholds are heuristic in gauge units.",
    "Detections are boxes only (no instance masks in production yet); lifted extents can include background pixels.",
    "Support / mounting relations rely on the job frame being Y-up after canonicalization.",
)


def _quat_to_R(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    n = np.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _R_to_quat(R: np.ndarray):
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        return (0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s)
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
    q = [0.0] * 4
    q[0] = (R[k, j] - R[j, k]) / s
    q[i + 1] = 0.25 * s
    q[j + 1] = (R[j, i] + R[i, j]) / s
    q[k + 1] = (R[k, i] + R[i, k]) / s
    return tuple(q)


def to_working_camera(cam: Any):
    """The same camera expressed in the Z-up working frame."""
    from engine.math import Quat, Vec3
    from world_ir.camera import CameraExtrinsics, PinholeCamera

    e = cam.extrinsics
    R = R_UP @ _quat_to_R((e.rotation.w, e.rotation.x, e.rotation.y, e.rotation.z))
    p = R_UP @ np.array([e.position.x, e.position.y, e.position.z])
    return PinholeCamera(cam.intrinsics, CameraExtrinsics(Vec3(*p), Quat(*_R_to_quat(R))))


def _to_job(v) -> np.ndarray:
    return R_UP.T @ np.asarray(v, float)


def _axis_aligned_extents_job(points_working: Sequence[Sequence[float]], fallback: Sequence[float]) -> list[float]:
    """5th-95th percentile extent along the JOB axes of the lifted point sample (what the box geometry consumer reads)."""
    if len(points_working) < 8:
        return [float(x) for x in fallback]
    pj = np.asarray(points_working, float) @ R_UP  # row-wise R_UP.T @ p
    return (np.percentile(pj, 95, axis=0) - np.percentile(pj, 5, axis=0)).tolist()


MIN_EXTENT_POINTS = 200  # fewer unprojected points than this and the extent (and its uncertainty) is not estimated from them
N_BOOT = 200


def _detection_points_job(det: Any, depth: np.ndarray, cam_work: Any, *, max_points: int = 2500) -> np.ndarray:
    """The detection's own depth points in the JOB frame: every mask pixel (box pixels when there is no mask) with a valid aligned depth,
    depth-outliers (> 3 robust sigma from the median) dropped, unprojected through the working camera. (n, 3), possibly empty."""
    h, w = depth.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in det.box_2d)
    u0, u1, v0, v1 = max(0, int(x1)), min(w, int(np.ceil(x2))), max(0, int(y1)), min(h, int(np.ceil(y2)))
    if u1 <= u0 or v1 <= v0:
        return np.zeros((0, 3))
    vv, uu = np.mgrid[v0:v1, u0:u1]
    sel = np.ones(vv.shape, bool)
    m = det.mask_2d
    if m is not None and np.asarray(m).shape[:2] == (h, w):
        sel &= np.asarray(m)[v0:v1, u0:u1].astype(bool)
    d = depth[v0:v1, u0:u1]
    sel &= np.isfinite(d) & (d > 0)
    u, v, z = uu[sel], vv[sel], d[sel].astype(float)
    if len(z) == 0:
        return np.zeros((0, 3))
    med = float(np.median(z))
    sig = max(1.4826 * float(np.median(np.abs(z - med))), 1e-9)
    keep = np.abs(z - med) <= 3.0 * sig
    u, v, z = u[keep], v[keep], z[keep]
    if len(z) > max_points:
        idx = np.random.default_rng(0).choice(len(z), max_points, replace=False)
        u, v, z = u[idx], v[idx], z[idx]
    pts = np.array([[p.x, p.y, p.z] for p in (cam_work.unproject(float(a), float(b), float(c)) for a, b, c in zip(u, v, z, strict=True))])
    return pts @ R_UP  # working (Z-up) -> job frame, row-wise R_UP.T @ p


def _extent_covariance(pts_job: np.ndarray, *, seed: int) -> tuple[list[float], np.ndarray] | None:
    """5th-95th percentile extent along the job axes and the covariance of that estimator by bootstrap over the points. This is a LOWER BOUND
    of the dimension uncertainty: it captures sampling spread only, not the aligned-depth systematic error or mask-boundary error."""
    if len(pts_job) < MIN_EXTENT_POINTS:
        return None
    ext = np.percentile(pts_job, 95, axis=0) - np.percentile(pts_job, 5, axis=0)
    rng = np.random.default_rng(seed)
    boots = np.array([np.percentile(s, 95, axis=0) - np.percentile(s, 5, axis=0)
                      for s in (pts_job[rng.integers(0, len(pts_job), len(pts_job))] for _ in range(N_BOOT))])
    cov = np.cov(boots, rowvar=False) + 1e-12 * np.eye(3)  # tiny ridge: a degenerate axis must stay positive semi-definite, not NaN
    return ext.tolist(), cov


def _bump(d: dict[str, int], key: str, n: int = 1) -> None:
    d[key] = d.get(key, 0) + n


def _lift_mask(enc: Any, photo_wh: tuple[int, int], depth_wh: tuple[int, int]) -> np.ndarray | None:
    """Decode a persisted RLE mask (photograph resolution) and resample it (nearest) to the depth map's resolution. A mask that
    does not decode or does not match the photograph is NOT used (the box stands alone); nothing is repaired or guessed."""
    if not isinstance(enc, dict):
        return None
    from PIL import Image

    from engine.multiengine.mask_codec import decode_mask

    try:
        m = decode_mask(enc)
    except Exception:  # noqa: BLE001
        return None
    if (m.shape[1], m.shape[0]) != tuple(photo_wh):
        return None
    if tuple(photo_wh) == tuple(depth_wh):
        return m
    return np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize(depth_wh, Image.NEAREST)) > 0


def run_spatial_stage(
    matrix: EvidenceMatrix,
    *,
    cameras: Mapping[str, Any],
    points3d: Sequence[Sequence[float]],
    run_id: str,
    sources: Sequence[str] | None = None,
    observations: Mapping[str, Mapping[str, Any]] | None = None,
    state: dict[str, Any] | None = None,
    target_frame: str = "world",
) -> dict[str, Any]:
    from PIL import Image

    from engine.multiengine.spatial_intelligence.integration import DetectionInput, reconstruct_spatial_scene

    state = state if state is not None else {}
    report: dict[str, Any] = {"status": "ran", "limitations": list(LIMITATIONS), "images": [], "lifted": 0, "rejected": 0,
                              "rejection_reasons": {}, "scale": "sfm_gauge (COLMAP_SFM_UNITS; not metres)", "target_frame": target_frame,
                              "engine": "spatial_intelligence.reconstruct_spatial_scene"}
    reasons: dict[str, int] = report["rejection_reasons"]
    pts_list = [tuple(p) for p in points3d]
    if not pts_list:
        return {**report, "status": "skipped", "reason": "no sparse SfM points: relative depth cannot be aligned to any scale"}
    if not cameras:
        return {**report, "status": "skipped", "reason": "no registered cameras with intrinsics: nothing can be unprojected"}

    depth_recs: dict[str, list[tuple[str, Metric3DObservation]]] = {}
    for o in matrix.store.query(quantity="depth_relative"):
        if matrix.status(o.observation_id) == "rejected":
            continue
        try:
            rec = Metric3DObservation.from_observation(o)
        except Exception:  # noqa: BLE001
            continue
        if rec.validation_status == "valid" and rec.raw_artifact_ref:
            depth_recs.setdefault(rec.source_id, []).append((o.observation_id, rec))
    if not depth_recs:
        return {**report, "status": "skipped", "reason": "no accepted relative-depth records in the evidence matrix"}

    boxes: dict[str, list] = {}
    for o in matrix.store.query(quantity="bbox"):
        if isinstance(o.predicted.get("label"), str) and o.image_region is not None and matrix.status(o.observation_id) != "rejected":
            boxes.setdefault(o.source_id, []).append(o)

    work_cams: dict[str, Any] = {}
    depth_maps: dict[str, np.ndarray] = {}
    images: dict[str, np.ndarray] = {}
    det_inputs: list[Any] = []
    det_obs: dict[str, Any] = {}
    image_of: dict[str, tuple] = {}
    for src in sorted(set(sources) if sources is not None else set(boxes)):
        cam, bxs = cameras.get(src), boxes.get(src, [])
        img: dict[str, Any] = {"source": src, "boxes": len(bxs), "depth_engines": [], "lifted": 0, "rejected": 0}
        report["images"].append(img)
        if not bxs and (cam is None or src not in depth_recs):
            continue  # an image with depth + a camera but NO detection still joins depth fusion (its depth is evidence of the surface)
        if cam is None or src not in depth_recs:
            img["note"] = "no registered camera for this image" if cam is None else "no relative-depth record for this image"
            continue
        K = cam.intrinsics
        try:
            with Image.open(src) as im:
                rgb = im.convert("RGB")
                iw, ih = rgb.size
        except Exception as exc:  # noqa: BLE001
            img["note"] = f"original image unreadable: {type(exc).__name__}: {exc}"
            continue
        if abs(K.width / K.height - iw / ih) > 0.01 * (iw / ih):
            img["note"] = f"camera size {K.width}x{K.height} and photograph {iw}x{ih} have different aspect ratios (frame mismatch)"
            continue
        best = None  # the depth record whose alignment generalises best on held-out points
        for depth_oid, drec in depth_recs[src]:
            d_info: dict[str, Any] = {"depth_observation": depth_oid, "depth_engine": drec.engine_id}
            img["depth_engines"].append(d_info)
            try:
                al = align_depth_record(matrix, drec, cam, (observations or {}).get(src), pts_list, d_info)
            except Exception as exc:  # noqa: BLE001
                d_info["status"] = f"depth artifact unreadable: {type(exc).__name__}: {exc}"
                continue
            d_info["status"] = "aligned" if al.is_aligned else "not_aligned"
            if al.is_aligned and (best is None or (al.held_out_relative_rmse or 9) < best[0]):
                best = ((al.held_out_relative_rmse or 9), al, drec, d_info)
        if best is None:
            recs = [_rejected(src, b.image_region, "depth not alignable to the SfM points (every depth engine refused)", frame=target_frame) for b in bxs]
            res = ingest(matrix, recs, run_id=run_id, model_version=MODEL_VERSION)
            img["rejected"] += len(res["rejected"])
            _bump(reasons, "depth_not_aligned", len(res["rejected"]))
            continue
        _, al, drec, d_info = best
        img["chosen_depth_engine"] = drec.engine_id
        sx, sy = K.width / iw, K.height / ih
        work_cams[src] = to_working_camera(cam)
        depth_maps[src] = al.aligned_depth_map
        images[src] = np.asarray(rgb.resize((K.width, K.height)))
        image_of[src] = (iw, ih, drec, d_info)
        dh, dw = depth_maps[src].shape[:2]
        for b in bxs:
            box = [b.image_region[0] * sx, b.image_region[1] * sy, b.image_region[2] * sx, b.image_region[3] * sy]
            mask = _lift_mask(b.predicted.get("mask"), (iw, ih), (dw, dh))  # the segmenter's mask at depth resolution, or None (box only)
            report["masks_used"] = report.get("masks_used", 0) + (mask is not None)
            det_inputs.append(DetectionInput(detection_id=b.observation_id, source_image_id=src, class_name=str(b.predicted["label"]),
                                             box_2d=box, confidence=float(b.confidence if b.confidence is not None else 0.5), mask_2d=mask))
            det_obs[b.observation_id] = b
    report["fusion"] = _fuse_depth(matrix, [(s, cameras[s], depth_maps[s], images[s]) for s in depth_maps if s in cameras])
    state["fusion"] = report["fusion"]  # the surface compiler prefers this fused surface over its own point-cloud mesh
    if not det_inputs:
        refused = sum(i["rejected"] for i in report["images"])
        if refused:  # the depth maps WERE examined and refused: that is a result (rejections on record), not a skipped stage
            return {**report, "lifted": 0, "rejected": refused, "reason": "no image's depth map could be aligned to the SfM geometry"}
        return {**report, "status": "skipped", "reason": "no image has both a registered camera and an alignable depth map"}

    z_med = float(np.median([np.median(m[np.isfinite(m) & (m > 0)]) for m in depth_maps.values()]))
    ray = RAY_FRACTION * z_med
    scene = reconstruct_spatial_scene(det_inputs, work_cams, depth_maps, images, scale_state="SFM_ALIGNED_UNMETRIC",
                                      scale_units="COLMAP_SFM_UNITS", is_physically_metric=False, max_ray_distance_m=ray)
    s_state, s_units = scene.metric3d_scale()
    if s_state is not ScaleState.SFM_GAUGE or scene.is_physically_metric:  # the stage never lets gauge become metres
        raise RuntimeError(f"scale contract violated: expected SFM_GAUGE, got {s_state} (physically metric: {scene.is_physically_metric})")

    lifted_ids: set[str] = set()
    for le in scene.lifted_entities:
        b = det_obs[le.entity_id]
        src = b.source_id
        iw, ih, drec, d_info = image_of[src]
        pos_job = _to_job(le.position_m)
        cov_job = R_UP.T @ np.asarray(le.covariance_xyz, float) @ R_UP
        dims_job = _axis_aligned_extents_job(le.point_cloud_sample, np.abs(_to_job(le.dimensions_m)))
        dim_cov, dim_unc = None, {"state": "unknown", "reason": f"fewer than {MIN_EXTENT_POINTS} usable depth points inside the detection, or no depth/camera"}
        # With no direct extent the value below is the lifter's, which floors every axis at 4 cm: an axis at that floor is a floor, not a measurement
        floored = [bool(abs(float(v) - 0.04) < 1e-9) for v in le.dimensions_m]
        det_in = next((d for d in det_inputs if d.detection_id == le.entity_id), None)
        if det_in is not None and src in depth_maps and src in work_cams:
            pj = _detection_points_job(det_in, depth_maps[src], work_cams[src])
            est = _extent_covariance(pj, seed=int(hashlib.sha256(le.entity_id.encode()).hexdigest()[:8], 16))
            if est is not None:
                dims_job, dcov = est[0], est[1]
                dim_cov = tuple(tuple(float(x) for x in r) for r in dcov)
                dim_unc = {"state": "estimated_from_visual_evidence", "method": "bootstrap of the 5-95 percentile extent over the detection's depth points",
                           "n_points": int(len(pj)), "lower_bound": True, "calibrated": False,
                           "includes": ["sampling spread of the extent estimator"],
                           "excludes": ["aligned-depth systematic error", "mask boundary error", "occlusion (unseen extent)"]}
        if dim_cov is None:
            dim_unc["extent_source"] = "lifter_oriented_extents_fallback"
            dim_unc["axes_at_lifter_floor_0.04"] = floored  # taxonomy / lifter floors are priors; flagged per axis, never presented as measured
        support = {"detection_observation": b.observation_id, "depth_observation": d_info["depth_observation"], "box_px": list(b.image_region),
                   "alignment": d_info["alignment"], "working_frame": "Z-up (job frame rotated +90 deg about X)",
                   "lifted_working_frame": le.to_dict(), "oriented_extents_w_d_h_working": list(le.dimensions_m),
                   "axis_aligned_extents_job_frame": dims_job, "dimension_uncertainty": dim_unc,
                   "point_sample": [list(p) for p in le.point_cloud_sample[:64]]}
        ref, _ = matrix.artifacts.put(json.dumps(support, sort_keys=True, default=float).encode("utf-8"))
        common: dict[str, Any] = dict(
            engine_id=ENGINE_ID, architecture_id=ARCH, checkpoint_id="none:geometric-algorithm", source_id=src, units=s_units,
            scale_state=s_state, frame=target_frame, camera_id=src,
            image_size=None if (min(b.image_region[:2]) < 0 or b.image_region[2] > iw or b.image_region[3] > ih) else (iw, ih),
            image_region=tuple(float(v) for v in b.image_region), raw_artifact_ref=ref)
        recs = [Metric3DObservation(quantity="position", value=tuple(float(v) for v in pos_job),
                                    covariance=tuple(tuple(float(x) for x in r) for r in cov_job), **common),
                Metric3DObservation(quantity="dimensions", value=tuple(float(v) for v in dims_job), covariance=dim_cov, **common)]
        res = ingest(matrix, recs, run_id=run_id, model_version=MODEL_VERSION)
        ok = len(res["accepted"]) == 2
        img = next(i for i in report["images"] if i["source"] == src)
        img["lifted"] += 1 if ok else 0
        img["rejected"] += len(res["rejected"])
        for r in res["rejected"]:
            _bump(reasons, "contract: " + r[:80])
        if ok:
            lifted_ids.add(le.entity_id)

    # material / colour / state claims: ONE per physical entity, attached to the member whose attributes the package used
    n_attr = 0
    for ment in scene.merged_entities:
        va = scene.visual_attributes.get(ment.canonical_entity_id)
        first = ment.member_observation_ids[0] if ment.member_observation_ids else None
        if va is None or first is None or first not in det_obs or va.material.value == "unknown":
            continue
        b = det_obs[first]
        raw, _ = matrix.artifacts.put(json.dumps({"detection_observation": first, "attributes": va.to_dict()}, sort_keys=True, default=str).encode("utf-8"))
        try:
            obs = make_observation(
                ObservationDraft(task="attributes", quantity="material", source_id=b.source_id, coordinate_system=f"image:{b.source_id}",
                                 image_region=b.image_region, predicted={"label": va.material.value, "raw_confidence": va.material_confidence,
                                                                          "surface_finish": va.surface_finish.value, "colour": va.dominant_color_name,
                                                                          "operational_state": va.operational_state.value},
                                 confidence=None),
                engine_id="spatial_intelligence.attributes", model_version=MODEL_VERSION, execution_config={"contract": "attributes/1"})
            matrix.add(obs, EvidenceContext(obs.observation_id, run_id=run_id, architecture_id="spatial_intelligence/extract_visual_attributes",
                                            checkpoint_id="none:heuristic", artifact_refs={"raw_output": raw}, validation="attributes_contract_1"))
            n_attr += 1
        except Exception as exc:  # noqa: BLE001 -- an attribute that cannot be recorded is reported, never faked
            _bump(reasons, "attribute: " + str(exc)[:60])

    entity_of = {m: e.canonical_entity_id for e in scene.merged_entities for m in e.member_observation_ids}
    state.update({
        "lifted_ids": lifted_ids, "entity_of": entity_of, "merged": {e.canonical_entity_id: e for e in scene.merged_entities},
        "relationships": list(scene.spatial_relationships), "refusals": dict(scene.refusal_reasons), "ray_threshold": ray,
        "aligned": {src: (depth_maps[src], cameras[src]) for src in depth_maps},  # per image: aligned depth + its JOB-frame camera (for surfaces)
        "aligned_depth_observations": {src: image_of[src][3]["depth_observation"] for src in depth_maps},  # the depth record each map came from
    })
    report.update({
        "lifted": len(lifted_ids), "detections_in": len(det_inputs), "merged_entities": len(scene.merged_entities),
        "multiview_entities": sum(e.is_multiview for e in scene.merged_entities), "relationships": len(scene.spatial_relationships),
        "attribute_claims": n_attr, "association_refusals": dict(scene.refusal_reasons), "max_ray_distance": ray,
        "ray_threshold_basis": f"{RAY_FRACTION} x median aligned scene depth ({z_med:.4g} gauge units)",
        "rejected": sum(i["rejected"] for i in report["images"]),
    })
    return report


def _rejected(src: str, region: Sequence[float], reason: str, *, frame: str) -> Metric3DObservation:
    return Metric3DObservation(
        engine_id=ENGINE_ID, architecture_id=ARCH, checkpoint_id="none:geometric-algorithm", source_id=src, quantity="position", value=(),
        units="sfm_units", scale_state=ScaleState.SFM_GAUGE, frame=frame, camera_id=src, image_region=tuple(float(v) for v in region),
        validation_status="rejected", rejection_reason=reason)


def make_scene_pair_scorer(state: dict[str, Any], base: Callable[[Sequence[Candidate], EvidenceMatrix], Iterable[EvidencePairScore]] | None = None):
    """Association from the package's multi-view ray triangulation. A pair the package judged (both detections lifted) is merged only
    when it placed them in one physical entity; any pair it could not judge is left to ``base`` (the 3D-IoU / epipolar scorer)."""

    def scorer(cands: Sequence[Candidate], matrix: EvidenceMatrix) -> Iterable[EvidencePairScore]:
        lifted, entity_of, merged = state.get("lifted_ids", set()), state.get("entity_of", {}), state.get("merged", {})
        base_by_pair = {tuple(sorted((s.a, s.b))): s for s in (base(cands, matrix) if base is not None else [])}
        out: list[EvidencePairScore] = []
        for i, a in enumerate(cands):
            for b in cands[i + 1:]:
                if a.source_id == b.source_id:
                    continue
                key = tuple(sorted((a.observation_id, b.observation_id)))
                if a.observation_id in lifted and b.observation_id in lifted:
                    ea, eb = entity_of.get(a.observation_id), entity_of.get(b.observation_id)
                    if ea is not None and ea == eb:
                        m = merged[ea]
                        out.append(EvidencePairScore(a.observation_id, b.observation_id, max(1e-3, float(m.confidence)), {
                            "method": "multi-view ray triangulation (spatial_intelligence)", "physical_entity": ea,
                            "view_count": m.details.get("view_count"), "reprojection_residual_px": m.reprojection_residual_px,
                            "ray_triangulation_residual": m.ray_triangulation_residual_m, "max_ray_distance": state.get("ray_threshold"),
                            "limitation": "ray threshold scaled to scene depth (untuned); box-only detections"}))
                    # judged and not merged: no score, so the pair stays separate
                elif key in base_by_pair:
                    out.append(base_by_pair[key])
        return out

    return scorer


def _fuse_depth(matrix: EvidenceMatrix, items: Sequence[tuple[str, Any, np.ndarray, np.ndarray]]) -> dict[str, Any]:
    """Fuse the aligned depth maps of every image that has a camera into one surface (Open3D TSDF) and PERSIST it. Cameras are the job-frame
    ones, so the mesh is already in the frame the entities use. Never raises: a failure is a recorded result, and the surface is stored
    content-addressed only when it exists, so a recorded artifact reference always resolves."""
    from engine.multiengine.depth_fusion import FusionView, fuse_depth_to_surface

    try:
        views = [FusionView(src, np.asarray(img) if np.asarray(img).shape[:2] == d.shape[:2] else _resize_rgb(img, d.shape), d, cam)
                 for src, cam, d, img in items]
        res = fuse_depth_to_surface(views)
    except Exception as exc:  # noqa: BLE001 -- recorded, never fatal to the stage
        return {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    out: dict[str, Any] = {"status": res.status, "reason": res.reason, **res.report, "frame": "job (cameras as registered; +Y up)"}
    if res.mesh_ply is not None:
        uri, sha = matrix.artifacts.put(res.mesh_ply)
        out["mesh_artifact"] = {"uri": uri, "sha256": sha, "bytes": len(res.mesh_ply), "format": "ply", "frame": "job",
                                "scale_state": "SFM_GAUGE", "scale_units": "sfm_units"}
        for key, blob in (("provenance_artifact", res.provenance_json), ("provenance_vertex_table_artifact", res.provenance_npz)):
            if blob is not None:  # per-vertex provenance sidecars: content-addressed beside the mesh, hashes also in report["provenance"]
                p_uri, p_sha = matrix.artifacts.put(blob)
                out[key] = {"uri": p_uri, "sha256": p_sha, "bytes": len(blob)}
    return out


def _resize_rgb(img: Any, shape: tuple[int, int]) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.fromarray(np.asarray(img)).resize((shape[1], shape[0])))


def make_scene_relations_fn(state: dict[str, Any]):
    """Bridge relations from the package's spatial relationships, mapped from detections to the canonical entities that own them."""
    from engine.multiengine.worldir_bridge import SpatialRelation
    from world_ir.schema_v1 import RelationshipKind

    kinds = {"mounted_on": RelationshipKind.MOUNTED_ON, "supported_by": RelationshipKind.SUPPORTED_BY, "supports": RelationshipKind.SUPPORTS,
             "contains": RelationshipKind.CONTAINS, "part_of": RelationshipKind.PART_OF, "adjacent_to": RelationshipKind.ADJACENT_TO,
             "facing": RelationshipKind.ADJACENT_TO}

    # Kept: relations that follow from contact / hosting / composition. Dropped: ``adjacent_to`` and ``facing`` -- the package's thresholds
    # are fixed metre distances (0.8-1.2), which relate almost every pair of objects in a room (11k relations for 91 entities on the
    # real office run); scale-aware adjacency comes from ``spatial_relations.derive_relations`` instead.
    structural = {"mounted_on", "supported_by", "supports", "contains", "part_of"}

    def fn(canonicals, matrix) -> list:
        owner = {m: ce.entity_id for ce in canonicals for m in ce.members}
        parent_of: dict[str, set[str]] = {}  # child -> supporters, to keep vertical support ACYCLIC

        def reaches(src: str, dst: str) -> bool:
            stack, seen_n = [src], set()
            while stack:
                n = stack.pop()
                if n == dst:
                    return True
                if n not in seen_n:
                    seen_n.add(n)
                    stack.extend(parent_of.get(n, ()))
            return False

        out, seen = [], set()
        for rel in sorted(state.get("relationships", []), key=lambda r: (-r.confidence, r.source_id, r.target_id)):
            if rel.relation.value not in structural:
                continue
            a, b = owner.get(rel.source_id), owner.get(rel.target_id)
            if a is None or b is None or a == b:
                continue
            kind = kinds.get(rel.relation.value, RelationshipKind.ATTACHED_TO)
            if (a, kind, b) in seen:
                continue
            if kind in (RelationshipKind.SUPPORTED_BY, RelationshipKind.SUPPORTS):
                child, parent = (a, b) if kind is RelationshipKind.SUPPORTED_BY else (b, a)
                if reaches(parent, child):  # accepting it would create a support cycle: the weaker (later) claim is refused
                    continue
                parent_of.setdefault(child, set()).add(parent)
            seen.add((a, kind, b))
            # the distance is stated in the units the package measured it in (``distance_m`` is None unless they are physical metres)
            units = {"m": "m", "sfm_units": "SfM gauge units", "synthetic_m": "synthetic metres"}.get(rel.distance_units, str(rel.distance_units))
            out.append(SpatialRelation(a, kind, b, f"spatial_intelligence relation '{rel.relation.value}' ({rel.basis}); distance {rel.distance:.3g} "
                                                    f"{units}; thresholds are metre-calibrated heuristics",
                                       (rel.source_id, rel.target_id)))
        return out

    return fn
