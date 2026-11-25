"""Depth-mask evidence, DINOv2+geometry cross-view association, and video/temporal
enrichment for Reality Engine's Scene Intelligence Engine.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from perception.detection.normalized import NormalizedDetection
from world_ir.observations import DepthSource

# ---------------------------------------------------------------------------
# 1. Observation-level Depth & Geometric Evidence
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ObservationDepthEvidence:
    observation_id: str
    median_depth: float | None
    depth_dispersion_iqr: float | None
    valid_depth_fraction: float
    boundary_depth_discontinuity: float | None
    foreground_background_boundary: bool
    relative_depth_order: int | None  # 0 = closest in view
    metric_scale_status: str  # "metric_calibrated" | "relative_uncalibrated" | "missing"
    depth_source: str
    depth_model_name: str
    depth_model_version: str
    unreliable_reason: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_observation_depth_evidence(
    observation_id: str,
    depth_map: np.ndarray | None,
    bbox_px: Tuple[float, float, float, float],
    *,
    mask: np.ndarray | None = None,
    depth_source: DepthSource = DepthSource.RELATIVE,
    depth_model_name: str = "depth_anything_v2_small",
    depth_model_version: str = "hf_v1",
    relative_depth_order: int | None = None,
) -> ObservationDepthEvidence:
    """Compute depth statistics for an observation from its mask (preferred) or box."""
    metric_status = (
        "metric_calibrated"
        if depth_source in (DepthSource.METRIC, DepthSource.MVS)
        else "relative_uncalibrated"
    )
    if depth_map is None or depth_map.size == 0:
        return ObservationDepthEvidence(
            observation_id=observation_id,
            median_depth=None,
            depth_dispersion_iqr=None,
            valid_depth_fraction=0.0,
            boundary_depth_discontinuity=None,
            foreground_background_boundary=False,
            relative_depth_order=relative_depth_order,
            metric_scale_status="missing",
            depth_source=depth_source.value,
            depth_model_name=depth_model_name,
            depth_model_version=depth_model_version,
            unreliable_reason="no_depth_map",
        )

    dh, dw = depth_map.shape[:2]
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        if m.shape != (dh, dw):
            m = cv2.resize(m.astype(np.uint8), (dw, dh), interpolation=cv2.INTER_NEAREST).astype(bool)
    else:
        x0 = int(max(0, min(dw - 1, math.floor(bbox_px[0]))))
        y0 = int(max(0, min(dh - 1, math.floor(bbox_px[1]))))
        x1 = int(max(x0 + 1, min(dw, math.ceil(bbox_px[2]))))
        y1 = int(max(y0 + 1, min(dh, math.ceil(bbox_px[3]))))
        m = np.zeros((dh, dw), dtype=bool)
        m[y0:y1, x0:x1] = True

    total_px = int(m.sum())
    if total_px == 0:
        return ObservationDepthEvidence(
            observation_id=observation_id,
            median_depth=None,
            depth_dispersion_iqr=None,
            valid_depth_fraction=0.0,
            boundary_depth_discontinuity=None,
            foreground_background_boundary=False,
            relative_depth_order=relative_depth_order,
            metric_scale_status=metric_status,
            depth_source=depth_source.value,
            depth_model_name=depth_model_name,
            depth_model_version=depth_model_version,
            unreliable_reason="empty_region",
        )

    vals = depth_map[m]
    valid_mask = np.isfinite(vals) & (vals > 0)
    valid_frac = float(valid_mask.sum()) / float(total_px)
    if not valid_mask.any():
        return ObservationDepthEvidence(
            observation_id=observation_id,
            median_depth=None,
            depth_dispersion_iqr=None,
            valid_depth_fraction=round(valid_frac, 4),
            boundary_depth_discontinuity=None,
            foreground_background_boundary=False,
            relative_depth_order=relative_depth_order,
            metric_scale_status="missing",
            depth_source=depth_source.value,
            depth_model_name=depth_model_name,
            depth_model_version=depth_model_version,
            unreliable_reason="no_valid_depth_pixels",
        )

    v = vals[valid_mask].astype(np.float64)
    q25, med, q75 = np.percentile(v, [25, 50, 75])
    iqr = float(q75 - q25)

    # Boundary depth discontinuity: compare mask interior edge vs 5px outer ring
    u8 = m.astype(np.uint8)
    dil = cv2.dilate(u8, np.ones((7, 7), np.uint8), iterations=1) > 0
    ring = dil & (~m)
    ring_vals = depth_map[ring]
    ring_valid = ring_vals[np.isfinite(ring_vals) & (ring_vals > 0)]
    if ring_valid.size >= 8:
        ring_med = float(np.median(ring_valid))
        discontinuity = round(abs(ring_med - float(med)), 4)
        fg_bg = bool(discontinuity >= max(0.15, 0.15 * max(abs(float(med)), 1e-3)))
    else:
        discontinuity = None
        fg_bg = False

    unreliable = None
    if valid_frac < 0.25:
        unreliable = "low_valid_depth_fraction"
    elif med > 0 and iqr / med > 0.65:
        unreliable = "high_depth_dispersion"

    return ObservationDepthEvidence(
        observation_id=observation_id,
        median_depth=round(float(med), 4),
        depth_dispersion_iqr=round(iqr, 4),
        valid_depth_fraction=round(valid_frac, 4),
        boundary_depth_discontinuity=discontinuity,
        foreground_background_boundary=fg_bg,
        relative_depth_order=relative_depth_order,
        metric_scale_status=metric_status,
        depth_source=depth_source.value,
        depth_model_name=depth_model_name,
        depth_model_version=depth_model_version,
        unreliable_reason=unreliable,
    )


# ---------------------------------------------------------------------------
# 2. Cross-View Association (DINOv2 + Appearance + Camera Geometry)
# ---------------------------------------------------------------------------

class DINOv2LocalEmbedder:
    """Local-cache DINOv2 (``facebook/dinov2-small``) crop embedder with unload()."""

    def __init__(self, model_id: str = "facebook/dinov2-small", device: str | None = None) -> None:
        self.model_id = model_id
        self._device = device
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is not None:
            return self._model, self._processor
        import torch
        from transformers import AutoImageProcessor, AutoModel

        dev = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        proc = AutoImageProcessor.from_pretrained(self.model_id, local_files_only=True)
        mod = AutoModel.from_pretrained(self.model_id, local_files_only=True).to(dev).eval()
        self._device, self._model, self._processor = dev, mod, proc
        return mod, proc

    def embed_crops(self, crops_rgb: Sequence[np.ndarray]) -> List[np.ndarray]:
        if not crops_rgb:
            return []
        import torch
        from PIL import Image

        mod, proc = self._load()
        pils = [Image.fromarray(np.asarray(c, dtype=np.uint8)) for c in crops_rgb]
        inputs = proc(images=pils, return_tensors="pt").to(self._device)
        with torch.no_grad():
            out = mod(**inputs)
            cls = out.last_hidden_state[:, 0, :]
            cls = torch.nn.functional.normalize(cls, dim=-1)
        arr = cls.detach().cpu().numpy().astype(np.float32)
        return [arr[i] for i in range(arr.shape[0])]

    def unload(self) -> None:
        self._model = self._processor = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass


def _crop_rgb_hist_embedding(rgb: np.ndarray, bbox_px: Tuple[float, float, float, float]) -> np.ndarray:
    h, w = rgb.shape[:2]
    x0 = int(max(0, min(w - 1, math.floor(bbox_px[0]))))
    y0 = int(max(0, min(h - 1, math.floor(bbox_px[1]))))
    x1 = int(max(x0 + 1, min(w, math.ceil(bbox_px[2]))))
    y1 = int(max(y0 + 1, math.ceil(bbox_px[3]))) if False else int(max(y0 + 1, min(h, math.ceil(bbox_px[3]))))
    crop = rgb[y0:y1, x0:x1]
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [12, 8], [0, 180, 0, 256]).flatten().astype(np.float32)
    norm = float(np.linalg.norm(hist))
    return hist / norm if norm > 1e-8 else hist


@dataclass(frozen=True)
class CrossViewAssociationCandidate:
    obs_a_id: str
    obs_b_id: str
    view_a: str
    view_b: str
    label_a: str
    label_b: str
    appearance_similarity: float
    geometric_support: float | None
    status: str  # "matched" | "ambiguous_repeated_structure" | "contradictory" | "rejected"
    uncertainty: float
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def associate_cross_view_detections(
    detections: Sequence[NormalizedDetection],
    *,
    embeddings_by_id: Mapping[str, np.ndarray] | None = None,
    cameras_by_view: Mapping[str, Any] | None = None,
    rgb_by_view: Mapping[str, np.ndarray] | None = None,
    min_appearance_sim: float = 0.72,
) -> List[CrossViewAssociationCandidate]:
    """Associate detections across views while disambiguating repeated structures.

    Image-space appearance similarity alone is NEVER treated as proof of physical
    identity: without geometric support or distinct layout context, high-similarity
    repeated elements (doors, windows, chairs) remain ``ambiguous_repeated_structure``.
    """
    embs: Dict[str, np.ndarray] = dict(embeddings_by_id or {})
    if rgb_by_view:
        for d in detections:
            if d.detection_id not in embs and d.image_id in rgb_by_view:
                embs[d.detection_id] = _crop_rgb_hist_embedding(rgb_by_view[d.image_id], d.bbox_px)

    results: List[CrossViewAssociationCandidate] = []
    for i in range(len(detections)):
        da = detections[i]
        for j in range(i + 1, len(detections)):
            db = detections[j]
            if da.image_id == db.image_id:
                # Same view: repeated instances cannot be merged into one physical identity
                if da.canonical_label == db.canonical_label:
                    results.append(
                        CrossViewAssociationCandidate(
                            obs_a_id=da.detection_id,
                            obs_b_id=db.detection_id,
                            view_a=da.image_id,
                            view_b=db.image_id,
                            label_a=da.canonical_label,
                            label_b=db.canonical_label,
                            appearance_similarity=1.0,
                            geometric_support=0.0,
                            status="ambiguous_repeated_structure",
                            uncertainty=0.8,
                            reason="same_view_distinct_boxes_repeated_structure",
                        )
                    )
                continue

            ea = embs.get(da.detection_id)
            eb = embs.get(db.detection_id)
            sim = float(np.dot(ea, eb)) if (ea is not None and eb is not None and ea.shape == eb.shape) else 0.0
            sim = round(max(-1.0, min(1.0, sim)), 4)

            if sim < min_appearance_sim:
                continue

            # Check camera epipolar / ray support if cameras are provided
            geom_sup: float | None = None
            if cameras_by_view and da.image_id in cameras_by_view and db.image_id in cameras_by_view:
                ca = cameras_by_view[da.image_id]
                cb = cameras_by_view[db.image_id]
                try:
                    from perception.instances.epipolar import epipolar_residual_px

                    cax = (da.bbox_px[0] + da.bbox_px[2]) / 2.0
                    cay = (da.bbox_px[1] + da.bbox_px[3]) / 2.0
                    cbx = (db.bbox_px[0] + db.bbox_px[2]) / 2.0
                    cby = (db.bbox_px[1] + db.bbox_px[3]) / 2.0
                    dist_px = float(epipolar_residual_px(ca, (cax, cay), cb, (cbx, cby)))
                    geom_sup = round(max(0.0, 1.0 - min(1.0, dist_px / 60.0)), 4)
                except Exception:  # noqa: BLE001
                    geom_sup = None

            if da.canonical_label != db.canonical_label:
                results.append(
                    CrossViewAssociationCandidate(
                        obs_a_id=da.detection_id,
                        obs_b_id=db.detection_id,
                        view_a=da.image_id,
                        view_b=db.image_id,
                        label_a=da.canonical_label,
                        label_b=db.canonical_label,
                        appearance_similarity=sim,
                        geometric_support=geom_sup,
                        status="contradictory",
                        uncertainty=0.85,
                        reason=f"high_appearance_sim_{sim}_but_conflicting_labels_{da.canonical_label}_vs_{db.canonical_label}",
                    )
                )
            elif geom_sup is not None and geom_sup < 0.35:
                results.append(
                    CrossViewAssociationCandidate(
                        obs_a_id=da.detection_id,
                        obs_b_id=db.detection_id,
                        view_a=da.image_id,
                        view_b=db.image_id,
                        label_a=da.canonical_label,
                        label_b=db.canonical_label,
                        appearance_similarity=sim,
                        geometric_support=geom_sup,
                        status="rejected",
                        uncertainty=0.75,
                        reason="epipolar_geometry_disagrees_with_appearance",
                    )
                )
            elif geom_sup is not None and geom_sup >= 0.5:
                results.append(
                    CrossViewAssociationCandidate(
                        obs_a_id=da.detection_id,
                        obs_b_id=db.detection_id,
                        view_a=da.image_id,
                        view_b=db.image_id,
                        label_a=da.canonical_label,
                        label_b=db.canonical_label,
                        appearance_similarity=sim,
                        geometric_support=geom_sup,
                        status="matched",
                        uncertainty=round(max(0.1, 1.0 - 0.5 * (sim + geom_sup)), 4),
                        reason="appearance_and_epipolar_geometry_agree",
                    )
                )
            else:
                # Appearance-only match without geometry: keep as ambiguous for repeated structures
                results.append(
                    CrossViewAssociationCandidate(
                        obs_a_id=da.detection_id,
                        obs_b_id=db.detection_id,
                        view_a=da.image_id,
                        view_b=db.image_id,
                        label_a=da.canonical_label,
                        label_b=db.canonical_label,
                        appearance_similarity=sim,
                        geometric_support=geom_sup,
                        status="ambiguous_repeated_structure",
                        uncertainty=0.6,
                        reason="appearance_only_no_camera_geometry_proof",
                    )
                )
    return results


# ---------------------------------------------------------------------------
# 3. Video Frame Deduplication & Keyframe Selection
# ---------------------------------------------------------------------------

def deduplicate_video_frames(
    frames_rgb: Sequence[np.ndarray],
    timestamps_s: Sequence[float],
    *,
    min_l1_diff: float = 4.0,
) -> Dict[str, Any]:
    """Select non-duplicate keyframes while preserving frame indices and timestamps."""
    if len(frames_rgb) != len(timestamps_s):
        raise ValueError("frames_rgb and timestamps_s must have identical length")
    if not frames_rgb:
        return {"kept_indices": [], "dropped_duplicate_indices": [], "keyframes": []}

    kept_idx: List[int] = [0]
    dropped_idx: List[int] = []
    prev_small = cv2.resize(cv2.cvtColor(frames_rgb[0], cv2.COLOR_RGB2GRAY), (64, 64))

    for idx in range(1, len(frames_rgb)):
        cur_small = cv2.resize(cv2.cvtColor(frames_rgb[idx], cv2.COLOR_RGB2GRAY), (64, 64))
        diff = float(np.mean(np.abs(cur_small.astype(np.float32) - prev_small.astype(np.float32))))
        if diff >= min_l1_diff:
            kept_idx.append(idx)
            prev_small = cur_small
        else:
            dropped_idx.append(idx)

    return {
        "kept_indices": kept_idx,
        "dropped_duplicate_indices": dropped_idx,
        "keyframes": [
            {"frame_index": i, "timestamp_s": float(timestamps_s[i])}
            for i in kept_idx
        ],
    }


# ---------------------------------------------------------------------------
# 4. Camera Bundle Validation (Programme C2)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CameraValidationReport:
    total_cameras: int
    valid_cameras: int
    invalid_cameras: Dict[str, str]
    all_valid: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def validate_camera_bundle(cameras: Mapping[str, Any] | None) -> CameraValidationReport:
    """Validate camera intrinsics, finite extrinsics, and quaternion norms without inventing poses."""
    if not cameras:
        return CameraValidationReport(total_cameras=0, valid_cameras=0, invalid_cameras={}, all_valid=True)

    invalid: Dict[str, str] = {}
    valid = 0
    for cid, cam in cameras.items():
        intr = getattr(cam, "intrinsics", None)
        ext = getattr(cam, "extrinsics", None)
        if intr is None or ext is None:
            invalid[str(cid)] = "missing_intrinsics_or_extrinsics"
            continue
        fx = float(getattr(intr, "fx", 0.0))
        fy = float(getattr(intr, "fy", 0.0))
        w = int(getattr(intr, "width", 0))
        h = int(getattr(intr, "height", 0))
        if not (math.isfinite(fx) and math.isfinite(fy) and fx > 0 and fy > 0 and w > 0 and h > 0):
            invalid[str(cid)] = "invalid_intrinsics"
            continue
        pos = getattr(ext, "position", None)
        rot = getattr(ext, "rotation", None)
        if pos is None or rot is None:
            invalid[str(cid)] = "missing_pose"
            continue
        pvals = [float(pos.x), float(pos.y), float(pos.z)]
        qvals = [float(rot.w), float(rot.x), float(rot.y), float(rot.z)]
        if any(not math.isfinite(v) for v in pvals + qvals):
            invalid[str(cid)] = "non_finite_pose"
            continue
        qnorm = math.sqrt(sum(q * q for q in qvals))
        if abs(qnorm - 1.0) > 0.05:
            invalid[str(cid)] = f"unnormalized_quaternion_{qnorm:.4f}"
            continue
        valid += 1

    return CameraValidationReport(
        total_cameras=len(cameras),
        valid_cameras=valid,
        invalid_cameras=invalid,
        all_valid=(len(invalid) == 0),
    )


# ---------------------------------------------------------------------------
# 5. Association Quality Benchmark & Failure Diagnostics (Programme D3)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AssociationQualityReport:
    precision: float
    recall: float
    false_merges: int
    false_splits: int
    duplicate_rate: float
    ambiguous_rate: float
    failure_diagnostics: Tuple[Dict[str, Any], ...]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["failure_diagnostics"] = list(self.failure_diagnostics)
        return d


def evaluate_association_quality(
    predictions: Sequence[CrossViewAssociationCandidate],
    ground_truth_pairs: Sequence[Tuple[str, str]],
) -> AssociationQualityReport:
    """Measure precision, recall, false merges/splits, ambiguous rate, and failure kinds."""
    from perception.evaluation.failure_detector import FailureKind

    gt = {tuple(sorted((a, b))) for a, b in ground_truth_pairs}
    matched_preds = [p for p in predictions if p.status == "matched"]
    pred_pairs = {tuple(sorted((p.obs_a_id, p.obs_b_id))) for p in matched_preds}

    tp = len(pred_pairs & gt)
    fp = len(pred_pairs - gt)
    fn = len(gt - pred_pairs)

    precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else (1.0 if not gt else 0.0)
    recall = round(tp / (tp + fn), 4) if (tp + fn) > 0 else (1.0 if not pred_pairs else 0.0)
    total_preds = max(1, len(predictions))
    ambiguous = sum(1 for p in predictions if p.status == "ambiguous_repeated_structure")
    duplicates = sum(1 for p in predictions if p.view_a == p.view_b)

    diags: List[Dict[str, Any]] = []
    if fp > 0:
        diags.append({
            "failure_kind": FailureKind.EMBEDDING_SYMMETRY_CONFUSION.value,
            "count": fp,
            "reason": "false_merge_of_distinct_physical_instances",
        })
    if fn > 0:
        diags.append({
            "failure_kind": FailureKind.SFM_LOW_OVERLAP_DISCONNECTION.value,
            "count": fn,
            "reason": "false_split_missing_cross_view_association",
        })

    return AssociationQualityReport(
        precision=precision,
        recall=recall,
        false_merges=fp,
        false_splits=fn,
        duplicate_rate=round(duplicates / total_preds, 4),
        ambiguous_rate=round(ambiguous / total_preds, 4),
        failure_diagnostics=tuple(diags),
    )


# ---------------------------------------------------------------------------
# 6. Documented Observation Quality Scoring Formula (Programme H)
# ---------------------------------------------------------------------------

def compute_observation_quality_score(
    *,
    confidence: float | None,
    mask_validity: str,
    depth_evidence: ObservationDepthEvidence | None,
    image_is_blurry: bool = False,
    image_is_underexposed: bool = False,
    image_is_overexposed: bool = False,
) -> Dict[str, Any]:
    """Documented quality score in [0, 1] keeping confidence & uncertainty separate.

    Formula:
      quality = 0.35 * det_term + 0.25 * mask_term + 0.20 * depth_term + 0.20 * img_term
    """
    det_term = float(confidence) if confidence is not None else 0.35
    mask_term = 1.0 if mask_validity == "valid" else (0.35 if mask_validity == "degenerate_box_fill" else 0.15)
    if depth_evidence is None or depth_evidence.median_depth is None:
        depth_term = 0.10
    else:
        depth_term = min(1.0, max(0.1, depth_evidence.valid_depth_fraction))
        if depth_evidence.unreliable_reason:
            depth_term *= 0.6
    img_term = 1.0
    if image_is_blurry:
        img_term *= 0.5
    if image_is_underexposed or image_is_overexposed:
        img_term *= 0.6

    score = round(0.35 * det_term + 0.25 * mask_term + 0.20 * depth_term + 0.20 * img_term, 4)
    return {
        "quality_score": score,
        "formula": "0.35*det + 0.25*mask + 0.20*depth + 0.20*img",
        "components": {
            "det_term": round(det_term, 4),
            "mask_term": round(mask_term, 4),
            "depth_term": round(depth_term, 4),
            "img_term": round(img_term, 4),
        },
    }


# ---------------------------------------------------------------------------
# 7. Real Video File Decoding & Adaptive Keyframe Sampling (Programme G)
# ---------------------------------------------------------------------------

def process_video_file(
    video_path: str,
    *,
    max_keyframes: int = 24,
    min_l1_diff: float = 4.0,
) -> Dict[str, Any]:
    """Decode a real video file, suppress duplicate frames, and estimate camera motion."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Unable to open video file: {video_path}")

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        if not math.isfinite(fps) or fps <= 0:
            fps = 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        stride = max(1, total_frames // max(1, max_keyframes * 2)) if total_frames > 0 else 1

        sampled_rgb: List[np.ndarray] = []
        sampled_ts: List[float] = []
        sampled_idx: List[int] = []
        frame_i = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if frame_i % stride == 0:
                sampled_rgb.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                sampled_ts.append(round(frame_i / fps, 4))
                sampled_idx.append(frame_i)
            frame_i += 1
    finally:
        cap.release()

    dedup = deduplicate_video_frames(sampled_rgb, sampled_ts, min_l1_diff=min_l1_diff)
    kept_positions = dedup["kept_indices"][:max_keyframes]
    keyframes = []
    prev_gray = None
    for pos in kept_positions:
        rgb = sampled_rgb[pos]
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        motion_px = 0.0
        if prev_gray is not None:
            pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=60, qualityLevel=0.05, minDistance=10)
            if pts is not None and len(pts) >= 4:
                nxt, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None)
                good = (status.flatten() == 1) if status is not None else np.zeros(len(pts), dtype=bool)
                if good.any():
                    disp = np.linalg.norm(nxt[good].reshape(-1, 2) - pts[good].reshape(-1, 2), axis=1)
                    motion_px = round(float(np.median(disp)), 3)
        prev_gray = gray
        keyframes.append({
            "frame_index": sampled_idx[pos],
            "timestamp_s": sampled_ts[pos],
            "source_video_uri": str(video_path),
            "median_optical_flow_px": motion_px,
            "camera_moving": motion_px > 2.5,
            "rgb": rgb,
        })

    return {
        "video_path": str(video_path),
        "fps": fps,
        "decoded_frames": frame_i,
        "sampled_frames": len(sampled_rgb),
        "dropped_duplicate_count": len(dedup["dropped_duplicate_indices"]),
        "keyframes": keyframes,
    }


# ---------------------------------------------------------------------------
# 8. Depth-Based Surface Orientation Hypotheses (Workstream E)
# ---------------------------------------------------------------------------

def estimate_surface_orientation_from_depth(
    depth_map: np.ndarray,
    *,
    mask: Optional[np.ndarray] = None,
    bbox_px: Optional[Tuple[float, float, float, float]] = None,
) -> Dict[str, Any]:
    """Qualitative surface orientation hypothesis from depth gradients (relative basis)."""
    dh, dw = depth_map.shape[:2]
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        if m.shape != (dh, dw):
            m = cv2.resize(m.astype(np.uint8), (dw, dh), interpolation=cv2.INTER_NEAREST).astype(bool)
    elif bbox_px is not None:
        x0 = int(max(0, min(dw - 1, math.floor(bbox_px[0]))))
        y0 = int(max(0, min(dh - 1, math.floor(bbox_px[1]))))
        x1 = int(max(x0 + 1, min(dw, math.ceil(bbox_px[2]))))
        y1 = int(max(y0 + 1, min(dh, math.ceil(bbox_px[3]))))
        m = np.zeros((dh, dw), dtype=bool)
        m[y0:y1, x0:x1] = True
    else:
        m = np.ones((dh, dw), dtype=bool)

    valid = m & np.isfinite(depth_map) & (depth_map > 0)
    if int(valid.sum()) < 25:
        return {"orientation_hypothesis": "unknown", "dz_dx": None, "dz_dy": None, "relative_depth_basis": True}

    dz_dy_map, dz_dx_map = np.gradient(depth_map.astype(np.float32))
    gx = float(np.median(dz_dx_map[valid]))
    gy = float(np.median(dz_dy_map[valid]))
    if abs(gy) > 2.0 * max(abs(gx), 1e-4) and abs(gy) > 0.005:
        hyp = "horizontal_floor_or_ceiling"
    elif abs(gx) > 0.005 or abs(gy) > 0.002:
        hyp = "vertical_or_angled_wall"
    else:
        hyp = "fronto_parallel_plane"
    return {
        "orientation_hypothesis": hyp,
        "dz_dx": round(gx, 5),
        "dz_dy": round(gy, 5),
        "relative_depth_basis": True,
    }


# ---------------------------------------------------------------------------
# 9. Video Track Continuity & Occlusion Fragmentation Recovery (Workstream G)
# ---------------------------------------------------------------------------

def link_video_object_tracks(
    per_frame_detections: Sequence[Sequence[NormalizedDetection]],
    *,
    max_frame_gap: int = 2,
    min_iou: float = 0.25,
) -> Dict[str, Any]:
    """Associate detections across video keyframes, bridging short occlusion gaps."""
    from perception.detection.normalized import _box_iou_xyxy

    active_tracks: List[Dict[str, Any]] = []
    for f_idx, dets in enumerate(per_frame_detections):
        used_tracks: set[int] = set()
        for det in dets:
            best_t, best_iou = None, 0.0
            for t_i, tr in enumerate(active_tracks):
                if t_i in used_tracks:
                    continue
                if tr["label"] != det.canonical_label:
                    continue
                if f_idx - tr["last_frame"] > max_frame_gap + 1 or f_idx == tr["last_frame"]:
                    continue
                v = _box_iou_xyxy(tr["last_box"], det.bbox_px)
                if v >= min_iou and v > best_iou:
                    best_iou, best_t = v, t_i
            if best_t is not None:
                tr = active_tracks[best_t]
                gap = f_idx - tr["last_frame"] - 1
                if gap > 0:
                    tr["occluded_gaps"].append({"from_frame": tr["last_frame"], "to_frame": f_idx, "missed_frames": gap})
                tr["last_frame"] = f_idx
                tr["last_box"] = det.bbox_px
                tr["detection_ids"].append(det.detection_id)
                tr["frame_indices"].append(det.frame_index if det.frame_index is not None else f_idx)
                tr["timestamps_s"].append(det.timestamp_s)
                used_tracks.add(best_t)
            else:
                active_tracks.append({
                    "track_id": f"vtrk_{len(active_tracks):04d}",
                    "label": det.canonical_label,
                    "last_frame": f_idx,
                    "last_box": det.bbox_px,
                    "detection_ids": [det.detection_id],
                    "frame_indices": [det.frame_index if det.frame_index is not None else f_idx],
                    "timestamps_s": [det.timestamp_s],
                    "occluded_gaps": [],
                })

    for tr in active_tracks:
        tr["persistent"] = len(tr["detection_ids"]) >= 2
        tr["recovered_from_occlusion"] = len(tr["occluded_gaps"]) > 0
        tr.pop("last_frame", None)
        tr.pop("last_box", None)

    return {
        "track_count": len(active_tracks),
        "persistent_tracks": sum(1 for t in active_tracks if t["persistent"]),
        "occlusion_recovered_tracks": sum(1 for t in active_tracks if t["recovered_from_occlusion"]),
        "tracks": active_tracks,
    }


# ---------------------------------------------------------------------------
# 10. Local Feature Matcher Comparison (SIFT vs LightGlue DISK/ALIKED)
# ---------------------------------------------------------------------------

def compare_local_feature_matchers(
    image_a_path: str,
    image_b_path: str,
    *,
    max_side: int = 640,
) -> Dict[str, Any]:
    """Compare classical SIFT against LightGlue (DISK & ALIKED) on an image pair."""
    from pathlib import Path
    import time
    import torch
    from reconstruction.matching.lightglue import LightGlueMatcher
    from reconstruction.matching.sift import SiftMatcher

    paths = {"a": Path(image_a_path), "b": Path(image_b_path)}
    out: Dict[str, Any] = {}

    t0 = time.perf_counter()
    r_sift = SiftMatcher(paths, max_side=max_side).match("a", "b")
    out["sift"] = {
        "matches": r_sift.matches,
        "inliers": r_sift.inliers,
        "inlier_ratio": round(float(r_sift.inlier_ratio), 4),
        "time_s": round(time.perf_counter() - t0, 3),
    }

    for feat in ("disk", "aliked"):
        t1 = time.perf_counter()
        matcher = LightGlueMatcher(paths, max_side=max_side, features=feat)
        try:
            r_lg = matcher.match("a", "b")
            out[f"lightglue_{feat}"] = {
                "matches": r_lg.matches,
                "inliers": r_lg.inliers,
                "inlier_ratio": round(float(r_lg.inlier_ratio), 4),
                "time_s": round(time.perf_counter() - t1, 3),
            }
        finally:
            if hasattr(matcher, "unload"):
                matcher.unload()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return out
