"""Canonical Visual Observation & Perception-to-World Bridge (perception/world_bridge.py).

Connects image, video, detection, segmentation, and depth perception outputs to
Claude's spatial reasoning compiler (world_ir.observations, world_ir.constraints,
world_ir.uncertainty_model, world_ir.completeness, worldstore.store) without
inventing metric geometry or overwriting immutable source evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from provenance import Provenance
from world_ir.completeness import WorldCompletenessReport, assess_world_completeness
from world_ir.constraints import ConstraintViolation, check_architectural_constraints
from world_ir.observations import (
    AssociationDecision,
    AssociationState,
    DepthSource,
    Observation as LedgerObservation,
    Visibility,
    attach_ledger,
    ledger_of,
    sigma_for,
)
from world_ir.schema_v1 import EntityType, Observation as EntityObservation
from world_ir.uncertainty_model import EntityUncertaintyProfile, propagate_entity_uncertainty
from world_ir.world_v1 import WorldIR


ARCHITECTURAL_LABELS = frozenset({
    "wall", "floor", "ceiling", "door", "window", "opening", "column",
    "beam", "stairs", "stair", "railing", "corridor", "room",
})

OBJECT_MATERIAL_PRIORS: Dict[str, str] = {
    "chair": "wood_or_fabric",
    "couch": "fabric_or_leather",
    "dining table": "wood",
    "bed": "fabric_and_wood",
    "tv": "glass_and_polymer",
    "refrigerator": "brushed_metal",
    "sink": "ceramic_or_steel",
    "toilet": "ceramic",
    "door": "wood_or_metal",
    "window": "glass_and_frame",
}


def deterministic_observation_id(parts: Mapping[str, Any]) -> str:
    """Content-addressed SHA-256 observation ID for idempotent reprocessing."""
    canonical = json.dumps(parts, sort_keys=True, default=str)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return f"vobs_{digest}"


@dataclass(frozen=True)
class ImageQualityMetrics:
    width: int
    height: int
    blur_laplacian_var: float
    mean_luminance: float
    clipped_fraction: float
    is_blurry: bool
    is_underexposed: bool
    is_overexposed: bool
    underexposed_fraction: float = 0.0
    overexposed_fraction: float = 0.0
    noise_estimate: float = 0.0
    motion_blur_score: float = 0.0
    is_motion_blurred: bool = False
    is_noisy: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def assess_image_quality(
    image_rgb: np.ndarray,
    *,
    blur_threshold: float = 50.0,
) -> ImageQualityMetrics:
    """Measure resolution, blur, directional motion blur, exposure, clipped pixels, and noise."""
    import cv2

    h, w = int(image_rgb.shape[0]), int(image_rgb.shape[1])
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY) if image_rgb.ndim == 3 else image_rgb
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean_lum = float(np.mean(gray))
    under_frac = float(np.mean(gray <= 25))
    over_frac = float(np.mean(gray >= 235))
    clipped = float(np.mean((gray <= 3) | (gray >= 252)))

    # High-frequency residual MAD noise estimate
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
    resid = np.abs(gray.astype(np.float32) - blurred.astype(np.float32))
    noise_est = float(np.median(resid) * 1.4826)

    # Directional gradient anisotropy for motion blur detection
    gx = float(np.mean(np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))))
    gy = float(np.mean(np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))))
    anisotropy = abs(gx - gy) / max(1e-6, gx + gy)
    is_blurry = lap_var < blur_threshold
    is_motion_blurred = bool(is_blurry and anisotropy > 0.25)

    return ImageQualityMetrics(
        width=w,
        height=h,
        blur_laplacian_var=round(lap_var, 3),
        mean_luminance=round(mean_lum, 2),
        clipped_fraction=round(clipped, 4),
        is_blurry=is_blurry,
        is_underexposed=mean_lum < 35.0 or under_frac > 0.65,
        is_overexposed=mean_lum > 225.0 or over_frac > 0.65,
        underexposed_fraction=round(under_frac, 4),
        overexposed_fraction=round(over_frac, 4),
        noise_estimate=round(noise_est, 3),
        motion_blur_score=round(anisotropy, 4),
        is_motion_blurred=is_motion_blurred,
        is_noisy=noise_est > 12.0,
    )


@dataclass(frozen=True)
class TaskConfidences:
    """Separate per-task confidences; high detection confidence never implies accurate depth/dimensions."""

    detection: Optional[float] = None
    segmentation: Optional[float] = None
    depth: Optional[float] = None
    dimensions: Optional[float] = None
    material: Optional[float] = None

    def to_dict(self) -> Dict[str, Optional[float]]:
        return asdict(self)


@dataclass(frozen=True)
class WallSlabObservation:
    start_xyz_m: Optional[Tuple[float, float, float]]
    end_xyz_m: Optional[Tuple[float, float, float]]
    coordinate_frame: str
    source_camera: Optional[str]
    source_evidence: str
    depth_basis: str  # "metric" | "mvs" | "relative" | "unknown"
    length_m: Optional[float]
    orientation_rad: Optional[float]
    measurement_uncertainty_m: Optional[float]
    visibility_state: str  # "observed" | "partial" | "inferred" | "unresolved"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.start_xyz_m is not None:
            d["start_xyz_m"] = list(self.start_xyz_m)
        if self.end_xyz_m is not None:
            d["end_xyz_m"] = list(self.end_xyz_m)
        return d


@dataclass(frozen=True)
class OpeningObservation:
    opening_class: str
    bbox_px: Optional[Tuple[float, float, float, float]]
    host_wall_candidate: Optional[str]
    relative_location: Optional[Tuple[float, float]]
    estimated_width_m: Optional[float]
    estimated_height_m: Optional[float]
    evidence_refs: Tuple[str, ...]
    uncertainty_m: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VisualObservation:
    """Canonical perception observation record."""

    observation_id: str
    source_evidence_id: str
    source_image_id: str
    source_video_id: Optional[str]
    frame_index: Optional[int]
    timestamp_s: Optional[float]
    camera_id: Optional[str]
    camera_pose: Optional[Mapping[str, Any]]
    detection_category: str  # "architecture" | "opening" | "object" | "material" | "text"
    semantic_label: str
    bbox_px: Optional[Tuple[float, float, float, float]]
    mask_ref: Optional[str]
    depth_source: DepthSource
    world_pos: Optional[Tuple[float, float, float]]
    bounds_min: Optional[Tuple[float, float, float]]
    bounds_max: Optional[Tuple[float, float, float]]
    sigma_m: Optional[float]
    wall_slab: Optional[WallSlabObservation]
    opening: Optional[OpeningObservation]
    material_hypothesis: Optional[str]
    material_verified: bool
    visual_attributes: Mapping[str, Any]
    model_id: str
    model_version: str
    processing_config: Mapping[str, Any]
    confidences: TaskConfidences
    provenance: Provenance
    visibility: Optional[Visibility]
    processing_status: str = "processed"
    entity_hint_id: Optional[str] = None

    def to_ledger_observation(self) -> Optional[LedgerObservation]:
        """Convert to world_ir.observations.Observation when 3D position is known."""
        if self.world_pos is None or self.sigma_m is None:
            return None
        return LedgerObservation(
            observation_id=self.observation_id,
            image_id=self.source_image_id,
            label=self.semantic_label,
            world_pos=self.world_pos,
            sigma_m=self.sigma_m,
            depth_source=self.depth_source,
            model=f"{self.model_id}@{self.model_version}",
            bounds_min=self.bounds_min,
            bounds_max=self.bounds_max,
            confidence=self.confidences.detection if self.confidences.detection is not None else 0.5,
            mask_uri=self.mask_ref,
            bbox_px=self.bbox_px,
            visibility=self.visibility,
            session_id=self.source_video_id,
            created_at=self.timestamp_s,
        )

    def to_entity_observation(self) -> EntityObservation:
        """Convert to world_ir.schema_v1.Observation for entity.observations."""
        meta: Dict[str, Any] = {
            "source_evidence_id": self.source_evidence_id,
            "source_image_id": self.source_image_id,
            "detection_category": self.detection_category,
            "semantic_label": self.semantic_label,
            "depth_source": self.depth_source.value,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "confidences": self.confidences.to_dict(),
            "processing_status": self.processing_status,
        }
        if self.source_video_id is not None:
            meta["source_video_id"] = self.source_video_id
        if self.frame_index is not None:
            meta["frame_index"] = self.frame_index
        if self.bbox_px is not None:
            meta["bbox_px"] = list(self.bbox_px)
        if self.mask_ref is not None:
            meta["mask_ref"] = self.mask_ref
        if self.material_hypothesis is not None:
            meta["material_hypothesis"] = self.material_hypothesis
            meta["material_verified"] = self.material_verified
        if self.wall_slab is not None:
            meta["wall_slab"] = self.wall_slab.to_dict()
        if self.opening is not None:
            meta["opening"] = self.opening.to_dict()
        return EntityObservation(
            id=self.observation_id,
            sensor_type="video_frame" if self.source_video_id else "camera",
            timestamp=self.timestamp_s,
            frame_id=str(self.frame_index) if self.frame_index is not None else self.camera_id,
            data_uri=self.mask_ref or self.source_evidence_id,
            metadata=meta,
            confidence=self.confidences.detection,
        )


def extract_wall_slab_from_points(
    points_xyz: np.ndarray,
    *,
    source_evidence: str,
    source_camera: Optional[str],
    depth_basis: str,
    coordinate_frame: str = "world_metric",
    visibility_state: str = "observed",
) -> WallSlabObservation:
    """Fit a wall segment (start_xyz_m, end_xyz_m) from 3D points ONLY when metric/MVS scale is valid."""
    if depth_basis not in ("metric", "mvs", "calibrated") or points_xyz.shape[0] < 2:
        return WallSlabObservation(
            start_xyz_m=None,
            end_xyz_m=None,
            coordinate_frame=coordinate_frame,
            source_camera=source_camera,
            source_evidence=source_evidence,
            depth_basis=depth_basis,
            length_m=None,
            orientation_rad=None,
            measurement_uncertainty_m=None,
            visibility_state="unresolved" if points_xyz.shape[0] < 2 else visibility_state,
        )
    pts_2d = points_xyz[:, :2]
    mean_xy = pts_2d.mean(axis=0)
    centered = pts_2d - mean_xy
    cov = np.cov(centered.T) if centered.shape[0] > 1 else np.eye(2)
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal = eigvecs[:, int(np.argmax(eigvals))]
    proj = centered @ principal
    p_min = mean_xy + float(proj.min()) * principal
    p_max = mean_xy + float(proj.max()) * principal
    z_base = float(np.percentile(points_xyz[:, 2], 5))
    start = (round(float(p_min[0]), 4), round(float(p_min[1]), 4), round(z_base, 4))
    end = (round(float(p_max[0]), 4), round(float(p_max[1]), 4), round(z_base, 4))
    length = float(math.hypot(end[0] - start[0], end[1] - start[1]))
    orient = float(math.atan2(end[1] - start[1], end[0] - start[0]))
    residuals = np.abs(centered @ np.array([-principal[1], principal[0]]))
    rms = float(np.sqrt(np.mean(residuals ** 2))) if residuals.size else 0.05
    return WallSlabObservation(
        start_xyz_m=start,
        end_xyz_m=end,
        coordinate_frame=coordinate_frame,
        source_camera=source_camera,
        source_evidence=source_evidence,
        depth_basis=depth_basis,
        length_m=round(length, 4),
        orientation_rad=round(orient, 4),
        measurement_uncertainty_m=round(max(0.02, rms), 4),
        visibility_state=visibility_state,
    )


def build_visual_observations_from_detections(
    *,
    source_evidence_id: str,
    source_image_id: str,
    detections: Sequence[Mapping[str, Any]],
    model_id: str,
    model_version: str,
    depth_map: Optional[np.ndarray] = None,
    depth_source: DepthSource = DepthSource.RELATIVE,
    camera_id: Optional[str] = None,
    camera_pose: Optional[Mapping[str, Any]] = None,
    source_video_id: Optional[str] = None,
    frame_index: Optional[int] = None,
    timestamp_s: Optional[float] = None,
    processing_config: Optional[Mapping[str, Any]] = None,
    quality: Optional[ImageQualityMetrics] = None,
) -> List[VisualObservation]:
    """Convert raw detector/segmentation/depth outputs into canonical VisualObservations."""
    out: List[VisualObservation] = []
    cfg = dict(processing_config or {})
    if quality is not None:
        cfg["image_quality"] = quality.to_dict()

    for idx, det in enumerate(detections):
        label = str(det.get("label") or det.get("class_name") or "unknown").strip()
        low = label.lower()
        category = (
            "opening" if low in ("door", "window", "opening")
            else "architecture" if low in ARCHITECTURAL_LABELS
            else "object"
        )
        raw_bbox = det.get("bbox_px") or det.get("bbox")
        bbox_px: Optional[Tuple[float, float, float, float]] = (
            (float(raw_bbox[0]), float(raw_bbox[1]), float(raw_bbox[2]), float(raw_bbox[3]))
            if isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) == 4
            else None
        )
        det_conf_raw = det.get("score") if det.get("score") is not None else det.get("confidence")
        det_conf = float(det_conf_raw) if isinstance(det_conf_raw, (int, float)) else None

        # Sample depth inside bbox if available (never fabricate metric world_pos from uncalibrated relative depth)
        est_depth_m: Optional[float] = None
        if depth_map is not None and bbox_px is not None:
            h, w = depth_map.shape[:2]
            x0 = max(0, min(w - 1, int(bbox_px[0])))
            y0 = max(0, min(h - 1, int(bbox_px[1])))
            x1 = max(x0 + 1, min(w, int(bbox_px[2])))
            y1 = max(y0 + 1, min(h, int(bbox_px[3])))
            patch = depth_map[y0:y1, x0:x1]
            if patch.size > 0 and np.any(np.isfinite(patch)):
                est_depth_m = float(np.nanmedian(patch))

        world_pos: Optional[Tuple[float, float, float]] = None
        sigma_m: Optional[float] = None
        if det.get("world_pos") is not None:
            wp = det["world_pos"]
            world_pos = (float(wp[0]), float(wp[1]), float(wp[2]))
            sigma_m = float(det.get("sigma_m") or sigma_for(depth_source, float(np.linalg.norm(world_pos))))
        elif depth_source in (DepthSource.METRIC, DepthSource.MVS) and est_depth_m is not None and camera_pose is not None:
            pos = camera_pose.get("position_m") or [0.0, 0.0, 0.0]
            world_pos = (float(pos[0]), float(pos[1]) + est_depth_m, float(pos[2]))
            sigma_m = sigma_for(depth_source, est_depth_m)

        obs_id = deterministic_observation_id({
            "source_evidence_id": source_evidence_id,
            "source_image_id": source_image_id,
            "frame_index": frame_index,
            "model_id": model_id,
            "model_version": model_version,
            "idx": idx,
            "label": label,
            "bbox_px": list(bbox_px) if bbox_px else None,
        })

        opening_obs = None
        if category == "opening":
            opening_obs = OpeningObservation(
                opening_class=low,
                bbox_px=bbox_px,
                host_wall_candidate=det.get("host_wall_candidate"),
                relative_location=None,
                estimated_width_m=det.get("estimated_width_m"),
                estimated_height_m=det.get("estimated_height_m"),
                evidence_refs=(source_evidence_id,),
                uncertainty_m=sigma_m,
            )

        out.append(VisualObservation(
            observation_id=obs_id,
            source_evidence_id=source_evidence_id,
            source_image_id=source_image_id,
            source_video_id=source_video_id,
            frame_index=frame_index,
            timestamp_s=timestamp_s,
            camera_id=camera_id,
            camera_pose=camera_pose,
            detection_category=category,
            semantic_label=label,
            bbox_px=bbox_px,
            mask_ref=det.get("mask_ref"),
            depth_source=depth_source,
            world_pos=world_pos,
            bounds_min=tuple(det["bounds_min"]) if det.get("bounds_min") else None,
            bounds_max=tuple(det["bounds_max"]) if det.get("bounds_max") else None,
            sigma_m=sigma_m,
            wall_slab=det.get("wall_slab"),
            opening=opening_obs,
            material_hypothesis=OBJECT_MATERIAL_PRIORS.get(low),
            material_verified=False,
            visual_attributes={"estimated_depth_raw": round(est_depth_m, 4)} if est_depth_m is not None else {},
            model_id=model_id,
            model_version=model_version,
            processing_config=cfg,
            confidences=TaskConfidences(
                detection=det_conf,
                segmentation=det.get("mask_confidence"),
                depth=None if est_depth_m is None else (0.7 if depth_source in (DepthSource.METRIC, DepthSource.MVS) else None),
                dimensions=None,
                material=None,
            ),
            provenance=Provenance.OBSERVED,
            visibility=Visibility.PARTIAL if quality and quality.is_blurry else Visibility.FULL,
            entity_hint_id=det.get("entity_hint_id"),
        ))
    return out


def detect_observation_conflicts(
    observations: Sequence[VisualObservation],
    *,
    spatial_radius_m: float = 0.6,
) -> List[AssociationDecision]:
    """Generate candidate multi-view correspondences and surface conflicting classifications/dimensions."""
    decisions: List[AssociationDecision] = []
    positioned = [o for o in observations if o.world_pos is not None]
    for i, a in enumerate(positioned):
        for b in positioned[i + 1:]:
            if a.source_image_id == b.source_image_id and a.frame_index == b.frame_index:
                continue
            dist = math.dist(a.world_pos, b.world_pos)  # type: ignore[arg-type]
            if dist > spatial_radius_m:
                continue
            if a.semantic_label.lower() != b.semantic_label.lower():
                decisions.append(AssociationDecision(
                    observation_id=b.observation_id,
                    state=AssociationState.AMBIGUOUS,
                    entity_id=a.entity_hint_id,
                    probability=0.5,
                    reason=f"Conflicting labels ({a.semantic_label} vs {b.semantic_label}) within {dist:.2f}m",
                    dist_m=round(dist, 4),
                    candidates=((a.entity_hint_id or a.observation_id, 0.5), (b.entity_hint_id or b.observation_id, 0.5)),
                ))
            else:
                decisions.append(AssociationDecision(
                    observation_id=b.observation_id,
                    state=AssociationState.SAME_ENTITY_CONFIRMED,
                    entity_id=a.entity_hint_id,
                    probability=round(max(0.5, 1.0 - dist / max(spatial_radius_m, 1e-6)), 4),
                    reason=f"Multi-view {a.semantic_label} correspondence within {dist:.2f}m",
                    dist_m=round(dist, 4),
                    candidates=((a.entity_hint_id or a.observation_id, 0.9),),
                ))
    return decisions


@dataclass(frozen=True)
class BridgeExecutionReport:
    added_ledger_observations: int
    skipped_duplicate_observations: int
    entity_observations_attached: int
    wall_slabs_populated: int
    constraint_violations: List[ConstraintViolation]
    uncertainty_profiles: Dict[str, EntityUncertaintyProfile]
    completeness: WorldCompletenessReport


def bridge_observations_to_world(
    world: WorldIR,
    visual_observations: Sequence[VisualObservation],
    *,
    extra_decisions: Sequence[AssociationDecision] = (),
) -> BridgeExecutionReport:
    """Idempotently attach VisualObservations to WorldIR ledger & entities, then run Claude's compiler checks."""
    existing_obs, existing_decs = ledger_of(world)
    obs_by_id: Dict[str, LedgerObservation] = {o.observation_id: o for o in existing_obs}
    dec_by_key: Dict[Tuple[str, str], AssociationDecision] = {
        (d.observation_id, d.state.value): d for d in existing_decs
    }

    added_ledger = 0
    skipped_dup = 0
    attached_entities = 0
    wall_slabs_populated = 0

    for vobs in visual_observations:
        lobs = vobs.to_ledger_observation()
        if lobs is not None:
            if lobs.observation_id in obs_by_id:
                skipped_dup += 1
            else:
                obs_by_id[lobs.observation_id] = lobs
                added_ledger += 1

        target_eid = vobs.entity_hint_id
        if target_eid and target_eid in world.entities:
            ent = world.entities[target_eid]
            existing_eobs = {o.id for o in ent.observations}
            if vobs.observation_id not in existing_eobs:
                ent.observations.append(vobs.to_entity_observation())
                attached_entities += 1
            if (
                ent.type is EntityType.WALL
                and vobs.wall_slab is not None
                and vobs.wall_slab.start_xyz_m is not None
                and vobs.wall_slab.end_xyz_m is not None
            ):
                cp = dict(ent.custom_properties or {})
                cp["wall_slab"] = vobs.wall_slab.to_dict()
                ent.custom_properties = cp
                wall_slabs_populated += 1

    auto_decs = detect_observation_conflicts(visual_observations)
    for d in (*auto_decs, *extra_decisions):
        dec_by_key[(d.observation_id, d.state.value)] = d

    attach_ledger(world, list(obs_by_id.values()), list(dec_by_key.values()))

    constraints = check_architectural_constraints(world)
    profiles = {eid: propagate_entity_uncertainty(world, eid) for eid in sorted(world.entities)}
    completeness = assess_world_completeness(world)

    return BridgeExecutionReport(
        added_ledger_observations=added_ledger,
        skipped_duplicate_observations=skipped_dup,
        entity_observations_attached=attached_entities,
        wall_slabs_populated=wall_slabs_populated,
        constraint_violations=constraints,
        uncertainty_profiles=profiles,
        completeness=completeness,
    )
