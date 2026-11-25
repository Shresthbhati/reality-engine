"""Shared producer-consumer contract ``spatial_inputs.v1`` (Phase 9).

Produces validated, deterministic ``spatial_inputs.v1`` payloads from Reality
Engine's perception pipeline (``VisualSceneGraph`` / ``VisualObservation``s)
and from API job Evidence Matrix snapshots, and exposes the
``register_inputs_provider`` / ``get_inputs_provider`` seam for Claude's
downstream ``spatial_global`` reconstruction consumer.

Invariants:
  1. Never fabricates camera poses (absent poses stay empty).
  2. Never assumes metric scale from relative depth.
  3. Never emits unsupported 3D geometry.
  4. Deduplicates observations deterministically by ``observation_id``.
  5. Validates every output via ``SpatialInputsV1.validate()``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

from world_ir.observations import DepthSource

SCHEMA_VERSION = "spatial_inputs.v1"
SPATIAL_INPUTS_FILE = "spatial_inputs_v1.json"


class SpatialInputValidationError(ValueError):
    """Raised when a ``spatial_inputs.v1`` payload violates schema or honesty invariants."""


@dataclass(frozen=True)
class ScaleStateInput:
    basis: str  # "metric_calibrated" | "sfm_gauge" | "relative_uncalibrated" | "unknown"
    unit: str   # "metres" | "sfm_gauge" | "relative" | "unknown"
    is_metric: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CameraInput:
    camera_id: str
    image_id: str
    source_uri: str
    position_m: Tuple[float, float, float] | None
    orientation_wxyz: Tuple[float, float, float, float] | None
    intrinsics: Dict[str, Any] | None
    pose_source: str  # "colmap" | "vio" | "registered" | "unknown"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.position_m is not None:
            d["position_m"] = list(self.position_m)
        if self.orientation_wxyz is not None:
            d["orientation_wxyz"] = list(self.orientation_wxyz)
        return d


@dataclass(frozen=True)
class SpatialObservationInput:
    observation_id: str
    evidence_id: str
    image_id: str
    frame_index: int | None
    timestamp_s: float | None
    label: str
    category: str
    bbox_px: Tuple[float, float, float, float] | None
    has_mask: bool
    mask_ref: str | None
    depth_source: str
    depth_evidence: Dict[str, Any] | None
    world_pos: Tuple[float, float, float] | None
    bounds_min: Tuple[float, float, float] | None
    bounds_max: Tuple[float, float, float] | None
    sigma_m: float | None
    model_id: str
    model_version: str
    confidence: float | None
    uncertainty: Dict[str, Any]
    material_hypotheses: Tuple[Dict[str, Any], ...] = ()
    failure_flags: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        for k in ("bbox_px", "world_pos", "bounds_min", "bounds_max"):
            if d[k] is not None:
                d[k] = list(d[k])
        d["material_hypotheses"] = list(self.material_hypotheses)
        d["failure_flags"] = list(self.failure_flags)
        return d


@dataclass(frozen=True)
class StructuralInput:
    observation_id: str
    kind: str  # "wall_slab" | "opening" | "structural_boundary"
    label: str
    evidence_id: str
    payload: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SpatialInputsV1:
    run_id: str
    scale_state: ScaleStateInput
    cameras: Tuple[CameraInput, ...]
    observations: Tuple[SpatialObservationInput, ...]
    structural_observations: Tuple[StructuralInput, ...]
    cross_view_associations: Tuple[Dict[str, Any], ...]
    relationships: Tuple[Dict[str, Any], ...]
    room_hypotheses: Tuple[Dict[str, Any], ...]
    storey_id: str | None = None
    provenance: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise SpatialInputValidationError(f"invalid schema_version: {self.schema_version!r}")
        if not self.run_id or not str(self.run_id).strip():
            raise SpatialInputValidationError("run_id must be non-empty")
        seen_ids: set[str] = set()
        for obs in self.observations:
            if not obs.observation_id:
                raise SpatialInputValidationError("observation_id must be non-empty")
            if obs.observation_id in seen_ids:
                raise SpatialInputValidationError(f"duplicate observation_id: {obs.observation_id}")
            seen_ids.add(obs.observation_id)
            if not obs.evidence_id:
                raise SpatialInputValidationError(f"observation {obs.observation_id} missing evidence_id")
            if obs.world_pos is not None and obs.depth_source in ("relative", "unknown") and not self.cameras:
                raise SpatialInputValidationError(
                    f"observation {obs.observation_id} carries world_pos under {obs.depth_source!r} depth without registered cameras"
                )
            if obs.confidence is not None and not (0.0 <= float(obs.confidence) <= 1.0):
                raise SpatialInputValidationError(
                    f"observation {obs.observation_id} confidence out of [0,1]: {obs.confidence}"
                )

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        body = {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "storey_id": self.storey_id,
            "scale_state": self.scale_state.to_dict(),
            "cameras": [c.to_dict() for c in self.cameras],
            "observations": [o.to_dict() for o in self.observations],
            "structural_observations": [s.to_dict() for s in self.structural_observations],
            "cross_view_associations": list(self.cross_view_associations),
            "relationships": list(self.relationships),
            "room_hypotheses": list(self.room_hypotheses),
            "provenance": dict(self.provenance),
        }
        digest = hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode("utf-8")).hexdigest()
        body["content_sha256"] = digest
        return body

    def write_json(self, path: Path) -> Dict[str, Any]:
        payload = self.to_dict()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return payload


# ---------------------------------------------------------------------------
# Provider Registry Seam (register_inputs_provider)
# ---------------------------------------------------------------------------

InputsProvider = Callable[..., SpatialInputsV1]
_PROVIDERS: Dict[str, InputsProvider] = {}


def register_inputs_provider(name: str, provider: InputsProvider) -> None:
    if not name or not callable(provider):
        raise ValueError("provider name must be non-empty and provider must be callable")
    _PROVIDERS[name] = provider


def get_inputs_provider(name: str = "perception_scene_intelligence") -> InputsProvider:
    if name not in _PROVIDERS:
        raise KeyError(f"spatial_inputs.v1 provider {name!r} not registered; available: {sorted(_PROVIDERS)}")
    return _PROVIDERS[name]


def list_inputs_providers() -> Tuple[str, ...]:
    return tuple(sorted(_PROVIDERS))


# ---------------------------------------------------------------------------
# Producers
# ---------------------------------------------------------------------------

def _cameras_to_inputs(cameras: Mapping[str, Any] | None) -> Tuple[CameraInput, ...]:
    if not cameras:
        return ()
    out: List[CameraInput] = []
    for key in sorted(cameras):
        cam = cameras[key]
        ext = getattr(cam, "extrinsics", None)
        intr = getattr(cam, "intrinsics", None)
        pos = getattr(ext, "position", None)
        rot = getattr(ext, "rotation", None)
        pos_t = (float(pos.x), float(pos.y), float(pos.z)) if pos is not None else None
        rot_t = (float(rot.w), float(rot.x), float(rot.y), float(rot.z)) if rot is not None else None
        intr_d = (
            {
                "fx": float(intr.fx),
                "fy": float(intr.fy),
                "cx": float(intr.cx),
                "cy": float(intr.cy),
                "width": int(intr.width),
                "height": int(intr.height),
            }
            if intr is not None
            else None
        )
        out.append(
            CameraInput(
                camera_id=str(key),
                image_id=str(key),
                source_uri=str(key),
                position_m=pos_t,
                orientation_wxyz=rot_t,
                intrinsics=intr_d,
                pose_source="registered" if pos_t is not None else "unknown",
            )
        )
    return tuple(out)


def produce_spatial_inputs_from_scene_graph(
    run_id: str,
    scene_graph: Any,
    *,
    cameras: Mapping[str, Any] | None = None,
    cross_view_associations: Sequence[Mapping[str, Any]] | None = None,
    storey_id: str | None = None,
    scale_basis: str | None = None,
) -> SpatialInputsV1:
    """Convert a ``VisualSceneGraph`` into a validated ``SpatialInputsV1`` contract."""
    obs_by_id: Dict[str, SpatialObservationInput] = {}
    structural: List[StructuralInput] = []
    has_metric = False
    models_used: set[str] = set()
    source_uris: set[str] = set()

    for vobs in sorted(getattr(scene_graph, "observations", ()), key=lambda o: o.observation_id):
        if vobs.observation_id in obs_by_id:
            continue
        models_used.add(f"{vobs.model_id}:{vobs.model_version}")
        source_uris.add(str(vobs.source_image_id))
        ds = vobs.depth_source.value if isinstance(vobs.depth_source, DepthSource) else str(vobs.depth_source)
        if ds == DepthSource.METRIC.value:
            has_metric = True

        depth_ev = (vobs.processing_config or {}).get("depth_evidence")
        seg_q = (vobs.processing_config or {}).get("segmentation_quality")
        has_mask = bool(vobs.mask_ref) or (isinstance(seg_q, dict) and seg_q.get("validity") == "valid")
        mat_hyps = tuple((vobs.visual_attributes or {}).get("material_hypotheses") or ())
        failure_flags: List[str] = []
        if isinstance(depth_ev, dict) and depth_ev.get("unreliable_reason"):
            failure_flags.append(str(depth_ev["unreliable_reason"]))
        if vobs.semantic_label.lower() == "unknown":
            failure_flags.append("unknown_semantic_label")

        obs_by_id[vobs.observation_id] = SpatialObservationInput(
            observation_id=vobs.observation_id,
            evidence_id=vobs.source_evidence_id,
            image_id=vobs.source_image_id,
            frame_index=vobs.frame_index,
            timestamp_s=vobs.timestamp_s,
            label=vobs.semantic_label,
            category=vobs.detection_category,
            bbox_px=vobs.bbox_px,
            has_mask=has_mask,
            mask_ref=vobs.mask_ref,
            depth_source=ds,
            depth_evidence=depth_ev if isinstance(depth_ev, dict) else None,
            world_pos=vobs.world_pos,
            bounds_min=vobs.bounds_min,
            bounds_max=vobs.bounds_max,
            sigma_m=vobs.sigma_m,
            model_id=vobs.model_id,
            model_version=vobs.model_version,
            confidence=vobs.confidences.detection,
            uncertainty={"sigma_m": vobs.sigma_m, "task_confidences": vobs.confidences.to_dict()},
            material_hypotheses=mat_hyps,
            failure_flags=tuple(failure_flags),
        )
        if vobs.wall_slab is not None:
            structural.append(
                StructuralInput(
                    observation_id=vobs.observation_id,
                    kind="wall_slab",
                    label=vobs.semantic_label,
                    evidence_id=vobs.source_evidence_id,
                    payload=vobs.wall_slab.to_dict(),
                )
            )
        if vobs.opening is not None:
            structural.append(
                StructuralInput(
                    observation_id=vobs.observation_id,
                    kind="opening",
                    label=vobs.semantic_label,
                    evidence_id=vobs.source_evidence_id,
                    payload=vobs.opening.to_dict(),
                )
            )

    if scale_basis == "metric_calibrated" or has_metric:
        scale_state = ScaleStateInput(basis="metric_calibrated", unit="metres", is_metric=True)
    elif scale_basis == "sfm_gauge":
        scale_state = ScaleStateInput(basis="sfm_gauge", unit="sfm_gauge", is_metric=False)
    else:
        scale_state = ScaleStateInput(basis="relative_uncalibrated", unit="relative", is_metric=False)

    payload = SpatialInputsV1(
        run_id=run_id,
        scale_state=scale_state,
        cameras=_cameras_to_inputs(cameras),
        observations=tuple(obs_by_id[k] for k in sorted(obs_by_id)),
        structural_observations=tuple(structural),
        cross_view_associations=tuple(dict(a) for a in (cross_view_associations or ())),
        relationships=tuple(r.to_dict() for r in getattr(scene_graph, "relationships", ())),
        room_hypotheses=tuple(h.to_dict() for h in getattr(scene_graph, "room_hypotheses", ())),
        storey_id=storey_id,
        provenance={
            "producer": "perception.spatial_inputs.produce_spatial_inputs_from_scene_graph",
            "models_used": sorted(models_used),
            "source_uris": sorted(source_uris),
        },
    )
    payload.validate()
    return payload


def produce_spatial_inputs_from_snapshot(
    run_id: str,
    snapshot: Mapping[str, Any],
    *,
    cameras: Mapping[str, Any] | None = None,
    sources: Sequence[str] = (),
) -> SpatialInputsV1:
    """Produce a validated ``SpatialInputsV1`` payload from an Evidence Matrix job snapshot."""
    obs_by_id: Dict[str, SpatialObservationInput] = {}
    models_used: set[str] = set()
    structural: List[StructuralInput] = []

    # 1. Entity-attached observations from snapshot["entities"]
    for ent in snapshot.get("entities") or ():
        we = ent.get("world_entity") or {}
        label = str(ent.get("ontology_path") or we.get("name") or "unknown")
        cat = str((we.get("custom_properties") or {}).get("evidence", {}).get("ontology", {}).get("category") or we.get("type") or "object")
        for raw in we.get("observations") or ():
            oid = str(raw.get("id") or "")
            if not oid or oid in obs_by_id:
                continue
            meta = raw.get("metadata") or {}
            eng = str(meta.get("engine_id") or "evidence_matrix")
            ver = str(meta.get("model_version") or "")
            models_used.add(f"{eng}:{ver}" if ver else eng)
            src = str(raw.get("data_uri") or "")
            reg = meta.get("image_region")
            bbox_px = tuple(float(v) for v in reg) if isinstance(reg, (list, tuple)) and len(reg) == 4 else None
            raw_conf = raw.get("confidence") if raw.get("confidence") is not None else meta.get("raw_confidence")
            obs_by_id[oid] = SpatialObservationInput(
                observation_id=oid,
                evidence_id=src or oid,
                image_id=str(raw.get("frame_id") or src or oid),
                frame_index=None,
                timestamp_s=float(raw["timestamp"]) if raw.get("timestamp") is not None else None,
                label=label,
                category=cat,
                bbox_px=bbox_px,
                has_mask=bool(meta.get("quantity") == "mask"),
                mask_ref=None,
                depth_source="relative",
                depth_evidence=None,
                world_pos=None,
                bounds_min=None,
                bounds_max=None,
                sigma_m=None,
                model_id=eng,
                model_version=ver,
                confidence=float(raw_conf) if raw_conf is not None else None,
                uncertainty=dict(raw.get("uncertainty") or {"calibrated": False}),
            )

    # 2. Flat observations if present
    for raw in snapshot.get("observations") or ():
        oid = str(raw.get("observation_id") or "")
        if not oid or oid in obs_by_id:
            continue
        src = str(raw.get("source") or raw.get("evidence_id") or "")
        eng = str(raw.get("engine_id") or "evidence_matrix")
        models_used.add(eng)
        kind = str(raw.get("kind") or "observation")
        val = raw.get("value")
        bbox_px = tuple(float(v) for v in val["bbox_px"]) if isinstance(val, Mapping) and isinstance(val.get("bbox_px"), (list, tuple)) and len(val["bbox_px"]) == 4 else None
        world_pos = tuple(float(v) for v in val["position"]) if isinstance(val, Mapping) and isinstance(val.get("position"), (list, tuple)) and len(val["position"]) == 3 else None
        obs_by_id[oid] = SpatialObservationInput(
            observation_id=oid,
            evidence_id=src or oid,
            image_id=src or oid,
            frame_index=raw.get("frame_index"),
            timestamp_s=raw.get("timestamp_s"),
            label=str(raw.get("label") or kind),
            category=kind,
            bbox_px=bbox_px,
            has_mask=bool(isinstance(val, Mapping) and val.get("has_mask")),
            mask_ref=val.get("mask_ref") if isinstance(val, Mapping) else None,
            depth_source="sfm_gauge" if world_pos is not None else "relative",
            depth_evidence=None,
            world_pos=world_pos,
            bounds_min=None,
            bounds_max=None,
            sigma_m=None,
            model_id=eng,
            model_version="",
            confidence=float(raw["confidence"]) if raw.get("confidence") is not None else None,
            uncertainty={"calibrated": False},
        )

    cam_inputs = _cameras_to_inputs(cameras)
    basis = "sfm_gauge" if cam_inputs else "relative_uncalibrated"
    unit = "sfm_gauge" if cam_inputs else "relative"
    payload = SpatialInputsV1(
        run_id=run_id,
        scale_state=ScaleStateInput(basis=basis, unit=unit, is_metric=False),
        cameras=cam_inputs,
        observations=tuple(obs_by_id[k] for k in sorted(obs_by_id)),
        structural_observations=tuple(structural),
        cross_view_associations=tuple(snapshot.get("associations") or ()),
        relationships=tuple(snapshot.get("relations") or ()),
        room_hypotheses=(),
        storey_id=None,
        provenance={
            "producer": "perception.spatial_inputs.produce_spatial_inputs_from_snapshot",
            "models_used": sorted(models_used),
            "source_uris": sorted(str(s) for s in sources),
        },
    )
    payload.validate()
    return payload


register_inputs_provider("perception_scene_intelligence", produce_spatial_inputs_from_scene_graph)
register_inputs_provider("evidence_matrix_snapshot", produce_spatial_inputs_from_snapshot)
