"""Human-Reviewed Perception Ground-Truth System (A4.2).

Supports:
  - Bounding boxes, instance masks, semantic masks, architectural boundaries
  - Door/window/furniture/component/occlusion/material/cross-view identity annotations
  - Ambiguous & unknown classes
  - Inter-annotator agreement (IoU & label kappa/agreement)
  - Review status (``unreviewed`` | ``in_review`` | ``human_reviewed`` | ``disputed``)
  - Versioning, dataset quality scoring, provenance, and JSON import/export
  - 15-image human-review workflow (never marks images reviewed without human review)
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from engine.multiengine.spatial_intelligence.detection_pipeline import compute_box_iou
from engine.multiengine.spatial_intelligence.taxonomy import resolve_taxonomy_class


class ReviewStatus(str, Enum):
    UNREVIEWED = "unreviewed"
    IN_REVIEW = "in_review"
    HUMAN_REVIEWED = "human_reviewed"
    DISPUTED = "disputed"


@dataclass(frozen=True)
class AnnotationProvenance:
    annotator_id: str
    review_status: ReviewStatus = ReviewStatus.UNREVIEWED
    reviewer_id: str | None = None
    reviewed_at_utc: str | None = None
    tool_version: str = "reality_annotator_v1"

    def __post_init__(self) -> None:
        if self.review_status is ReviewStatus.HUMAN_REVIEWED and not (self.reviewer_id and self.reviewed_at_utc):
            raise ValueError("HUMAN_REVIEWED requires explicit reviewer_id and reviewed_at_utc")


@dataclass(frozen=True)
class GroundTruthAnnotation:
    annotation_id: str
    image_id: str
    raw_label: str
    canonical_label: str
    bbox_px: tuple[float, float, float, float]
    image_width: int
    image_height: int
    provenance: AnnotationProvenance
    parent_annotation_id: str | None = None
    cross_view_identity_id: str | None = None
    is_occluded: bool = False
    is_truncated: bool = False
    is_ambiguous: bool = False
    material_label: str | None = None
    mask_rle: dict[str, Any] | None = None
    boundary_polygon_px: tuple[tuple[float, float], ...] = ()
    schema_version: str = "gt.perception.v1"

    def __post_init__(self) -> None:
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("image dimensions must be positive")
        x1, y1, x2, y2 = self.bbox_px
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"degenerate bbox_px: {self.bbox_px!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "annotation_id": self.annotation_id,
            "image_id": self.image_id,
            "raw_label": self.raw_label,
            "canonical_label": self.canonical_label,
            "bbox_px": list(self.bbox_px),
            "image_width": self.image_width,
            "image_height": self.image_height,
            "provenance": {
                "annotator_id": self.provenance.annotator_id,
                "review_status": self.provenance.review_status.value,
                "reviewer_id": self.provenance.reviewer_id,
                "reviewed_at_utc": self.provenance.reviewed_at_utc,
                "tool_version": self.provenance.tool_version,
            },
            "parent_annotation_id": self.parent_annotation_id,
            "cross_view_identity_id": self.cross_view_identity_id,
            "is_occluded": self.is_occluded,
            "is_truncated": self.is_truncated,
            "is_ambiguous": self.is_ambiguous,
            "material_label": self.material_label,
            "mask_rle": self.mask_rle,
            "boundary_polygon_px": [list(p) for p in self.boundary_polygon_px],
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GroundTruthAnnotation:
        pr = data["provenance"]
        prov = AnnotationProvenance(
            annotator_id=str(pr["annotator_id"]),
            review_status=ReviewStatus(pr["review_status"]),
            reviewer_id=pr.get("reviewer_id"),
            reviewed_at_utc=pr.get("reviewed_at_utc"),
            tool_version=str(pr.get("tool_version", "reality_annotator_v1")),
        )
        return cls(
            annotation_id=str(data["annotation_id"]),
            image_id=str(data["image_id"]),
            raw_label=str(data["raw_label"]),
            canonical_label=str(data["canonical_label"]),
            bbox_px=tuple(float(v) for v in data["bbox_px"]),
            image_width=int(data["image_width"]),
            image_height=int(data["image_height"]),
            provenance=prov,
            parent_annotation_id=data.get("parent_annotation_id"),
            cross_view_identity_id=data.get("cross_view_identity_id"),
            is_occluded=bool(data.get("is_occluded", False)),
            is_truncated=bool(data.get("is_truncated", False)),
            is_ambiguous=bool(data.get("is_ambiguous", False)),
            material_label=data.get("material_label"),
            mask_rle=data.get("mask_rle"),
            boundary_polygon_px=tuple((float(p[0]), float(p[1])) for p in data.get("boundary_polygon_px", ())),
            schema_version=str(data.get("schema_version", "gt.perception.v1")),
        )


def compute_inter_annotator_agreement(
    annotator_a: Sequence[GroundTruthAnnotation],
    annotator_b: Sequence[GroundTruthAnnotation],
    *,
    iou_threshold: float = 0.50,
) -> dict[str, Any]:
    """Measure box IoU agreement and label agreement between two human annotators."""
    matched = 0
    label_agreements = 0
    ious: list[float] = []
    used_b: set[int] = set()

    for a in annotator_a:
        best_iou = 0.0
        best_idx: int | None = None
        for bi, b in enumerate(annotator_b):
            if bi in used_b or b.image_id != a.image_id:
                continue
            iou = compute_box_iou(a.bbox_px, b.bbox_px)
            if iou >= iou_threshold and iou > best_iou:
                best_iou = iou
                best_idx = bi
        if best_idx is not None:
            used_b.add(best_idx)
            matched += 1
            ious.append(best_iou)
            if annotator_b[best_idx].canonical_label == a.canonical_label:
                label_agreements += 1

    total = max(len(annotator_a), len(annotator_b), 1)
    return {
        "matched_pairs": matched,
        "box_agreement_rate": round(matched / total, 4),
        "label_agreement_on_matched": round(label_agreements / max(1, matched), 4) if matched else 0.0,
        "mean_matched_iou": round(sum(ious) / len(ious), 4) if ious else 0.0,
    }


def score_ground_truth_dataset_quality(annotations: Sequence[GroundTruthAnnotation]) -> dict[str, Any]:
    """Score dataset readiness: reviewed fraction, mask coverage, identity coverage, and ambiguity rate."""
    n = len(annotations)
    if n == 0:
        return {"total_annotations": 0, "quality_score": 0.0, "human_reviewed_fraction": 0.0}
    reviewed = sum(1 for a in annotations if a.provenance.review_status is ReviewStatus.HUMAN_REVIEWED)
    masked = sum(1 for a in annotations if a.mask_rle is not None)
    with_id = sum(1 for a in annotations if a.cross_view_identity_id is not None)
    disputed = sum(1 for a in annotations if a.provenance.review_status is ReviewStatus.DISPUTED)

    rev_frac = reviewed / n
    mask_frac = masked / n
    id_frac = with_id / n
    disp_frac = disputed / n
    score = max(0.0, min(1.0, 0.50 * rev_frac + 0.25 * mask_frac + 0.25 * id_frac - 0.20 * disp_frac))
    return {
        "total_annotations": n,
        "human_reviewed_fraction": round(rev_frac, 4),
        "mask_coverage": round(mask_frac, 4),
        "identity_coverage": round(id_frac, 4),
        "disputed_fraction": round(disp_frac, 4),
        "quality_score": round(score, 4),
    }


def build_human_review_manifest(
    image_paths: Sequence[str],
    *,
    required_count: int = 15,
    preannotations: Sequence[GroundTruthAnnotation] = (),
) -> dict[str, Any]:
    """Prepare a 15-image human-review workflow manifest (all images start UNREVIEWED)."""
    selected = list(image_paths[:required_count])
    by_img: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ann in preannotations:
        by_img[ann.image_id].append(ann.to_dict())

    items = [
        {
            "image_id": Path(p).name,
            "image_path": p,
            "review_status": ReviewStatus.UNREVIEWED.value,
            "reviewer_id": None,
            "reviewed_at_utc": None,
            "preannotations": by_img.get(Path(p).name, []),
        }
        for p in selected
    ]
    return {
        "schema_version": "gt.review_manifest.v1",
        "required_image_count": required_count,
        "selected_image_count": len(selected),
        "reviewed_count": 0,
        "ready_for_calibration": False,
        "items": items,
    }


def export_ground_truth_json(annotations: Sequence[GroundTruthAnnotation], out_path: Path | str) -> None:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "gt.perception.v1",
        "annotations": [a.to_dict() for a in annotations],
    }
    p.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")


def import_ground_truth_json(in_path: Path | str) -> list[GroundTruthAnnotation]:
    data = json.loads(Path(in_path).read_text(encoding="utf-8"))
    return [GroundTruthAnnotation.from_dict(row) for row in data.get("annotations", ())]


__all__ = [
    "AnnotationProvenance",
    "GroundTruthAnnotation",
    "ReviewStatus",
    "build_human_review_manifest",
    "compute_inter_annotator_agreement",
    "export_ground_truth_json",
    "import_ground_truth_json",
    "score_ground_truth_dataset_quality",
]
