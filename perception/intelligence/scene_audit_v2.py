"""A4 Survey Completeness & Scene Perception Audit Engine (A4.10 / A4.13).

Evaluates whether a set of perception outputs is ready for downstream
as-built architectural and furniture reconstruction:
  - Multi-view coverage per image and per canonical entity
  - Architectural envelope readiness (walls, floor, ceiling, openings)
  - Furniture & fixture coverage and arrangement density
  - Depth & calibration honesty checks
  - Prioritized remediation checklist for operators and active perception
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from perception.intelligence.architectural_elevation import (
    ArchitecturalElevationReport,
    analyze_architectural_elevations,
)
from perception.intelligence.data_model import NormalizedPerceptionObservation
from perception.intelligence.furniture_v2 import Furniture2AnalysisResult, analyze_furniture_v2
from perception.intelligence.multiview import MultiViewPerceptionReport


@dataclass(frozen=True)
class AuditChecklistItem:
    check_id: str
    category: str  # "architecture" | "furniture" | "multiview" | "calibration" | "depth"
    passed: bool
    score: float
    detail: str
    remediation: str | None = None


@dataclass(frozen=True)
class ScenePerceptionAuditReport:
    run_id: str
    image_count: int
    observation_count: int
    overall_readiness_score: float
    is_survey_ready: bool
    checklist: tuple[AuditChecklistItem, ...]
    class_histogram: dict[str, int]
    remediation_actions: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "image_count": self.image_count,
            "observation_count": self.observation_count,
            "overall_readiness_score": self.overall_readiness_score,
            "is_survey_ready": self.is_survey_ready,
            "checklist": [
                {
                    "check_id": c.check_id,
                    "category": c.category,
                    "passed": c.passed,
                    "score": c.score,
                    "detail": c.detail,
                    "remediation": c.remediation,
                }
                for c in self.checklist
            ],
            "class_histogram": dict(self.class_histogram),
            "remediation_actions": list(self.remediation_actions),
        }


def run_scene_perception_audit(
    run_id: str,
    observations: Sequence[NormalizedPerceptionObservation],
    *,
    elevation_report: ArchitecturalElevationReport | None = None,
    furniture_report: Furniture2AnalysisResult | None = None,
    multiview_report: MultiViewPerceptionReport | None = None,
    has_depth: bool = False,
    has_human_reviewed_calibration: bool = False,
) -> ScenePerceptionAuditReport:
    """Audit perception outputs for architectural completeness, furniture density, and honesty."""
    by_img: dict[str, list[NormalizedPerceptionObservation]] = defaultdict(list)
    class_counts: Counter[str] = Counter()
    for o in observations:
        by_img[o.provenance.source_image_id].append(o)
        class_counts[o.canonical_label] += 1

    elev = elevation_report or analyze_architectural_elevations(observations)
    furn = furniture_report or analyze_furniture_v2(observations)

    checklist: list[AuditChecklistItem] = []
    remediations: list[str] = []

    # 1. Architectural envelope completeness
    env_score = float(elev.summary.get("mean_envelope_completeness", 0.0))
    arch_ok = env_score >= 0.60 and len(elev.profiles) >= 1
    checklist.append(AuditChecklistItem(
        check_id="arch_envelope_completeness",
        category="architecture",
        passed=arch_ok,
        score=round(env_score, 4),
        detail=f"Mean envelope completeness={env_score:.2f} across {len(elev.envelopes)} images.",
        remediation=None if arch_ok else "Capture floor, ceiling, and multi-wall corners with wider FOV.",
    ))
    if not arch_ok:
        remediations.append("Capture floor, ceiling, and multi-wall corners with wider FOV.")

    # 2. Opening & boundary representation
    opening_count = sum(
        class_counts[lbl] for lbl in ("door", "window", "opening", "door_frame", "window_frame")
    )
    open_ok = opening_count >= 1
    open_score = min(1.0, opening_count / 2.0)
    checklist.append(AuditChecklistItem(
        check_id="architectural_openings",
        category="architecture",
        passed=open_ok,
        score=round(open_score, 4),
        detail=f"Detected {opening_count} architectural openings/frames.",
        remediation=None if open_ok else "Capture doors/windows head-on to verify wall containment.",
    ))
    if not open_ok:
        remediations.append("Capture doors/windows head-on to verify wall containment.")

    # 3. Furniture & arrangement coverage
    furn_count = int(furn.summary.get("classified_objects", 0))
    arr_count = int(furn.summary.get("arrangements", 0))
    furn_ok = furn_count >= 1
    furn_score = min(1.0, 0.5 * (furn_count > 0) + 0.25 * min(2, furn_count) / 2.0 + 0.25 * (arr_count > 0))
    checklist.append(AuditChecklistItem(
        check_id="furniture_and_arrangements",
        category="furniture",
        passed=furn_ok,
        score=round(furn_score, 4),
        detail=f"Classified {furn_count} furniture/fixture items and {arr_count} arrangement groups.",
        remediation=None if furn_ok else "Run tiled/crop inference on interior furnishings.",
    ))
    if not furn_ok:
        remediations.append("Run tiled/crop inference on interior furnishings.")

    # 4. Multi-view coverage
    n_imgs = len(by_img)
    confirmed_tracks = 0
    if multiview_report is not None:
        confirmed_tracks = sum(
            1 for ids in multiview_report.visibility_tracks.values() if len(ids) >= 2
        )
    mv_ok = n_imgs >= 2 and (multiview_report is None or confirmed_tracks >= 1)
    mv_score = min(1.0, 0.5 * min(2, n_imgs) / 2.0 + 0.5 * (1.0 if confirmed_tracks >= 1 else 0.0))
    checklist.append(AuditChecklistItem(
        check_id="multiview_association",
        category="multiview",
        passed=mv_ok,
        score=round(mv_score, 4),
        detail=f"{n_imgs} images, {confirmed_tracks} multi-view object tracks.",
        remediation=None if mv_ok else "Provide overlapping multi-view frames (>=2 views per object).",
    ))
    if not mv_ok:
        remediations.append("Provide overlapping multi-view frames (>=2 views per object).")

    # 5. Depth & calibration honesty
    depth_score = 1.0 if has_depth else 0.25
    checklist.append(AuditChecklistItem(
        check_id="depth_association",
        category="depth",
        passed=has_depth,
        score=depth_score,
        detail="Depth map contract attached." if has_depth else "No depth maps associated; ordering is 2D only.",
        remediation=None if has_depth else "Attach relative or metric DepthMapContract outputs.",
    ))

    cal_score = 1.0 if has_human_reviewed_calibration else 0.5
    checklist.append(AuditChecklistItem(
        check_id="calibration_honesty",
        category="calibration",
        passed=True,
        score=cal_score,
        detail=(
            "Human-reviewed real calibration applied."
            if has_human_reviewed_calibration
            else "Scores explicitly marked uncalibrated (no human-reviewed real calibration)."
        ),
    ))

    overall = round(sum(c.score for c in checklist) / len(checklist), 4) if checklist else 0.0
    is_ready = arch_ok and furn_ok and n_imgs >= 2
    return ScenePerceptionAuditReport(
        run_id=run_id,
        image_count=n_imgs,
        observation_count=len(observations),
        overall_readiness_score=overall,
        is_survey_ready=is_ready,
        checklist=tuple(checklist),
        class_histogram=dict(sorted(class_counts.items())),
        remediation_actions=tuple(remediations),
    )


__all__ = [
    "AuditChecklistItem",
    "ScenePerceptionAuditReport",
    "run_scene_perception_audit",
]
