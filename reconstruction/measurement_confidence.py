"""Measurement confidence and tolerances (plan item D.19).

Attaches confidence intervals and provenance to every dimension in a world,
distinguishing directly measured distances from image-derived estimates.

Honesty rules:
* A dimension with no supporting measurement is "unknown", not "zero".
* Every reported value carries: the number, the unit, the confidence (0..1),
  the provenance (tape_measured | image_derived | heuristic | unknown), and
  the evidence source ids.
* Heuristic defaults (e.g. wall thickness 0.10 m) are labelled and never
  presented as measurements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from world_ir import Entity, EntityType


@dataclass(frozen=True)
class MeasuredDimension:
    """One dimension with confidence and provenance."""
    value_m: float
    confidence: float          # 0..1
    provenance: str            # "tape_measured" | "image_derived" | "heuristic" | "unknown"
    evidence_ids: List[str]    # source photos / sensor observations
    method: str                # how it was obtained
    tolerance_m: Optional[float] = None  # +/- tolerance if known
    note: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "value_m": round(self.value_m, 4),
            "confidence": round(self.confidence, 3),
            "provenance": self.provenance,
            "evidence_ids": list(self.evidence_ids),
            "method": self.method,
            "tolerance_m": None if self.tolerance_m is None else round(self.tolerance_m, 4),
            "note": self.note,
        }


@dataclass(frozen=True)
class DimensionReport:
    """All dimensions for one entity or measurement."""
    dimensions: Dict[str, MeasuredDimension]
    overall_confidence: float  # 0..1
    note: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "dimensions": {k: d.to_dict() for k, d in self.dimensions.items()},
            "overall_confidence": round(self.overall_confidence, 3),
            "note": self.note,
        }


class MeasurementConfidence:
    """Computes confidence-weighted dimensions for WorldIR entities."""

    #: Confidence for a dimension directly tape-measured.
    CONF_TAPE = 0.95
    #: Confidence for a dimension derived from calibrated imagery.
    CONF_IMAGE = 0.60
    #: Confidence for a labelled heuristic default (e.g. wall thickness).
    CONF_HEURISTIC = 0.30
    #: Confidence for an unprovenanced value.
    CONF_UNKNOWN = 0.0

    def dimension_for_entity(
        self,
        entity: Entity,
        kind: str,                     # "length" | "height" | "thickness" | "width"
        tape_value: Optional[float] = None,
        tape_error_m: Optional[float] = None,
        image_value: Optional[float] = None,
        image_baseline_m: Optional[float] = None,
        heuristic_value: Optional[float] = None,
        note: Optional[str] = None,
    ) -> MeasuredDimension:
        """Return the best dimension for ``kind`` with confidence.

        Priority: tape_measured > image_derived > heuristic > unknown.
        """
        # ---- tape measurement wins outright --------------------------------
        if tape_value is not None:
            conf = self.CONF_TAPE
            tol = tape_error_m
            note = "Directly measured with a tape or laser distancer."
            return MeasuredDimension(
                value_m=float(tape_value), confidence=conf,
                provenance="tape_measured", evidence_ids=[],
                method="tape_measurement", tolerance_m=tol, note=note)

        # ---- image-derived --------------------------------------------------
        if image_value is not None:
            # confidence falls with distance from camera and baseline quality
            conf = self.CONF_IMAGE
            note = "Derived from captured imagery; uncalibrated without a scale reference."
            if image_baseline_m is not None and image_baseline_m < 0.5:
                conf = round(conf * 0.7, 3)
                note += " Short baseline: treat as approximate."
            return MeasuredDimension(
                value_m=float(image_value), confidence=conf,
                provenance="image_derived", evidence_ids=[],
                method="photogrammetry", tolerance_m=None, note=note)

        # ---- heuristic default ----------------------------------------------
        if heuristic_value is not None:
            base = ("Labelled default, not a measurement. No pipeline stage "
                    "observes both faces of this element.")
            note = base if note is None else base + " " + note
            return MeasuredDimension(
                value_m=float(heuristic_value), confidence=self.CONF_HEURISTIC,
                provenance="heuristic", evidence_ids=[],
                method="labelled_default", note=note)

        # ---- nothing known --------------------------------------------------
        return MeasuredDimension(
            value_m=0.0, confidence=self.CONF_UNKNOWN,
            provenance="unknown", evidence_ids=[], method="none",
            note="No measurement or estimate available for this dimension.",
        )

    def wall_dimensions(
        self,
        wall: Entity,
        tape_length_m: Optional[float] = None,
        tape_height_m: Optional[float] = None,
        image_length_m: Optional[float] = None,
        image_height_m: Optional[float] = None,
    ) -> DimensionReport:
        """Dimensions for a wall entity: length, height, thickness."""
        cp = wall.custom_properties
        heur_thick = cp.get("thickness_m")
        heur_thick_basis = cp.get("thickness_basis", "default_unmeasured")

        length = self.dimension_for_entity(
            wall, "length",
            tape_value=tape_length_m, image_value=image_length_m,
            heuristic_value=cp.get("length_m"))
        height = self.dimension_for_entity(
            wall, "height",
            tape_value=tape_height_m, image_value=image_height_m,
            heuristic_value=cp.get("height_m"))
        thickness = self.dimension_for_entity(
            wall, "thickness",
            heuristic_value=heur_thick,
            note=(f"Wall thickness basis: {heur_thick_basis}." if heur_thick else None))

        dims = {"length": length, "height": height, "thickness": thickness}
        overall = round(float(np_mean([d.confidence for d in dims.values()])), 3)
        return DimensionReport(dimensions=dims, overall_confidence=overall,
                               note="Wall thickness is a labelled default unless tape-measured.")


def np_mean(values: Sequence[float]) -> float:
    """Mean of a non-empty sequence; 0.0 if empty."""
    vals = [float(v) for v in values]
    return float(sum(vals) / len(vals)) if vals else 0.0


__all__ = ["MeasurementConfidence", "MeasuredDimension", "DimensionReport", "np_mean"]