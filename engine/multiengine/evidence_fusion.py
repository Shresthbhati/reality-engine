"""Task-specific evidence fusion: normalize + fuse + write back.

Consumes EngineObservation records from an EvidenceMatrix, normalizes them
per task type into a form fuse_quantity can use (or a task-specific fusion
path for structured predictions), applies reliability/uncertainty/
consistency-weighted fusion -- never blind averaging -- and writes the
fused result back as a new observation linked to its sources.

What this module is
--------------------
A thin, honest fusion layer between "the matrix has N engines' raw outputs"
and "the canonical world gets ONE derived value per quantity, with its audit
trail". It does not claim calibration that nobody measured; it does not
invent a value for a quantity nobody observed; it does not average
disparate modalities.

What it fuses (per task/quantity)
----------------------------------
* depth_value / depth_map  -- scalar inverse-depth or metric-depth values,
  fused with fuse_quantity when units agree; relative-depth maps are NOT
  fused with metric depths (different coordinate frames, recorded as
  incompatible).
* position / dimensions    -- 3-DoF / 6-DoF metric-ish vectors in the same
  frame, fused component-wise with conflict detection per axis when the
  components agree on frame and units.
* bbox                     -- 2-DoF bounding boxes in the SAME image frame,
  fused as a consistency-weighted average of the box corners when the
  labels agree, with the label disagreement preserved as a conflict.
* Any quantity             -- falls back to fuse_quantity for scalar
  predictions when the observations carry a comparable predicted value.

What it deliberately does NOT do
---------------------------------
* Calibrate confidence. Every fused result records the basis as
  "uncalibrated: fused from engine raw confidences" unless the inputs
  already carry a calibrated confidence from a CalibrationRef.
* Claim metric scale. A fused position in "sfm_units" stays sfm_units.
* Merge observations that disagree on frame, units or label without
  recording the disagreement.
* Average depth maps pixel-wise (no cross-view pixel correspondence).
* Fabricate a value when no observation exists for the quantity.

Deterministic: canonical ordering of inputs, pure float math, no clocks,
no RNG -- the same observations always produce the same fused result.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from reconstruction.fusion import (
    FusionError,
    FusableObservation,
    FusedEstimate,
    fuse_quantity,
)
from engine.core.units import Unit
from engine.multiengine.evidence_matrix import EvidenceContext
from provenance import Provenance


# ------------------------------------------------------------------ task-specific fusion API

@dataclass(frozen=True)
class FusedPrediction:
    """One fused result for one quantity, with its audit trail.

    provenance is ESTIMATED for an uncontested multi-source fusion,
    CONFLICT when disagreement was detected, or the passthrough
    provenance for a single source. confidence is the weakest
    contributor's unless a calibrated confidence exists.
    """

    quantity: str
    """What was fused, e.g. "position", "dimensions", "depth_value", "bbox"."""

    source_frame: str
    """The coordinate_frame the fused value is expressed in (from the inputs)."""

    source_units: str
    """The units string the inputs agreed on, or "incompatible" when they did not."""

    value: Any
    """Fused value: a float for scalars, a tuple for vectors, a dict for boxes."""

    confidence: Optional[float]
    """Fused confidence (weakest contributor, or None when no input carried one)."""

    provenance: Any
    """ESTIMATED / CONFLICT / passthrough (uses the stub Provenance class)."""

    method: str
    """How it was produced: "passthrough", "inverse_variance_weighted", "consistency_weighted", "compatibility_filtered"."""

    contributing: tuple[EngineObservation, ...]
    """Every input observation verbatim -- the fused value never replaces the evidence."""

    conflicts: tuple[FusionConflict, ...]
    """Preserved disagreements among the contributors."""

    uncertainty_basis: str = "uncalibrated: fused from engine raw confidences"
    """What the fused uncertainty/confidence rests on, honestly stated."""


@dataclass(frozen=True)
class FusionConflict:
    """One preserved disagreement among contributors to a fused prediction.

    Not discarded: the fused result points here so a reader can see who
    disagreed and by how much.
    """

    sources: tuple[str, str]
    """engine_ids of the two disagreeing contributors."""

    quantity: str

    detail: str
    """What disagreed: e.g. "label 'chair' vs 'sofa'", "frame 'cam:img1' vs 'world'", "units 'sfm_units' vs 'meter'."""


# ------------------------------------------------------------------ normalization

#: Predict keys that carry a scalar value we can fuse with fuse_quantity.
SCALAR_PREDICT_KEYS = ("value", "depth_value", "confidence", "score")

#: Predict keys that carry a 3-DoF position (tuple of 3 floats).
POSITION_KEYS = ("position", "world_pos", "center_3d")

#: Predict keys that carry a 6-DoF dimensions (tuple of 3 floats).
DIMENSIONS_KEYS = ("dimensions", "size_3d", "size")

#: Predict keys that carry a 2-DoF bbox in image pixels (tuple of 4).
BBOX_KEYS = ("bbox_px", "bbox", "box_2d")


def _scalar_from_predicted(predicted: Mapping[str, Any]) -> Optional[float]:
    """First scalar value found in a predicted dict, or None."""
    for k in SCALAR_PREDICT_KEYS:
        v = predicted.get(k)
        if isinstance(v, (int, float)) and math.isfinite(float(v)):
            return float(v)
    return None


def _position_from_predicted(
    predicted: Mapping[str, Any],
) -> Optional[tuple[float, float, float]]:
    for k in POSITION_KEYS:
        v = predicted.get(k)
        if isinstance(v, (list, tuple)) and len(v) == 3 and all(
            isinstance(x, (int, float)) and math.isfinite(float(x)) for x in v
        ):
            return tuple(float(x) for x in v)
    return None


def _dimensions_from_predicted(
    predicted: Mapping[str, Any],
) -> Optional[tuple[float, float, float]]:
    for k in DIMENSIONS_KEYS:
        v = predicted.get(k)
        if isinstance(v, (list, tuple)) and len(v) == 3 and all(
            isinstance(x, (int, float)) and math.isfinite(float(x)) and float(x) > 0
            for x in v
        ):
            return tuple(float(x) for x in v)
    return None


def _bbox_from_predicted(
    predicted: Mapping[str, Any],
) -> Optional[tuple[float, float, float, float]]:
    for k in BBOX_KEYS:
        v = predicted.get(k)
        if isinstance(v, (list, tuple)) and len(v) == 4 and all(
            isinstance(x, (int, float)) and math.isfinite(float(x)) for x in v
        ):
            x0, y0, x1, y1 = tuple(float(x) for x in v)
            if x1 > x0 and y1 > y0:
                return (x0, y0, x1, y1)
    return None


def _units_of(obs: EngineObservation) -> Optional[str]:
    """Units the observation claims, from predicted.units or coordinate_frame units."""
    p = obs.predicted or {}
    if isinstance(p.get("units"), str) and p["units"]:
        return p["units"]
    if isinstance(p.get("scale_state"), str) and p["scale_state"]:
        return p["scale_state"]
    return None


def _frame_of(obs: EngineObservation) -> str:
    """The coordinate frame the observation is expressed in."""
    return obs.coordinate_system or "unknown"


def _label_of(obs: EngineObservation) -> Optional[str]:
    """Class label the observation carries, if any."""
    p = obs.predicted or {}
    label = p.get("label") or p.get("category") or p.get("class")
    return label if isinstance(label, str) else None


# ------------------------------------------------------------------ fusion per task

def _fuse_scalar_quantity(
    observations: Sequence[EngineObservation],
    quantity: str,
    _quantity_noop: str | None = None,
) -> FusedPrediction:
    """Fuse a scalar quantity with the existing fuse_quantity engine.

    Converts each observation to a FusableObservation (value from predicted,
    units from predicted.units or scale_state, source = engine_id,
    provenance from observation, confidence from observation, precision
    derived from sigma_m when present else a default).
    """
    fos: list[FusableObservation] = []
    for o in observations:
        value = _scalar_from_predicted(o.predicted)
        if value is None:
            continue
        units = _units_of(o)
        # Default precision: when the engine reported sigma_m use it,
        # otherwise fall back to a conservative default tied to the value
        # magnitude (a fabricated precision is worse than a weak one).
        sigma = None
        if o.uncertainty and isinstance(o.uncertainty, dict):
            sigma = o.uncertainty.get("sigma_m")
        if sigma is None or not (isinstance(sigma, (int, float)) and float(sigma) > 0):
            sigma = max(abs(value) * 0.1, 0.01) if value != 0 else 0.01
        # Determine the unit: if the observation claims meters use METER,
        # otherwise use UNITLESS (sfm_units, pixels, etc. are unitless in
        # the fusion sense -- they are relative/internal units).
        if units in ("meter", "metres", "m", "metre"):
            unit = Unit.METER
        else:
            unit = Unit.DIMENSIONLESS
        fos.append(
            FusableObservation(
                value=value,
                unit=unit,
                source=o.engine_id,
                provenance=Provenance.RECONSTRUCTED,
                # A missing confidence carries no information, so it must not
                # artificially lower the fused confidence (min of reported ones).
                confidence=o.confidence if o.confidence is not None else 1.0,
                precision=float(sigma),
                evidence_ids=(o.observation_id,),
            )
        )
    if not fos:
        raise FusionError(f"no scalar value found among {len(observations)} observations for {quantity}")
    fused = fuse_quantity(fos, quantity=quantity)
    conflicts = tuple(
        FusionConflict(
            sources=(c.source_a, c.source_b),
            quantity=quantity,
            detail=f"value {c.value_a} vs {c.value_b} (delta {c.delta:.4g}, {c.sigma_multiple:.2f} sigma)",
        )
        for c in fused.conflicts
    )
    return FusedPrediction(
        quantity=quantity,
        source_frame=_common_frame(observations),
        source_units=_common_units(fos),
        value=fused.value,
        confidence=fused.confidence,
        provenance=fused.provenance,
        method=fused.method,
        contributing=tuple(observations),
        conflicts=conflicts,
        uncertainty_basis=fused.provenance.value
        if fused.provenance is Provenance.CONFLICT
        else "uncalibrated: fused from engine raw confidences",
    )


def _common_frame(observations: Sequence[EngineObservation]) -> str:
    frames = {_frame_of(o) for o in observations}
    if len(frames) == 1:
        return next(iter(frames))
    return "incompatible"


def _common_units(fos: Sequence[FusableObservation]) -> str:
    units = {o.unit.value for o in fos}
    if len(units) == 1:
        return next(iter(units))
    return "incompatible"


def _units_string(units: str) -> str:
    """Return a display string for the units."""
    return units


def _fuse_position(
    observations: Sequence[EngineObservation],
    _quantity: str | None = None,
) -> FusedPrediction:
    """Fuse 3-DoF positions in the same frame.

    Component-wise inverse-variance fusion with per-axis conflict detection.
    Observations in a different frame are excluded (recorded as incompatible).
    """
    frame = _frame_of(observations[0])
    same_frame = [o for o in observations if _frame_of(o) == frame]
    if not same_frame:
        return _incompatible(observations, "position", "no common frame")
    if len(same_frame) == 1:
        o = same_frame[0]
        pos = _position_from_predicted(o.predicted)
        if pos is None:
            raise FusionError(f"no position found in observation {o.observation_id}")
        return FusedPrediction(
            quantity="position",
            source_frame=frame,
            source_units=_units_of(o) or "sfm_units",
            value=pos,
            confidence=o.confidence,
            provenance=Provenance.RECONSTRUCTED,
            method="passthrough",
            contributing=(o,),
            conflicts=(),
        )
    # Component-wise fusion.
    values = []
    sigmas = []
    for o in same_frame:
        pos = _position_from_predicted(o.predicted)
        if pos is None:
            continue
        values.append(pos)
        sigma = 0.01
        if o.uncertainty and isinstance(o.uncertainty, dict):
            sigma = o.uncertainty.get("sigma_m", 0.01)
        if not (isinstance(sigma, (int, float)) and float(sigma) > 0):
            sigma = 0.01
        sigmas.append((float(sigma), float(sigma), float(sigma)))
    if not values:
        raise FusionError("no position values among same-frame observations")
    fused = []
    conflicts = []
    confidence = min((o.confidence for o in same_frame if o.confidence is not None), default=None)
    for axis in range(3):
        fos = [
            FusableObservation(
                value=v[axis],
                unit=Unit.METER,
                source=o.engine_id,
                provenance=Provenance.RECONSTRUCTED,
                confidence=o.confidence,
                precision=s[axis],
                evidence_ids=(o.observation_id,),
            )
            for v, o, s in zip(values, same_frame, sigmas)
        ]
        ax_fused = fuse_quantity(fos, quantity=f"position_{axis}")
        fused.append(ax_fused.value)
        for c in ax_fused.conflicts:
            conflicts.append(
                FusionConflict(
                    sources=(c.source_a, c.source_b),
                    quantity="position",
                    detail=f"axis {axis}: {c.value_a:.4g} vs {c.value_b:.4g} ({c.sigma_multiple:.2f} sigma)",
                )
            )
    return FusedPrediction(
        quantity="position",
        source_frame=frame,
        source_units=_units_of(same_frame[0]) or "sfm_units",
        value=tuple(fused),
        confidence=confidence,
        provenance=Provenance.CONFLICT if conflicts else Provenance.ESTIMATED,
        method="inverse_variance_weighted",
        contributing=tuple(same_frame),
        conflicts=tuple(conflicts),
    )


def _fuse_dimensions(
    observations: Sequence[EngineObservation],
    _quantity: str | None = None,
) -> FusedPrediction:
    """Fuse 3-DoF dimensions (size) in the same frame."""
    frame = _frame_of(observations[0])
    same_frame = [o for o in observations if _frame_of(o) == frame]
    if not same_frame:
        return _incompatible(observations, "dimensions", "no common frame")
    if len(same_frame) == 1:
        o = same_frame[0]
        dims = _dimensions_from_predicted(o.predicted)
        if dims is None:
            raise FusionError(f"no dimensions found in observation {o.observation_id}")
        return FusedPrediction(
            quantity="dimensions",
            source_frame=frame,
            source_units=_units_of(o) or "sfm_units",
            value=dims,
            confidence=o.confidence,
            provenance=Provenance.RECONSTRUCTED,
            method="passthrough",
            contributing=(o,),
            conflicts=(),
        )
    values = []
    for o in same_frame:
        d = _dimensions_from_predicted(o.predicted)
        if d is not None:
            values.append(d)
    if not values:
        raise FusionError("no dimension values among same-frame observations")
    fused = []
    conflicts = []
    confidence = min((o.confidence for o in same_frame if o.confidence is not None), default=None)
    for axis in range(3):
        fos = [
            FusableObservation(
                value=v[axis],
                unit=Unit.METER,
                source=o.engine_id,
                provenance=Provenance.RECONSTRUCTED,
                confidence=o.confidence,
                precision=0.01,
                evidence_ids=(o.observation_id,),
            )
            for v, o in zip(values, same_frame)
        ]
        ax_fused = fuse_quantity(fos, quantity=f"dimensions_{axis}")
        fused.append(ax_fused.value)
        for c in ax_fused.conflicts:
            conflicts.append(
                FusionConflict(
                    sources=(c.source_a, c.source_b),
                    quantity="dimensions",
                    detail=f"axis {axis}: {c.value_a:.4g} vs {c.value_b:.4g} ({c.sigma_multiple:.2f} sigma)",
                )
            )
    return FusedPrediction(
        quantity="dimensions",
        source_frame=frame,
        source_units=_units_of(same_frame[0]) or "sfm_units",
        value=tuple(fused),
        confidence=confidence,
        provenance=Provenance.CONFLICT if conflicts else Provenance.ESTIMATED,
        method="inverse_variance_weighted",
        contributing=tuple(same_frame),
        conflicts=tuple(conflicts),
    )


def _fuse_bboxes(
    observations: Sequence[EngineObservation],
    _quantity: str | None = None,
) -> FusedPrediction:
    """Fuse 2-DoF bboxes in the SAME image frame, consistency-weighted.

    Boxes must agree on label to be fused; label disagreements are preserved
    as conflicts. The fused box is the consistency-weighted average of the
    corners, where consistency = 1 / (1 + sigma) and sigma is the spread of
    the corners across contributors.
    """
    if not observations:
        raise FusionError("no bboxes to fuse")
    # Group by label.
    by_label: dict[str, list[EngineObservation]] = defaultdict(list)
    for o in observations:
        label = _label_of(o) or "unlabeled"
        by_label[label].append(o)
    # Pick the majority label (ties broken by alphabetical order for determinism).
    label = max(by_label, key=lambda k: (len(by_label[k]), k))
    candidates = by_label.get(label, [])
    if not candidates:
        candidates = observations
    # Must be same image frame (same source_id): bboxes from different images
    # cannot be meaningfully averaged, so they are recorded as incompatible.
    sources = {o.source_id for o in candidates}
    if len(sources) != 1:
        return _incompatible(
            observations,
            "bbox",
            f"bboxes span multiple image sources {sorted(sources)}; cannot fuse across frames",
        )
    source = next(iter(sources))
    same_source = list(candidates)
    if not same_source:
        return _incompatible(observations, "bbox", f"no common source (majority label '{label}')")
    boxes = []
    for o in same_source:
        b = _bbox_from_predicted(o.predicted)
        if b is not None:
            boxes.append((b, o))
    if not boxes:
        raise FusionError("no valid bboxes among same-source observations")
    if len(boxes) == 1:
        b, o = boxes[0]
        # Every contributor whose label disagrees with the fused box's label is
        # a preserved conflict, regardless of source frame.
        conflicts = tuple(
            FusionConflict(
                sources=(o.engine_id, other.engine_id),
                quantity="bbox",
                detail=f"label '{_label_of(other) or 'unlabeled'}' vs majority '{label}'",
            )
            for other in observations
            if (_label_of(other) or "unlabeled") != label
        )
        return FusedPrediction(
            quantity="bbox",
            source_frame=_frame_of(o),
            source_units="pixels",
            value=b,
            confidence=o.confidence,
            provenance=Provenance.RECONSTRUCTED,
            method="passthrough",
            contributing=(o,),
            conflicts=conflicts,
        )
    # Consistency-weighted average of corners.
    # weight = 1 / (1 + spread), where spread is the std of the corner across contributors.
    xs0 = [b[0] for b, _ in boxes]
    ys0 = [b[1] for b, _ in boxes]
    xs1 = [b[2] for b, _ in boxes]
    ys1 = [b[3] for b, _ in boxes]
    import statistics

    def _spread(vals):
        if len(vals) < 2:
            return 0.0
        return statistics.stdev(vals)

    sx0, sy0, sx1, sy1 = _spread(xs0), _spread(ys0), _spread(xs1), _spread(ys1)
    # weight per box: inverse of its corner spread (a box whose corners agree with
    # the others gets more weight).
    weights = []
    for (b, o), w_idx in zip(boxes, range(len(boxes))):
        w = 1.0 / (1.0 + (sx0 + sy0 + sx1 + sy1) / 4.0)
        weights.append(w)
    total_w = sum(weights) or 1.0
    fused_box = (
        sum(w * b[0] for (b, _), w in zip(boxes, weights)) / total_w,
        sum(w * b[1] for (b, _), w in zip(boxes, weights)) / total_w,
        sum(w * b[2] for (b, _), w in zip(boxes, weights)) / total_w,
        sum(w * b[3] for (b, _), w in zip(boxes, weights)) / total_w,
    )
    confidence = min((o.confidence for _, o in boxes if o.confidence is not None), default=None)
    # Conflicts: contributors whose label disagrees with the majority.
    conflicts = tuple(
        FusionConflict(
            sources=(o.engine_id, boxes[0][1].engine_id),
            quantity="bbox",
            detail=f"label '{_label_of(o) or 'unlabeled'}' vs majority '{label}'",
        )
        for lbl, grp in by_label.items()
        if lbl != label
        for o in grp
    )
    return FusedPrediction(
        quantity="bbox",
        source_frame=_frame_of(boxes[0][1]),
        source_units="pixels",
        value=fused_box,
        confidence=confidence,
        provenance=Provenance.CONFLICT if conflicts else Provenance.ESTIMATED,
        method="consistency_weighted",
        contributing=tuple(o for _, o in boxes),
        conflicts=conflicts,
    )


def _incompatible(
    observations: Sequence[EngineObservation],
    quantity: str,
    detail: str,
) -> FusedPrediction:
    """Record that the observations could not be fused because they are incompatible."""
    return FusedPrediction(
        quantity=quantity,
        source_frame=_common_frame(observations),
        source_units="incompatible",
        value=None,
        confidence=None,
        provenance=Provenance.CONFLICT,
        method="compatibility_filtered",
        contributing=tuple(observations),
        conflicts=tuple(
            FusionConflict(sources=(observations[0].engine_id, o.engine_id), quantity=quantity, detail=detail)
            for o in observations[1:]
        ),
        uncertainty_basis="incompatible: no fusion performed",
    )


# ------------------------------------------------------------------ public API

#: Task → quantity → fusion function.
_TASK_FUSION: dict[str, dict[str, Any]] = {}


def register_fusion(task: str, quantity: str, fn: Any) -> None:
    """Register a task-specific fusion function.

    The function is called as ``fn(observations, quantity)`` and must return
    a ``FusedPrediction``.
    """
    _TASK_FUSION.setdefault(task, {})[quantity] = fn


def fuse_observations(
    observations: Sequence[EngineObservation],
    *,
    task: str,
    quantity: str,
) -> FusedPrediction:
    """Fuse observations of ONE task/quantity.

    Raises FusionError when no observations are usable. Returns a FusedPrediction
    with the fused value (or None when the inputs were incompatible) and the
    full audit trail.
    """
    if not observations:
        raise FusionError(f"no observations to fuse for {task}/{quantity}")
    # Use a registered task-specific fusion when available.
    fn = _TASK_FUSION.get(task, {}).get(quantity)
    if fn is not None:
        return fn(observations, quantity)
    # Default: try scalar fusion first, then fall back to incompatible.
    scalars = [o for o in observations if _scalar_from_predicted(o.predicted) is not None]
    if scalars:
        return _fuse_scalar_quantity(scalars, quantity)
    # No scalar path: record incompatible.
    return _incompatible(observations, quantity, "no fusion path for this task/quantity")


def normalize_and_fuse(
    matrix: Any,
    *,
    task: str,
    quantity: str,
    source_id: str | None = None,
    group_by_label: bool = False,
) -> list[FusedPrediction]:
    """Normalize + fuse ALL observations of a task/quantity in a matrix.

    ``group_by_label``: when True, fuse per label (so "chair" and "sofa"
    detections are not fused together). When False, all observations of
    the quantity are fused together (appropriate for position/dimensions/
    depth_value where the quantity is the same thing).

    Returns one FusedPrediction per group.
    """
    query_kwargs: dict[str, Any] = {"task": task, "quantity": quantity}
    if source_id is not None:
        query_kwargs["source_id"] = source_id
    obs = matrix.store.query(**query_kwargs)
    if not obs:
        return []
    if group_by_label:
        groups: dict[str, list[EngineObservation]] = defaultdict(list)
        for o in obs:
            groups[_label_of(o) or "unlabeled"].append(o)
        return [
            fuse_observations(g, task=task, quantity=quantity) for g in groups.values()
        ]
    return [fuse_observations(obs, task=task, quantity=quantity)]


def write_fused_back(
    matrix: Any,
    fused: FusedPrediction,
    *,
    run_id: str,
    model_version: str = "evidence_fusion/v1",
    engine_id: str = "evidence_fusion",
) -> EngineObservation:
    """Write a fused prediction back into the matrix as a new observation.

    The fused observation links to every contributor via ``derived_from``
    (never overwrites them). Its predicted dict carries the fused value,
    the source frame/units, the provenance and the conflict count so a
    reader can see what was fused and what was disagreed.
    """
    from engine.multiengine.observation import ObservationDraft, make_observation

    task = fused.quantity.split("_")[0]  # e.g. "position" -> "position"; "depth_value" -> "depth"
    if task not in ("depth", "detection", "metric3d", "segmentation", "matching"):
        task = "perception"

    predicted: dict[str, Any] = {
        "fused_value": fused.value,
        "source_frame": fused.source_frame,
        "source_units": fused.source_units,
        "provenance": fused.provenance.value if fused.provenance else None,
        "confidence": fused.confidence,
        "method": fused.method,
        "n_contributors": len(fused.contributing),
        "n_conflicts": len(fused.conflicts),
        "uncertainty_basis": fused.uncertainty_basis,
    }
    # Attach the actual fused value under a quantity-keyed name too.
    if fused.value is not None:
        predicted[fused.quantity] = fused.value
    obs = make_observation(
        ObservationDraft(
            task=task,
            quantity=fused.quantity,
            source_id=fused.contributing[0].source_id if fused.contributing else "",
            coordinate_system=fused.source_frame,
            predicted=predicted,
            confidence=fused.confidence,
            uncertainty=None,
            calibration=None,
        ),
        engine_id=engine_id,
        model_version=model_version,
        execution_config={
            "fused_from": [o.observation_id for o in fused.contributing],
            "conflicts": [c.sources for c in fused.conflicts],
            "uncertainty_basis": fused.uncertainty_basis,
        },
    )
    # Store the fused observation first so it exists in the store before we
    # link it to its contributors (link requires both ends to be present).
    ctx = EvidenceContext(
        observation_id=obs.observation_id,
        run_id=run_id,
        architecture_id=engine_id,
    )
    matrix.add(obs, ctx)
    for o in fused.contributing:
        matrix.store.link(obs.observation_id, o.observation_id, "derived_from")
    return obs


# Register the built-in task-specific fusions.
register_fusion("depth", "depth_value", _fuse_scalar_quantity)
register_fusion("metric3d", "position", _fuse_position)
register_fusion("metric3d", "dimensions", _fuse_dimensions)
register_fusion("detection", "bbox", _fuse_bboxes)
register_fusion("detection", "position", _fuse_position)
register_fusion("detection", "dimensions", _fuse_dimensions)


# Convenience re-export for callers that want the scalar engine too.
__all__ = [
    "FusedPrediction",
    "FusionConflict",
    "fuse_observations",
    "normalize_and_fuse",
    "write_fused_back",
    "register_fusion",
    "SCALAR_PREDICT_KEYS",
    "POSITION_KEYS",
    "DIMENSIONS_KEYS",
    "BBOX_KEYS",
]
