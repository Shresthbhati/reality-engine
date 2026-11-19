"""Quality gates that prevent false success.

A gate is a named measurement with a direction and a threshold. Two rules are enforced in code:

* A threshold is DATA, supplied by the caller from a documented benchmark. There are no built-in numbers: a gate without a
  threshold can only ever be UNVERIFIED ("no benchmark-established threshold"), never PASS.
* A missing measurement is UNVERIFIED, never PASS. The overall result is COMPLETE only when every required gate was measured and
  passed; a reconstruction whose pipeline stages all returned without exceptions is not complete on that basis.

Overall results: COMPLETE (all required gates pass), FAILED (a required gate failed or a regression was found), PARTIAL (some
required gates pass, the rest could not be verified), UNVERIFIED (no required gate could be verified).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple


class GateVerdict(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    UNVERIFIED = "unverified"


class Overall(str, Enum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    UNVERIFIED = "unverified"
    FAILED = "failed"


#: The metrics the task requires tracking. Names only -- thresholds come from benchmarks, not from this file.
STANDARD_GATES: Tuple[Tuple[str, str, str], ...] = (
    ("camera_pose_accuracy", "lower_is_better", "m"),
    ("reprojection_error", "lower_is_better", "px"),
    ("depth_consistency", "lower_is_better", ""),
    ("pointcloud_accuracy", "lower_is_better", "m"),
    ("pointcloud_completeness", "higher_is_better", "fraction"),
    ("mesh_accuracy", "lower_is_better", "m"),
    ("metric_scale_confidence", "higher_is_better", ""),
    ("surface_coverage", "higher_is_better", "fraction"),
    ("detection_precision", "higher_is_better", "fraction"),
    ("detection_recall", "higher_is_better", "fraction"),
    ("segmentation_quality", "higher_is_better", ""),
    ("identity_consistency", "higher_is_better", "fraction"),
    ("spatial_relationship_accuracy", "higher_is_better", "fraction"),
    ("uncertainty_calibration", "lower_is_better", ""),
    ("missing_region_coverage", "higher_is_better", "fraction"),
)


@dataclass(frozen=True)
class GateSpec:
    name: str
    direction: str  # "lower_is_better" | "higher_is_better"
    threshold: float | None = None  # None = no benchmark-established threshold exists yet
    threshold_source: str = (
        ""  # the benchmark record the threshold came from; required when threshold is set
    )
    required: bool = True
    unit: str = ""

    def __post_init__(self):
        if self.direction not in ("lower_is_better", "higher_is_better"):
            raise ValueError(f"gate {self.name!r}: unknown direction {self.direction!r}")
        if self.threshold is not None and not self.threshold_source:
            raise ValueError(
                f"gate {self.name!r}: a threshold must cite the benchmark it was established from"
            )


@dataclass(frozen=True)
class Measurement:
    value: float
    method: str  # how it was measured (e.g. "reprojection over registered views")
    n: int | None = None  # sample size, if applicable
    source: str = ""  # run / artifact the measurement came from


@dataclass(frozen=True)
class GateResult:
    name: str
    verdict: GateVerdict
    reason: str
    value: float | None = None
    threshold: float | None = None
    required: bool = True


@dataclass(frozen=True)
class Regression:
    feature_id: str
    kind: str  # "missing" | "changed"
    previous: object
    current: object


QualityGateResult = GateResult


@dataclass(frozen=True)
class QualityReport:
    overall: Overall
    gates: Tuple[GateResult, ...]
    regressions: Tuple[Regression, ...]
    stages_ok: bool | None
    notes: Tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "overall": self.overall.value,
            "stages_ok": self.stages_ok,
            "notes": list(self.notes),
            "gates": [
                {
                    "name": g.name,
                    "verdict": g.verdict.value,
                    "reason": g.reason,
                    "value": g.value,
                    "threshold": g.threshold,
                    "required": g.required,
                }
                for g in self.gates
            ],
            "regressions": [
                {
                    "feature_id": r.feature_id,
                    "kind": r.kind,
                    "previous": r.previous,
                    "current": r.current,
                }
                for r in self.regressions
            ],
        }


def default_gate_specs(
    thresholds: Mapping[str, Tuple[float, str]] | None = None,
    *,
    required: Iterable[str] | None = None,
) -> List[GateSpec]:
    """The standard gates. ``thresholds`` maps gate name -> (threshold, benchmark record). Anything not given stays threshold-less
    (UNVERIFIED). ``required`` limits which gates are required (default: all)."""
    t = dict(thresholds or {})
    unknown = set(t) - {n for n, _, _ in STANDARD_GATES}
    if unknown:
        raise ValueError(f"thresholds given for unknown gates: {sorted(unknown)}")
    req = None if required is None else set(required)
    out = []
    for name, direction, unit in STANDARD_GATES:
        thr, src = t.get(name, (None, ""))
        out.append(
            GateSpec(name, direction, thr, src, required=(req is None or name in req), unit=unit)
        )
    return out


def check_regressions(
    previous: Mapping[str, object], current: Mapping[str, object], *, tolerance: Mapping[str, float]
) -> List[Regression]:
    """Features validated earlier must not silently vanish or change. ``tolerance`` maps feature id -> allowed absolute change for
    numeric values; a feature with no stated tolerance must be exactly equal (no default is invented)."""
    out: List[Regression] = []
    for fid, prev in sorted(previous.items()):
        if fid not in current:
            out.append(Regression(fid, "missing", prev, None))
            continue
        cur = current[fid]
        tol = tolerance.get(fid)
        numeric = (
            isinstance(prev, (int, float))
            and isinstance(cur, (int, float))
            and not isinstance(prev, bool)
            and not isinstance(cur, bool)
        )
        if numeric and tol is not None:
            if abs(float(cur) - float(prev)) > tol:
                out.append(Regression(fid, "changed", prev, cur))
        elif cur != prev:
            out.append(Regression(fid, "changed", prev, cur))
    return out


def evaluate(
    specs: Sequence[GateSpec],
    measurements: Mapping[str, Measurement | None],
    *,
    stages_ok: bool | None = None,
    regressions: Sequence[Regression] = (),
) -> QualityReport:
    results: List[GateResult] = []
    for s in specs:
        m = measurements.get(s.name)
        if m is None:
            results.append(
                GateResult(
                    s.name, GateVerdict.UNVERIFIED, "not measured", None, s.threshold, s.required
                )
            )
        elif s.threshold is None:
            results.append(
                GateResult(
                    s.name,
                    GateVerdict.UNVERIFIED,
                    "measured, but no benchmark-established threshold exists to judge it against",
                    m.value,
                    None,
                    s.required,
                )
            )
        else:
            ok = (
                m.value <= s.threshold
                if s.direction == "lower_is_better"
                else m.value >= s.threshold
            )
            results.append(
                GateResult(
                    s.name,
                    GateVerdict.PASS if ok else GateVerdict.FAIL,
                    f"{m.value:g} vs threshold {s.threshold:g} ({s.direction}, from {s.threshold_source})",
                    m.value,
                    s.threshold,
                    s.required,
                )
            )
    required = [r for r in results if r.required]
    notes: List[str] = []
    if stages_ok is True:
        notes.append(
            "all pipeline stages returned without error; that alone does not make the reconstruction complete"
        )
    if not required:
        overall = Overall.UNVERIFIED
        notes.append("no required gate is configured")
    elif regressions or any(r.verdict is GateVerdict.FAIL for r in required) or stages_ok is False:
        overall = Overall.FAILED
        if regressions:
            notes.append(f"{len(regressions)} previously validated feature(s) regressed")
        if stages_ok is False:
            notes.append("a required pipeline stage failed")
    elif all(r.verdict is GateVerdict.PASS for r in required):
        overall = Overall.COMPLETE
    elif any(r.verdict is GateVerdict.PASS for r in required):
        overall = Overall.PARTIAL
        notes.append(
            "some required gates could not be verified: "
            + ", ".join(r.name for r in required if r.verdict is GateVerdict.UNVERIFIED)
        )
    else:
        overall = Overall.UNVERIFIED
        notes.append("no required gate could be verified")
    return QualityReport(overall, tuple(results), tuple(regressions), stages_ok, tuple(notes))


class QualityGateManager:
    """Convenience manager for evaluating quality gates across stages."""

    def __init__(self, specs: Sequence[GateSpec] | None = None) -> None:
        self.specs = tuple(specs) if specs is not None else tuple(default_gate_specs())

    def evaluate(
        self,
        measurements: Mapping[str, Measurement | float],
        *,
        stages_ok: bool = True,
    ) -> QualityReport:
        norm_meas: dict[str, Measurement] = {}
        for k, v in measurements.items():
            if isinstance(v, Measurement):
                norm_meas[k] = v
            else:
                norm_meas[k] = Measurement(value=float(v), method="pipeline_direct")
        return evaluate(self.specs, norm_meas, stages_ok=stages_ok)

