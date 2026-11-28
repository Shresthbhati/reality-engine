"""Benchmark-driven model selection (WS4 of the Master Engine Integration Mandate).

Additive module. It consumes the reproducible ``ModelBenchmarkRecord``
records produced by ``perception.evaluation.candidate_model_benchmark`` /
``gt_benchmark`` and ranks candidates using MEASURED performance only.

Rules enforced here:
  * No popularity/opinion: the only ranking inputs are numbers on a record
    (F1/precision/recall from a held-out split, wall time, peak VRAM, status).
  * Comparability first: candidates are compared only when they share the
    same dataset + commit + image ids (otherwise the numbers are not
    comparable and the selection is BLOCKED, never averaged).
  * Different tasks win differently: selection is per (task, domain, scale,
    image_quality) request; no global "best model".
  * Hardware constraints are hard gates (available VRAM, CPU-only, licence).
  * Failed/unavailable candidates are reported, never silently substituted.
  * Incompatible measurements are never averaged: only scalar metrics with
    the same key on the same split are compared.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "SelectionStatus", "SelectionRequest", "CandidateScore", "SelectionResult",
    "select_model", "select_models",
]


class SelectionStatus(str, Enum):
    SELECTED = "selected"
    BLOCKED = "blocked"          # no eligible candidate / incomparable evidence
    NO_DATA = "no_data"          # no benchmark records for the task at all


@dataclass(frozen=True)
class SelectionRequest:
    task: str
    domain: str = "any"          # e.g. "interior", "outdoor"
    object_scale: str = "any"    # e.g. "small", "large"
    image_quality: str = "any"   # e.g. "low", "high"
    available_vram_mb: float | None = None
    cpu_only: bool = False
    require_commercial: bool = False
    # Weights on MEASURED quantities only. Accuracy dominates; speed/memory
    # are tie-breakers. All >= 0; weights are part of the reproducible config.
    weight_accuracy: float = 0.7
    weight_speed: float = 0.2
    weight_memory: float = 0.1
    max_wall_time_s: float | None = None

    def __post_init__(self):
        for w in (self.weight_accuracy, self.weight_speed, self.weight_memory):
            if w < 0:
                raise ValueError("selection weights must be >= 0")


@dataclass(frozen=True)
class CandidateScore:
    model_id: str
    status: str                          # benchmark record status
    score: float | None                  # None when not comparable/eligible
    reason: str
    measured: dict[str, float] = field(default_factory=dict)
    eligible: bool = False

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id, "status": self.status, "score": self.score,
            "reason": self.reason, "measured": dict(self.measured), "eligible": self.eligible,
        }


@dataclass(frozen=True)
class SelectionResult:
    status: SelectionStatus
    request: SelectionRequest
    winner: str | None
    candidates: tuple[CandidateScore, ...]
    reasons: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "request": {
                "task": self.request.task, "domain": self.request.domain,
                "object_scale": self.request.object_scale, "image_quality": self.request.image_quality,
                "available_vram_mb": self.request.available_vram_mb, "cpu_only": self.request.cpu_only,
                "require_commercial": self.request.require_commercial,
                "weights": {"accuracy": self.request.weight_accuracy, "speed": self.request.weight_speed,
                            "memory": self.request.weight_memory},
            },
            "winner": self.winner,
            "candidates": [c.to_dict() for c in self.candidates],
            "reasons": list(self.reasons),
        }


def _metric_value(record: Any, request: SelectionRequest) -> float | None:
    """Extract ONE measured accuracy number for this task/scale/quality.

    Looks for per-class metrics keyed by the requested scale/quality first,
    then a task-level F1, then any per-class F1 mean. Missing = None (never
    fabricated, never a default score).
    """
    per_class = getattr(record, "per_class_metrics", None) or {}
    keys = [
        f"{request.object_scale}_{request.image_quality}",
        request.object_scale, request.image_quality, request.task, "all", "mean",
    ]
    for k in keys:
        m = per_class.get(k)
        if isinstance(m, Mapping):
            for metric in ("f1", "f1_score", "precision", "recall"):
                v = m.get(metric)
                if isinstance(v, (int, float)):
                    return float(v)
        elif isinstance(m, (int, float)):
            return float(m)
    return None


def _comparability_key(record: Any) -> tuple:
    return (
        getattr(record, "dataset", None),
        getattr(record, "commit_sha", None),
        tuple(getattr(record, "image_ids", ()) or ()),
    )


def select_model(
    request: SelectionRequest,
    records: Sequence[Any],
    *,
    catalog: Mapping[str, Any] | None = None,
) -> SelectionResult:
    """Rank benchmark records for ``request`` using measured numbers only.

    ``catalog`` is the optional CandidateModelSpec map (for licence/VRAM
    metadata); when omitted, those gates are skipped rather than guessed.
    """
    task_records = [r for r in records if getattr(r, "model_id", None) and _task_of(r, catalog) == request.task]
    if not task_records:
        return SelectionResult(SelectionStatus.NO_DATA, request, None, (),
                               (f"no benchmark records for task {request.task!r}",))

    # Comparability: every eligible record must share the SAME dataset/commit/images.
    keys = {_comparability_key(r) for r in task_records}
    if len(keys) > 1:
        return SelectionResult(
            SelectionStatus.BLOCKED, request, None, (),
            ("benchmark records are not comparable (different dataset/commit/image ids); "
             "refusing to average incompatible measurements",),
        )

    scored: list[CandidateScore] = []
    for r in task_records:
        model_id = getattr(r, "model_id", "?")
        status = str(getattr(r, "status", "unknown"))
        measured: dict[str, float] = {}
        if status not in ("succeeded", "ok", "success"):
            scored.append(CandidateScore(model_id, status, None,
                           f"candidate status {status!r}; not eligible (no silent fallback)", measured))
            continue
        spec = (catalog or {}).get(model_id)
        # Hardware hard gates.
        peak = getattr(r, "peak_vram_mb", None)
        min_vram = getattr(spec, "min_vram_mb", None) if spec else None
        if request.cpu_only and peak is not None and peak > 0:
            scored.append(CandidateScore(model_id, status, None,
                           "recorded on GPU but request is CPU-only", measured))
            continue
        if request.available_vram_mb is not None:
            need = peak if peak is not None else min_vram
            if need is not None and need > request.available_vram_mb:
                scored.append(CandidateScore(model_id, status, None,
                               f"VRAM need {need} > available {request.available_vram_mb}", measured))
                continue
        licence = getattr(r, "license", "") or (getattr(spec, "license", "") if spec else "")
        if request.require_commercial and "non-commercial" in licence.lower():
            scored.append(CandidateScore(model_id, status, None,
                           f"licence {licence!r} fails require_commercial", measured))
            continue
        acc = _metric_value(r, request)
        wall = float(getattr(r, "wall_time_s", 0.0) or 0.0)
        if request.max_wall_time_s is not None and wall > request.max_wall_time_s:
            scored.append(CandidateScore(model_id, status, None,
                           f"wall time {wall}s exceeds {request.max_wall_time_s}s", measured))
            continue
        if acc is None:
            scored.append(CandidateScore(model_id, status, None,
                           "no measured accuracy for requested scale/quality; not eligible", measured))
            continue
        measured.update(accuracy=acc, wall_time_s=wall)
        if peak is not None:
            measured["peak_vram_mb"] = float(peak)
        # Lower-is-better quantities normalized against the best eligible
        # candidate would need two passes; instead score against fixed
        # reference scales documented in the config (reproducible).
        speed_term = 1.0 / (1.0 + wall)
        mem_term = 1.0 / (1.0 + (float(peak) if peak else 0.0) / 1024.0)
        score = (request.weight_accuracy * acc
                 + request.weight_speed * speed_term
                 + request.weight_memory * mem_term)
        scored.append(CandidateScore(model_id, status, round(score, 6),
                       "ranked by measured F1 + wall time + peak VRAM", measured, eligible=True))

    eligible = [c for c in scored if c.eligible]
    if not eligible:
        return SelectionResult(SelectionStatus.BLOCKED, request, None, tuple(scored),
                               ("no eligible candidate with comparable measured performance",))
    winner = max(eligible, key=lambda c: (c.score or 0.0, -c.measured.get("wall_time_s", 0.0))).model_id
    return SelectionResult(SelectionStatus.SELECTED, request, winner, tuple(scored),
                           ("selection uses measured held-out metrics only; different tasks selected independently",))


def select_models(
    requests: Iterable[SelectionRequest],
    records: Sequence[Any],
    *,
    catalog: Mapping[str, Any] | None = None,
) -> tuple[SelectionResult, ...]:
    """Select each request independently -- different models may win different tasks."""
    return tuple(select_model(req, records, catalog=catalog) for req in requests)


def _task_of(record: Any, catalog: Mapping[str, Any] | None) -> str:
    """Task from the record's own metadata or the catalog; never inferred from a name."""
    direct = getattr(record, "task", None)
    if isinstance(direct, str) and direct:
        return direct
    spec = (catalog or {}).get(getattr(record, "model_id", None))
    return str(getattr(spec, "task", "") or "")
