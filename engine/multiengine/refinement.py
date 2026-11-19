"""Iterative refinement: measure -> find problems -> classify cause -> reprocess ONLY what can help -> reconcile -> check global
consistency and regressions -> measure improvement -> repeat until a stated stopping condition.

The loop is orchestration; the domain steps are injected as callables (``measure``, ``detect``, ``apply``, ``consistency``,
``features``), so it carries no reconstruction logic of its own and cannot fake a result.

Rules enforced here:
* A problem whose cause is INSUFFICIENT_EVIDENCE is never "fixed" by rerunning models: it becomes a capture request (or stays
  unresolved when no measured fact justifies asking for photos).
* An engine already tried for a problem is not tried again for it; when no untried available engine can address it the problem is
  recorded as exhausted instead of looping.
* Every local update must pass the global consistency check and the regression check; otherwise it is REJECTED and the previous
  state is kept (the rejection is recorded).
* The loop stops with an explicit reason and never reports success it did not measure.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from engine.multiengine.capture_requests import CaptureRequestSpec, ProblemEvidence, request_for_problem
from engine.multiengine.contract import EngineRequest, Task
from engine.multiengine.executor import ExecutionManager, ExecutionReport
from engine.multiengine.quality_gates import (
    GateSpec, GateVerdict, Measurement, Overall, QualityReport, check_regressions, evaluate,
)
from engine.multiengine.registry import EngineRegistry


class ProblemKind(str, Enum):
    MISSING_SURFACE = "missing_surface"
    INCOMPLETE_WALL = "incomplete_wall"
    MISSING_CORNER = "missing_corner"
    INCORRECT_DIMENSION = "incorrect_dimension"
    UNCLASSIFIED_OBJECT = "unclassified_object"
    MISSED_SMALL_OBJECT = "missed_small_object"
    SEGMENTATION_ERROR = "segmentation_error"
    IDENTITY_ERROR = "identity_error"
    POOR_REGISTRATION = "poor_registration"
    INCONSISTENT_DEPTH = "inconsistent_depth"
    UNCERTAIN_MATERIAL = "uncertain_material"
    MISSING_ARCHITECTURAL_FEATURE = "missing_architectural_feature"


class ProblemCause(str, Enum):
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"    # the photos do not contain enough: only new evidence helps
    ENGINE_DISAGREEMENT = "engine_disagreement"        # engines conflict: another engine / the image evidence may resolve it
    ENGINE_MISS = "engine_miss"                        # evidence exists but the engines run so far did not extract it
    UNKNOWN = "unknown"


class StopReason(str, Enum):
    TARGET_MET = "target_met"
    MAX_ITERATIONS = "max_iterations"
    NO_IMPROVEMENT = "no_improvement"
    NO_ENGINE_AVAILABLE = "no_engine_available"        # every remaining problem has no untried available engine
    AWAITING_EVIDENCE = "awaiting_evidence"            # remaining problems need new photographs
    NOTHING_TO_REFINE = "nothing_to_refine"            # no problem found, yet the result is not verified complete


@dataclass(frozen=True)
class Problem:
    problem_id: str
    kind: ProblemKind
    cause: ProblemCause
    location: Mapping[str, Any]
    tasks: Tuple[Task, ...] = ()                       # tasks whose engines could re-examine it
    evidence: ProblemEvidence = ProblemEvidence()
    target_ids: Tuple[str, ...] = ()
    detail: str = ""
    basis: Tuple[str, ...] = ()                        # observation / measurement ids that revealed it


@dataclass(frozen=True)
class ActionRecord:
    problem_id: str
    outcome: str       # applied | unchanged | rejected_inconsistent | rejected_regression | engine_failed | exhausted | needs_evidence | unresolved
    engines: Tuple[str, ...] = ()
    detail: str = ""


@dataclass(frozen=True)
class IterationRecord:
    index: int
    quality_before: QualityReport
    problems: Tuple[Problem, ...]
    actions: Tuple[ActionRecord, ...]
    quality_after: Optional[QualityReport]
    improved: bool


@dataclass(frozen=True)
class RefinementResult:
    state: Any
    stop_reason: StopReason
    iterations: Tuple[IterationRecord, ...]
    capture_requests: Tuple[CaptureRequestSpec, ...]
    final_quality: QualityReport
    unresolved: Tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "stop_reason": self.stop_reason.value, "final_quality": self.final_quality.to_dict(),
            "unresolved": list(self.unresolved),
            "capture_requests": [{"reason": c.reason, "why_insufficient": list(c.why_insufficient), "instruction": c.instruction,
                                  "overlap_with": list(c.overlap_with), "target_ids": list(c.target_ids)}
                                 for c in self.capture_requests],
            "iterations": [{"index": i.index, "overall_before": i.quality_before.overall.value,
                            "overall_after": None if i.quality_after is None else i.quality_after.overall.value,
                            "improved": i.improved, "problems": [p.problem_id for p in i.problems],
                            "actions": [{"problem_id": a.problem_id, "outcome": a.outcome, "engines": list(a.engines),
                                         "detail": a.detail} for a in i.actions]} for i in self.iterations],
        }


_RANK = {Overall.FAILED: 0, Overall.UNVERIFIED: 1, Overall.PARTIAL: 2, Overall.COMPLETE: 3}


def _score(q: QualityReport, n_problems: int) -> Tuple[int, int, int, int]:
    passes = sum(1 for g in q.gates if g.verdict is GateVerdict.PASS)
    fails = sum(1 for g in q.gates if g.verdict is GateVerdict.FAIL)
    return (_RANK[q.overall], passes, -fails, -n_problems)


class RefinementLoop:
    def __init__(self, *, manager: ExecutionManager, registry: EngineRegistry, specs: Sequence[GateSpec],
                 measure: Callable[[Any], Tuple[Mapping[str, Optional[Measurement]], Optional[bool]]],
                 detect: Callable[[Any, QualityReport], Sequence[Problem]],
                 apply: Callable[[Any, Problem, ExecutionReport], Any],
                 consistency: Callable[[Any], Sequence[str]],
                 features: Callable[[Any], Mapping[str, object]] = lambda s: {},
                 regression_tolerance: Optional[Mapping[str, float]] = None,
                 max_iterations: int = 5, patience: int = 1) -> None:
        if max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")
        self.manager, self.registry, self.specs = manager, registry, list(specs)
        self.measure, self.detect, self.apply, self.consistency, self.features = measure, detect, apply, consistency, features
        self.regression_tolerance = dict(regression_tolerance or {})
        self.max_iterations, self.patience = max_iterations, max(1, patience)

    def _quality(self, state: Any, validated: Mapping[str, object]) -> QualityReport:
        measurements, stages_ok = self.measure(state)
        regs = check_regressions(validated, self.features(state), tolerance=self.regression_tolerance) if validated else []
        return evaluate(self.specs, measurements, stages_ok=stages_ok, regressions=regs)

    def run(self, state: Any) -> RefinementResult:
        validated: Mapping[str, object] = dict(self.features(state))
        tried: Dict[str, Set[str]] = {}
        exhausted: Set[str] = set()
        capture: Dict[str, CaptureRequestSpec] = {}
        history: List[IterationRecord] = []
        stagnant = 0
        quality = self._quality(state, validated)
        problems: List[Problem] = list(self.detect(state, quality))
        for index in range(1, self.max_iterations + 1):
            if quality.overall is Overall.COMPLETE and not problems:
                return self._done(state, StopReason.TARGET_MET, history, capture, quality, problems)
            if not problems:
                return self._done(state, StopReason.NOTHING_TO_REFINE, history, capture, quality, problems)
            actions: List[ActionRecord] = []
            for p in problems:
                if p.problem_id in exhausted:
                    continue
                if p.cause in (ProblemCause.INSUFFICIENT_EVIDENCE, ProblemCause.UNKNOWN):
                    spec = request_for_problem(problem_kind=p.kind.value, location=dict(p.location), target_ids=p.target_ids,
                                               evidence=p.evidence, basis=p.basis)
                    exhausted.add(p.problem_id)
                    if spec is None:
                        actions.append(ActionRecord(p.problem_id, "unresolved", (), "no measured fact shows that more photos would help"))
                    else:
                        capture.setdefault(p.problem_id, spec)
                        actions.append(ActionRecord(p.problem_id, "needs_evidence", (), spec.reason))
                    continue
                state, record = self._reprocess(state, p, tried, validated)
                actions.append(record)
                if record.outcome == "exhausted":
                    exhausted.add(p.problem_id)
            after = self._quality(state, validated)
            new_problems = list(self.detect(state, after))
            improved = _score(after, len(new_problems)) > _score(quality, len(problems))
            history.append(IterationRecord(index, quality, tuple(problems), tuple(actions), after, improved))
            quality, problems = after, new_problems
            stagnant = 0 if improved else stagnant + 1
            open_problems = [p for p in problems if p.problem_id not in exhausted]
            if quality.overall is Overall.COMPLETE and not problems:
                return self._done(state, StopReason.TARGET_MET, history, capture, quality, problems)
            if not open_problems:
                reason = StopReason.AWAITING_EVIDENCE if capture else StopReason.NO_ENGINE_AVAILABLE
                return self._done(state, reason, history, capture, quality, problems)
            if stagnant >= self.patience:
                return self._done(state, StopReason.NO_IMPROVEMENT, history, capture, quality, problems)
        return self._done(state, StopReason.MAX_ITERATIONS, history, capture, quality, problems)

    # ------------------------------------------------------------------ one targeted reprocessing step
    def _reprocess(self, state: Any, p: Problem, tried: Dict[str, Set[str]], validated: Mapping[str, object]
                   ) -> Tuple[Any, ActionRecord]:
        done = tried.setdefault(p.problem_id, set())
        candidates: List[Tuple[Task, List[str]]] = []
        for task in p.tasks:
            ids = [s.engine_id for s in self.registry.available(task) if s.engine_id not in done]
            if ids:
                candidates.append((task, ids))
        if not candidates:
            return state, ActionRecord(p.problem_id, "exhausted", (), "no untried available engine can address this problem")
        task, ids = candidates[0]
        request = EngineRequest(sources=tuple(p.target_ids) or ("region",),
                                payload={"problem": p.problem_id, "kind": p.kind.value, "location": dict(p.location)})
        report = self.manager.run(task, request, only=ids)
        done.update(r.engine_id for r in report.runs)
        ran = tuple(r.engine_id for r in report.runs if r.ok)
        if not ran:
            return state, ActionRecord(p.problem_id, "engine_failed", tuple(r.engine_id for r in report.runs),
                                       "; ".join(f"{r.engine_id}: {r.error}" for r in report.runs if r.error))
        candidate = self.apply(state, p, report)
        if candidate is state or candidate is None:
            return state, ActionRecord(p.problem_id, "unchanged", ran, "reconciliation produced no change")
        violations = list(self.consistency(candidate))
        if violations:
            return state, ActionRecord(p.problem_id, "rejected_inconsistent", ran, "; ".join(violations))
        regs = check_regressions(validated, self.features(candidate), tolerance=self.regression_tolerance) if validated else []
        if regs:
            return state, ActionRecord(p.problem_id, "rejected_regression", ran, "regressed: " + ", ".join(r.feature_id for r in regs))
        return candidate, ActionRecord(p.problem_id, "applied", ran, "")

    def _done(self, state, reason, history, capture, quality, problems) -> RefinementResult:
        return RefinementResult(state, reason, tuple(history), tuple(capture.values()), quality,
                                tuple(sorted(p.problem_id for p in problems)))
