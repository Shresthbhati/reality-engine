"""The refinement loop for the evidence-to-world flow: the domain callbacks for ``refinement.RefinementLoop``.

State is a SELECTION of observation ids (``RefineState.active``). The raw matrix is append-only and is never edited by refinement:

* reprocessing (``ExecutionManager.run(..., only=untried engines)``) adds new raw observations to the matrix;
* ``apply`` proposes a new selection that includes them;
* the loop's CONSISTENCY check (no entity holds two different detections from one image; reprocessing introduced no new label
  conflicts) and REGRESSION check (every established class of an entity survives; none may vanish or change) decide whether the new
  selection is adopted. If not, the previous selection stays -- ROLLBACK -- and the rejected reprocessing observations stay in the matrix,
  marked ``rejected`` with the reason, so what an engine said is never lost.
* a problem is reprocessed ONLY when an untried available engine could change the answer (cause ENGINE_MISS / ENGINE_DISAGREEMENT).
  When none exists, the cause is INSUFFICIENT_EVIDENCE and the loop turns it into a capture request from measured facts instead.

Stopping reasons are the loop's explicit ones (target met / max iterations / no improvement / no engine available / awaiting evidence /
nothing to refine). With every gate threshold unset (none is benchmark-established) the quality verdict stays UNVERIFIED, so success is
never claimed from here; progress is measured as the number of open problems.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from engine.multiengine.capture_requests import ProblemEvidence
from engine.multiengine.conflicts import semantic_conflicts
from engine.multiengine.contract import Task
from engine.multiengine.evidence_flow import (
    SAME_IMAGE_IOU_MIN,
    Derived,
    derive_entities,
    problem_facts,
)
from engine.multiengine.evidence_matrix import EvidenceMatrix
from engine.multiengine.executor import ExecutionManager, ExecutionReport
from engine.multiengine.identity import IdentityRegistry
from engine.multiengine.quality_gates import GateSpec, Measurement, QualityReport
from engine.multiengine.refinement import (
    Problem,
    ProblemCause,
    ProblemKind,
    RefinementLoop,
    RefinementResult,
)
from engine.multiengine.registry import EngineRegistry

GATES = (GateSpec("unresolved_problems", "lower_is_better", unit="count"),)  # no threshold: UNVERIFIED by design


def _kind(value: str) -> ProblemKind:
    try:
        return ProblemKind(value)
    except ValueError:
        return ProblemKind.INCORRECT_DIMENSION


@dataclass(frozen=True)
class RefineState:
    active: frozenset[str]


@dataclass
class RefineOutcome:
    state: RefineState
    result: RefinementResult
    rolled_back: tuple[str, ...]  # observation ids produced by reprocessing that were NOT adopted (kept raw, marked rejected)

    def to_dict(self) -> dict[str, Any]:
        d = self.result.to_dict()
        d["final_observations"] = len(self.state.active)
        d["rolled_back_observations"] = list(self.rolled_back)
        # Refinement is only claimed when an engine actually produced NEW inference during the loop (adopted or rolled back).
        # A failed or impossible reprocessing attempt is recorded as an attempt, never as refinement.
        actions = [a for it in d.get("iterations", []) for a in it.get("actions", [])]
        inferred = [a for a in actions if a.get("outcome") in ("applied", "unchanged", "rejected_inconsistent", "rejected_regression")]
        d["new_inference"] = bool(inferred)
        d["reprocessing_attempts"] = len([a for a in actions if a.get("engines")])
        d["refinement_claim"] = "refined" if any(a.get("outcome") == "applied" for a in actions) else "not_refined"
        return d


class FlowRefiner:
    def __init__(
        self,
        *,
        matrix: EvidenceMatrix,
        manager: ExecutionManager,
        registry: EngineRegistry,
        onto: Any,
        pair_scorer: Any = None,
        tolerances: Mapping[str, float] | None = None,
        identity: IdentityRegistry | None = None,
        overlap_tolerance_m: float | None = None,
        tasks: Sequence[Task] = (Task.DETECTION,),
        max_iterations: int = 3,
        patience: int = 1,
        same_image_iou_min: float = SAME_IMAGE_IOU_MIN,
    ) -> None:
        self.m, self.mgr, self.registry, self.onto = matrix, manager, registry, onto
        self.pair_scorer, self.tolerances, self.overlap = pair_scorer, tolerances, overlap_tolerance_m
        self.tasks, self.iou = tuple(tasks), same_image_iou_min
        self._identity_base = identity.clone() if identity is not None else None  # trial states never touch the real registry
        self._memo: dict[frozenset[str], Derived] = {}
        self._baseline_label_conflicts = 0
        self.loop = RefinementLoop(
            manager=manager, registry=registry, specs=list(GATES), measure=self._measure, detect=self._detect,
            apply=self._apply, consistency=self._consistency, features=self._features, regression_tolerance={},
            tried_before=lambda task: [r.engine_id for r in self.m.runs() if r.task == task.value],
            max_iterations=max_iterations, patience=patience,
        )

    # ------------------------------------------------------------------ derivation (memoised per selection)
    def _derive(self, state: RefineState) -> Derived:
        if state.active not in self._memo:
            self._memo[state.active] = derive_entities(
                self.m, state.active, registry=self.registry, onto=self.onto, pair_scorer=self.pair_scorer,
                tolerances=self.tolerances, identity=self._identity_base.clone() if self._identity_base else None,
                overlap_tolerance_m=self.overlap, same_image_iou_min=self.iou, record=False,
            )
        return self._memo[state.active]

    def _untried(self, task: Task) -> list[str]:
        ran = {r.engine_id for r in self.m.runs() if r.task == task.value}
        return [s.engine_id for s in self.registry.available(task) if s.engine_id not in ran]

    # ------------------------------------------------------------------ loop callbacks
    def _problems(self, state: RefineState) -> list[Problem]:
        d = self._derive(state)
        untried = any(self._untried(t) for t in self.tasks)
        out: list[Problem] = []
        for ce in d.canonicals:
            sources = sorted({self.m.store.get(o).source_id for o in ce.members})
            kind, ev, loc = problem_facts(ce, self.m, self.onto)
            loc = {**loc, "sources": sources}  # what reprocessing must re-run on: the original IMAGES
            basis = tuple(sorted(ce.members))
            conflict = ce.class_status == "conflict"
            if conflict or ce.ontology_path.startswith("fallback"):
                if not untried:
                    cause = ProblemCause.INSUFFICIENT_EVIDENCE
                else:
                    cause = ProblemCause.ENGINE_DISAGREEMENT if conflict else ProblemCause.ENGINE_MISS
                out.append(Problem(
                    f"class:{ce.entity_id}", ProblemKind.UNCLASSIFIED_OBJECT, cause, loc, tasks=self.tasks, evidence=ev,
                    target_ids=ce.members, detail="class unresolved" if conflict else "no class established", basis=basis,
                ))
            out.append(Problem(
                f"measure:{ce.entity_id}", _kind(kind), ProblemCause.INSUFFICIENT_EVIDENCE, loc, tasks=(), evidence=ev,
                target_ids=ce.members, detail="size not determined / rear unobserved / too few pixels", basis=basis,
            ))
        by_id = {c.entity_id: c for c in d.canonicals}
        for cf in d.conflicts:
            if cf.kind == "interpenetration":
                cause = ProblemCause.ENGINE_DISAGREEMENT if untried else ProblemCause.INSUFFICIENT_EVIDENCE
                ents = [by_id[e] for e in cf.entity_ids if e in by_id]
                srcs = sorted({self.m.store.get(o).source_id for e in ents for o in e.members})
                out.append(Problem(
                    f"overlap:{'+'.join(sorted(cf.entity_ids))}", ProblemKind.POOR_REGISTRATION, cause,
                    {"entity_ids": list(cf.entity_ids), "sources": srcs}, tasks=self.tasks,
                    evidence=ProblemEvidence(inconsistent_estimates=True), target_ids=tuple(m for e in ents for m in e.members),
                    detail=cf.detail,
                ))
        return out

    def _measure(self, state: RefineState) -> tuple[Mapping[str, Measurement | None], bool | None]:
        n = len(self._problems(state))
        m = Measurement(float(n), "open problems derived from the entities' own evidence", n=len(self._derive(state).canonicals),
                        source="evidence-flow")
        return {"unresolved_problems": m}, True

    def _detect(self, state: RefineState, quality: QualityReport) -> Sequence[Problem]:
        return self._problems(state)

    def _apply(self, state: RefineState, problem: Problem, report: ExecutionReport) -> RefineState:
        new = {
            o.observation_id for r in report.runs for o in r.observations
            if o.quantity == "bbox" and isinstance(o.predicted.get("label"), str)
        } - state.active
        return state if not new else RefineState(state.active | frozenset(new))

    def _consistency(self, state: RefineState) -> Sequence[str]:
        d = self._derive(state)
        group_of = {m: g.group_id for g in d.groups for m in g.members}
        bad: list[str] = []
        for ce in d.canonicals:
            per_source: dict[str, set[str]] = {}
            for oid in ce.members:
                if oid not in group_of:  # derived evidence (lifted position / dimensions, attributes) is not a detection: no group to compare
                    continue
                per_source.setdefault(self.m.store.get(oid).source_id, set()).add(group_of[oid])
            for src, groups in per_source.items():
                if len(groups) > 1:
                    bad.append(f"entity {ce.entity_id} holds {len(groups)} different detections from image {src}")
        n_conf = len(semantic_conflicts(d.assoc))
        if n_conf > self._baseline_label_conflicts:
            bad.append(f"reprocessing introduced {n_conf - self._baseline_label_conflicts} new label conflict(s)")
        return bad

    def _features(self, state: RefineState) -> Mapping[str, object]:
        """Established classes only: an unknown class is not a validated feature, so replacing it with a real one is not a regression,
        but replacing or losing an ESTABLISHED class is."""
        return {
            f"class:{c.entity_id}": c.ontology_path for c in self._derive(state).canonicals
            if c.class_status in ("agreed_unverified", "resolved_by_evidence") and not c.ontology_path.startswith("fallback")
        }

    # ------------------------------------------------------------------ run
    def run(self) -> RefineOutcome:
        start = RefineState(frozenset(
            o.observation_id for o in self.m.store.query(quantity="bbox")
            if isinstance(o.predicted.get("label"), str) and self.m.status(o.observation_id) != "rejected"
        ))
        before = {o.observation_id for o in self.m.store.query(quantity="bbox")}
        self._baseline_label_conflicts = len(semantic_conflicts(self._derive(start).assoc))
        result = self.loop.run(start)
        final: RefineState = result.state
        produced = {o.observation_id for o in self.m.store.query(quantity="bbox")} - before
        dropped = tuple(sorted(produced - final.active))
        reason = ("reprocessing result not adopted: it failed the consistency/regression check or changed nothing "
                  "(see refinement history); the raw prediction is kept")
        for oid in dropped:
            if self.m.status(oid) != "rejected":
                self.m.set_status(oid, "rejected", by="refinement", reason=reason)
        return RefineOutcome(final, result, dropped)
