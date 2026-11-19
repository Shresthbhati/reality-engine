"""Evidence-to-world flow: original images -> executed engines -> Evidence Matrix -> association -> canonical entities -> conflicts ->
capture requests -> WorldIR entities, as ONE call with a durable, inspectable result.

This composes the pieces built so far; it adds no reconstruction logic of its own. What it can and cannot do today is stated in
``LIMITATIONS`` and copied into every snapshot, so a reader of the result cannot mistake it for more than it is:

* the detectors emit 2D boxes and labels. No engine in the repository emits metric 3D ``position`` / ``dimensions`` observations yet
  (that needs depth + camera pose lifting), so a canonical entity built here has a class and traceable 2D evidence but NO geometry --
  the bridge does not invent one -- and the capture requests ask for what is missing;
* cross-view association needs pair scores; unless the caller supplies a ``pair_scorer`` every detection is its own entity.

Cancellation is checked between engines (``ExecutionManager``) and between stages; what already finished stays persisted.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from engine.multiengine.alignment import FrameGraph
from engine.multiengine.association import (
    AssociationResult,
    Candidate,
    PairScore,
    associate,
    ontology_label_compat,
)
from engine.multiengine.canonical import (
    DIMENSIONS,
    POSITION,
    CanonicalEntity,
    build_canonical_entity,
    explain,
)
from engine.multiengine.capture_requests import (
    CaptureRequestSpec,
    ProblemEvidence,
    request_for_problem,
)
from engine.multiengine.conflicts import (
    Box,
    Conflict,
    box_from_entity,
    find_overlaps,
    ontology_contact_allowed,
    record_conflicts,
    semantic_conflicts,
)
from engine.multiengine.contract import EngineAdapter, EngineRequest, Task
from engine.multiengine.evidence_matrix import EvidenceMatrix
from engine.multiengine.executor import (
    ExecutionCancelled,
    ExecutionManager,
    Policy,
    ReportStatus,
    ResourceProbe,
)
from engine.multiengine.registry import EngineRegistry
from engine.multiengine.spatial_hierarchy import SpatialHierarchy
from engine.multiengine.worldir_bridge import BridgeResult, convert

SNAPSHOT_FILE = "snapshot.json"
SNAPSHOT_VERSION = 1

LIMITATIONS = (
    "Detectors emit 2D boxes and labels only; no engine here emits metric 3D position or dimensions, so entities carry no "
    "geometry (none is fabricated).",
    "All confidences are uncalibrated raw model outputs; some adapters emit fixed values rather than model scores. Nothing here "
    "is physically validated against measured ground truth.",
    "The refinement loop (targeted re-processing with rollback) is not run: with no 3D estimates there is nothing to re-measure; "
    "unresolved problems become capture requests instead.",
)
PAIR_SCORER_LIMITATION = "No cross-view pair scorer was supplied, so every detection is its own entity (no cross-view fusion)."

PairScorer = Callable[[Sequence[Candidate], EvidenceMatrix], Iterable[PairScore]]


@dataclass(frozen=True)
class StageSpec:
    task: Task
    only: tuple[str, ...] | None = None  # restrict to named engines; None = every approved engine for the task


@dataclass
class EvidenceFlowResult:
    run_id: str
    status: str  # complete | partial | failed | cancelled
    root: Path
    snapshot: dict[str, Any]
    canonicals: list[CanonicalEntity] = field(default_factory=list)
    association: AssociationResult | None = None
    conflicts: list[Conflict] = field(default_factory=list)
    capture_requests: list[CaptureRequestSpec] = field(default_factory=list)
    bridge: BridgeResult | None = None

    @property
    def cancelled(self) -> bool:
        return self.status == "cancelled"


def _atomic_write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(data, fh, sort_keys=True, separators=(",", ":"))
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def load_snapshot(root: Path) -> dict[str, Any]:
    return json.loads((Path(root) / SNAPSHOT_FILE).read_text(encoding="utf-8"))


def _box_area_px(matrix: EvidenceMatrix, oid: str) -> float | None:
    r = matrix.store.get(oid).image_region
    return None if r is None else max(0.0, (r[2] - r[0]) * (r[3] - r[1]))


def _problem_for(
    ce: CanonicalEntity, matrix: EvidenceMatrix, onto: Any, conflict_entities: set[str]
) -> list[CaptureRequestSpec]:
    """Capture requests that rest ONLY on measured facts about this entity's evidence."""
    node = onto.get(ce.ontology_path)
    sources = sorted({matrix.store.get(m).source_id for m in ce.members})
    dims = ce.determinations.get(DIMENSIONS)
    out: list[CaptureRequestSpec] = []
    location = {"entity_id": ce.entity_id, "ontology_path": ce.ontology_path, "frame": "unresolved"}
    basis = tuple(sorted(ce.members))
    group = ce.ontology_path.replace(".", "/").split("/")[0]

    # furniture / fixtures / objects: size not determined, rear unobserved, too few pixels for a small fitting
    unmeasured = ("w", "d", "h") if (dims is None or dims.status != "resolved") else ()
    pixels = max((a for a in (_box_area_px(matrix, m) for m in ce.members) if a is not None), default=None)
    needed = float(node.min_pixels) if node is not None and node.min_pixels > 0 else None
    ev = ProblemEvidence(
        view_ids=tuple(sources),
        unmeasured_dimension_axes=unmeasured,
        rear_observed=False if len(sources) == 1 else None,  # one viewpoint -> rear unobserved; >= 2 is not proof either way
        pixels_on_target=pixels,
        pixels_on_target_needed=needed,
    )
    kind = "unclassified_object" if ce.ontology_path.startswith("fallback") else "incorrect_dimension"
    if node is not None and getattr(node, "small", False):
        kind = "missed_small_object"
    spec = request_for_problem(problem_kind=kind, location=location, target_ids=ce.members, evidence=ev, basis=basis)
    if spec is not None:
        out.append(spec)
    if ce.entity_id in conflict_entities and group in ("architecture", "structure", "wall", "room"):
        arch = request_for_problem(
            problem_kind="incomplete_wall", location=location, target_ids=ce.members, basis=basis,
            evidence=ProblemEvidence(view_ids=tuple(sources), inconsistent_estimates=True),
        )
        if arch is not None:
            out.append(arch)
    return out


def run_evidence_flow(
    root: Path,
    run_id: str,
    sources: Sequence[str],
    *,
    registry: EngineRegistry,
    adapters: Mapping[str, EngineAdapter],
    stages: Sequence[StageSpec],
    should_cancel: Callable[[], bool] | None = None,
    camera_refs: Mapping[str, str] | None = None,
    pair_scorer: PairScorer | None = None,
    tolerances: Mapping[str, float] | None = None,
    ontology: Any | None = None,
    probe: ResourceProbe | None = None,
    policy: Policy = Policy.MAXIMUM_FIDELITY,
    metadata: Mapping[str, Any] | None = None,
    overlap_tolerance_m: float | None = None,
) -> EvidenceFlowResult:
    from perception.ontology import get_default_ontology

    root = Path(root)
    onto = ontology or get_default_ontology()
    matrix = EvidenceMatrix(root)
    mgr = ExecutionManager(
        registry, adapters, matrix=matrix, run_id=run_id, should_cancel=should_cancel, probe=probe, policy=policy,
        max_parallel=1, camera_refs=camera_refs,
    )
    stage_summaries: list[dict[str, Any]] = []
    statuses: list[ReportStatus] = []
    cancelled = False
    for st in stages:
        if should_cancel is not None and should_cancel():
            cancelled = True
            break
        try:
            report = mgr.run(
                st.task, EngineRequest(sources=tuple(sources), payload=dict(metadata or {}), seed=42), only=st.only
            )
        except ExecutionCancelled as exc:
            report, cancelled = exc.report, True
        statuses.append(report.status)
        stage_summaries.append({
            "task": st.task.value, "status": report.status.value, "warnings": list(report.warnings),
            "engines": {r.engine_id: r.status.value for r in report.runs},
        })
        if cancelled:
            break

    n_obs = len(matrix)
    if cancelled:
        status = "cancelled"
    elif n_obs == 0:
        status = "failed"
    elif all(s is ReportStatus.COMPLETE for s in statuses):
        status = "complete"
    else:
        status = "partial"

    # ---- evidence -> entities (whatever the evidence supports)
    cands = [
        Candidate(o.observation_id, o.source_id, str(o.predicted["label"]))
        for o in matrix.store.query(quantity="bbox")
        if isinstance(o.predicted.get("label"), str) and matrix.status(o.observation_id) != "rejected"
    ]
    limitations = list(LIMITATIONS) + ([] if pair_scorer is not None else [PAIR_SCORER_LIMITATION])
    scores = list(pair_scorer(cands, matrix)) if pair_scorer is not None and cands else []
    assoc = associate(cands, scores, labels_compatible=ontology_label_compat(onto), store=matrix.store)
    labels = {c.observation_id: c.label for c in cands}
    canonicals = [
        build_canonical_entity(
            e, matrix, target_frame="world", tolerances=tolerances or {}, labels=labels, registry=registry,
            ontology=onto, association=assoc, graph=None,
        )
        for e in assoc.entities
    ]

    conflicts = semantic_conflicts(assoc)
    boxes: list[Box] = [b for b in (box_from_entity(c) for c in canonicals) if b is not None]
    if boxes and overlap_tolerance_m is not None:
        conflicts += find_overlaps(boxes, tolerance_m=overlap_tolerance_m, contact_allowed=ontology_contact_allowed(onto))
    elif boxes:
        limitations.append("Interpenetration was not checked: no overlap tolerance was supplied (it has no default).")
    supporting = {c.entity_id: list(c.determinations[POSITION].supporting) for c in canonicals if POSITION in c.determinations}
    record_conflicts(matrix.store, conflicts, supporting)
    conflict_entities = {eid for cf in conflicts if cf.kind == "interpenetration" for eid in cf.entity_ids}

    requests: list[CaptureRequestSpec] = []
    for ce in canonicals:
        requests.extend(_problem_for(ce, matrix, onto, conflict_entities))

    bridge = convert(canonicals, matrix, ontology=onto)
    hierarchy = SpatialHierarchy(FrameGraph()).readiness()

    snapshot: dict[str, Any] = {
        "snapshot_version": SNAPSHOT_VERSION, "run_id": run_id, "status": status, "sources": list(sources),
        "counts": {
            "observations": n_obs, "runs": len(matrix.runs()), "candidates": len(cands), "entities": len(canonicals),
            "conflicts": len(conflicts), "capture_requests": len(requests),
        },
        "stages": stage_summaries, "limitations": limitations,
        "entities": [
            {"entity_id": c.entity_id, "ontology_path": c.ontology_path, "members": list(c.members),
             "explain": explain(c, matrix, ontology=onto),
             "world_entity": bridge.entities[f"ent-{c.entity_id}"].to_dict()}
            for c in canonicals
        ],
        "associations": [
            {"a": d.a, "b": d.b, "score": d.score, "outcome": d.outcome, "reason": d.reason, "basis": dict(d.basis)}
            for d in assoc.decisions
        ],
        "conflicts": [
            {"kind": c.kind, "entity_ids": list(c.entity_ids), "detail": c.detail, "magnitude": c.magnitude,
             "basis": dict(c.basis)}
            for c in conflicts
        ],
        "capture_requests": [r.to_row(world_id="", request_id=f"{run_id}-cr{i}") for i, r in enumerate(requests)],
        "hierarchy": hierarchy,
        "refinement_history": [{
            "iteration": 1, "problems": len(requests),
            "actions": ["association", "canonical_fusion", "conflict_detection", "capture_requests"],
            "reprocessing": "not run: " + LIMITATIONS[2],
        }],
        "bridge_notes": list(bridge.notes),
        "matrix_problems": matrix.verify(),
    }
    _atomic_write_json(root / SNAPSHOT_FILE, snapshot)
    return EvidenceFlowResult(run_id, status, root, snapshot, canonicals, assoc, conflicts, requests, bridge)
