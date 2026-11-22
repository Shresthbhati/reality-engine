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
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field, replace
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
from engine.multiengine.detection_groups import DetectionGroup, group_detections
from engine.multiengine.evidence_matrix import EvidenceMatrix
from engine.multiengine.executor import (
    ExecutionCancelled,
    ExecutionManager,
    Policy,
    ReportStatus,
    ResourceProbe,
)
from engine.multiengine.identity import IdentityRegistry
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


def problem_facts(ce: CanonicalEntity, matrix: EvidenceMatrix, onto: Any) -> tuple[str, ProblemEvidence, dict[str, Any]]:
    """(problem kind, measured evidence facts, location) for one entity: furniture/fixtures/objects whose size is not determined,
    whose rear is unobserved (one viewpoint), or that cover too few pixels for the ontology's minimum."""
    node = onto.get(ce.ontology_path)
    sources = sorted({matrix.store.get(m).source_id for m in ce.members})
    dims = ce.determinations.get(DIMENSIONS)
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
    return kind, ev, {"entity_id": ce.entity_id, "ontology_path": ce.ontology_path, "frame": "unresolved"}


def _problem_for(
    ce: CanonicalEntity, matrix: EvidenceMatrix, onto: Any, conflict_entities: set[str]
) -> list[CaptureRequestSpec]:
    """Capture requests that rest ONLY on measured facts about this entity's evidence."""
    sources = sorted({matrix.store.get(m).source_id for m in ce.members})
    out: list[CaptureRequestSpec] = []
    kind, ev, location = problem_facts(ce, matrix, onto)
    basis = tuple(sorted(ce.members))
    group = ce.ontology_path.replace(".", "/").split("/")[0]
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


#: Starting value, NOT tuned against any dataset; recorded in every snapshot under ``grouping``.
SAME_IMAGE_IOU_MIN = 0.5


@dataclass
class Derived:
    """Entities derived from a SELECTION of the matrix's observations (raw evidence is never modified by deriving)."""

    cands: list[Candidate]  # one per image-level detection group (its representative observation)
    groups: list[DetectionGroup]
    assoc: AssociationResult  # entities expanded to EVERY member observation of their groups
    canonicals: list[CanonicalEntity]
    conflicts: list[Conflict]
    identity_events: list[dict[str, Any]]
    limitations: list[str]


def derive_entities(
    matrix: EvidenceMatrix,
    active: AbstractSet[str] | None,
    *,
    registry: EngineRegistry,
    onto: Any,
    pair_scorer: PairScorer | None = None,
    tolerances: Mapping[str, float] | None = None,
    identity: IdentityRegistry | None = None,
    overlap_tolerance_m: float | None = None,
    same_image_iou_min: float = SAME_IMAGE_IOU_MIN,
    record: bool = False,
) -> Derived:
    """detections -> image-level groups -> cross-view association -> stable ids -> canonical entities -> conflicts.
    ``active`` selects which observations take part (None = every non-rejected detection). ``record`` writes the ``same_entity`` /
    ``conflicts_with`` relations into the append-only store -- only the ADOPTED state should record, never a trial."""
    obs = [
        o for o in matrix.store.query(quantity="bbox")
        if isinstance(o.predicted.get("label"), str) and matrix.status(o.observation_id) != "rejected"
        and (active is None or o.observation_id in active)
    ]
    groups = group_detections(obs, iou_min=same_image_iou_min)
    by_rep = {g.representative: g for g in groups}
    label_of = {o.observation_id: str(o.predicted["label"]) for o in obs}
    # A sub-part detection (handle on a door, ...) names its host box in the SAME image (``parent_pipeline_box_id``); the host box's
    # image-level group representative becomes the part's parent, so association/canonical/bridge can record PART_OF. Detections that
    # carry no box ids (every whole-image detector) have no parent: nothing is inferred.
    obs_by_id = {o.observation_id: o for o in obs}
    box_obs = {(o.source_id, o.predicted["pipeline_box_id"]): o.observation_id for o in obs if o.predicted.get("pipeline_box_id") is not None}
    rep_of_obs = {m: g.representative for g in groups for m in g.members}

    def parent_rep(rep: str) -> str | None:
        o = obs_by_id[rep]
        host = o.predicted.get("parent_pipeline_box_id")
        pid = box_obs.get((o.source_id, host)) if host is not None else None
        p = rep_of_obs.get(pid) if pid else None
        return p if p and p != rep else None

    cands = [Candidate(g.representative, g.source_id, label_of[g.representative], parent_rep(g.representative)) for g in groups]
    limitations = list(LIMITATIONS) + ([] if pair_scorer is not None else [PAIR_SCORER_LIMITATION])
    scores = list(pair_scorer(cands, matrix)) if pair_scorer is not None and cands else []
    assoc = associate(cands, scores, labels_compatible=ontology_label_compat(onto))
    from engine.multiengine.lift_stage import lifted_members

    lifted = lifted_members(matrix, obs)  # lifted position/dimensions of the SAME box in the SAME image join that detection's entity
    entities = [
        replace(e, members=tuple(sorted({x for m in ms for x in (m, *lifted.get(m, ()))})),
                labels=tuple(sorted({label_of[m] for m in ms})))
        for e in assoc.entities
        for ms in [tuple(sorted(m for rep in e.members for m in by_rep[rep].members))]
    ]
    identity_events: list[dict[str, Any]] = []
    if identity is not None:  # keep entity ids stable when later evidence extends, merges or splits entities
        entities, events = identity.resolve(entities)
        identity_events = [e.to_dict() for e in events]
    assoc = AssociationResult(tuple(entities), assoc.decisions)
    ent_of_member = {m: e.entity_key for e in assoc.entities for m in e.members}

    def parent_entity(e: Any) -> str | None:  # the entity that hosts this one's sub-part detections (None for whole objects)
        ps = {ent_of_member.get(c.parent_observation_id) for c in cands if c.observation_id in e.members and c.parent_observation_id}
        ps.discard(None)
        ps.discard(e.entity_key)
        return sorted(ps)[0] if ps else None

    canonicals = [
        build_canonical_entity(
            e, matrix, target_frame="world", tolerances=tolerances or {}, labels=label_of, registry=registry,
            ontology=onto, association=assoc, graph=None, parent_entity_id=parent_entity(e),
        )
        for e in assoc.entities
    ]
    conflicts = semantic_conflicts(assoc)
    boxes: list[Box] = [b for b in (box_from_entity(c) for c in canonicals) if b is not None]
    if boxes and overlap_tolerance_m is not None:
        conflicts += find_overlaps(boxes, tolerance_m=overlap_tolerance_m, contact_allowed=ontology_contact_allowed(onto))
    elif boxes:
        limitations.append("Interpenetration was not checked: no overlap tolerance was supplied (it has no default).")
    if record:
        for d in assoc.decisions:
            if d.outcome == "merged":
                matrix.store.link(d.a, d.b, "same_entity")
        supporting = {c.entity_id: list(c.determinations[POSITION].supporting) for c in canonicals if POSITION in c.determinations}
        record_conflicts(matrix.store, conflicts, supporting)
    return Derived(cands, groups, assoc, canonicals, conflicts, identity_events, limitations)


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
    identity_registry: IdentityRegistry | None = None,
    scoring_note: Mapping[str, Any] | None = None,
    refine: bool = True,
    refine_max_iterations: int = 3,
    after_stages: Callable[[EvidenceMatrix], Mapping[str, Any]] | None = None,
    relations_fn: Callable[[Sequence[CanonicalEntity], EvidenceMatrix], Sequence[Any]] | None = None,
    surfaces_fn: Callable[[EvidenceMatrix], tuple[Sequence[Any], Sequence[Any], Sequence[str], Mapping[str, Any]]] | None = None,
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
    elif n_obs == 0 and not any(r.status == "succeeded" for r in matrix.runs()):
        status = "failed"  # nothing ran successfully; a successful run that found nothing is NOT a failure
    elif all(s is ReportStatus.COMPLETE for s in statuses):
        status = "complete"
    else:
        status = "partial"

    # ---- caller-supplied step on the SAME matrix (depth + lifting): runs before refinement and entity derivation so both see its
    # observations. A failure is recorded in the snapshot and the flow carries on with whatever evidence exists (never silent).
    after_report: dict[str, Any] | None = None
    if after_stages is not None and not cancelled:
        try:
            after_report = dict(after_stages(matrix))
        except Exception as exc:  # noqa: BLE001
            after_report = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}

    # ---- refinement: reprocess with untried engines where that can help, roll back what fails the checks, stop with a reason
    refine_outcome = None
    active: frozenset[str] | None = None
    start_active = frozenset(
        o.observation_id for o in matrix.store.query(quantity="bbox")
        if isinstance(o.predicted.get("label"), str) and matrix.status(o.observation_id) != "rejected"
    )
    if refine and not cancelled and n_obs > 0:
        from engine.multiengine.refine_flow import FlowRefiner

        try:
            refine_outcome = FlowRefiner(
                matrix=matrix, manager=mgr, registry=registry, onto=onto, pair_scorer=pair_scorer, tolerances=tolerances,
                identity=identity_registry, overlap_tolerance_m=overlap_tolerance_m, max_iterations=refine_max_iterations,
            ).run()
            active = refine_outcome.state.active
        except ExecutionCancelled as exc:  # cancelled mid-reprocessing: keep the pre-refinement selection; finished work is persisted
            cancelled, status, active = True, "cancelled", start_active
            stage_summaries.append({"task": "refinement", "status": "cancelled", "warnings": list(exc.report.warnings), "engines": {}})

    # ---- evidence -> entities (whatever the evidence supports)
    derived = derive_entities(
        matrix, active, registry=registry, onto=onto, pair_scorer=pair_scorer, tolerances=tolerances,
        identity=identity_registry, overlap_tolerance_m=overlap_tolerance_m, record=True,
    )
    cands, assoc, canonicals, conflicts = derived.cands, derived.assoc, derived.canonicals, derived.conflicts
    identity_events, limitations = derived.identity_events, derived.limitations
    if after_report and after_report.get("lifted"):  # geometry now exists: the "no geometry" statement no longer holds as written
        limitations[0:1] = list(after_report.get("limitations") or ())
    if n_obs == 0 and status != "failed":
        limitations.append("The engines ran successfully but produced no detections on these images (no entity is invented).")
    conflict_entities = {eid for cf in conflicts if cf.kind == "interpenetration" for eid in cf.entity_ids}

    requests: list[CaptureRequestSpec] = []
    if refine_outcome is not None:  # requests the loop derived from measured facts (it also reprocessed whatever could be)
        requests = list(refine_outcome.result.capture_requests)
    else:
        for ce in canonicals:
            requests.extend(_problem_for(ce, matrix, onto, conflict_entities))

    from engine.multiengine.spatial_relations import derive_relations

    relations = list(derive_relations(canonicals, matrix))
    if relations_fn is not None:  # e.g. the spatial-intelligence relationships, mapped onto these canonical entities
        known = {(r.source_entity_id, r.kind, r.target_entity_id) for r in relations}
        relations += [r for r in relations_fn(canonicals, matrix) if (r.source_entity_id, r.kind, r.target_entity_id) not in known]
    bridge = convert(canonicals, matrix, ontology=onto, relations=relations)
    surface_report: dict[str, Any] | None = None
    if surfaces_fn is not None:  # observed floor / wall / ceiling planes + mesh: entities of the SAME world, never a second representation
        try:
            s_ents, s_geoms, s_notes, surface_report = surfaces_fn(matrix)
            for e in s_ents:
                bridge.entities[e.id] = e
            for g in s_geoms:
                bridge.geometries[g.id] = g
            bridge.notes.extend(s_notes)
        except Exception as exc:  # noqa: BLE001 -- recorded in the snapshot, never silent and never fatal to the objects
            surface_report = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    hierarchy = SpatialHierarchy(FrameGraph()).readiness()

    snapshot: dict[str, Any] = {
        "snapshot_version": SNAPSHOT_VERSION, "run_id": run_id, "status": status, "sources": list(sources),
        "counts": {
            "observations": len(matrix), "runs": len(matrix.runs()), "candidates": len(cands), "entities": len(canonicals),
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
        "refinement_history": (
            [refine_outcome.to_dict()] if refine_outcome is not None
            else [{"stop_reason": "not_run", "reason": "refinement was disabled, cancelled or there was no evidence"}]
        ),
        "grouping": {"same_image_iou_min": SAME_IMAGE_IOU_MIN, "caveat": "untuned starting value"},
        "identity_events": identity_events,
        "scoring": dict(scoring_note) if scoring_note else None,  # which pair scorer + thresholds were used (None = no scorer)
        "bridge_notes": list(bridge.notes),
        "after_stages": after_report,  # depth + lifting facts (None = no such step was supplied)
        "surfaces": surface_report,  # observed planes / mesh facts (None = not requested)
        "matrix_problems": matrix.verify(),
    }
    _atomic_write_json(root / SNAPSHOT_FILE, snapshot)
    return EvidenceFlowResult(run_id, status, root, snapshot, canonicals, assoc, conflicts, requests, bridge)
