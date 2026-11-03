"""Evidence-backed change proposals: a candidate world (e.g. from a new scan) is compared with a trusted stored version
and turned into REVIEWABLE changes. Nothing here modifies a stored version; accepting a proposal creates a NEW
version whose parent is the proposal's base (versions stay immutable, history stays intact).

Assessment rules (deterministic, ordered; the first matching rule decides an entity's verdict):

  REMOVED   -> needs_review. Absence from a new scan is not evidence that something is gone.
  ADDED     -> supported when the candidate entity cites at least one observation, or is OBSERVED/RECONSTRUCTED with
               confidence >= MIN_ADD_CONFIDENCE; otherwise needs_review ("no evidence").
  MODIFIED  -> conflict when the new provenance ranks below the old one and the old one was trusted
               (>= RECONSTRUCTED): weaker data must not silently replace stronger data;
            -> conflict when placement/geometry moved beyond GEOMETRY_TOLERANCE_M while confidence DROPPED;
            -> needs_review when a geometry/transform/semantic change cites no observation that the base entity did
               not already have ("change without new evidence");
            -> supported otherwise.

Accepting: by default only `supported` changes are applied. `needs_review` changes must be named explicitly in
``entity_ids``; `conflict` changes additionally need ``override_conflicts`` and a non-empty ``reason``, both recorded
on the proposal. The merged world must pass world validation, and the saved version must pass read-back
verification, otherwise it is quarantined and nothing is adopted.

Known limits: geometry is carried over per entity (its referenced geometries); materials/surfaces referenced by an
applied entity are not copied (validation reports dangling references and refuses); relationship edges from OTHER
entities to an added/removed entity are not rewritten.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from provenance import Provenance
from world_ir.diff import ChangeKind, diff_worlds
from world_ir.validation import validate_world_ir
from world_ir.world_v1 import WorldIR
from worldstore.store import StoredVersion, WorldStore, WorldStoreError, _atomic_write_text

MIN_ADD_CONFIDENCE = 0.5
GEOMETRY_TOLERANCE_M = 0.01
TRUST = {Provenance.OBSERVED: 4, Provenance.RECONSTRUCTED: 3, Provenance.ESTIMATED: 2, Provenance.INFERRED: 1,
         Provenance.GENERATED: 1, Provenance.UNKNOWN: 0, Provenance.CONFLICT: 0}
_TRUSTED_FLOOR = TRUST[Provenance.RECONSTRUCTED]

SUPPORTED, NEEDS_REVIEW, CONFLICT = "supported", "needs_review", "conflict"


class ProposalError(WorldStoreError):
    """An acceptance/rejection the rules refuse."""


class UnknownProposalError(ProposalError):
    """No such proposal."""


class ProposalStateError(ProposalError):
    """The proposal was already accepted or rejected."""


@dataclass(frozen=True)
class ChangeAssessment:
    entity_id: str
    kind: str                                  # added | removed | modified
    verdict: str                               # supported | needs_review | conflict
    reasons: tuple = ()
    categories: tuple = ()
    new_evidence: tuple = ()                   # observation ids the candidate cites that the base entity lacked
    old_confidence: Optional[float] = None
    new_confidence: Optional[float] = None

    def to_dict(self) -> dict:
        return {"entity_id": self.entity_id, "kind": self.kind, "verdict": self.verdict, "reasons": list(self.reasons),
                "categories": list(self.categories), "new_evidence": list(self.new_evidence),
                "old_confidence": self.old_confidence, "new_confidence": self.new_confidence}


@dataclass
class ChangeProposal:
    proposal_id: str
    base_version_id: str
    candidate_artifact_uri: str
    candidate_artifact_hash: str
    assessments: List[ChangeAssessment]
    summary: dict
    candidate_validation: dict
    source_session_ids: tuple = ()
    status: str = "pending"                    # pending | accepted | rejected
    decision: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"proposal_id": self.proposal_id, "base_version_id": self.base_version_id,
                "candidate_artifact_uri": self.candidate_artifact_uri,
                "candidate_artifact_hash": self.candidate_artifact_hash,
                "assessments": [a.to_dict() for a in self.assessments], "summary": self.summary,
                "candidate_validation": self.candidate_validation, "source_session_ids": list(self.source_session_ids),
                "status": self.status, "decision": self.decision}

    @staticmethod
    def from_dict(d: dict) -> "ChangeProposal":
        return ChangeProposal(
            d["proposal_id"], d["base_version_id"], d["candidate_artifact_uri"], d["candidate_artifact_hash"],
            [ChangeAssessment(a["entity_id"], a["kind"], a["verdict"], tuple(a["reasons"]), tuple(a["categories"]),
                              tuple(a["new_evidence"]), a["old_confidence"], a["new_confidence"])
             for a in d["assessments"]],
            d["summary"], d["candidate_validation"], tuple(d.get("source_session_ids", ())), d["status"],
            d.get("decision", {}))


def _dir(store: WorldStore) -> Path:
    p = store._root / "proposals"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _save(store: WorldStore, p: ChangeProposal) -> None:
    _atomic_write_text(_dir(store) / f"{p.proposal_id}.json", json.dumps(p.to_dict(), indent=2, sort_keys=True))


def _delta(a, b) -> float:
    return max(abs(a.x - b.x), abs(a.y - b.y), abs(a.z - b.z)) if a is not None and b is not None else 0.0


def _moved(base: WorldIR, cand: WorldIR, old, new) -> bool:
    """True if placement or any referenced geometry moved beyond tolerance (identity-level edits count as moved)."""
    if (old.transform or None) != (new.transform or None):
        return True
    old_g, new_g = list(old.geometry_ids), list(new.geometry_ids)
    if old_g != new_g:
        return True
    for gid in old_g:
        a, b = base.geometries.get(gid), cand.geometries.get(gid)
        if a is None or b is None:
            return True
        if a.type != b.type or max(_delta(a.bounds_min, b.bounds_min),
                                   _delta(a.bounds_max, b.bounds_max)) > GEOMETRY_TOLERANCE_M:
            return True
    return False


@dataclass(frozen=True)
class _GeometryOnly:
    """Stand-in entity diff: the entity record is unchanged but a geometry it references was modified."""
    entity_id: str
    kind: ChangeKind = ChangeKind.MODIFIED
    changes: tuple = ()


def _assess(base: WorldIR, cand: WorldIR, diff) -> List[ChangeAssessment]:
    out: List[ChangeAssessment] = []
    changes = {d.entity_id: d for d in diff.entity_diffs}
    # A geometry edited IN PLACE (same ids) leaves the entity record identical: attribute it to every entity that
    # references it, or the change would be invisible to review.
    touched = {g.geometry_id for g in diff.geometry_diffs if g.kind is ChangeKind.MODIFIED}
    for eid, e in base.entities.items():
        if eid not in changes and eid in cand.entities and touched & set(e.geometry_ids):
            changes[eid] = _GeometryOnly(eid)
    for d in sorted(changes.values(), key=lambda x: x.entity_id):
        old, new = base.entities.get(d.entity_id), cand.entities.get(d.entity_id)
        if d.kind is ChangeKind.REMOVED:
            out.append(ChangeAssessment(d.entity_id, "removed", NEEDS_REVIEW,
                                        ("absent from the candidate; absence is not evidence of removal",),
                                        ("identity",), (), old.confidence, None))
            continue
        cites = tuple(sorted(o.id for o in new.observations))
        if d.kind is ChangeKind.ADDED:
            ok = bool(cites) or (new.provenance in (Provenance.OBSERVED, Provenance.RECONSTRUCTED)
                                 and new.confidence >= MIN_ADD_CONFIDENCE)
            out.append(ChangeAssessment(
                d.entity_id, "added", SUPPORTED if ok else NEEDS_REVIEW,
                () if ok else ("no cited observation and not an OBSERVED/RECONSTRUCTED entity above the confidence "
                               "floor",),
                ("identity",), cites, None, new.confidence))
            continue
        fields = {c.field for c in d.changes}
        cats = {"geometry" if f == "geometry_ids" else
                "transform" if f == "transform" else
                "semantic" if f in ("type", "name", "semantic_labels", "statement_state", "material_ids",
                                    "surface_ids", "component_ids") else
                "confidence" if f == "confidence" else
                "provenance" if f == "provenance" else
                "topology" if f == "relationships" else "other" for f in fields}
        geometry_moved = _moved(base, cand, old, new)
        if geometry_moved:
            cats.add("geometry")
        new_ev = tuple(sorted(set(cites) - {o.id for o in old.observations}))
        reasons: List[str] = []
        verdict = SUPPORTED
        if TRUST[new.provenance] < TRUST[old.provenance] and TRUST[old.provenance] >= _TRUSTED_FLOOR:
            verdict = CONFLICT
            reasons.append(f"would replace {old.provenance.value} data with weaker {new.provenance.value} data")
        elif geometry_moved and new.confidence < old.confidence:
            verdict = CONFLICT
            reasons.append(f"moves geometry/placement beyond {GEOMETRY_TOLERANCE_M} m while confidence drops "
                           f"{old.confidence:.2f} -> {new.confidence:.2f}")
        elif (geometry_moved or "semantic" in cats) and not new_ev:
            verdict = NEEDS_REVIEW
            reasons.append("change without new evidence: the candidate cites no observation the base lacked")
        out.append(ChangeAssessment(d.entity_id, "modified", verdict, tuple(reasons), tuple(sorted(cats)), new_ev,
                                    old.confidence, new.confidence))
    return out


def propose_update(store: WorldStore, base_version_id: str, candidate: WorldIR,
                   source_session_ids: Sequence[str] = ()) -> ChangeProposal:
    """Compare ``candidate`` with the stored ``base_version_id`` and persist a pending proposal. Idempotent: the same
    base + candidate always yields the same proposal id (an existing proposal, with its decision, is returned)."""
    base = store.load_version(base_version_id)
    payload = json.dumps(candidate.to_dict(), sort_keys=True).encode("utf-8")
    uri, digest = store._store.put(payload)
    pid = "p-" + hashlib.sha256(f"{base_version_id}:{digest}".encode()).hexdigest()[:12]
    path = _dir(store) / f"{pid}.json"
    if path.exists():
        return ChangeProposal.from_dict(json.loads(path.read_text(encoding="utf-8")))
    diff = diff_worlds(base, candidate)
    report = validate_world_ir(candidate)
    assessments = _assess(base, candidate, diff)
    counts = {v: sum(1 for a in assessments if a.verdict == v) for v in (SUPPORTED, NEEDS_REVIEW, CONFLICT)}
    proposal = ChangeProposal(
        pid, base_version_id, uri, digest, assessments,
        {**counts, "changes": len(assessments), "diff": diff.summary(), "categories": diff.categories()},
        {"valid": report.is_valid(), "errors": [str(i) for i in report.errors],
         "warnings": [str(i) for i in report.warnings]},
        tuple(source_session_ids))
    _save(store, proposal)
    return proposal


def get_proposal(store: WorldStore, proposal_id: str) -> ChangeProposal:
    path = _dir(store) / f"{proposal_id}.json"
    if not proposal_id.startswith("p-") or not path.is_file():
        raise UnknownProposalError(f"unknown proposal: {proposal_id!r}")
    return ChangeProposal.from_dict(json.loads(path.read_text(encoding="utf-8")))


def list_proposals(store: WorldStore) -> List[ChangeProposal]:
    return [ChangeProposal.from_dict(json.loads(p.read_text(encoding="utf-8")))
            for p in sorted(_dir(store).glob("p-*.json"))]


def reject_proposal(store: WorldStore, proposal_id: str, reason: str) -> ChangeProposal:
    p = get_proposal(store, proposal_id)
    if p.status != "pending":
        raise ProposalStateError(f"proposal {proposal_id} is already {p.status}")
    if not reason.strip():
        raise ProposalError("a rejection needs a reason")
    p.status, p.decision = "rejected", {"reason": reason}
    _save(store, p)
    return p


def prepare_acceptance(store: WorldStore, proposal_id: str, entity_ids: Optional[Iterable[str]] = None, *,
                       override_conflicts: Iterable[str] = (), reason: str = ""):
    """All acceptance checks, then the merged (validated) world. Returns (proposal, merged, chosen, overrides).
    Writes nothing: callers commit the world their own way (WorldStore directly, or the API's DB-mirrored commit)."""
    p = get_proposal(store, proposal_id)
    if p.status != "pending":
        raise ProposalStateError(f"proposal {proposal_id} is already {p.status}")
    by_id = {a.entity_id: a for a in p.assessments}
    overrides = set(override_conflicts)
    chosen = sorted(a.entity_id for a in p.assessments if a.verdict == SUPPORTED) if entity_ids is None         else sorted(set(entity_ids))
    unknown = [e for e in chosen + sorted(overrides) if e not in by_id]
    if unknown:
        raise ProposalError(f"not part of proposal {proposal_id}: {unknown}")
    blocked = [e for e in chosen if by_id[e].verdict == CONFLICT and e not in overrides]
    if blocked:
        raise ProposalError(f"conflicting changes need override_conflicts: {blocked}")
    if (overrides & set(chosen)) and not reason.strip():
        raise ProposalError("overriding a conflict needs a non-empty reason")
    if not chosen:
        raise ProposalError("nothing to apply: no supported changes and none selected explicitly")

    merged = WorldIR.from_dict(store.load_version(p.base_version_id).to_dict())
    cand = WorldIR.from_dict(json.loads(store._store.get(p.candidate_artifact_uri).decode("utf-8")))
    for eid in chosen:
        a = by_id[eid]
        if a.kind == "removed":
            gone = merged.entities.pop(eid)
            still = {g for e in merged.entities.values() for g in e.geometry_ids}
            for gid in gone.geometry_ids:
                if gid not in still:
                    merged.geometries.pop(gid, None)
        else:
            entity = cand.entities[eid]
            merged.entities[eid] = entity
            for gid in entity.geometry_ids:
                if gid in cand.geometries:
                    merged.geometries[gid] = cand.geometries[gid]
    report = validate_world_ir(merged)
    if not report.is_valid():
        raise ProposalError("merged world fails validation, nothing adopted: " + "; ".join(report.messages()[:5]))
    return p, merged, chosen, overrides


def record_acceptance(store: WorldStore, p: ChangeProposal, version_id: str, chosen: Sequence[str],
                      overrides: Iterable[str], reason: str) -> ChangeProposal:
    """Mark ``p`` accepted once its merged world has been committed as ``version_id``."""
    overrides = set(overrides)
    p.status = "accepted"
    p.decision = {"version_id": version_id, "applied": list(chosen),
                  "overridden_conflicts": sorted(overrides & set(chosen)), "reason": reason,
                  "skipped": sorted({a.entity_id for a in p.assessments} - set(chosen))}
    _save(store, p)
    return p


def accept_proposal(store: WorldStore, proposal_id: str, entity_ids: Optional[Iterable[str]] = None, *,
                    override_conflicts: Iterable[str] = (), reason: str = "") -> StoredVersion:
    """Apply selected changes of a pending proposal as a NEW version (parent = the proposal's base), directly on the
    WorldStore, with read-back verification (quarantined on failure)."""
    p, merged, chosen, overrides = prepare_acceptance(store, proposal_id, entity_ids,
                                                      override_conflicts=override_conflicts, reason=reason)
    version = store.save_version(merged, parent=p.base_version_id, source_session_ids=list(p.source_session_ids))
    failures = store.verify_version(version.version_id)
    if failures:
        store.quarantine_version(version.version_id)
        raise ProposalError(f"saved version failed read-back verification and was quarantined: {failures}")
    record_acceptance(store, p, version.version_id, chosen, overrides, reason)
    return version


# ---------------------------------------------------------------------------
# Compatibility / Pipeline Workflow
# ---------------------------------------------------------------------------

import threading
import time
import uuid
from enum import Enum
from typing import Any, Callable, Dict


class ProposalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    APPLIED = "applied"


@dataclass
class Proposal:
    proposal_id: str
    kind: str
    payload: Dict[str, Any]
    status: ProposalStatus = ProposalStatus.PENDING
    created_by: str = "system"
    created_at: float = field(default_factory=time.time)
    reviewed_by: Optional[str] = None
    review_reason: str = ""
    applied_at: Optional[float] = None
    apply_result: Any = None


class ProposalPipeline:
    """Thread-safe in-memory proposal store + review/apply workflow."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._proposals: Dict[str, Proposal] = {}
        self._order: List[str] = []

    def submit(self, kind: str, payload: Optional[Dict[str, Any]] = None,
               created_by: str = "system") -> Proposal:
        if not kind:
            raise ProposalError("proposal kind must be non-empty")
        p = Proposal(
            proposal_id=str(uuid.uuid4()),
            kind=kind,
            payload=dict(payload or {}),
            created_by=created_by,
        )
        with self._lock:
            self._proposals[p.proposal_id] = p
            self._order.append(p.proposal_id)
        return p

    def get(self, proposal_id: str) -> Optional[Proposal]:
        with self._lock:
            return self._proposals.get(proposal_id)

    def list(self, status: Optional[ProposalStatus] = None) -> List[Proposal]:
        with self._lock:
            out = [self._proposals[pid] for pid in self._order]
        if status is not None:
            out = [p for p in out if p.status == status]
        return out

    def approve(self, proposal_id: str, reviewer: str = "reviewer",
                reason: str = "") -> Proposal:
        with self._lock:
            p = self._require(proposal_id)
            if p.status != ProposalStatus.PENDING:
                raise ProposalError(
                    f"cannot approve proposal in status {p.status.value}"
                )
            p.status = ProposalStatus.APPROVED
            p.reviewed_by = reviewer
            p.review_reason = reason
        return p

    def reject(self, proposal_id: str, reviewer: str = "reviewer",
               reason: str = "") -> Proposal:
        with self._lock:
            p = self._require(proposal_id)
            if p.status != ProposalStatus.PENDING:
                raise ProposalError(
                    f"cannot reject proposal in status {p.status.value}"
                )
            p.status = ProposalStatus.REJECTED
            p.reviewed_by = reviewer
            p.review_reason = reason
        return p

    def apply(self, proposal_id: str,
              apply_fn: Callable[[Proposal], Any]) -> Any:
        with self._lock:
            p = self._require(proposal_id)
            if p.status == ProposalStatus.REJECTED:
                raise ProposalError("cannot apply a rejected proposal")
            if p.status == ProposalStatus.APPLIED:
                raise ProposalError("proposal already applied")
            if p.status != ProposalStatus.APPROVED:
                raise ProposalError(
                    "proposal must be approved before application"
                )
            p.status = ProposalStatus.APPLIED
            p.applied_at = time.time()
        try:
            result = apply_fn(p)
        except Exception:
            with self._lock:
                p.status = ProposalStatus.APPROVED
                p.applied_at = None
            raise
        with self._lock:
            p.apply_result = result
        return result

    def _require(self, proposal_id: str) -> Proposal:
        p = self._proposals.get(proposal_id)
        if p is None:
            raise ProposalError(f"unknown proposal: {proposal_id}")
        return p


_default_pipeline: Optional[ProposalPipeline] = None
_default_lock = threading.Lock()


def get_default_pipeline() -> ProposalPipeline:
    global _default_pipeline
    with _default_lock:
        if _default_pipeline is None:
            _default_pipeline = ProposalPipeline()
        return _default_pipeline


def reset_default_pipeline() -> ProposalPipeline:
    global _default_pipeline
    with _default_lock:
        _default_pipeline = ProposalPipeline()
    return _default_pipeline
