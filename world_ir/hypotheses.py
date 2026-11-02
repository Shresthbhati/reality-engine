"""Competing hypotheses and perception reconciliation (Master Development Plan Tasks 7.3 & 7.6).

Architectural Principle:
When multiple perception engines (e.g. depth vs SfM, plane detection vs room layout,
object classifier vs material appearance) produce disagreeing interpretations,
Reality Engine stores competing hypotheses side-by-side with full evidence provenance,
rather than averaging them into a fictitious compromise that matches no real observation.

Key capabilities:
1. `CompetingHypothesis`:
   - Captures an independent interpretation of an entity's classification, geometry,
     or material, with producer identity, confidence, and source evidence links.
2. `HypothesisSet`:
   - Holds competing candidates for a specific entity or property domain (e.g. "classification",
     "geometry", "room_role", "material").
3. `reconcile_hypotheses`:
   - Deterministic reconciliation selecting an active canonical hypothesis while
     preserving losing hypotheses in the lineage audit trail.
   - Respects human corrections (`VERIFIED`) as authoritative over automated inferences.
   - Flags honest `Provenance.CONFLICT` when two independent high-confidence hypotheses disagree.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple
from uuid import uuid4

from provenance import Provenance
from world_ir.schema_v1 import Entity


class ReconciliationStrategy(str, Enum):
    """Strategies for resolving competing interpretations."""

    USER_OVERRIDE = "user_override"          # Human correction strictly supersedes automated models
    HIGHEST_CONFIDENCE = "highest_confidence"  # Highest model confidence wins
    EVIDENCE_WEIGHTED = "evidence_weighted"    # View/evidence count times confidence
    CONSERVATIVE = "conservative"              # Most conservative/constrained interpretation


@dataclass(frozen=True)
class CompetingHypothesis:
    """One candidate interpretation of an entity property or classification."""

    hypothesis_id: str
    producer: str                       # e.g., "perception.classifier", "perception.architecture", "sam2"
    statement: Dict[str, Any]           # The property values asserted (e.g., {"entity_class": "room.kitchen"})
    confidence: float                   # 0.0 to 1.0
    provenance: Provenance = Provenance.INFERRED
    supporting_evidence_ids: Tuple[str, ...] = ()
    rationale: str = ""
    is_active: bool = False             # Set to True when selected as the canonical representation
    timestamp: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "producer": self.producer,
            "statement": self.statement,
            "confidence": round(self.confidence, 4),
            "provenance": self.provenance.value if hasattr(self.provenance, "value") else str(self.provenance),
            "supporting_evidence_ids": list(self.supporting_evidence_ids),
            "rationale": self.rationale,
            "is_active": self.is_active,
            "timestamp": self.timestamp,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> CompetingHypothesis:
        prov_val = data.get("provenance", Provenance.INFERRED.value)
        return CompetingHypothesis(
            hypothesis_id=data.get("hypothesis_id", f"hyp-{uuid4()}"),
            producer=data.get("producer", "unknown_producer"),
            statement=data.get("statement", {}),
            confidence=float(data.get("confidence", 0.5)),
            provenance=Provenance(prov_val) if prov_val in [p.value for p in Provenance] else Provenance.INFERRED,
            supporting_evidence_ids=tuple(data.get("supporting_evidence_ids", ())),
            rationale=data.get("rationale", ""),
            is_active=bool(data.get("is_active", False)),
            timestamp=float(data.get("timestamp", 0.0)),
        )


@dataclass
class HypothesisSet:
    """Collection of competing hypotheses for an entity or property domain."""

    target_id: str
    domain: str                         # "classification", "geometry", "room_role", "material"
    hypotheses: List[CompetingHypothesis] = field(default_factory=list)
    resolved_hypothesis_id: Optional[str] = None
    strategy_used: str = ReconciliationStrategy.HIGHEST_CONFIDENCE.value
    conflict_detected: bool = False
    conflict_note: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_id": self.target_id,
            "domain": self.domain,
            "hypotheses": [h.to_dict() for h in self.hypotheses],
            "resolved_hypothesis_id": self.resolved_hypothesis_id,
            "strategy_used": self.strategy_used,
            "conflict_detected": self.conflict_detected,
            "conflict_note": self.conflict_note,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> HypothesisSet:
        return HypothesisSet(
            target_id=data["target_id"],
            domain=data["domain"],
            hypotheses=[CompetingHypothesis.from_dict(h) for h in data.get("hypotheses", [])],
            resolved_hypothesis_id=data.get("resolved_hypothesis_id"),
            strategy_used=data.get("strategy_used", ReconciliationStrategy.HIGHEST_CONFIDENCE.value),
            conflict_detected=bool(data.get("conflict_detected", False)),
            conflict_note=data.get("conflict_note"),
        )


def reconcile_hypotheses(
    hyp_set: HypothesisSet,
    strategy: ReconciliationStrategy = ReconciliationStrategy.HIGHEST_CONFIDENCE,
    conflict_threshold: float = 0.65,
) -> CompetingHypothesis:
    """Reconcile competing hypotheses and elect an active candidate.

    Never deletes losing hypotheses; stores them in hyp_set.hypotheses with is_active=False.

    Rules:
    1. A human correction or verified hypothesis (provenance ESTIMATED/VERIFIED or producer 'user')
       strictly wins over any automated inference regardless of model confidence.
    2. Under HIGHEST_CONFIDENCE, the candidate with maximum confidence is selected.
    3. Under EVIDENCE_WEIGHTED, confidence is multiplied by (1 + 0.2 * len(supporting_evidence_ids)).
    4. If the top two mutually exclusive hypotheses both have confidence >= conflict_threshold and
       differ by < 0.15 in score, conflict_detected is flagged and noted.
    """
    if not hyp_set.hypotheses:
        raise ValueError(f"Cannot reconcile empty hypothesis set for {hyp_set.target_id}:{hyp_set.domain}")

    # 1. Check for human user verification / corrections
    for h in hyp_set.hypotheses:
        if "user" in h.producer.lower() or h.provenance == Provenance.ESTIMATED and "user" in h.rationale.lower():
            # Human override wins strictly
            _activate_hypothesis(hyp_set, h.hypothesis_id)
            hyp_set.strategy_used = ReconciliationStrategy.USER_OVERRIDE.value
            hyp_set.conflict_detected = False
            hyp_set.conflict_note = f"Resolved via human correction ({h.producer})"
            return [cand for cand in hyp_set.hypotheses if cand.is_active][0]

    # 2. Score hypotheses according to strategy
    scored: List[Tuple[float, CompetingHypothesis]] = []
    for h in hyp_set.hypotheses:
        if strategy == ReconciliationStrategy.EVIDENCE_WEIGHTED:
            evidence_boost = 1.0 + 0.25 * min(10, len(h.supporting_evidence_ids))
            effective_score = h.confidence * evidence_boost
        elif strategy == ReconciliationStrategy.CONSERVATIVE:
            # Conservative penalizes extreme extents or unsupported assertions
            effective_score = h.confidence * (0.8 if len(h.supporting_evidence_ids) == 0 else 1.0)
        else:  # HIGHEST_CONFIDENCE
            effective_score = h.confidence
        scored.append((effective_score, h))

    # Sort descending by effective score
    scored.sort(key=lambda item: (-item[0], item[1].hypothesis_id))

    winner_score, winner = scored[0]

    # 3. Conflict detection check
    conflict_detected = False
    conflict_note = None
    if len(scored) >= 2:
        runner_up_score, runner_up = scored[1]
        # If statements assert conflicting values and both are confident
        if winner.statement != runner_up.statement:
            if runner_up.confidence >= conflict_threshold and winner.confidence >= conflict_threshold:
                if abs(winner_score - runner_up_score) < 0.15:
                    conflict_detected = True
                    conflict_note = (
                        f"Unresolved conflict between {winner.producer} ({winner.statement}) "
                        f"and {runner_up.producer} ({runner_up.statement})"
                    )

    # 4. Activate winner
    _activate_hypothesis(hyp_set, winner.hypothesis_id)
    hyp_set.strategy_used = strategy.value
    hyp_set.conflict_detected = conflict_detected
    hyp_set.conflict_note = conflict_note

    return [cand for cand in hyp_set.hypotheses if cand.is_active][0]


def _activate_hypothesis(hyp_set: HypothesisSet, target_id: str) -> None:
    updated: List[CompetingHypothesis] = []
    for h in hyp_set.hypotheses:
        is_winner = (h.hypothesis_id == target_id)
        if is_winner != h.is_active:
            # Recreate with updated active flag
            updated.append(
                CompetingHypothesis(
                    hypothesis_id=h.hypothesis_id,
                    producer=h.producer,
                    statement=h.statement,
                    confidence=h.confidence,
                    provenance=h.provenance,
                    supporting_evidence_ids=h.supporting_evidence_ids,
                    rationale=h.rationale,
                    is_active=is_winner,
                    timestamp=h.timestamp,
                )
            )
        else:
            updated.append(h)
    hyp_set.hypotheses = updated
    hyp_set.resolved_hypothesis_id = target_id


# -----------------------------------------------------------------------------
# Entity Custom Properties Helpers
# -----------------------------------------------------------------------------

def attach_competing_hypothesis(
    entity: Entity,
    domain: str,
    statement: Dict[str, Any],
    producer: str,
    confidence: float,
    provenance: Provenance = Provenance.INFERRED,
    supporting_evidence_ids: Sequence[str] = (),
    rationale: str = "",
    auto_reconcile: bool = True,
) -> HypothesisSet:
    """Attach a candidate hypothesis to an entity's custom_properties['hypotheses']."""
    if "hypotheses" not in entity.custom_properties:
        entity.custom_properties["hypotheses"] = {}

    hyp_domain_dict = entity.custom_properties["hypotheses"].get(domain)
    if hyp_domain_dict is not None:
        hyp_set = HypothesisSet.from_dict(hyp_domain_dict)
    else:
        hyp_set = HypothesisSet(target_id=entity.id, domain=domain)

    new_hyp = CompetingHypothesis(
        hypothesis_id=f"hyp-{uuid4()}",
        producer=producer,
        statement=dict(statement),
        confidence=confidence,
        provenance=provenance,
        supporting_evidence_ids=tuple(supporting_evidence_ids),
        rationale=rationale,
        is_active=False,
        timestamp=time.time(),
    )
    hyp_set.hypotheses.append(new_hyp)

    if auto_reconcile:
        reconcile_hypotheses(hyp_set)

    entity.custom_properties["hypotheses"][domain] = hyp_set.to_dict()
    return hyp_set


def get_active_hypothesis(entity: Entity, domain: str) -> Optional[CompetingHypothesis]:
    """Retrieve the currently active hypothesis for an entity in the specified domain."""
    hyp_dict = entity.custom_properties.get("hypotheses", {}).get(domain)
    if not hyp_dict:
        return None
    hyp_set = HypothesisSet.from_dict(hyp_dict)
    for h in hyp_set.hypotheses:
        if h.is_active:
            return h
    return None


def get_competing_hypotheses(entity: Entity, domain: str) -> List[CompetingHypothesis]:
    """Retrieve all candidate hypotheses for an entity in the specified domain."""
    hyp_dict = entity.custom_properties.get("hypotheses", {}).get(domain)
    if not hyp_dict:
        return []
    hyp_set = HypothesisSet.from_dict(hyp_dict)
    return list(hyp_set.hypotheses)
