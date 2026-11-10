"""Evidence and provenance governance (N-40).

Tracks every spatial claim back to source images, sensor observations,
reconstruction stages, manual edits or inference. Preserves the distinction
between observed, reconstructed, inferred, generated, unknown and conflicting
data.

Honesty rules:
* Every entity carries a provenance enum; nothing defaults to "observed".
* The provenance chain is queryable end-to-end: media -> frame -> calibration
  -> pose -> depth -> point -> hypothesis -> entity -> edit -> export.
* Conflicting provenance for the same entity is recorded, not suppressed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class ProvenanceKind(Enum):
    OBSERVED = "observed"
    RECONSTRUCTED = "reconstructed"
    INFERRED = "inferred"
    GENERATED = "generated"
    UNKNOWN = "unknown"
    CONFLICT = "conflict"


@dataclass(frozen=True)
class ProvenanceLink:
    """One link in the provenance chain."""
    step: str                      # media | frame | calibration | pose | depth |
                                   # point | hypothesis | entity | edit | export
    source_id: str                 # id of the thing at this step
    target_id: str                 # id of the thing produced
    kind: ProvenanceKind
    confidence: float              # 0..1
    timestamp: Optional[str] = None
    note: Optional[str] = None


@dataclass(frozen=True)
class EntityProvenance:
    """Complete provenance for one entity."""
    entity_id: str
    entity_type: str
    primary_kind: ProvenanceKind
    confidence: float
    chain: List[ProvenanceLink]
    conflicting: List[ProvenanceKind] = field(default_factory=list)
    note: Optional[str] = None

    def is_observed(self) -> bool:
        return self.primary_kind == ProvenanceKind.OBSERVED and not self.conflicting

    def is_inferred(self) -> bool:
        return self.primary_kind in (ProvenanceKind.INFERRED, ProvenanceKind.GENERATED)

    def is_trustworthy(self, threshold: float = 0.7) -> bool:
        return self.confidence >= threshold and self.primary_kind == ProvenanceKind.OBSERVED


@dataclass(frozen=True)
class GovernanceReport:
    """Provenance audit for a world."""
    entities: Dict[str, EntityProvenance]
    unprovenanced: List[str]
    conflicting: List[str]
    summary: Dict[str, int]
    provisional: bool = True


class EvidenceGovernance:
    """Tracks and audits provenance across the reconstruction pipeline."""

    def __init__(self):
        self._chains: Dict[str, List[ProvenanceLink]] = {}
        self._kinds: Dict[str, ProvenanceKind] = {}
        self._confidences: Dict[str, float] = {}
        self._conflicts: Dict[str, List[ProvenanceKind]] = {}

    def record_link(
        self,
        entity_id: str,
        step: str,
        source_id: str,
        target_id: str,
        kind: ProvenanceKind,
        confidence: float,
        timestamp: Optional[str] = None,
        note: Optional[str] = None,
    ) -> None:
        link = ProvenanceLink(
            step=step, source_id=source_id, target_id=target_id,
            kind=kind, confidence=max(0.0, min(1.0, confidence)),
            timestamp=timestamp, note=note,
        )
        self._chains.setdefault(entity_id, []).append(link)

        prev = self._kinds.get(entity_id)
        if prev is None:
            self._kinds[entity_id] = kind
        elif prev != kind:
            self._conflicts.setdefault(entity_id, [])
            if kind not in self._conflicts[entity_id]:
                self._conflicts[entity_id].append(kind)

        self._confidences[entity_id] = max(
            self._confidences.get(entity_id, 0.0),
            max(0.0, min(1.0, confidence)),
        )

    def provenance_for(self, entity_id: str) -> Optional[EntityProvenance]:
        chain = self._chains.get(entity_id)
        if chain is None and entity_id not in self._kinds:
            return None
        kind = self._kinds.get(entity_id, ProvenanceKind.UNKNOWN)
        conf = self._confidences.get(entity_id, 0.0)
        conflicts = list(self._conflicts.get(entity_id, []))
        note = self._derive_note(kind, conf, conflicts)
        return EntityProvenance(
            entity_id=entity_id, entity_type="unknown",
            primary_kind=kind, confidence=round(conf, 3),
            chain=list(chain), conflicting=conflicts, note=note,
        )

    def audit_world(self, entities: Dict[str, Entity]) -> GovernanceReport:
        entity_provs: Dict[str, EntityProvenance] = {}
        unprovenanced: List[str] = []
        conflicting: List[str] = []
        summary: Dict[str, int] = {k.value: 0 for k in ProvenanceKind}

        for eid, ent in entities.items():
            prov = self.provenance_for(eid)
            if prov is None:
                unprovenanced.append(eid)
                summary[ProvenanceKind.UNKNOWN.value] += 1
                entity_provs[eid] = EntityProvenance(
                    entity_id=eid, entity_type=ent.type if hasattr(ent.type, 'value') else str(ent.type),
                    primary_kind=ProvenanceKind.UNKNOWN, confidence=0.0,
                    chain=[], conflicting=[], note="No provenance recorded.",
                )
                continue
            entity_provs[eid] = EntityProvenance(
                entity_id=eid,
                entity_type=ent.type.value if hasattr(ent.type, 'value') else str(ent.type),
                primary_kind=prov.primary_kind, confidence=prov.confidence,
                chain=prov.chain, conflicting=prov.conflicting, note=prov.note,
            )
            summary[prov.primary_kind.value] += 1
            if prov.conflicting:
                conflicting.append(eid)

        return GovernanceReport(
            entities=entity_provs, unprovenanced=unprovenanced,
            conflicting=conflicting, summary=summary, provisional=True,
        )

    @staticmethod
    def _derive_note(kind: ProvenanceKind, conf: float, conflicts: List[ProvenanceKind]) -> str:
        parts = [f"{kind.value} (confidence {conf:.2f})"]
        if conflicts:
            parts.append(f"also seen as: {', '.join(c.value for c in conflicts)}")
        if kind == ProvenanceKind.UNKNOWN:
            parts.append("treat as untrusted")
        elif kind in (ProvenanceKind.INFERRED, ProvenanceKind.GENERATED):
            parts.append("not directly observed")
        return "; ".join(parts)


__all__ = ["EvidenceGovernance", "EntityProvenance", "ProvenanceLink", "GovernanceReport", "ProvenanceKind"]