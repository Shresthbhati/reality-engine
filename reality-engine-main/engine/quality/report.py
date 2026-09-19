"""World Quality Report: aggregate confidence/completeness metrics over a WorldIR.

Directive items #29 (QUALITY ENGINE) and #64 (WORLD COMPLETENESS): answer
"how confident are we?" with real numbers computed from WorldIR's actual
entity data -- not fabricated scores. Reuses Inspector for entity-level
queries (low_confidence_entities) rather than re-implementing them; the
provenance/canonical/relationship scans here are a genuinely new aggregate
layer, not duplicated from Inspector's summary().
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from engine.inspector.inspector import Inspector
from provenance import Provenance, Provenanced
from world_ir import RelationshipKind, WorldIR


@dataclass(frozen=True)
class WorldQualityReport:
    entity_count: int
    provenance_counts: Dict[str, int]
    mean_confidence: Optional[float]
    canonical_fraction: Optional[float]
    low_confidence_ids: List[str] = field(default_factory=list)
    unresolved_relationship_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "entity_count": self.entity_count,
            "provenance_counts": dict(self.provenance_counts),
            "mean_confidence": self.mean_confidence,
            "canonical_fraction": self.canonical_fraction,
            "low_confidence_ids": list(self.low_confidence_ids),
            "unresolved_relationship_ids": list(self.unresolved_relationship_ids),
        }

    def summary_text(self) -> str:
        if self.entity_count == 0:
            return "Entities: 0\nNo entities in this world -- nothing to score."

        canon_pct = round((self.canonical_fraction or 0.0) * 100)
        mean_conf = round(self.mean_confidence, 2) if self.mean_confidence is not None else "n/a"
        lines = [
            f"Entities: {self.entity_count} ({canon_pct}% canonical, mean confidence {mean_conf})",
            "Provenance: " + ", ".join(f"{k}={v}" for k, v in self.provenance_counts.items() if v),
            f"Low-confidence (<0.5): {len(self.low_confidence_ids)} entities",
            f"Unresolved relationships: {len(self.unresolved_relationship_ids)} entities",
        ]
        return "\n".join(lines)


def provenance_breakdown(world: WorldIR) -> Dict[str, int]:
    counts = {p.value: 0 for p in Provenance}
    for entity in world.entities.values():
        counts[entity.provenance.value] += 1
    return counts


def mean_confidence(world: WorldIR) -> Optional[float]:
    entities = list(world.entities.values())
    if not entities:
        return None
    return sum(e.confidence for e in entities) / len(entities)


def canonical_fraction(world: WorldIR) -> Optional[float]:
    entities = list(world.entities.values())
    if not entities:
        return None
    canonical = sum(
        1 for e in entities
        if Provenanced(value=None, provenance=e.provenance).is_canonical()
    )
    return canonical / len(entities)


def low_confidence_entity_ids(world: WorldIR, threshold: float = 0.5) -> List[str]:
    return [e.id for e in Inspector(world).low_confidence_entities(threshold)]


def unresolved_relationship_entity_ids(world: WorldIR) -> List[str]:
    return [
        entity.id
        for entity in world.entities.values()
        if any(rel.kind == RelationshipKind.UNKNOWN for rel in entity.relationships)
    ]


def compute_quality_report(world: WorldIR) -> WorldQualityReport:
    return WorldQualityReport(
        entity_count=len(world.entities),
        provenance_counts=provenance_breakdown(world),
        mean_confidence=mean_confidence(world),
        canonical_fraction=canonical_fraction(world),
        low_confidence_ids=low_confidence_entity_ids(world),
        unresolved_relationship_ids=unresolved_relationship_entity_ids(world),
    )
