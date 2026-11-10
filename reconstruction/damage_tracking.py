"""Physical damage and deterioration tracking (N-24).

Tracks supported cracks, stains, staining, deformation and deterioration
over repeated captures. Distinguishes newly observed defects from existing
conditions while accounting for lighting/viewpoint differences.

Honesty rules:
* A defect is only recorded when visual evidence supports it; absence of
  evidence is "unobserved", not "defect-free".
* Lighting/viewpoint differences are flagged as uncertainty, not treated as
  damage.
* Every damage record carries confidence, evidence ids, and the capture it
  was observed in.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence

from world_ir import Entity, EntityType


@dataclass(frozen=True)
class DamageRecord:
    """One observed defect or deterioration on one object."""
    object_id: str
    object_type: str
    object_name: Optional[str]
    damage_type: str                 # "crack" | "stain" | "deformation" | "surface_damage" | "other"
    location_m: Optional[Tuple[float, float, float]]  # where on the object
    severity: str                    # "minor" | "moderate" | "major" | "unknown"
    confidence: float                # 0..1
    first_observed: str              # ISO-8601 UTC capture id / timestamp
    last_observed: str
    evidence_ids: List[str]
    lighting_note: Optional[str]     # e.g. "low light, confidence reduced"
    viewpoint_note: Optional[str]    # e.g. "oblique angle, partial occlusion"
    is_new: bool                     # True if not present in a previous version

    def to_dict(self) -> dict:
        d = asdict(self)
        d["location_m"] = None if self.location_m is None else list(self.location_m)
        return d


@dataclass(frozen=True)
class DamageReport:
    """All damage observations for one world version."""
    objects: Dict[str, List[DamageRecord]]
    new_since_previous: List[str]    # object ids with newly observed damage
    resolved_since_previous: List[str]  # object ids where prior damage no longer observed
    provisional: bool = True


class DamageTracker:
    """Tracks physical damage across world versions."""

    MIN_CONFIDENCE = 0.40

    def __init__(self):
        self._history: Dict[str, List[DamageRecord]] = {}
        self._current: Dict[str, DamageRecord] = {}

    def record_damage(
        self,
        object_id: str,
        object_type: EntityType,
        object_name: Optional[str],
        damage_type: str,
        location_m: Optional[Sequence[float]] = None,
        severity: str = "unknown",
        confidence: float = 0.5,
        capture_id: str = "",
        evidence_ids: Optional[Sequence[str]] = None,
        lighting_note: Optional[str] = None,
        viewpoint_note: Optional[str] = None,
        timestamp: Optional[str] = None,
    ) -> Optional[DamageRecord]:
        """Record a damage observation for one object in one capture."""
        now = timestamp or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        conf = max(0.0, min(1.0, confidence))
        if conf < self.MIN_CONFIDENCE:
            return None

        loc = None if location_m is None else tuple(float(c) for c in location_m)
        prev = self._current.get(object_id)
        is_new = prev is None

        record = DamageRecord(
            object_id=object_id, object_type=object_type.value,
            object_name=object_name, damage_type=damage_type,
            location_m=loc, severity=severity, confidence=round(conf, 3),
            first_observed=capture_id or now, last_observed=capture_id or now,
            evidence_ids=list(evidence_ids or []),
            lighting_note=lighting_note, viewpoint_note=viewpoint_note,
            is_new=is_new,
        )

        self._history.setdefault(object_id, []).append(record)
        self._current[object_id] = record
        return record

    def clear_damage(self, object_id: str) -> None:
        """Mark that no damage is observed for this object in the current version."""
        self._current.pop(object_id, None)

    def report_for_version(
        self,
        previous_current: Optional[Dict[str, DamageRecord]] = None,
    ) -> DamageReport:
        objects: Dict[str, List[DamageRecord]] = {}
        new_ids: List[str] = []
        resolved_ids: List[str] = []

        prev_ids = set(previous_current or {})
        for oid, rec in self._current.items():
            objects.setdefault(oid, []).append(rec)
            if oid not in prev_ids:
                new_ids.append(oid)

        for oid in prev_ids:
            if oid not in self._current:
                resolved_ids.append(oid)

        return DamageReport(
            objects=objects,
            new_since_previous=sorted(new_ids),
            resolved_since_previous=sorted(resolved_ids),
            provisional=True,
        )

    def history_for(self, object_id: str) -> Optional[List[DamageRecord]]:
        return list(self._history.get(object_id, []))


__all__ = ["DamageTracker", "DamageRecord", "DamageReport"]