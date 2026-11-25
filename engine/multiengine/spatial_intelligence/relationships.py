"""Object Relationship Intelligence and Spatial Semantics (Sprint 6 / Sprint 7).

Implements evidence-backed spatial relationships, physical attachment verification,
and semantic contradiction detection across 16 canonical relationship types.
Does not infer physical attachment merely from proximity.

Sprint 7 additions:
- Scale-state guards: metric-threshold-dependent relationships are only applied when
  BOTH entities have ``scale_state == "metric_anchored"`` and ``is_physically_metric``
  is True (or the equivalent attribute on the entity is set). For non-metric entities,
  absolute-distance thresholds are skipped and relationships are either omitted or
  downgraded to ``validation_status="hypothesized"``.
- Cross-frame detection: if the two entities advertise different coordinate frames,
  metric relationships are marked ``"hypothesized"`` rather than ``"validated"``.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from engine.multiengine.spatial_intelligence.taxonomy import (
    TaxonomyClass,
    are_semantically_compatible,
    get_subparts_for_class,
    resolve_taxonomy_class,
)

# ---------------------------------------------------------------------------
# Scale-state helpers
# ---------------------------------------------------------------------------

#: Scale-state values that represent physically calibrated metric coordinates.
_METRIC_SCALE_STATES: frozenset[str] = frozenset({
    "metric_anchored",
    "METRIC_ANCHORED",
})


def _entity_is_metric(entity: Any) -> bool:
    """Returns True only when the entity carries physically-anchored metric coordinates.

    Checks ``scale_state`` (string attribute, possibly an enum) and
    ``is_physically_metric`` (bool) on the entity.  SfM-gauge, relative-depth
    and synthetic-scale entities are explicitly NOT metric.
    """
    ss = getattr(entity, "scale_state", None)
    if ss is None:
        return False
    ss_str = str(ss).upper().replace("-", "_")
    is_metric_state = "METRIC_ANCHORED" in ss_str
    is_physical = bool(getattr(entity, "is_physically_metric", False))
    return is_metric_state and is_physical


def _entities_share_frame(ent_a: Any, ent_b: Any) -> bool:
    """Returns True when both entities use the same coordinate frame string."""
    frame_a = getattr(ent_a, "coordinate_frame", None)
    frame_b = getattr(ent_b, "coordinate_frame", None)
    if frame_a is None or frame_b is None:
        return True  # Unknown → assume compatible (conservative)
    return frame_a == frame_b


def _metric_validation_status(ent_a: Any, ent_b: Any) -> str:
    """Returns the appropriate validation_status for a metric-threshold-based relationship.

    * ``"validated"``  — both entities are metric and share the same frame.
    * ``"hypothesized"`` — frames differ, or one entity is not metric.
    """
    if _entity_is_metric(ent_a) and _entity_is_metric(ent_b) and _entities_share_frame(ent_a, ent_b):
        return "validated"
    return "hypothesized"


class RelationshipType(str, Enum):
    """Canonical spatial and structural relationship types."""

    ATTACHED_TO = "attached_to"
    MOUNTED_ON = "mounted_on"
    EMBEDDED_IN = "embedded_in"
    SUPPORTED_BY = "supported_by"
    INSIDE = "inside"
    CONTAINS = "contains"
    ADJACENT_TO = "adjacent_to"
    INTERSECTS = "intersects"
    ABOVE = "above"
    BELOW = "below"
    IN_FRONT_OF = "in_front_of"
    BEHIND = "behind"
    CONNECTED_TO = "connected_to"
    PART_OF = "part_of"
    OPENS_INTO = "opens_into"
    ALIGNED_WITH = "aligned_with"


@dataclass(frozen=True)
class EvidenceRelationship:
    """Directed relationship between two physical entities grounded in geometric and semantic evidence."""

    source_id: str
    target_id: str
    relation_type: RelationshipType
    confidence: float = 0.5
    uncertainty: float = 0.5
    supporting_observations: tuple[str, ...] = ()
    geometric_evidence: dict[str, Any] = field(default_factory=dict)
    validation_status: str = "validated"  # "validated", "hypothesized", "conflicted", "rejected"
    basis: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type.value,
            "confidence": round(self.confidence, 4),
            "uncertainty": round(self.uncertainty, 4),
            "supporting_observations": list(self.supporting_observations),
            "geometric_evidence": self.geometric_evidence,
            "validation_status": self.validation_status,
            "basis": self.basis,
        }


# Mapping of inverse relations
_INVERSE_RELATIONS: dict[RelationshipType, RelationshipType] = {
    RelationshipType.CONTAINS: RelationshipType.INSIDE,
    RelationshipType.INSIDE: RelationshipType.CONTAINS,
    RelationshipType.ABOVE: RelationshipType.BELOW,
    RelationshipType.BELOW: RelationshipType.ABOVE,
    RelationshipType.IN_FRONT_OF: RelationshipType.BEHIND,
    RelationshipType.BEHIND: RelationshipType.IN_FRONT_OF,
}

# Incompatible pairs between the same source and target
_INCOMPATIBLE_PAIRS: frozenset[frozenset[RelationshipType]] = frozenset({
    frozenset({RelationshipType.PART_OF, RelationshipType.CONTAINS}),
    frozenset({RelationshipType.INSIDE, RelationshipType.CONTAINS}),
    frozenset({RelationshipType.ABOVE, RelationshipType.BELOW}),
    frozenset({RelationshipType.IN_FRONT_OF, RelationshipType.BEHIND}),
    frozenset({RelationshipType.PART_OF, RelationshipType.SUPPORTED_BY}),  # A subpart is structurally part of host, not an independent rested entity
})


class RelationshipEngine:
    """Infers, verifies, and audits spatial relationships between physical entities."""

    def __init__(self) -> None:
        pass

    def evaluate_relationships(
        self,
        entities: Sequence[Any],
        wall_contact_threshold_m: float = 0.15,
        support_contact_threshold_m: float = 0.08,
    ) -> list[EvidenceRelationship]:
        """Evaluates pairwise physical and structural relationships across entities.

        Scale-safety contract (Sprint 7):
        - Relationships whose inference depends on absolute metric distance thresholds
          (wall contact, support contact, adjacency radius, above/below vertical offset)
          are only ``validation_status="validated"`` when BOTH entities carry
          ``scale_state="metric_anchored"`` and ``is_physically_metric=True``.
        - For non-metric entities (relative-depth, SfM-gauge, synthetic) metric-distance
          checks are still performed, but the resulting relationship is marked
          ``validation_status="hypothesized"`` so downstream code can filter accordingly.
        - Cross-frame relationships (entities in different coordinate frames) are
          likewise marked ``"hypothesized"``.
        - Scale-invariant relationships (structural hierarchy from explicit parent/child
          attributes, surface-normal alignment) are unaffected.
        """
        results: list[EvidenceRelationship] = []
        n = len(entities)

        for i in range(n):
            ent_a = entities[i]
            id_a = getattr(ent_a, "entity_id", getattr(ent_a, "canonical_entity_id", f"ent_{i}"))
            cls_a = getattr(ent_a, "class_name", getattr(ent_a, "primary_class", "unknown")).lower()
            tax_a = resolve_taxonomy_class(cls_a)
            pos_a = getattr(ent_a, "position_m", getattr(ent_a, "position", None))
            bounds_min_a = getattr(ent_a, "bounds_min", None)
            bounds_max_a = getattr(ent_a, "bounds_max", None)
            normal_a = getattr(ent_a, "surface_normal", None)

            # Check subpart relationship from direct attributes
            parent_id_a = getattr(ent_a, "parent_id", getattr(ent_a, "parent_box_id", None))
            child_ids_a = tuple(getattr(ent_a, "child_ids", getattr(ent_a, "child_box_ids", ())))

            for j in range(n):
                if i == j:
                    continue
                ent_b = entities[j]
                id_b = getattr(ent_b, "entity_id", getattr(ent_b, "canonical_entity_id", f"ent_{j}"))
                cls_b = getattr(ent_b, "class_name", getattr(ent_b, "primary_class", "unknown")).lower()
                tax_b = resolve_taxonomy_class(cls_b)
                pos_b = getattr(ent_b, "position_m", getattr(ent_b, "position", None))
                bounds_min_b = getattr(ent_b, "bounds_min", None)
                bounds_max_b = getattr(ent_b, "bounds_max", None)
                normal_b = getattr(ent_b, "surface_normal", None)

                # Determine scale-state validity for metric-threshold-dependent rules.
                # This is computed once per pair for efficiency.
                metric_vstatus = _metric_validation_status(ent_a, ent_b)

                # 1. Structural PART_OF / CONTAINS hierarchy (scale-invariant: uses
                #    explicit parent/child ID linkage, not distance).
                if parent_id_a == id_b or id_a in getattr(ent_b, "child_ids", ()):
                    results.append(EvidenceRelationship(
                        source_id=id_a,
                        target_id=id_b,
                        relation_type=RelationshipType.PART_OF,
                        confidence=0.95,
                        uncertainty=0.05,
                        geometric_evidence={"hierarchy": "explicit_parent_child"},
                        validation_status="validated",
                        basis="structural_hierarchy",
                    ))
                    results.append(EvidenceRelationship(
                        source_id=id_b,
                        target_id=id_a,
                        relation_type=RelationshipType.CONTAINS,
                        confidence=0.95,
                        uncertainty=0.05,
                        geometric_evidence={"hierarchy": "explicit_parent_child"},
                        validation_status="validated",
                        basis="structural_hierarchy",
                    ))
                    continue

                # 2. Mounting and Attachment: verify against host taxonomy constraints.
                # Do NOT infer attachment merely from proximity: require host compatibility.
                is_valid_host = (
                    cls_b in tax_a.hosts
                    or any(h in cls_b for h in tax_a.hosts)
                    or ("wall" in cls_b and tax_a.category in ("electrical", "lighting", "opening", "detail"))
                )

                if pos_a is not None and pos_b is not None:
                    pa = np.array(pos_a[:3], dtype=np.float32)
                    pb = np.array(pos_b[:3], dtype=np.float32)
                    dist = float(np.linalg.norm(pa - pb))

                    # 2a. Planar contact distance if wall host.
                    # Uses metric thresholds → mark with scale-state-derived validation_status.
                    if "wall" in cls_b and is_valid_host:
                        if normal_b is not None:
                            nb = np.array(normal_b[:3], dtype=np.float32)
                            norm_nb = float(np.linalg.norm(nb))
                            if norm_nb > 1e-4:
                                nb = nb / norm_nb
                            planar_dist = float(abs(np.dot(pa - pb, nb)))
                        else:
                            planar_dist = float(np.hypot(pa[0] - pb[0], pa[1] - pb[1]))

                        if planar_dist <= wall_contact_threshold_m:
                            # Verify if embedded (e.g. socket, electrical panel) or surface-mounted
                            is_embedded = any(k in cls_a for k in ("socket", "outlet", "switch", "panel", "junction_box"))
                            if is_embedded:
                                results.append(EvidenceRelationship(
                                    source_id=id_a,
                                    target_id=id_b,
                                    relation_type=RelationshipType.EMBEDDED_IN,
                                    confidence=0.88,
                                    uncertainty=0.12,
                                    geometric_evidence={
                                        "wall_planar_clearance_m": planar_dist,
                                        "host_class": cls_b,
                                        "threshold_m": wall_contact_threshold_m,
                                    },
                                    validation_status=metric_vstatus,
                                    basis="coplanar_wall_host_mounting",
                                ))
                            results.append(EvidenceRelationship(
                                source_id=id_a,
                                target_id=id_b,
                                relation_type=RelationshipType.MOUNTED_ON,
                                confidence=0.88,
                                uncertainty=0.12,
                                geometric_evidence={
                                    "wall_planar_clearance_m": planar_dist,
                                    "host_class": cls_b,
                                    "threshold_m": wall_contact_threshold_m,
                                },
                                validation_status=metric_vstatus,
                                basis="coplanar_wall_host_mounting",
                            ))
                            results.append(EvidenceRelationship(
                                source_id=id_a,
                                target_id=id_b,
                                relation_type=RelationshipType.ATTACHED_TO,
                                confidence=0.85,
                                uncertainty=0.15,
                                geometric_evidence={
                                    "attachment_clearance_m": planar_dist,
                                    "threshold_m": wall_contact_threshold_m,
                                },
                                validation_status=metric_vstatus,
                                basis="surface_attachment",
                            ))

                    # 3. Floor / Furniture Support (Vertical Stacking).
                    # Requires footprint overlap and vertical contact — metric threshold.
                    if ("floor" in cls_b or "table" in cls_b or "desk" in cls_b or "shelf" in cls_b) and bounds_min_a and bounds_max_b:
                        vert_gap = abs(bounds_min_a[2] - bounds_max_b[2])
                        # Check horizontal overlap
                        ox = max(0.0, min(bounds_max_a[0], bounds_max_b[0]) - max(bounds_min_a[0], bounds_min_b[0]))
                        oy = max(0.0, min(bounds_max_a[1], bounds_max_b[1]) - max(bounds_min_a[1], bounds_min_b[1]))
                        has_overlap = (ox > 0.05 and oy > 0.05) if "floor" not in cls_b else True

                        if vert_gap <= support_contact_threshold_m and has_overlap:
                            results.append(EvidenceRelationship(
                                source_id=id_a,
                                target_id=id_b,
                                relation_type=RelationshipType.SUPPORTED_BY,
                                confidence=0.85,
                                uncertainty=0.15,
                                geometric_evidence={
                                    "vertical_contact_gap_m": vert_gap,
                                    "horizontal_overlap_area": ox * oy,
                                    "threshold_m": support_contact_threshold_m,
                                },
                                validation_status=metric_vstatus,
                                basis="vertical_support_contact",
                            ))

                    # 4. Vertical Above / Below — metric threshold (0.30 m minimum offset).
                    if pa[2] > pb[2] + 0.30:
                        results.append(EvidenceRelationship(
                            source_id=id_a,
                            target_id=id_b,
                            relation_type=RelationshipType.ABOVE,
                            confidence=0.80,
                            uncertainty=0.20,
                            geometric_evidence={
                                "vertical_delta_m": float(pa[2] - pb[2]),
                                "threshold_m": 0.30,
                            },
                            validation_status=metric_vstatus,
                            basis="coordinate_elevation",
                        ))
                    elif pb[2] > pa[2] + 0.30:
                        results.append(EvidenceRelationship(
                            source_id=id_a,
                            target_id=id_b,
                            relation_type=RelationshipType.BELOW,
                            confidence=0.80,
                            uncertainty=0.20,
                            geometric_evidence={
                                "vertical_delta_m": float(pb[2] - pa[2]),
                                "threshold_m": 0.30,
                            },
                            validation_status=metric_vstatus,
                            basis="coordinate_elevation",
                        ))

                    # 5. Adjacency: horizontal proximity without vertical support.
                    # 1.20 m upper bound is a metric threshold → mark accordingly.
                    if 0.10 < dist <= 1.20 and not ("wall" in cls_b or "floor" in cls_b):
                        results.append(EvidenceRelationship(
                            source_id=id_a,
                            target_id=id_b,
                            relation_type=RelationshipType.ADJACENT_TO,
                            confidence=0.75,
                            uncertainty=0.25,
                            geometric_evidence={
                                "distance_m": dist,
                                "threshold_m": 1.20,
                            },
                            validation_status=metric_vstatus,
                            basis="spatial_proximity_clearance",
                        ))

                    # 6. Surface alignment (aligned surface normals) — scale-invariant:
                    #    uses only direction cosines, not distance magnitudes.
                    if normal_a is not None and normal_b is not None:
                        na = np.array(normal_a[:3], dtype=np.float32)
                        nb = np.array(normal_b[:3], dtype=np.float32)
                        dot = float(np.dot(na, nb))
                        if dot > 0.90:
                            results.append(EvidenceRelationship(
                                source_id=id_a,
                                target_id=id_b,
                                relation_type=RelationshipType.ALIGNED_WITH,
                                confidence=0.82,
                                uncertainty=0.18,
                                geometric_evidence={"normal_alignment_cosine": dot},
                                validation_status="validated",
                                basis="coplanar_parallel_normals",
                            ))

                    # 7. Opens Into: doorway / portal transition into adjacent room or corridor (Sprint 7)
                    if "door" in cls_a and any(k in cls_b for k in ("corridor", "room", "passage", "space", "interior")):
                        results.append(EvidenceRelationship(
                            source_id=id_a,
                            target_id=id_b,
                            relation_type=RelationshipType.OPENS_INTO,
                            confidence=0.86,
                            uncertainty=0.14,
                            geometric_evidence={"source_door": id_a, "target_space": id_b},
                            validation_status="validated",
                            basis="doorway_portal_transition",
                        ))

                    # 8. Connected To: utility / plumbing pipe to fixture connection (Sprint 7)
                    is_utility_a = any(k in cls_a for k in ("pipe", "drain", "wire", "cable", "conduit", "hose", "fitting"))
                    is_fixture_b = any(k in cls_b for k in ("sink", "toilet", "faucet", "radiator", "basin", "appliance", "water_heater"))
                    if is_utility_a and is_fixture_b and dist <= 0.80:
                        results.append(EvidenceRelationship(
                            source_id=id_a,
                            target_id=id_b,
                            relation_type=RelationshipType.CONNECTED_TO,
                            confidence=0.84,
                            uncertainty=0.16,
                            geometric_evidence={"clearance_m": dist, "utility_type": cls_a, "fixture_type": cls_b},
                            validation_status=metric_vstatus,
                            basis="utility_fixture_continuity",
                        ))

                    # 9. Intersects: orthogonal structural boundaries forming room corners / floor junctions (Sprint 7)
                    if ("wall" in cls_a and "wall" in cls_b) or ("wall" in cls_a and "floor" in cls_b):
                        if normal_a is not None and normal_b is not None:
                            na_v = np.array(normal_a[:3], dtype=np.float32)
                            nb_v = np.array(normal_b[:3], dtype=np.float32)
                            len_a = float(np.linalg.norm(na_v))
                            len_b = float(np.linalg.norm(nb_v))
                            if len_a > 1e-4 and len_b > 1e-4:
                                cos_val = abs(float(np.dot(na_v / len_a, nb_v / len_b)))
                                if cos_val < 0.30 and dist <= 3.0:
                                    results.append(EvidenceRelationship(
                                        source_id=id_a,
                                        target_id=id_b,
                                        relation_type=RelationshipType.INTERSECTS,
                                        confidence=0.83,
                                        uncertainty=0.17,
                                        geometric_evidence={
                                            "normal_orthogonality_cosine": cos_val,
                                            "junction_distance_m": dist,
                                            "threshold_m": 3.0,
                                        },
                                        validation_status=metric_vstatus,
                                        basis="orthogonal_structural_junction",
                                    ))

        return results

    def detect_contradictions(self, relationships: Sequence[EvidenceRelationship]) -> list[dict[str, Any]]:
        """Audits relationship set for logical and physical contradictions."""
        contradictions: list[dict[str, Any]] = []

        # Group by pair
        pair_map: dict[tuple[str, str], list[EvidenceRelationship]] = defaultdict(list)
        for r in relationships:
            pair_map[(r.source_id, r.target_id)].append(r)

        # 1. Direct pairwise incompatibilities on (src, tgt)
        for (src, tgt), rels in pair_map.items():
            for i in range(len(rels)):
                for j in range(i + 1, len(rels)):
                    pair_kinds = frozenset({rels[i].relation_type, rels[j].relation_type})
                    if pair_kinds in _INCOMPATIBLE_PAIRS:
                        contradictions.append({
                            "type": "incompatible_relation_pair",
                            "source_id": src,
                            "target_id": tgt,
                            "relations": (rels[i].relation_type.value, rels[j].relation_type.value),
                            "reason": f"Mutually exclusive relations '{rels[i].relation_type.value}' and '{rels[j].relation_type.value}' asserted between {src} and {tgt}.",
                        })

        # 2. Reciprocal contradictions (A -> B and B -> A)
        seen_pairs: set[frozenset[str]] = set()
        for (src, tgt), rels_fwd in pair_map.items():
            pair_key = frozenset({src, tgt})
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            rels_rev = pair_map.get((tgt, src), [])
            for rf in rels_fwd:
                for rr in rels_rev:
                    # Inverted part_of
                    if rf.relation_type == RelationshipType.PART_OF and rr.relation_type == RelationshipType.PART_OF:
                        contradictions.append({
                            "type": "cyclical_part_of",
                            "source_id": src,
                            "target_id": tgt,
                            "reason": f"Cyclical part_of: {src} part_of {tgt} and {tgt} part_of {src}.",
                        })
                    # Inverted above
                    if rf.relation_type == RelationshipType.ABOVE and rr.relation_type == RelationshipType.ABOVE:
                        contradictions.append({
                            "type": "reciprocal_vertical_contradiction",
                            "source_id": src,
                            "target_id": tgt,
                            "reason": f"Both {src} and {tgt} asserted to be above each other.",
                        })

        return contradictions
