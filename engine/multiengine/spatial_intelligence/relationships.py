"""Object Relationship Intelligence and Spatial Semantics (Sprint 6).

Implements evidence-backed spatial relationships, physical attachment verification,
and semantic contradiction detection across 16 canonical relationship types.
Does not infer physical attachment merely from proximity.
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
        """Evaluates pairwise physical and structural relationships across entities."""
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

                # 1. Structural PART_OF / CONTAINS hierarchy
                if parent_id_a == id_b or id_a in getattr(ent_b, "child_ids", ()):
                    results.append(EvidenceRelationship(
                        source_id=id_a,
                        target_id=id_b,
                        relation_type=RelationshipType.PART_OF,
                        confidence=0.95,
                        uncertainty=0.05,
                        geometric_evidence={"hierarchy": "explicit_parent_child"},
                        basis="structural_hierarchy",
                    ))
                    results.append(EvidenceRelationship(
                        source_id=id_b,
                        target_id=id_a,
                        relation_type=RelationshipType.CONTAINS,
                        confidence=0.95,
                        uncertainty=0.05,
                        geometric_evidence={"hierarchy": "explicit_parent_child"},
                        basis="structural_hierarchy",
                    ))
                    continue

                # 2. Mounting and Attachment: verify against host taxonomy constraints
                # Do NOT infer attachment merely from proximity: require host compatibility
                is_valid_host = (
                    cls_b in tax_a.hosts
                    or any(h in cls_b for h in tax_a.hosts)
                    or ("wall" in cls_b and tax_a.category in ("electrical", "lighting", "opening", "detail"))
                )

                if pos_a is not None and pos_b is not None:
                    pa = np.array(pos_a[:3], dtype=np.float32)
                    pb = np.array(pos_b[:3], dtype=np.float32)
                    dist = float(np.linalg.norm(pa - pb))

                    # Planar contact distance if wall
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
                                    geometric_evidence={"wall_planar_clearance_m": planar_dist, "host_class": cls_b},
                                    basis="coplanar_wall_host_mounting",
                                ))
                            results.append(EvidenceRelationship(
                                source_id=id_a,
                                target_id=id_b,
                                relation_type=RelationshipType.MOUNTED_ON,
                                confidence=0.88,
                                uncertainty=0.12,
                                geometric_evidence={"wall_planar_clearance_m": planar_dist, "host_class": cls_b},
                                basis="coplanar_wall_host_mounting",
                            ))
                            results.append(EvidenceRelationship(
                                source_id=id_a,
                                target_id=id_b,
                                relation_type=RelationshipType.ATTACHED_TO,
                                confidence=0.85,
                                uncertainty=0.15,
                                geometric_evidence={"attachment_clearance_m": planar_dist},
                                basis="surface_attachment",
                            ))

                    # 3. Floor / Furniture Support (Vertical Stacking)
                    # Requires footprint overlap and vertical contact
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
                                geometric_evidence={"vertical_contact_gap_m": vert_gap, "horizontal_overlap_area": ox * oy},
                                basis="vertical_support_contact",
                            ))

                    # 4. Vertical Above / Below
                    if pa[2] > pb[2] + 0.30:
                        results.append(EvidenceRelationship(
                            source_id=id_a,
                            target_id=id_b,
                            relation_type=RelationshipType.ABOVE,
                            confidence=0.80,
                            uncertainty=0.20,
                            geometric_evidence={"vertical_delta_m": float(pa[2] - pb[2])},
                            basis="coordinate_elevation",
                        ))
                    elif pb[2] > pa[2] + 0.30:
                        results.append(EvidenceRelationship(
                            source_id=id_a,
                            target_id=id_b,
                            relation_type=RelationshipType.BELOW,
                            confidence=0.80,
                            uncertainty=0.20,
                            geometric_evidence={"vertical_delta_m": float(pb[2] - pa[2])},
                            basis="coordinate_elevation",
                        ))

                    # 5. Adjacency: horizontal proximity without vertical support
                    if 0.10 < dist <= 1.20 and not ("wall" in cls_b or "floor" in cls_b):
                        results.append(EvidenceRelationship(
                            source_id=id_a,
                            target_id=id_b,
                            relation_type=RelationshipType.ADJACENT_TO,
                            confidence=0.75,
                            uncertainty=0.25,
                            geometric_evidence={"distance_m": dist},
                            basis="spatial_proximity_clearance",
                        ))

                    # 6. Surface alignment (aligned surface normals)
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
                                basis="coplanar_parallel_normals",
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
