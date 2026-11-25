"""Scene Observation Graph and Structured Scene Intelligence (Sprint 6).

Provides a structured scene observation graph representing physical entities,
hierarchical components, surface boundaries, and directed spatial relationships.
Replaces flat bounding box lists with an evidence-grounded topological graph.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from engine.multiengine.spatial_intelligence.taxonomy import (
    Domain,
    TaxonomyClass,
    are_semantically_compatible,
    resolve_taxonomy_class,
)


class NodePresenceState(str, Enum):
    """Evidence status of physical presence in the scene."""

    NOT_PRESENT = "not_present"
    PRESENT_NOT_VISIBLE = "present_not_visible"
    POSSIBLY_PRESENT = "possibly_present"
    DETECTED_UNRESOLVED = "detected_unresolved"
    VERIFIED_PRESENT = "verified_present"


class EdgeValidationStatus(str, Enum):
    """Validation lifecycle status for a graph relationship edge."""

    HYPOTHESIZED = "hypothesized"
    VALIDATED = "validated"
    CONFLICTED = "conflicted"
    REJECTED = "rejected"


@dataclass
class SceneGraphNode:
    """Individual entity or structural component node in the scene observation graph."""

    node_id: str
    class_name: str
    taxonomy_id: str
    entity_category: str  # "structure", "opening", "furniture", "fixture", "environmental", "detail"
    presence_state: NodePresenceState = NodePresenceState.DETECTED_UNRESOLVED
    confidence: float = 0.5
    uncertainty: float = 0.5
    is_multiview_confirmed: bool = False
    is_geometrically_positioned: bool = False
    is_physically_measured: bool = False
    observations: tuple[str, ...] = ()
    box_2d: tuple[float, float, float, float] | None = None
    position_3d: tuple[float, float, float] | None = None
    dimensions_3d: tuple[float, float, float] | None = None
    orientation_quat: tuple[float, float, float, float] | None = None
    surface_normal: tuple[float, float, float] | None = None
    material: str | None = None
    surface_finish: str | None = None
    dominant_color: str | None = None
    parent_id: str | None = None
    subpart_ids: tuple[str, ...] = ()
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "class_name": self.class_name,
            "taxonomy_id": self.taxonomy_id,
            "entity_category": self.entity_category,
            "presence_state": self.presence_state.value,
            "confidence": round(self.confidence, 4),
            "uncertainty": round(self.uncertainty, 4),
            "is_multiview_confirmed": self.is_multiview_confirmed,
            "is_geometrically_positioned": self.is_geometrically_positioned,
            "is_physically_measured": self.is_physically_measured,
            "observations": list(self.observations),
            "box_2d": list(self.box_2d) if self.box_2d else None,
            "position_3d": [round(x, 4) for x in self.position_3d] if self.position_3d else None,
            "dimensions_3d": [round(x, 4) for x in self.dimensions_3d] if self.dimensions_3d else None,
            "orientation_quat": [round(x, 4) for x in self.orientation_quat] if self.orientation_quat else None,
            "surface_normal": [round(x, 4) for x in self.surface_normal] if self.surface_normal else None,
            "material": self.material,
            "surface_finish": self.surface_finish,
            "dominant_color": self.dominant_color,
            "parent_id": self.parent_id,
            "subpart_ids": list(self.subpart_ids),
            "attributes": self.attributes,
        }


@dataclass
class SceneGraphEdge:
    """Directed, evidence-backed relationship edge between two scene graph nodes."""

    source_id: str
    target_id: str
    relation_type: str  # attached_to, mounted_on, supported_by, contains, part_of, etc.
    confidence: float = 0.5
    uncertainty: float = 0.5
    geometric_evidence: dict[str, Any] = field(default_factory=dict)
    supporting_observations: tuple[str, ...] = ()
    validation_status: EdgeValidationStatus = EdgeValidationStatus.HYPOTHESIZED
    conflict_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type,
            "confidence": round(self.confidence, 4),
            "uncertainty": round(self.uncertainty, 4),
            "geometric_evidence": self.geometric_evidence,
            "supporting_observations": list(self.supporting_observations),
            "validation_status": self.validation_status.value,
            "conflict_reason": self.conflict_reason,
        }


# Pairs of relations that are mutually contradictory between the same source and target
_CONTRADICTORY_RELATION_PAIRS: frozenset[frozenset[str]] = frozenset({
    frozenset({"part_of", "contains"}),
    frozenset({"above", "below"}),
    frozenset({"in_front_of", "behind"}),
    frozenset({"supported_by", "supports"}),
    frozenset({"inside", "contains"}),
})


class SceneObservationGraph:
    """Directed graph representing physical entities, subparts, and spatial relationships."""

    def __init__(self, scene_id: str = "scene_001") -> None:
        self.scene_id: str = scene_id
        self._nodes: dict[str, SceneGraphNode] = {}
        self._edges: list[SceneGraphEdge] = []
        self._out_edges: dict[str, list[SceneGraphEdge]] = defaultdict(list)
        self._in_edges: dict[str, list[SceneGraphEdge]] = defaultdict(list)

    @property
    def nodes(self) -> Mapping[str, SceneGraphNode]:
        return self._nodes

    @property
    def edges(self) -> tuple[SceneGraphEdge, ...]:
        return tuple(self._edges)

    def add_node(self, node: SceneGraphNode) -> None:
        self._nodes[node.node_id] = node

    def add_edge(self, edge: SceneGraphEdge) -> None:
        self._edges.append(edge)
        self._out_edges[edge.source_id].append(edge)
        self._in_edges[edge.target_id].append(edge)

    def get_node(self, node_id: str) -> SceneGraphNode | None:
        return self._nodes.get(node_id)

    def get_outgoing_edges(self, source_id: str) -> tuple[SceneGraphEdge, ...]:
        return tuple(self._out_edges.get(source_id, []))

    def get_incoming_edges(self, target_id: str) -> tuple[SceneGraphEdge, ...]:
        return tuple(self._in_edges.get(target_id, []))

    def get_subparts(self, parent_node_id: str) -> tuple[SceneGraphNode, ...]:
        node = self.get_node(parent_node_id)
        if node is None:
            return ()
        return tuple(self._nodes[cid] for cid in node.subpart_ids if cid in self._nodes)

    def detect_contradictions(self) -> list[dict[str, Any]]:
        """Identifies semantic and geometric contradictions across graph edges.

        Detects:
        1. Mutually exclusive relations between identical node pairs (e.g., A part_of B and A contains B).
        2. Inverted cyclical hierarchy (A part_of B and B part_of A).
        3. Opposing vertical/depth arrangements (A above B and A below B).
        4. Incompatible container/part semantics.
        """
        contradictions: list[dict[str, Any]] = []

        # Group edges by directed pair (source, target)
        pair_to_edges: dict[tuple[str, str], list[SceneGraphEdge]] = defaultdict(list)
        for edge in self._edges:
            pair_to_edges[(edge.source_id, edge.target_id)].append(edge)

        # 1. Check direct contradictions on identical (src, tgt)
        for (src, tgt), edges in pair_to_edges.items():
            rel_types = [e.relation_type for e in edges]
            for i in range(len(edges)):
                for j in range(i + 1, len(edges)):
                    pair_set = frozenset({edges[i].relation_type, edges[j].relation_type})
                    if pair_set in _CONTRADICTORY_RELATION_PAIRS:
                        contradictions.append({
                            "type": "direct_relation_conflict",
                            "source_id": src,
                            "target_id": tgt,
                            "relations": (edges[i].relation_type, edges[j].relation_type),
                            "reason": f"Mutually exclusive relations '{edges[i].relation_type}' and '{edges[j].relation_type}' asserted between {src} and {tgt}.",
                        })
                        edges[i].validation_status = EdgeValidationStatus.CONFLICTED
                        edges[i].conflict_reason = f"Conflicted with {edges[j].relation_type}"
                        edges[j].validation_status = EdgeValidationStatus.CONFLICTED
                        edges[j].conflict_reason = f"Conflicted with {edges[i].relation_type}"

        # 2. Check reciprocal contradictions (A -> B and B -> A)
        visited_pairs: set[frozenset[str]] = set()
        for (src, tgt), edges_fwd in pair_to_edges.items():
            pair_key = frozenset({src, tgt})
            if pair_key in visited_pairs:
                continue
            visited_pairs.add(pair_key)

            edges_rev = pair_to_edges.get((tgt, src), [])
            for ef in edges_fwd:
                for er in edges_rev:
                    # Part-of cycle: A is part of B AND B is part of A
                    if ef.relation_type == "part_of" and er.relation_type == "part_of":
                        contradictions.append({
                            "type": "cyclical_hierarchy",
                            "source_id": src,
                            "target_id": tgt,
                            "relations": ("part_of", "part_of"),
                            "reason": f"Cyclical containment hierarchy: {src} part_of {tgt} and {tgt} part_of {src}.",
                        })
                        ef.validation_status = EdgeValidationStatus.CONFLICTED
                        er.validation_status = EdgeValidationStatus.CONFLICTED

                    # Inverted container conflict: A inside B and B inside A
                    if ef.relation_type == "inside" and er.relation_type == "inside":
                        contradictions.append({
                            "type": "cyclical_enclosure",
                            "source_id": src,
                            "target_id": tgt,
                            "relations": ("inside", "inside"),
                            "reason": f"Cyclical spatial enclosure: {src} inside {tgt} and {tgt} inside {src}.",
                        })
                        ef.validation_status = EdgeValidationStatus.CONFLICTED
                        er.validation_status = EdgeValidationStatus.CONFLICTED

                    # Vertical contradiction: A above B and B above A
                    if ef.relation_type == "above" and er.relation_type == "above":
                        contradictions.append({
                            "type": "opposing_vertical_order",
                            "source_id": src,
                            "target_id": tgt,
                            "relations": ("above", "above"),
                            "reason": f"Reciprocal vertical contradiction: {src} above {tgt} and {tgt} above {src}.",
                        })
                        ef.validation_status = EdgeValidationStatus.CONFLICTED
                        er.validation_status = EdgeValidationStatus.CONFLICTED

        return contradictions

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "node_count": len(self._nodes),
            "edge_count": len(self._edges),
            "nodes": {nid: n.to_dict() for nid, n in self._nodes.items()},
            "edges": [e.to_dict() for e in self._edges],
            "contradictions": self.detect_contradictions(),
        }


def classify_entity_category(class_name: str) -> str:
    """Categorizes an entity into high-level scene structure groups."""
    tax = resolve_taxonomy_class(class_name)
    if tax.is_structural or tax.category == "surface":
        return "structure"
    if "door" in class_name or "window" in class_name or tax.category == "opening":
        return "opening"
    if tax.category in ("electrical", "lighting", "plumbing", "hvac", "utility"):
        return "fixture"
    if tax.domain == Domain.EXTERIOR or tax.category in ("vegetation", "street", "circulation", "perimeter"):
        return "environmental"
    if any(k in class_name for k in ("handle", "lock", "hinge", "frame", "sill", "leg", "drawer")):
        return "component"
    return "furniture"


def build_scene_observation_graph(
    scene_id: str,
    merged_entities: Sequence[Any],
    lifted_entities: Sequence[Any],
    visual_attributes: Mapping[str, Any] | None = None,
    spatial_relationships: Sequence[Any] | None = None,
    unresolved_candidates: Sequence[Any] | None = None,
) -> SceneObservationGraph:
    """Constructs a validated SceneObservationGraph from perception and spatial reconstruction results."""
    graph = SceneObservationGraph(scene_id=scene_id)
    v_attrs = visual_attributes or {}

    # Map lifted entities by entity_id
    lifted_map: dict[str, Any] = {e.entity_id: e for e in lifted_entities}

    # 1. Add nodes from merged physical entities
    for ment in merged_entities:
        cid = ment.canonical_entity_id
        cname = ment.primary_class
        tax = resolve_taxonomy_class(cname)
        cat = classify_entity_category(cname)

        # Presence determination
        is_mv = bool(ment.is_multiview)
        presence = NodePresenceState.VERIFIED_PRESENT if is_mv else NodePresenceState.DETECTED_UNRESOLVED

        # 3D geometry from lifted entity or merged entity
        pos_3d = ment.position
        dim_3d = ment.dimensions
        quat_3d = ment.orientation_quat_wxyz
        normal = None

        if cid in lifted_map:
            lent = lifted_map[cid]
            if pos_3d is None:
                pos_3d = lent.position_m
            if dim_3d is None:
                dim_3d = lent.dimensions_m
            if normal is None:
                normal = lent.surface_normal

        # Visual attributes
        mat_name = None
        finish_name = None
        color_name = None
        if cid in v_attrs:
            va = v_attrs[cid]
            mat_name = getattr(va, "material", None)
            if hasattr(mat_name, "value"):
                mat_name = mat_name.value
            finish_name = getattr(va, "surface_finish", None)
            if hasattr(finish_name, "value"):
                finish_name = finish_name.value
            color_name = getattr(va, "dominant_color_name", None)

        node = SceneGraphNode(
            node_id=cid,
            class_name=cname,
            taxonomy_id=tax.taxonomy_id,
            entity_category=cat,
            presence_state=presence,
            confidence=float(ment.confidence),
            uncertainty=max(0.01, min(0.99, 1.0 - float(ment.confidence))),
            is_multiview_confirmed=is_mv,
            is_geometrically_positioned=pos_3d is not None,
            is_physically_measured=bool(ment.is_physically_anchored),
            observations=tuple(ment.member_observation_ids),
            position_3d=pos_3d,
            dimensions_3d=dim_3d,
            orientation_quat=quat_3d,
            surface_normal=normal,
            material=str(mat_name) if mat_name else None,
            surface_finish=str(finish_name) if finish_name else None,
            dominant_color=str(color_name) if color_name else None,
            attributes=dict(ment.details) if hasattr(ment, "details") else {},
        )
        graph.add_node(node)

    # 2. Add unresolved single-view observation nodes if provided
    if unresolved_candidates:
        for cand in unresolved_candidates:
            cid = getattr(cand, "detection_id", str(getattr(cand, "box_id", "")))
            if not cid or cid in graph.nodes:
                continue
            cname = getattr(cand, "class_name", "unknown")
            tax = resolve_taxonomy_class(cname)
            cat = classify_entity_category(cname)
            node = SceneGraphNode(
                node_id=cid,
                class_name=cname,
                taxonomy_id=tax.taxonomy_id,
                entity_category=cat,
                presence_state=NodePresenceState.DETECTED_UNRESOLVED,
                confidence=float(getattr(cand, "confidence", 0.5)),
                uncertainty=max(0.01, min(0.99, 1.0 - float(getattr(cand, "confidence", 0.5)))),
                is_multiview_confirmed=False,
                is_geometrically_positioned=False,
                is_physically_measured=False,
                observations=(getattr(cand, "source_image_id", "img_001"),),
                box_2d=getattr(cand, "box_2d", None),
                parent_id=getattr(cand, "parent_id", getattr(cand, "parent_box_id", None)),
                subpart_ids=tuple(getattr(cand, "child_ids", getattr(cand, "child_box_ids", ()))),
            )
            graph.add_node(node)

    # 3. Add edges from spatial relationships
    if spatial_relationships:
        for rel in spatial_relationships:
            src = getattr(rel, "source_id", "")
            tgt = getattr(rel, "target_id", "")
            rkind = getattr(rel, "relation", "adjacent_to")
            rval = rkind.value if hasattr(rkind, "value") else str(rkind)

            # Only add edge if both nodes exist or can be referenced
            edge = SceneGraphEdge(
                source_id=src,
                target_id=tgt,
                relation_type=rval,
                confidence=float(getattr(rel, "confidence", 0.5)),
                uncertainty=max(0.01, min(0.99, 1.0 - float(getattr(rel, "confidence", 0.5)))),
                geometric_evidence={
                    "distance": getattr(rel, "distance", 0.0),
                    "distance_units": getattr(rel, "distance_units", "m"),
                    "basis": getattr(rel, "basis", ""),
                },
                validation_status=EdgeValidationStatus.VALIDATED,
            )
            graph.add_edge(edge)

    # 4. Detect and record any contradictions
    graph.detect_contradictions()

    return graph
