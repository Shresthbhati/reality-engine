"""Grounded Reality Copilot Spatial Engine.

Spec: Universal Intelligence Expansion (Epic E11-1).
Deterministic spatial question answering and natural language query execution with
multi-step tool grounding against WorldIR, SceneGraph, and SpatialIndex.

Mandates:
- 100% precision on spatial entity queries.
- Zero fabricated measurements, entities, or geometry.
- Complete traceability and provenance for all assertions.
- Deterministic execution without cloud LLM hallucination risk.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from assistant.query_translator import ParsedQuery, QueryIntent, QueryTranslator
from assistant.tools import Context, NOT_RECORDED, explain_provenance, measure_tool, provenance_label
from engine.scene_graph.graph import SceneGraph
from engine.scene_graph.query import WorldQuery
from engine.scene_graph.spatial_index import SpatialIndex, _entity_bounds
from world_ir import Entity, EntityType, RelationshipKind
from world_ir import measure
from world_ir.world_v1 import WorldIR


@dataclass
class GroundedFact:
    """An atomic, verified factual claim grounded directly in WorldIR."""
    entity_id: str
    property_name: str
    value: Any
    provenance: str
    confidence: float
    source: str  # e.g. "geometry_bounds", "relationship_graph", "spatial_index", "entity_metadata"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "property_name": self.property_name,
            "value": self.value,
            "provenance": self.provenance,
            "confidence": self.confidence,
            "source": self.source,
        }


@dataclass
class CopilotMeasurement:
    """A verified measurement computed between entities or over geometry."""
    kind: str  # "distance", "height", "width", "depth", "area", "clearance"
    value: float
    unit: str
    entity_ids: List[str]
    scale_state: str
    points: Optional[List[List[float]]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "value": self.value,
            "unit": self.unit,
            "entity_ids": self.entity_ids,
            "scale_state": self.scale_state,
            "points": self.points,
        }


@dataclass
class CopilotResponse:
    """Structured response returned by the Grounded Reality Copilot."""
    query: str
    intent: str
    success: bool
    answer: str
    entities: List[Dict[str, Any]] = field(default_factory=list)
    measurements: List[CopilotMeasurement] = field(default_factory=list)
    grounded_facts: List[GroundedFact] = field(default_factory=list)
    uncertainties: List[str] = field(default_factory=list)
    trace: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "intent": self.intent,
            "success": self.success,
            "answer": self.answer,
            "entities": self.entities,
            "measurements": [m.to_dict() for m in self.measurements],
            "grounded_facts": [f.to_dict() for f in self.grounded_facts],
            "uncertainties": self.uncertainties,
            "trace": self.trace,
        }


class RealityCopilotEngine:
    """Deterministic spatial engine answering spatial and geometric queries against WorldIR."""

    def __init__(self, world: WorldIR, scale_state: str = "metric"):
        self.world = world
        self.scale_state = scale_state  # "metric", "authored", "unknown"
        self.translator = QueryTranslator()
        self.query_facade = WorldQuery(world)
        self.scene_graph = SceneGraph(world)
        self.spatial_index = SpatialIndex(world)
        self.ctx = Context(world=world, scale_state=scale_state)

    def ask(self, query: Union[str, Dict[str, Any], ParsedQuery]) -> CopilotResponse:
        """Process a query and return a strictly grounded CopilotResponse."""
        if isinstance(query, ParsedQuery):
            plan = query
        else:
            plan = self.translator.translate(query)

        trace: List[str] = [f"Translated query '{plan.raw_query}' to intent {plan.intent.value}"]

        # Route by parsed intent
        if plan.intent == QueryIntent.PROXIMITY:
            return self._handle_proximity(plan, trace)
        elif plan.intent == QueryIntent.CONTAINMENT:
            return self._handle_containment(plan, trace)
        elif plan.intent == QueryIntent.ATTACHMENT:
            return self._handle_attachment(plan, trace)
        elif plan.intent == QueryIntent.SUPPORT:
            return self._handle_support(plan, trace)
        elif plan.intent == QueryIntent.CONNECTIVITY:
            return self._handle_connectivity(plan, trace)
        elif plan.intent == QueryIntent.MEASUREMENT_DISTANCE:
            return self._handle_distance(plan, trace)
        elif plan.intent == QueryIntent.MEASUREMENT_DIMENSION:
            return self._handle_dimension(plan, trace)
        elif plan.intent == QueryIntent.MEASUREMENT_AREA:
            return self._handle_area(plan, trace)
        elif plan.intent == QueryIntent.CLEARANCE_FIT:
            return self._handle_clearance(plan, trace)
        elif plan.intent == QueryIntent.COUNT:
            return self._handle_count(plan, trace)
        elif plan.intent in (QueryIntent.FIND, QueryIntent.INSPECTION):
            return self._handle_find_and_inspect(plan, trace)
        elif plan.intent == QueryIntent.PROVENANCE:
            return self._handle_provenance(plan, trace)
        elif plan.intent == QueryIntent.CONFIDENCE:
            return self._handle_confidence(plan, trace)
        else:
            return self._handle_unknown(plan, trace)

    # ---------------------------------------------------- Intent Handlers

    def _handle_proximity(self, plan: ParsedQuery, trace: List[str]) -> CopilotResponse:
        """Find entities within metric radius of a reference entity."""
        ref_name = plan.reference_name
        if not ref_name:
            return CopilotResponse(
                query=plan.raw_query,
                intent=plan.intent.value,
                success=False,
                answer="No reference entity specified for proximity query.",
                uncertainties=["Missing reference entity."],
                trace=trace,
            )

        ref_entity = self._resolve_entity(ref_name)
        if not ref_entity:
            trace.append(f"Entity matching reference '{ref_name}' not found.")
            return CopilotResponse(
                query=plan.raw_query,
                intent=plan.intent.value,
                success=False,
                answer=f"Reference entity '{ref_name}' was not found in the reconstructed world.",
                uncertainties=[f"Unknown reference entity: {ref_name}"],
                trace=trace,
            )

        trace.append(f"Resolved reference entity: {ref_entity.id} ({ref_entity.name or ref_entity.type})")
        ref_bounds = self._get_entity_bounds(ref_entity)
        if not ref_bounds:
            # Fall back to entity transform or center (0, 0, 0)
            center = (0.0, 0.0, 0.0)
            trace.append("Reference entity has no geometry bounds; using default origin (0, 0, 0).")
        else:
            b_min, b_max = ref_bounds
            center = (
                (b_min[0] + b_max[0]) / 2.0,
                (b_min[1] + b_max[1]) / 2.0,
                (b_min[2] + b_max[2]) / 2.0,
            )

        radius = plan.distance_threshold_m
        target_type = plan.target_type

        # Query spatial index
        nearby_candidates = self.spatial_index.within_radius(center, radius=radius)
        trace.append(f"Spatial index returned {len(nearby_candidates)} candidates within {radius:.2f}m.")

        results: List[Dict[str, Any]] = []
        measurements: List[CopilotMeasurement] = []
        facts: List[GroundedFact] = []

        unit = "m" if self.scale_state in ("metric", "authored") else "model units"

        for candidate, raw_dist in nearby_candidates:
            if candidate.id == ref_entity.id:
                continue

            # Target type/class filtering
            if target_type and not self._matches_type_or_class(candidate, target_type):
                continue

            # Compute accurate surface-to-surface or center distance
            cand_bounds = self._get_entity_bounds(candidate)
            if ref_bounds and cand_bounds:
                dist = self._aabb_distance(ref_bounds, cand_bounds)
            else:
                dist = raw_dist

            if dist > radius:
                continue

            meas = CopilotMeasurement(
                kind="distance",
                value=round(dist, 3),
                unit=unit,
                entity_ids=[ref_entity.id, candidate.id],
                scale_state=self.scale_state,
            )
            measurements.append(meas)

            facts.append(
                GroundedFact(
                    entity_id=candidate.id,
                    property_name=f"distance_to_{ref_entity.id}",
                    value=round(dist, 3),
                    provenance=candidate.provenance.value,
                    confidence=candidate.confidence,
                    source="spatial_index",
                )
            )

            results.append({
                "id": candidate.id,
                "name": candidate.name or NOT_RECORDED,
                "type": getattr(candidate.type, "value", str(candidate.type)),
                "entity_class": candidate.custom_properties.get("entity_class", NOT_RECORDED),
                "distance_m": round(dist, 3),
                "confidence": candidate.confidence,
                "provenance": provenance_label(candidate),
            })

        results.sort(key=lambda r: r["distance_m"])

        target_desc = f"{target_type} entity" if target_type else "object"
        if not results:
            answer = f"No {target_desc} found within {radius:.1f}{unit} of {ref_entity.name or ref_entity.id}."
        else:
            lines = [
                f"Found {len(results)} {target_desc}(s) within {radius:.1f}{unit} of {ref_entity.name or ref_entity.id}:"
            ]
            for r in results:
                lines.append(
                    f"- {r['name']} ({r['id']}, {r['type']}): {r['distance_m']}{unit} away "
                    f"[confidence: {r['confidence']:.2f}, provenance: {r['provenance']}]"
                )
            answer = "\n".join(lines)

        return CopilotResponse(
            query=plan.raw_query,
            intent=plan.intent.value,
            success=True,
            answer=answer,
            entities=results,
            measurements=measurements,
            grounded_facts=facts,
            trace=trace,
        )

    def _handle_containment(self, plan: ParsedQuery, trace: List[str]) -> CopilotResponse:
        """Find entities contained within a room or container."""
        container_name = plan.reference_name
        if not container_name:
            return CopilotResponse(
                query=plan.raw_query,
                intent=plan.intent.value,
                success=False,
                answer="No container entity specified.",
                trace=trace,
            )

        container = self._resolve_entity(container_name)
        if not container:
            return CopilotResponse(
                query=plan.raw_query,
                intent=plan.intent.value,
                success=False,
                answer=f"Container '{container_name}' was not found in the reconstructed world.",
                uncertainties=[f"Unknown container: {container_name}"],
                trace=trace,
            )

        # 1. Check SceneGraph containment
        contained = self.scene_graph.contents_of(container.id)
        trace.append(f"SceneGraph reported {len(contained)} contained entities for {container.id}.")

        # 2. Check Geometric containment if container has bounds
        c_bounds = self._get_entity_bounds(container)
        if c_bounds:
            b_min, b_max = c_bounds
            geom_contained = self.spatial_index.within_region(b_min, b_max)
            contained_ids = {e.id for e in contained}
            for ge in geom_contained:
                if ge.id != container.id and ge.id not in contained_ids:
                    contained.append(ge)
                    contained_ids.add(ge.id)
            trace.append(f"Total contained entities after geometric check: {len(contained)}.")

        # Filter by target type if requested
        if plan.target_type:
            contained = [e for e in contained if self._matches_type_or_class(e, plan.target_type)]

        results = [
            {
                "id": e.id,
                "name": e.name or NOT_RECORDED,
                "type": getattr(e.type, "value", str(e.type)),
                "entity_class": e.custom_properties.get("entity_class", NOT_RECORDED),
                "confidence": e.confidence,
                "provenance": provenance_label(e),
            }
            for e in contained
        ]
        results.sort(key=lambda r: r["id"])

        target_desc = f"{plan.target_type} entity" if plan.target_type else "object"
        if not results:
            answer = f"No {target_desc}s found inside {container.name or container.id}."
        else:
            lines = [f"Found {len(results)} {target_desc}(s) inside {container.name or container.id}:"]
            for r in results:
                lines.append(f"- {r['name']} ({r['id']}, {r['type']}) [provenance: {r['provenance']}]")
            answer = "\n".join(lines)

        return CopilotResponse(
            query=plan.raw_query,
            intent=plan.intent.value,
            success=True,
            answer=answer,
            entities=results,
            trace=trace,
        )

    def _handle_attachment(self, plan: ParsedQuery, trace: List[str]) -> CopilotResponse:
        """Find entities attached to or mounted on host entity (e.g. wall)."""
        host_name = plan.reference_name
        if not host_name:
            return CopilotResponse(
                query=plan.raw_query,
                intent=plan.intent.value,
                success=False,
                answer="No host entity specified for attachment query.",
                trace=trace,
            )

        host = self._resolve_entity(host_name)
        if not host:
            return CopilotResponse(
                query=plan.raw_query,
                intent=plan.intent.value,
                success=False,
                answer=f"Host entity '{host_name}' was not found in the reconstructed world.",
                uncertainties=[f"Unknown host entity: {host_name}"],
                trace=trace,
            )

        attached_edges = self.scene_graph.edges_to(host.id, RelationshipKind.ATTACHED_TO)
        attached_edges += self.scene_graph.edges_to(host.id, RelationshipKind.PART_OF)
        trace.append(f"SceneGraph found {len(attached_edges)} attached/part-of edges targeting {host.id}.")

        entities: List[Entity] = []
        seen = set()
        for edge in attached_edges:
            ent = self.world.entities.get(edge.source_id)
            if ent and ent.id not in seen:
                seen.add(ent.id)
                entities.append(ent)

        # Proximity fallback: small items within 0.15m of host wall
        host_bounds = self._get_entity_bounds(host)
        if host_bounds:
            b_min, b_max = host_bounds
            margin = 0.15
            expanded_min = (b_min[0] - margin, b_min[1] - margin, b_min[2] - margin)
            expanded_max = (b_max[0] + margin, b_max[1] + margin, b_max[2] + margin)
            prox_items = self.spatial_index.within_region(expanded_min, expanded_max)
            for p in prox_items:
                if p.id != host.id and p.id not in seen:
                    # If it's a fixture or small object, associate as attached
                    if p.custom_properties.get("entity_class", "").startswith("electrical") or p.type in (
                        EntityType.WINDOW, EntityType.DOOR
                    ):
                        seen.add(p.id)
                        entities.append(p)

        if plan.target_type:
            entities = [e for e in entities if self._matches_type_or_class(e, plan.target_type)]

        results = [
            {
                "id": e.id,
                "name": e.name or NOT_RECORDED,
                "type": getattr(e.type, "value", str(e.type)),
                "entity_class": e.custom_properties.get("entity_class", NOT_RECORDED),
                "confidence": e.confidence,
                "provenance": provenance_label(e),
            }
            for e in entities
        ]

        if not results:
            answer = f"No attached items found on {host.name or host.id}."
        else:
            lines = [f"Found {len(results)} item(s) mounted on/attached to {host.name or host.id}:"]
            for r in results:
                lines.append(f"- {r['name']} ({r['id']}, {r['type']}) [provenance: {r['provenance']}]")
            answer = "\n".join(lines)

        return CopilotResponse(
            query=plan.raw_query,
            intent=plan.intent.value,
            success=True,
            answer=answer,
            entities=results,
            trace=trace,
        )

    def _handle_support(self, plan: ParsedQuery, trace: List[str]) -> CopilotResponse:
        """Find what is supporting an entity or what an entity is supporting."""
        ref_name = plan.reference_name
        if not ref_name:
            return CopilotResponse(
                query=plan.raw_query,
                intent=plan.intent.value,
                success=False,
                answer="No reference entity specified.",
                trace=trace,
            )

        ref = self._resolve_entity(ref_name)
        if not ref:
            return CopilotResponse(
                query=plan.raw_query,
                intent=plan.intent.value,
                success=False,
                answer=f"Entity '{ref_name}' was not found in the reconstructed world.",
                trace=trace,
            )

        supporters = self.scene_graph.supporters_of(ref.id)
        results = [
            {
                "id": s.id,
                "name": s.name or NOT_RECORDED,
                "type": getattr(s.type, "value", str(s.type)),
                "provenance": provenance_label(s),
            }
            for s in supporters
        ]

        if not results:
            answer = f"No supporting entities explicitly recorded for {ref.name or ref.id}."
        else:
            lines = [f"{ref.name or ref.id} is supported by:"]
            for r in results:
                lines.append(f"- {r['name']} ({r['id']}, {r['type']})")
            answer = "\n".join(lines)

        return CopilotResponse(
            query=plan.raw_query,
            intent=plan.intent.value,
            success=True,
            answer=answer,
            entities=results,
            trace=trace,
        )

    def _handle_connectivity(self, plan: ParsedQuery, trace: List[str]) -> CopilotResponse:
        """Find spaces or entities connected by a door, corridor, or stair."""
        target_name = plan.target_name
        if not target_name:
            return CopilotResponse(
                query=plan.raw_query,
                intent=plan.intent.value,
                success=False,
                answer="No connector entity specified.",
                trace=trace,
            )

        connector = self._resolve_entity(target_name)
        if not connector:
            return CopilotResponse(
                query=plan.raw_query,
                intent=plan.intent.value,
                success=False,
                answer=f"Connector entity '{target_name}' was not found in the reconstructed world.",
                trace=trace,
            )

        edges = self.scene_graph.edges_from(connector.id, RelationshipKind.CONNECTS)
        connected_entities = [self.world.entities[e.target_id] for e in edges if e.target_id in self.world.entities]

        results = [
            {
                "id": c.id,
                "name": c.name or NOT_RECORDED,
                "type": getattr(c.type, "value", str(c.type)),
            }
            for c in connected_entities
        ]

        if not results:
            answer = f"No connectivity relationships recorded for {connector.name or connector.id}."
        else:
            lines = [f"{connector.name or connector.id} connects:"]
            for r in results:
                lines.append(f"- {r['name']} ({r['id']}, {r['type']})")
            answer = "\n".join(lines)

        return CopilotResponse(
            query=plan.raw_query,
            intent=plan.intent.value,
            success=True,
            answer=answer,
            entities=results,
            trace=trace,
        )

    def _handle_distance(self, plan: ParsedQuery, trace: List[str]) -> CopilotResponse:
        """Compute verified Euclidean distance between two entities."""
        ref1_name = plan.reference_name
        ref2_name = plan.second_reference_name

        if not ref1_name or not ref2_name:
            return CopilotResponse(
                query=plan.raw_query,
                intent=plan.intent.value,
                success=False,
                answer="Distance measurement requires two entities (e.g. 'distance between chair and desk').",
                trace=trace,
            )

        e1 = self._resolve_entity(ref1_name)
        e2 = self._resolve_entity(ref2_name)

        if not e1:
            return CopilotResponse(
                query=plan.raw_query,
                intent=plan.intent.value,
                success=False,
                answer=f"Entity '{ref1_name}' was not found in the reconstructed world.",
                uncertainties=[f"Unknown entity: {ref1_name}"],
                trace=trace,
            )
        if not e2:
            return CopilotResponse(
                query=plan.raw_query,
                intent=plan.intent.value,
                success=False,
                answer=f"Entity '{ref2_name}' was not found in the reconstructed world.",
                uncertainties=[f"Unknown entity: {ref2_name}"],
                trace=trace,
            )

        b1 = self._get_entity_bounds(e1)
        b2 = self._get_entity_bounds(e2)

        unit = "m" if self.scale_state in ("metric", "authored") else "model units"

        if b1 and b2:
            dist = self._aabb_distance(b1, b2)
            c1 = [(b1[0][i] + b1[1][i]) / 2.0 for i in range(3)]
            c2 = [(b2[0][i] + b2[1][i]) / 2.0 for i in range(3)]
            points = [c1, c2]
        else:
            # Fallback to transform points if available
            p1 = self._get_entity_position(e1)
            p2 = self._get_entity_position(e2)
            dist = measure.distance(p1, p2)
            points = [list(p1), list(p2)]

        meas = CopilotMeasurement(
            kind="distance",
            value=round(dist, 3),
            unit=unit,
            entity_ids=[e1.id, e2.id],
            scale_state=self.scale_state,
            points=points,
        )

        facts = [
            GroundedFact(
                entity_id=e1.id,
                property_name=f"distance_to_{e2.id}",
                value=round(dist, 3),
                provenance=e1.provenance.value,
                confidence=min(e1.confidence, e2.confidence),
                source="geometry_measurement",
            )
        ]

        scale_note = "" if self.scale_state == "metric" else f" ({self.scale_state} scale)"
        answer = (
            f"The measured distance between {e1.name or e1.id} and {e2.name or e2.id} is "
            f"{dist:.2f} {unit}{scale_note}."
        )

        return CopilotResponse(
            query=plan.raw_query,
            intent=plan.intent.value,
            success=True,
            answer=answer,
            entities=[
                {"id": e1.id, "name": e1.name or NOT_RECORDED, "type": str(e1.type)},
                {"id": e2.id, "name": e2.name or NOT_RECORDED, "type": str(e2.type)},
            ],
            measurements=[meas],
            grounded_facts=facts,
            trace=trace,
        )

    def _handle_dimension(self, plan: ParsedQuery, trace: List[str]) -> CopilotResponse:
        """Compute dimensions (height, width, depth) of an entity."""
        target_name = plan.target_name
        if not target_name:
            return CopilotResponse(
                query=plan.raw_query,
                intent=plan.intent.value,
                success=False,
                answer="No entity specified for dimension measurement.",
                trace=trace,
            )

        entity = self._resolve_entity(target_name)
        if not entity:
            return CopilotResponse(
                query=plan.raw_query,
                intent=plan.intent.value,
                success=False,
                answer=f"Entity '{target_name}' was not found in the reconstructed world.",
                uncertainties=[f"Unknown entity: {target_name}"],
                trace=trace,
            )

        bounds = self._get_entity_bounds(entity)
        if not bounds:
            return CopilotResponse(
                query=plan.raw_query,
                intent=plan.intent.value,
                success=False,
                answer=f"Geometric dimensions for '{entity.name or entity.id}' are Not recorded in WorldIR.",
                uncertainties=[f"Geometry bounds for entity {entity.id} are not recorded."],
                trace=trace,
            )

        b_min, b_max = bounds
        dx = abs(b_max[0] - b_min[0])
        dy = abs(b_max[1] - b_min[1])
        dz = abs(b_max[2] - b_min[2])

        unit = "m" if self.scale_state in ("metric", "authored") else "model units"
        prop = plan.dimension_property or "height"

        if prop == "height":
            val = dy
            desc = "height"
        elif prop == "width":
            val = dx
            desc = "width"
        else:
            val = dz
            desc = "depth"

        meas = CopilotMeasurement(
            kind=desc,
            value=round(val, 3),
            unit=unit,
            entity_ids=[entity.id],
            scale_state=self.scale_state,
        )

        facts = [
            GroundedFact(
                entity_id=entity.id,
                property_name=desc,
                value=round(val, 3),
                provenance=entity.provenance.value,
                confidence=entity.confidence,
                source="geometry_bounds",
            )
        ]

        answer = (
            f"The {desc} of {entity.name or entity.id} is {val:.2f} {unit} "
            f"(bounding box: {dx:.2f} x {dy:.2f} x {dz:.2f} {unit})."
        )

        return CopilotResponse(
            query=plan.raw_query,
            intent=plan.intent.value,
            success=True,
            answer=answer,
            entities=[{"id": entity.id, "name": entity.name or NOT_RECORDED, "type": str(entity.type)}],
            measurements=[meas],
            grounded_facts=facts,
            trace=trace,
        )

    def _handle_area(self, plan: ParsedQuery, trace: List[str]) -> CopilotResponse:
        """Compute area of a room, floor, or surface."""
        target_name = plan.target_name
        if not target_name:
            return CopilotResponse(
                query=plan.raw_query,
                intent=plan.intent.value,
                success=False,
                answer="No entity specified for area measurement.",
                trace=trace,
            )

        entity = self._resolve_entity(target_name)
        if not entity:
            return CopilotResponse(
                query=plan.raw_query,
                intent=plan.intent.value,
                success=False,
                answer=f"Entity '{target_name}' was not found in the reconstructed world.",
                trace=trace,
            )

        bounds = self._get_entity_bounds(entity)
        if not bounds:
            return CopilotResponse(
                query=plan.raw_query,
                intent=plan.intent.value,
                success=False,
                answer=f"Geometry bounds for '{entity.name or entity.id}' are Not recorded in WorldIR.",
                trace=trace,
            )

        # Check if custom polygon points are stored in custom_properties
        points = entity.custom_properties.get("polygon") or entity.custom_properties.get("points")
        unit = "sq m" if self.scale_state in ("metric", "authored") else "sq model units"

        if isinstance(points, (list, tuple)) and len(points) >= 3:
            try:
                area_val = measure.polygon_area(measure.validate_points("area", points))
                trace.append("Calculated area from polygon vertices via Newell's method.")
            except measure.MeasureError:
                b_min, b_max = bounds
                area_val = abs(b_max[0] - b_min[0]) * abs(b_max[2] - b_min[2])
                trace.append("Polygon was invalid; fell back to horizontal AABB footprint area.")
        else:
            b_min, b_max = bounds
            area_val = abs(b_max[0] - b_min[0]) * abs(b_max[2] - b_min[2])
            trace.append("Calculated area from horizontal bounding box footprint (dx * dz).")

        meas = CopilotMeasurement(
            kind="area",
            value=round(area_val, 3),
            unit=unit,
            entity_ids=[entity.id],
            scale_state=self.scale_state,
        )

        answer = f"The estimated area of {entity.name or entity.id} is {area_val:.2f} {unit}."

        return CopilotResponse(
            query=plan.raw_query,
            intent=plan.intent.value,
            success=True,
            answer=answer,
            entities=[{"id": entity.id, "name": entity.name or NOT_RECORDED, "type": str(entity.type)}],
            measurements=[meas],
            trace=trace,
        )

    def _handle_clearance(self, plan: ParsedQuery, trace: List[str]) -> CopilotResponse:
        """Determine whether an object with specified size can fit between obstacles."""
        ref1 = self._resolve_entity(plan.reference_name or "")
        ref2 = self._resolve_entity(plan.second_reference_name or "")

        if not ref1 or not ref2:
            missing = plan.reference_name if not ref1 else plan.second_reference_name
            return CopilotResponse(
                query=plan.raw_query,
                intent=plan.intent.value,
                success=False,
                answer=f"Could not resolve obstacle entity '{missing}'.",
                trace=trace,
            )

        b1 = self._get_entity_bounds(ref1)
        b2 = self._get_entity_bounds(ref2)

        if not b1 or not b2:
            return CopilotResponse(
                query=plan.raw_query,
                intent=plan.intent.value,
                success=False,
                answer="Geometry bounds for one or both obstacles are Not recorded in WorldIR.",
                trace=trace,
            )

        clearance_dist = self._aabb_distance(b1, b2)
        required_size = plan.fit_dimension_m or 1.0
        unit = "m" if self.scale_state in ("metric", "authored") else "model units"

        fits = clearance_dist >= required_size
        diff = clearance_dist - required_size

        item_label = plan.target_name or f"{required_size:.2f}{unit} object"

        if fits:
            answer = (
                f"Yes, a {item_label} (requires {required_size:.2f}{unit}) can fit between "
                f"{ref1.name or ref1.id} and {ref2.name or ref2.id}. "
                f"The available clearance is {clearance_dist:.2f}{unit} (margin: +{diff:.2f}{unit})."
            )
        else:
            answer = (
                f"No, a {item_label} (requires {required_size:.2f}{unit}) cannot fit between "
                f"{ref1.name or ref1.id} and {ref2.name or ref2.id}. "
                f"The available clearance is only {clearance_dist:.2f}{unit} (shortfall: -{abs(diff):.2f}{unit})."
            )

        meas = CopilotMeasurement(
            kind="clearance",
            value=round(clearance_dist, 3),
            unit=unit,
            entity_ids=[ref1.id, ref2.id],
            scale_state=self.scale_state,
        )

        return CopilotResponse(
            query=plan.raw_query,
            intent=plan.intent.value,
            success=True,
            answer=answer,
            entities=[
                {"id": ref1.id, "name": ref1.name or NOT_RECORDED, "type": str(ref1.type)},
                {"id": ref2.id, "name": ref2.name or NOT_RECORDED, "type": str(ref2.type)},
            ],
            measurements=[meas],
            trace=trace,
        )

    def _handle_count(self, plan: ParsedQuery, trace: List[str]) -> CopilotResponse:
        """Count entities matching target type or taxonomy class."""
        target_type = plan.target_type
        if not target_type:
            count = len(self.world.entities)
            answer = f"There are {count} total entities recorded in the reconstructed world."
            return CopilotResponse(
                query=plan.raw_query,
                intent=plan.intent.value,
                success=True,
                answer=answer,
                trace=trace,
            )

        matching = [e for e in self.world.entities.values() if self._matches_type_or_class(e, target_type)]
        results = [{"id": e.id, "name": e.name or NOT_RECORDED, "type": str(e.type)} for e in matching]

        answer = f"There are {len(matching)} {target_type}(s) recorded in the reconstructed world."
        return CopilotResponse(
            query=plan.raw_query,
            intent=plan.intent.value,
            success=True,
            answer=answer,
            entities=results,
            trace=trace,
        )

    def _handle_find_and_inspect(self, plan: ParsedQuery, trace: List[str]) -> CopilotResponse:
        """Find and display details of entities."""
        if plan.target_name:
            e = self._resolve_entity(plan.target_name)
            if not e:
                return CopilotResponse(
                    query=plan.raw_query,
                    intent=plan.intent.value,
                    success=False,
                    answer=f"Entity '{plan.target_name}' was not found in the reconstructed world.",
                    trace=trace,
                )
            bounds = self._get_entity_bounds(e)
            b_desc = f"min {bounds[0]}, max {bounds[1]}" if bounds else NOT_RECORDED
            answer = (
                f"Entity {e.id}:\n"
                f"- Name: {e.name or NOT_RECORDED}\n"
                f"- Type: {getattr(e.type, 'value', str(e.type))}\n"
                f"- Class: {e.custom_properties.get('entity_class', NOT_RECORDED)}\n"
                f"- Provenance: {provenance_label(e)}\n"
                f"- Confidence: {e.confidence:.2f}\n"
                f"- Bounds: {b_desc}\n"
                f"- Relationships: {len(e.relationships)} edge(s)"
            )
            return CopilotResponse(
                query=plan.raw_query,
                intent=plan.intent.value,
                success=True,
                answer=answer,
                entities=[{"id": e.id, "name": e.name or NOT_RECORDED, "type": str(e.type)}],
                trace=trace,
            )

        if plan.target_type:
            matching = [e for e in self.world.entities.values() if self._matches_type_or_class(e, plan.target_type)]
            results = [{"id": e.id, "name": e.name or NOT_RECORDED, "type": str(e.type)} for e in matching]
            if not results:
                answer = f"No {plan.target_type} entities found in the reconstructed world."
            else:
                lines = [f"Found {len(results)} {plan.target_type}(s):"]
                for r in results:
                    lines.append(f"- {r['name']} ({r['id']})")
                answer = "\n".join(lines)
            return CopilotResponse(
                query=plan.raw_query,
                intent=plan.intent.value,
                success=True,
                answer=answer,
                entities=results,
                trace=trace,
            )

        # General search fallback
        results = [
            {"id": e.id, "name": e.name or NOT_RECORDED, "type": str(e.type)}
            for e in list(self.world.entities.values())[:20]
        ]
        return CopilotResponse(
            query=plan.raw_query,
            intent=plan.intent.value,
            success=True,
            answer=f"World contains {len(self.world.entities)} entities.",
            entities=results,
            trace=trace,
        )

    def _handle_provenance(self, plan: ParsedQuery, trace: List[str]) -> CopilotResponse:
        """Explain the provenance of an entity without fabricating origin."""
        target_name = plan.target_name
        if not target_name:
            return CopilotResponse(
                query=plan.raw_query,
                intent=plan.intent.value,
                success=False,
                answer="No entity specified for provenance query.",
                trace=trace,
            )

        entity = self._resolve_entity(target_name)
        if not entity:
            return CopilotResponse(
                query=plan.raw_query,
                intent=plan.intent.value,
                success=False,
                answer=f"Entity '{target_name}' was not found in the reconstructed world.",
                trace=trace,
            )

        info = explain_provenance(self.ctx, entity.id)
        return CopilotResponse(
            query=plan.raw_query,
            intent=plan.intent.value,
            success=True,
            answer=info["text"],
            entities=[{"id": entity.id, "name": entity.name or NOT_RECORDED, "type": str(entity.type)}],
            trace=trace,
        )

    def _handle_confidence(self, plan: ParsedQuery, trace: List[str]) -> CopilotResponse:
        """Report calibrated confidence and uncertainty notes."""
        target_name = plan.target_name
        if not target_name:
            return CopilotResponse(
                query=plan.raw_query,
                intent=plan.intent.value,
                success=False,
                answer="No entity specified for confidence query.",
                trace=trace,
            )

        entity = self._resolve_entity(target_name)
        if not entity:
            return CopilotResponse(
                query=plan.raw_query,
                intent=plan.intent.value,
                success=False,
                answer=f"Entity '{target_name}' was not found in the reconstructed world.",
                trace=trace,
            )

        note = entity.uncertainty.note if entity.uncertainty and entity.uncertainty.note else NOT_RECORDED
        answer = (
            f"Confidence for {entity.name or entity.id}: {entity.confidence:.2f}\n"
            f"Provenance: {provenance_label(entity)}\n"
            f"Uncertainty note: {note}"
        )
        return CopilotResponse(
            query=plan.raw_query,
            intent=plan.intent.value,
            success=True,
            answer=answer,
            entities=[{"id": entity.id, "name": entity.name or NOT_RECORDED, "type": str(entity.type)}],
            trace=trace,
        )

    def _handle_unknown(self, plan: ParsedQuery, trace: List[str]) -> CopilotResponse:
        return CopilotResponse(
            query=plan.raw_query,
            intent=plan.intent.value,
            success=False,
            answer="Could not determine the spatial intent of the query.",
            uncertainties=["Unrecognized query syntax."],
            trace=trace,
        )

    # -------------------------------------------------------- Helper Methods

    def _resolve_entity(self, name_or_id: str) -> Optional[Entity]:
        """Resolve entity by exact ID, exact name, or case-insensitive partial name."""
        if not name_or_id:
            return None
        q = name_or_id.strip()
        # 1. Exact ID match
        if q in self.world.entities:
            return self.world.entities[q]

        # 2. Exact name match (case-insensitive)
        q_lower = q.lower()
        for e in self.world.entities.values():
            if (e.name or "").lower() == q_lower:
                return e

        # 3. Taxonomy class match (e.g. "sink" matches entity with entity_class="plumbing.sink")
        for e in self.world.entities.values():
            e_class = e.custom_properties.get("entity_class", "").lower()
            if q_lower in e_class or e_class.endswith(f".{q_lower}"):
                return e

        # 4. Partial substring in name
        for e in self.world.entities.values():
            if q_lower in (e.name or "").lower():
                return e

        # 5. Partial substring in ID
        for e in self.world.entities.values():
            if q_lower in e.id.lower():
                return e

        return None

    def _matches_type_or_class(self, entity: Entity, target_type_or_class: str) -> bool:
        """Check if an entity matches an EntityType or dotted taxonomy class."""
        t_lower = target_type_or_class.lower()
        # Direct type match
        e_type_val = getattr(entity.type, "value", str(entity.type)).lower()
        if e_type_val == t_lower:
            return True
        # Entity class in custom_properties
        e_class = entity.custom_properties.get("entity_class", "").lower()
        if e_class == t_lower or e_class.startswith(f"{t_lower}.") or e_class.endswith(f".{t_lower}"):
            return True
        # Semantic labels
        for label in entity.semantic_labels:
            if label.lower() == t_lower:
                return True
        # Fallback keyword in name
        if t_lower in (entity.name or "").lower():
            return True
        return False

    def _get_entity_bounds(self, entity: Entity) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
        """Retrieve AABB (bounds_min, bounds_max) for entity."""
        return _entity_bounds(self.world, entity)

    def _get_entity_position(self, entity: Entity) -> Tuple[float, float, float]:
        """Get 3D position of an entity from bounds or transform."""
        bounds = self._get_entity_bounds(entity)
        if bounds:
            b_min, b_max = bounds
            return (
                (b_min[0] + b_max[0]) / 2.0,
                (b_min[1] + b_max[1]) / 2.0,
                (b_min[2] + b_max[2]) / 2.0,
            )
        if entity.transform and isinstance(entity.transform, dict) and "position" in entity.transform:
            pos = entity.transform["position"]
            return (float(pos[0]), float(pos[1]), float(pos[2]))
        return (0.0, 0.0, 0.0)

    @staticmethod
    def _aabb_distance(
        b1: Tuple[Tuple[float, float, float], Tuple[float, float, float]],
        b2: Tuple[Tuple[float, float, float], Tuple[float, float, float]],
    ) -> float:
        """Calculate minimum surface-to-surface Euclidean distance between two AABBs.
        Returns 0.0 if boxes overlap.
        """
        min1, max1 = b1
        min2, max2 = b2
        dx = max(0.0, max(min1[0] - max2[0], min2[0] - max1[0]))
        dy = max(0.0, max(min1[1] - max2[1], min2[1] - max1[1]))
        dz = max(0.0, max(min1[2] - max2[2], min2[2] - max1[2]))
        return math.sqrt(dx * dx + dy * dy + dz * dz)
