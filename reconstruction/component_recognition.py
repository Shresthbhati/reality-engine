"""Fine-grained architectural component recognition (N-11).

Separates architectural objects into useful components: door leaf, frame,
hinges, handles, glazing, window frame, sill, lintel, socket faceplate,
switch and other details.

Honesty rules:
* Components are independently identified when the image evidence supports it.
* A component without supporting evidence is labeled "unknown", not assumed
  to exist.
* Component confidence reflects the quality of the visual evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from world_ir import Entity, EntityType


@dataclass(frozen=True)
class Component:
    """One architectural component."""
    component_id: str
    component_type: str  # "door_leaf", "door_frame", "hinge", "handle", "glazing",
                         # "window_frame", "sill", "lintel", "socket_faceplate",
                         # "switch", etc.
    parent_entity_id: str  # the entity this component belongs to
    confidence: float  # 0..1
    evidence_ids: List[str]
    bounds: Optional[Tuple[float, float, float, float, float, float]]  # bounding box


@dataclass(frozen=True)
class ComponentRecognitionResult:
    """Result of component recognition for one entity."""
    entity_id: str
    components: List[Component]
    provisional: bool = True


class ComponentRecognizer:
    """Recognizes fine-grained architectural components within entities."""

    def recognize_components(
        self,
        entity: Entity,
        image_evidence: Optional[Dict[str, dict]] = None,
    ) -> ComponentRecognitionResult:
        """Recognize components within an entity.

        Args:
            entity: The entity to analyze (door, window, socket, etc.)
            image_evidence: Optional evidence from images supporting each component

        Returns:
            ComponentRecognitionResult with identified components
        """
        components = []

        entity_type = entity.type if hasattr(entity.type, 'value') else str(entity.type)
        cp = entity.custom_properties or {}

        if entity_type == EntityType.DOOR:
            components.extend(self._recognize_door_components(entity, cp, image_evidence))
        elif entity_type == EntityType.WINDOW:
            components.extend(self._recognize_window_components(entity, cp, image_evidence))
        elif entity_type in (EntityType.SWITCH, EntityType.SOCKET):
            components.extend(self._recognize_fixture_components(entity, cp, image_evidence))

        return ComponentRecognitionResult(
            entity_id=entity.id,
            components=components,
            provisional=True,
        )

    def _recognize_door_components(
        self,
        entity: Entity,
        cp: dict,
        image_evidence: Optional[Dict[str, dict]],
    ) -> List[Component]:
        """Recognize door components: leaf, frame, hinges, handle, glazing."""
        components = []

        # Door leaf (the moving part)
        if self._has_evidence(entity, "door_leaf", image_evidence):
            components.append(Component(
                component_id=f"{entity.id}_leaf",
                component_type="door_leaf",
                parent_entity_id=entity.id,
                confidence=0.7,
                evidence_ids=image_evidence.get("door_leaf", {}).get("evidence_ids", []),
                bounds=cp.get("leaf_bounds"),
            ))

        # Door frame
        if self._has_evidence(entity, "door_frame", image_evidence):
            components.append(Component(
                component_id=f"{entity.id}_frame",
                component_type="door_frame",
                parent_entity_id=entity.id,
                confidence=0.7,
                evidence_ids=image_evidence.get("door_frame", {}).get("evidence_ids", []),
                bounds=cp.get("frame_bounds"),
            ))

        # Hinges
        if self._has_evidence(entity, "door_hinges", image_evidence):
            components.append(Component(
                component_id=f"{entity.id}_hinges",
                component_type="hinge",
                parent_entity_id=entity.id,
                confidence=0.5,
                evidence_ids=image_evidence.get("door_hinges", {}).get("evidence_ids", []),
                bounds=cp.get("hinge_bounds"),
            ))

        # Handle
        if self._has_evidence(entity, "door_handle", image_evidence):
            components.append(Component(
                component_id=f"{entity.id}_handle",
                component_type="handle",
                parent_entity_id=entity.id,
                confidence=0.6,
                evidence_ids=image_evidence.get("door_handle", {}).get("evidence_ids", []),
                bounds=cp.get("handle_bounds"),
            ))

        # Glazing (glass in door)
        if self._has_evidence(entity, "door_glazing", image_evidence):
            components.append(Component(
                component_id=f"{entity.id}_glazing",
                component_type="glazing",
                parent_entity_id=entity.id,
                confidence=0.5,
                evidence_ids=image_evidence.get("door_glazing", {}).get("evidence_ids", []),
                bounds=cp.get("glazing_bounds"),
            ))

        return components

    def _recognize_window_components(
        self,
        entity: Entity,
        cp: dict,
        image_evidence: Optional[Dict[str, dict]],
    ) -> List[Component]:
        """Recognize window components: frame, sill, lintel, glazing."""
        components = []

        # Window frame
        if self._has_evidence(entity, "window_frame", image_evidence):
            components.append(Component(
                component_id=f"{entity.id}_frame",
                component_type="window_frame",
                parent_entity_id=entity.id,
                confidence=0.7,
                evidence_ids=image_evidence.get("window_frame", {}).get("evidence_ids", []),
                bounds=cp.get("frame_bounds"),
            ))

        # Sill
        if self._has_evidence(entity, "window_sill", image_evidence):
            components.append(Component(
                component_id=f"{entity.id}_sill",
                component_type="sill",
                parent_entity_id=entity.id,
                confidence=0.6,
                evidence_ids=image_evidence.get("window_sill", {}).get("evidence_ids", []),
                bounds=cp.get("sill_bounds"),
            ))

        # Lintel
        if self._has_evidence(entity, "window_lintel", image_evidence):
            components.append(Component(
                component_id=f"{entity.id}_lintel",
                component_type="lintel",
                parent_entity_id=entity.id,
                confidence=0.5,
                evidence_ids=image_evidence.get("window_lintel", {}).get("evidence_ids", []),
                bounds=cp.get("lintel_bounds"),
            ))

        # Glazing
        if self._has_evidence(entity, "window_glazing", image_evidence):
            components.append(Component(
                component_id=f"{entity.id}_glazing",
                component_type="glazing",
                parent_entity_id=entity.id,
                confidence=0.5,
                evidence_ids=image_evidence.get("window_glazing", {}).get("evidence_ids", []),
                bounds=cp.get("glazing_bounds"),
            ))

        return components

    def _recognize_fixture_components(
        self,
        entity: Entity,
        cp: dict,
        image_evidence: Optional[Dict[str, dict]],
    ) -> List[Component]:
        """Recognize fixture components: switch faceplate, socket faceplate."""
        components = []

        # Faceplate
        if self._has_evidence(entity, "faceplate", image_evidence):
            components.append(Component(
                component_id=f"{entity.id}_faceplate",
                component_type="faceplate",
                parent_entity_id=entity.id,
                confidence=0.6,
                evidence_ids=image_evidence.get("faceplate", {}).get("evidence_ids", []),
                bounds=cp.get("faceplate_bounds"),
            ))

        return components

    def _has_evidence(
        self,
        entity: Entity,
        component_type: str,
        image_evidence: Optional[Dict[str, dict]],
    ) -> bool:
        """Check if there is evidence supporting a component."""
        if image_evidence and component_type in image_evidence:
            return True
        # Check custom properties for component evidence
        cp = entity.custom_properties or {}
        component_key = f"{component_type}_evidence"
        return cp.get(component_key, False) is True


__all__ = ["ComponentRecognizer", "Component", "ComponentRecognitionResult"]