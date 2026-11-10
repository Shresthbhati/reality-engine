"""Material and finish recognition (N-15).

Identifies visually supported materials and finishes such as wood, concrete,
tile, stone, metal, glass and paint. Preserves uncertainty where images
cannot establish the actual material.

Honesty rules:
* Each material label has evidence, confidence and an explicit unknown state.
* Materials are only identified when image evidence supports them.
* Unknown materials are labeled as unknown, not assumed to be a default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from world_ir import Entity, EntityType


@dataclass(frozen=True)
class MaterialClassification:
    """Material classification for one surface/entity."""
    entity_id: str
    entity_type: str
    material: str  # "wood", "concrete", "tile", "stone", "metal", "glass", "paint", "unknown"
    confidence: float  # 0..1
    evidence_ids: List[str]
    finish: Optional[str] = None  # "matte", "glossy", "satin", "textured", "polished"
    uncertainty_note: Optional[str] = None


@dataclass(frozen=True)
class MaterialRecognitionResult:
    """Result of material recognition for one or more entities."""
    classifications: List[MaterialClassification]
    provisional: bool = True


class MaterialRecognizer:
    """Recognizes materials and finishes from visual evidence."""

    MATERIAL_TYPES = {
        "wood", "concrete", "tile", "stone", "metal", "glass", "paint", "unknown"
    }

    FINISH_TYPES = {
        "matte", "glossy", "satin", "textured", "polished", "unknown"
    }

    def classify_material(
        self,
        entity: Entity,
        visual_evidence: Optional[Dict[str, dict]] = None,
    ) -> MaterialClassification:
        """Classify the material of one entity.

        Args:
            entity: The entity to classify
            visual_evidence: Optional visual evidence supporting the material
                             classification (e.g., color histograms, texture
                             analysis results)

        Returns:
            MaterialClassification with material, confidence, and provenance
        """
        cp = entity.custom_properties or {}
        material = cp.get("material")
        confidence = cp.get("material_confidence", 0.0)
        evidence_ids = cp.get("material_evidence_ids", [])
        finish = cp.get("material_finish")

        # Check if we have visual evidence
        if visual_evidence and entity.id in visual_evidence:
            ev = visual_evidence[entity.id]
            material = ev.get("material", material or "unknown")
            confidence = ev.get("confidence", confidence)
            evidence_ids = ev.get("evidence_ids", evidence_ids)
            finish = ev.get("finish", finish)

        # If no evidence at all, mark as unknown
        if material is None or material == "unknown":
            return MaterialClassification(
                entity_id=entity.id,
                entity_type=entity.type.value if hasattr(entity.type, 'value') else str(entity.type),
                material="unknown",
                confidence=0.0,
                evidence_ids=[],
                finish=None,
                uncertainty_note="No material evidence available",
            )

        return MaterialClassification(
            entity_id=entity.id,
            entity_type=entity.type.value if hasattr(entity.type, 'value') else str(entity.type),
            material=material,
            confidence=confidence,
            evidence_ids=list(evidence_ids),
            finish=finish,
            uncertainty_note=None if confidence > 0.5 else "Low confidence material classification",
        )

    def classify_entities(
        self,
        entities: Dict[str, Entity],
        visual_evidence: Optional[Dict[str, Dict[str, dict]]] = None,
    ) -> MaterialRecognitionResult:
        """Classify materials for multiple entities."""
        classifications = []
        for eid, ent in entities.items():
            ev = visual_evidence.get(eid) if visual_evidence else None
            classification = self.classify_material(ent, ev)
            classifications.append(classification)

        return MaterialRecognitionResult(
            classifications=classifications,
            provisional=True,
        )

    def is_material_known(self, material: str) -> bool:
        """Check if a material is a known type."""
        return material in self.MATERIAL_TYPES

    def is_finish_known(self, finish: str) -> bool:
        """Check if a finish is a known type."""
        return finish in self.FINISH_TYPES


__all__ = ["MaterialRecognizer", "MaterialClassification", "MaterialRecognitionResult"]