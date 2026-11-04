"""Tests for Autonomous Quality and Correction Loop (Directive Rule 9).

Verifies:
- Multi-dimensional quality audit (completeness, scale fidelity, identity integrity, structural soundness)
- Autonomous detection and correction of metric scale drift
- Autonomous consolidation of fragmented co-located duplicate entities
- Selective rejection of candidate modifications that do not improve quality
- Strict recording of correction provenance in the world audit log
"""

from __future__ import annotations

import pytest

from engine.quality.autonomous_correction_loop import (
    AutonomousCorrectionLoop,
    CorrectionHypothesis,
    DefectKind,
    QualityAuditor,
)
from provenance import Provenance
from world_ir import (
    Entity,
    EntityType,
    Geometry,
    GeometryType,
    Observation,
    Vector3,
    WorldIR,
)


def _add_box(
    world: WorldIR,
    entity_id: str,
    name: str,
    entity_type: EntityType,
    min_pt: tuple[float, float, float],
    max_pt: tuple[float, float, float],
) -> Entity:
    g_id = f"geom-{entity_id}"
    geom = Geometry(
        id=g_id,
        type=GeometryType.BOX,
        bounds_min=Vector3(*min_pt),
        bounds_max=Vector3(*max_pt),
        provenance=Provenance.RECONSTRUCTED,
    )
    world.geometries[g_id] = geom

    ent = Entity(
        id=entity_id,
        name=name,
        type=entity_type,
        geometry_ids=[g_id],
        provenance=Provenance.INFERRED,
    )
    world.entities[entity_id] = ent
    return ent


class TestAutonomousQualityLoopRule9:
    def test_quality_auditor_baseline(self):
        """Test multi-dimensional quality metric computation on a standard scene."""
        world = WorldIR()
        _add_box(world, "floor-1", "Ground Slab", EntityType.FLOOR, (0, 0, 0), (6, 6, 0.2))
        _add_box(world, "wall-1", "North Wall", EntityType.WALL, (0, 0, 0.2), (6, 0.2, 2.8))
        _add_box(world, "door-1", "Entry Door", EntityType.DOOR, (2, 0, 0.2), (2.9, 0.1, 2.25))

        audit = QualityAuditor.audit(world)

        assert 0.0 <= audit.completeness_score <= 1.0
        assert audit.scale_fidelity_score > 0.80  # Door is 2.05m high, perfect match
        assert audit.identity_integrity_score == 1.0  # No duplicates
        assert audit.overall_quality_score > 0.60

    def test_scale_drift_detection_and_autonomous_correction(self):
        """Test autonomous detection and correction of metric scale drift."""
        world = WorldIR()
        # Scaled-down world where door height is only 0.205m (factor of 10 too small)
        _add_box(world, "floor-1", "Tiny Slab", EntityType.FLOOR, (0, 0, 0), (0.6, 0.6, 0.02))
        _add_box(world, "wall-1", "Tiny Wall", EntityType.WALL, (0, 0, 0.02), (0.6, 0.02, 0.28))
        _add_box(world, "door-1", "Tiny Door", EntityType.DOOR, (0.2, 0, 0.02), (0.29, 0.01, 0.225))

        initial_audit = QualityAuditor.audit(world)
        assert initial_audit.scale_fidelity_score < 0.60  # Scale drift detected

        # Run autonomous loop
        report = AutonomousCorrectionLoop.run_autonomous_loop(world, max_iterations=2)

        assert report.corrections_accepted >= 1
        assert report.final_metrics.overall_quality_score > report.initial_metrics.overall_quality_score
        assert "autonomous_corrections_log" in world.metadata

        # Verify door has been scaled back to ~2.05m
        door_geom = world.geometries["geom-door-1"]
        h = door_geom.bounds_max.z - door_geom.bounds_min.z
        assert 1.80 <= h <= 2.40

    def test_identity_fragmentation_consolidation(self):
        """Test autonomous consolidation of co-located duplicate entities."""
        world = WorldIR()
        _add_box(world, "floor-1", "Ground Slab", EntityType.FLOOR, (0, 0, 0), (6, 6, 0.2))

        # Duplicate chairs at the same location
        _add_box(world, "chair-dup1", "Office Chair A", EntityType.STRUCTURE, (2.0, 2.0, 0.2), (2.5, 2.5, 1.0))
        _add_box(world, "chair-dup2", "Office Chair B", EntityType.STRUCTURE, (2.02, 2.01, 0.2), (2.52, 2.51, 1.0))

        initial_audit = QualityAuditor.audit(world)
        assert initial_audit.identity_integrity_score < 1.0

        report = AutonomousCorrectionLoop.run_autonomous_loop(world, max_iterations=2)

        assert report.corrections_accepted >= 1
        # One of the duplicates must have been pruned
        remaining_chairs = [e for e in world.entities.values() if "chair" in e.id]
        assert len(remaining_chairs) == 1
        assert report.final_metrics.identity_integrity_score == 1.0

    def test_selective_rejection_of_harmful_correction(self):
        """Test that candidate corrections offering zero or negative quality gain are rejected."""
        world = WorldIR()
        _add_box(world, "floor-1", "Ground Slab", EntityType.FLOOR, (0, 0, 0), (6, 6, 0.2))
        _add_box(world, "door-1", "Standard Door", EntityType.DOOR, (2, 0, 0.2), (2.9, 0.1, 2.25))

        audit_before = QualityAuditor.audit(world)

        # Harmful hypothesis: rescale an already well-calibrated world by 5x
        bad_hyp = CorrectionHypothesis(
            hypothesis_id="hyp-bad-scale",
            defect_kind=DefectKind.SCALE_DRIFT,
            target_entity_ids=tuple(world.entities.keys()),
            action_type="rescale_world",
            parameters={"scale_factor": 5.0},
            confidence=0.10,
            rationale="Harmful arbitrary scale change",
        )

        candidate = WorldIR.from_dict(world.to_dict())
        AutonomousCorrectionLoop.apply_hypothesis_in_candidate(candidate, bad_hyp)
        cand_audit = QualityAuditor.audit(candidate)

        # Quality should drop significantly
        assert cand_audit.overall_quality_score < audit_before.overall_quality_score
