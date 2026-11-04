"""Unit and integration tests for Universal Spatial Intelligence Directive Rule 2:
Verification Model Repair & Truthful Execution Classifications.
"""

import pytest

from engine.technologies.adapters import get_adapter_for_technology
from engine.technologies.benchmark import TechnologyBenchmarkHarness
from engine.technologies.registry import get_technology_registry
from engine.verification.classification import (
    CapabilityVerificationStatus,
    ExecutionClassification,
    validate_observation_provenance,
)
from perception.reasoning.physical_plausibility import validate_physical_plausibility
from planning.completeness import evaluate_reconstruction_completeness
from provenance import Provenance
from world_ir.schema_v1 import Entity, EntityType, Geometry, GeometryType, Vector3
from world_ir.world_v1 import WorldIR


def test_provenance_invariant_enforcement():
    """Verify that simulated or synthetic outputs can never enter production as Provenance.OBSERVED."""
    # Valid pairings
    validate_observation_provenance(ExecutionClassification.INTERNAL_IMPLEMENTATION, Provenance.RECONSTRUCTED)
    validate_observation_provenance(ExecutionClassification.NATIVE_EXECUTED, Provenance.OBSERVED)
    validate_observation_provenance(ExecutionClassification.WRAPPED_EXECUTED, Provenance.OBSERVED)
    validate_observation_provenance(ExecutionClassification.SIMULATED, Provenance.INFERRED)
    validate_observation_provenance(ExecutionClassification.SYNTHETIC_TEST, Provenance.INFERRED)

    # Invariant violations must raise ValueError
    with pytest.raises(ValueError, match="CRITICAL INVARIANT VIOLATION"):
        validate_observation_provenance(ExecutionClassification.SIMULATED, Provenance.OBSERVED)

    with pytest.raises(ValueError, match="CRITICAL INVARIANT VIOLATION"):
        validate_observation_provenance(ExecutionClassification.SYNTHETIC_TEST, Provenance.OBSERVED)


def test_benchmark_result_contains_all_rule_2_fields():
    """Verify benchmark assessment reports all metadata required by Rule 2."""
    harness = TechnologyBenchmarkHarness()

    # Benchmark TEASER++ (Internal Implementation)
    res_teaser = harness.benchmark_technology("teaser_plus_plus")
    assert res_teaser.classification == ExecutionClassification.INTERNAL_IMPLEMENTATION.value
    assert res_teaser.verification_status == CapabilityVerificationStatus.SYNTHETICALLY_VERIFIED.value
    assert len(res_teaser.dataset_id) > 0
    assert len(res_teaser.dataset_checksum) > 0
    assert len(res_teaser.ground_truth_source) > 0
    assert len(res_teaser.hardware_provider) > 0
    assert res_teaser.execution_time_ms > 0
    assert "registration_rmse_m" in res_teaser.metrics
    # Must be measured empirical value (sub-millimeter in synthetic affine registration)
    assert res_teaser.metrics["registration_rmse_m"].value < 0.001
    assert res_teaser.metrics["registration_rmse_m"].measurement_methodology != ""

    # Benchmark Manifold (Internal Implementation)
    res_manifold = harness.benchmark_technology("manifold")
    assert res_manifold.classification == ExecutionClassification.INTERNAL_IMPLEMENTATION.value
    assert res_manifold.metrics["manifold_integrity_pct"].value == 100.0

    # Benchmark a technology requiring fallback (e.g. Detectron2)
    res_detectron = harness.benchmark_technology("detectron2")
    assert res_detectron.classification == ExecutionClassification.SIMULATED.value
    assert res_detectron.verification_status == CapabilityVerificationStatus.NOT_IMPLEMENTED.value
    assert "Simulated fallback model" in res_detectron.measurement_methodology


def test_completeness_evidence_honesty():
    """Verify completeness never fabricates 0.95 observed fraction for unobserved or inferred entities."""
    world = WorldIR()

    # Inferred Wall (closure)
    w_inferred = Entity(
        id="wall_inferred_1",
        name="Inferred Wall Closure",
        type=EntityType.WALL,
        provenance=Provenance.INFERRED,
    )
    # Directly Observed Floor
    fl_observed = Entity(
        id="floor_obs_1",
        name="Observed Tile Floor",
        type=EntityType.FLOOR,
        provenance=Provenance.OBSERVED,
    )
    world.entities[w_inferred.id] = w_inferred
    world.entities[fl_observed.id] = fl_observed

    report = evaluate_reconstruction_completeness(world)

    # Inferred entity must have 0.0 directly observed fraction (not 0.35 or 0.95!)
    assert report.coverage_heatmap["wall_inferred_1"] == 0.0
    # Observed entity has evidence-backed confidence (1.0 default)
    assert report.coverage_heatmap["floor_obs_1"] == 1.0


def test_physical_plausibility_enforces_execution_provenance():
    """Verify physical plausibility validation rejects simulated outputs tagged as OBSERVED."""
    world = WorldIR()

    # Geometry for floor
    g_fl = Geometry(id="g_fl", type=GeometryType.BOX, bounds_min=Vector3(x=0, y=0, z=0), bounds_max=Vector3(x=2, y=0.1, z=2))
    world.geometries[g_fl.id] = g_fl

    # Entity tagged as SIMULATED execution classification but falsely marked as OBSERVED provenance
    e_bad = Entity(
        id="bad_simulated_entity",
        name="Fake Observed Entity",
        type=EntityType.FLOOR,
        geometry_ids=["g_fl"],
        provenance=Provenance.OBSERVED,
        custom_properties={"execution_classification": "SIMULATED"},
    )
    world.entities[e_bad.id] = e_bad

    with pytest.raises(ValueError, match="violates physical provenance rules"):
        validate_physical_plausibility(world)
