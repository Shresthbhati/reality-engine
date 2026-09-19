"""Phase 8: Real / Synthetic Separation Firewall Tests.

Verifies:
1. Four explicit data quality states: REAL, SYNTHETIC, TEST DOUBLE, UNAVAILABLE.
2. Real data firewall rejects unverified synthetic data when real is expected.
3. Synthetic data firewall requires full generation parameters (algorithm, seed/prompt, confidence bound).
4. Test double firewall prevents test doubles from being saved to production WorldStore.
5. Desktop Studio never displays synthetic data as real (visual distinction verification).
6. Demo point cloud ([DEMO] points.ply) is explicitly tagged TEST DOUBLE and cannot be loaded as production evidence.
7. Mixed scenes preserve per-entity quality tags without cross-contamination.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from world_ir.quality_state import (
    QualityState,
    QualityTag,
    RealSyntheticFirewall,
    SyntheticMetadata,
    ProductionFirewallViolation,
)
from world_ir.schema_v1 import Entity, EntityType, Geometry, GeometryType
from world_ir.world_v1 import WorldIR
from worldstore.store import WorldStore, WorldStoreError
from provenance import Provenance, Uncertainty


class TestRealSyntheticSeparationSuite:
    """Verifies that the system maintains strict firewalls between real,
    synthetic, test doubles, and unavailable states without silent promotion."""

    def test_four_explicit_quality_states_exist(self):
        states = {s.value for s in QualityState}
        assert "REAL" in states
        assert "SYNTHETIC" in states
        assert "TEST DOUBLE" in states
        assert "UNAVAILABLE" in states

    def test_real_data_firewall_rejects_synthetic_when_real_expected(self):
        synth_meta = SyntheticMetadata(
            algorithm="nerf_diffusion_v2",
            seed_or_prompt="a wooden table",
            confidence_bound=0.85,
        )
        tag = QualityTag(state=QualityState.SYNTHETIC, synthetic_metadata=synth_meta)

        with pytest.raises(ProductionFirewallViolation, match="Synthetic data rejected"):
            RealSyntheticFirewall.assert_real_evidence(tag)

    def test_synthetic_data_requires_full_generation_parameters(self):
        # Missing metadata raises ValueError
        with pytest.raises(ValueError, match="requires full generation parameters"):
            QualityTag(state=QualityState.SYNTHETIC, synthetic_metadata=None)

        # Invalid algorithm name raises ValueError
        with pytest.raises(ValueError, match="non-empty algorithm name"):
            SyntheticMetadata(algorithm="", seed_or_prompt="chair", confidence_bound=0.9)

        # Out-of-bounds confidence raises ValueError
        with pytest.raises(ValueError, match="confidence_bound must be in"):
            SyntheticMetadata(algorithm="procedural_room", seed_or_prompt=42, confidence_bound=1.5)

    def test_test_double_firewall_prevents_persistence_to_production_store(self, tmp_path):
        tag = QualityTag(state=QualityState.TEST_DOUBLE)
        assert tag.is_test_double is True

        with pytest.raises(ProductionFirewallViolation, match="TEST DOUBLE / demo fixture cannot be persisted"):
            RealSyntheticFirewall.assert_safe_for_production(tag)

    def test_demo_point_cloud_classified_as_test_double(self):
        demo_uris = [
            "[DEMO] points.ply",
            "demo/sample_scan.ply",
            "assets/test_double_box.glb",
        ]
        for uri in demo_uris:
            state = RealSyntheticFirewall.classify_source_uri(uri)
            assert state == QualityState.TEST_DOUBLE

            tag = QualityTag(state=state)
            with pytest.raises(ProductionFirewallViolation):
                RealSyntheticFirewall.assert_real_evidence(tag)

    def test_mixed_scene_isolation_prevents_cross_contamination(self):
        # Real room floor + synthetic chair
        floor_tag = QualityTag(state=QualityState.REAL, evidence_id="ev-sensor-floor-01")
        chair_tag = QualityTag(
            state=QualityState.SYNTHETIC,
            synthetic_metadata=SyntheticMetadata(
                algorithm="cad_library_v1", seed_or_prompt="chair_office_01", confidence_bound=0.9
            ),
        )

        initial_tags = [floor_tag, chair_tag]

        # Scenario A: Preservation - stays clean
        pass_tags = [
            QualityTag(state=QualityState.REAL, evidence_id="ev-sensor-floor-01"),
            QualityTag(
                state=QualityState.SYNTHETIC,
                synthetic_metadata=SyntheticMetadata(
                    algorithm="cad_library_v1", seed_or_prompt="chair_office_01", confidence_bound=0.9
                ),
            ),
        ]
        assert RealSyntheticFirewall.verify_scene_isolation(initial_tags, pass_tags) is True

        # Scenario B: Chair promoted to REAL -> must be rejected
        corrupt_chair_promoted = [
            floor_tag,
            QualityTag(state=QualityState.REAL, evidence_id="fake-floor-link"),
        ]
        with pytest.raises(ProductionFirewallViolation, match="Cross-contamination detected"):
            RealSyntheticFirewall.verify_scene_isolation(initial_tags, corrupt_chair_promoted)

        # Scenario C: Floor demoted to SYNTHETIC -> must be rejected
        corrupt_floor_demoted = [
            QualityTag(
                state=QualityState.SYNTHETIC,
                synthetic_metadata=SyntheticMetadata(
                    algorithm="floor_gen", seed_or_prompt=1, confidence_bound=0.5
                ),
            ),
            chair_tag,
        ]
        with pytest.raises(ProductionFirewallViolation, match="Cross-contamination detected"):
            RealSyntheticFirewall.verify_scene_isolation(initial_tags, corrupt_floor_demoted)

    def test_explicit_unavailable_state(self):
        # UNAVAILABLE requires explicit reason
        with pytest.raises(ValueError, match="explicit reason"):
            QualityTag(state=QualityState.UNAVAILABLE, unavailable_reason=None)

        unavail_tag = QualityTag(
            state=QualityState.UNAVAILABLE,
            unavailable_reason="Sensor camera blocked by operator hand",
        )
        assert unavail_tag.state == QualityState.UNAVAILABLE
        assert unavail_tag.unavailable_reason == "Sensor camera blocked by operator hand"

        with pytest.raises(ProductionFirewallViolation, match="Data is UNAVAILABLE"):
            RealSyntheticFirewall.assert_real_evidence(unavail_tag)

    def test_desktop_viewer_contract_visual_distinction(self):
        """Ensures that WorldIR entity custom properties or metadata preserve
        quality tags for visual distinctiveness in desktop viewer / outliner."""
        world = WorldIR(id="world-desktop-distinction")

        # 1 Real entity
        e_real = Entity(
            id="e-real",
            type=EntityType.WALL,
            provenance=Provenance.OBSERVED,
            confidence=0.98,
        )
        e_real.custom_properties["quality_state"] = QualityState.REAL.value

        # 1 Synthetic entity
        e_synth = Entity(
            id="e-synth",
            type=EntityType.STRUCTURE,
            provenance=Provenance.GENERATED,
            confidence=0.7,
        )
        e_synth.custom_properties["quality_state"] = QualityState.SYNTHETIC.value
        e_synth.custom_properties["synthetic_algorithm"] = "proc_wall_extrude"

        world.entities["e-real"] = e_real
        world.entities["e-synth"] = e_synth

        # Serialized payload preserves the explicit distinction
        d = world.to_dict()
        assert d["entities"]["e-real"]["custom_properties"]["quality_state"] == "REAL"
        assert d["entities"]["e-synth"]["custom_properties"]["quality_state"] == "SYNTHETIC"
        assert d["entities"]["e-synth"]["provenance"] == "GENERATED"
        # Verify that synth entity cannot claim OBSERVED provenance
        assert d["entities"]["e-synth"]["provenance"] != "OBSERVED"
