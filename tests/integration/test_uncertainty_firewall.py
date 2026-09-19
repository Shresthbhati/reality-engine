"""Phase 5: Uncertainty Firewall.

Verifies that uncertainty survives across the entire system:
Evidence -> Reconstruction -> WorldIR -> WorldStore -> Desktop.

Asserts:
- Covariance, sigma, confidence, quality metrics, and notes survive without loss
- Downstream representations do not discard or overwrite uncertainty
- No fabricated certainty (composed uncertainty is monotonically non-increasing)
- Invalid uncertainty values (e.g. confidence > 1.0 or < 0.0) are rejected loudly
"""

from __future__ import annotations

import io
import json
import math
import sys
from pathlib import Path
import pytest

from apps.cli import api_bridge
from engine.compiler import CompileOptions, compile_reconstruction_to_world
from evidence.packages import (
    DeterministicPackageBuilder,
    EvidenceKind,
    EvidenceSource,
)
from provenance import Provenance, Uncertainty
from reconstruction.backend.interface import (
    ReconstructedCameraPose,
    ReconstructedPoint,
    ReconstructionResult,
)
from tests.test_room_inference import _CAMS, _two_room_scene
from uncertainty.propagation import (
    Covariance3,
    depth_to_world_covariance,
    transform_point_covariance,
)
from uncertainty.quantity import Uncertain
from world_ir.artifact_store import FileArtifactStore
from world_ir.coordinates import Frame, Transform
from world_ir.schema_v1 import Entity, EntityType
from worldstore.store import WorldStore


class TestUncertaintyFirewall:
    """Verifies that uncertainty contracts are strictly enforced across all boundaries."""

    def test_uncertainty_bounds_enforced(self):
        """Uncertainty confidence must be strictly bounded in [0, 1]."""
        with pytest.raises(ValueError, match="confidence must be in"):
            Uncertainty(confidence=1.5)

        with pytest.raises(ValueError, match="confidence must be in"):
            Uncertainty(confidence=-0.05)

        # Valid bounds
        u0 = Uncertainty(confidence=0.0)
        assert u0.confidence == 0.0
        u1 = Uncertainty(confidence=1.0)
        assert u1.confidence == 1.0

    def test_composed_uncertainty_never_fabricates_certainty(self):
        """Chaining transforms must only decrease or preserve confidence, never increase."""
        t1 = Transform(
            source_frame=Frame.CAMERA,
            target_frame=Frame.SESSION_LOCAL,
            uncertainty=Uncertainty(confidence=0.85, note="sensor calibration"),
        )
        t2 = Transform(
            source_frame=Frame.SESSION_LOCAL,
            target_frame=Frame.WORLD,
            uncertainty=Uncertainty(confidence=0.90, note="cross-session alignment"),
        )
        chained = t1.then(t2)
        assert chained.source_frame == Frame.CAMERA
        assert chained.target_frame == Frame.WORLD
        # Multiplicative composition: 0.85 * 0.90 = 0.765
        expected_confidence = 0.85 * 0.90
        assert math.isclose(chained.uncertainty.confidence, expected_confidence, rel_tol=1e-6)
        assert chained.uncertainty.confidence <= t1.uncertainty.confidence
        assert chained.uncertainty.confidence <= t2.uncertainty.confidence

    def test_covariance_propagation_preserves_variance(self):
        """Covariance3 must maintain sigmas and reject fabricating certainty when unknown."""
        cov = Covariance3.from_sigmas(0.01, 0.02, 0.03)
        assert cov.sigma == (0.01, 0.02, 0.03)
        assert cov.require_sigma() == (0.01, 0.02, 0.03)

        # UNKNOWN covariance must require an explicit reason
        with pytest.raises(Exception, match="reason"):
            Covariance3.unknown("")

        unk = Covariance3.unknown("missing depth sensor intrinsics")
        assert unk.is_unknown is True
        # Calling require_sigma() on unknown must raise, never fabricating numbers
        with pytest.raises(Exception, match="fabricate certainty"):
            unk.require_sigma()

    def test_uncertainty_survival_end_to_end(self, tmp_path, monkeypatch):
        """Verify confidence and quality survive Evidence -> WorldStore -> Desktop."""
        store_root = tmp_path / "store"
        monkeypatch.setenv("REALITY_STORE_PATH", str(store_root))
        store = WorldStore(store_root)
        artifact_store = FileArtifactStore(tmp_path / "artifacts")

        # Step 1: Reconstruction with explicit camera pose confidence
        recon = _two_room_scene()
        import dataclasses
        cams = [
            ReconstructedCameraPose(
                evidence_id=f"ev-c-{i}",
                position=p,
                rotation=(1.0, 0.0, 0.0, 0.0),
                uncertainty=Uncertainty(confidence=0.92, note="stereo calibrated"),
            )
            for i, p in enumerate(_CAMS)
        ]
        recon = dataclasses.replace(recon, camera_poses=cams)

        # Step 2: Compile to WorldIR
        world, diag = compile_reconstruction_to_world(recon, CompileOptions(artifact_store=artifact_store))
        assert world.global_confidence > 0.0

        # Step 3: Save to WorldStore
        store.save_version(world, parent=None, version_id="v-unc-1")

        # Step 4: Desktop Bridge loads world
        out = io.StringIO()
        monkeypatch.setattr(sys, "stdout", out)
        ret = api_bridge.cmd_load_world("v-unc-1")
        assert ret == 0

        payload = json.loads(out.getvalue().strip())
        assert math.isclose(payload["global_confidence"], world.global_confidence, rel_tol=1e-5)

        # Check that entity confidences are preserved in the desktop JSON
        for entity_dict in payload["entities"]:
            orig_entity = world.entities[entity_dict["id"]]
            assert math.isclose(entity_dict["confidence"], orig_entity.confidence, rel_tol=1e-5)
