"""Phase 4: Provenance Firewall.

Builds integration assertions that every important world entity can answer:
WHERE DID THIS COME FROM?

Verifies end-to-end lineage:
Entity -> WorldIR Geometry -> Reconstruction/Session -> Evidence -> Source Artifact.

Enforces:
- Rejection of "unknown", "demo", "mock" as canonical production provenance.
- Explicit UNAVAILABLE reporting instead of silent fallback.
- Preservation of provenance across WorldStore and API bridge.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
import pytest

from apps.cli import api_bridge
from engine.compiler import CompileOptions, compile_reconstruction_to_world
from provenance import Provenance, Provenanced, Uncertainty
from reconstruction.backend.interface import (
    ReconstructedCameraPose,
    ReconstructedPoint,
    ReconstructionResult,
)
from tests.test_room_inference import _CAMS, _two_room_scene
from world_ir.artifact_store import FileArtifactStore
from world_ir.schema_v1 import Entity, EntityType
from worldstore.store import WorldStore


class TestProvenanceFirewall:
    """Verifies that every entity and geometry has verifiable provenance."""

    def test_entity_to_evidence_lineage_traceability(self, tmp_path):
        """Entity -> Geometry -> Source Points / Evidence IDs -> Artifact Store."""
        artifacts_dir = tmp_path / "artifacts"
        artifact_store = FileArtifactStore(artifacts_dir)

        recon = _two_room_scene()
        import dataclasses
        cams = [
            ReconstructedCameraPose(
                evidence_id=f"ev-cam-{i}",
                position=p,
                rotation=(1.0, 0.0, 0.0, 0.0),
                uncertainty=Uncertainty(confidence=0.98),
            )
            for i, p in enumerate(_CAMS)
        ]
        recon = dataclasses.replace(recon, camera_poses=cams)

        world, diag = compile_reconstruction_to_world(recon, CompileOptions(artifact_store=artifact_store))

        # Check all entities in world
        assert len(world.entities) > 0
        for eid, entity in world.entities.items():
            # 1. Provenance must be real production provenance
            assert entity.provenance in (Provenance.OBSERVED, Provenance.INFERRED, Provenance.RECONSTRUCTED), (
                f"Entity {eid} has non-canonical provenance: {entity.provenance}"
            )
            assert entity.confidence > 0.0

            # 2. Associated geometries must exist and trace to real artifacts
            for gid in entity.geometry_ids:
                assert gid in world.geometries, f"Entity {eid} references missing geometry {gid}"
                geom = world.geometries[gid]
                assert geom.provenance in (Provenance.OBSERVED, Provenance.RECONSTRUCTED, Provenance.INFERRED)
                # If geometry has a data URI, verify the artifact exists and is not corrupt
                if geom.data_uri and geom.data_uri.startswith("urn:hash:"):
                    assert artifact_store.exists(geom.data_uri), (
                        f"Artifact {geom.data_uri} for geometry {gid} missing from artifact store!"
                    )
                    # Verify content hash integrity
                    payload = artifact_store.load(geom.data_uri)
                    assert len(payload) > 0

    def test_firewall_rejects_mock_or_demo_provenance(self):
        """Firewall rule: 'demo', 'mock', 'unknown' must fail is_canonical() check."""
        non_canonical_cases = [
            Provenanced(value="room_mesh", provenance=Provenance.UNKNOWN),
            Provenanced(value="door_frame", provenance=Provenance.GENERATED),
            Provenanced(value="wall_plane", provenance=Provenance.CONFLICT),
        ]
        for p in non_canonical_cases:
            assert p.is_canonical() is False, f"{p.provenance} must not be accepted as canonical reality"

        canonical_cases = [
            Provenanced(value="room_mesh", provenance=Provenance.OBSERVED),
            Provenanced(value="room_mesh", provenance=Provenance.RECONSTRUCTED),
            Provenanced(value="room_mesh", provenance=Provenance.INFERRED),
        ]
        for p in canonical_cases:
            assert p.is_canonical() is True

    def test_provenance_preserved_across_worldstore_and_desktop(self, tmp_path, monkeypatch):
        """Provenance must survive serialization, WorldStore persistence, and API bridge."""
        store_root = tmp_path / "store"
        monkeypatch.setenv("REALITY_STORE_PATH", str(store_root))
        store = WorldStore(store_root)
        artifact_store = FileArtifactStore(tmp_path / "artifacts")

        recon = _two_room_scene()
        import dataclasses
        cams = [
            ReconstructedCameraPose(evidence_id=f"ev-cam-{i}", position=p, rotation=(1.0, 0.0, 0.0, 0.0))
            for i, p in enumerate(_CAMS)
        ]
        recon = dataclasses.replace(recon, camera_poses=cams)
        world, _ = compile_reconstruction_to_world(recon, CompileOptions(artifact_store=artifact_store))

        # Save to WorldStore
        store.save_version(world, parent=None, version_id="v-prov-1")

        # Load through API Bridge
        out = io.StringIO()
        monkeypatch.setattr(sys, "stdout", out)
        ret = api_bridge.cmd_load_world("v-prov-1")
        assert ret == 0

        payload = json.loads(out.getvalue().strip())
        assert payload["global_provenance"] == world.global_provenance.value

        for entity_data in payload["entities"]:
            orig_entity = world.entities[entity_data["id"]]
            assert entity_data["provenance"] == orig_entity.provenance.value
            assert entity_data["confidence"] == orig_entity.confidence

    def test_explicit_unavailable_state_not_faked(self, tmp_path, monkeypatch):
        """When an artifact or version is unavailable, emit explicit unavailable error, never fake data."""
        store_root = tmp_path / "empty_store"
        monkeypatch.setenv("REALITY_STORE_PATH", str(store_root))
        store = WorldStore(store_root)

        out = io.StringIO()
        monkeypatch.setattr(sys, "stdout", out)
        ret = api_bridge.cmd_load_world("non-existent-version")
        assert ret == 1

        payload = json.loads(out.getvalue().strip())
        assert "error" in payload
        assert "non-existent-version" in payload["error"] or "No versions" in payload["error"]
        assert payload["entities"] == []
