"""Phase 3: Cross-Agent Contract Compatibility Tests.

Catches contract drift across:
Cline EvidencePackage -> Session -> FreeBuff Reconstruction -> WorldIR Compiler -> Claude WorldStore -> Desktop.

Asserts:
- No missing fields
- Compatible enums across all subsystem boundaries
- Stable identifiers and hashes
- Provenance and Uncertainty preservation
- Coordinate frame compatibility
- Round-trip serialization fidelity
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
import pytest

from apps.cli import api_bridge
from engine.compiler import CompileOptions, compile_reconstruction_to_world
from evidence.packages import (
    DeterministicPackageBuilder,
    EvidenceAsset,
    EvidencePackage,
    EvidenceSource,
)
from evidence.session import (
    EvidenceItem,
    EvidenceKind,
    Session,
    SessionStatus,
)
from provenance import Provenance, Uncertainty
from reconstruction.backend.interface import (
    ReconstructedCameraPose,
    ReconstructedPoint,
    ReconstructionResult,
)
from reconstruction.robustness import classify_evidence_items, ACCEPTED
from tests.test_room_inference import _CAMS, _two_room_scene
from world_ir.artifact_store import FileArtifactStore
from world_ir.coordinates import Frame
from world_ir.schema_v1 import EntityType, GeometryType
from world_ir.world_v1 import WorldIR
from worldstore.store import WorldStore


class TestClineToSessionToFreebuff:
    """Verifies Cline EvidencePackage -> Session -> FreeBuff contract flow."""

    def test_evidence_package_to_session_contract(self):
        builder = DeterministicPackageBuilder("mobile-run-01")
        source = EvidenceSource(source_id="mobile-phone-01", platform="phone", device="Pixel 8 Pro")
        builder.register_source(source)

        jpeg_payload = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00" + b"\x00" * 32
        asset_id = builder.add_payload(
            payload=jpeg_payload,
            kind=EvidenceKind.PHOTO,
            source=source,
            source_uri="photo_001.jpg",
            acquired_at=1700000000.0,
            coordinate_frame=Frame.CAMERA,
            sensor_metadata={
                "focal_length": 24.0,
                "exposure_time": 0.01,
                "quality": {"laplacian_variance": 250.0, "clipped_fraction": 0.05, "measured": 1.0},
            },
            quality={"laplacian_variance": 250.0, "clipped_fraction": 0.05, "measured": 1.0},
            provenance=Provenance.OBSERVED,
            uncertainty=Uncertainty(confidence=0.99, note="high quality mobile capture"),
        )
        pkg = builder.build()
        asset = pkg.assets[asset_id]

        # 1. Asset -> EvidenceItem conversion
        item = asset.to_evidence_item()
        assert isinstance(item, EvidenceItem)
        assert item.id == asset.id
        assert item.kind == asset.kind
        assert item.source_uri == asset.source_uri
        assert item.captured_at == asset.acquired_at
        assert item.sha256 == asset.sha256
        assert item.provenance == asset.provenance
        assert item.uncertainty.confidence == asset.uncertainty.confidence
        assert item.metadata["focal_length"] == 24.0

        # 2. Session ingestion
        session = Session("sess-mobile-01", name="Mobile Ingest", coordinate_frame=Frame.SESSION_LOCAL)
        session.add_evidence(item)
        assert len(session.all_evidence()) == 1

        # 3. Session -> FreeBuff robustness admission
        admissions = classify_evidence_items(session.all_evidence())
        assert item.id in admissions.accepted_ids
        assert admissions.counts[ACCEPTED] == 1

    def test_session_serialization_fidelity(self):
        session = Session("sess-roundtrip", name="Roundtrip Session", coordinate_frame=Frame.SESSION_LOCAL)
        item = EvidenceItem(
            id="ev-rt-01",
            kind=EvidenceKind.PHOTO,
            source_uri="file:///photos/1.jpg",
            captured_at=1700000100.0,
            sha256="abc123" * 10 + "abcd",
            provenance=Provenance.OBSERVED,
            uncertainty=Uncertainty(confidence=0.95),
            metadata={"width": 1920, "height": 1080},
        )
        session.add_evidence(item)
        data = session.to_dict()

        # Deserialization check
        new_session = Session.from_dict(data)
        assert new_session.id == "sess-roundtrip"
        assert len(new_session.all_evidence()) == 1
        rt_item = new_session.get_evidence("ev-rt-01")
        assert rt_item.id == item.id
        assert rt_item.kind == item.kind
        assert rt_item.sha256 == item.sha256
        assert rt_item.provenance == item.provenance
        assert rt_item.uncertainty.confidence == item.uncertainty.confidence


class TestFreebuffToWorldIRToWorldStoreToDesktop:
    """Verifies FreeBuff Reconstruction -> WorldIR -> WorldStore -> Desktop contract flow."""

    def test_reconstruction_to_worldir_compiler_contract(self, tmp_path):
        artifact_store = FileArtifactStore(tmp_path / "artifacts")
        recon = _two_room_scene()
        import dataclasses
        cams = [
            ReconstructedCameraPose(
                evidence_id=f"ev-c-{i}",
                position=p,
                rotation=(1.0, 0.0, 0.0, 0.0),
                uncertainty=Uncertainty(confidence=0.95),
            )
            for i, p in enumerate(_CAMS)
        ]
        recon = dataclasses.replace(recon, camera_poses=cams)

        world, diag = compile_reconstruction_to_world(recon, CompileOptions(artifact_store=artifact_store))

        # Contract checks:
        assert isinstance(world, WorldIR)
        assert hasattr(world, "entities")
        assert hasattr(world, "geometries")
        assert hasattr(world, "global_provenance")
        assert hasattr(world, "coordinate_frame")

        # Every entity must have an EntityType enum and valid Provenance
        for e in world.entities.values():
            assert isinstance(e.type, EntityType)
            assert isinstance(e.provenance, Provenance)
            assert isinstance(e.confidence, float)
            assert 0.0 <= e.confidence <= 1.0

        # Every geometry must have a GeometryType enum
        for g in world.geometries.values():
            assert isinstance(g.type, GeometryType)
            assert isinstance(g.provenance, Provenance)

    def test_worldir_to_worldstore_contract(self, tmp_path):
        store_root = tmp_path / "store"
        store = WorldStore(store_root)
        artifact_store = FileArtifactStore(tmp_path / "artifacts")

        recon = _two_room_scene()
        import dataclasses
        cams = [
            ReconstructedCameraPose(evidence_id=f"ev-c-{i}", position=p, rotation=(1.0, 0.0, 0.0, 0.0))
            for i, p in enumerate(_CAMS)
        ]
        recon = dataclasses.replace(recon, camera_poses=cams)
        world, _ = compile_reconstruction_to_world(recon, CompileOptions(artifact_store=artifact_store))

        v1 = store.save_version(world, parent=None, version_id="v-contract-1", source_session_ids=["sess-1"])
        assert v1.version_id == "v-contract-1"
        assert v1.world_id == world.id

        # Reload from store and verify schema consistency
        reloaded = store.load_version("v-contract-1")
        assert reloaded.id == world.id
        assert len(reloaded.entities) == len(world.entities)
        for eid, orig_e in world.entities.items():
            loaded_e = reloaded.entities[eid]
            assert loaded_e.type == orig_e.type
            assert loaded_e.name == orig_e.name
            assert loaded_e.confidence == orig_e.confidence
            assert loaded_e.provenance == orig_e.provenance

    def test_worldstore_to_desktop_bridge_contract(self, tmp_path, monkeypatch):
        store_root = tmp_path / "store"
        monkeypatch.setenv("REALITY_STORE_PATH", str(store_root))
        store = WorldStore(store_root)
        artifact_store = FileArtifactStore(tmp_path / "artifacts")

        recon = _two_room_scene()
        import dataclasses
        cams = [
            ReconstructedCameraPose(evidence_id=f"ev-c-{i}", position=p, rotation=(1.0, 0.0, 0.0, 0.0))
            for i, p in enumerate(_CAMS)
        ]
        recon = dataclasses.replace(recon, camera_poses=cams)
        world, _ = compile_reconstruction_to_world(recon, CompileOptions(artifact_store=artifact_store))
        store.save_version(world, parent=None, version_id="v-desktop-contract")

        out = io.StringIO()
        monkeypatch.setattr(sys, "stdout", out)
        ret = api_bridge.cmd_load_world("v-desktop-contract")
        assert ret == 0

        payload = json.loads(out.getvalue().strip())
        # Contract required by Desktop Frontend:
        # 1. Envelope fields
        assert "version_id" in payload
        assert "world_id" in payload
        assert "global_provenance" in payload
        assert "global_confidence" in payload
        assert "coordinate_frame" in payload
        assert "entities" in payload
        assert "geometries" in payload

        # 2. Canonical Entity fields in payload
        for entity in payload["entities"]:
            assert "id" in entity
            assert "type" in entity
            assert "provenance" in entity
            assert "confidence" in entity
            assert "geometry_ids" in entity
            assert "relationships" in entity
            assert "observations" in entity

        # 3. Canonical Geometry fields in payload
        for gid, geom in payload["geometries"].items():
            assert "id" in geom
            assert "type" in geom
            assert "lod_level" in geom
            assert "provenance" in geom
            assert "confidence" in geom
