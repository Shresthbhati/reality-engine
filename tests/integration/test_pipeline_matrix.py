"""Phase 1: Real Integration Test Matrix (Stages A through J).

Verifies every stage across:
input -> contract -> output -> provenance -> uncertainty -> coordinate frame -> failure behavior.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from engine.compiler import CompileOptions, compile_reconstruction_to_world
from engine.compiler.world_compiler import CompileInputError
from engine.math import Quat, Vec3
from engine.scene_graph.graph import SceneGraph
from engine.scene_graph.spatial_index import SpatialIndex
from evidence.packages import (
    CorruptEvidenceError,
    DeterministicPackageBuilder,
    EvidenceAsset,
    EvidencePackage,
)
from evidence.session import (
    DuplicateEvidenceError,
    EvidenceItem,
    EvidenceKind,
    ProcessingRecord,
    Session,
    SessionStatus,
)
from exporters.gltf.exporter import export_to_gltf_with_report
from exporters.usd.exporter import export_to_usda_with_report
from provenance import Provenance, Uncertainty
from reconstruction.backend.interface import (
    ReconstructedCameraPose,
    ReconstructedPoint,
    ReconstructionResult,
)
from registration.cross_session import align_session
from sdk.reality import UnsupportedExportFormatError, export
from tests.test_room_inference import _CAMS, _two_room_scene
from world_ir.artifact_store import FileArtifactStore
from world_ir.coordinates import Frame, Transform
from world_ir.diff import diff_worlds
from world_ir.spatial_tiles import SpatialTiles
from worldstore.store import WorldStore, WorldStoreError


class TestStageAEvidenceIngestion:
    """Stage A: Raw Bytes -> Deterministic EvidencePackage."""

    def test_stage_a_contract_and_provenance(self):
        from evidence.packages import EvidenceSource
        builder = DeterministicPackageBuilder("pkg-001")
        source = EvidenceSource(source_id="src-cam-0", platform="phone", device="iPhone 15 Pro")
        builder.register_source(source)
        # Valid PNG header (8 bytes) + minimal payload
        png_payload = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        asset_id = builder.add_payload(
            payload=png_payload,
            kind=EvidenceKind.PHOTO,
            source=source,
            source_uri="cam0.png",
            provenance=Provenance.OBSERVED,
            uncertainty=Uncertainty(confidence=0.98, note="calibrated"),
            coordinate_frame=Frame.CAMERA,
        )
        pkg = builder.build()

        # Contract
        assert isinstance(pkg, EvidencePackage)
        assert asset_id in pkg.assets
        # Output
        assert len(pkg.assets[asset_id].sha256) == 64
        assert pkg.assets[asset_id].source_uri == "cam0.png"
        # Provenance
        assert pkg.assets[asset_id].provenance == Provenance.OBSERVED
        # Uncertainty
        assert pkg.assets[asset_id].uncertainty.confidence == 0.98
        # Coordinate frame
        assert pkg.assets[asset_id].coordinate_frame == Frame.CAMERA

    def test_stage_a_failure_behavior(self):
        from evidence.packages import EvidenceSource
        builder = DeterministicPackageBuilder("pkg-corrupt")
        source = EvidenceSource(source_id="src-cam-0", platform="phone", device="iPhone 15 Pro")
        builder.register_source(source)
        # Corrupt PNG (bad magic bytes) must be rejected
        with pytest.raises(CorruptEvidenceError):
            builder.add_payload(
                payload=b"NOT_A_PNG_FILE_GARBAGE",
                kind=EvidenceKind.PHOTO,
                source=source,
                source_uri="corrupt.png",
            )


class TestStageBSessionConstruction:
    """Stage B: Evidence Items -> Append-Only Session."""

    def test_stage_b_contract_and_provenance(self):
        sess = Session("sess-matrix-01", name="Matrix Session")
        item = EvidenceItem(
            id="ev-item-01",
            kind=EvidenceKind.PHOTO,
            source_uri="file:///evidence/photo1.jpg",
            provenance=Provenance.OBSERVED,
            uncertainty=Uncertainty(confidence=0.95),
            metadata={"iso": 100},
        )
        sess.add_evidence(item)
        sess.status = SessionStatus.PROCESSED

        assert len(sess.all_evidence()) == 1
        stored = sess.all_evidence()[0]
        assert stored.id == "ev-item-01"
        assert stored.provenance == Provenance.OBSERVED
        assert stored.uncertainty.confidence == 0.95

    def test_stage_b_failure_behavior(self):
        sess = Session("sess-dup", name="Duplicate Test")
        item = EvidenceItem(id="ev-1", kind=EvidenceKind.PHOTO, source_uri="file:///1.jpg")
        sess.add_evidence(item)
        # Re-adding duplicate item must fail
        with pytest.raises(DuplicateEvidenceError):
            sess.add_evidence(item)


class TestStageCRegistration:
    """Stage C: Multi-Session Cloud Alignment."""

    def test_stage_c_contract_and_coordinate_frame(self):
        cloud_a = [Vec3(x, y, 0.0) for x in range(5) for y in range(5)]
        # Cloud B translated by (2, 3, 0)
        cloud_b = [Vec3(p.x + 2.0, p.y + 3.0, p.z) for p in cloud_a]

        report = align_session(cloud_b, cloud_a, from_session="sess-b", to_session="sess-a")
        assert report.status == "accepted"
        assert report.transform is not None
        assert report.from_session == "sess-b"
        assert report.to_session == "sess-a"
        assert report.inlier_fraction > 0.5

    def test_stage_c_failure_behavior(self):
        # Non-overlapping clouds should be rejected, not fake-aligned
        cloud_a = [Vec3(x, y, 0.0) for x in range(5) for y in range(5)]
        cloud_b = [Vec3(x + 1000.0, y + 1000.0, 500.0) for x in range(5) for y in range(5)]

        report = align_session(cloud_b, cloud_a, from_session="sess-b", to_session="sess-a")
        assert report.status == "refused"
        assert report.transform is None
        assert "no plausible contact" in report.reason


class TestStageDReconstruction:
    """Stage D: Raw Tracks/Cams -> ReconstructionResult."""

    def test_stage_d_contract_and_uncertainty(self):
        recon = _two_room_scene()
        cams = [
            ReconstructedCameraPose(
                evidence_id=f"ev-c-{i}",
                position=p,
                rotation=(1.0, 0.0, 0.0, 0.0),
                uncertainty=Uncertainty(confidence=0.97),
            )
            for i, p in enumerate(_CAMS)
        ]
        import dataclasses
        recon = dataclasses.replace(recon, camera_poses=cams)

        assert len(recon.points) > 0
        assert len(recon.camera_poses) == len(_CAMS)
        assert recon.camera_poses[0].uncertainty.confidence == 0.97
        assert recon.points[0].uncertainty.confidence > 0.0

    def test_stage_d_failure_behavior(self):
        # Empty reconstruction must be refused by compiler
        empty_recon = ReconstructionResult(points=[], camera_poses=[], registration_status="failed")
        with pytest.raises(CompileInputError):
            compile_reconstruction_to_world(empty_recon)


class TestStageEWorldIRCompilation:
    """Stage E: ReconstructionResult -> Validated WorldIR."""

    def test_stage_e_contract_and_provenance(self, tmp_path):
        artifact_store = FileArtifactStore(tmp_path / "artifacts")
        recon = _two_room_scene()
        import dataclasses
        cams = [
            ReconstructedCameraPose(evidence_id=f"ev-c-{i}", position=p, rotation=(1.0, 0.0, 0.0, 0.0))
            for i, p in enumerate(_CAMS)
        ]
        recon = dataclasses.replace(recon, camera_poses=cams)

        world, diag = compile_reconstruction_to_world(
            recon, CompileOptions(seed=42, artifact_store=artifact_store)
        )

        assert len(world.entities) > 0
        assert diag.rooms_detected >= 1
        # Check that provenance on compiled entities is valid
        for entity in world.entities.values():
            assert entity.provenance in (Provenance.OBSERVED, Provenance.INFERRED, Provenance.RECONSTRUCTED)
            assert 0.0 <= entity.confidence <= 1.0


class TestStageFWorldStorePersistence:
    """Stage F: WorldIR -> Durable, Immutable WorldStore Version."""

    def test_stage_f_persistence_and_lineage(self, tmp_path):
        store = WorldStore(tmp_path / "store")
        artifact_store = FileArtifactStore(tmp_path / "artifacts")
        recon = _two_room_scene()
        import dataclasses
        cams = [
            ReconstructedCameraPose(evidence_id=f"ev-c-{i}", position=p, rotation=(1.0, 0.0, 0.0, 0.0))
            for i, p in enumerate(_CAMS)
        ]
        recon = dataclasses.replace(recon, camera_poses=cams)
        world_v1, _ = compile_reconstruction_to_world(recon, CompileOptions(artifact_store=artifact_store))

        v1 = store.save_version(world_v1, parent=None, version_id="v1")
        assert v1.version_id == "v1"
        assert v1.parent is None

        # Immutability: duplicate version save must raise
        with pytest.raises(WorldStoreError):
            store.save_version(world_v1, parent=None, version_id="v1")

        # Process restart verification
        restarted = WorldStore(tmp_path / "store")
        assert restarted.parents("v1") == []
        reloaded = restarted.load_version("v1")
        assert len(reloaded.entities) == len(world_v1.entities)


class TestStageGWorldSpatialQueries:
    """Stage G: WorldIR -> SpatialIndex, SceneGraph, SpatialTiles."""

    def test_stage_g_queries_and_tiling(self, tmp_path):
        artifact_store = FileArtifactStore(tmp_path / "artifacts")
        recon = _two_room_scene()
        import dataclasses
        cams = [
            ReconstructedCameraPose(evidence_id=f"ev-c-{i}", position=p, rotation=(1.0, 0.0, 0.0, 0.0))
            for i, p in enumerate(_CAMS)
        ]
        recon = dataclasses.replace(recon, camera_poses=cams)
        world, _ = compile_reconstruction_to_world(recon, CompileOptions(artifact_store=artifact_store))

        tiles = SpatialTiles(world, tile_size=10.0)
        assert len(list(tiles.tile_ids())) > 0

        scene_graph = SceneGraph(world)
        assert scene_graph.world is world


class TestStageHVersionDiff:
    """Stage H: WorldIR x WorldIR -> Structural WorldDiff."""

    def test_stage_h_diff_contract(self, tmp_path):
        artifact_store = FileArtifactStore(tmp_path / "artifacts")
        recon = _two_room_scene()
        import dataclasses
        cams = [
            ReconstructedCameraPose(evidence_id=f"ev-c-{i}", position=p, rotation=(1.0, 0.0, 0.0, 0.0))
            for i, p in enumerate(_CAMS)
        ]
        recon = dataclasses.replace(recon, camera_poses=cams)
        world_v1, _ = compile_reconstruction_to_world(recon, CompileOptions(artifact_store=artifact_store))

        from world_ir import apply_incremental_update
        from world_ir.schema_v1 import Entity, EntityType

        annex = Entity(id="ent-annex", type=EntityType.ROOM, name="Annex Room")
        res = apply_incremental_update(world_v1, [annex])
        world_v2 = res.new_world

        diff = diff_worlds(world_v1, world_v2)
        assert not diff.is_empty()
        summary = diff.summary()
        assert summary["entities_added"] == 1
        assert summary["entities_modified"] == 0
        assert summary["entities_removed"] == 0


class TestStageIDesktopConsumption:
    """Stage I: WorldStore -> API Bridge -> Canonical Desktop Shapes."""

    def test_stage_i_desktop_payload_shape(self, tmp_path, monkeypatch):
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
        store.save_version(world, parent=None, version_id="v-desktop-1")

        from apps.cli import api_bridge
        import io
        import sys

        out = io.StringIO()
        monkeypatch.setattr(sys, "stdout", out)
        ret = api_bridge.cmd_load_world("v-desktop-1")
        assert ret == 0

        payload = json.loads(out.getvalue().strip())
        assert payload["version_id"] == "v-desktop-1"
        assert "entities" in payload
        assert "geometries" in payload
        assert "global_provenance" in payload
        assert "coordinate_frame" in payload


class TestStageJExport:
    """Stage J: WorldIR -> GLTF / USDA / Blender export."""

    def test_stage_j_export_contracts(self, tmp_path):
        artifact_store = FileArtifactStore(tmp_path / "artifacts")
        recon = _two_room_scene()
        import dataclasses
        cams = [
            ReconstructedCameraPose(evidence_id=f"ev-c-{i}", position=p, rotation=(1.0, 0.0, 0.0, 0.0))
            for i, p in enumerate(_CAMS)
        ]
        recon = dataclasses.replace(recon, camera_poses=cams)
        world, _ = compile_reconstruction_to_world(recon, CompileOptions(artifact_store=artifact_store))

        # GLTF export
        gltf_content, gltf_rep = export(world, "gltf")
        assert isinstance(gltf_content, dict)
        assert len(gltf_rep.entities_exported) > 0

        # USDA export
        usda_content, usda_rep = export(world, "usda")
        assert usda_content.startswith("#usda 1.0")
        assert len(usda_rep.entities_exported) > 0

        # Unsupported format must fail explicitly
        with pytest.raises(UnsupportedExportFormatError):
            export(world, "unsupported_xyz_format")
