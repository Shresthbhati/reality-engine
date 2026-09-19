"""Phase 7: Adversarial Failure Injection Suite.

Tests all 12 system failure modes:
1. Missing artifact
2. Corrupt artifact
3. Missing / empty reconstruction
4. Invalid WorldIR
5. Invalid provenance
6. Unavailable backend
7. Registration failure
8. Partial / degraded reconstruction
9. Failed tile parameter / write
10. Interrupted WorldStore update / immutability violation
11. Malformed EvidencePackage
12. Missing optional dependency / unsupported export format

Enforces:
EXPLICIT FAILURE + DIAGNOSTICS + NO CORRUPTED WORLD + NO FAKE SUCCESS.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
import pytest

from apps.cli import api_bridge
from engine.compiler import CompileOptions, compile_reconstruction_to_world
from engine.compiler.world_compiler import CompileInputError, WorldValidationGateError
from engine.math import Vec3
from evidence.packages import (
    CorruptEvidenceError,
    DeterministicPackageBuilder,
    EvidenceKind,
    EvidenceSource,
)
from evidence.session import EvidenceItem
from provenance import Provenance
from reconstruction.backend.interface import (
    ReconstructedCameraPose,
    ReconstructedPoint,
    ReconstructionResult,
)
from reconstruction.robustness import classify_evidence_items, REJECTED
from registration.cross_session import align_session
from sdk.reality import UnsupportedExportFormatError, export
from tests.test_room_inference import _CAMS, _two_room_scene
from world_ir.artifact_store import ArtifactNotFoundError, FileArtifactStore
from world_ir.schema_v1 import Entity, EntityType, Geometry, GeometryType
from world_ir.spatial_tiles import SpatialTiles
from world_ir.validation import validate_world_ir
from world_ir.world_v1 import WorldIR
from worldstore.store import WorldStore, WorldStoreError


class TestFailureInjectionSuite:
    """Verifies that all 12 failure modes fail explicitly with diagnostics and without data corruption."""

    # 1. Missing artifact
    def test_failure_mode_1_missing_artifact(self, tmp_path):
        store_root = tmp_path / "store"
        store = WorldStore(store_root)

        world = WorldIR(id="world-missing-artifact")
        world.entities["e1"] = Entity(id="e1", type=EntityType.STRUCTURE, provenance=Provenance.OBSERVED)
        ver = store.save_version(world, parent=None, version_id="v-missing-artifact")

        # Delete artifact file on disk
        artifact_file = store._store._path_for(ver.artifact_hash)
        assert artifact_file.exists()
        artifact_file.unlink()

        errors = store.verify_version("v-missing-artifact")
        assert len(errors) > 0
        assert any("missing" in err["reason"].lower() for err in errors)

        with pytest.raises(Exception):
            store.load_version("v-missing-artifact")

    # 2. Corrupt artifact
    def test_failure_mode_2_corrupt_artifact(self, tmp_path):
        store_root = tmp_path / "store"
        store = WorldStore(store_root)

        world = WorldIR(id="world-corrupt-artifact")
        world.entities["e1"] = Entity(id="e1", type=EntityType.STRUCTURE, provenance=Provenance.OBSERVED)
        ver = store.save_version(world, parent=None, version_id="v-corrupt-artifact")

        # Tamper with an artifact on disk
        artifact_file = store._store._path_for(ver.artifact_hash)
        assert artifact_file.exists()
        artifact_file.write_bytes(b"CORRUPTED_PAYLOAD_TAMPERED")

        errors = store.verify_version("v-corrupt-artifact")
        assert len(errors) > 0
        assert any("hash mismatch" in err["reason"].lower() for err in errors)

        with pytest.raises(WorldStoreError, match="artifact hash mismatch"):
            store.load_version("v-corrupt-artifact")

    # 3. Missing / empty reconstruction
    def test_failure_mode_3_missing_reconstruction(self):
        empty_recon = ReconstructionResult(points=[], camera_poses=[], registration_status="failed")
        with pytest.raises(CompileInputError, match="refusing to compile a failed reconstruction"):
            compile_reconstruction_to_world(empty_recon)

    # 4. Invalid WorldIR
    def test_failure_mode_4_invalid_worldir(self):
        world = WorldIR(id="world-invalid")
        world.entities["ghost-room"] = Entity(
            id="ghost-room",
            type=EntityType.ROOM,
            provenance=Provenance.INFERRED,
            confidence=0.5,
            geometry_ids=["geom-does-not-exist"],
        )
        report = validate_world_ir(world)
        assert not report.is_valid()
        assert any(i.code == "structural" for i in report.issues)

    # 5. Invalid provenance
    def test_failure_mode_5_invalid_provenance(self):
        with pytest.raises(ValueError):
            Provenance("INVALID_PROVENANCE_STRING")

    # 6. Unavailable backend
    def test_failure_mode_6_unavailable_backend(self, tmp_path, monkeypatch):
        empty_dir = tmp_path / "nonexistent_store"
        monkeypatch.setenv("REALITY_STORE_PATH", str(empty_dir))

        out = io.StringIO()
        monkeypatch.setattr(sys, "stdout", out)
        ret = api_bridge.cmd_load_world("v-does-not-exist")
        assert ret == 1
        payload = json.loads(out.getvalue().strip())
        assert "error" in payload

    # 7. Registration failure
    def test_failure_mode_7_registration_failure(self):
        cloud_a = [Vec3(x, y, 0.0) for x in range(5) for y in range(5)]
        cloud_b = [Vec3(x + 5000.0, y + 5000.0, 1000.0) for x in range(5) for y in range(5)]
        report = align_session(cloud_b, cloud_a, from_session="sess-b", to_session="sess-a")
        assert report.status == "refused"
        assert report.transform is None
        assert len(report.reason) > 0

    # 8. Partial / degraded reconstruction
    def test_failure_mode_8_partial_reconstruction(self):
        item_blurry = EvidenceItem(
            id="ev-blurry",
            kind=EvidenceKind.PHOTO,
            source_uri="file:///blurry.jpg",
            metadata={"quality": {"laplacian_variance": 15.0, "clipped_fraction": 0.01, "measured": 1.0}},
        )
        admissions = classify_evidence_items([item_blurry])
        assert admissions.counts[REJECTED] == 1
        assert "ev-blurry" in admissions.rejected_ids

    # 9. Failed tile parameter
    def test_failure_mode_9_failed_tile_parameter(self):
        world = WorldIR(id="world-tile")
        with pytest.raises(ValueError, match="tile_size must be positive"):
            SpatialTiles(world, tile_size=-5.0)

    # 10. Interrupted WorldStore update / immutability violation
    def test_failure_mode_10_worldstore_immutability(self, tmp_path):
        store = WorldStore(tmp_path / "store")
        world = WorldIR(id="world-immutable")
        store.save_version(world, parent=None, version_id="v-fixed-1")

        with pytest.raises(WorldStoreError, match="already exists -- versions are immutable"):
            store.save_version(world, parent=None, version_id="v-fixed-1")

    # 11. Malformed EvidencePackage
    def test_failure_mode_11_malformed_evidence_package(self):
        builder = DeterministicPackageBuilder("pkg-fail")
        source = EvidenceSource(source_id="src-0", platform="phone", device="test")
        builder.register_source(source)
        with pytest.raises(CorruptEvidenceError):
            builder.add_payload(
                payload=b"NOT_A_VALID_IMAGE_HEADER_12345",
                kind=EvidenceKind.PHOTO,
                source=source,
                source_uri="test.jpg",
            )

    # 12. Missing optional dependency / unsupported export format
    def test_failure_mode_12_unsupported_export_format(self):
        world = WorldIR(id="world-export")
        with pytest.raises(UnsupportedExportFormatError, match="unsupported export format"):
            export(world, "unsupported_cad_format_xyz")
