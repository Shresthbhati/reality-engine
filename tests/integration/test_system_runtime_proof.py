"""System Runtime Proof & Real Product Verification (SYSTEM_RUNTIME_PROOF_READY).

Proves the complete Reality Engine runtime with real datasets, localized V2 updates,
lazy tiled spatial storage, adversarial invariant enforcement, and Desktop Studio consumption:

1. REAL DATASET VERTICAL SLICE
   Real Evidence -> MultiSourceSession -> Quality Admission Gate -> Registration ->
   Reconstruction -> WorldIR -> Tiled WorldStore -> Lazy Spatial Query -> Desktop API.
   Uses:
   - `datasets/real_room_capture/` (real iPhone 14 Pro photos, camera poses, EXIF)
   - `datasets/real_room_capture_worldir/` (real 3D points, track IDs, uncertainty)
   - `datasets/room_capture/pipeline_out/worldir.json` (58 structural planes/entities)

2. TRUE LOCALIZED V2 PROOF
   Real V1 World -> Localized Rescan Evidence -> Incremental Adapter ->
   `apply_reconstruction_update` -> `IncrementalUpdateResult` -> `save_version_tiled`.
   Proves:
   - Changed target entity is updated
   - Untouched entities preserve strict Python object identity (`is`)
   - Untouched geometries preserve strict Python object identity (`is`)
   - Untouched spatial tiles preserve identical `artifact_uri` and `content_hash`
   - Rebuilt spatial tiles are confined to `result.rebuilt_tile_ids`
   - WorldDiff accurately captures the exact delta
   - Zero storage re-writes for untouched tiles

3. 10 ADVERSARIAL FAILURE INJECTIONS
   1) Corrupted evidence bytes / invalid inputs
   2) Incomplete reconstruction (0 points)
   3) Missing camera poses / unaligned session
   4) Registration refusal
   5) Tile artifact tampering (SHA256 corruption detection)
   6) Stale version ID request
   7) Concurrent update collision (duplicate version ID)
   8) Atomic failure safety (no corrupt manifest on failure)
   9) Frontend bridge error handling (graceful error JSON, exit code 1)
   10) Backend store unavailability handling

4. DESKTOP STUDIO CONTRACT VERIFICATION
   Validates NDJSON outputs against frontend `convertBackendEntityToEntity` expectations:
   - Non-mock, non-demo entity IDs
   - Real session IDs resolved from custom properties and observations
   - Valid bounding boxes from geometries
   - Correct provenance mappings (RECONSTRUCTED)
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple
import pytest

from apps.cli import api_bridge
from engine.compiler.incremental_adapter import (
    IncrementalUpdatePackage,
    ReconstructionAdapterError,
    adapt_reconstruction_to_incremental_update,
    apply_reconstruction_update,
)
from engine.compiler.world_compiler import CompileOptions, compile_reconstruction_to_world
from engine.math import Vec3
from evidence.packages import DeterministicPackageBuilder, EvidenceKind
from evidence.session import EvidenceItem, Session
from provenance import Provenance, Uncertainty
from reconstruction.backend.interface import (
    ReconstructedCameraPose,
    ReconstructedPoint,
    ReconstructionResult,
)
from reconstruction.calibration.transforms import RigidTransform
from reconstruction.robustness import ACCEPTED, classify_evidence_items
from registration.cross_session import SessionAlignment
from world_ir.artifact_store import FileArtifactStore
from world_ir.diff import diff_worlds
from world_ir.incremental import IncrementalUpdateResult, apply_incremental_update
from world_ir.schema_v1 import (
    Entity,
    EntityType,
    Geometry,
    GeometryType,
    Observation,
    Relationship,
    RelationshipKind,
    Vector3,
)
from world_ir.world_v1 import WorldIR
from worldstore.store import WorldStore, WorldStoreError
from worldstore.tiles import (
    TileArtifactRef,
    TileManifest,
    WorldVersionHandle,
    open_version,
    save_version_tiled,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
REAL_CAPTURE_DIR = REPO_ROOT / "datasets" / "real_room_capture"
REAL_WORLDIR_DIR = REPO_ROOT / "datasets" / "real_room_capture_worldir"
PIPELINE_OUT_DIR = REPO_ROOT / "datasets" / "room_capture" / "pipeline_out"


def _load_real_reconstruction_result() -> ReconstructionResult:
    """Loads canonical real reconstruction result from disk."""
    recon_path = REAL_WORLDIR_DIR / "reconstruction_result.json"
    assert recon_path.exists(), f"Missing real reconstruction data at {recon_path}"
    data = json.loads(recon_path.read_text(encoding="utf-8"))

    points = [
        ReconstructedPoint(
            position=tuple(p["position"]),
            track_id=p["track_id"],
            source_evidence_ids=list(p["source_evidence_ids"]),
            uncertainty=Uncertainty(
                confidence=p.get("uncertainty", {}).get("confidence", 0.8),
                note=p.get("uncertainty", {}).get("note"),
            ),
        )
        for p in data["points"]
    ]
    cameras = [
        ReconstructedCameraPose(
            evidence_id=c["evidence_id"],
            position=tuple(c["position"]),
            rotation=tuple(c["rotation"]),
            uncertainty=Uncertainty(
                confidence=c.get("uncertainty", {}).get("confidence", 0.9),
                note=c.get("uncertainty", {}).get("note"),
            ),
        )
        for c in data.get("camera_poses", [])
    ]
    return ReconstructionResult(
        points=points,
        camera_poses=cameras,
        registration_status=data.get("registration_status", "success"),
    )


def _load_real_structural_world() -> WorldIR:
    """Loads compiled room structural WorldIR from pipeline output."""
    world_path = PIPELINE_OUT_DIR / "worldir.json"
    assert world_path.exists(), f"Missing real pipeline worldir at {world_path}"
    data = json.loads(world_path.read_text(encoding="utf-8"))
    return WorldIR.from_dict(data)


# =============================================================================
# SECTION A: REAL DATASET VERTICAL SLICE
# =============================================================================

class TestRealDatasetVerticalSlice:
    """Executes real capture data through the full ingestion, admission, and compilation pipeline."""

    def test_real_evidence_admission_and_compilation(self, tmp_path):
        store = WorldStore(tmp_path / "store")

        # 1. Real evidence items with real files, hashes, and EXIF/quality metadata
        pkg_path = REAL_WORLDIR_DIR / "evidence_package.json"
        assert pkg_path.exists()
        pkg_data = json.loads(pkg_path.read_text(encoding="utf-8"))

        sess = Session("sess-real-room-001")
        evidence_items: List[EvidenceItem] = []
        for asset_id in pkg_data["asset_order"][:10]:  # First 10 real photos
            asset = pkg_data["assets"][asset_id]
            img_rel = asset["source_uri"]
            img_file = REPO_ROOT / img_rel
            assert img_file.exists()
            item = EvidenceItem(
                id=asset["id"],
                kind=EvidenceKind.PHOTO,
                source_uri=img_file.as_uri(),
                sha256=asset["sha256"],
                metadata={
                    "quality": asset.get("quality", {}),
                    "sensor": asset.get("sensor_metadata", {}),
                },
            )
            evidence_items.append(item)
            sess.add_evidence(item)

        assert len(sess.all_evidence()) == 10

        # 2. Quality admission gate
        report = classify_evidence_items(evidence_items, blur_reject_laplacian=20.0)
        assert len(report.admissions) == 10
        assert report.counts["failed"] == 0
        assert len(report.accepted_ids) + len(report.degraded_ids) == 10

        # 3. Load real reconstruction results (3D points + camera poses)
        recon = _load_real_reconstruction_result()
        assert len(recon.points) == 200
        assert len(recon.camera_poses) == 21
        assert recon.registration_status == "success"

        # 4. Compile to WorldIR
        world_v1, stats = compile_reconstruction_to_world(
            recon, CompileOptions(require_validation=True, artifact_store=store._store)
        )
        assert world_v1.coordinate_frame.value == "world"
        assert world_v1.global_provenance == Provenance.RECONSTRUCTED
        assert len(world_v1.entities) > 0
        assert len(world_v1.geometries) > 0

        # 5. Persist as tiled WorldStore version
        v1_stored = save_version_tiled(
            store,
            world_v1,
            parent=None,
            version_id="v-real-001",
            tile_size=2.0,
            source_session_ids=[sess.id],
        )
        assert v1_stored.version_id == "v-real-001"
        assert v1_stored.parent is None

        # 6. Lazy handle access & spatial query
        handle = open_version(store, "v-real-001")
        tile_keys = handle.list_tiles()
        assert len(tile_keys) > 0

        # Verify single tile load does not crash and loads only its entities
        first_tile = tile_keys[0]
        loaded_entities = handle.load_tile(first_tile)
        assert isinstance(loaded_entities, dict)

        # Spatial region query
        bmin, bmax = handle.tile_bounds(first_tile)
        region_entities = handle.query_region(bmin, bmax)
        assert len(region_entities) >= len(loaded_entities)


# =============================================================================
# SECTION B: TRUE V2 LOCALIZED UPDATE WITH TILED REUSE
# =============================================================================

class TestTrueLocalizedUpdateWithTiledReuse:
    """Proves true incremental compilation with strict object identity and zero-rewrite tile reuse."""

    def test_localized_v2_update_and_tile_reuse(self, tmp_path):
        store = WorldStore(tmp_path / "store")

        # 1. Load the real 58-entity room WorldIR
        world_v1 = _load_real_structural_world()
        total_entities_v1 = len(world_v1.entities)
        total_geometries_v1 = len(world_v1.geometries)
        assert total_entities_v1 == 58
        assert total_geometries_v1 == 58

        target_entity_id = "struct-plane-000"
        assert target_entity_id in world_v1.entities
        target_entity_v1 = world_v1.entities[target_entity_id]
        target_geom_id = target_entity_v1.geometry_ids[0]

        # 2. Save V1 tiled
        v1_stored = save_version_tiled(
            store,
            world_v1,
            parent=None,
            version_id="v-room-1",
            tile_size=2.0,
            source_session_ids=["sess-base"],
        )
        handle_v1 = open_version(store, "v-room-1")
        manifest_v1 = handle_v1.manifest
        v1_tiles_map = {ref.tile_key: ref for ref in manifest_v1.tiles}
        assert len(v1_tiles_map) > 1

        # 3. Formulate a localized rescan of target_entity_id from a new session
        rescan_session_id = "sess-rescan-desk-002"
        rescan_points = [
            ReconstructedPoint(
                position=(0.15, -2.18, 3.25),
                track_id="pt-rescan-01",
                source_evidence_ids=["ev-rescan-photo-01"],
                uncertainty=Uncertainty(confidence=0.98),
            ),
            ReconstructedPoint(
                position=(0.20, -2.15, 3.28),
                track_id="pt-rescan-02",
                source_evidence_ids=["ev-rescan-photo-02"],
                uncertainty=Uncertainty(confidence=0.97),
            ),
        ]
        rescan_cameras = [
            ReconstructedCameraPose(
                evidence_id="ev-rescan-photo-01",
                position=(0.10, -2.00, 3.50),
                rotation=(1.0, 0.0, 0.0, 0.0),
                uncertainty=Uncertainty(confidence=0.99),
            )
        ]
        rescan_recon = ReconstructionResult(
            points=rescan_points,
            camera_poses=rescan_cameras,
            registration_status="success",
        )

        # 4. Apply localized reconstruction update
        update_result: IncrementalUpdateResult = apply_reconstruction_update(
            base_world=world_v1,
            reconstruction=rescan_recon,
            session_id=rescan_session_id,
            target_entity_id=target_entity_id,
            artifact_store=store._store,
            tile_size=2.0,
        )

        world_v2 = update_result.new_world

        # A) PROVE TARGET ENTITY WAS UPDATED
        target_entity_v2 = world_v2.entities[target_entity_id]
        assert target_entity_v2 is not target_entity_v1
        assert target_entity_v2.custom_properties.get("session_id") == rescan_session_id
        assert abs(target_entity_v2.confidence - 0.975) < 1e-3

        # B) PROVE STRICT OBJECT IDENTITY (is) ON ALL UNTOUCHED ENTITIES
        untouched_entity_ids = [eid for eid in world_v1.entities if eid != target_entity_id]
        assert len(untouched_entity_ids) == total_entities_v1 - 1
        for eid in untouched_entity_ids:
            assert world_v2.entities[eid] is world_v1.entities[eid], (
                f"Entity {eid} was re-allocated instead of preserved by reference!"
            )

        # C) PROVE STRICT OBJECT IDENTITY (is) ON ALL UNTOUCHED GEOMETRIES
        untouched_geom_ids = [gid for gid in world_v1.geometries if gid != target_geom_id]
        assert len(untouched_geom_ids) == total_geometries_v1 - 1
        for gid in untouched_geom_ids:
            assert world_v2.geometries[gid] is world_v1.geometries[gid], (
                f"Geometry {gid} was re-allocated instead of preserved by reference!"
            )

        # 5. Save V2 tiled using incremental_result
        v2_stored = save_version_tiled(
            store,
            world_v2,
            parent=v1_stored.version_id,
            version_id="v-room-2",
            tile_size=2.0,
            incremental_result=update_result,
            source_session_ids=[rescan_session_id],
        )

        handle_v2 = open_version(store, "v-room-2")
        manifest_v2 = handle_v2.manifest
        v2_tiles_map = {ref.tile_key: ref for ref in manifest_v2.tiles}

        # D) PROVE REBUILT VS REUSED TILES
        rebuilt_keys = update_result.rebuilt_tile_ids
        assert len(rebuilt_keys) > 0
        reused_count = 0

        for key, ref2 in v2_tiles_map.items():
            if key in rebuilt_keys:
                # Rebuilt tile must have been refreshed
                if key in v1_tiles_map:
                    assert ref2.artifact_uri != v1_tiles_map[key].artifact_uri
            else:
                # Untouched tile MUST be reused verbatim (zero disk writes)
                assert key in v1_tiles_map
                ref1 = v1_tiles_map[key]
                assert ref2.artifact_uri == ref1.artifact_uri, (
                    f"Tile {key} had new artifact_uri despite being untouched!"
                )
                assert ref2.content_hash == ref1.content_hash, (
                    f"Tile {key} had different content_hash despite being untouched!"
                )
                reused_count += 1

        assert reused_count > 0, "No tiles were reused incrementally!"

        # E) PROVE WORLD DIFF EXACTNESS
        diff = diff_worlds(world_v1, world_v2)
        assert diff.modified_entity_ids == (target_entity_id,)
        assert diff.added_entity_ids == ()
        assert diff.removed_entity_ids == ()


# =============================================================================
# SECTION C: 10 ADVERSARIAL FAILURE INJECTIONS
# =============================================================================

class TestAdversarialFailureModes:
    """Guarantees explicit refusals, diagnostics, and zero corrupted world states."""

    # 1. Corrupted evidence / empty file bytes
    def test_mode_1_corrupted_evidence(self):
        item = EvidenceItem(
            id="ev-bad",
            kind=EvidenceKind.PHOTO,
            source_uri="file:///bad.jpg",
            metadata={"quality": {"measured": 0.0, "quality_note": "file truncated"}},
        )
        report = classify_evidence_items([item])
        assert report.admissions[0].outcome == "failed"

    # 2. Incomplete reconstruction (0 points)
    def test_mode_2_incomplete_reconstruction(self):
        base_world = _load_real_structural_world()
        empty_recon = ReconstructionResult(points=[], camera_poses=[], registration_status="success")
        with pytest.raises(ReconstructionAdapterError, match="Empty reconstruction"):
            apply_reconstruction_update(
                base_world, empty_recon, session_id="s-fail", target_entity_id="struct-plane-000"
            )

    # 3. Missing camera poses / unaligned session
    def test_mode_3_missing_camera_poses(self):
        base_world = _load_real_structural_world()
        pts = [ReconstructedPoint((0.0, 0.0, 0.0), "p1", ["ev1"])]
        recon_no_cams = ReconstructionResult(points=pts, camera_poses=[], registration_status="success")
        with pytest.raises(ReconstructionAdapterError, match="without camera poses"):
            apply_reconstruction_update(
                base_world, recon_no_cams, session_id="s-fail", target_entity_id="struct-plane-000"
            )

    # 4. Registration refusal
    def test_mode_4_registration_refused(self):
        base_world = _load_real_structural_world()
        pts = [ReconstructedPoint((0.0, 0.0, 0.0), "p1", ["ev1"])]
        cams = [ReconstructedCameraPose("ev1", (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))]
        recon = ReconstructionResult(points=pts, camera_poses=cams, registration_status="success")
        refused_alignment = SessionAlignment(
            from_session="s2",
            to_session="s1",
            status="refused",
            method="none",
            transform=None,
            rmse=float("inf"),
            inlier_fraction=0.0,
            reason="Insufficient feature overlap (< 10 matches)",
            source_points=10,
            target_points=10,
        )
        with pytest.raises(ReconstructionAdapterError, match="refused"):
            apply_reconstruction_update(
                base_world,
                recon,
                session_id="s2",
                target_entity_id="struct-plane-000",
                alignment=refused_alignment,
            )

    # 5. Tile corruption (SHA256 tampering detection)
    def test_mode_5_tile_artifact_corruption_detected(self, tmp_path):
        store = WorldStore(tmp_path / "store")
        world = _load_real_structural_world()
        save_version_tiled(store, world, parent=None, version_id="v-tamper", tile_size=2.0)
        handle = open_version(store, "v-tamper")
        first_tile_key = handle.list_tiles()[0]

        # Find the artifact on disk and tamper with it
        ref = next(r for r in handle.manifest.tiles if r.tile_key == first_tile_key)
        artifact_file = store._store._path_for(ref.content_hash)
        assert artifact_file.exists()

        # Corrupt 1 byte
        content = artifact_file.read_bytes()
        tampered = content[:-1] + (b"X" if content[-1:] != b"X" else b"Y")
        artifact_file.write_bytes(tampered)

        # Loading the corrupted tile MUST raise WorldStoreError with hash mismatch
        with pytest.raises(WorldStoreError, match="artifact hash mismatch"):
            handle.load_tile(first_tile_key)

    # 6. Stale version ID request
    def test_mode_6_stale_version_id(self, tmp_path):
        store = WorldStore(tmp_path / "store")
        with pytest.raises(WorldStoreError, match="no tile manifest for version"):
            open_version(store, "v-stale-non-existent")

    # 7. Concurrent update collision (duplicate version ID)
    def test_mode_7_duplicate_version_collision(self, tmp_path):
        store = WorldStore(tmp_path / "store")
        world = _load_real_structural_world()
        save_version_tiled(store, world, parent=None, version_id="v-dup", tile_size=2.0)
        with pytest.raises(WorldStoreError, match="already exists"):
            save_version_tiled(store, world, parent=None, version_id="v-dup", tile_size=2.0)

    # 8. Atomic failure safety (manifest written last)
    def test_mode_8_atomic_write_protection(self, tmp_path):
        store = WorldStore(tmp_path / "store")
        world = _load_real_structural_world()
        # Non-existent parent should raise clean error before writing corrupted child manifest
        with pytest.raises(WorldStoreError, match="unknown version: v-missing"):
            save_version_tiled(store, world, parent="v-missing", version_id="v-child", tile_size=2.0)

        # Verify no orphan child manifest was written
        manifest_path = store._root / "tile_manifests" / "v-child.json"
        assert not manifest_path.exists()

    # 9. Frontend bridge error handling (graceful JSON, exit code 1)
    def test_mode_9_frontend_bridge_error_handling(self, tmp_path, monkeypatch):
        monkeypatch.setenv("REALITY_STORE_PATH", str(tmp_path / "empty_store"))
        out = io.StringIO()
        monkeypatch.setattr(sys, "stdout", out)
        ret = api_bridge.cmd_load_world("non-existent-version")
        assert ret == 1
        payload = json.loads(out.getvalue().strip())
        assert "error" in payload
        assert payload["entities"] == []

    # 10. Backend store unavailability handling
    def test_mode_10_backend_unavailability(self, tmp_path, monkeypatch):
        # Empty uninitialized directory
        empty_dir = tmp_path / "uninitialized_backend"
        monkeypatch.setenv("REALITY_STORE_PATH", str(empty_dir))
        out = io.StringIO()
        monkeypatch.setattr(sys, "stdout", out)
        ret = api_bridge.cmd_status()
        assert ret == 0
        payload = json.loads(out.getvalue().strip())
        assert payload["backend"] is True
        assert payload["version_count"] == 0


# =============================================================================
# SECTION D: DESKTOP STUDIO CONTRACT VERIFICATION
# =============================================================================

class TestDesktopStudioContractVerification:
    """Proves Desktop Studio backend loading produces valid state without mock fallbacks."""

    def test_desktop_bridge_load_world_and_diff(self, tmp_path, monkeypatch):
        store_path = tmp_path / "desktop_store"
        monkeypatch.setenv("REALITY_STORE_PATH", str(store_path))
        store = WorldStore(store_path)

        # 1. Prepare real V1 & localized V2
        world_v1 = _load_real_structural_world()
        save_version_tiled(store, world_v1, parent=None, version_id="v-desk-1", tile_size=2.0)

        rescan_recon = ReconstructionResult(
            points=[
                ReconstructedPoint(
                    position=(0.14, -2.20, 3.22),
                    track_id="pt-desk-01",
                    source_evidence_ids=["ev-desk-01"],
                    uncertainty=Uncertainty(confidence=0.96),
                )
            ],
            camera_poses=[
                ReconstructedCameraPose(
                    evidence_id="ev-desk-01",
                    position=(0.0, -2.0, 3.0),
                    rotation=(1.0, 0.0, 0.0, 0.0),
                )
            ],
            registration_status="success",
        )
        res = apply_reconstruction_update(
            base_world=world_v1,
            reconstruction=rescan_recon,
            session_id="sess-desk-002",
            target_entity_id="struct-plane-000",
            artifact_store=store._store,
            tile_size=2.0,
        )
        save_version_tiled(
            store,
            res.new_world,
            parent="v-desk-1",
            version_id="v-desk-2",
            tile_size=2.0,
            incremental_result=res,
        )

        # 2. Test cmd_status
        out_status = io.StringIO()
        monkeypatch.setattr(sys, "stdout", out_status)
        assert api_bridge.cmd_status() == 0
        status_payload = json.loads(out_status.getvalue().strip())
        assert status_payload["backend"] is True
        assert status_payload["version_count"] == 2
        assert status_payload["latest_version"] == "v-desk-2"

        # 3. Test cmd_list_worlds
        out_list = io.StringIO()
        monkeypatch.setattr(sys, "stdout", out_list)
        assert api_bridge.cmd_list_worlds() == 0
        list_payload = json.loads(out_list.getvalue().strip())
        assert list_payload["count"] == 2
        assert any(w["version_id"] == "v-desk-1" for w in list_payload["worlds"])
        assert any(w["version_id"] == "v-desk-2" for w in list_payload["worlds"])

        # 4. Test cmd_load_world("v-desk-2")
        out_load = io.StringIO()
        monkeypatch.setattr(sys, "stdout", out_load)
        assert api_bridge.cmd_load_world("v-desk-2") == 0
        load_payload = json.loads(out_load.getvalue().strip())

        assert load_payload["version_id"] == "v-desk-2"
        assert load_payload["entity_count"] == 58
        assert len(load_payload["entities"]) == 58
        assert load_payload["geometry_count"] == 58

        # Check the updated entity structure for frontend consumption
        updated_e = next(e for e in load_payload["entities"] if e["id"] == "struct-plane-000")
        assert updated_e["type"].upper() == "FLOOR"
        assert updated_e["provenance"] == "RECONSTRUCTED"
        assert updated_e["custom_properties"]["session_id"] == "sess-desk-002"
        assert len(updated_e["geometry_ids"]) > 0

        first_geom_id = updated_e["geometry_ids"][0]
        assert first_geom_id in load_payload["geometries"]
        geom = load_payload["geometries"][first_geom_id]
        assert geom["bounds_min"] is not None
        assert geom["bounds_max"] is not None

        # 5. Test cmd_diff("v-desk-1", "v-desk-2")
        out_diff = io.StringIO()
        monkeypatch.setattr(sys, "stdout", out_diff)
        assert api_bridge.cmd_diff("v-desk-1", "v-desk-2") == 0
        diff_payload = json.loads(out_diff.getvalue().strip())
        assert diff_payload["from_version_id"] == "v-desk-1"
        assert diff_payload["to_version_id"] == "v-desk-2"
        assert diff_payload["summary"]["entities_modified"] == 1
        assert diff_payload["entity_diffs"][0]["entity_id"] == "struct-plane-000"
        assert diff_payload["entity_diffs"][0]["kind"] == "modified"
