"""End-to-End Integration: FreeBuff Reconstruction -> Claude WorldOS -> WorldStore -> Desktop.

Verifies the complete real cross-agent contract:
1. FreeBuff canonical reconstruction outputs (ReconstructionResult, camera poses, uncertainty)
2. Cross-session registration alignment (RigidTransform)
3. WorldOS incremental adapter (adapt_reconstruction_to_incremental_update)
4. Localized WorldStore V2 update (apply_reconstruction_update)
5. Locality proof (strict object reference identity 'is' on untouched regions)
6. Provenance firewall (entity -> geometry -> observation -> session -> evidence -> artifact)
7. Uncertainty firewall (survival without loss or fabricated certainty)
8. Coordinate frame firewall (explicit session-to-world rigid registration)
9. 10 failure injection modes (explicit refusals, diagnostics, no corrupted world)
10. Mobile consumption flow (Cline EvidencePackage -> Session -> FreeBuff -> WorldOS -> Desktop)
"""

from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path
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
from evidence.packages import (
    DeterministicPackageBuilder,
    EvidenceKind,
    EvidenceSource,
)
from evidence.session import EvidenceItem, Session
from provenance import Provenance, Uncertainty
from reconstruction.backend.interface import (
    ReconstructedCameraPose,
    ReconstructedPoint,
    ReconstructionResult,
)
from reconstruction.calibration.transforms import RigidTransform
from reconstruction.robustness import classify_evidence_items, ACCEPTED
from registration.cross_session import SessionAlignment, align_session
from world_ir.artifact_store import FileArtifactStore
from world_ir.coordinates import Frame, Transform
from world_ir.diff import diff_worlds
from world_ir.schema_v1 import (
    Entity,
    EntityType,
    Geometry,
    GeometryType,
    Relationship,
    RelationshipKind,
    Vector3,
)
from world_ir.spatial_tiles import SpatialTiles
from world_ir.world_v1 import WorldIR
from worldstore.store import WorldStore, WorldStoreError


def _create_synthetic_base_world(store: WorldStore) -> Tuple[WorldIR, Session]:
    """Sets up a clean World V1 containing room-1 and room-annex."""
    sess_1 = Session("sess-base-001")
    item_1 = EvidenceItem(
        id="ev-base-001",
        kind=EvidenceKind.PHOTO,
        source_uri="file:///captures/base_room.jpg",
        metadata={"source_id": "device-01"},
    )
    sess_1.add_evidence(item_1)

    # Base points: room 1 in [0, 4], annex in [5, 8]
    pts_r1 = [
        ReconstructedPoint(
            position=(float(x), float(y), 1.0),
            track_id=f"pt-r1-{x}-{y}",
            source_evidence_ids=[item_1.id],
            uncertainty=Uncertainty(confidence=0.95),
        )
        for x in range(0, 4)
        for y in range(0, 4)
    ]
    pts_ax = [
        ReconstructedPoint(
            position=(float(x), float(y), 1.0),
            track_id=f"pt-ax-{x}-{y}",
            source_evidence_ids=[item_1.id],
            uncertainty=Uncertainty(confidence=0.92),
        )
        for x in range(5, 8)
        for y in range(0, 4)
    ]
    cams = [
        ReconstructedCameraPose(
            evidence_id=item_1.id,
            position=(2.0, 2.0, 1.5),
            rotation=(1.0, 0.0, 0.0, 0.0),
            uncertainty=Uncertainty(confidence=0.98),
        )
    ]
    recon = ReconstructionResult(
        points=pts_r1 + pts_ax,
        camera_poses=cams,
        registration_status="success",
    )
    world_v1, _ = compile_reconstruction_to_world(
        recon, CompileOptions(require_validation=True, artifact_store=store._store)
    )

    # Concrete entity structure
    geom_r1 = Geometry(
        id="geom-room-1",
        type=GeometryType.BOX,
        bounds_min=Vector3(0.0, 0.0, 0.0),
        bounds_max=Vector3(4.0, 4.0, 3.0),
        provenance=Provenance.RECONSTRUCTED,
        confidence=0.95,
    )
    e_r1 = Entity(
        id="room-1",
        name="Main Room",
        type=EntityType.ROOM,
        geometry_ids=["geom-room-1"],
        provenance=Provenance.RECONSTRUCTED,
        confidence=0.95,
    )

    geom_ax = Geometry(
        id="geom-room-annex",
        type=GeometryType.BOX,
        bounds_min=Vector3(5.0, 0.0, 0.0),
        bounds_max=Vector3(8.0, 4.0, 3.0),
        provenance=Provenance.RECONSTRUCTED,
        confidence=0.92,
    )
    e_ax = Entity(
        id="room-annex",
        name="Annex Room",
        type=EntityType.ROOM,
        geometry_ids=["geom-room-annex"],
        relationships=[
            Relationship(kind=RelationshipKind.ADJACENT_TO, target_id="room-1", provenance=Provenance.OBSERVED)
        ],
        provenance=Provenance.RECONSTRUCTED,
        confidence=0.92,
    )

    world_v1.entities["room-1"] = e_r1
    world_v1.geometries["geom-room-1"] = geom_r1
    world_v1.entities["room-annex"] = e_ax
    world_v1.geometries["geom-room-annex"] = geom_ax

    store.save_version(world_v1, parent=None, version_id="v-base-1", source_session_ids=[sess_1.id])
    return world_v1, sess_1


class TestGoldenRealisticFlow:
    """Phase 3: Golden end-to-end flow from real/realistic reconstruction to WorldOS update."""

    def test_full_reconstruction_to_worldos_update_flow(self, tmp_path, monkeypatch):
        store_root = tmp_path / "store"
        store = WorldStore(store_root)
        monkeypatch.setenv("REALITY_STORE_PATH", str(store_root))

        # 1. Base World V1
        world_v1, sess_1 = _create_synthetic_base_world(store)

        # 2. Session 2 Rescan (Annex only, with slight translation shift)
        sess_2 = Session("sess-annex-rescan-002")
        item_2a = EvidenceItem(
            id="ev-rescan-001",
            kind=EvidenceKind.PHOTO,
            source_uri="file:///captures/annex_rescan_1.jpg",
            metadata={"source_id": "device-02"},
        )
        item_2b = EvidenceItem(
            id="ev-rescan-002",
            kind=EvidenceKind.PHOTO,
            source_uri="file:///captures/annex_rescan_2.jpg",
            metadata={"source_id": "device-02"},
        )
        sess_2.add_evidence(item_2a)
        sess_2.add_evidence(item_2b)

        # Session 2 reconstruction output (FreeBuff)
        # Shifted by +0.02m due to independent local coordinate frame
        rescan_pts = [
            ReconstructedPoint(
                position=(float(x) + 0.02, float(y) + 0.01, 1.005),
                track_id=f"pt-rescan-{x}-{y}",
                source_evidence_ids=[item_2a.id, item_2b.id],
                uncertainty=Uncertainty(confidence=0.96),
            )
            for x in range(5, 9)  # Annex expanded to x=9
            for y in range(0, 5)  # Annex expanded to y=5
        ]
        rescan_cams = [
            ReconstructedCameraPose(
                evidence_id=item_2a.id,
                position=(6.5, 2.0, 1.6),
                rotation=(1.0, 0.0, 0.0, 0.0),
                uncertainty=Uncertainty(confidence=0.97),
            )
        ]
        recon_result_2 = ReconstructionResult(
            points=rescan_pts,
            camera_poses=rescan_cams,
            registration_status="success",
        )

        # 3. Cross-session registration alignment
        cloud_base_annex = [Vec3(x, y, 1.0) for x in range(5, 8) for y in range(0, 4)]
        cloud_rescan = [Vec3(p.position[0], p.position[1], p.position[2]) for p in rescan_pts]
        alignment = align_session(cloud_rescan, cloud_base_annex, from_session=sess_2.id, to_session=sess_1.id)
        assert alignment.status == "accepted"
        assert alignment.transform is not None

        # 4. Bridge to WorldOS incremental update via Adapter
        update_result = apply_reconstruction_update(
            base_world=world_v1,
            reconstruction=recon_result_2,
            session_id=sess_2.id,
            target_entity_id="room-annex",
            alignment=alignment,
            artifact_store=store._store,
            tile_size=4.0,
        )
        world_v2 = update_result.new_world

        # 5. Persist World V2 to WorldStore
        ver_2 = store.save_version(
            world_v2,
            parent="v-base-1",
            version_id="v-base-2",
            source_session_ids=[sess_2.id],
        )
        assert ver_2.version_id == "v-base-2"
        assert ver_2.parent == "v-base-1"
        assert store.parents("v-base-2") == ["v-base-1"]

        # 6. Verify Diff
        diff = diff_worlds(world_v1, world_v2)
        diff_entities = {d.entity_id for d in diff.entity_diffs}
        assert "room-annex" in diff_entities
        assert "room-1" not in diff_entities

        # 7. Desktop API bridge verification
        out_diff = io.StringIO()
        monkeypatch.setattr(sys, "stdout", out_diff)
        ret = api_bridge.cmd_diff("v-base-1", "v-base-2")
        assert ret == 0
        diff_payload = json.loads(out_diff.getvalue().strip())
        assert any(d["entity_id"] == "room-annex" for d in diff_payload["entities"])
        assert not any(d["entity_id"] == "room-1" for d in diff_payload["entities"])


class TestLocalityProof:
    """Phase 4: Proves true locality - unchanged regions remain 100% reused by reference."""

    def test_locality_invariants(self, tmp_path):
        store = WorldStore(tmp_path / "store")
        world_v1, sess_1 = _create_synthetic_base_world(store)

        # Rescan modifying only room-annex
        rescan_pts = [
            ReconstructedPoint(
                position=(float(x), float(y), 1.0),
                track_id=f"pt-{x}-{y}",
                source_evidence_ids=["ev-01"],
                uncertainty=Uncertainty(confidence=0.94),
            )
            for x in range(5, 9)
            for y in range(0, 4)
        ]
        rescan_cams = [
            ReconstructedCameraPose(
                evidence_id="ev-01",
                position=(6.0, 2.0, 1.5),
                rotation=(1.0, 0.0, 0.0, 0.0),
            )
        ]
        recon = ReconstructionResult(
            points=rescan_pts,
            camera_poses=rescan_cams,
            registration_status="success",
        )

        update_result = apply_reconstruction_update(
            base_world=world_v1,
            reconstruction=recon,
            session_id="sess-loc-02",
            target_entity_id="room-annex",
            artifact_store=store._store,
            tile_size=4.0,
        )
        world_v2 = update_result.new_world

        # MEASUREMENTS:
        # Changed entities
        assert update_result.changed_entity_ids == {"room-annex"}
        # Reused entities
        assert "room-1" in update_result.reused_entity_ids
        # Strict memory identity 'is'
        assert world_v2.entities["room-1"] is world_v1.entities["room-1"]
        # Reused geometries
        assert "geom-room-1" in update_result.reused_geometry_ids
        assert world_v2.geometries["geom-room-1"] is world_v1.geometries["geom-room-1"]
        # Unaffected tile (0, 0, 0) was NOT invalidated
        assert (0, 0, 0) not in update_result.invalidated_tile_ids


class TestProvenanceFirewallE2E:
    """Phase 5: Verifies end-to-end provenance and evidence traceability."""

    def test_entity_to_source_artifact_lineage(self, tmp_path):
        store = WorldStore(tmp_path / "store")
        world_v1, _ = _create_synthetic_base_world(store)

        recon_pts = [
            ReconstructedPoint(
                position=(5.0, 2.0, 1.0),
                track_id="pt-1",
                source_evidence_ids=["ev-capture-001", "ev-capture-002"],
                uncertainty=Uncertainty(confidence=0.93),
            )
        ]
        recon_cams = [
            ReconstructedCameraPose(
                evidence_id="ev-capture-001",
                position=(5.0, 1.0, 1.5),
                rotation=(1.0, 0.0, 0.0, 0.0),
            )
        ]
        recon = ReconstructionResult(
            points=recon_pts,
            camera_poses=recon_cams,
            registration_status="success",
        )

        update_result = apply_reconstruction_update(
            base_world=world_v1,
            reconstruction=recon,
            session_id="sess-prov-001",
            target_entity_id="room-annex",
            artifact_store=store._store,
        )
        world_v2 = update_result.new_world
        updated_entity = world_v2.entities["room-annex"]

        # 1. Entity provenance
        assert updated_entity.provenance == Provenance.RECONSTRUCTED
        assert updated_entity.provenance != Provenance.OBSERVED  # Must not claim observed

        # 2. Geometry provenance & content-addressed storage
        geom = world_v2.geometries[updated_entity.geometry_ids[0]]
        assert geom.provenance == Provenance.RECONSTRUCTED
        assert geom.data_uri.startswith("artifact://")
        assert geom.data_hash != ""

        # 3. Observations trace to source evidence IDs
        obs_evidence_uris = {o.data_uri for o in updated_entity.observations}
        assert "evidence://ev-capture-001" in obs_evidence_uris
        assert "evidence://ev-capture-002" in obs_evidence_uris

        # 4. Session traceability
        assert updated_entity.custom_properties["session_id"] == "sess-prov-001"


class TestUncertaintyFirewallE2E:
    """Phase 6: Verifies uncertainty survives from points to WorldStore and Desktop."""

    def test_uncertainty_survival(self, tmp_path, monkeypatch):
        store_root = tmp_path / "store"
        store = WorldStore(store_root)
        monkeypatch.setenv("REALITY_STORE_PATH", str(store_root))

        world_v1, _ = _create_synthetic_base_world(store)

        # Points with specific confidence scores
        pts = [
            ReconstructedPoint(
                position=(5.0, 2.0, 1.0),
                track_id="pt-u1",
                source_evidence_ids=["ev-u1"],
                uncertainty=Uncertainty(confidence=0.88),
            ),
            ReconstructedPoint(
                position=(5.5, 2.0, 1.0),
                track_id="pt-u2",
                source_evidence_ids=["ev-u2"],
                uncertainty=Uncertainty(confidence=0.92),
            ),
        ]
        cams = [
            ReconstructedCameraPose(
                evidence_id="ev-u1",
                position=(5.0, 1.0, 1.5),
                rotation=(1.0, 0.0, 0.0, 0.0),
                uncertainty=Uncertainty(confidence=0.90),
            )
        ]
        recon = ReconstructionResult(points=pts, camera_poses=cams, registration_status="success")

        update_result = apply_reconstruction_update(
            base_world=world_v1,
            reconstruction=recon,
            session_id="sess-unc-001",
            target_entity_id="room-annex",
            artifact_store=store._store,
        )
        world_v2 = update_result.new_world
        ver = store.save_version(world_v2, parent=None, version_id="v-unc-test")

        # Mean confidence = (0.88 + 0.92) / 2 = 0.90
        assert abs(world_v2.entities["room-annex"].confidence - 0.90) < 1e-4

        # Verify survival through Desktop API bridge
        out = io.StringIO()
        monkeypatch.setattr(sys, "stdout", out)
        ret = api_bridge.cmd_load_world("v-unc-test")
        assert ret == 0
        loaded = json.loads(out.getvalue().strip())
        annex_loaded = next(e for e in loaded["entities"] if e["id"] == "room-annex")
        assert abs(annex_loaded["confidence"] - 0.90) < 1e-4


class TestCoordinateFrameFirewallE2E:
    """Phase 7: Verifies cross-session rigid registration and coordinate frame discipline."""

    def test_session_registration_transform_applied(self, tmp_path):
        store = WorldStore(tmp_path / "store")
        world_v1, _ = _create_synthetic_base_world(store)

        # Local capture with +100m offset in x
        pts_offset = [
            ReconstructedPoint(
                position=(105.0, 2.0, 1.0),
                track_id="pt-off",
                source_evidence_ids=["ev-off"],
                uncertainty=Uncertainty(confidence=0.9),
            )
        ]
        cams_offset = [
            ReconstructedCameraPose(
                evidence_id="ev-off",
                position=(105.0, 1.0, 1.5),
                rotation=(1.0, 0.0, 0.0, 0.0),
            )
        ]
        recon = ReconstructionResult(points=pts_offset, camera_poses=cams_offset, registration_status="success")

        # Alignment correcting the -100m offset back into world frame
        alignment_transform = RigidTransform(
            from_frame="sess-offset",
            to_frame="world",
            translation=Vec3(-100.0, 0.0, 0.0),
        )

        update_result = apply_reconstruction_update(
            base_world=world_v1,
            reconstruction=recon,
            session_id="sess-offset",
            target_entity_id="room-annex",
            alignment=alignment_transform,
            artifact_store=store._store,
        )
        world_v2 = update_result.new_world
        geom = world_v2.geometries["geom-room-annex"]

        # Points transformed from x=105 to x=5
        assert abs(geom.bounds_min.x - 5.0) < 1e-4
        assert abs(geom.bounds_max.x - 5.0) < 1e-4


class TestFailureInjection10Modes:
    """Phase 8: Tests 10 failure injection modes for explicit refusals and diagnostics."""

    # 1. Reconstruction unavailable / empty points
    def test_failure_mode_1_empty_reconstruction(self, tmp_path):
        world = WorldIR(id="w-fail-1")
        recon = ReconstructionResult(points=[], camera_poses=[], registration_status="success")
        with pytest.raises(ReconstructionAdapterError, match="Empty reconstruction"):
            apply_reconstruction_update(world, recon, session_id="s1", target_entity_id="e1")

    # 2. Registration failure (refused alignment)
    def test_failure_mode_2_registration_refused(self, tmp_path):
        world = WorldIR(id="w-fail-2")
        pts = [ReconstructedPoint((0.0, 0.0, 0.0), "p1", ["ev1"])]
        cams = [ReconstructedCameraPose("ev1", (0.0, 0.0, 1.0), (1.0, 0.0, 0.0, 0.0))]
        recon = ReconstructionResult(points=pts, camera_poses=cams, registration_status="success")

        refused_alignment = SessionAlignment(
            from_session="s2",
            to_session="s1",
            status="refused",
            method="none",
            transform=None,
            rmse=999.0,
            inlier_fraction=0.0,
            reason="insufficient point overlap",
            source_points=10,
            target_points=10,
        )
        with pytest.raises(ReconstructionAdapterError, match="Registration refused"):
            apply_reconstruction_update(
                world, recon, session_id="s2", target_entity_id="e1", alignment=refused_alignment
            )

    # 3. Malformed ReconstructionResult (missing camera poses)
    def test_failure_mode_3_missing_cameras(self, tmp_path):
        world = WorldIR(id="w-fail-3")
        pts = [ReconstructedPoint((0.0, 0.0, 0.0), "p1", ["ev1"])]
        recon = ReconstructionResult(points=pts, camera_poses=[], registration_status="success")
        with pytest.raises(ReconstructionAdapterError, match="without camera poses"):
            apply_reconstruction_update(world, recon, session_id="s1", target_entity_id="e1")

    # 4. Missing / invalid provenance
    def test_failure_mode_4_invalid_provenance(self):
        with pytest.raises(ValueError):
            Provenance("NON_EXISTENT_PROVENANCE")

    # 5. Missing / out-of-bounds uncertainty
    def test_failure_mode_5_invalid_uncertainty(self):
        with pytest.raises(ValueError, match="confidence must be in"):
            Uncertainty(confidence=1.5)

    # 6. Coordinate-frame mismatch (unsupported alignment type)
    def test_failure_mode_6_invalid_alignment_type(self, tmp_path):
        world = WorldIR(id="w-fail-6")
        pts = [ReconstructedPoint((0.0, 0.0, 0.0), "p1", ["ev1"])]
        cams = [ReconstructedCameraPose("ev1", (0.0, 0.0, 1.0), (1.0, 0.0, 0.0, 0.0))]
        recon = ReconstructionResult(points=pts, camera_poses=cams, registration_status="success")
        with pytest.raises(TypeError, match="Unsupported alignment type"):
            apply_reconstruction_update(
                world, recon, session_id="s1", target_entity_id="e1", alignment="NOT_A_TRANSFORM"
            )

    # 7. Invalid entity (validation gate rejection)
    def test_failure_mode_7_failed_reconstruction_status(self, tmp_path):
        world = WorldIR(id="w-fail-7")
        pts = [ReconstructedPoint((0.0, 0.0, 0.0), "p1", ["ev1"])]
        cams = [ReconstructedCameraPose("ev1", (0.0, 0.0, 1.0), (1.0, 0.0, 0.0, 0.0))]
        recon = ReconstructionResult(points=pts, camera_poses=cams, registration_status="failed")
        with pytest.raises(ReconstructionAdapterError, match="Cannot adapt failed reconstruction"):
            apply_reconstruction_update(world, recon, session_id="s1", target_entity_id="e1")

    # 8. Tile parameter error
    def test_failure_mode_8_tile_parameter_error(self, tmp_path):
        world = WorldIR(id="w-fail-8")
        pts = [ReconstructedPoint((0.0, 0.0, 0.0), "p1", ["ev1"])]
        cams = [ReconstructedCameraPose("ev1", (0.0, 0.0, 1.0), (1.0, 0.0, 0.0, 0.0))]
        recon = ReconstructionResult(points=pts, camera_poses=cams, registration_status="success")
        with pytest.raises(ValueError, match="tile_size must be positive"):
            apply_reconstruction_update(world, recon, session_id="s1", target_entity_id="e1", tile_size=-2.0)

    # 9. WorldStore immutability violation
    def test_failure_mode_9_worldstore_immutability(self, tmp_path):
        store = WorldStore(tmp_path / "store")
        world = WorldIR(id="w-immutable")
        store.save_version(world, parent=None, version_id="v-fixed")
        with pytest.raises(WorldStoreError, match="already exists -- versions are immutable"):
            store.save_version(world, parent=None, version_id="v-fixed")

    # 10. Desktop backend unavailable
    def test_failure_mode_10_desktop_backend_unavailable(self, tmp_path, monkeypatch):
        empty_dir = tmp_path / "nonexistent"
        monkeypatch.setenv("REALITY_STORE_PATH", str(empty_dir))
        out = io.StringIO()
        monkeypatch.setattr(sys, "stdout", out)
        ret = api_bridge.cmd_load_world("v-does-not-exist")
        assert ret == 1
        payload = json.loads(out.getvalue().strip())
        assert "error" in payload


class TestMobileConsumptionE2E:
    """Phase 10: Verifies full path from Cline EvidencePackage -> Session -> FreeBuff -> WorldOS -> Desktop."""

    def test_mobile_package_to_worldos_flow(self, tmp_path, monkeypatch):
        store_root = tmp_path / "store"
        store = WorldStore(store_root)
        monkeypatch.setenv("REALITY_STORE_PATH", str(store_root))

        world_v1, _ = _create_synthetic_base_world(store)

        # 1. Mobile EvidencePackage created
        source = EvidenceSource(source_id="mobile-phone-01", platform="phone", device="Pixel 8 Pro")
        builder = DeterministicPackageBuilder("pkg-mobile-01")
        builder.register_source(source)

        jpeg_payload = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00" + b"\x00" * 32
        asset_id = builder.add_payload(
            payload=jpeg_payload,
            kind=EvidenceKind.PHOTO,
            source=source,
            source_uri="photo_001.jpg",
            acquired_at=1700000000.0,
            sensor_metadata={
                "focal_length": 24.0,
                "quality": {"laplacian_variance": 250.0, "clipped_fraction": 0.05, "measured": 1.0},
            },
            quality={"laplacian_variance": 250.0, "clipped_fraction": 0.05, "measured": 1.0},
        )
        pkg = builder.build()

        # 2. Session created
        sess = Session("sess-mobile-01")
        item = pkg.assets[asset_id].to_evidence_item()
        sess.add_evidence(item)

        # 3. Robustness admission gate
        admissions = classify_evidence_items([item])
        assert admissions.counts[ACCEPTED] == 1

        # 4. Reconstruction result
        recon = ReconstructionResult(
            points=[
                ReconstructedPoint(
                    position=(5.2, 2.1, 1.0),
                    track_id="pt-mob-1",
                    source_evidence_ids=[item.id],
                    uncertainty=Uncertainty(confidence=0.96),
                )
            ],
            camera_poses=[
                ReconstructedCameraPose(
                    evidence_id=item.id,
                    position=(5.0, 1.0, 1.5),
                    rotation=(1.0, 0.0, 0.0, 0.0),
                    uncertainty=Uncertainty(confidence=0.98),
                )
            ],
            registration_status="success",
        )

        # 5. Adapt to WorldOS update
        update_result = apply_reconstruction_update(
            base_world=world_v1,
            reconstruction=recon,
            session_id=sess.id,
            target_entity_id="room-annex",
            artifact_store=store._store,
        )
        world_v2 = update_result.new_world
        store.save_version(world_v2, parent="v-base-1", version_id="v-mobile-update", source_session_ids=[sess.id])

        # 6. Desktop consumption
        out = io.StringIO()
        monkeypatch.setattr(sys, "stdout", out)
        ret = api_bridge.cmd_load_world("v-mobile-update")
        assert ret == 0
        loaded = json.loads(out.getvalue().strip())
        assert any(e["id"] == "room-annex" for e in loaded["entities"])
        annex_entity = next(e for e in loaded["entities"] if e["id"] == "room-annex")
        assert annex_entity["custom_properties"]["session_id"] == "sess-mobile-01"


class TestDesktopProof:
    """Phase 9: Verifies the Desktop user-visible contract for World V2."""

    def test_desktop_v2_user_visible_contract(self, tmp_path, monkeypatch):
        store_root = tmp_path / "store"
        store = WorldStore(store_root)
        monkeypatch.setenv("REALITY_STORE_PATH", str(store_root))

        world_v1, _ = _create_synthetic_base_world(store)

        # Update Annex
        recon = ReconstructionResult(
            points=[
                ReconstructedPoint((5.5, 2.0, 1.0), "pt-dt-1", ["ev-dt-1"], Uncertainty(0.95)),
                ReconstructedPoint((8.5, 4.0, 3.0), "pt-dt-2", ["ev-dt-2"], Uncertainty(0.93)),
            ],
            camera_poses=[ReconstructedCameraPose("ev-dt-1", (5.0, 1.0, 1.5), (1.0, 0.0, 0.0, 0.0))],
            registration_status="success",
        )
        update_result = apply_reconstruction_update(
            base_world=world_v1,
            reconstruction=recon,
            session_id="sess-desktop-02",
            target_entity_id="room-annex",
            target_entity_name="Annex Suite Rescanned",
            artifact_store=store._store,
        )
        world_v2 = update_result.new_world
        store.save_version(world_v2, parent="v-base-1", version_id="v-desktop-v2", source_session_ids=["sess-desktop-02"])

        # 1. Desktop World Load API
        out_load = io.StringIO()
        monkeypatch.setattr(sys, "stdout", out_load)
        ret = api_bridge.cmd_load_world("v-desktop-v2")
        assert ret == 0
        loaded = json.loads(out_load.getvalue().strip())

        # Verify Changed Entity: Annex
        annex_ui = next((e for e in loaded["entities"] if e["id"] == "room-annex"), None)
        assert annex_ui is not None
        assert annex_ui["provenance"] == "RECONSTRUCTED"
        assert abs(annex_ui["confidence"] - 0.94) < 1e-4
        assert annex_ui["custom_properties"]["session_id"] == "sess-desktop-02"

        # Verify Unchanged Entity: Room 1
        room1_ui = next((e for e in loaded["entities"] if e["id"] == "room-1"), None)
        assert room1_ui is not None
        assert room1_ui["provenance"] == "RECONSTRUCTED"
        assert abs(room1_ui["confidence"] - 0.95) < 1e-4

        # Verify Geometries and Bounds
        annex_geom = loaded["geometries"]["geom-room-annex"]
        assert annex_geom["bounds_min"] == {"x": 5.5, "y": 2.0, "z": 1.0}
        assert annex_geom["bounds_max"] == {"x": 8.5, "y": 4.0, "z": 3.0}

        # 2. Desktop Diff API
        out_diff = io.StringIO()
        monkeypatch.setattr(sys, "stdout", out_diff)
        ret_diff = api_bridge.cmd_diff("v-base-1", "v-desktop-v2")
        assert ret_diff == 0
        diff_payload = json.loads(out_diff.getvalue().strip())
        changed_ids = [d["entity_id"] for d in diff_payload["entities"]]
        assert "room-annex" in changed_ids
        assert "room-1" not in changed_ids

