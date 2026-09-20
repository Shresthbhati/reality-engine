"""Phase 9: Golden Vertical Slice Acceptance Test Suite.

Verifies the entire golden path end-to-end:
Capture 1:
  Evidence (Package / Source / Camera)
   ↓
  Session 1
   ↓
  Registration
   ↓
  Reconstruction
   ↓
  WorldIR Compiler
   ↓
  WorldStore V1
   ↓
  Spatial Query / Index
   ↓
  Desktop Backend Bridge (cmd_load_world)

Then Capture 2 (MANDATORY SECOND PASS):
  Second Evidence Capture (Localized to Annex)
   ↓
  Session 2
   ↓
  Registration / Alignment to Session 1
   ↓
  Localized Update via affected_closure() + apply_incremental_update()
   ↓
  WorldStore V2 (with parent lineage to V1)
   ↓
  WorldDiff (strict structural diff: untouched Room 1 preserved)
   ↓
  Desktop Comparison via api_bridge.cmd_diff

This proves the Reality Engine's core claim:
Incremental updates update the existing world locally without recompiling from zero.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
import pytest

from apps.cli import api_bridge
from apps.cli.mobile_bridge import (
    derive_capture_tasks,
    ingest_mobile_bundle_to_dir,
    load_mobile_bundle,
)
from engine.compiler import CompileOptions, compile_reconstruction_to_world
from engine.math import Vec3
from evidence.multi_source import MultiSourceSession
from evidence.packages import (
    DeterministicPackageBuilder,
    EvidenceKind,
    EvidenceSource,
)
from evidence.session import EvidenceItem, Session
from exporters.citygml.exporter import export_to_citygml
from exporters.cityjson.exporter import export_to_cityjson
from exporters.gltf.exporter import export_to_gltf
from exporters.usd.exporter import export_to_usda
from provenance import Provenance, Uncertainty
from reconstruction.backend.interface import (
    ReconstructedCameraPose,
    ReconstructedPoint,
    ReconstructionResult,
)
from registration.cross_session import align_session
from world_ir.diff import diff_worlds, ChangeKind
from world_ir.incremental import affected_closure, apply_incremental_update
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
from worldstore.store import WorldStore


class TestGoldenWorldFlowSuite:
    """End-to-end integration test of the golden vertical slice and true localized update."""

    def test_golden_vertical_slice_two_pass_flow(self, tmp_path, monkeypatch):
        # Setup clean store environment
        store_root = tmp_path / "store"
        store = WorldStore(store_root)
        monkeypatch.setenv("REALITY_STORE_PATH", str(store_root))

        # =====================================================================
        # PASS 1: Base World (Room 1 + Room Annex)
        # =====================================================================

        # 0. Mobile Capture Bundle Creation (Phone Export)
        raw_jpeg_1 = (
            b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00C\x00\x08\x06"
            b"\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14"
            b"\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342\xff\xc0\x00\x0b"
            b"\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01"
            b"\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xda\x00"
            b"\x08\x01\x01\x00\x00?\x00\xbf\x00\xff\xd9"
        )
        sha_1 = hashlib.sha256(raw_jpeg_1).hexdigest()
        frame_id_1 = f"frame:{sha_1[:16]}"
        bundle_doc_1 = {
            "schema": "re.mobile-session-bundle/v1",
            "bundleId": "bundle:golden-001",
            "exportedAt": "2026-09-20T10:00:00Z",
            "device": "iPhone 15 Pro",
            "session": {
                "sessionId": "sess-base-001",
                "name": "Golden Capture Site",
                "intent": "inspection",
            },
            "frames": [
                {
                    "frameId": frame_id_1,
                    "contentSha256": sha_1,
                    "capturedAt": "2026-09-20T09:59:00Z",
                    "width": 1920,
                    "height": 1080,
                    "bytes": len(raw_jpeg_1),
                    "mime": "image/jpeg",
                    "verdict": "USEFUL",
                    "reasons": ["sharpness_ok", "exposure_ok"],
                    "telemetry": {
                        "headingDeg": 15.0,
                        "geolocation": {
                            "latitude": 37.7749,
                            "longitude": -122.4194,
                            "accuracyM": 5,
                        },
                    },
                    "payloadBase64": base64.b64encode(raw_jpeg_1).decode("ascii"),
                }
            ],
            "skippedFrames": [],
            "taskOutcomes": [],
        }
        bundle_path_1 = tmp_path / "mobile_bundle_1.json"
        bundle_path_1.write_text(json.dumps(bundle_doc_1), encoding="utf-8")

        # 1. Desktop Loader: Ingest Mobile Bundle into Session Store
        sess_dir_1 = tmp_path / "session_1"
        sess_dir_1.mkdir(parents=True, exist_ok=True)
        bundle_1 = load_mobile_bundle(str(bundle_path_1))
        sess_1 = MultiSourceSession("sess-base-001", name="Golden Capture Site")
        ingest_report_1 = ingest_mobile_bundle_to_dir(bundle_1, sess_1, sess_dir_1)
        assert ingest_report_1["frames_verified"] == 1
        assert ingest_report_1["integrity_failures"] == []
        assert (sess_dir_1 / "mobile" / "rgb" / f"{sha_1}.jpg").exists()

        # Follow-up capture tasks derived from real bundle telemetry
        tasks_doc_1 = derive_capture_tasks(bundle_1)
        assert tasks_doc_1["coverageAnalysis"]["state"] == "AVAILABLE"
        assert tasks_doc_1["gpsBounds"] is not None

        # Extract evidence item from ingested session
        item_1 = sess_1.package.all_assets()[0].to_evidence_item()

        # 3. Reconstruction 1: Two rooms (room-1: x in [0, 4], annex: x in [5, 8])
        pts_room1 = [
            ReconstructedPoint(
                position=(float(x), float(y), 1.0),
                track_id=f"pt-r1-{x}-{y}",
                source_evidence_ids=[item_1.id],
            )
            for x in range(0, 4)
            for y in range(0, 4)
        ]
        pts_annex = [
            ReconstructedPoint(
                position=(float(x), float(y), 1.0),
                track_id=f"pt-ax-{x}-{y}",
                source_evidence_ids=[item_1.id],
            )
            for x in range(5, 8)
            for y in range(0, 4)
        ]
        cams_1 = [
            ReconstructedCameraPose(
                evidence_id=item_1.id,
                position=(2.0, 2.0, 1.5),
                rotation=(1.0, 0.0, 0.0, 0.0),
            )
        ]
        recon_1 = ReconstructionResult(
            points=pts_room1 + pts_annex,
            camera_poses=cams_1,
            registration_status="success",
        )

        # 4. Compile to WorldIR V1
        world_v1, diag_1 = compile_reconstruction_to_world(
            recon_1,
            CompileOptions(require_validation=True, artifact_store=store._store),
        )
        assert world_v1 is not None
        assert len(world_v1.entities) > 0

        # Synthesize well-defined entities for spatial tile and localized diff testing
        e_room1 = Entity(
            id="room-1",
            type=EntityType.ROOM,
            provenance=Provenance.RECONSTRUCTED,
            confidence=0.95,
        )
        e_room1.transform = {"position": {"x": 2.0, "y": 2.0, "z": 1.5}}
        geom_r1 = Geometry(
            id="geom-r1",
            type=GeometryType.BOX,
            bounds_min=Vector3(0.0, 0.0, 0.0),
            bounds_max=Vector3(4.0, 4.0, 3.0),
            provenance=Provenance.RECONSTRUCTED,
        )
        e_room1.geometry_ids.append("geom-r1")

        e_annex = Entity(
            id="room-annex",
            type=EntityType.ROOM,
            provenance=Provenance.RECONSTRUCTED,
            confidence=0.92,
        )
        e_annex.transform = {"position": {"x": 6.5, "y": 2.0, "z": 1.5}}
        geom_annex = Geometry(
            id="geom-annex-v1",
            type=GeometryType.BOX,
            bounds_min=Vector3(5.0, 0.0, 0.0),
            bounds_max=Vector3(8.0, 4.0, 3.0),
            provenance=Provenance.RECONSTRUCTED,
        )
        e_annex.geometry_ids.append("geom-annex-v1")

        # Partitioning relationships
        e_annex.relationships.append(
            Relationship(
                kind=RelationshipKind.ADJACENT_TO,
                target_id="room-1",
                provenance=Provenance.OBSERVED,
            )
        )

        world_v1.entities["room-1"] = e_room1
        world_v1.geometries["geom-r1"] = geom_r1
        world_v1.entities["room-annex"] = e_annex
        world_v1.geometries["geom-annex-v1"] = geom_annex

        # 5. Save Version 1 to WorldStore
        ver_1 = store.save_version(
            world_v1,
            parent=None,
            version_id="v-golden-1",
            source_session_ids=[sess_1.session_id],
        )
        assert ver_1.version_id == "v-golden-1"
        assert ver_1.parent is None
        assert store.verify_version("v-golden-1") == []

        # 6. Spatial Query / Index verification
        tiles_v1 = SpatialTiles(world_v1, tile_size=4.0)
        # Point (2, 2, 1) belongs to tile containing room-1
        tile_00_entities = tiles_v1.tile_of((2.0, 2.0, 1.0))
        assert "room-1" in tile_00_entities
        assert "room-annex" not in tile_00_entities
        # Point (6, 2, 1) belongs to tile containing room-annex
        tile_10_entities = tiles_v1.tile_of((6.0, 2.0, 1.0))
        assert "room-annex" in tile_10_entities

        # 7. Desktop API bridge verification (Pass 1)
        out_1 = io.StringIO()
        monkeypatch.setattr(sys, "stdout", out_1)
        ret_load_1 = api_bridge.cmd_load_world("v-golden-1")
        assert ret_load_1 == 0
        loaded_desktop_1 = json.loads(out_1.getvalue().strip())
        assert loaded_desktop_1["version_id"] == "v-golden-1"
        assert any(e["id"] == "room-1" for e in loaded_desktop_1["entities"])
        assert any(e["id"] == "room-annex" for e in loaded_desktop_1["entities"])

        # =====================================================================
        # PASS 2: Localized Incremental Update (Capture 2 of Annex only)
        # =====================================================================

        # 1. Capture 2
        source_2 = EvidenceSource(source_id="device-iphone-1", platform="ios", device="iPhone 15 Pro")
        pkg_builder_2 = DeterministicPackageBuilder("pkg-sess-2")
        pkg_builder_2.register_source(source_2)
        payload_img_2 = b"\xff\xd8\xff\xe0" + b"REAL_SENSOR_PAYLOAD_ANNEX_RESCAN" + b"\xff\xd9"
        asset_id_2 = pkg_builder_2.add_payload(
            payload=payload_img_2,
            kind=EvidenceKind.PHOTO,
            source=source_2,
            source_uri="file:///captures/annex_rescan.jpg",
        )
        pkg_2 = pkg_builder_2.build()
        assert len(pkg_2.assets) == 1

        sess_2 = Session("sess-annex-update-002")
        item_2 = pkg_2.assets[asset_id_2].to_evidence_item()
        sess_2.add_evidence(item_2)

        # 2. Alignment: Align Session 2 cloud to Session 1
        cloud_sess1 = [Vec3(x, y, 1.0) for x in range(5, 8) for y in range(0, 4)]
        cloud_sess2 = [Vec3(x + 0.05, y + 0.02, 1.01) for x in range(5, 8) for y in range(0, 4)]
        align_report = align_session(cloud_sess2, cloud_sess1, from_session=sess_2.id, to_session=sess_1.session_id)
        assert align_report.status == "accepted"
        assert align_report.transform is not None

        # 3. Affected Closure computation on target change
        # Only room-annex is affected; room-1 must NOT be affected!
        closure = affected_closure(world_v1, ["room-annex"])
        assert "room-annex" in closure
        assert "room-1" not in closure

        # 4. Apply localized incremental update
        geom_annex_v2 = Geometry(
            id="geom-annex-v2",
            type=GeometryType.BOX,
            bounds_min=Vector3(5.0, 0.0, 0.0),
            bounds_max=Vector3(8.5, 4.5, 3.2),  # Expanded bounds
            provenance=Provenance.RECONSTRUCTED,
        )
        e_annex_v2 = Entity(
            id="room-annex",
            type=EntityType.ROOM,
            provenance=Provenance.RECONSTRUCTED,
            confidence=0.98,
            geometry_ids=["geom-annex-v2"],
        )
        e_annex_v2.transform = {"position": {"x": 6.75, "y": 2.25, "z": 1.6}}

        update_result = apply_incremental_update(
            base_world=world_v1,
            updated_entities=[e_annex_v2],
            updated_geometries=[geom_annex_v2],
        )
        world_v2 = update_result.new_world

        # CRITICAL VERIFICATION: Untouched room-1 maintains strict identity
        assert world_v2.entities["room-1"] is world_v1.entities["room-1"]
        assert world_v2.geometries["geom-r1"] is world_v1.geometries["geom-r1"]
        # Updated room-annex is updated
        assert world_v2.entities["room-annex"] is e_annex_v2
        assert "geom-annex-v2" in world_v2.geometries

        # 5. Save Version 2 to WorldStore with lineage to V1
        ver_2 = store.save_version(
            world_v2,
            parent="v-golden-1",
            version_id="v-golden-2",
            source_session_ids=[sess_2.id],
        )
        assert ver_2.version_id == "v-golden-2"
        assert ver_2.parent == "v-golden-1"
        assert store.parents("v-golden-2") == ["v-golden-1"]
        assert store.ancestors("v-golden-2") == ["v-golden-1"]
        assert store.verify_version("v-golden-2") == []

        # 6. WorldDiff verification
        diff = diff_worlds(world_v1, world_v2)
        diff_entity_ids = {d.entity_id for d in diff.entity_diffs}
        assert "room-annex" in diff_entity_ids
        assert "room-1" not in diff_entity_ids  # Room 1 completely absent from diff!

        # 7. Desktop API bridge verification (Pass 2 & Diff)
        out_diff = io.StringIO()
        monkeypatch.setattr(sys, "stdout", out_diff)
        ret_diff = api_bridge.cmd_diff("v-golden-1", "v-golden-2")
        assert ret_diff == 0
        diff_payload = json.loads(out_diff.getvalue().strip())
        assert diff_payload["from_version_id"] == "v-golden-1"
        assert diff_payload["to_version_id"] == "v-golden-2"
        changed_entities = [d["entity_id"] for d in diff_payload["entities"]]
        assert "room-annex" in changed_entities
        assert "room-1" not in changed_entities

        # =====================================================================
        # STEP 8: Export Pipeline & Readback Validation
        # =====================================================================

        # 1. glTF 2.0 Export & Readback
        gltf_out = export_to_gltf(world_v2, artifact_store=store._store)
        assert gltf_out["asset"]["version"] == "2.0"
        node_names = [n.get("name") for n in gltf_out.get("nodes", [])]
        assert "room-1" in node_names
        assert "room-annex" in node_names
        assert len(gltf_out.get("meshes", [])) >= 1
        assert len(gltf_out.get("buffers", [])) >= 1
        buf_uri = gltf_out["buffers"][0]["uri"]
        assert buf_uri.startswith("data:application/octet-stream;base64,")
        b64_data = buf_uri.split(",", 1)[1]
        decoded_bytes = base64.b64decode(b64_data)
        assert len(decoded_bytes) > 0

        # 2. CityGML 2.0 Export & Readback
        citygml_out = export_to_citygml(world_v2)
        assert "CityModel" in citygml_out
        root_elem = ET.fromstring(citygml_out)
        assert root_elem.tag.endswith("CityModel")
        assert "room-1" in citygml_out
        assert "room-annex" in citygml_out
        assert "Solid" in citygml_out

        # 3. USDA (Universal Scene Description) Export & Readback
        usda_out = export_to_usda(world_v2, artifact_store=store._store)
        assert usda_out.startswith("#usda 1.0")
        assert '"room_1"' in usda_out
        assert '"room_annex"' in usda_out

        # 4. CityJSON Export & Readback
        cityjson_out = export_to_cityjson(world_v2)
        assert cityjson_out["type"] == "CityJSON"
        assert cityjson_out["version"] == "1.1"
        assert "room-1" in cityjson_out["CityObjects"]
        assert "room-annex" in cityjson_out["CityObjects"]
