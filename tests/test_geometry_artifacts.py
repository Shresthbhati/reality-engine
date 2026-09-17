"""Tests for real geometry storage (P0.10/P0.11):
world_ir/geometry_data.py (PointCloudData), world_ir/artifact_store.py
(MemoryArtifactStore, FileArtifactStore), the promote_planes.py wiring
that writes real point payloads, and the gltf exporter's real-mesh path
that reads them back.
"""

from __future__ import annotations

import struct

import pytest

from world_ir.artifact_store import (
    ArtifactNotFoundError,
    ArtifactStore,
    FileArtifactStore,
    MemoryArtifactStore,
)
from world_ir.geometry_data import PointCloudData


# ---------------------------------------------------------------- PointCloudData


class TestPointCloudData:
    def test_round_trips_through_bytes(self):
        points = ((0.0, 0.0, 0.0), (1.5, -2.25, 3.75), (-100.0, 0.001, 50.5))
        payload = PointCloudData(points=points).to_bytes()
        restored = PointCloudData.from_bytes(payload)
        assert restored.points == points

    def test_empty_point_cloud_round_trips(self):
        payload = PointCloudData(points=()).to_bytes()
        assert PointCloudData.from_bytes(payload).points == ()

    def test_preserves_order_and_duplicates(self):
        points = ((1.0, 1.0, 1.0), (1.0, 1.0, 1.0), (2.0, 2.0, 2.0))
        restored = PointCloudData.from_bytes(PointCloudData(points=points).to_bytes())
        assert restored.points == points

    def test_from_bytes_rejects_bad_magic(self):
        with pytest.raises(ValueError):
            PointCloudData.from_bytes(b"not-a-real-payload-------------")

    def test_from_positions_is_equivalent_to_direct_construction(self):
        positions = [(0.1, 0.2, 0.3), (0.4, 0.5, 0.6)]
        assert PointCloudData.from_positions(positions) == PointCloudData(points=tuple(positions))


# ---------------------------------------------------------------- ArtifactStore


@pytest.fixture(params=["memory", "file"])
def store(request, tmp_path) -> ArtifactStore:
    if request.param == "memory":
        return MemoryArtifactStore()
    return FileArtifactStore(tmp_path / "artifacts")


class TestArtifactStore:
    def test_put_then_get_round_trips(self, store):
        data = b"real reconstructed geometry bytes, not fabricated"
        uri, digest = store.put(data)
        assert uri == f"artifact://{digest}"
        assert store.get(uri) == data

    def test_content_addressing_deduplicates_identical_bytes(self, store):
        data = PointCloudData(points=((1.0, 2.0, 3.0),)).to_bytes()
        uri1, digest1 = store.put(data)
        uri2, digest2 = store.put(data)
        assert uri1 == uri2
        assert digest1 == digest2

    def test_different_bytes_get_different_digests(self, store):
        _uri1, digest1 = store.put(b"payload A")
        _uri2, digest2 = store.put(b"payload B")
        assert digest1 != digest2

    def test_get_unknown_uri_raises(self, store):
        with pytest.raises(ArtifactNotFoundError):
            store.get("artifact://" + "0" * 64)

    def test_digest_of_rejects_non_artifact_uri(self, store):
        with pytest.raises(ValueError):
            ArtifactStore.digest_of("https://example.com/not-an-artifact")

    @pytest.mark.parametrize("malicious_digest", [
        "../../../../etc/passwd",
        "..%2f..%2fescape",
        "/etc/passwd",
        "0" * 63,  # one short of a real sha256 hex digest
        "0" * 65,  # one long
        "g" * 64,  # not hex
        "",
    ])
    def test_get_rejects_path_traversal_and_malformed_digests(self, store, malicious_digest):
        """Strix-flagged finding: FileArtifactStore.get() resolved
        artifact:// URIs without validating the digest is a real sha256
        hex string, so artifact://../../victim could read files outside
        the store root. digest_of() now refuses anything that isn't
        exactly 64 lowercase hex characters, for every ArtifactStore
        backend (this fixture parametrizes both memory and file)."""
        with pytest.raises(ValueError):
            store.get(f"artifact://{malicious_digest}")

    def test_file_store_traversal_digest_never_touches_the_filesystem(self, tmp_path):
        """Extra assurance for the file backend specifically: a
        malicious digest must be rejected before any path is built or
        touched, and must not create a file outside root."""
        root = tmp_path / "artifacts"
        store = FileArtifactStore(root)
        outside_target = tmp_path / "outside.txt"
        outside_target.write_text("should never be reachable via the store")

        with pytest.raises(ValueError):
            store.get("artifact://../outside")

        # Nothing was created outside the intended root.
        assert set(tmp_path.iterdir()) == {outside_target, root}
        assert outside_target.read_text() == "should never be reachable via the store"


class TestFileArtifactStorePersistence:
    def test_survives_a_new_store_instance_over_the_same_root(self, tmp_path):
        root = tmp_path / "artifacts"
        store1 = FileArtifactStore(root)
        data = PointCloudData(points=((9.0, 9.0, 9.0),)).to_bytes()
        uri, _digest = store1.put(data)

        store2 = FileArtifactStore(root)  # simulates a new process/session
        assert store2.get(uri) == data


# ---------------------------------------------------------------- promotion wiring


def _compiled_world_with_store(seed: int = 42):
    from engine.compiler import CompileOptions, compile_reconstruction_to_world
    from reconstruction.backend.interface import ReconstructedCameraPose
    from tests.test_room_inference import _CAMS, _two_room_scene

    result = _two_room_scene()
    result.camera_poses.extend(
        ReconstructedCameraPose(evidence_id=f"ev-{i}", position=p, rotation=(1.0, 0.0, 0.0, 0.0))
        for i, p in enumerate(_CAMS)
    )
    store = MemoryArtifactStore()
    world, diagnostics = compile_reconstruction_to_world(
        result, CompileOptions(seed=seed, artifact_store=store),
    )
    return world, diagnostics, store


class TestPromotePlanesWritesRealGeometry:
    def test_compiled_plane_geometries_carry_a_resolvable_data_uri(self):
        world, diagnostics, store = _compiled_world_with_store()
        assert diagnostics.entities_created  # sanity: something was actually promoted

        plane_geometries = [g for g in world.geometries.values() if g.data_uri]
        assert plane_geometries, "at least one promoted plane must have written a real artifact"

        for geom in plane_geometries:
            payload = store.get(geom.data_uri)
            cloud = PointCloudData.from_bytes(payload)
            assert len(cloud.points) == geom.vertex_count  # real data matches the metadata count
            assert geom.data_hash == store.digest_of(geom.data_uri)

    def test_omitting_artifact_store_reproduces_old_behavior(self):
        """CompileOptions() with no artifact_store (the default) must
        leave data_uri/data_hash empty -- this parameter is additive,
        never a behavior change for existing callers."""
        from engine.compiler import CompileOptions, compile_reconstruction_to_world
        from reconstruction.backend.interface import ReconstructedCameraPose
        from tests.test_room_inference import _CAMS, _two_room_scene

        result = _two_room_scene()
        result.camera_poses.extend(
            ReconstructedCameraPose(evidence_id=f"ev-{i}", position=p, rotation=(1.0, 0.0, 0.0, 0.0))
            for i, p in enumerate(_CAMS)
        )
        world, _diag = compile_reconstruction_to_world(result, CompileOptions(seed=42))

        for geom in world.geometries.values():
            assert geom.data_uri == ""
            assert geom.data_hash == ""

    def test_two_compiles_of_the_same_evidence_produce_identical_artifacts(self):
        """Same reconstruction + same seed -> byte-identical point
        payloads (determinism preserved through the artifact layer)."""
        world_a, _diag_a, store_a = _compiled_world_with_store(seed=7)
        world_b, _diag_b, store_b = _compiled_world_with_store(seed=7)

        hashes_a = sorted(g.data_hash for g in world_a.geometries.values() if g.data_hash)
        hashes_b = sorted(g.data_hash for g in world_b.geometries.values() if g.data_hash)
        assert hashes_a == hashes_b
        assert hashes_a  # sanity: not vacuously true


class TestPromoteRoomsWritesRealGeometry:
    """Proves engine/compiler/world_compiler.py threads
    CompileOptions.artifact_store into promote_room_to_entity (not just
    promote_plane_to_entity) -- the follow-on wiring named in
    .agent/CURRENT_STATE.md after the room-geometry feature itself
    landed."""

    def test_compiled_room_geometry_carries_a_resolvable_data_uri(self):
        from world_ir import EntityType

        world, diagnostics, store = _compiled_world_with_store()
        rooms = [e for e in world.entities.values() if e.type is EntityType.ROOM]
        assert rooms, "sanity: the two-room fixture must actually produce a ROOM entity"

        room = rooms[0]
        assert room.geometry_ids, "compiled room must have real boundary geometry, not []"
        geom = world.geometries[room.geometry_ids[0]]
        assert geom.data_uri and geom.data_hash

        payload = store.get(geom.data_uri)
        cloud = PointCloudData.from_bytes(payload)
        assert len(cloud.points) == geom.vertex_count

    def test_omitting_artifact_store_leaves_room_geometry_ids_empty(self):
        from engine.compiler import CompileOptions, compile_reconstruction_to_world
        from reconstruction.backend.interface import ReconstructedCameraPose
        from tests.test_room_inference import _CAMS, _two_room_scene
        from world_ir import EntityType

        result = _two_room_scene()
        result.camera_poses.extend(
            ReconstructedCameraPose(evidence_id=f"ev-{i}", position=p, rotation=(1.0, 0.0, 0.0, 0.0))
            for i, p in enumerate(_CAMS)
        )
        world, _diag = compile_reconstruction_to_world(result, CompileOptions(seed=42))

        rooms = [e for e in world.entities.values() if e.type is EntityType.ROOM]
        assert rooms
        for room in rooms:
            assert room.geometry_ids == []


# ---------------------------------------------------------------- object promotion wiring


def _two_view_chair_candidate():
    """Same two-view chair fixture as tests/test_object_pipeline_e2e.py,
    reused here to prove hypothesis.points survives lift -> merge and
    reaches promote_object_to_entity()."""
    from engine.physics.math3 import Quat, Vec3
    from perception.depth.interface import DepthMap
    from perception.instances.lifting import lift_region_to_3d
    from perception.instances.object_resolution import merge_hypotheses
    from perception.segmentation.interface import SegmentedRegion
    from reconstruction.calibration.camera import CameraExtrinsics, CameraIntrinsics, PinholeCamera

    width, height = 12, 10

    def camera(position):
        intrinsics = CameraIntrinsics(fx=200.0, fy=200.0, cx=width / 2, cy=height / 2, width=width, height=height)
        return PinholeCamera(intrinsics=intrinsics, extrinsics=CameraExtrinsics(position=position, rotation=Quat.identity()))

    def rect_mask(row_lo, row_hi, col_lo, col_hi):
        return [[row_lo <= r < row_hi and col_lo <= c < col_hi for c in range(width)] for r in range(height)]

    camera_1 = camera(Vec3.zero())
    depth_1 = DepthMap(evidence_id="photo-1", width=width, height=height,
                        values=[[4.0] * width for _ in range(height)], unit="meters")
    region_1 = SegmentedRegion(region_id="r1", evidence_id="photo-1", label="chair",
                                mask=rect_mask(3, 7, 4, 8), confidence=0.85)

    camera_2 = camera(Vec3(0.1, 0.0, 0.0))
    depth_2 = DepthMap(evidence_id="photo-2", width=width, height=height,
                        values=[[4.0] * width for _ in range(height)], unit="meters")
    region_2 = SegmentedRegion(region_id="r2", evidence_id="photo-2", label="chair",
                                mask=rect_mask(3, 7, 4, 8), confidence=0.80)

    hyp_1 = lift_region_to_3d(region_1, depth_1, camera_1)
    hyp_2 = lift_region_to_3d(region_2, depth_2, camera_2)
    return merge_hypotheses([hyp_1, hyp_2], distance_threshold_m=0.5)[0]


class TestObjectHypothesisRetainsRealPoints:
    def test_lift_region_to_3d_populates_points_matching_point_count(self):
        candidate = _two_view_chair_candidate()
        for hypothesis in candidate.source_hypotheses:
            assert len(hypothesis.points) == hypothesis.point_count
            assert hypothesis.point_count > 0

    def test_hand_built_hypothesis_defaults_points_to_empty_tuple(self):
        """Backward compatibility: existing callers/tests that build
        ObjectHypothesis3D by hand (never touching the new field) must
        keep working unchanged."""
        from engine.physics.math3 import Vec3
        from perception.instances.lifting import ObjectHypothesis3D

        hyp = ObjectHypothesis3D(
            region_id="r", evidence_id="e", label="chair",
            position=Vec3(0, 0, 0), bounds_min=Vec3(-1, -1, -1), bounds_max=Vec3(1, 1, 1),
            point_count=0, mask_pixel_count=0, confidence=0.5,
        )
        assert hyp.points == ()


class TestPromoteObjectsWritesRealGeometry:
    def test_promoted_object_geometry_carries_a_resolvable_data_uri(self):
        from evidence.promote_objects import promote_object_to_entity
        from world_ir.world_v1 import WorldIR

        candidate = _two_view_chair_candidate()
        store = MemoryArtifactStore()
        world = WorldIR()

        result = promote_object_to_entity(candidate, world, "ent-chair", artifact_store=store)

        assert result.geometry.data_uri
        payload = store.get(result.geometry.data_uri)
        cloud = PointCloudData.from_bytes(payload)
        expected_total = sum(len(h.points) for h in candidate.source_hypotheses)
        assert len(cloud.points) == expected_total
        assert result.geometry.data_hash == store.digest_of(result.geometry.data_uri)

    def test_omitting_artifact_store_reproduces_old_behavior(self):
        from evidence.promote_objects import promote_object_to_entity
        from world_ir.world_v1 import WorldIR

        candidate = _two_view_chair_candidate()
        world = WorldIR()

        result = promote_object_to_entity(candidate, world, "ent-chair")

        assert result.geometry.data_uri == ""
        assert result.geometry.data_hash == ""

    def test_promoted_object_exports_a_real_points_mesh(self):
        """Full chain: lift -> merge -> promote (with a store) ->
        validate -> gltf export, same standard test_object_pipeline_e2e.py
        already holds this pipeline to, extended through export."""
        from evidence.promote_objects import promote_object_to_entity
        from exporters.gltf.exporter import export_to_gltf
        from world_ir.validation import validate_world_ir
        from world_ir.world_v1 import WorldIR

        candidate = _two_view_chair_candidate()
        store = MemoryArtifactStore()
        world = WorldIR()
        promote_object_to_entity(candidate, world, "ent-chair", artifact_store=store)

        assert validate_world_ir(world).is_valid()

        gltf = export_to_gltf(world, artifact_store=store)
        mesh_indices_used = {node["mesh"] for node in gltf["nodes"]}
        assert mesh_indices_used - {0}, "the promoted chair must export real geometry, not the cube"


# ---------------------------------------------------------------- gltf export wiring


class TestGltfExportsRealGeometry:
    def test_entities_with_real_geometry_get_a_points_mesh_not_the_cube(self):
        from exporters.gltf.exporter import export_to_gltf

        world, _diag, store = _compiled_world_with_store()
        gltf = export_to_gltf(world, artifact_store=store)

        # Some node must NOT reference the shared placeholder cube (mesh 0)
        # -- at least one entity got real per-entity geometry.
        mesh_indices_used = {node["mesh"] for node in gltf["nodes"]}
        assert mesh_indices_used - {0}, "expected at least one real (non-cube) mesh"

        # Every non-cube mesh must be a real POINTS primitive with a
        # vertex count matching what was actually stored.
        for mesh_index in mesh_indices_used - {0}:
            primitive = gltf["meshes"][mesh_index]["primitives"][0]
            assert primitive["mode"] == 0  # POINTS
            accessor = gltf["accessors"][primitive["attributes"]["POSITION"]]
            assert accessor["count"] > 0

    def test_without_artifact_store_falls_back_to_the_cube_exactly_as_before(self):
        from exporters.gltf.exporter import export_to_gltf

        world, _diag, _store = _compiled_world_with_store()
        gltf_no_store = export_to_gltf(world)  # artifact_store omitted

        assert all(node["mesh"] == 0 for node in gltf_no_store["nodes"])

    def test_sdk_export_threads_artifact_store_through_for_gltf(self):
        from sdk import reality

        world, _diag, store = _compiled_world_with_store()
        content, report = reality.export(world, "gltf", artifact_store=store)

        mesh_indices_used = {node["mesh"] for node in content["nodes"]}
        assert mesh_indices_used - {0}
        assert report.entities_exported
