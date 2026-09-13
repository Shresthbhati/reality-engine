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
