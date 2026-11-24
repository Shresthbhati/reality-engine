"""P0 regression tests: a persisted evidence-world surface must be resolvable by export.

Two independent defects are covered here.

1. The evidence-to-world flow writes geometry payloads into the RUN's Evidence Matrix store (``<run>/artifacts``) while a
   WorldStore keeps its own (``<run>/worldstore/artifacts``). A reader handed a single store loses the other's geometry, so
   ``evidence_world_stage.run_artifact_store`` returns a chain and export must resolve through it.
2. The surfaces stage stores its reconstructed mesh as binary PLY (Open3D marching cubes). The glTF exporter only understood
   the canonical ``MeshData`` container and therefore replaced a REAL surface with the placeholder unit cube.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from world_ir.artifact_store import (
    ArtifactNotFoundError,
    ChainedArtifactStore,
    FileArtifactStore,
    MemoryArtifactStore,
)
from world_ir.geometry_data import MeshData
from world_ir.schema_v1 import Geometry, GeometryType, Vector3


def _tetra_ply() -> tuple[bytes, int]:
    """A real 4-vertex triangle mesh as binary PLY (what Open3D writes), and its vertex count."""
    mesh = MeshData(
        vertices=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        faces=((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)),
    )
    payload = mesh.to_ply_bytes()
    assert payload.startswith(b"ply\n")
    return payload, len(mesh.vertices)


class TestChainedArtifactStore:
    def test_resolves_a_blob_held_by_the_second_store(self):
        first, second = MemoryArtifactStore(), MemoryArtifactStore()
        uri, digest = second.put(b"surface mesh bytes")
        chain = ChainedArtifactStore([first, second])
        assert chain.get(uri) == b"surface mesh bytes"
        assert chain.has(digest)

    def test_put_writes_to_the_first_store_only(self):
        first, second = MemoryArtifactStore(), MemoryArtifactStore()
        chain = ChainedArtifactStore([first, second])
        uri, digest = chain.put(b"new payload")
        assert first.has(digest)
        assert not second.has(digest)
        assert chain.get(uri) == b"new payload"

    def test_missing_blob_raises_rather_than_returning_empty_bytes(self):
        chain = ChainedArtifactStore([MemoryArtifactStore(), MemoryArtifactStore()])
        with pytest.raises(ArtifactNotFoundError):
            chain.get("artifact://" + "0" * 64)

    def test_rejects_an_empty_chain(self):
        with pytest.raises(ValueError):
            ChainedArtifactStore([])


class TestRunArtifactStore:
    def test_resolves_geometry_from_the_run_matrix_store(self, tmp_path: Path):
        from apps.api.evidence_world_stage import run_artifact_store

        run = tmp_path / "run"
        (run / "artifacts").mkdir(parents=True)
        uri, _ = FileArtifactStore(run / "artifacts").put(b"fused tsdf ply")
        assert run_artifact_store(run).get(uri) == b"fused tsdf ply"

    def test_resolves_geometry_from_the_run_worldstore_too(self, tmp_path: Path):
        from apps.api.evidence_world_stage import run_artifact_store

        run = tmp_path / "run"
        (run / "worldstore" / "artifacts").mkdir(parents=True)
        uri, _ = FileArtifactStore(run / "worldstore" / "artifacts").put(b"worldstore payload")
        assert run_artifact_store(run).get(uri) == b"worldstore payload"


class TestPlyMeshExport:
    def test_gltf_resolves_a_real_ply_mesh_instead_of_the_placeholder_cube(self):
        from exporters.gltf.exporter import _resolve_real_mesh

        payload, n_vertices = _tetra_ply()
        store = MemoryArtifactStore()
        uri, _ = store.put(payload)
        geometry = Geometry(
            id="geom-fused-surface", type=GeometryType.MESH, lod_level=0, vertex_count=n_vertices, triangle_count=4,
            data_uri=uri, bounds_min=Vector3(0.0, 0.0, 0.0), bounds_max=Vector3(1.0, 1.0, 1.0),
        )
        resolved = _resolve_real_mesh(store, geometry)
        assert resolved is not None, "a real PLY surface must not fall through to the placeholder cube"
        vertices, faces = resolved[0], resolved[1]
        assert len(vertices) == n_vertices
        assert len(faces) == 4

    def test_gltf_still_returns_none_for_a_payload_it_cannot_read(self):
        from exporters.gltf.exporter import _resolve_real_mesh

        store = MemoryArtifactStore()
        uri, _ = store.put(b"not a mesh at all")
        geometry = Geometry(
            id="geom-bad", type=GeometryType.MESH, lod_level=0, vertex_count=1, triangle_count=0,
            data_uri=uri, bounds_min=Vector3(0.0, 0.0, 0.0), bounds_max=Vector3(1.0, 1.0, 1.0),
        )
        assert _resolve_real_mesh(store, geometry) is None
