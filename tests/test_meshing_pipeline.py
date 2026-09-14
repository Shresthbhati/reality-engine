"""Regression tests for the dense-geometry -> mesh -> WorldIR leg
(mapping campaign P0.11-P0.13).

Deterministic coverage (no COLMAP, no torch):

  - reconstruction/meshing/mesh.py      (MeshData: binary + PLY roundtrips,
                                        face validation, header-driven face
                                        parsing incl. COLMAP's `list int int`)
  - reconstruction/meshing/preprocess.py (voxel downsample determinism,
                                        outlier filter, camera-oriented PCA
                                        normals)
  - reconstruction/meshing/surface.py   (honest unavailability errors)
  - engine/pipeline/vertical_slice.py   (_mesh_stage skip paths + promoted
                                        artifact roundtrip with a fake
                                        mesher; the real COLMAP path is a
                                        dependency-gated test)
  - exporters/gltf/exporter.py          (real MESH triangle primitive)

The real COLMAP poisson_mesher path is exercised in
`test_poisson_real_backend` (skipped with a reason when the configured
COLMAP binary lacks the command) -- never faked here.
"""

from __future__ import annotations

import struct
import types
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest

from reconstruction.meshing import MeshData, mesh_summary
from reconstruction.meshing.preprocess import (
    PreprocessError,
    estimate_oriented_normals,
    statistical_outlier_filter,
    voxel_downsample,
)
from reconstruction.meshing.surface import (
    MeshingUnavailableError,
    reconstruct_surface,
)

TRI = Tuple[int, int, int]
VERT = Tuple[float, float, float]

PYRAMID_V: List[VERT] = [
    (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0), (0.5, 0.5, 1.0),
]
PYRAMID_F: List[TRI] = [
    (0, 1, 2), (0, 2, 3), (0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4),
]


# ---------------------------------------------------------------- MeshData

class TestMeshData:
    def test_binary_roundtrip_is_byte_identical(self):
        m = MeshData(vertices=tuple(PYRAMID_V), faces=tuple(PYRAMID_F))
        m2 = MeshData.from_bytes(m.to_bytes())
        assert m2 == m
        assert m.to_bytes() == m2.to_bytes()

    def test_binary_roundtrip_with_normals(self):
        m = MeshData(
            vertices=tuple(PYRAMID_V), faces=tuple(PYRAMID_F),
            normals=((0.0, 0.0, -1.0),) * 5,
        )
        m2 = MeshData.from_bytes(m.to_bytes())
        assert m2.normals == ((0.0, 0.0, -1.0),) * 5

    def test_bad_magic_rejected(self):
        with pytest.raises(ValueError, match="not a MeshData payload"):
            MeshData.from_bytes(b"NOPE0000" + b"\x00" * 20)

    def test_face_index_validation(self):
        with pytest.raises(ValueError, match="references vertex 6"):
            MeshData(
                vertices=tuple(PYRAMID_V),
                faces=((0, 1, 6),),  # 6 out of range
            )

    def test_normals_length_validation(self):
        with pytest.raises(ValueError, match="per-vertex"):
            MeshData(
                vertices=tuple(PYRAMID_V), faces=tuple(PYRAMID_F),
                normals=((0, 0, 1),),  # 1 normal, 5 vertices
            )

    def test_ply_roundtrip_ours(self):
        m = MeshData(vertices=tuple(PYRAMID_V), faces=tuple(PYRAMID_F))
        p = MeshData.from_ply_bytes(m.to_ply_bytes())
        assert list(p.vertices) == PYRAMID_V
        assert list(p.faces) == PYRAMID_F

    def test_ply_reads_colmap_style_list_int_header(self):
        """COLMAP's mesher writes `property list int int vertex_indices`
        (4-byte count). The reader must follow the header, not assume a
        uchar count -- observed as garbage-face errors otherwise."""
        header = (
            "ply\nformat binary_little_endian 1.0\n"
            "element vertex 3\n"
            "property float x\nproperty float y\nproperty float z\n"
            "element face 1\n"
            "property list int int vertex_indices\n"
            "end_header\n"
        ).encode("ascii")
        body = (
            struct.pack("<3f", 0, 0, 0)
            + struct.pack("<3f", 1, 0, 0)
            + struct.pack("<3f", 0, 1, 0)
            + struct.pack("<i", 3)          # int32 count (COLMAP style)
            + struct.pack("<3I", 0, 1, 2)
        )
        m = MeshData.from_ply_bytes(header + body)
        assert m.vertices == ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
        assert m.faces == ((0, 1, 2),)

    def test_ply_with_normals_and_colors_roundtrip(self):
        m = MeshData(
            vertices=tuple(PYRAMID_V), faces=tuple(PYRAMID_F),
            normals=((0.0, -1.0, 0.0),) * 5,
        )
        p = MeshData.from_ply_bytes(
            m.to_ply_bytes(with_colors=[(0.5, 0.5, 0.5)] * 5)
        )
        assert p.normals == ((0.0, -1.0, 0.0),) * 5
        assert p.vertices == tuple(PYRAMID_V)

    def test_summary_reports_watertight_without_enforcing(self):
        open_mesh = MeshData(vertices=tuple(PYRAMID_V), faces=((0, 1, 2),))
        s = mesh_summary(open_mesh)
        assert s["watertight"] is False  # reported, never enforced
        closed = MeshData(vertices=tuple(PYRAMID_V), faces=tuple(PYRAMID_F))
        assert mesh_summary(closed)["watertight"] is True


# ------------------------------------------------------------ preprocessing

class TestPreprocess:
    def _cloud(self, n=4000, seed=42):
        rng = np.random.default_rng(seed)
        pts = rng.normal(0, 0.05, size=(n, 3)) + np.array([0.0, 0.0, 0.5])
        return [tuple(map(float, p)) for p in pts]

    def test_voxel_downsample_is_deterministic(self):
        pts = self._cloud()
        a, _ = voxel_downsample(pts, 0.02)
        b, _ = voxel_downsample(pts, 0.02)
        assert a == b
        assert 0 < len(a) < len(pts)

    def test_voxel_downsample_rejects_bad_size(self):
        with pytest.raises(PreprocessError):
            voxel_downsample(self._cloud(10), 0.0)

    def test_outlier_filter_removes_gross_outliers(self):
        pts = self._cloud(3000)
        pts.append((50.0, 50.0, 50.0))  # one gross outlier
        kept, facts = statistical_outlier_filter(pts)
        assert len(kept) == facts["kept"]
        assert facts["removed"] >= 1
        assert (50.0, 50.0, 50.0) not in kept

    def test_normals_point_toward_observing_camera(self):
        pts = self._cloud(1000)
        normals = estimate_oriented_normals(pts[:300], [(0.0, 0.0, -2.0)])
        # cloud is a blob around z=0.5, camera far below -> normals point down
        downward = sum(1 for n in normals if n[2] < 0)
        assert downward > 280

    def test_normals_require_camera_centers(self):
        with pytest.raises(PreprocessError, match="camera_centers"):
            estimate_oriented_normals(self._cloud(10), [])

    def test_normals_require_finite_cameras(self):
        with pytest.raises(PreprocessError, match="non-finite"):
            estimate_oriented_normals(self._cloud(10), [(float("nan"), 0, 0)])


# ----------------------------------------------------------------- surface

class TestSurfaceUnavailability:
    def test_missing_binary_raises_unavailable(self):
        with pytest.raises(MeshingUnavailableError, match="not found on PATH"):
            reconstruct_surface(
                [(0, 0, 0)], [(0, 0, 1)],
                colmap_binary="definitely-not-a-real-colmap-binary-xyz",
            )

    def test_length_mismatch_raises(self):
        with pytest.raises(MeshingError_cls := __import__(
            "reconstruction.meshing.surface", fromlist=["MeshingError"]
        ).MeshingError, match="1:1"):
            reconstruct_surface([(0, 0, 0), (1, 1, 1)], [(0, 0, 1)])


# ----------------------------------------------------------- pipeline stage

def _fake_result(points: List[VERT], centers: List[VERT]):
    return types.SimpleNamespace(
        points=[types.SimpleNamespace(position=p) for p in points],
        camera_poses=[types.SimpleNamespace(position=c) for c in centers],
    )


def _metric_world():
    from world_ir.world_v1 import WorldIR

    w = WorldIR(id="w-test")
    w.metadata["scale"] = {"state": "metric", "meters_per_unit": 1.0}
    return w


def _plane_cloud(n=4000, seed=7):
    rng = np.random.default_rng(seed)
    pts = [
        (float(x), float(y), float(0.004 * rng.standard_normal()))
        for x, y in rng.uniform(-1, 1, size=(n, 2))
    ]
    return pts


class TestMeshStage:
    def _options(self, **overrides):
        from engine.pipeline.vertical_slice import VerticalSliceOptions
        from world_ir.artifact_store import MemoryArtifactStore

        defaults = dict(
            artifact_store=MemoryArtifactStore(),
            mesh_enabled=True,
            mesh_voxel_size_m=0.02,
            mesh_poisson_depth=6,
            colmap_binary="unused-by-fake-mesher",
        )
        defaults.update(overrides)
        return VerticalSliceOptions(**defaults)

    def _stage(self, world, result, options, monkeypatch, mesh=None):
        import engine.pipeline.vertical_slice as vs
        from reconstruction.meshing import MeshData as MD

        if mesh is None:
            mesh = MD(vertices=tuple(PYRAMID_V), faces=tuple(PYRAMID_F))
        monkeypatch.setattr(
            "reconstruction.meshing.surface.reconstruct_surface",
            lambda *a, **k: mesh,
        )
        monkeypatch.setattr(
            "reconstruction.meshing.surface.poisson_mesher_available",
            lambda binary: True,
        )
        return vs._mesh_stage(result, world, options)

    def test_disabled_skips(self):
        import engine.pipeline.vertical_slice as vs
        from engine.pipeline.vertical_slice import VerticalSliceOptions

        world = _metric_world()
        options = VerticalSliceOptions(mesh_enabled=False)
        facts = vs._mesh_stage(
            _fake_result(_plane_cloud(200), [(0, 0, 2)]), world, options
        )
        assert facts["status"] == "skipped"
        assert "disabled" in facts["note"]

    def test_no_store_skips(self):
        import engine.pipeline.vertical_slice as vs
        from engine.pipeline.vertical_slice import VerticalSliceOptions

        world = _metric_world()
        options = VerticalSliceOptions(artifact_store=None)
        facts = vs._mesh_stage(
            _fake_result(_plane_cloud(200), [(0, 0, 2)]), world, options
        )
        assert facts["status"] == "skipped"
        assert "artifact_store" in facts["note"]

    def test_relative_world_skips(self):
        import engine.pipeline.vertical_slice as vs
        from engine.pipeline.vertical_slice import VerticalSliceOptions
        from world_ir.artifact_store import MemoryArtifactStore

        world = _metric_world()
        world.metadata["scale"]["state"] = "relative"
        options = VerticalSliceOptions(artifact_store=MemoryArtifactStore())
        facts = vs._mesh_stage(
            _fake_result(_plane_cloud(200), [(0, 0, 2)]), world, options
        )
        assert facts["status"] == "skipped"
        assert "METRIC" in facts["note"].upper() or "metric" in facts["note"]

    def test_unavailable_colmap_skips_honestly(self, monkeypatch):
        import engine.pipeline.vertical_slice as vs
        from world_ir.artifact_store import MemoryArtifactStore

        world = _metric_world()
        options = self._options()
        monkeypatch.setattr(
            "reconstruction.meshing.surface.poisson_mesher_available",
            lambda binary: False,
        )
        facts = vs._mesh_stage(
            _fake_result(_plane_cloud(200), [(0, 0, 2)]), world, options
        )
        assert facts["status"] == "skipped"
        assert "poisson_mesher" in facts["note"]

    def test_meshed_world_carries_real_artifact(self, monkeypatch):
        from reconstruction.meshing import MeshData as MD
        from world_ir.schema_v1 import GeometryType
        from world_ir.geometry_data import PointCloudData  # different payload

        world = _metric_world()
        options = self._options()
        result = _fake_result(_plane_cloud(3000), [(0, 0, 2)])
        facts = self._stage(world, result, options, monkeypatch)

        assert facts["status"] == "ran"
        assert facts["vertices"] == len(PYRAMID_V)
        assert world.geometries["geom-mesh-room"].type is GeometryType.MESH
        geometry = world.geometries["geom-mesh-room"]
        assert geometry.data_uri.startswith("artifact://")
        # the artifact bytes decode to the mesh the fake mesher returned
        from reconstruction.meshing.mesh import MeshData as MD2

        mesh = MD2.from_bytes(options.artifact_store.get(geometry.data_uri))
        assert mesh.vertices == tuple(PYRAMID_V)
        assert "entity-mesh-room" in world.entities
        assert geometry.data_hash == geometry.data_uri.split("//")[1]

    def test_too_few_points_skips(self):
        import engine.pipeline.vertical_slice as vs
        from engine.pipeline.vertical_slice import VerticalSliceOptions
        from world_ir.artifact_store import MemoryArtifactStore

        world = _metric_world()
        options = VerticalSliceOptions(artifact_store=MemoryArtifactStore())
        facts = vs._mesh_stage(
            _fake_result(_plane_cloud(50), [(0, 0, 2)]), world, options
        )
        assert facts["status"] == "skipped"
        assert "too few" in facts["note"]


# ---------------------------------------------------------------- exporter

class TestGltfMeshExport:
    def _mesh_world(self):
        from provenance import Provenance
        from world_ir.artifact_store import MemoryArtifactStore
        from world_ir.schema_v1 import Entity, EntityType, Geometry, GeometryType, Vector3
        from world_ir.world_v1 import WorldIR

        mesh = MeshData(vertices=tuple(PYRAMID_V), faces=tuple(PYRAMID_F))
        store = MemoryArtifactStore()
        uri, digest = store.put(mesh.to_bytes())

        world = WorldIR(id="w-test")
        world.geometries["geom-m1"] = Geometry(
            id="geom-m1", type=GeometryType.MESH, vertex_count=5, triangle_count=6,
            data_uri=uri, data_hash=digest,
            bounds_min=Vector3(x=0, y=0, z=0), bounds_max=Vector3(x=1, y=1, z=1),
            provenance=Provenance.RECONSTRUCTED,
        )
        world.entities["ent-m1"] = Entity(
            id="ent-m1", type=EntityType.STRUCTURE, name="mesh room",
            transform={"position": {"x": 0.5, "y": 0.5, "z": 0.5}},
            geometry_ids=["geom-m1"], provenance=Provenance.RECONSTRUCTED,
        )
        return world, store

    def test_mesh_payload_becomes_triangles_primitive(self):
        from exporters.gltf.exporter import export_to_gltf

        world, store = self._mesh_world()
        gltf = export_to_gltf(world, artifact_store=store)
        assert len(gltf["meshes"]) == 2  # shared cube + real mesh
        prim = gltf["meshes"][-1]["primitives"][0]
        assert prim["mode"] == 4  # TRIANGLES
        assert "indices" in prim
        pos_acc = gltf["accessors"][prim["attributes"]["POSITION"]]
        assert pos_acc["count"] == 5
        assert pos_acc["min"] == [-0.5, -0.5, -0.5]  # translated to local
        assert gltf["accessors"][prim["indices"]]["count"] == 18

    def test_without_store_falls_back_to_cube(self):
        from exporters.gltf.exporter import export_to_gltf

        world, _ = self._mesh_world()
        gltf = export_to_gltf(world)
        assert len(gltf["meshes"]) == 1  # placeholder cube only


# ------------------------------------------------- real backend (gated)

def _colmap_binary() -> str:
    import os

    return os.environ.get("REALITY_COLMAP", "colmap")


def _poisson_available() -> bool:
    try:
        from reconstruction.meshing.surface import poisson_mesher_available

        return poisson_mesher_available(_colmap_binary())
    except Exception:
        return False


@pytest.mark.skipif(
    not _poisson_available(),
    reason="COLMAP binary with poisson_mesher not available "
    "(set REALITY_COLMAP to a full COLMAP build to run)",
)
class TestPoissonRealBackend:
    def test_plane_cloud_years_a_mesh(self):
        rng = np.random.default_rng(7)
        pts = [
            (float(x), float(y), float(0.005 * rng.standard_normal()))
            for x, y in rng.uniform(-1, 1, size=(3000, 2))
        ]
        normals = [(0.0, 0.0, 1.0)] * len(pts)
        mesh = reconstruct_surface(
            pts, normals, colmap_binary=_colmap_binary(), depth=8
        )
        assert len(mesh.faces) > 100
        (mnx, mny, _), (mxx, mxy, _) = mesh.bounds()
        # 2 m plane, Poisson inflates slightly: extents within 25%
        assert 1.5 <= (mxx - mnx) <= 2.6
        assert 1.5 <= (mxy - mny) <= 2.6
        # and the output must roundtrip through our own PLY writer
        rt = MeshData.from_ply_bytes(mesh.to_ply_bytes())
        assert rt.vertices == mesh.vertices and rt.faces == mesh.faces
