"""T-2.4: Plain mesh export for SketchUp/3ds Max/Blender workflows (OBJ/DAE/GLB).

Verifies:
- Metric scale (1.0 meter = 1 unit, bounds match WorldIR dimensions)
- Per-object nodes (<node name="..."> in DAE, o <name> in OBJ, nodes named in GLB)
- Materials named by surface (surface_wall, surface_floor, surface_window, etc. or authored materials)
- Real mesh resolution from ArtifactStore when available, falling back to bounds
- Clean loading and scale verification via trimesh and XML schema parsing
"""

import io
import xml.etree.ElementTree as ET
import pytest
import trimesh

from exporters.gltf.exporter import export_to_glb, export_to_gltf
from exporters.mesh.dae import export_to_dae, export_to_dae_with_report
from exporters.mesh.obj import export_to_obj, export_to_obj_with_report
from sdk import reality
from tests.test_floorplan_export import _world
from world_ir.artifact_store import MemoryArtifactStore
from world_ir.geometry_data import MeshData
from world_ir.schema_v1 import Entity, Geometry, GeometryType, Vector3


def test_obj_export_metric_scale_nodes_and_surface_materials():
    w = _world()
    obj_str, mtl_str = export_to_obj(w)

    # 1. Per-object nodes
    assert "o " in obj_str
    # 2. Materials named by surface
    assert "usemtl surface_wall" in obj_str or "usemtl surface_floor" in obj_str
    assert "newmtl surface_" in mtl_str

    # 3. Clean trimesh parse and metric scale check
    scene = trimesh.load(io.StringIO(obj_str), file_type="obj")
    bounds = scene.bounds
    # Room in _world is approx 5m x 4m x 2.8m, bounds span meters
    dx = bounds[1][0] - bounds[0][0]
    dy = bounds[1][1] - bounds[0][1]
    dz = bounds[1][2] - bounds[0][2]
    assert 2.0 < dx < 20.0
    assert 2.0 < dy < 20.0
    assert 1.0 < dz < 10.0


def test_glb_export_metric_scale_nodes_and_surface_materials():
    w = _world()
    glb_bytes = export_to_glb(w)
    gltf_dict = export_to_gltf(w)

    # 1. Nodes named per entity
    node_names = [n.get("name") for n in gltf_dict.get("nodes", [])]
    assert any("Wall" in n or "Room" in n for n in node_names if n)

    # 2. Materials named by surface
    mat_names = [m.get("name") for m in gltf_dict.get("materials", [])]
    assert any(m.startswith("surface_") for m in mat_names if m)

    # 3. Trimesh import asserts metric scale
    scene = trimesh.load(io.BytesIO(glb_bytes), file_type="glb")
    bounds = scene.bounds
    dx = bounds[1][0] - bounds[0][0]
    dy = bounds[1][1] - bounds[0][1]
    dz = bounds[1][2] - bounds[0][2]
    assert 2.0 < dx < 20.0
    assert 2.0 < dy < 20.0
    assert 1.0 < dz < 10.0


def test_dae_export_metric_scale_nodes_and_surface_materials():
    w = _world()
    dae_str, report = export_to_dae_with_report(w)
    assert report.format == "dae"
    assert len(report.entities_exported) > 0

    # 1. Valid XML parse
    root = ET.fromstring(dae_str)
    ns = {"c": "http://www.collada.org/2005/11/COLLADASchema"}

    # 2. Metric scale asset unit
    unit = root.find(".//c:asset/c:unit", ns)
    assert unit is not None
    assert unit.attrib.get("meter") == "1.0"
    assert unit.attrib.get("name") == "meter"
    up_axis = root.find(".//c:asset/c:up_axis", ns)
    assert up_axis is not None and up_axis.text == "Z_UP"

    # 3. Per-object nodes in visual scene
    nodes = root.findall(".//c:library_visual_scenes/c:visual_scene/c:node", ns)
    assert len(nodes) == len(report.entities_exported)
    node_names = [n.attrib.get("name") for n in nodes]
    assert any("Wall" in name for name in node_names if name)

    # 4. Materials named by surface
    materials = root.findall(".//c:library_materials/c:material", ns)
    mat_names = [m.attrib.get("name") for m in materials]
    assert any(m.startswith("surface_") for m in mat_names if m)

    # 5. Extract all positions from COLLADA float_arrays and assert metric dimensions
    all_floats = []
    for fa in root.findall(".//c:float_array", ns):
        all_floats.extend([float(x) for x in (fa.text or "").split()])
    xs = all_floats[0::3]
    ys = all_floats[1::3]
    zs = all_floats[2::3]
    dx = max(xs) - min(xs)
    dy = max(ys) - min(ys)
    dz = max(zs) - min(zs)
    assert 2.0 < dx < 20.0
    assert 2.0 < dy < 20.0
    assert 1.0 < dz < 10.0


def test_dae_and_obj_real_mesh_support():
    """Verify real MeshData from artifact store is preserved in DAE and OBJ."""
    w = _world()
    store = MemoryArtifactStore()

    # Create real mesh data (a triangular prism)
    verts = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.5, 1.0, 0.0),
             (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (0.5, 1.0, 1.0)]
    faces = [(0, 1, 2), (3, 5, 4), (0, 3, 4), (0, 4, 1), (1, 4, 5), (1, 5, 2), (2, 5, 3), (2, 3, 0)]
    mesh_data = MeshData(vertices=verts, faces=faces)
    put_res = store.put(mesh_data.to_bytes())
    uri = put_res[0] if isinstance(put_res, tuple) else put_res

    mesh_geom = Geometry(id="geom_prism", type=GeometryType.MESH, data_uri=uri)
    w.geometries["geom_prism"] = mesh_geom

    prism_entity = Entity(
        id="ent_prism",
        name="Prism Object",
        type=w.entities[list(w.entities.keys())[0]].type,
        geometry_ids=["geom_prism"],
        transform={"position": {"x": 1.0, "y": 1.0, "z": 0.0}},
    )
    w.entities["ent_prism"] = prism_entity

    dae_str, _ = export_to_dae_with_report(w, artifact_store=store)
    assert "geom-ent_prism" in dae_str
    # 6 vertices * 3 coords = 18 floats in the position array
    assert 'count="18"' in dae_str

    obj_str, _ = export_to_obj_with_report(w, artifact_store=store)
    assert "o Prism Object" in obj_str


def test_reality_sdk_export_dae():
    """Verify reality.export(world, 'dae') works out of the box."""
    w = _world()
    data, report = reality.export(w, "dae")
    assert isinstance(data, str)
    assert report.format == "dae"
    assert "<COLLADA" in data
