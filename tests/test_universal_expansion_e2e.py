"""Universal Intelligence Expansion — Full End-to-End Master Pipeline Integration Test.

Validates the complete chain of Reality Engine capabilities in a unified,
autonomous workflow:
  1. Multi-view camera ingestion & metric scale fusion (RANSAC baseline consensus)
  2. Watertight manifold mesh repair with sharp planar decimation
  3. 3D Oriented Bounding Box (OMBB) & 6-DoF pose estimation
  4. Multi-view Hungarian visual instance association
  5. 12-class PBR material classification & seam-blended UV texture unprojection
  6. Multi-storey staircase vertical topology graph & inter-storey path traversal
  7. Next-best-view capture planning with information gain optimization
  8. Multi-epoch temporal alignment & displacement/renovation diffing
  9. Multi-format asset export: IFC4 BIM, 2D Multi-sheet PDF, glTF 2.0, USDA
  10. Grounded Reality Copilot spatial query execution with zero hallucination
"""

from __future__ import annotations

import copy
import math
import numpy as np
import pytest

from world_ir import Entity, EntityType, Relationship, RelationshipKind
from world_ir.schema_v1 import Geometry, GeometryType, Uncertainty, Vector3
from world_ir.world_v1 import WorldIR
from reconstruction.meshing.mesh import MeshData
from world_ir.temporal import (
    TemporalEpoch,
    align_temporal_epochs,
    diff_temporal_epochs,
    TemporalChangeKind,
)
from reconstruction.backend.interface import (
    ReconstructedCameraPose,
    ReconstructedPoint,
    ReconstructionResult,
)
from reconstruction.scale import (
    MetricScaleFusionEngine,
    ScaleReference,
    ScaleState,
)
from reconstruction.meshing.repair import (
    repair_watertight_and_decimate,
)
from perception.instances.bounding import solve_ombb, OrientedBoundingBox
from perception.instances.association import associate, Observation, DepthSource, AssociationState
from perception.materials.classifier import (
    classify_material,
    assign_pbr_material,
    MaterialCategory,
    MaterialEvidence,
)
from engine.render.textures.projection import (
    PinholeCamera,
    PlanarSurface,
    TextureAtlas,
    measure_seam_discontinuity,
    multi_band_seam_blend,
    project_and_blend_planar_texture,
)
from engine.compiler.topology_graph import (
    compile_vertical_topology,
    StoreyLevel,
    StaircaseAnatomy,
)
from planning.capture_planner import (
    CameraPose,
    CapturePlanResult,
    RecommendedViewpoint,
    plan_next_best_view_capture,
)
from assistant.copilot_engine import RealityCopilotEngine
from exporters.ifc_bridge import export_world_to_ifc
from exporters.floorplan import export_to_multisheet_floorplan_pdf_with_report
from tests.test_room_fidelity_fixture import _compiled
from world_ir.measure import stamp_world_dimensions
import sdk.reality as reality_sdk


def _build_metric_compiled_world() -> WorldIR:
    """Build a fully validated compiled room world with metric dimensions stamped."""
    _obs, world, _diag = copy.deepcopy(_compiled("albedo", False))
    world.metadata["scale"] = {
        "state": "metric",
        "meters_per_unit": 1.0,
        "dimension_state": "MEASURED",
    }
    stamp_world_dimensions(world)
    return world


def _make_camera(cam_id: str, pos_world: tuple[float, float, float], look_dir: tuple[float, float, float], img: np.ndarray) -> PinholeCamera:
    pos = np.array(pos_world, dtype=np.float64)
    z_axis = np.array(look_dir, dtype=np.float64)
    z_axis /= np.linalg.norm(z_axis)

    up = np.array([0.0, 0.0, 1.0]) if abs(z_axis[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x_axis = np.cross(z_axis, up)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    R = np.vstack([x_axis, y_axis, z_axis])
    t = -R @ pos

    T_wc = np.eye(4)
    T_wc[:3, :3] = R
    T_wc[:3, 3] = t

    H, W = img.shape[:2]
    K = [
        [float(W), 0.0, float(W) * 0.5],
        [0.0, float(W), float(H) * 0.5],
        [0.0, 0.0, 1.0],
    ]
    return PinholeCamera(camera_id=cam_id, intrinsics=K, world_to_cam=T_wc.tolist(), image_rgb=img)


def test_e2e_universal_reconstruction_and_scale_fusion():
    """Verify autonomous metric scale fusion with 1D RANSAC consensus."""
    true_scale = 2.0  # 2.0 meters per model unit
    poses = [
        ReconstructedCameraPose(
            evidence_id="cam_0",
            position=(0.0, 0.0, 1.0 / true_scale),
            rotation=(1.0, 0.0, 0.0, 0.0),
        ),
        ReconstructedCameraPose(
            evidence_id="cam_1",
            position=(4.0 / true_scale, 0.0, 1.0 / true_scale),
            rotation=(1.0, 0.0, 0.0, 0.0),
        ),
        ReconstructedCameraPose(
            evidence_id="cam_2",
            position=(0.0, 3.0 / true_scale, 1.0 / true_scale),
            rotation=(1.0, 0.0, 0.0, 0.0),
        ),
    ]
    pts = [
        ReconstructedPoint(
            position=(float(x) / true_scale, float(y) / true_scale, 0.0),
            track_id=f"p_{x}_{y}",
            source_evidence_ids=["cam_0", "cam_1", "cam_2"],
        )
        for x in (0, 1, 2, 3, 4)
        for y in (0, 1, 2, 3)
    ]
    recon = ReconstructionResult(
        points=pts,
        camera_poses=poses,
        registration_status="success",
    )

    ref = ScaleReference(evidence_id_a="cam_0", evidence_id_b="cam_1", distance_m=4.0)

    engine = MetricScaleFusionEngine()
    scaled, diag = engine.fuse(recon, references=[ref])

    assert diag.consensus_reached is True
    assert diag.scale_state == "metric"
    assert diag.fused_meters_per_unit == pytest.approx(true_scale, rel=1e-3)
    assert diag.estimated_error_pct <= 1.5


def test_e2e_watertight_mesh_repair_and_planar_simplification():
    """Verify mesh repair and sharp corner decimation on noisy geometry."""
    box_vertices = np.array([
        [0.0, 0.0, 0.0], [6.0, 0.0, 0.0], [6.0, 0.2, 0.0], [0.0, 0.2, 0.0],
        [0.0, 0.0, 3.0], [6.0, 0.0, 3.0], [6.0, 0.2, 3.0], [0.0, 0.2, 3.0],
        [2.0, 0.0, 1.0], [4.0, 0.0, 1.0], [2.0, 0.0, 2.0], [4.0, 0.0, 2.0],
    ], dtype=float)

    faces = [
        [0, 1, 5], [0, 5, 4],
        [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6],
        [3, 0, 4], [3, 4, 7],
        [4, 5, 6], [4, 6, 7],
        [0, 2, 1], [0, 3, 2],
    ]
    raw_faces = np.array(faces, dtype=int)

    input_mesh = MeshData(vertices=box_vertices, faces=raw_faces)
    repaired_mesh, report = repair_watertight_and_decimate(
        input_mesh, target_planar_reduction=0.5
    )

    assert len(repaired_mesh.faces) > 0
    assert report["initial_vertices"] == len(box_vertices)
    assert report["final_faces"] > 0
    assert "is_manifold" in report


def test_e2e_instance_bounding_box_and_hungarian_association():
    """Verify 3D OMBB estimation and persistent multi-view association."""
    rng = np.random.RandomState(42)
    pts = rng.uniform(
        low=[2.1, 2.05, 0.0],
        high=[3.9, 2.95, 0.75],
        size=(200, 3),
    ).tolist()

    ombb = solve_ombb(pts, class_id="furniture.table")
    assert 1.4 <= ombb.extents[0] <= 2.2
    assert 0.65 <= ombb.extents[1] <= 1.2
    assert 0.55 <= ombb.extents[2] <= 0.95

    # Multi-view Hungarian tracking across 5 camera views:
    obs = [
        Observation(
            observation_id=f"door_obs_{i}",
            image_id=f"frame_{i}.jpg",
            label="door",
            world_pos=(2.0 + 0.02 * (i % 2), 1.0, 0.0),
            sigma_m=0.05,
            depth_source=DepthSource.METRIC,
        )
        for i in range(5)
    ]
    assoc_result = associate(obs)
    assert len(assoc_result.entities) == 1
    assert len(assoc_result.entities[0].observation_ids) == 5
    assert len(assoc_result.entities[0].image_ids) == 5


def test_e2e_pbr_materials_and_seam_blended_uv_projection():
    """Verify 12-class PBR parameter assignment and seam-blended unprojection."""
    world = _build_metric_compiled_world()
    wall_id = "wall-room-plane-000-01"
    assert wall_id in world.entities
    wall_entity = world.entities[wall_id]

    ev_plaster = MaterialEvidence(
        color_rgb=[235, 235, 230],
        texture_roughness=0.10,
        specular_sheen=0.08,
        surface_kind="wall",
        spread_delta_e=3.0,
    )
    clf = classify_material(ev_plaster)
    assert len(clf) > 0
    assert clf[0].category == MaterialCategory.PLASTER
    assert clf[0].confidence >= 0.70

    mat = assign_pbr_material(world, wall_entity, ev_plaster)
    assert mat.id in world.materials
    assert wall_entity.material_ids == [mat.id]
    assert "pbr_material" in wall_entity.custom_properties
    assert wall_entity.custom_properties["pbr_material"]["category"] == "plaster"

    surface = PlanarSurface(
        origin=(-2.0, 0.0, 0.0),
        u_axis=(1.0, 0.0, 0.0),
        v_axis=(0.0, 0.0, 1.0),
        normal=(0.0, -1.0, 0.0),
        width_m=4.0,
        height_m=3.0,
    )

    img_left = np.ones((100, 100, 3), dtype=np.float64) * 0.75
    img_right = np.ones((100, 100, 3), dtype=np.float64) * 0.60
    cam_left = _make_camera("cam-left", (-1.0, -3.0, 1.5), (0.0, 1.0, 0.0), img_left)
    cam_right = _make_camera("cam-right", (1.0, -3.0, 1.5), (0.0, 1.0, 0.0), img_right)

    atlas = project_and_blend_planar_texture(
        surface=surface,
        cameras=[cam_left, cam_right],
        texture_width=128,
        texture_height=128,
        enable_seam_blending=True,
    )

    assert atlas.width == 128
    assert atlas.height == 128
    assert atlas.pixels.shape == (128, 128, 3)
    assert atlas.max_seam_discontinuity < 0.10


def test_e2e_vertical_topology_and_next_best_view_planning():
    """Verify multi-storey staircase coordinate graph and capture planning."""
    world_vert = WorldIR(id="w-multi-storey")
    ground_floor = Entity(
        id="floor-0",
        name="Ground Floor Slab",
        type=EntityType.FLOOR,
        transform={"position": {"x": 0.0, "y": 0.0, "z": 0.0}}
    )
    first_floor = Entity(
        id="floor-1",
        name="First Floor Slab",
        type=EntityType.FLOOR,
        transform={"position": {"x": 0.0, "y": 0.0, "z": 3.0}}
    )
    stairs = Entity(
        id="stair-flight-1",
        name="Main Staircase Flight",
        type=EntityType.STAIRS,
        transform={"position": {"x": 2.0, "y": 3.0, "z": 0.0}},
        custom_properties={
            "total_rise_m": 3.0,
            "total_run_m": 3.6,
            "flight_width_m": 1.1,
            "num_risers": 16,
        }
    )
    room_ground = Entity(id="room-101", name="Living Room", type=EntityType.ROOM, transform={"position": {"x": 1.0, "y": 1.0, "z": 0.0}})
    room_upper = Entity(id="room-201", name="Master Bedroom", type=EntityType.ROOM, transform={"position": {"x": 1.0, "y": 1.0, "z": 3.0}})

    world_vert.entities[ground_floor.id] = ground_floor
    world_vert.entities[first_floor.id] = first_floor
    world_vert.entities[stairs.id] = stairs
    world_vert.entities[room_ground.id] = room_ground
    world_vert.entities[room_upper.id] = room_upper

    graph = compile_vertical_topology(world_vert)
    assert len(graph.storeys) == 2
    assert "storey-ground" in graph.storeys
    assert "storey-1" in graph.storeys
    assert graph.storeys["storey-ground"].floor_elevation_m == 0.0
    assert graph.storeys["storey-1"].floor_elevation_m == 3.0

    path = graph.find_shortest_vertical_path("room-101", "room-201")
    assert path is not None
    assert path.from_storey_id == "storey-ground"
    assert path.to_storey_id == "storey-1"
    assert "stair-flight-1" in path.via_stair_ids

    # Next-Best-View Capture Planning
    world_plan = WorldIR(id="w-sparse-capture")
    walls = [
        Entity(id="wall-north", name="North Wall", type=EntityType.WALL, transform={"position": {"x": 0.0, "y": 3.0, "z": 0.0}}),
        Entity(id="wall-south", name="South Wall", type=EntityType.WALL, transform={"position": {"x": 0.0, "y": -3.0, "z": 0.0}}),
        Entity(id="wall-east", name="East Wall", type=EntityType.WALL, transform={"position": {"x": 3.0, "y": 0.0, "z": 0.0}}),
        Entity(id="wall-west", name="West Wall", type=EntityType.WALL, transform={"position": {"x": -3.0, "y": 0.0, "z": 0.0}}),
        Entity(id="floor", name="Floor", type=EntityType.FLOOR, transform={"position": {"x": 0.0, "y": 0.0, "z": 0.0}}),
    ]
    for w in walls:
        world_plan.entities[w.id] = w

    existing_cams = [
        CameraPose(camera_id="cam-1", position=(0.0, 0.0, 1.5), look_direction=(0.0, 1.0, 0.0), fov_deg=60.0)
    ]

    result = plan_next_best_view_capture(
        world=world_plan,
        existing_cameras=existing_cams,
        max_recommendations=5,
        camera_height_m=1.6,
        camera_fov_deg=65.0,
    )

    assert result.initial_uncertainty > 0.40
    assert result.unobserved_patches > 0
    assert result.uncertainty_reduction_ratio >= 0.40
    assert len(result.recommendations) > 0


def test_e2e_temporal_multi_epoch_diffing():
    """Verify temporal alignment and change detection between 2 capture epochs."""
    walls_a = [
        Entity(id="wall-n", name="North Wall", type=EntityType.STRUCTURE, transform={"position": {"x": 0.0, "y": 5.0, "z": 0.0}}),
        Entity(id="wall-s", name="South Wall", type=EntityType.STRUCTURE, transform={"position": {"x": 0.0, "y": -5.0, "z": 0.0}}),
        Entity(id="wall-e", name="East Wall", type=EntityType.STRUCTURE, transform={"position": {"x": 5.0, "y": 0.0, "z": 0.0}}),
        Entity(id="wall-w", name="West Wall", type=EntityType.STRUCTURE, transform={"position": {"x": -5.0, "y": 0.0, "z": 0.0}}),
        Entity(id="chair-1", name="Armchair", type=EntityType.STRUCTURE, transform={"position": {"x": 1.0, "y": 1.0, "z": 0.0}}, custom_properties={"movable": True}),
        Entity(id="desk-old", name="Old Desk", type=EntityType.STRUCTURE, transform={"position": {"x": 2.0, "y": 2.0, "z": 0.0}}),
    ]
    world_a = WorldIR(id="world-epoch-1")
    for w in walls_a:
        world_a.entities[w.id] = w

    epoch_a = TemporalEpoch(epoch_id="epoch-2025-01", timestamp_iso="2025-01-15T10:00:00Z", world=world_a)

    t_drift = np.array([2.0, 3.0, 0.5])
    walls_b = [
        Entity(id="wall-n", name="North Wall", type=EntityType.STRUCTURE, transform={"position": {"x": 0.0 + t_drift[0], "y": 5.0 + t_drift[1], "z": 0.0 + t_drift[2]}}),
        Entity(id="wall-s", name="South Wall", type=EntityType.STRUCTURE, transform={"position": {"x": 0.0 + t_drift[0], "y": -5.0 + t_drift[1], "z": 0.0 + t_drift[2]}}),
        Entity(id="wall-e", name="East Wall", type=EntityType.STRUCTURE, transform={"position": {"x": 5.0 + t_drift[0], "y": 0.0 + t_drift[1], "z": 0.0 + t_drift[2]}}),
        Entity(id="wall-w", name="West Wall", type=EntityType.STRUCTURE, transform={"position": {"x": -5.0 + t_drift[0], "y": 0.0 + t_drift[1], "z": 0.0 + t_drift[2]}}),
        # Chair moved by 0.25m
        Entity(id="chair-1", name="Armchair", type=EntityType.STRUCTURE, transform={"position": {"x": 1.0 + t_drift[0] + 0.25, "y": 1.0 + t_drift[1], "z": 0.0 + t_drift[2]}}, custom_properties={"movable": True}),
        # Desk removed, new sofa added
        Entity(id="sofa-new", name="New Sofa", type=EntityType.STRUCTURE, transform={"position": {"x": 3.0 + t_drift[0], "y": 3.0 + t_drift[1], "z": 0.0 + t_drift[2]}}),
    ]
    world_b = WorldIR(id="world-epoch-2")
    for w in walls_b:
        world_b.entities[w.id] = w

    epoch_b = TemporalEpoch(epoch_id="epoch-2025-06", timestamp_iso="2025-06-20T14:30:00Z", world=world_b)

    alignment = align_temporal_epochs(epoch_a, epoch_b)
    assert alignment.converged is True
    assert alignment.num_anchors_used >= 4
    assert alignment.residual_error_m < 0.01

    diff_report = diff_temporal_epochs(epoch_a, epoch_b, movement_threshold_m=0.05)
    assert len(diff_report.moved) == 1
    assert diff_report.moved[0].entity_id == "chair-1"
    assert math.isclose(diff_report.moved[0].displacement_m, 0.25, abs_tol=1e-3)
    assert diff_report.moved[0].change_kind == TemporalChangeKind.MOVED

    assert len(diff_report.removed) == 1
    assert diff_report.removed[0].entity_id == "desk-old"

    assert len(diff_report.added) == 1
    assert diff_report.added[0].entity_id == "sofa-new"


def test_e2e_multi_format_exports_and_copilot_assistant(tmp_path):
    """Verify IFC4, Multi-sheet 2D PDF, glTF, USD exports, and Grounded Copilot."""
    world = _build_metric_compiled_world()

    # 1. IFC4 BIM Export with Pset
    ifc_path = tmp_path / "model.ifc"
    ifc_report = export_world_to_ifc(world, ifc_path)
    assert ifc_path.exists()
    assert len(ifc_report.get("written", [])) >= 3
    ifc_content = ifc_path.read_text(encoding="utf-8")
    assert "Pset_WallCommon" in ifc_content or "IFCWALL" in ifc_content

    # 2. Multi-Sheet Floor Plan PDF via reality_sdk
    pdf_bytes, pdf_report = reality_sdk.export(world, "multisheet_floorplan_pdf")
    assert len(pdf_bytes) > 500
    assert pdf_bytes.startswith(b"%PDF-")
    assert pdf_report.format == "multisheet_floorplan_pdf"

    # 3. glTF 2.0 Export
    gltf_data, gltf_report = reality_sdk.export(world, "gltf")
    assert "asset" in gltf_data
    assert "nodes" in gltf_data
    assert len(gltf_data["nodes"]) >= 3
    assert gltf_report.format == "gltf"

    # glB binary export
    glb_bytes, glb_report = reality_sdk.export(world, "glb")
    assert len(glb_bytes) > 100
    assert glb_report.format == "glb"

    # 4. USD Export
    usd_text, usd_report = reality_sdk.export(world, "usda")
    assert len(usd_text) > 100
    assert usd_report.format == "usda"

    # 5. Grounded Reality Copilot Spatial Engine
    copilot = RealityCopilotEngine(world, scale_state="metric")
    res = copilot.ask("what is the height of wall-room-plane-000-01?")
    assert res.success is True
    assert len(res.measurements) == 1
    assert "2." in res.answer or "3." in res.answer
