"""End-to-End Acceptance Test for Metric Reconstruction and Failure Paths.

Covers the complete perception-to-world chain:
1. Full Success Path: Calibrated camera -> metric depth -> 2D detections -> segmentation masks ->
   metric 3D lifting -> part-parent association -> WorldIR compilation -> validation -> persistence -> reload.
2. Failure Path: Missing or malformed camera calibration (fx, fy, cx, cy, width, height).
3. Failure Path: Relative unscaled depth (is_metric_depth=False).
4. Failure Path: Incompatible or empty segmentation mask.
5. Failure Path: Non-rigid extrinsics (det(R) != 1 or non-orthogonal).
6. Failure Path: Unloaded adapter inference attempt.
7. Persistence & Reload: Serialization round-trip maintains exact schema and confidence contracts.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

from engine.math import Quat, Vec3
from engine.multiengine.contract import EngineError, EngineErrorKind, EngineRequest, EngineSpec, Task
from engine.multiengine.adapters.metric_3d_lifter import Metric3DLifterAdapter
from engine.multiengine.metric_lifting import Metric3DLifter
from engine.multiengine.visibility_reprojection import VisibilityReprojectionValidator
from world.ontology.universal_object_ontology import get_universal_ontology
from world_ir.camera import CameraExtrinsics, CameraIntrinsics, PinholeCamera
from world_ir.schema_v1 import Entity, EntityType, Provenance
from world_ir.serialization import load_world, save_world
from world_ir.validation import validate_world_ir
from world_ir.world_v1 import WorldIR

ANCHORED = dict(  # SYNTHETIC test fixtures declare their scale explicitly; nothing defaults to metres
    is_metric_depth=True, scale_state="PHYSICALLY_ANCHORED", scale_source="synthetic-test-fixture-anchor",
    depth_source="synthetic_test_fixture", calibration_source="synthetic_test_fixture",
)


@pytest.fixture
def calibrated_rig() -> tuple[PinholeCamera, np.ndarray, dict]:
    """Standard calibrated camera at origin and 2.5m metric depth plane."""
    intr = CameraIntrinsics(
        fx=800.0,
        fy=800.0,
        cx=400.0,
        cy=300.0,
        width=800,
        height=600,
    )
    extr = CameraExtrinsics(
        position=Vec3(0.0, 0.0, 0.0),
        rotation=Quat.identity(),
    )
    camera = PinholeCamera(intrinsics=intr, extrinsics=extr)
    depth_map = np.full((600, 800), 2.5, dtype=np.float32)

    cam_meta = {
        "intrinsics": {
            "fx": 800.0, "fy": 800.0, "cx": 400.0, "cy": 300.0,
            "width": 800, "height": 600,
        },
        "extrinsics": {
            "position": [0.0, 0.0, 0.0],
            "rotation": [1.0, 0.0, 0.0, 0.0],
        },
        "is_canonical_frame": True,
        "source": "calibrated_acceptance_rig",
        "method": "pinhole_canonical",
        "coordinate_frame": "world:canonical",
    }

    return camera, depth_map, cam_meta


def test_e2e_reconstruction_full_success_path(
    calibrated_rig: tuple[PinholeCamera, np.ndarray, dict], tmp_path: Path
) -> None:
    """Full perception-to-world reconstruction e2e path."""
    camera, depth_map, cam_meta = calibrated_rig
    lifter = Metric3DLifter(get_universal_ontology())

    # 1. Lift a parent entity (facade / wall)
    wall_lifted = lifter.lift_detection(
        entity_id="facade_main",
        class_name="wall",
        box_2d=(100.0, 50.0, 700.0, 550.0),
        depth_map=depth_map,
        camera=camera,
        confidence=0.95,
        is_metric_depth=True, scale_state="PHYSICALLY_ANCHORED", scale_source="synthetic-test-fixture-anchor",
        depth_source="synthetic_test_fixture",
    )
    assert wall_lifted is not None
    assert wall_lifted.class_name == "wall"

    # 2. Lift a subpart entity inside parent (window)
    window_lifted = lifter.lift_detection(
        entity_id="window_01",
        class_name="window",
        box_2d=(250.0, 150.0, 450.0, 350.0),
        depth_map=depth_map,
        camera=camera,
        confidence=0.90,
        is_metric_depth=True, scale_state="PHYSICALLY_ANCHORED", scale_source="synthetic-test-fixture-anchor",
        depth_source="synthetic_test_fixture",
    )
    assert window_lifted is not None
    assert window_lifted.class_name == "window"

    # 3. Associate subpart to parent in 3D
    associated = lifter.associate_parts_and_parents([wall_lifted, window_lifted])
    assert len(associated) == 2

    # 4. Compile into WorldIR
    world = WorldIR()
    for lent in associated:
        world_ent = Entity(
            id=f"ent-{lent.entity_id}",
            type=EntityType.STRUCTURE,
            name=lent.class_name.capitalize(),
            transform={
                "position": {"x": lent.position[0], "y": lent.position[1], "z": lent.position[2]},
                "orientation": (
                    {"w": lent.orientation[0], "x": lent.orientation[1], "y": lent.orientation[2], "z": lent.orientation[3]}
                    if lent.orientation is not None
                    else None
                ),
                "frame_id": lent.coordinate_frame,
            },
            provenance=Provenance.RECONSTRUCTED,
            confidence=lent.confidence,
            confidence_basis="depth_unprojection_pca_quantile",
            custom_properties={
                "dimensions": list(lent.dimensions),
                "point_count": lent.point_count,
                "is_part": lent.is_part,
                "parent_entity_id": lent.parent_entity_id,
            },
        )
        world.entities[world_ent.id] = world_ent

    # 5. Validate WorldIR
    val_report = validate_world_ir(world)
    errors = [i for i in val_report.issues if i.severity.value == "error"]
    assert len(errors) == 0

    # 6. Persist to disk and reload
    save_path = tmp_path / "reconstructed_world.json"
    save_world(world, str(save_path))
    assert save_path.exists()

    reloaded_world = load_world(str(save_path))
    assert len(reloaded_world.entities) == 2
    assert "ent-facade_main" in reloaded_world.entities
    assert "ent-window_01" in reloaded_world.entities

    reloaded_window = reloaded_world.entities["ent-window_01"]
    assert reloaded_window.confidence == 0.90
    assert reloaded_window.confidence_basis == "depth_unprojection_pca_quantile"


def test_e2e_failure_path_missing_calibration() -> None:
    """Missing or uncalibrated camera metadata returns uncalibrated and prevents fake geometry."""
    bad_meta = {
        "intrinsics": {"fx": 500.0},  # Missing fy, cx, cy, width, height!
        "is_canonical_frame": True,
    }
    val = VisibilityReprojectionValidator.validate_camera_metadata(bad_meta)
    assert val.is_calibrated is False
    assert val.camera is None
    assert "missing" in str(val.rejection_reason)


def test_e2e_failure_path_relative_unscaled_depth(
    calibrated_rig: tuple[PinholeCamera, np.ndarray, dict]
) -> None:
    """Relative monocular depth without scale alignment is strictly rejected."""
    camera, depth_map, _ = calibrated_rig
    lifter = Metric3DLifter()

    result = lifter.lift_detection(
        entity_id="chair_relative",
        class_name="chair",
        box_2d=(200.0, 150.0, 400.0, 350.0),
        depth_map=depth_map,
        camera=camera,
        is_metric_depth=False,  # Unscaled relative depth
    )
    assert result is None


def test_e2e_failure_path_incompatible_mask(
    calibrated_rig: tuple[PinholeCamera, np.ndarray, dict]
) -> None:
    """Mask dimensions mismatched with depth map are strictly rejected."""
    camera, depth_map, _ = calibrated_rig
    lifter = Metric3DLifter()

    bad_mask = np.ones((100, 100), dtype=bool)  # Depth map is (600, 800)
    result = lifter.lift_detection(
        **ANCHORED,
        entity_id="mismatched_mask",
        class_name="door",
        box_2d=(200.0, 150.0, 400.0, 350.0),
        depth_map=depth_map,
        camera=camera,
        mask_2d=bad_mask,
    )
    assert result is None


def test_e2e_failure_path_non_rigid_extrinsics() -> None:
    """Camera extrinsics with non-rigid rotation matrix (reflection det=-1) are rejected."""
    reflection_matrix = [
        [-1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 2.0],
        [0.0, 0.0, 1.0, 3.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    meta = {
        "intrinsics": {
            "fx": 600.0, "fy": 600.0, "cx": 320.0, "cy": 240.0,
            "width": 640, "height": 480,
        },
        "extrinsics": reflection_matrix,
    }
    val = VisibilityReprojectionValidator.validate_camera_metadata(meta)
    assert val.is_calibrated is False
    assert "non_rigid_rotation" in str(val.rejection_reason)


def test_e2e_failure_path_unloaded_adapter_inference() -> None:
    """Calling infer() on an adapter before load() raises an explicit EngineError."""
    from engine.multiengine.registry import maximum_fidelity_registry

    spec = maximum_fidelity_registry().get("sam")
    adapter = Metric3DLifterAdapter(spec)
    req = EngineRequest(sources=("photo.jpg",))

    with pytest.raises(EngineError) as exc_info:
        adapter.infer(req)

    assert exc_info.value.kind is EngineErrorKind.UNAVAILABLE
    assert "not loaded" in str(exc_info.value)
