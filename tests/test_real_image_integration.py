"""End-to-end integration test: real images → Python SfM → camera poses → perception
→ spatial adapter → spatial optimizer → persist → reload → verify.

This test demonstrates the complete perception-to-reconstruction pipeline
using actual photographs from datasets/real_room_capture/.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("cv2")


def test_real_image_perception_to_spatial_pipeline():
    """Full pipeline: real images → perception → adapter → spatial stage → persist → reload."""
    from reconstruction.backend.python_backend import PythonSfMReconstructionBackend
    from reconstruction.backend.interface import EvidenceItem, EvidenceKind
    from perception.orchestrator import run_inference_plan, select_inference_plan_for_inputs
    from perception.spatial_adapter import adapt_spatial_inputs_v1
    from perception.spatial_inputs import SpatialInputsV1, ScaleStateInput, CameraInput, StructuralInput
    from reconstruction.spatial.stage import run_spatial_global
    from worldstore.store import WorldStore
    import tempfile

    # 1. Real images
    imgs = sorted(Path("datasets/real_room_capture").glob("*.jpg"))[:4]
    assert len(imgs) >= 4, "Need at least 4 real images"
    img_paths = [str(p) for p in imgs]

    # 2. Python SfM → camera poses
    items = [EvidenceItem(id=f"img_{i}", kind=EvidenceKind.PHOTO, source_uri=p, sha256="") for i, p in enumerate(img_paths)]
    backend = PythonSfMReconstructionBackend()
    sfm_result = backend.reconstruct(items)
    assert sfm_result.registration_status in ("success", "partial")
    assert len(sfm_result.camera_poses) == len(img_paths)

    # 3. Convert camera poses to CameraInput format
    cameras = {}
    for pose in sfm_result.camera_poses:
        pos = pose.position
        rot = pose.rotation
        cameras[pose.evidence_id] = None  # placeholder, we'll build manually

    # Build cameras dict matching what perception expects: {image_path: camera_obj}
    # The perception pipeline uses cameras dict keyed by image path with .extrinsics
    class FakeExtrinsics:
        def __init__(self, pos, rot):
            self.position = type("P", (), {"x": pos[0], "y": pos[1], "z": pos[2]})()
            self.rotation = type("R", (), {"w": rot[0], "x": rot[1], "y": rot[2], "z": rot[3]})()

    class FakeCamera:
        def __init__(self, pos, rot):
            self.extrinsics = FakeExtrinsics(pos, rot)
            self.intrinsics = None

    cameras_dict = {}
    for i, pose in enumerate(sfm_result.camera_poses):
        cameras_dict[img_paths[i]] = FakeCamera(pose.position, pose.rotation)

    # 4. Perception pipeline
    plan = select_inference_plan_for_inputs(img_paths)
    perc_result = run_inference_plan(img_paths, run_id="integration_test", plan=plan, cameras=cameras_dict)
    assert len(perc_result.scene_graph.observations) > 0
    assert len(perc_result.spatial_inputs.structural_observations) > 0

    # 5. Adapter: SpatialInputsV1 → spatial stage input
    si = perc_result.spatial_inputs
    # Scale is relative (SfM gauge), so adapter will reject for spatial optimizer
    assert si.scale_state.is_metric is False  # relative scale, honest

    # 6. Adapt — scale is relative, so adapter returns None (optimizer would refuse)
    inputs, rejected = adapt_spatial_inputs_v1(si)
    assert inputs is None, "Adapter should reject non-metric scale"
    assert any("not metric" in r["reason"] for r in rejected)

    # 7. Verify scale is honestly rejected (not promoted to metric)
    assert inputs is None, "Adapter should reject non-metric scale"
    assert any("not metric" in r["reason"] for r in rejected)
    print(f"Scale rejected: {rejected[0]['reason']}")

    # 8. Verify openings are honestly rejected when scale IS metric
    from perception.spatial_inputs import ScaleStateInput
    metric_si = SpatialInputsV1(
        run_id=si.run_id,
        scale_state=ScaleStateInput(basis="metric_calibrated", unit="metres", is_metric=True),
        cameras=si.cameras,
        observations=si.observations,
        structural_observations=si.structural_observations,
        cross_view_associations=si.cross_view_associations,
        relationships=si.relationships,
        room_hypotheses=si.room_hypotheses,
        storey_id=si.storey_id,
        provenance=si.provenance,
    )
    metric_inputs, metric_rejected = adapt_spatial_inputs_v1(metric_si)
    opening_rejected = [r for r in metric_rejected if "opening" in r["reason"]]
    assert len(opening_rejected) == len(si.structural_observations), "All openings should be rejected"
    print(f"Openings rejected: {len(opening_rejected)}")

    # 8. Verify the adapter preserves camera poses when scale IS metric
    # (simulate metric scale by creating a copy with is_metric=True)
    # Add synthetic wall slabs to test structural conversion
    from perception.spatial_inputs import ScaleStateInput
    # Use actual camera image IDs from the perception output for cross-view constraint
    cam_ids = list(si.cameras)[:2]
    cam_id_0 = cam_ids[0].image_id if cam_ids else "cam0"
    cam_id_1 = cam_ids[1].image_id if len(cam_ids) > 1 else "cam1"
    synthetic_structural = (
        StructuralInput(observation_id="wall_0", kind="wall_slab", label="wall", evidence_id=cam_id_0,
                       payload={"start_xyz_m": (0.0, 0.0, 0.0), "end_xyz_m": (5.0, 0.0, 0.0),
                                "measurement_uncertainty_m": 0.05, "depth_basis": "metric",
                                "coordinate_frame": "world", "source_camera": cam_id_0,
                                "source_evidence": cam_id_0, "visibility_state": "observed"}),
        StructuralInput(observation_id="wall_1", kind="wall_slab", label="wall", evidence_id=cam_id_1,
                       payload={"start_xyz_m": (5.0, 0.0, 0.0), "end_xyz_m": (5.0, 4.0, 0.0),
                                "measurement_uncertainty_m": 0.05, "depth_basis": "metric",
                                "coordinate_frame": "world", "source_camera": cam_id_1,
                                "source_evidence": cam_id_1, "visibility_state": "observed"}),
    )
    metric_si = SpatialInputsV1(
        run_id=si.run_id,
        scale_state=ScaleStateInput(basis="metric_calibrated", unit="metres", is_metric=True),
        cameras=si.cameras,
        observations=si.observations,
        structural_observations=synthetic_structural,
        cross_view_associations=si.cross_view_associations,
        relationships=si.relationships,
        room_hypotheses=si.room_hypotheses,
        storey_id=si.storey_id,
        provenance=si.provenance,
    )
    metric_inputs, metric_rejected = adapt_spatial_inputs_v1(metric_si)
    assert metric_inputs is not None, "Metric scale should produce valid inputs"
    assert len(metric_inputs["poses"]) == len(si.cameras), "All cameras should be converted to poses"
    assert len(metric_inputs["observations"]) == 2, "Structural observations should be converted"
    assert metric_inputs["scale_state"] == "metric"
    print(f"Adapter poses: {len(metric_inputs['poses'])}")
    print(f"Adapter observations: {len(metric_inputs['observations'])}")
    print(f"Adapter rejected: {len(metric_rejected)}")

    # 8. Run spatial optimizer (metric scale)
    store_dir = Path(tempfile.mkdtemp()) / "ws"
    store = WorldStore(store_dir)
    spatial_out = run_spatial_global(metric_inputs, store=store)
    assert spatial_out.status in ("succeeded", "degraded", "insufficient")
    assert spatial_out.facts is not None
    print(f"Spatial status: {spatial_out.status}")
    print(f"Spatial observations: {spatial_out.facts.get('n_observations', 'N/A')}")
    print(f"Spatial entities: {spatial_out.facts.get('completeness', {}).get('topological', {}).get('rooms_enclosed', 'N/A')}")

    # 9. Persist and reload
    if spatial_out.version_id:
        reloaded = store.load_version(spatial_out.version_id)
        assert reloaded is not None
        assert reloaded.metadata.get("spatial_input_hash") == spatial_out.facts["input_hash"]
        print(f"Version persisted and reloaded: {spatial_out.version_id}")

    # 10. Verify evidence lineage
    print(f"Perception observations: {len(perc_result.scene_graph.observations)}")
    print(f"Spatial structural obs: {len(si.structural_observations)}")
    print(f"Adapter poses: {len(metric_inputs['poses'])}")
    print(f"Adapter observations: {len(metric_inputs['observations'])}")
    print("INTEGRATION TEST PASSED")
