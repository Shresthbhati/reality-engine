"""Difficult-image failure tests and Perception-to-WorldIR end-to-end validation (PR-D)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from perception.engine_evidence import (
    CoordinateFrame,
    ModelIdentity,
    ObservationRecord,
)
from perception.evaluation.failure_detector import EngineFailureDetector, FailureKind
from perception.mask_persistence import load_mask_store, persist_masks, save_mask_store
from perception.spatial_reasoning import infer_spatial_relationships
from perception.world_bridge import assess_image_quality
from provenance import Provenance
from world_ir import Entity, EntityType, Geometry, GeometryType, Vector3, WorldIR
from world_ir.mask_store import MaskStore
from worldstore.store import WorldStore


def test_difficult_images_low_light_glare_motion_blur_and_low_res() -> None:
    # 1. Low light
    dark = np.full((120, 160, 3), 12, dtype=np.uint8)
    q_dark = assess_image_quality(dark)
    assert q_dark.is_underexposed is True

    # 2. Glare / overexposed
    glare = np.full((120, 160, 3), 254, dtype=np.uint8)
    q_glare = assess_image_quality(glare)
    assert q_glare.is_overexposed is True
    assert q_glare.clipped_fraction > 0.5

    # 3. Directional motion blur (vertical step edge -> strong horizontal gradient, zero vertical gradient)
    stripe = np.full((128, 128, 3), 80, dtype=np.uint8)
    stripe[:, 64:] = 180
    q_blur = assess_image_quality(stripe, blur_threshold=5000.0)
    assert q_blur.is_blurry is True
    assert q_blur.is_motion_blurred is True

    # 4. Low-res image check
    low_res = np.random.default_rng(0).integers(40, 200, size=(120, 160, 3), dtype=np.uint8)
    q_low = assess_image_quality(low_res)
    assert q_low.width == 160 and q_low.height == 120


def test_difficult_images_depth_drywall_and_segmentation_failures() -> None:
    # 5. Glass / mirror / flat depth failure
    flat_depth = np.ones((64, 64), dtype=np.float32) * 2.0
    diag_flat = EngineFailureDetector.diagnose_depth(flat_depth)
    assert diag_flat.is_failure_detected is True
    assert diag_flat.failure_kind == FailureKind.DEPTH_INVERTED_OR_FLAT

    # 6. Out-of-bounds depth
    oob_depth = np.linspace(-1.0, 120.0, 64 * 64, dtype=np.float32).reshape(64, 64)
    diag_oob = EngineFailureDetector.diagnose_depth(oob_depth)
    assert diag_oob.is_failure_detected is True
    assert diag_oob.failure_kind == FailureKind.DEPTH_OUT_OF_BOUNDS

    # 7. Textureless painted drywall / repetitive structure starvation
    drywall_a = np.full((128, 128, 3), 210, dtype=np.uint8)
    drywall_b = np.full((128, 128, 3), 208, dtype=np.uint8)
    diag_feat = EngineFailureDetector.diagnose_feature_matching(drywall_a, drywall_b, num_inlier_matches=3)
    assert diag_feat.is_failure_detected is True
    assert diag_feat.failure_kind == FailureKind.TEXTURELESS_FEATURE_STARVATION

    # 8. Heavy occlusion / segmentation boundary leakage
    leaked_mask = np.zeros((100, 100), dtype=bool)
    leaked_mask[10:90, 10:90] = True  # 6400 px
    diag_seg = EngineFailureDetector.diagnose_segmentation_leakage(
        leaked_mask, prompt_box_xyxy=(40.0, 40.0, 60.0, 60.0), max_leak_ratio=1.40
    )
    assert diag_seg.is_failure_detected is True
    assert diag_seg.failure_kind == FailureKind.SEGMENTATION_BOUNDARY_LEAKAGE


def test_perception_to_worldir_and_worldstore_roundtrip(tmp_path: Path) -> None:
    # 1. ObservationRecord with mask -> MaskStore persistence
    mask = np.zeros((64, 64), dtype=bool)
    mask[16:40, 20:44] = True
    obs = ObservationRecord(
        evidence_id="obs:img0:switch:0",
        source_image_id="img0.jpg",
        coordinate_frame=CoordinateFrame(frame_id="image", scale_state="image", units="pixels"),
        model_id=ModelIdentity(
            engine_id="segmentation",
            architecture_id="sam2",
            checkpoint_id="sam2_hiera_tiny",
            license="Apache-2.0",
            hardware_class="cpu",
        ),
        label="switch",
        category="instance_segmentation",
        bbox_px=(20.0, 16.0, 44.0, 40.0),
        mask=mask,
        confidence=0.91,
        calibration_quality="unverified",
        metadata={"ontology_path": "fixtures/electrical/switch"},
    )
    mstore = MaskStore()
    stamped = persist_masks([obs], mstore)
    assert stamped[0].mask_ref == 0

    mask_npz = tmp_path / "masks.npz"
    save_mask_store(mstore, mask_npz)
    reloaded_masks = load_mask_store(mask_npz)
    assert np.array_equal(reloaded_masks.get(0), mask)

    # 2. Construct WorldIR with Wall + Switch + Spatial Reasoning -> WorldStore
    world = WorldIR(id="world-e2e-1")
    world.geometries["geom-wall"] = Geometry(
        id="geom-wall",
        type=GeometryType.BOX,
        bounds_min=Vector3(0.0, 0.0, 0.0),
        bounds_max=Vector3(4.0, 0.2, 2.8),
        provenance=Provenance.RECONSTRUCTED,
    )
    world.entities["wall-1"] = Entity(
        id="wall-1",
        type=EntityType.WALL,
        name="North Wall",
        transform={"position": {"x": 2.0, "y": 0.1, "z": 1.4}},
        geometry_ids=["geom-wall"],
        provenance=Provenance.RECONSTRUCTED,
        confidence=0.88,
    )
    world.geometries["geom-switch"] = Geometry(
        id="geom-switch",
        type=GeometryType.BOX,
        bounds_min=Vector3(1.5, 0.18, 1.2),
        bounds_max=Vector3(1.6, 0.22, 1.3),
        provenance=Provenance.OBSERVED,
    )
    world.entities["switch-1"] = Entity(
        id="switch-1",
        type=EntityType.SENSOR,
        name="switch",
        transform={"position": {"x": 2.0, "y": 0.2, "z": 1.4}},
        geometry_ids=["geom-switch"],
        semantic_labels=["switch"],
        provenance=Provenance.OBSERVED,
        confidence=0.91,
        custom_properties={
            "mask_ref": stamped[0].mask_ref,
            "ontology_path": "fixtures/electrical/switch",
        },
    )

    infer_spatial_relationships(world)

    ws_root = tmp_path / "worldstore"
    store1 = WorldStore(ws_root)
    ver = store1.save_version(world, parent=None)

    store2 = WorldStore(ws_root)
    loaded = store2.load_version(ver.version_id)
    assert "wall-1" in loaded.entities
    assert "switch-1" in loaded.entities
    assert loaded.entities["switch-1"].custom_properties["mask_ref"] == 0
    assert any(r.target_id == "wall-1" for r in loaded.entities["switch-1"].relationships)
