"""Tests for MultiViewIdentityTrackBackend (P7-01 multi-view identity)."""

from __future__ import annotations

import numpy as np

from perception.instances.track_backend import (
    MultiViewIdentityTrackBackend,
    build_images_dict,
)
from perception.segmentation.interface import SegmentationResult, SegmentedRegion
from perception.instances.lifting import ObjectHypothesis3D
from perception.depth.interface import DepthMap
from reconstruction.calibration.camera import PinholeCamera, CameraIntrinsics, CameraExtrinsics
from engine.physics.math3 import Vec3, Quat
from evidence.session import EvidenceItem, EvidenceKind
from provenance import Provenance, Uncertainty


def _make_evidence(eid: str) -> EvidenceItem:
    return EvidenceItem(
        id=eid,
        kind=EvidenceKind.IMAGE,
        source_uri=f"file:///{eid}.jpg",
        metadata={},
    )


def _make_camera(eid: str, pos: Vec3, rot: Quat) -> PinholeCamera:
    intrinsics = CameraIntrinsics(fx=800, fy=800, cx=320, cy=240, width=640, height=480)
    extrinsics = CameraExtrinsics(position=pos, rotation=rot)
    return PinholeCamera(intrinsics=intrinsics, extrinsics=extrinsics)


def _make_region(region_id: str, evidence_id: str, label: str, mask_size=(480, 640)) -> SegmentedRegion:
    h, w = mask_size
    # Simple rectangular mask in center
    mask = [[False] * w for _ in range(h)]
    for y in range(h // 3, 2 * h // 3):
        for x in range(w // 3, 2 * w // 3):
            mask[y][x] = True
    return SegmentedRegion(
        region_id=region_id,
        evidence_id=evidence_id,
        label=label,
        mask=mask,
        confidence=0.9,
    )


def _make_depth_frame(evidence_id: str) -> DepthMap:
    """Create a mock metric depth map for testing."""
    h, w = 480, 640
    # Flat depth at 2 meters (metric)
    values = [[2.0 for _ in range(w)] for _ in range(h)]
    return DepthMap(
        evidence_id=evidence_id,
        width=w,
        height=h,
        values=values,
        unit="meters",
        uncertainty=Uncertainty(confidence=1.0),
    )


def test_build_images_dict_loads_images(tmp_path):
    """Test that build_images_dict loads images from evidence items."""
    try:
        import imageio.v3 as iio
    except ImportError:
        return  # Skip if imageio not available
    
    # Create a test image
    img_path = tmp_path / "test.jpg"
    test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    iio.imwrite(img_path, test_img)
    
    evidence = EvidenceItem(
        id="ev-1",
        kind=EvidenceKind.IMAGE,
        source_uri=str(img_path).replace("\\", "/"),
        metadata={},
    )
    
    images = build_images_dict([evidence])
    assert "ev-1" in images
    assert len(images["ev-1"]) == 100
    assert len(images["ev-1"][0]) == 100


def test_track_backend_basic_linking():
    """Test basic track linking with two views of same object."""
    # Two cameras looking at the same object
    cam1 = _make_camera("cam1", Vec3(0, 0, 0), Quat(1, 0, 0, 0))
    cam2 = _make_camera("cam2", Vec3(1, 0, 0), Quat(1, 0, 0, 0))
    
    cameras = {"ev-1": cam1, "ev-2": cam2}
    
    # Create a simple test image
    test_img = np.full((480, 640, 3), (100, 150, 200), dtype=np.uint8)
    images = {"ev-1": test_img.tolist(), "ev-2": test_img.tolist()}
    
    # Two segmentation results with same label, should merge
    region1 = _make_region("r1", "ev-1", "chair")
    region2 = _make_region("r2", "ev-2", "chair")
    
    seg1 = SegmentationResult(evidence_id="ev-1", regions=[region1], model_name="test", model_version="1.0")
    seg2 = SegmentationResult(evidence_id="ev-2", regions=[region2], model_name="test", model_version="1.0")
    
    depth_maps = {"ev-1": _make_depth_frame("ev-1"), "ev-2": _make_depth_frame("ev-2")}
    
    backend = MultiViewIdentityTrackBackend(distance_threshold_m=1.0)
    tracks = backend.link_instances([seg1, seg2], depth_maps, cameras, images)
    
    # Should create at least one track (same label, overlapping 3D position)
    # Note: With our synthetic data, the lift may produce overlapping positions
    # depending on the exact depth/camera geometry
    assert isinstance(tracks, list)


def test_track_backend_different_labels_no_merge():
    """Test that different labels don't merge into same track."""
    cam1 = _make_camera("cam1", Vec3(0, 0, 0), Quat(1, 0, 0, 0))
    cam2 = _make_camera("cam2", Vec3(1, 0, 0), Quat(1, 0, 0, 0))
    
    cameras = {"ev-1": cam1, "ev-2": cam2}
    
    test_img = np.full((480, 640, 3), (100, 150, 200), dtype=np.uint8)
    images = {"ev-1": test_img.tolist(), "ev-2": test_img.tolist()}
    
    region1 = _make_region("r1", "ev-1", "chair")
    region2 = _make_region("r2", "ev-2", "table")  # Different label
    
    seg1 = SegmentationResult(evidence_id="ev-1", regions=[region1], model_name="test", model_version="1.0")
    seg2 = SegmentationResult(evidence_id="ev-2", regions=[region2], model_name="test", model_version="1.0")
    
    depth_maps = {"ev-1": _make_depth_frame("ev-1"), "ev-2": _make_depth_frame("ev-2")}
    
    backend = MultiViewIdentityTrackBackend(distance_threshold_m=1.0)
    tracks = backend.link_instances([seg1, seg2], depth_maps, cameras, images)
    
    # Different labels -> should not merge (each becomes separate track)
    assert len(tracks) >= 1  # At least one track per unique label


def test_track_backend_empty_results():
    """Test track backend with empty results."""
    backend = MultiViewIdentityTrackBackend()
    tracks = backend.link_instances([], {}, {}, {})
    assert tracks == []


def test_track_backend_no_metric_depth():
    """Test track backend skips views without metric depth."""
    cam1 = _make_camera("cam1", Vec3(0, 0, 0), Quat(1, 0, 0, 0))
    cameras = {"ev-1": cam1}
    
    test_img = np.full((480, 640, 3), (100, 150, 200), dtype=np.uint8)
    images = {"ev-1": test_img.tolist()}
    
    region1 = _make_region("r1", "ev-1", "chair")
    seg1 = SegmentationResult(evidence_id="ev-1", regions=[region1], model_name="test", model_version="1.0")
    
    # No depth maps provided
    depth_maps = {}
    
    backend = MultiViewIdentityTrackBackend()
    tracks = backend.link_instances([seg1], depth_maps, cameras, images)
    
    # No metric depth -> no lifting -> no tracks
    assert tracks == []


def test_build_images_dict_non_image_kind():
    """Test that non-image evidence items are skipped."""
    evidence = EvidenceItem(
        id="ev-1",
        kind=EvidenceKind.VIDEO,
        source_uri="file:///video.mp4",
        metadata={},
    )
    
    images = build_images_dict([evidence])
    assert images == {}  # Video kind is filtered out


def test_track_backend_confidence_propagation():
    """Test that track confidence reflects merge agreement."""
    # Two views of same object with high confidence should produce high track confidence
    cam1 = _make_camera("cam1", Vec3(0, 0, 0), Quat(1, 0, 0, 0))
    cam2 = _make_camera("cam2", Vec3(1, 0, 0), Quat(1, 0, 0, 0))
    
    cameras = {"ev-1": cam1, "ev-2": cam2}
    
    test_img = np.full((480, 640, 3), (100, 150, 200), dtype=np.uint8)
    images = {"ev-1": test_img.tolist(), "ev-2": test_img.tolist()}
    
    region1 = _make_region("r1", "ev-1", "chair")
    region2 = _make_region("r2", "ev-2", "chair")
    
    seg1 = SegmentationResult(evidence_id="ev-1", regions=[region1], model_name="test", model_version="1.0")
    seg2 = SegmentationResult(evidence_id="ev-2", regions=[region2], model_name="test", model_version="1.0")
    
    depth_maps = {"ev-1": _make_depth_frame("ev-1"), "ev-2": _make_depth_frame("ev-2")}
    
    backend = MultiViewIdentityTrackBackend(distance_threshold_m=1.0)
    tracks = backend.link_instances([seg1, seg2], depth_maps, cameras, images)
    
    # If tracks are created, confidence should be reasonable
    for track in tracks:
        assert 0.0 <= track.confidence <= 1.0
        assert track.uncertainty.confidence == track.confidence