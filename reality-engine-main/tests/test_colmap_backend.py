"""Tests for the COLMAP backend's parsing logic and honest-unavailability path.

Parsing is tested against COLMAP's own documented text-output format
(https://colmap.github.io/format.html) so it's verified without the
colmap binary installed. reconstruct() itself is only tested for the
"colmap not on PATH" failure -- this repo doesn't have COLMAP installed,
so that's the real, exercisable path here.
"""

import pytest

from evidence.session import EvidenceItem, EvidenceKind
from reconstruction.backend.colmap_backend import (
    ColmapReconstructionBackend,
    ReconstructionBackendUnavailableError,
    _confidence_from_reprojection_error,
    _parse_images_txt,
    _parse_points3d_txt,
)

IMAGES_TXT = """\
# Image list with two lines of data per image:
#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
#   POINTS2D[] as (X, Y, POINT3D_ID)
# Number of images: 2, mean observations per image: 1
1 0.851773 0.0165051 0.503764 -0.142941 -0.737434 1.02973 3.74354 1 ev-p1.jpg
2362.39 248.498 63390
2 0.9 0.0 0.0 0.1 1.0 2.0 3.0 1 ev-p2.jpg
100.0 200.0 63390
"""

POINTS3D_TXT = """\
# 3D point list with one line of data per point:
#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)
# Number of points: 1, mean track length: 2.0
63390 1.67241 0.191763 0.466115 154 174 102 1.14007 1 0 2 0
"""


def test_parse_images_txt_extracts_pose_lines_only():
    poses = _parse_images_txt(IMAGES_TXT, evidence_id_by_name={})
    assert len(poses) == 2
    assert poses[0].evidence_id == "ev-p1.jpg"
    assert poses[0].position == (-0.737434, 1.02973, 3.74354)
    assert poses[0].rotation == (0.851773, 0.0165051, 0.503764, -0.142941)


def test_parse_images_txt_maps_filename_back_to_evidence_id():
    poses = _parse_images_txt(IMAGES_TXT, evidence_id_by_name={"ev-p1.jpg": "p1", "ev-p2.jpg": "p2"})
    assert [p.evidence_id for p in poses] == ["p1", "p2"]


def test_parse_points3d_txt_extracts_position_and_track():
    points = _parse_points3d_txt(POINTS3D_TXT, image_id_to_evidence_id={"1": "p1", "2": "p2"})
    assert len(points) == 1
    assert points[0].position == (1.67241, 0.191763, 0.466115)
    assert points[0].track_id == "63390"
    assert points[0].source_evidence_ids == ["p1", "p2"]


def test_confidence_from_reprojection_error_is_bounded_and_monotonic():
    low_error_conf = _confidence_from_reprojection_error(0.1)
    high_error_conf = _confidence_from_reprojection_error(10.0)
    assert 0.0 < high_error_conf < low_error_conf <= 1.0


def test_reconstruct_raises_when_colmap_binary_missing():
    backend = ColmapReconstructionBackend(colmap_binary="definitely-not-a-real-binary-xyz")
    with pytest.raises(ReconstructionBackendUnavailableError):
        backend.reconstruct([
            EvidenceItem(id="p1", kind=EvidenceKind.PHOTO, source_uri="file://a.jpg"),
            EvidenceItem(id="p2", kind=EvidenceKind.PHOTO, source_uri="file://b.jpg"),
        ])


def test_reconstruct_reports_failed_status_with_fewer_than_two_photos_without_touching_colmap():
    """Evidence-count validation happens before the PATH check -- cheap,
    unconditionally-correct rejections shouldn't depend on COLMAP being
    installed at all."""
    backend = ColmapReconstructionBackend(colmap_binary="definitely-not-a-real-binary-xyz")
    result = backend.reconstruct([EvidenceItem(id="p1", kind=EvidenceKind.PHOTO, source_uri="file://a.jpg")])
    assert result.registration_status == "failed"
