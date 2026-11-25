"""Tests for the perception adapter interfaces (IDepthBackend,
ISegmentationBackend). No concrete backend exists yet -- these tests
verify the interface contract itself (shape validation, ABC enforcement)
using minimal fake implementations, the same pattern as
test_colmap_backend.py's honest-unavailability tests for IReconstructionBackend.
"""

import pytest

from evidence.session import EvidenceItem, EvidenceKind
from perception.depth.interface import DepthMap, IDepthBackend
from perception.segmentation.interface import ISegmentationBackend, SegmentationResult, SegmentedRegion
from provenance import Uncertainty


def _photo(id_):
    return EvidenceItem(id=id_, kind=EvidenceKind.PHOTO, source_uri=f"file://{id_}.jpg")


# ---- DepthMap ----

def test_depth_map_validates_row_count():
    with pytest.raises(ValueError):
        DepthMap(evidence_id="p1", width=2, height=2, values=[[1.0, 2.0]])  # only 1 row, expected 2


def test_depth_map_validates_row_width():
    with pytest.raises(ValueError):
        DepthMap(evidence_id="p1", width=3, height=1, values=[[1.0, 2.0]])  # row has 2 cols, expected 3


def test_depth_map_accepts_well_formed_grid():
    depth = DepthMap(evidence_id="p1", width=2, height=2, values=[[1.0, 2.0], [3.0, 4.0]])
    assert depth.unit == "relative"
    assert depth.quantity == "depth"          # default: larger = farther
    assert depth.valid_mask is None           # no mask claimed unless the producer supplies one
    assert depth.invalid_pixels is None
    assert depth.values[1][1] == 4.0


def test_depth_map_rejects_an_unknown_quantity():
    with pytest.raises(ValueError, match="quantity must be"):
        DepthMap(evidence_id="p1", width=1, height=1, values=[[1.0]], quantity="disparity")


def test_depth_map_validates_the_mask_shape():
    with pytest.raises(ValueError, match="valid_mask"):
        DepthMap(evidence_id="p1", width=2, height=2, values=[[1.0, 2.0], [3.0, 4.0]],
                 valid_mask=[[True, True]])


def test_depth_map_counts_pixels_the_producer_could_not_measure():
    depth = DepthMap(evidence_id="p1", width=2, height=2,
                     values=[[1.0, float("nan")], [3.0, 4.0]],
                     valid_mask=[[True, False], [True, True]])
    assert depth.invalid_pixels == 1
    assert DepthMap(evidence_id="p1", width=1, height=1, values=[[float("nan")]],
                    valid_mask=[[False]]).invalid_pixels == 1


# ---- IDepthBackend ----

def test_idepth_backend_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        IDepthBackend()


class _FakeDepthBackend(IDepthBackend):
    """Minimal real implementation used only to exercise the interface
    contract -- not a claim that real depth estimation is implemented."""

    def estimate_depth(self, evidence):
        maps = []
        for item in evidence:
            if item.kind != EvidenceKind.PHOTO:
                continue  # honest skip, not a fabricated map
            maps.append(DepthMap(evidence_id=item.id, width=1, height=1, values=[[1.0]]))
        return maps


def test_fake_depth_backend_skips_unsupported_evidence_kind():
    backend = _FakeDepthBackend()
    lidar = EvidenceItem(id="l1", kind=EvidenceKind.LIDAR, source_uri="file://l1.las")
    result = backend.estimate_depth([_photo("p1"), lidar])
    assert [m.evidence_id for m in result] == ["p1"]


# ---- SegmentedRegion / SegmentationResult ----

def test_segmented_region_validates_confidence_bounds():
    with pytest.raises(ValueError):
        SegmentedRegion(region_id="r1", evidence_id="p1", label="wall", mask=[[True]], confidence=1.5)


def test_segmented_region_accepts_valid_confidence():
    region = SegmentedRegion(region_id="r1", evidence_id="p1", label="wall", mask=[[True]], confidence=0.9)
    assert region.uncertainty == Uncertainty()


# ---- ISegmentationBackend ----

def test_isegmentation_backend_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        ISegmentationBackend()


class _FakeSegmentationBackend(ISegmentationBackend):
    def segment(self, evidence):
        results = []
        for item in evidence:
            if item.kind != EvidenceKind.PHOTO:
                continue
            region = SegmentedRegion(
                region_id=f"reg-{item.id}", evidence_id=item.id, label="wall",
                mask=[[True]], confidence=0.8,
            )
            results.append(SegmentationResult(evidence_id=item.id, regions=[region], model_name="fake-v0"))
        return results


def test_fake_segmentation_backend_labels_are_backend_owned_not_invented_by_interface():
    backend = _FakeSegmentationBackend()
    results = backend.segment([_photo("p1")])
    assert len(results) == 1
    assert results[0].model_name == "fake-v0"
    assert results[0].regions[0].label == "wall"


def test_fake_segmentation_backend_skips_unsupported_evidence_kind():
    backend = _FakeSegmentationBackend()
    lidar = EvidenceItem(id="l1", kind=EvidenceKind.LIDAR, source_uri="file://l1.las")
    results = backend.segment([lidar])
    assert results == []


def test_depth_map_accepts_a_frozen_float32_array_and_still_validates():
    import numpy as np
    import pytest

    from perception.depth.interface import DepthMap

    arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    dm = DepthMap("e", width=4, height=3, values=arr, unit="meters")
    assert dm.values[1][2] == 6.0 and len(dm.values) == 3 and len(dm.values[0]) == 4
    with pytest.raises(ValueError):
        dm.values[0, 0] = 1.0                                   # shared between stages: read-only
    with pytest.raises(ValueError):
        DepthMap("e", width=5, height=3, values=np.zeros((3, 4), np.float32))      # width mismatch is still caught
