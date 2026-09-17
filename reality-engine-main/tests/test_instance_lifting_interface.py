"""Tests for the instance-lifting interface (ITrackBackend). No concrete
backend exists yet -- these tests verify the interface contract itself
(shape validation, ABC enforcement) using a minimal fake implementation,
the same pattern as test_perception_interfaces.py's fakes for
IDepthBackend/ISegmentationBackend.
"""

import pytest

from perception.instances.interface import InstanceTrack, ITrackBackend
from perception.segmentation.interface import SegmentationResult, SegmentedRegion
from provenance import Uncertainty


def _region(region_id, evidence_id, label="chair", confidence=0.9):
    return SegmentedRegion(
        region_id=region_id, evidence_id=evidence_id, label=label,
        mask=[[True]], confidence=confidence,
    )


# ---- InstanceTrack ----

def test_instance_track_validates_confidence_bounds():
    with pytest.raises(ValueError):
        InstanceTrack(track_id="t1", regions=[_region("r1", "e1")], label="chair", confidence=1.5)


def test_instance_track_accepts_valid_confidence():
    track = InstanceTrack(track_id="t1", regions=[_region("r1", "e1")], label="chair", confidence=0.8)
    assert track.uncertainty == Uncertainty()
    assert track.label == "chair"


# ---- ITrackBackend ----

def test_itrack_backend_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        ITrackBackend()


class _FakeTrackBackend(ITrackBackend):
    """Minimal real implementation used only to exercise the interface
    contract -- groups same-label regions within one call into a single
    track. This is a TOY fake for testing the interface shape only; it
    is NOT a claim that real cross-view identity association (appearance
    embeddings, geometric consistency, etc.) is implemented."""

    def link_instances(self, results):
        by_label = {}
        for result in results:
            for region in result.regions:
                by_label.setdefault(region.label, []).append(region)

        tracks = []
        for label, regions in by_label.items():
            tracks.append(InstanceTrack(
                track_id=f"track-{label}", regions=regions, label=label, confidence=0.7,
            ))
        return tracks


def test_fake_track_backend_groups_same_label_regions_across_evidence_items():
    backend = _FakeTrackBackend()
    results = [
        SegmentationResult(evidence_id="e1", regions=[_region("r1", "e1", "chair")], model_name="fake-v0"),
        SegmentationResult(evidence_id="e2", regions=[_region("r2", "e2", "chair")], model_name="fake-v0"),
        SegmentationResult(evidence_id="e3", regions=[_region("r3", "e3", "table")], model_name="fake-v0"),
    ]

    tracks = backend.link_instances(results)

    chair_tracks = [t for t in tracks if t.label == "chair"]
    assert len(chair_tracks) == 1
    assert {r.evidence_id for r in chair_tracks[0].regions} == {"e1", "e2"}

    table_tracks = [t for t in tracks if t.label == "table"]
    assert len(table_tracks) == 1
    assert {r.evidence_id for r in table_tracks[0].regions} == {"e3"}


def test_fake_track_backend_on_empty_input_produces_no_tracks():
    backend = _FakeTrackBackend()
    assert backend.link_instances([]) == []
