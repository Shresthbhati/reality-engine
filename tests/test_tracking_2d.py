"""Tests for the P7-02 temporal-tracking completion: 2D box
association (BoxTrack/AssociationResult/track_boxes_2d) on top of the
existing 3D link_temporally chains.

Directive section 14 rules under test:
  - every association produces an explicit AssociationResult
    (matched / new_track / ambiguous) with measured scores;
  - an ambiguous candidate (>=2 open tracks within ambiguity_ratio)
    is NOT silently merged -- it starts a new track and the decision
    records the risk;
  - ID-switch diagnostics: a track member that joined with a runner-up
    in the provisional band is flagged on the track;
  - camera-motion compensation is the caller's explicit input (the
    tracker compares what it is given and never invents ego-motion);
  - deterministic ordering; duplicate ids raise.
"""

import math

import pytest

from perception.tracking.temporal import (
    AssociationResult,
    Detection2D,
    track_boxes_2d,
)


def _det(did, label, box, t, eid="img-0"):
    return Detection2D(
        detection_id=did, label=label, box=box,
        evidence_id=eid, captured_at=t,
    )


class TestTrackBoxes2D:
    def test_stationary_object_becomes_one_track(self):
        dets = [
            _det("d1", "chair", (10, 10, 50, 50), 0.0),
            _det("d2", "chair", (11, 11, 51, 51), 0.5),
            _det("d3", "chair", (12, 12, 52, 52), 1.0),
        ]
        tracks, decisions = track_boxes_2d(dets)
        assert len(tracks) == 1
        assert tracks[0].observation_count == 3
        assert [d.decision for d in decisions] == ["new_track", "matched", "matched"]
        assert tracks[0].duration_s == pytest.approx(1.0)

    def test_fast_object_starts_new_track_not_forced_merge(self):
        # Two far-apart detections of the same label: no gate passed.
        dets = [
            _det("d1", "car", (0, 0, 40, 40), 0.0),
            _det("d2", "car", (500, 500, 540, 540), 0.5),
        ]
        tracks, decisions = track_boxes_2d(dets)
        assert len(tracks) == 2
        assert all(d.decision == "new_track" for d in decisions)

    def test_ambiguous_runner_up_refused_not_silently_merged(self):
        # Two heavily overlapping open tracks; the next detection sits
        # between them (>= min_iou to both) -> ambiguous, new track.
        dets = [
            _det("a1", "person", (0, 0, 100, 100), 0.0),
            _det("b1", "person", (30, 0, 130, 100), 0.0),
            _det("x", "person", (15, 0, 115, 100), 0.5),
        ]
        tracks, decisions = track_boxes_2d(dets)
        assert len(tracks) == 3
        amb = [d for d in decisions if d.decision == "ambiguous"]
        assert len(amb) == 1
        assert amb[0].detection_id == "x"
        assert amb[0].ambiguity >= 0.7

    def test_id_switch_diagnostic_flags_provisional_links(self):
        # The provisional band is a documented caller knob (under pure
        # IoU greedy association it is geometrically narrow -- see the
        # module note). b1 seeds its own track; a2's best IoU is 0.82
        # (track a), its runner-up IoU 0.29 (track b, inside the motion
        # budget) -> ratio 0.354. With provisional_ratio=0.25 the match
        # is kept but flagged as an ID-switch risk on the track.
        dets = [
            _det("a1", "person", (0, 0, 100, 100), 0.0),
            _det("b1", "person", (65, 0, 165, 100), 0.1),
            _det("a2", "person", (10, 0, 110, 100), 0.5),
        ]
        tracks, decisions = track_boxes_2d(
            dets, max_distance_px=0.6, provisional_ratio=0.25
        )
        assert len(tracks) == 2
        matched = [d for d in decisions if d.decision == "matched" and d.detection_id == "a2"]
        assert matched and matched[0].ambiguity >= 0.25
        flagged = [t for t in tracks if t.ambiguous_links]
        assert len(flagged) == 1
        assert flagged[0].ambiguous_links == ("a2",)

    def test_labels_do_not_cross_associate(self):
        dets = [
            _det("c1", "chair", (10, 10, 50, 50), 0.0),
            _det("t1", "table", (10, 10, 50, 50), 0.5),
        ]
        tracks, _ = track_boxes_2d(dets)
        assert len(tracks) == 2
        assert {t.label for t in tracks} == {"chair", "table"}

    def test_untimed_detections_are_excluded_and_reported_via_decisions(self):
        dets = [
            _det("u1", "chair", (10, 10, 50, 50), None),
            _det("d2", "chair", (11, 11, 51, 51), 1.0),
        ]
        tracks, decisions = track_boxes_2d(dets)
        # The untimed detection is never associated (would corrupt the
        # time order); it becomes its own track and every association
        # is explicit.
        assert len(tracks) == 2
        assert all(d.decision == "new_track" for d in decisions)

    def test_duplicate_detection_id_raises(self):
        dets = [
            _det("d1", "chair", (10, 10, 50, 50), 0.0),
            _det("d1", "chair", (11, 11, 51, 51), 0.5),
        ]
        with pytest.raises(ValueError):
            track_boxes_2d(dets)

    def test_deterministic_order(self):
        dets = [
            _det("d1", "chair", (10, 10, 50, 50), 0.0),
            _det("d2", "table", (30, 30, 90, 90), 0.2),
            _det("d3", "chair", (12, 12, 52, 52), 0.4),
        ]
        t1, dec1 = track_boxes_2d(dets)
        t2, dec2 = track_boxes_2d(list(reversed(dets)))
        assert [t.to_dict() for t in t1] == [t.to_dict() for t in t2]
        assert [d.to_dict() for d in dec1] == [d.to_dict() for d in dec2]

    def test_camera_motion_compensation_is_caller_supplied(self):
        """A translating camera shifts every box by the same amount:
        UNcompensated, the static object's boxes diverge -> new tracks
        (honest). Compensated (caller shifts boxes back), the same
        object links into one track. The tracker never invents the
        ego-motion either way."""
        shift = 200.0
        uncompensated = [
            _det("d1", "chair", (10, 10, 50, 50), 0.0),
            _det("d2", "chair", (10 + shift, 10, 50 + shift, 50), 0.5),
        ]
        t_u, _ = track_boxes_2d(uncompensated)
        assert len(t_u) == 2

        compensated = [
            _det("d1", "chair", (10, 10, 50, 50), 0.0),
            _det("d2", "chair", (10, 10, 50, 50), 0.5),
        ]
        t_c, d_c = track_boxes_2d(compensated)
        assert len(t_c) == 1
        assert d_c[1].decision == "matched"
