"""Tests for temporal tracking (P7-02): associating lifted 3D
observations of the same object ACROSS TIME into persistent tracks --
the identity-lifecycle layer the ledger names as P7-01's open item.

Contract (ledger P7-02 scope + ITrackBackend's honesty rule in
perception/instances/interface.py):

  - Observations chain in TIME order per label: an observation joins
    the track it continues iff the time gap is within `max_gap_s` AND
    the implied motion is physically plausible
    (distance <= max(merge_distance_m, max_speed_m_s * gap)).
  - An observation that fits no open track STARTS A NEW TRACK -- it is
    never forced into one (the interface's documented honesty rule).
  - Cross-label linking never happens.
  - Observations with NO capture timestamp are excluded from timed
    chains and REPORTED separately -- never silently mixed in, never
    dropped.
  - Implied speed is MEASURED (path length / duration), not claimed.
  - Deterministic: sorted iteration everywhere; ties broken by
    observation id.
  - `instance_tracks_from` adapts track records onto the existing
    InstanceTrack interface using the caller's real regions -- a
    missing region is an error, not a fabricated placeholder.
"""

from __future__ import annotations

import pytest

from engine.math import Vec3
from perception.instances.interface import InstanceTrack
from perception.instances.lifting import ObjectHypothesis3D
from perception.tracking.temporal import (
    TimedObservation,
    TrackRecord,
    instance_tracks_from,
    link_temporally,
)


def _obs(oid, x, t, label="chair", evidence="img-1", confidence=None):
    return TimedObservation(
        observation_id=oid,
        label=label,
        position=Vec3(x, 0.0, 0.0),
        evidence_id=evidence,
        captured_at=t,
        confidence=confidence,
    )


class TestChaining:
    def test_moving_object_chains_in_time_order(self):
        # One object moving 1 m/s: t=0,1,2,3 at x=0,1,2,3.
        obs = [
            _obs("r-d", 3.0, 3.0),
            _obs("r-b", 1.0, 1.0),
            _obs("r-a", 0.0, 0.0),
            _obs("r-c", 2.0, 2.0),
        ]
        tracks, _untimed = link_temporally(obs, max_speed_m_s=2.0, max_gap_s=5.0)
        assert len(tracks) == 1
        rec = tracks[0]
        assert [o.observation_id for o in rec.observations] == ["r-a", "r-b", "r-c", "r-d"]
        assert rec.observation_count == 4
        assert rec.duration_s == pytest.approx(3.0)

    def test_speed_gate_rejects_impossible_motion(self):
        # 10 m apart, 1 s gap: no physical object at max 2 m/s.
        obs = [_obs("a", 0.0, 0.0), _obs("b", 10.0, 1.0)]
        tracks, _u = link_temporally(obs, max_speed_m_s=2.0, max_gap_s=5.0)
        assert len(tracks) == 2

    def test_same_distance_longer_gap_is_plausible(self):
        # 10 m apart, 10 s gap: 1 m/s < 2 m/s max -> same track.
        obs = [_obs("a", 0.0, 0.0), _obs("b", 10.0, 10.0)]
        tracks, _u = link_temporally(obs, max_speed_m_s=2.0, max_gap_s=20.0)
        assert len(tracks) == 1

    def test_gap_gate_rejects_stale_links(self):
        # Plausible speed but the gap exceeds max_gap_s -> new track.
        obs = [_obs("a", 0.0, 0.0), _obs("b", 1.0, 100.0)]
        tracks, _u = link_temporally(obs, max_speed_m_s=2.0, max_gap_s=5.0)
        assert len(tracks) == 2

    def test_merge_distance_floor_still_applies(self):
        # Slow drift but jump larger than the static merge distance AND
        # the speed budget -> separate.
        obs = [_obs("a", 0.0, 0.0), _obs("b", 50.0, 1.0)]
        tracks, _u = link_temporally(obs, max_speed_m_s=2.0, max_gap_s=5.0,
                                 merge_distance_m=0.5)
        assert len(tracks) == 2

    def test_unlinkable_observation_starts_own_track(self):
        # Three stationary sightings + one teleport: the teleport must
        # not be forced onto the stationary track.
        obs = [
            _obs("s1", 0.0, 0.0), _obs("s2", 0.0, 1.0), _obs("s3", 0.0, 2.0),
            _obs("tp", 9.0, 3.0),
        ]
        tracks, _u = link_temporally(obs, max_speed_m_s=2.0, max_gap_s=5.0)
        assert len(tracks) == 2
        solo = [t for t in tracks if t.observation_count == 1][0]
        assert solo.observations[0].observation_id == "tp"


class TestSeparationAndHonesty:
    def test_labels_never_cross(self):
        obs = [
            _obs("c1", 0.0, 0.0, label="chair"),
            _obs("t1", 0.5, 0.5, label="table"),
            _obs("c2", 1.0, 1.0, label="chair"),
            _obs("t2", 1.5, 1.5, label="table"),
        ]
        tracks, _u = link_temporally(obs, max_speed_m_s=5.0, max_gap_s=5.0)
        assert len(tracks) == 2
        assert {t.label for t in tracks} == {"chair", "table"}

    def test_untimed_observations_reported_not_mixed(self):
        obs = [
            _obs("a", 0.0, 0.0),
            _obs("b", 1.0, 1.0),
            TimedObservation(
                observation_id="u", label="chair", position=Vec3(2.0, 0.0, 0.0),
                evidence_id="img-x", captured_at=None,
            ),
        ]
        tracks, untimed = link_temporally(obs, max_speed_m_s=2.0, max_gap_s=5.0)
        assert len(tracks) == 1
        assert [o.observation_id for o in tracks[0].observations] == ["a", "b"]
        assert [o.observation_id for o in untimed] == ["u"]

    def test_measured_implied_speed(self):
        obs = [_obs("a", 0.0, 0.0), _obs("b", 4.0, 2.0)]
        tracks, _u = link_temporally(obs, max_speed_m_s=5.0, max_gap_s=5.0)
        assert tracks[0].implied_speed_m_s == pytest.approx(2.0)

    def test_single_observation_track_speed_is_none(self):
        tracks, _u = link_temporally([_obs("a", 0.0, 0.0)], max_speed_m_s=2.0)
        assert tracks[0].implied_speed_m_s is None
        assert tracks[0].duration_s == 0.0

    def test_deterministic_tie_breaking(self):
        # Two observations at the same instant: id order decides.
        obs = [_obs("z", 0.0, 5.0), _obs("a", 0.0, 5.0)]
        tracks1, _u1 = link_temporally(obs, max_speed_m_s=2.0)
        tracks2, _u2 = link_temporally(list(reversed(obs)), max_speed_m_s=2.0)
        assert [t.to_dict() for t in tracks1] == [t.to_dict() for t in tracks2]


class TestInstanceTrackAdapter:
    def test_groups_real_regions_by_membership(self):
        from perception.segmentation.interface import SegmentedRegion

        def region(rid, evidence):
            return SegmentedRegion(
                region_id=rid, evidence_id=evidence, label="chair",
                mask=[[True]], confidence=0.9,
            )

        regions = {"r-a": region("r-a", "img-0"), "r-b": region("r-b", "img-1")}
        obs = [
            _obs("r-a", 0.0, 0.0, evidence="img-0", confidence=0.9),
            _obs("r-b", 1.0, 1.0, evidence="img-1", confidence=0.7),
        ]
        tracks, _u = link_temporally(obs, max_speed_m_s=2.0)
        assert tracks[0].confidence == 0.7  # min, not fabricated
        adapted = instance_tracks_from(tracks, regions)
        assert len(adapted) == 1
        assert isinstance(adapted[0], InstanceTrack)
        assert adapted[0].confidence == 0.7
        assert [r.region_id for r in adapted[0].regions] == ["r-a", "r-b"]

    def test_no_measured_confidence_refuses_instead_of_fabricating(self):
        obs = [_obs("r-a", 0.0, 0.0), _obs("r-b", 1.0, 1.0)]
        tracks, _u = link_temporally(obs, max_speed_m_s=2.0)
        assert tracks[0].confidence is None
        with pytest.raises(ValueError, match="confidence"):
            instance_tracks_from(tracks, {})

    def test_missing_region_is_an_error_not_a_placeholder(self):
        obs = [_obs("r-a", 0.0, 0.0, confidence=0.8), _obs("r-b", 1.0, 1.0, confidence=0.8)]
        tracks, _u = link_temporally(obs, max_speed_m_s=2.0)
        with pytest.raises(KeyError):
            instance_tracks_from(tracks, {})
