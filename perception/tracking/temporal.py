"""Temporal tracking (P7-02): associating 3D observations of the same
object ACROSS TIME into persistent tracks -- the identity-lifecycle
layer of the multi-view identity chain (ledger P7-01's declared open
item; P7-02 scope: "object/camera/scene tracks with track_id/state/
velocity/history/confidence/uncertainty").

Position in the perception stack:

    per-frame lifting (instances/lifting.py)
    -> multi-view resolution (instances/object_resolution.py:
       same-moment cross-view association)
    -> TEMPORAL TRACKING (this module: same-object association
       across time -> persistent track with history + implied speed)
    -> WorldIR entity

Method: per label, process observations in time order; each joins the
open track it continues iff BOTH gates pass --
  speed gate: distance <= max(merge_distance_m, max_speed_m_s * gap)
  gap gate:   gap <= max_gap_s
-- else it starts a NEW track. An observation is never forced into a
track it does not fit (perception/instances/interface.py's documented
honesty rule: an ungrouped observation is the honest failure mode).

Observations with NO capture timestamp are excluded from timed chains
and returned separately -- silently mixing untimed data into timed
chains would corrupt every duration/speed measurement downstream.

All quantities are measured (path length, duration, implied speed =
path length / duration), never claimed. Deterministic: sorted
iteration, ties broken by observation_id.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from engine.math import Vec3

#: Defaults; a caller overrides per scene (not tuned against any real
#: dataset -- the same honesty as object_resolution's thresholds).
DEFAULT_MAX_SPEED_M_S = 2.0
DEFAULT_MAX_GAP_S = 5.0
DEFAULT_MERGE_DISTANCE_M = 0.5


@dataclass(frozen=True)
class TimedObservation:
    """One 3D observation with its capture time. `captured_at` None
    means the timestamp was not recorded -- unknown stays unknown."""

    observation_id: str
    label: str
    position: Vec3
    evidence_id: str
    captured_at: Optional[float]
    #: Measured observation confidence (e.g. segmentation x valid-depth
    #: fraction from the lifting path), when the caller has one.
    #: None = not measured; unknown stays unknown, never defaulted.
    confidence: Optional[float] = None


@dataclass(frozen=True)
class TrackRecord:
    """A persistent track: the observation history plus MEASURED
    motion facts. `implied_speed_m_s` is path length over duration --
    None for single-observation tracks (no motion is measurable), not
    zero."""

    track_id: str
    label: str
    observations: Tuple[TimedObservation, ...]
    path_length_m: float
    duration_s: float
    implied_speed_m_s: Optional[float]
    #: Min of the member observations' measured confidences; None when
    #: no member carries one (never fabricated to 1.0).
    confidence: Optional[float] = None

    @property
    def observation_count(self) -> int:
        return len(self.observations)

    @property
    def evidence_ids(self) -> Tuple[str, ...]:
        return tuple(sorted({o.evidence_id for o in self.observations}))

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "label": self.label,
            "observation_count": self.observation_count,
            "observation_ids": [o.observation_id for o in self.observations],
            "evidence_ids": list(self.evidence_ids),
            "path_length_m": self.path_length_m,
            "duration_s": self.duration_s,
            "implied_speed_m_s": self.implied_speed_m_s,
        }


def _dist(a: Vec3, b: Vec3) -> float:
    return math.sqrt(
        (a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2
    )


def link_temporally(
    observations: Sequence[TimedObservation],
    max_speed_m_s: float = DEFAULT_MAX_SPEED_M_S,
    max_gap_s: float = DEFAULT_MAX_GAP_S,
    merge_distance_m: float = DEFAULT_MERGE_DISTANCE_M,
) -> Tuple[List[TrackRecord], List[TimedObservation]]:
    """Chain observations into temporal tracks, per label, in time
    order. Returns (tracks, untimed) -- the second list holds
    observations with no capture timestamp, reported so the caller can
    surface them rather than silently dropping or mixing them.

    A duplicate observation_id is a caller bug and raises -- silent
    dedup would hide real data problems.
    """
    if max_speed_m_s <= 0:
        raise ValueError("max_speed_m_s must be positive")
    if max_gap_s <= 0:
        raise ValueError("max_gap_s must be positive")
    if merge_distance_m < 0:
        raise ValueError("merge_distance_m must be non-negative")

    ids = [o.observation_id for o in observations]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate observation_id in input -- a caller bug")

    timed = sorted(
        (o for o in observations if o.captured_at is not None),
        key=lambda o: (o.captured_at, o.observation_id),
    )
    untimed = sorted(
        (o for o in observations if o.captured_at is None),
        key=lambda o: o.observation_id,
    )

    # Open tracks per label; each is a mutable list of observations
    # plus its last position/time.
    open_tracks: List[dict] = []
    by_label: Dict[str, List[int]] = {}
    for obs in timed:
        best_idx = None
        best_gap = None
        for idx in reversed(by_label.get(obs.label, [])):
            tr = open_tracks[idx]
            gap = obs.captured_at - tr["last_t"]
            if gap > max_gap_s or gap < 0:
                continue
            budget = max(merge_distance_m, max_speed_m_s * gap)
            if _dist(obs.position, tr["last_pos"]) <= budget:
                # Among open tracks of this label, take the most
                # recent continuation (reversed iteration; first hit).
                if best_idx is None or gap < best_gap:
                    best_idx = idx
                    best_gap = gap
        if best_idx is None:
            open_tracks.append({
                "label": obs.label,
                "obs": [obs],
                "last_t": obs.captured_at,
                "last_pos": obs.position,
                "path": 0.0,
            })
            by_label.setdefault(obs.label, []).append(len(open_tracks) - 1)
        else:
            tr = open_tracks[best_idx]
            step = _dist(obs.position, tr["last_pos"])
            tr["obs"].append(obs)
            tr["path"] += step
            tr["last_t"] = obs.captured_at
            tr["last_pos"] = obs.position

    tracks: List[TrackRecord] = []
    for i, tr in enumerate(open_tracks):
        obs_list = tr["obs"]
        duration = obs_list[-1].captured_at - obs_list[0].captured_at
        speed = (tr["path"] / duration) if duration > 0 else None
        confidences = [o.confidence for o in obs_list if o.confidence is not None]
        confidence = min(confidences) if confidences else None
        tracks.append(TrackRecord(
            track_id=f"track-{tr['label']}-{i:04d}",
            label=tr["label"],
            observations=tuple(obs_list),
            path_length_m=tr["path"],
            duration_s=duration,
            implied_speed_m_s=speed,
            confidence=confidence,
        ))
    tracks.sort(key=lambda t: (t.label, t.observations[0].captured_at,
                               t.observations[0].observation_id))
    return tracks, untimed


def instance_tracks_from(
    tracks: Sequence[TrackRecord],
    regions_by_id: Dict[str, "object"],
) -> List["InstanceTrack"]:
    """Adapt TrackRecords onto the existing InstanceTrack interface,
    attaching the caller's REAL SegmentedRegions by observation_id.

    Raises KeyError when an observation's region is missing -- a
    fabricated placeholder region would be exactly the dishonesty the
    tracking contract forbids. Raises ValueError for a track with no
    measured confidence -- InstanceTrack.confidence is a required
    number, and writing 1.0 where nothing was measured would lie.
    (Import is local: perception.instances is a sibling subsystem, and
    deferring it keeps this module importable from tooling that only
    needs TrackRecords.)
    """
    from perception.instances.interface import InstanceTrack

    out: List[InstanceTrack] = []
    for tr in tracks:
        if tr.confidence is None:
            raise ValueError(
                f"track {tr.track_id!r} has no measured confidence -- "
                f"supply per-observation confidences instead of "
                f"fabricating one"
            )
        regions = []
        for obs in tr.observations:
            region = regions_by_id.get(obs.observation_id)
            if region is None:
                raise KeyError(
                    f"no region supplied for observation "
                    f"{obs.observation_id!r} -- refusing to fabricate a "
                    f"placeholder region"
                )
            regions.append(region)
        out.append(InstanceTrack(
            track_id=tr.track_id,
            regions=regions,
            label=tr.label,
            confidence=tr.confidence,
        ))
    return out
