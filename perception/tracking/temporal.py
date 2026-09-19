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


def _iou(a, b) -> float:
    """Intersection-over-union of two (x0, y0, x1, y1) boxes."""
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass(frozen=True)
class AssociationResult:
    """The outcome of ONE frame-to-frame 2D association decision
    (directive section 14: never silently merge; every association
    carries its measured scores and an explicit decision).

    decision is "matched" | "new_track" | "ambiguous". An ambiguous
    candidate pair is NOT merged (the honest failure mode); it is
    reported so the caller can surface it.
    """

    decision: str
    track_id: Optional[str] = None
    detection_id: Optional[str] = None
    iou: float = 0.0
    distance_px: Optional[float] = None
    #: Ambiguity = runner-up IoU / best IoU (0 when no runner-up).
    ambiguity: float = 0.0
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "decision": self.decision,
            "track_id": self.track_id,
            "detection_id": self.detection_id,
            "iou": self.iou,
            "distance_px": self.distance_px,
            "ambiguity": self.ambiguity,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class Detection2D:
    """One 2D detection box at one timestamp (the P7-02 chain's
    front end: 2D detection -> 2D tracking -> multi-view association)."""

    detection_id: str
    label: str
    box: Tuple[float, float, float, float]  # (x0, y0, x1, y1) pixels
    evidence_id: str
    captured_at: Optional[float]
    #: Compensated camera translation between frames (px/frame in the
    # image plane, from the known camera motion): boxes are compared
    # AFTER compensation so a moving camera does not fabricate object
    # motion. Optional; zero when the camera is static.

    @property
    def center(self) -> Tuple[float, float]:
        return (
            (self.box[0] + self.box[2]) / 2.0,
            (self.box[1] + self.box[3]) / 2.0,
        )


@dataclass(frozen=True)
class BoxTrack:
    """A 2D box track: the detection history plus MEASURED motion.

    `id_switches` records, per member detection, whether it was linked
    despite an ambiguous runner-up (directive section 14's ID-switch
    diagnostic) -- reported, never hidden.
    """

    track_id: str
    label: str
    detections: Tuple[Detection2D, ...]
    #: Detection ids that joined an open track while a runner-up IoU
    # was within AMBIGUITY_RATIO of the winner's -- the measured
    # ID-switch risk set.
    ambiguous_links: Tuple[str, ...] = ()

    @property
    def observation_count(self) -> int:
        return len(self.detections)

    @property
    def duration_s(self) -> Optional[float]:
        """Measured track duration; None for single-detection tracks
        (no duration is measurable), not zero."""
        if len(self.detections) < 2:
            return None
        d0, d1 = self.detections[0], self.detections[-1]
        if d0.captured_at is None or d1.captured_at is None:
            return None
        return d1.captured_at - d0.captured_at

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "label": self.label,
            "observation_count": self.observation_count,
            "detection_ids": [d.detection_id for d in self.detections],
            "evidence_ids": sorted({d.evidence_id for d in self.detections}),
            "ambiguous_links": list(self.ambiguous_links),
        }


def track_boxes_2d(
    detections: Sequence[Detection2D],
    max_distance_px: float = 0.25,
    min_iou: float = 0.1,
    ambiguity_ratio: float = 0.7,
    provisional_ratio: float = 0.5,
) -> Tuple[List[BoxTrack], List[AssociationResult]]:
    """Associate 2D detections into box tracks, per label, in time
    order (ByteTrack-style IoU association, kept honest and
    deterministic).

    Gates (defaults are documented deferrals, not tuned to a dataset):
      - IoU >= min_iou between the detection and the track's last box
        (shape+position consistency),
      - center distance <= max_distance_px * max(box_w, box_h) of the
        track's last box (motion budget, scale-relative).
    Camera-motion compensation: the caller supplies boxes ALREADY
    compensated (see Detection2D note) -- this function compares what
    it is given and does not invent ego-motion.

    A detection matching >= 2 open tracks within ambiguity_ratio is
    AMBIGUOUS: it starts a NEW track rather than silently merging
    (the ID-switch risk is reported, not taken). A matched detection
    whose runner-up reaches `provisional_ratio` of the winner is
    flagged on its track (ambiguous_links) -- the measured ID-switch
    risk band. (Under pure-IoU greedy association the provisional
    band is geometrically narrow -- heavy overlap tends to fall
    through to either clear-match or full ambiguity -- which is why
    the band is a documented, untuned caller knob.)

    Returns (tracks, decisions): every association's explicit outcome.
    """
    if not (0.0 <= min_iou <= 1.0):
        raise ValueError("min_iou must be in [0, 1]")
    if max_distance_px <= 0:
        raise ValueError("max_distance_px must be positive")
    if not (0.0 < ambiguity_ratio <= 1.0):
        raise ValueError("ambiguity_ratio must be in (0, 1]")
    if not (0.0 <= provisional_ratio < ambiguity_ratio):
        raise ValueError(
            "provisional_ratio must be in [0, ambiguity_ratio)"
        )

    ids = [d.detection_id for d in detections]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate detection_id in input -- a caller bug")

    timed = sorted(
        (d for d in detections if d.captured_at is not None),
        key=lambda d: (d.captured_at, d.detection_id),
    )
    decisions: List[AssociationResult] = []

    open_tracks: List[dict] = []
    by_label: Dict[str, List[int]] = {}

    for det in timed:
        candidates = []
        for idx in reversed(by_label.get(det.label, [])):
            tr = open_tracks[idx]
            iou = _iou(det.box, tr["last_box"])
            if iou < min_iou:
                continue
            w = tr["last_box"][2] - tr["last_box"][0]
            h = tr["last_box"][3] - tr["last_box"][1]
            scale = max(w, h)
            dx = det.center[0] - tr["last_center"][0]
            dy = det.center[1] - tr["last_center"][1]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist > max_distance_px * scale:
                continue
            candidates.append((idx, iou, dist))
        if not candidates:
            decisions.append(AssociationResult(
                decision="new_track", detection_id=det.detection_id,
                reason="no open track passed the IoU + motion gates",
            ))
            open_tracks.append({
                "label": det.label, "dets": [det],
                "last_box": det.box, "last_center": det.center,
                "track_id": f"boxtrack-{det.label}-{len(open_tracks):04d}",
            })
            by_label.setdefault(det.label, []).append(len(open_tracks) - 1)
            continue

        candidates.sort(key=lambda c: (-c[1], open_tracks[c[0]]["dets"][0].detection_id))
        best_idx, best_iou, best_dist = candidates[0]
        runner_up = candidates[1][1] if len(candidates) > 1 else 0.0
        ambiguity = runner_up / best_iou if best_iou > 0 else 0.0
        if len(candidates) > 1 and ambiguity >= ambiguity_ratio:
            # Refuse the risky merge: a new track with the diagnostic.
            decisions.append(AssociationResult(
                decision="ambiguous", detection_id=det.detection_id,
                iou=best_iou, distance_px=best_dist, ambiguity=ambiguity,
                reason=(
                    f"{len(candidates)} tracks within ambiguity_ratio "
                    f"({ambiguity:.2f}) -- not silently merged"
                ),
            ))
            open_tracks.append({
                "label": det.label, "dets": [det],
                "last_box": det.box, "last_center": det.center,
                "track_id": f"boxtrack-{det.label}-{len(open_tracks):04d}",
            })
            by_label.setdefault(det.label, []).append(len(open_tracks) - 1)
            continue

        tr = open_tracks[best_idx]
        tr["dets"].append(det)
        tr["last_box"] = det.box
        tr["last_center"] = det.center
        decisions.append(AssociationResult(
            decision="matched", track_id=tr["track_id"],
            detection_id=det.detection_id,
            iou=best_iou, distance_px=best_dist, ambiguity=ambiguity,
            reason="IoU + motion gates passed",
        ))

    # Untimed detections: real observations that cannot enter timed
    # association (silently dropping them would lose evidence; mixing
    # them in would corrupt the time order). Each becomes its own
    # single-detection track with an explicit decision.
    for det in sorted(
        (d for d in detections if d.captured_at is None),
        key=lambda d: d.detection_id,
    ):
        open_tracks.append({
            "label": det.label, "dets": [det],
            "last_box": det.box, "last_center": det.center,
            "track_id": f"boxtrack-{det.label}-{len(open_tracks):04d}",
        })
        decisions.append(AssociationResult(
            decision="new_track", detection_id=det.detection_id,
            reason="no capture timestamp -- excluded from timed association",
        ))

    tracks: List[BoxTrack] = []
    for tr in open_tracks:
        dets = tr["dets"]
        # ID-switch diagnostic (directive section 14): a member that
        # JOINED a track while a runner-up was within the provisional
        # band (ambiguity >= 0.5, below the refusal ratio) is flagged
        # on the track -- measured risk, reported not hidden.
        provisional = tuple(
            dec.detection_id for dec in decisions
            if dec.decision == "matched"
            and dec.track_id == tr["track_id"]
            and dec.ambiguity >= provisional_ratio
        )
        tracks.append(BoxTrack(
            track_id=tr["track_id"],
            label=tr["label"],
            detections=tuple(dets),
            ambiguous_links=provisional,
        ))
    def _sort_key(t: BoxTrack):
        first = t.detections[0]
        # Untimed tracks sort after timed ones (captured_at None is
        # not comparable to floats).
        timed_key = (0, first.captured_at) if first.captured_at is not None else (1, 0.0)
        return (t.label, timed_key, first.detection_id)

    tracks.sort(key=_sort_key)
    return tracks, decisions


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
