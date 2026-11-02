"""Structured reconstruction diagnostics (P5): what is measurably wrong or weak in a multi-view result.

Pure analysis of a ``ReconstructionResult``'s points and camera poses. It changes nothing and corrects nothing; a later
correction engine consumes these records and decides what to try. Every record carries the measurement that triggered
it, the threshold it was compared with, a severity, and whether an automatic correction would be safe (``True`` only
when the correction is non-destructive, such as excluding a camera from dense fusion; anything that needs new evidence
or could change canonical geometry is ``False``).

Thresholds are EXPERIMENTAL and named here, not buried. ``WEAK_OBSERVATIONS`` mirrors COLMAP's own absolute-pose
minimum (30 inliers): a camera seen by fewer triangulated points than that was placed on thin evidence. Measured on the
South Building set, relaxing registration to admit more photographs added cameras with 10 to 15 observations against a
median of 663 for the rest; this module is what makes that kind of weak camera visible.

Complements, does not replace: ``reconstruction/evaluation.py`` (pose and point accuracy against a reference model),
``reconstruction/validation.py`` (two reconstructions compared) and ``reconstruction/robustness.py`` (evidence
admission and run classification). None of those measure how well each placed camera is supported.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from statistics import mean, median
from typing import Dict, Iterable, List, Optional, Sequence

WEAK_OBSERVATIONS = 30          # below: warning
VERY_WEAK_OBSERVATIONS = 15     # below: error
LOW_REGISTRATION_FRACTION = 0.5  # fewer than this share of photographs placed: reported as a model-level finding
#: Two cameras are "tied" when they share at least this many triangulated points. A group of cameras tied to the rest
#: of the model by fewer than this is internally consistent but its placement relative to the model is unverified.
#: EXPERIMENTAL. Measured on South Building (32-photo stride-4 subset, session model with 25 registered cameras): three
#: cameras sharing 55-72 points with each other but at most 11 with any other camera were placed 104-129 deg wrong
#: against the reference, while looking healthy by every per-camera measure (76-78 observations, 0.25 px reprojection).
#: Thresholds 20 and 30 isolate exactly those three; the committed 32-camera reference stays one group up to 50.
BRIDGE_MIN_SHARED_POINTS = 30

SEVERITIES = ("info", "warning", "error")


@dataclass(frozen=True)
class Diagnostic:
    type: str
    severity: str
    subject: str                       # evidence id, or "reconstruction" for model-level findings
    message: str
    measured: dict = field(default_factory=dict)
    threshold: Optional[dict] = None
    safe_to_auto_correct: bool = False
    recommended: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


def _finite(values: Iterable[float]) -> bool:
    return all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)


_NON_SPARSE_PREFIXES = ("dense_mvs:", "dense:", "depth-")


def sparse_points(points: Sequence) -> List:
    """Triangulated structure only. Dense-MVS and monocular-depth points cite every source camera, so counting them
    would make each camera look supported by hundreds of thousands of points it never triangulated."""
    return [p for p in points if not str(getattr(p, "track_id", "") or "").startswith(_NON_SPARSE_PREFIXES)]


def bridged_groups(points: Sequence, registered: Iterable[str], min_shared: int = BRIDGE_MIN_SHARED_POINTS) -> List[dict]:
    """Camera groups tied to the largest group by fewer than ``min_shared`` shared points per camera pair.

    Union-find over the covisibility graph (edge = pair of cameras sharing >= ``min_shared`` sparse points). Every
    group of two or more cameras except the largest is returned with the evidence for it: how strongly it is tied
    inside and to the rest. Single cameras are never reported as a group."""
    registered = sorted(set(registered))
    known = set(registered)
    if len(registered) < 2:
        return []
    shared: Dict[tuple, int] = {}
    for pt in points:
        ids = sorted({e for e in (getattr(pt, "source_evidence_ids", None) or []) if e in known})
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                shared[(ids[i], ids[j])] = shared.get((ids[i], ids[j]), 0) + 1
    parent = {e: e for e in registered}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for (a, b), n in shared.items():
        if n >= min_shared:
            parent[find(a)] = find(b)
    groups: Dict[str, List[str]] = {}
    for e in registered:
        groups.setdefault(find(e), []).append(e)
    ordered = sorted(groups.values(), key=lambda g: (-len(g), g))
    out = []
    for g in ordered[1:]:
        if len(g) < 2:
            # A lone weakly connected camera is not "a group consistent with each other" (and is already covered by
            # WEAKLY_SUPPORTED_CAMERA); without this, any small capture whose every camera pair shares fewer than
            # min_shared points would report each camera as its own group.
            continue
        gs = set(g)
        inside = [n for (a, b), n in shared.items() if a in gs and b in gs]
        outside = [n for (a, b), n in shared.items() if (a in gs) != (b in gs)]
        out.append({"cameras": g, "internal_max_shared": max(inside) if inside else 0,
                    "to_rest_max_shared": max(outside) if outside else 0})
    return out


def camera_support(points: Sequence, poses: Sequence) -> Dict[str, dict]:
    """Per registered photograph: how many triangulated points observed it and their mean confidence."""
    points = sparse_points(points)
    support: Dict[str, dict] = {p.evidence_id: {"observations": 0, "_conf": []} for p in poses}
    for pt in points:
        conf = getattr(getattr(pt, "uncertainty", None), "confidence", None)
        for eid in getattr(pt, "source_evidence_ids", None) or []:
            if eid in support:
                support[eid]["observations"] += 1
                if isinstance(conf, (int, float)) and math.isfinite(conf):
                    support[eid]["_conf"].append(conf)
    return {
        eid: {"observations": s["observations"],
              "mean_point_confidence": round(mean(s["_conf"]), 3) if s["_conf"] else None}
        for eid, s in support.items()
    }


def analyze(points: Sequence, poses: Sequence, input_ids: Sequence[str]) -> dict:
    """Summary, per-camera support and diagnostics for one multi-view result. JSON-serialisable."""
    registered = {p.evidence_id for p in poses}
    all_points, points = points, sparse_points(points)
    support = camera_support(points, poses)
    diags: List[Diagnostic] = []

    # model level -----------------------------------------------------------------------------------------------
    n_input = len(input_ids)
    placed = len(registered & set(input_ids))
    fraction = (placed / n_input) if n_input else 0.0
    if n_input and fraction < LOW_REGISTRATION_FRACTION:
        diags.append(Diagnostic(
            "LOW_REGISTRATION", "warning", "reconstruction",
            f"Only {placed} of {n_input} photographs could be placed.",
            measured={"registered": placed, "input": n_input, "fraction": round(fraction, 3)},
            threshold={"fraction": LOW_REGISTRATION_FRACTION},
            recommended="Add photographs that overlap each other from different positions."))

    # non-finite numbers and inconsistent references (a backend fault, never auto-corrected) -------------------
    bad_points = sum(1 for pt in points if not _finite(getattr(pt, "position", ()) or ()))
    bad_poses = sum(1 for p in poses if not (_finite(p.position) and _finite(p.rotation)))
    if bad_points or bad_poses:
        diags.append(Diagnostic(
            "NON_FINITE_GEOMETRY", "error", "reconstruction",
            "The reconstruction contains non-finite (NaN or infinite) numbers.",
            measured={"points": bad_points, "poses": bad_poses},
            recommended="Treat this result as invalid and re-run the backend."))
    dangling = sum(1 for pt in points for eid in (getattr(pt, "source_evidence_ids", None) or []) if eid not in registered)
    if dangling:
        diags.append(Diagnostic(
            "DANGLING_OBSERVATION", "error", "reconstruction",
            "Some points cite photographs that have no camera pose.",
            measured={"observations": dangling},
            recommended="Treat this result as inconsistent and re-run the backend."))
    unknown = sorted(registered - set(input_ids)) if n_input else []
    if unknown:
        diags.append(Diagnostic(
            "UNKNOWN_CAMERA", "error", "reconstruction",
            "Some camera poses belong to photographs that were not supplied.",
            measured={"cameras": len(unknown)}, recommended="Re-run the backend; do not use this result."))

    # per photograph --------------------------------------------------------------------------------------------
    for eid in input_ids:
        if eid not in registered:
            diags.append(Diagnostic(
                "UNREGISTERED_IMAGE", "warning", eid, "This photograph could not be placed in the model.",
                measured={"observations": 0},
                recommended="Take or add photographs that overlap this one and its neighbours."))
    for eid, s in support.items():
        obs = s["observations"]
        if obs < WEAK_OBSERVATIONS:
            diags.append(Diagnostic(
                "WEAKLY_SUPPORTED_CAMERA", "error" if obs < VERY_WEAK_OBSERVATIONS else "warning", eid,
                "This photograph was placed on thin evidence, so its position is unverified.",
                measured={"observations": obs},
                threshold={"warning_below": WEAK_OBSERVATIONS, "error_below": VERY_WEAK_OBSERVATIONS},
                safe_to_auto_correct=True,  # excluding it from dense fusion and measurement removes nothing
                recommended="exclude_camera_from_dense_and_measurement"))

    for g in bridged_groups(points, registered):
        diags.append(Diagnostic(
            "WEAKLY_BRIDGED_GROUP", "error", "group:" + g["cameras"][0],
            f"{len(g['cameras'])} camera(s) are consistent with each other but only thinly tied to the rest of the "
            "model, so where they sit in it is unverified.",
            measured=g, threshold={"min_shared_points_per_camera_pair": BRIDGE_MIN_SHARED_POINTS},
            safe_to_auto_correct=True,  # excluding the group from dense fusion and measurement removes nothing verified
            recommended="exclude_group_from_dense_and_measurement"))

    counts = [s["observations"] for s in support.values()]
    track_lengths = [len(getattr(pt, "source_evidence_ids", None) or []) for pt in points]
    summary = {
        "registered": len(registered), "input": n_input,
        "registered_fraction": round(fraction, 3) if n_input else None,
        "points": len(points), "dense_points": len(all_points) - len(points),
        "median_observations_per_camera": median(counts) if counts else None,
        "min_observations_per_camera": min(counts) if counts else None,
        "mean_track_length": round(mean(track_lengths), 2) if track_lengths else None,
        "by_severity": {sev: sum(1 for d in diags if d.severity == sev) for sev in SEVERITIES},
    }
    return {"summary": summary, "cameras": support, "diagnostics": [d.to_dict() for d in diags]}
