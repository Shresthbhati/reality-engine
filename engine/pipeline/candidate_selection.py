"""Which reconstruction candidate is the better EVOLUTION of the existing world?

COLMAP can hand the pipeline two candidates for the same evidence: the incremental one (new photos
registered into the established model, coordinate frame preserved) and a full re-solve. Choosing by
"who registered more cameras" can pick a candidate that places one more photo but destroys established
geometry. Here both candidates are turned into world snapshots and put through the SAME machinery that
gates adoption (world_delta.compute_delta / reconcile_conflicts / decide, which uses spatial_continuity).

There is no scalar score. The decision is an ordered hierarchy; the first rule on which the candidates
differ decides, and the deciding rule is recorded:

    1. validity            a candidate that could not be built loses
    2. acceptance          REJECT loses to non-REJECT (both REJECT: neither is good, the job gate will
                           keep HEAD; the one with fewer reasons is returned so the record is honest)
    3. established loss    fewer previously-registered photos lost, then fewer established structural
                           surfaces removed/only-partly-reproduced, then fewer unsupported moves/conflicts
    4. camera stability    smaller drift of shared cameras (only when the difference exceeds a margin)
    5. uncertainty         fewer recorded uncertainties
    6. frame preservation  with an established HEAD, a candidate that KEEPS HEAD's coordinate frame beats one that
                           re-expresses the world in a new frame, unless the frame-changing candidate places at
                           least FRAME_CHANGE_MIN_EXTRA_PHOTOS more photographs (see _frame_preservation)
    7. useful evidence     more photographs placed
    8. new understanding   more surfaces refined / extended / new -- FIRST VERSIONS ONLY: with an established HEAD
                           the coordinate frame outranks this rule (see _new_understanding)
    9. tie                 the incremental candidate (keeps the established coordinate frame and avoids a
                           needless replacement)

Margins are tolerances, not weights: a difference smaller than CAMERA_DRIFT_MARGIN is treated as noise.

The coordinate frame matters because world_delta aligns candidate cameras to HEAD with a best-fit similarity
before comparing them (COLMAP's frame is arbitrary), so geometry-continuity rules are blind to a frame change by
design. Yet every earlier version, scale reference, entity identity and user-facing coordinate was expressed in
HEAD's frame: replacing it is a real cost, and is only worth paying for a material gain in evidence placed.
``frame_shift`` measures it from that same alignment; the tolerances below are PROVISIONAL (set from COLMAP's
incremental bundle-adjustment jitter, not yet calibrated on more than one dataset).
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

from engine.pipeline import world_delta as wd

#: drift differences below this (fraction of scene extent) are indistinguishable
CAMERA_DRIFT_MARGIN = 0.02
#: a candidate whose frame differs from HEAD's by more than ANY of these has "changed the coordinate frame".
#: Incremental registration + bundle adjustment re-solves every pose slightly, so identity is a tolerance, not 0.
FRAME_ROTATION_TOL_DEG = 1.0
FRAME_SCALE_TOL = 0.01              # |ln(scale)|
FRAME_TRANSLATION_TOL = 0.01        # origin displacement as a fraction of the scene extent
#: placing at least this many MORE photographs than a frame-preserving candidate justifies a frame change
FRAME_CHANGE_MIN_EXTRA_PHOTOS = 2


def frame_shift(alignment: Optional[dict], cand_cameras: Dict[str, list]) -> Optional[dict]:
    """How far the candidate's coordinate frame is from HEAD's, from the similarity ``world_delta`` already fitted
    (cand ~= s R (prev - ma) + mb). None when there was no alignment (nothing to compare against)."""
    if not alignment or not cand_cameras:
        return None
    import math

    import numpy as np

    r = np.asarray(alignment["R"], float)
    s = float(alignment["s"])
    if s <= 1e-12:
        return None
    pts = np.asarray(list(cand_cameras.values()), float)
    extent = float(np.sqrt(((pts - pts.mean(0)) ** 2).sum(1).mean()))
    if extent <= 1e-12:
        return None
    ma, mb = np.asarray(alignment["ma"], float), np.asarray(alignment["mb"], float)
    origin = mb - s * (r @ ma)                         # where HEAD's origin lands in the candidate's frame
    rot = math.degrees(math.acos(max(-1.0, min(1.0, (float(np.trace(r)) - 1.0) / 2.0))))
    out = {"rotation_deg": round(rot, 4), "scale_change": round(abs(math.log(s)), 5),
           "translation_rel": round(float(np.linalg.norm(origin)) / extent, 5)}
    out["changed"] = bool(out["rotation_deg"] > FRAME_ROTATION_TOL_DEG or out["scale_change"] > FRAME_SCALE_TOL
                          or out["translation_rel"] > FRAME_TRANSLATION_TOL)
    return out


def assess(prev_snapshot: Optional[dict], cand_snapshot: dict, prev_conflicts: Optional[List[dict]] = None,
           tag: str = "candidate") -> dict:
    """The acceptance machinery applied to one candidate, plus the measured facts the hierarchy compares."""
    delta = wd.compute_delta(prev_snapshot, cand_snapshot)
    conflicts = wd.reconcile_conflicts(prev_conflicts, delta, tag)
    decision = wd.decide(delta, conflicts)
    ev = delta["evidence"]
    ents = delta.get("entities") or {}
    rels = ents.get("relations", [])
    struct_lost = sum(len(r["prev"]) for r in rels
                      if r["type"] in wd.STRUCTURAL and r["kind"] in ("removed", "reduced"))
    cc = delta.get("camera_consistency")
    fresh = [c for c in conflicts if c.get("status") == "unresolved"
             and (c.get("history") or [{}])[-1].get("event") in ("opened", "moved_again")]
    counts = ents.get("counts") or {}
    fs = frame_shift(delta.get("alignment"), cand_snapshot.get("cameras") or {})
    measures = {
        "cameras_lost": len(ev.get("lost", [])),
        "structural_lost": struct_lost,
        "open_conflicts": len(fresh),
        "camera_drift": cc.get("relative") if cc else None,
        "uncertainties": len(decision["uncertainties"]),
        "registered": len(cand_snapshot.get("registered_ids", [])),
        "new_understanding": counts.get("refined", 0) + counts.get("extended", 0) + counts.get("new", 0),
        "first_version": bool(delta.get("first_version")),
        "frame_shift": fs,
        "frame_changed": None if fs is None else fs["changed"],
    }
    return {"verdict": decision["verdict"], "reasons": decision["reasons"],
            "uncertainties": decision["uncertainties"], "measures": measures}


def _lower(key: str, margin: float = 0.0):
    def cmp(a: dict, b: dict):
        x, y = a["measures"][key], b["measures"][key]
        if x is None or y is None or abs(x - y) <= margin:
            return None
        return "a" if x < y else "b"
    return cmp


def _higher(key: str):
    def cmp(a: dict, b: dict):
        x, y = a["measures"][key], b["measures"][key]
        return None if x == y else ("a" if x > y else "b")
    return cmp


def _new_understanding(a: dict, b: dict):
    """More refined / extended / new surfaces wins -- but only when there is NO established coordinate frame to
    lose. Once the world has a HEAD, preserving its frame outranks this tie-break: a full re-solve replaces the
    frame every earlier version, scale reference and entity identity was expressed in, so a marginal gain in
    understanding (by one entity) is not worth it. The incremental candidate then wins the final tie."""
    if not a["measures"].get("first_version"):
        return None
    return _higher("new_understanding")(a, b)


def _frame_preservation(a: dict, b: dict):
    """With an established frame, the candidate that keeps it wins unless the other places materially more
    photographs. Undecided (None) when both keep it, both change it, or either frame is not measurable."""
    ca, cb = a["measures"].get("frame_changed"), b["measures"].get("frame_changed")
    if ca is None or cb is None or ca == cb:
        return None
    keeper, changer = ("a", b) if cb else ("b", a)
    kept = a if keeper == "a" else b
    if changer["measures"]["registered"] - kept["measures"]["registered"] >= FRAME_CHANGE_MIN_EXTRA_PHOTOS:
        return None
    return keeper


# (rule name, measure key, comparator)
_HIERARCHY = (
    ("established photos lost", "cameras_lost", _lower("cameras_lost")),
    ("established surfaces lost", "structural_lost", _lower("structural_lost")),
    ("unsupported moves / conflicts", "open_conflicts", _lower("open_conflicts")),
    ("camera stability", "camera_drift", _lower("camera_drift", CAMERA_DRIFT_MARGIN)),
    ("recorded uncertainty", "uncertainties", _lower("uncertainties")),
    ("coordinate frame preserved", "frame_changed", _frame_preservation),
    ("useful evidence placed", "registered", _higher("registered")),
    ("new understanding (refined / extended / new)", "new_understanding", _new_understanding),
)


def choose(incremental: Optional[dict], full: Optional[dict]) -> dict:
    """Pick between two ``assess`` results (either may be None = could not be built)."""
    if incremental is None and full is None:
        raise ValueError("no candidate to choose from")
    if full is None:
        return {"choice": "incremental", "deciding": "validity", "why": "the full rebuild could not be built"}
    if incremental is None:
        return {"choice": "full", "deciding": "validity", "why": "the incremental model could not be built"}
    ri, rf = incremental["verdict"] == wd.REJECT, full["verdict"] == wd.REJECT
    if ri and rf:
        return {"choice": "full" if len(full["reasons"]) < len(incremental["reasons"]) else "incremental",
                "deciding": "acceptance", "both_questionable": True,
                "why": "both candidates fail world-level acceptance; the adoption gate keeps HEAD"}
    if ri != rf:
        loser, side = (incremental, "incremental") if ri else (full, "full")
        return {"choice": "full" if ri else "incremental", "deciding": "acceptance",
                "why": f"the {side} candidate fails world-level acceptance: " + "; ".join(loser["reasons"])}
    for name, key, cmp in _HIERARCHY:
        side = cmp(incremental, full)
        if side:
            winner = incremental if side == "a" else full
            why = f"decided by {name}: incremental {incremental['measures'][key]}, full {full['measures'][key]}"
            if winner["measures"].get("frame_changed"):
                why += "; the established coordinate frame is replaced by this choice"
            return {"choice": "incremental" if side == "a" else "full", "deciding": name, "why": why}
    return {"choice": "incremental", "deciding": "tie",
            "why": "no measurable difference; the incremental candidate keeps the established coordinate frame"}


class WorldArbiter:
    """What the COLMAP backend consults. ``to_snapshot`` compiles a ReconstructionResult into the same
    snapshot the adoption gate stores, so both candidates are judged as worlds against the world's HEAD."""

    def __init__(self, prev_snapshot: Optional[dict], prev_conflicts: Optional[List[dict]],
                 to_snapshot: Callable[[object], dict]):
        self.prev_snapshot = prev_snapshot
        self.prev_conflicts = prev_conflicts
        self.to_snapshot = to_snapshot

    def assess(self, result, tag: str) -> dict:
        return assess(self.prev_snapshot, self.to_snapshot(result), self.prev_conflicts, tag)

    @staticmethod
    def choose(incremental: Optional[dict], full: Optional[dict]) -> dict:
        return choose(incremental, full)
