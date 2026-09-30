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
    6. useful evidence     more photographs placed
    7. new understanding   more surfaces refined / extended / new -- FIRST VERSIONS ONLY: with an established HEAD
                           the coordinate frame outranks this rule (see _new_understanding)
    8. tie                 the incremental candidate (keeps the established coordinate frame and avoids a
                           needless replacement)

Margins are tolerances, not weights: a difference smaller than CAMERA_DRIFT_MARGIN is treated as noise.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

from engine.pipeline import world_delta as wd

#: drift differences below this (fraction of scene extent) are indistinguishable
CAMERA_DRIFT_MARGIN = 0.02


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
    measures = {
        "cameras_lost": len(ev.get("lost", [])),
        "structural_lost": struct_lost,
        "open_conflicts": len(fresh),
        "camera_drift": cc.get("relative") if cc else None,
        "uncertainties": len(decision["uncertainties"]),
        "registered": len(cand_snapshot.get("registered_ids", [])),
        "new_understanding": counts.get("refined", 0) + counts.get("extended", 0) + counts.get("new", 0),
        "first_version": bool(delta.get("first_version")),
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


# (rule name, measure key, comparator)
_HIERARCHY = (
    ("established photos lost", "cameras_lost", _lower("cameras_lost")),
    ("established surfaces lost", "structural_lost", _lower("structural_lost")),
    ("unsupported moves / conflicts", "open_conflicts", _lower("open_conflicts")),
    ("camera stability", "camera_drift", _lower("camera_drift", CAMERA_DRIFT_MARGIN)),
    ("recorded uncertainty", "uncertainties", _lower("uncertainties")),
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
            return {"choice": "incremental" if side == "a" else "full", "deciding": name,
                    "why": (f"decided by {name}: incremental {incremental['measures'][key]}, "
                            f"full {full['measures'][key]}")}
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
