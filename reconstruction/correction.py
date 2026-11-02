"""Bounded, evidence-driven correction of a multi-view result (P5).

Consumes the structured findings of ``reconstruction.diagnostics`` and tries only corrections that (a) the diagnostics
marked ``safe_to_auto_correct`` and (b) are registered here as executable and checkable. It is not an agent: every
candidate is produced by a named, deterministic function, scored with the SAME metrics as the baseline, and accepted
only when predefined criteria pass. The baseline is never mutated or overwritten; the caller decides what to do with
the returned selection.

Supported actions (anything else is recorded as skipped, never improvised):

  EXCLUDE_BRIDGED_GROUP  drop camera groups that are consistent with each other but only thinly tied to the rest of
                         the model (diagnostics.BRIDGE_MIN_SHARED_POINTS): their placement is unverified. Measured on
                         real data to remove exactly the misplaced cameras (see tests/test_reconstruction_correction.py).
  EXCLUDE_WEAK_CAMERAS   drop cameras placed on thin evidence (diagnostics.WEAK_OBSERVATIONS) and their observations.
                         Executable offline in milliseconds. Expected effect: removes unverified poses; it cannot
                         add geometry and it cannot make a model better than its remaining cameras support.

Not executable here, reported as an evidence request instead: unregistered photographs. A missing photo needs new
photographs, not a quiet fix; speculative geometry generation is out of scope.

Compute is bounded twice: ``max_attempts`` candidates and ``max_seconds`` of wall-clock across all of them (each
action is checked before it starts). A candidate that raises, returns something invalid, or regresses a metric is
REJECTED/FAILED with the reason recorded; the selected result then stays the baseline.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from reconstruction.backend.interface import ReconstructionResult
from reconstruction.diagnostics import WEAK_OBSERVATIONS, analyze, bridged_groups, camera_support, sparse_points

ACCEPTED, REJECTED, FAILED, SKIPPED = "ACCEPTED", "REJECTED", "FAILED", "SKIPPED"


@dataclass(frozen=True)
class Budget:
    max_attempts: int = 2
    max_seconds: float = 60.0


@dataclass(frozen=True)
class AcceptanceCriteria:
    """Predefined before any candidate runs. EXPERIMENTAL thresholds, named so they can be changed with evidence."""
    min_point_retention: float = 0.9      # candidate keeps at least this share of the baseline's points
    max_pose_regression: float = 1.05     # GT median errors may not exceed baseline x this ...
    pose_noise_floor_deg: float = 0.02    # ... plus this absolute slack on rotation (degrees)


# ---------------------------------------------------------------------------------------------------------------------
# actions: result -> (candidate result, ids the action declares it removed)
# ---------------------------------------------------------------------------------------------------------------------

def _without(result: ReconstructionResult, drop: set) -> Tuple[ReconstructionResult, List[str]]:
    if not drop:
        return result, []
    weak = drop
    points = []
    for pt in sparse_points(result.points):  # dense/depth points were fused with the removed poses: dropped
        kept = [e for e in pt.source_evidence_ids if e not in weak]
        if len(kept) >= 2:  # a structure point needs two views to remain a triangulation
            points.append(replace(pt, source_evidence_ids=kept))
    poses = [p for p in result.camera_poses if p.evidence_id not in weak]
    # dense output was fused with the removed cameras' poses, so it is no longer valid for this model
    return replace(result, points=points, camera_poses=poses, dense_points=None, dense_report=None), sorted(weak)


def exclude_weak_cameras(result: ReconstructionResult) -> Tuple[ReconstructionResult, List[str]]:
    support = camera_support(result.points, result.camera_poses)
    return _without(result, {eid for eid, s in support.items() if s["observations"] < WEAK_OBSERVATIONS})


def exclude_bridged_group(result: ReconstructionResult) -> Tuple[ReconstructionResult, List[str]]:
    groups = bridged_groups(sparse_points(result.points), [p.evidence_id for p in result.camera_poses])
    return _without(result, {e for g in groups for e in g["cameras"]})


#: diagnostic type -> (action name, rationale, expected effect). The only executable mapping.
EXECUTABLE: Dict[str, Tuple[str, str, str]] = {
    "WEAKLY_BRIDGED_GROUP": (
        "EXCLUDE_BRIDGED_GROUP",
        "These cameras agree with each other but share too few points with the rest of the model to verify where "
        "they sit in it.",
        "Removes the unverified group; cameras it never touched keep their poses. Dense output is invalidated and must "
        "be recomputed from the remaining cameras.",
    ),
    "WEAKLY_SUPPORTED_CAMERA": (
        "EXCLUDE_WEAK_CAMERAS",
        "Cameras seen by too few triangulated points were placed on thin evidence; their poses are unverified.",
        "Fewer registered cameras, no unverified poses; remaining geometry unchanged except points that only they saw.",
    ),
}
ACTIONS: Dict[str, Callable[[ReconstructionResult], Tuple[ReconstructionResult, List[str]]]] = {
    "EXCLUDE_BRIDGED_GROUP": exclude_bridged_group,
    "EXCLUDE_WEAK_CAMERAS": exclude_weak_cameras,
}


# ---------------------------------------------------------------------------------------------------------------------
# metrics (identical for baseline and every candidate)
# ---------------------------------------------------------------------------------------------------------------------

def _finite_result(result: ReconstructionResult) -> bool:
    for pt in result.points:
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in pt.position):
            return False
    for p in result.camera_poses:
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (*p.position, *p.rotation)):
            return False
    return True


def measure(result: ReconstructionResult, input_ids: Sequence[str], gt_images: Optional[dict] = None) -> dict:
    report = analyze(result.points, result.camera_poses, input_ids)
    by_type: Dict[str, int] = {}
    for d in report["diagnostics"]:
        by_type[d["type"]] = by_type.get(d["type"], 0) + 1
    out = {
        "summary": report["summary"],
        "diagnostic_counts": by_type,
        "finite": _finite_result(result),
        "pose_evaluation": None,
    }
    if gt_images:
        from reconstruction.evaluation import evaluate_absolute_poses
        out["pose_evaluation"] = evaluate_absolute_poses(list(result.camera_poses), gt_images)
    return out


def judge(base: dict, cand: dict, removed: Sequence[str], criteria: AcceptanceCriteria) -> List[str]:
    """Reasons to reject (empty = every criterion passed)."""
    why: List[str] = []
    if not cand["finite"]:
        why.append("candidate contains non-finite geometry")
    bc, cc = base["diagnostic_counts"], cand["diagnostic_counts"]
    for kind in ("NON_FINITE_GEOMETRY", "UNKNOWN_CAMERA", "DANGLING_OBSERVATION"):
        if cc.get(kind, 0) > bc.get(kind, 0):
            why.append(f"{kind} increased ({bc.get(kind, 0)} -> {cc.get(kind, 0)})")
    bs, cs = base["summary"], cand["summary"]
    if cs["registered"] < bs["registered"] - len(removed):
        why.append(f"registered cameras fell by more than the {len(removed)} the action declared "
                   f"({bs['registered']} -> {cs['registered']})")
    if bs["points"] and cs["points"] < criteria.min_point_retention * bs["points"]:
        why.append(f"points retained {cs['points']}/{bs['points']} < {criteria.min_point_retention:.0%}")
    if cc.get("WEAKLY_SUPPORTED_CAMERA", 0) > bc.get("WEAKLY_SUPPORTED_CAMERA", 0):
        why.append("weakly supported cameras increased")
    if cc.get("WEAKLY_BRIDGED_GROUP", 0) > bc.get("WEAKLY_BRIDGED_GROUP", 0):
        why.append("weakly bridged groups increased")
    improved = (cc.get("WEAKLY_SUPPORTED_CAMERA", 0) < bc.get("WEAKLY_SUPPORTED_CAMERA", 0)
                or cc.get("WEAKLY_BRIDGED_GROUP", 0) < bc.get("WEAKLY_BRIDGED_GROUP", 0)
                or (cs["min_observations_per_camera"] or 0) > (bs["min_observations_per_camera"] or 0))
    if not why and not improved:
        why.append("no measured improvement over the baseline")
    be, ce = base.get("pose_evaluation"), cand.get("pose_evaluation")
    if be and not be.get("refused"):
        if not ce or ce.get("refused"):
            why.append("reference evaluation refused for the candidate but not for the baseline")
        else:
            if ce["rotation_deg_median"] > be["rotation_deg_median"] * criteria.max_pose_regression + criteria.pose_noise_floor_deg:
                why.append(f"median rotation error regressed ({be['rotation_deg_median']} -> {ce['rotation_deg_median']} deg)")
            if ce["center_error_median"] > be["center_error_median"] * criteria.max_pose_regression:
                why.append(f"median camera-centre error regressed ({be['center_error_median']} -> {ce['center_error_median']})")
    return why


# ---------------------------------------------------------------------------------------------------------------------
# orchestrator
# ---------------------------------------------------------------------------------------------------------------------

def correct(
    baseline: ReconstructionResult,
    input_ids: Sequence[str],
    *,
    gt_images: Optional[dict] = None,
    budget: Budget = Budget(),
    criteria: AcceptanceCriteria = AcceptanceCriteria(),
    actions: Optional[Dict[str, Callable]] = None,
    clock: Callable[[], float] = time.monotonic,
) -> Tuple[ReconstructionResult, dict]:
    """Return ``(selected_result, record)``. ``selected_result`` is ``baseline`` itself unless a candidate was accepted."""
    actions = ACTIONS if actions is None else actions
    base_m = measure(baseline, input_ids, gt_images)
    diags = analyze(baseline.points, baseline.camera_poses, input_ids)["diagnostics"]
    record: dict = {"baseline": base_m, "attempts": [], "evidence_requests": [], "selected": "baseline",
                    "budget": {"max_attempts": budget.max_attempts, "max_seconds": budget.max_seconds}}

    missing = [d["subject"] for d in diags if d["type"] == "UNREGISTERED_IMAGE"]
    if missing:
        record["evidence_requests"].append({
            "reason": "These photographs could not be placed; no safe correction can add them.",
            "images": missing,
            "ask": "Add photographs that overlap these images and their registered neighbours from intermediate positions."})

    started, selected, planned = clock(), baseline, []
    for d in diags:
        if d["type"] in planned or d["type"] == "UNREGISTERED_IMAGE":
            continue
        planned.append(d["type"])
        step = {"diagnostic": d["type"], "action": None, "status": SKIPPED, "reasons": []}
        record["attempts"].append(step)
        mapped = EXECUTABLE.get(d["type"])
        if not d.get("safe_to_auto_correct") or not mapped or mapped[0] not in actions:
            step["reasons"].append("no supported, safe correction for this finding")
            continue
        name, rationale, effect = mapped
        step.update(action=name, rationale=rationale, expected_effect=effect)
        if sum(1 for a in record["attempts"] if a["status"] != SKIPPED) >= budget.max_attempts:
            step["reasons"].append("attempt budget exhausted")
            continue
        if clock() - started >= budget.max_seconds:
            step["reasons"].append("compute budget exhausted")
            continue
        t0 = clock()
        try:
            cand, removed = actions[name](baseline)
            cand_m = measure(cand, input_ids, gt_images)
        except Exception as exc:  # noqa: BLE001 -- a failed candidate must never take the build down
            step.update(status=FAILED, reasons=[f"{type(exc).__name__}: {exc}"])
            step["seconds"] = round(clock() - t0, 3)
            continue
        step["seconds"] = round(clock() - t0, 3)
        step.update(removed=list(removed), candidate=cand_m)
        reasons = judge(base_m, cand_m, removed, criteria)
        if reasons:
            step.update(status=REJECTED, reasons=reasons)
        else:
            step.update(status=ACCEPTED, reasons=["every acceptance criterion passed"])
            selected, record["selected"] = cand, name
            break  # one accepted correction per call; re-diagnose before stacking another
    return selected, record
