"""Reconstruction readiness verdict: READY | PARTIAL | BLOCKED | FAILED, derived ONLY from what a run persisted.

No universal numeric thresholds: each gate reports the quantity it measured and a status from a structural rule on that quantity
("every entity positioned"), never "the pipeline finished". A gate whose quantity could not be measured is ``not_measured`` and can
never count towards READY. The verdict also says what is missing, so partial worlds stay inspectable without being called complete.

* FAILED  -- the run or its persistence is broken (unreadable snapshot, stage failed, persisted world fails integrity verification).
* BLOCKED -- the run completed but produced nothing usable downstream (no persisted world, empty world, or nothing positioned).
* PARTIAL -- a verified world exists but is incomplete or unanchored (unpositioned / unsized / withheld entities, a partial stage,
             missing detection pieces, scale not physically anchored, an unmeasured gate).
* READY   -- every measured gate passes and none is unmeasured.
"""
from __future__ import annotations

from typing import Any, Mapping

PASS, FAIL, NOT_MEASURED = "pass", "fail", "not_measured"


def _gate(name: str, measured: Any, status: str, basis: str) -> dict[str, Any]:
    return {"name": name, "measured": measured, "status": status, "basis": basis}


def assess(readiness: Mapping[str, Any]) -> dict[str, Any]:
    """``readiness`` is the dict ``apps.api.evidence_world_stage.readiness`` builds from disk. Pure: reads it, returns the verdict."""
    stage = readiness.get("stage_status")
    world = readiness.get("world") or {}
    reasons: list[str] = []
    if stage in (None, "unreadable", "failed", "cancelled"):
        return {"verdict": "FAILED", "reasons": [readiness.get("reason") or f"reconstruction stage status: {stage}"], "gates": []}
    if world.get("status") in ("failed", "failed_verification"):
        why = world.get("verification") or world.get("reason") or ""
        return {"verdict": "FAILED", "reasons": [f"persisted world is not trustworthy: {world.get('status')} {why}".strip()],
                "gates": [_gate("world_integrity", world.get("verification"), FAIL, "WorldStore.verify_version reported problems")]}

    integ = readiness.get("integrity")
    if integ is not None and not integ.get("ok"):
        return {"verdict": "FAILED", "gates": [_gate("evidence_integrity", integ.get("problems"), FAIL, "recorded source photographs, manifest and world still match the disk")],
                "reasons": ["source evidence or the run record no longer matches what the run recorded: " + "; ".join(integ.get("problems") or ["unspecified"])]}

    gates: list[dict[str, Any]] = []
    gates.append(_gate("evidence_integrity", None if integ is None else "verified", NOT_MEASURED if integ is None else PASS,
                       "recorded source photographs, manifest and world still match the disk"))
    persisted = world.get("status") == "persisted"
    gates.append(_gate("world_persisted", world.get("status"), PASS if persisted else FAIL, "a verified immutable version exists on disk"))
    n = world.get("entities")
    placed, sized = world.get("entities_with_position"), world.get("entities_with_geometry")
    if n:
        gates.append(_gate("spatial_coverage", f"{placed}/{n} entities positioned", PASS if placed == n else FAIL, "every entity has a measured position"))
        gates.append(_gate("geometry_coverage", f"{sized}/{n} entities sized", PASS if sized == n else FAIL, "every entity has geometry"))
    else:
        gates.append(_gate("spatial_coverage", None, NOT_MEASURED, "no entities"))
    withheld = world.get("withheld_entities") or []
    gates.append(_gate("withheld_entities", len(withheld), PASS if not withheld else FAIL, "no entity was withheld for lacking evidence"))
    gates.append(_gate("stage_complete", stage, PASS if stage == "complete" else FAIL, "the evidence flow itself reported complete"))
    det = readiness.get("detection") or {}
    if det.get("status") == "ran":
        gates.append(_gate("detection_segmentation", det.get("segmentation"), PASS if det.get("segmentation") == "ran" else FAIL,
                           "segmentation executed for the detections"))
    else:
        gates.append(_gate("detection_segmentation", det.get("status"), NOT_MEASURED, "the tiled detection pipeline did not run"))
    if det.get("status") == "ran" and det.get("images"):
        got = det.get("images_with_detections")
        gates.append(_gate("detection_coverage", None if got is None else f"{got}/{det['images']} images with a detection",
                           NOT_MEASURED if got is None else PASS if got == det["images"] else FAIL, "every photograph produced at least one detection"))
    else:
        gates.append(_gate("detection_coverage", None, NOT_MEASURED, "the tiled detection pipeline did not run"))
    comp = readiness.get("completeness")  # only an INDEPENDENT inventory of what the room contains can supply this
    gates.append(_gate("object_completeness", None if not comp else f"{comp.get('covered')}/{comp.get('of')}",
                       NOT_MEASURED if not comp or not comp.get("of") else PASS if comp["covered"] == comp["of"] else FAIL,
                       "every object in an independent inventory of the room was reconstructed (no such inventory exists unless supplied)"))
    acc = readiness.get("physical_accuracy")
    gates.append(_gate("physical_accuracy", acc, NOT_MEASURED if acc in (None, "NOT_MEASURED") else PASS,
                       "an independent physical survey measured the reconstruction's error"))
    scale = readiness.get("scale") or {}
    if scale.get("records_by_scale_state"):
        gates.append(_gate("physical_scale", scale.get("status"), PASS if scale.get("physical_metric") else FAIL,
                           "positions are physically anchored metres (otherwise gauge/relative units)"))
    else:
        gates.append(_gate("physical_scale", scale.get("status"), NOT_MEASURED, "no depth/scale evidence"))

    cov = readiness.get("coverage")
    if cov:
        t = cov.get("topology") or {}
        bad = int(t.get("errors", 0)) + int(t.get("world_validation_errors", 0))
        gates.append(_gate("topology_validity", {"errors": bad, "warnings": t.get("warnings", 0)}, PASS if bad == 0 else FAIL,
                           "the persisted world validates with no topology or structural errors (warnings are reported, not failed)"))
        up = cov.get("uncertainty_propagated")
        if up is None or not up.get("of"):
            gates.append(_gate("uncertainty_propagated", None, NOT_MEASURED, "no geometry carries an evidence-derived uncertainty block"))
        else:
            gates.append(_gate("uncertainty_propagated", f"{up.get('covered')}/{up.get('of')}", PASS if up["covered"] == up["of"] else FAIL,
                               "every geometry's position and dimension uncertainty was derived from evidence (none is a stand-in)"))
        uc = cov.get("unresolved_conflicts")
        if uc is not None:
            gates.append(_gate("unresolved_conflicts", f"{uc.get('entities', 0)} entities / {uc.get('hypotheses', 0)} competing hypotheses",
                               PASS if not uc.get("entities") else FAIL, "no entity carries competing hypotheses the fusion could not resolve"))
        for key, why in (("source_traceability", "every entity cites at least one source photograph"),
                         ("provenance_declared", "every entity declares its provenance")):
            c = cov.get(key) or {}
            gates.append(_gate(key, f"{c.get('covered')}/{c.get('of')}", PASS if c.get("of") and c.get("covered") == c.get("of") else FAIL, why))
    else:
        gates.append(_gate("topology_validity", None, NOT_MEASURED, "no persisted world to measure"))

    if not persisted:
        return {"verdict": "BLOCKED", "reasons": [f"no persisted world ({world.get('status')}): {world.get('reason', 'nothing to inspect or export')}"], "gates": gates}
    if not n:
        return {"verdict": "BLOCKED", "reasons": ["the persisted world is empty: nothing was reconstructed"], "gates": gates}
    if not placed:
        return {"verdict": "BLOCKED", "reasons": ["no entity has a measured position; spatial reconstruction produced nothing usable"], "gates": gates}

    for g in gates:
        if g["status"] == FAIL:
            reasons.append(f"{g['name']}: {g['measured']} (needs: {g['basis']})")
        elif g["status"] == NOT_MEASURED:
            reasons.append(f"{g['name']} not measured ({g['basis']})")
    return {"verdict": "PARTIAL" if reasons else "READY", "reasons": reasons, "gates": gates,
            "coverage": cov or {}}  # multiview_association etc. are reported, not gated: a single-view object is weaker, not invalid
