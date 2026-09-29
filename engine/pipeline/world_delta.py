"""World-level acceptance: may this candidate replace the world's current HEAD?

A rebuild from all accumulated evidence can complete "successfully" and still be
WORSE than what the world already knows (fewer photographs placed, a fall back to
independent single views, a scrambled camera layout). A candidate must therefore
be compared with the previous version before it is adopted.

What can honestly be compared across two rebuilds of the same world:

  * evidence ids            -- stable, so "which photos were registered before / now"
                               is exact
  * camera layout           -- compared after a similarity (rotation+scale+shift)
                               alignment, because COLMAP's frame is arbitrary
  * structure counts        -- wall / floor / ... entity counts

What CANNOT be compared, and is therefore not pretended: individual entities
(ids are re-derived on every rebuild) and absolute positions (the frame differs).
Checks that need data an older version did not record are reported as
"not measurable", never skipped silently and never passed.

Verdicts:
  ACCEPT                    nothing measurable regressed
  ACCEPT_WITH_UNCERTAINTY   adopted, but something moved or shrank; reasons are recorded
  REJECT                    net loss of registered evidence (or a downgrade with no gain):
                            HEAD stays authoritative, the new evidence is kept and retried

There is deliberately no scalar "quality score": every reason cites a measured fact.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Sequence

ACCEPT = "ACCEPT"
ACCEPT_WITH_UNCERTAINTY = "ACCEPT_WITH_UNCERTAINTY"
REJECT = "REJECT"

#: relative RMS residual of previously-registered cameras after alignment
CAMERA_MOVED_REL = 0.25
MIN_COMMON_CAMERAS = 4
#: structural entity types whose disappearance is worth flagging
STRUCTURAL = ("wall", "floor", "ceiling", "storey", "room", "corridor", "door", "window", "opening")


def structure_of(world) -> dict:
    """Frame-independent counts of a WorldIR: entity types and provenance classes."""
    types = Counter(e.type.value for e in world.entities.values())
    prov = Counter(getattr(e.provenance, "value", str(e.provenance)) for e in world.entities.values())
    return {"entity_types": dict(types), "provenance": dict(prov)}


def snapshot(*, registered_ids: Sequence[str], input_ids: Sequence[str], level: int, model_state: str,
             points: int, camera_poses: Sequence[tuple], world) -> dict:
    """What is stored on the version report and later compared against."""
    cams = {}
    if level >= 1:  # level 0 poses are assumed / display layout, not registration
        reg = set(registered_ids)
        cams = {p[0]: [float(p[1][0]), float(p[1][1]), float(p[1][2])] for p in camera_poses if p[0] in reg}
    return {
        "registered_ids": list(registered_ids), "input_ids": list(input_ids), "level": level,
        "model_state": model_state, "points": points, "cameras": cams, **structure_of(world),
    }


def snapshot_from_report(report: Optional[dict]) -> Optional[dict]:
    """The stored snapshot of a version, or None (first version / nothing recorded)."""
    if not report:
        return None
    ev = report.get("evidence") or {}
    if not ev.get("registered_ids") and not ev.get("input_ids") and not report.get("structure"):
        return None
    st = report.get("structure") or {}
    return {
        "registered_ids": list(ev.get("registered_ids") or []), "input_ids": list(ev.get("input_ids") or []),
        "level": report.get("level"), "model_state": report.get("model_state"),
        "points": ((report.get("stages") or {}).get("reconstruction") or {}).get("points"),
        "cameras": dict(report.get("cameras") or {}),
        "entity_types": st.get("entity_types"), "provenance": st.get("provenance"),
    }


def camera_consistency(prev: Dict[str, list], cand: Dict[str, list]) -> Optional[dict]:
    """How far previously-registered cameras moved, after the best similarity alignment
    (Umeyama). None when fewer than MIN_COMMON_CAMERAS are shared (not measurable)."""
    import numpy as np

    common = sorted(set(prev) & set(cand))
    if len(common) < MIN_COMMON_CAMERAS:
        return None
    a = np.asarray([prev[k] for k in common], float)
    b = np.asarray([cand[k] for k in common], float)
    ma, mb = a.mean(0), b.mean(0)
    ac, bc = a - ma, b - mb
    var = (ac ** 2).sum() / len(common)
    if var < 1e-12:
        return None
    u, s, vt = np.linalg.svd(bc.T @ ac / len(common))
    d = np.ones(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        d[-1] = -1
    r = u @ np.diag(d) @ vt
    scale = (s * d).sum() / var
    resid = bc - scale * (ac @ r.T)
    rms = float(np.sqrt((resid ** 2).sum() / len(common)))
    extent = float(np.sqrt((bc ** 2).sum() / len(common)))
    return {"common": len(common), "rms_residual": round(rms, 6),
            "relative": round(rms / extent, 4) if extent > 1e-12 else None}


def compute_delta(prev: Optional[dict], cand: dict) -> dict:
    if prev is None:
        return {
            "first_version": True,
            "evidence": {"registered": sorted(cand["registered_ids"]), "gained": sorted(cand["registered_ids"]),
                         "lost": [], "kept": [],
                         "still_waiting": sorted(set(cand["input_ids"]) - set(cand["registered_ids"]))},
            "level": {"before": None, "after": cand["level"]}, "points": {"before": None, "after": cand["points"]},
            "structure": {}, "camera_consistency": None, "not_measurable": [],
        }
    # Level 0 "registered" ids are the single view's assumed anchor pose, not a registration
    # against anything: only level >= 1 counts as placed, on BOTH sides of the comparison.
    pr = set(prev["registered_ids"]) if (prev.get("level") or 0) >= 1 else set()
    cr = set(cand["registered_ids"]) if (cand.get("level") or 0) >= 1 else set()
    not_measurable: List[str] = []
    cc = None
    if prev.get("cameras") and cand.get("cameras"):
        cc = camera_consistency(prev["cameras"], cand["cameras"])
    if cc is None:
        not_measurable.append(f"camera layout consistency (fewer than {MIN_COMMON_CAMERAS} shared "
                              "registered cameras, or the previous version stored none)")
    struct: Dict[str, dict] = {}
    if prev.get("entity_types") is None:
        not_measurable.append("structure counts (previous version did not record them)")
    else:
        for t in sorted(set(prev["entity_types"]) | set(cand["entity_types"])):
            b, a = prev["entity_types"].get(t, 0), cand["entity_types"].get(t, 0)
            if a != b:
                struct[t] = {"before": b, "after": a}
    return {
        "first_version": False,
        "evidence": {
            "registered": sorted(cr), "gained": sorted(cr - pr), "lost": sorted(pr - cr),
            "kept": sorted(pr & cr),
            "still_waiting": sorted(set(cand["input_ids"]) - cr),
            "new_input": sorted(set(cand["input_ids"]) - set(prev["input_ids"])),
        },
        "level": {"before": prev.get("level"), "after": cand.get("level")},
        "points": {"before": prev.get("points"), "after": cand.get("points")},
        "structure": struct,
        "camera_consistency": cc,
        "not_measurable": not_measurable,
    }


def decide(delta: dict) -> dict:
    """ACCEPT / ACCEPT_WITH_UNCERTAINTY / REJECT, each reason citing a measured fact."""
    if delta.get("first_version"):
        return {"verdict": ACCEPT, "reasons": [], "uncertainties": []}
    ev, lvl = delta["evidence"], delta["level"]
    reject: List[str] = []
    unc: List[str] = []
    lost, gained = ev["lost"], ev["gained"]
    if lost and len(gained) <= len(lost):
        reject.append(f"{len(lost)} previously registered photo(s) are no longer placed and only "
                      f"{len(gained)} were gained (no net gain)")
    elif lost:
        unc.append(f"{len(lost)} previously registered photo(s) are no longer placed "
                   f"(offset by {len(gained)} newly placed)")
    if (lvl["before"] is not None and lvl["after"] is not None and lvl["after"] < lvl["before"]
            and not gained):
        reject.append(f"reconstruction fell back from level {lvl['before']} to {lvl['after']} "
                      "without placing any additional photo")
    cc = delta.get("camera_consistency")
    if cc and cc.get("relative") is not None and cc["relative"] > CAMERA_MOVED_REL:
        unc.append(f"previously registered cameras moved by {cc['relative']:.0%} of the scene extent "
                   f"(over {cc['common']} shared cameras)")
    for t, c in delta["structure"].items():
        if t in STRUCTURAL and c["before"] >= 2 and c["after"] < c["before"] / 2:
            unc.append(f"{t} count fell from {c['before']} to {c['after']}")
    verdict = REJECT if reject else (ACCEPT_WITH_UNCERTAINTY if unc else ACCEPT)
    return {"verdict": verdict, "reasons": reject, "uncertainties": unc}


def describe_changes(delta: dict, structure: bool = True) -> List[str]:
    """Plain-language statements, each derived from a measured field of the delta.
    ``structure=False`` for a candidate that was NOT adopted: its walls/floors are not
    part of the world, so only evidence facts are reported."""
    out: List[str] = []
    ev = delta["evidence"]

    def s(n):
        return "" if n == 1 else "s"

    if delta.get("first_version"):
        n = len(ev["registered"])
        out.append(f"First model built with {n} placed photo{s(n)}.")
    else:
        if ev["gained"]:
            out.append(f"{len(ev['gained'])} more photo{s(len(ev['gained']))} placed in the model.")
        if ev.get("kept"):
            out.append(f"{len(ev['kept'])} photo{s(len(ev['kept']))} stayed placed.")
        if ev["lost"]:
            out.append(f"{len(ev['lost'])} photo{s(len(ev['lost']))} could no longer be placed.")
        for t, c in (delta["structure"].items() if structure else ()):
            d = c["after"] - c["before"]
            out.append(f"{'+' if d > 0 else ''}{d} {t}{s(abs(d))} ({c['before']} -> {c['after']}).")
        cc = delta.get("camera_consistency")
        if structure and cc and cc.get("relative") is not None:
            out.append("Camera positions " + ("shifted" if cc["relative"] > CAMERA_MOVED_REL else "stayed consistent")
                       + f" ({cc['relative']:.0%} of scene extent).")
    w = ev["still_waiting"]
    if w:
        out.append(f"{len(w)} photo{s(len(w))} still waiting for a photo that connects {'it' if len(w) == 1 else 'them'}.")
    return out
