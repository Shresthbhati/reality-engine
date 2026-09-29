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
  * entity continuity       -- ONLY when a camera alignment exists: the candidate's
                               entity centres are mapped into the previous frame and
                               matched to previous entities by TYPE and POSITION
                               (ids are re-derived on every rebuild, so ids are useless)

Checks that need data an older version did not record are reported as
"not measurable", never skipped silently and never passed.

Verdicts:
  ACCEPT                    nothing measurable regressed
  ACCEPT_WITH_UNCERTAINTY   adopted, but something moved, shrank or conflicts; reasons recorded
  REJECT                    net loss of registered evidence (or a downgrade with no gain):
                            HEAD stays authoritative, the new evidence is kept and retried

Conflicts (explicit, never silently resolved): a previously registered camera that lands in
materially different places in two versions is two competing hypotheses. Both are kept, with
provenance, until a later version agrees with one of them.

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
#: one camera displaced by more than this (of scene extent) opens a conflict
CAMERA_CONFLICT_REL = 0.25
#: a later version that puts the camera within this of the newer hypothesis resolves it
CAMERA_AGREE_REL = 0.08
MIN_COMMON_CAMERAS = 4
#: entity continuity: same type within this fraction of extent = matched; within PRESERVED = unchanged
ENTITY_MATCH_REL = 0.20
ENTITY_PRESERVED_REL = 0.05
#: structural entity types whose disappearance is worth flagging
STRUCTURAL = ("wall", "floor", "ceiling", "storey", "room", "corridor", "door", "window", "opening")
_SKIP_TYPES = ("sensor", "unknown")
MAX_ENTITIES_STORED = 400


def structure_of(world) -> dict:
    """Frame-independent counts of a WorldIR: entity types and provenance classes."""
    types = Counter(e.type.value for e in world.entities.values())
    prov = Counter(getattr(e.provenance, "value", str(e.provenance)) for e in world.entities.values())
    return {"entity_types": dict(types), "provenance": dict(prov)}


def entity_centres(world) -> List[dict]:
    """Type, centre, provenance and confidence per entity, for later continuity checks.
    Centre = geometry bounds centre when there is one, else the entity transform position."""
    out: List[dict] = []
    for e in world.entities.values():
        if e.type.value in _SKIP_TYPES:
            continue
        c = None
        for gid in e.geometry_ids:
            g = world.geometries.get(gid)
            if g is not None and g.bounds_min is not None and g.bounds_max is not None:
                c = [(g.bounds_min.x + g.bounds_max.x) / 2, (g.bounds_min.y + g.bounds_max.y) / 2,
                     (g.bounds_min.z + g.bounds_max.z) / 2]
                break
        if c is None:
            pos = (e.transform or {}).get("position")
            if pos:
                c = [float(pos.get("x", 0)), float(pos.get("y", 0)), float(pos.get("z", 0))]
        if c is None:
            continue
        out.append({"id": e.id, "type": e.type.value, "center": [round(float(v), 5) for v in c],
                    "provenance": getattr(e.provenance, "value", str(e.provenance)),
                    "confidence": round(float(e.confidence), 4)})
    return out[:MAX_ENTITIES_STORED]


def snapshot(*, registered_ids: Sequence[str], input_ids: Sequence[str], level: int, model_state: str,
             points: int, camera_poses: Sequence[tuple], world) -> dict:
    """What is stored on the version report and later compared against."""
    cams = {}
    if level >= 1:  # level 0 poses are assumed / display layout, not registration
        reg = set(registered_ids)
        cams = {p[0]: [float(p[1][0]), float(p[1][1]), float(p[1][2])] for p in camera_poses if p[0] in reg}
    return {
        "registered_ids": list(registered_ids), "input_ids": list(input_ids), "level": level,
        "model_state": model_state, "points": points, "cameras": cams,
        "entities": entity_centres(world) if level >= 1 else [], **structure_of(world),
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
        "cameras": dict(report.get("cameras") or {}), "entities": list(report.get("entities") or []),
        "entity_types": st.get("entity_types"), "provenance": st.get("provenance"),
    }


def _align(prev: Dict[str, list], cand: Dict[str, list]) -> Optional[dict]:
    """Best similarity (Umeyama) mapping previous camera positions onto candidate ones.
    cand ~= s * R (prev - ma) + mb.  None when fewer than MIN_COMMON_CAMERAS are shared."""
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
    scale = float((s * d).sum() / var)
    resid = np.linalg.norm(bc - scale * (ac @ r.T), axis=1)
    extent_c = float(np.sqrt((bc ** 2).sum() / len(common)))
    extent_p = float(np.sqrt((ac ** 2).sum() / len(common)))
    return {"common": common, "R": r, "s": scale, "ma": ma, "mb": mb, "resid": resid,
            "extent_c": extent_c, "extent_p": extent_p, "a": a, "b": b}


def camera_consistency(prev: Dict[str, list], cand: Dict[str, list]) -> Optional[dict]:
    """How far previously-registered cameras moved, after the best similarity alignment.
    None when not measurable. ``per_camera`` is each camera's residual as a fraction of extent."""
    al = _align(prev, cand)
    if al is None or al["extent_c"] <= 1e-12:
        return None
    import numpy as np

    rms = float(np.sqrt((al["resid"] ** 2).mean()))
    return {"common": len(al["common"]), "rms_residual": round(rms, 6),
            "relative": round(rms / al["extent_c"], 4),
            "per_camera": {k: round(float(x) / al["extent_c"], 4) for k, x in zip(al["common"], al["resid"])}}


def entity_continuity(prev_entities: List[dict], cand_entities: List[dict], prev_cams, cand_cams) -> Optional[dict]:
    """PRESERVED / REFINED / NEW / REMOVED by type + position, in the previous frame.
    None when there is no camera alignment or an older version stored no entities."""
    import numpy as np

    al = _align(prev_cams, cand_cams) if prev_cams and cand_cams else None
    if al is None or not prev_entities or al["s"] <= 1e-12 or al["extent_p"] <= 1e-12:
        return None
    tol, keep = ENTITY_MATCH_REL * al["extent_p"], ENTITY_PRESERVED_REL * al["extent_p"]

    def to_prev(c):  # candidate frame -> previous frame
        return al["R"].T @ (np.asarray(c, float) - al["mb"]) / al["s"] + al["ma"]

    cand_pts = [(e, to_prev(e["center"])) for e in cand_entities]
    used: set = set()
    out = {"preserved": 0, "refined": 0, "removed": [], "new": 0, "matched_by": "type + position after camera alignment"}
    for pe in prev_entities:
        best, bd = None, None
        for i, (ce, cp) in enumerate(cand_pts):
            if i in used or ce["type"] != pe["type"]:
                continue
            dist = float(np.linalg.norm(cp - np.asarray(pe["center"], float)))
            if bd is None or dist < bd:
                best, bd = i, dist
        if best is not None and bd <= tol:
            used.add(best)
            out["preserved" if bd <= keep else "refined"] += 1
        else:
            out["removed"].append({"type": pe["type"], "provenance": pe.get("provenance")})
    out["new"] = len(cand_pts) - len(used)
    return out


def compute_delta(prev: Optional[dict], cand: dict) -> dict:
    if prev is None:
        return {
            "first_version": True,
            "evidence": {"registered": sorted(cand["registered_ids"]), "gained": sorted(cand["registered_ids"]),
                         "lost": [], "kept": [],
                         "still_waiting": sorted(set(cand["input_ids"]) - set(cand["registered_ids"]))},
            "level": {"before": None, "after": cand["level"]}, "points": {"before": None, "after": cand["points"]},
            "structure": {}, "camera_consistency": None, "entities": None, "moved_cameras": [],
            "not_measurable": [],
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
    ents = None
    if prev.get("entities") and cand.get("entities") and cc is not None:
        ents = entity_continuity(prev["entities"], cand["entities"], prev["cameras"], cand["cameras"])
    if ents is None:
        not_measurable.append("entity continuity (needs a camera alignment and entity positions from both versions)")
    struct: Dict[str, dict] = {}
    if prev.get("entity_types") is None:
        not_measurable.append("structure counts (previous version did not record them)")
    else:
        for t in sorted(set(prev["entity_types"]) | set(cand["entity_types"])):
            b, a = prev["entity_types"].get(t, 0), cand["entity_types"].get(t, 0)
            if a != b:
                struct[t] = {"before": b, "after": a}
    moved: List[dict] = []
    if cc:
        al = _align(prev["cameras"], cand["cameras"])
        for i, k in enumerate(al["common"]):
            rel = cc["per_camera"][k]
            if rel > CAMERA_CONFLICT_REL:
                # both hypotheses expressed in the PREVIOUS frame so they are comparable
                cand_in_prev = (al["R"].T @ (al["b"][i] - al["mb"]) / al["s"] + al["ma"]).tolist()
                moved.append({"evidence_id": k, "displacement_rel": rel,
                              "previous_position": [round(float(v), 5) for v in al["a"][i]],
                              "candidate_position": [round(float(v), 5) for v in cand_in_prev]})
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
        "entities": ents,
        "moved_cameras": moved,
        "not_measurable": not_measurable,
    }


def reconcile_conflicts(prev_conflicts: Optional[List[dict]], delta: dict, tag: str) -> List[dict]:
    """Carry conflicts forward and update them from the delta. NEVER silently drops one:
    a conflict is only marked resolved when a later version measurably agrees, and the
    record (with its hypotheses and history) is kept."""
    out: List[dict] = []
    per_cam = (delta.get("camera_consistency") or {}).get("per_camera") or {}
    moved_ids = {m["evidence_id"]: m for m in delta.get("moved_cameras", [])}
    for c in prev_conflicts or []:
        c = dict(c, history=list(c.get("history") or []), hypotheses=list(c.get("hypotheses") or []))
        if c.get("status") == "unresolved" and c.get("kind") == "camera_pose":
            rel = per_cam.get(c["subject"])
            if c["subject"] in moved_ids:
                m = moved_ids[c["subject"]]
                c["hypotheses"].append({"source": tag, "position": m["candidate_position"],
                                        "confidence": None, "provenance": [c["subject"]]})
                c["history"].append({"version": tag, "event": "moved_again",
                                     "detail": f"displacement {m['displacement_rel']:.0%} of scene extent"})
            elif rel is not None and rel <= CAMERA_AGREE_REL:
                c["status"] = "resolved"
                c["history"].append({"version": tag, "event": "resolved",
                                     "detail": f"camera stayed within {rel:.0%} of scene extent between versions"})
            elif rel is not None:
                c["history"].append({"version": tag, "event": "still_unresolved",
                                     "detail": f"moved {rel:.0%} of scene extent (agreement needs <= {CAMERA_AGREE_REL:.0%})"})
        out.append(c)
    open_subjects = {c["subject"] for c in out if c.get("status") == "unresolved" and c.get("kind") == "camera_pose"}
    for k, m in moved_ids.items():
        if k in open_subjects:
            continue
        out.append({
            "id": f"conflict-camera-{k}-{tag}", "kind": "camera_pose", "subject": k, "status": "unresolved",
            "summary": ("The estimated position of this photo changed materially between versions; "
                        "both estimates are kept until a later version agrees with one."),
            "hypotheses": [
                {"source": "previous_version", "position": m["previous_position"], "confidence": None,
                 "provenance": [k]},
                {"source": tag, "position": m["candidate_position"], "confidence": None, "provenance": [k]},
            ],
            "displacement_rel": m["displacement_rel"], "opened_in": tag,
            "history": [{"version": tag, "event": "opened", "detail": f"{m['displacement_rel']:.0%} of scene extent"}],
            "confidence_note": "not measured: the reconstruction does not report per-camera confidence",
        })
    return out


def decide(delta: dict, conflicts: Optional[List[dict]] = None) -> dict:
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
    ents = delta.get("entities")
    if ents:
        removed_struct = [r for r in ents["removed"] if r["type"] in STRUCTURAL]
        prev_struct = ents["preserved"] + ents["refined"] + len(removed_struct)
        if prev_struct >= 3 and len(removed_struct) / prev_struct > 0.5:
            unc.append(f"{len(removed_struct)} of {prev_struct} previously established structural elements "
                       f"were not reproduced (matched by {ents['matched_by']})")
    fresh = [c for c in (conflicts or []) if c.get("status") == "unresolved"
             and (c.get("history") or [{}])[-1].get("event") in ("opened", "moved_again")]
    if fresh:
        unc.append(f"{len(fresh)} photo position(s) now conflict with the previous version and stay unresolved")
    verdict = REJECT if reject else (ACCEPT_WITH_UNCERTAINTY if unc else ACCEPT)
    return {"verdict": verdict, "reasons": reject, "uncertainties": unc}


def describe_changes(delta: dict, structure: bool = True, conflicts: Optional[List[dict]] = None) -> List[str]:
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
        ents = delta.get("entities") if structure else None
        if ents:
            out.append(f"{ents['preserved']} structural element{s(ents['preserved'])} preserved, "
                       f"{ents['refined']} refined, {ents['new']} newly observed"
                       + (f", {len(ents['removed'])} not reproduced." if ents["removed"] else "."))
        else:
            for t, c in (delta["structure"].items() if structure else ()):
                d = c["after"] - c["before"]
                out.append(f"{'+' if d > 0 else ''}{d} {t}{s(abs(d))} ({c['before']} -> {c['after']}).")
        cc = delta.get("camera_consistency")
        if structure and cc and cc.get("relative") is not None:
            out.append("Camera positions " + ("shifted" if cc["relative"] > CAMERA_MOVED_REL else "stayed consistent")
                       + f" ({cc['relative']:.0%} of scene extent).")
    if structure and conflicts:
        n_open = sum(1 for c in conflicts if c.get("status") == "unresolved")
        n_done = sum(1 for c in conflicts if c.get("status") == "resolved"
                     and (c.get("history") or [{}])[-1].get("event") == "resolved")
        if n_open:
            out.append(f"{n_open} conflicting position estimate{s(n_open)} still unresolved.")
        if n_done:
            out.append(f"{n_done} earlier conflict{s(n_done)} resolved by this version's evidence.")
    w = ev["still_waiting"]
    if w:
        out.append(f"{len(w)} photo{s(len(w))} still waiting for a photo that connects {'it' if len(w) == 1 else 'them'}.")
    return out
