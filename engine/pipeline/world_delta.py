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
#: a report keeps at most this many sampled points in total; entities beyond it fall back to centre matching
MAX_STORED_POINTS = 24000
#: share of previously established structural surfaces that vanish: uncertainty above the first, regression above the second
UNCERTAIN_REMOVED_FRACTION = 0.34
REGRESSION_REMOVED_FRACTION = 0.60


def structure_of(world) -> dict:
    """Frame-independent counts of a WorldIR: entity types and provenance classes."""
    types = Counter(e.type.value for e in world.entities.values())
    prov = Counter(getattr(e.provenance, "value", str(e.provenance)) for e in world.entities.values())
    return {"entity_types": dict(types), "provenance": dict(prov)}


def entity_records(world, artifact_store=None) -> List[dict]:
    """Per entity, everything spatial_continuity can use: type, centre, bounds, provenance, confidence,
    the photographs that observed it and (when its geometry payload can be read back) a sample of its
    inlier points. Anything that is not available is simply absent -- never filled in."""
    import struct

    from world_ir.geometry_data import PointCloudData

    from engine.pipeline.spatial_continuity import sample_points

    out: List[dict] = []
    budget = MAX_STORED_POINTS          # total sampled points kept on one report
    for e in world.entities.values():
        if e.type.value in _SKIP_TYPES:
            continue
        c = bounds = pts = None
        for gid in e.geometry_ids:
            g = world.geometries.get(gid)
            if g is None:
                continue
            if g.bounds_min is not None and g.bounds_max is not None:
                bounds = [[g.bounds_min.x, g.bounds_min.y, g.bounds_min.z],
                          [g.bounds_max.x, g.bounds_max.y, g.bounds_max.z]]
                c = [(g.bounds_min.x + g.bounds_max.x) / 2, (g.bounds_min.y + g.bounds_max.y) / 2,
                     (g.bounds_min.z + g.bounds_max.z) / 2]
            if artifact_store is not None and g.data_uri and pts is None and budget > 0:
                try:
                    pts = sample_points(PointCloudData.from_bytes(artifact_store.get(g.data_uri)).points)
                except (KeyError, ValueError, OSError, struct.error):
                    pts = None      # unreadable payload = no shape signal, never a guess
            if c is not None:
                break
        if c is None:
            pos = (e.transform or {}).get("position")
            if pos:
                c = [float(pos.get("x", 0)), float(pos.get("y", 0)), float(pos.get("z", 0))]
        if c is None:
            continue
        rec = {"id": e.id, "type": e.type.value, "center": [round(float(v), 5) for v in c],
               "provenance": getattr(e.provenance, "value", str(e.provenance)),
               "confidence": round(float(e.confidence), 4)}
        if bounds is not None:
            rec["bounds"] = [[round(float(v), 4) for v in row] for row in bounds]
        if pts:
            rec["points"] = pts
            budget -= len(pts)
        support = e.custom_properties.get("supporting_evidence_ids")
        if support is not None:
            rec["evidence"] = list(support)
        out.append(rec)
    return out[:MAX_ENTITIES_STORED]


entity_centres = entity_records   # earlier name, same function (records are a superset)


def snapshot(*, registered_ids: Sequence[str], input_ids: Sequence[str], level: int, model_state: str,
             points: int, camera_poses: Sequence[tuple], world, artifact_store=None) -> dict:
    """What is stored on the version report and later compared against."""
    cams = {}
    if level >= 1:  # level 0 poses are assumed / display layout, not registration
        reg = set(registered_ids)
        cams = {p[0]: [float(p[1][0]), float(p[1][1]), float(p[1][2])] for p in camera_poses if p[0] in reg}
    return {
        "registered_ids": list(registered_ids), "input_ids": list(input_ids), "level": level,
        "model_state": model_state, "points": points, "cameras": cams,
        "entities": entity_records(world, artifact_store) if level >= 1 else [], **structure_of(world),
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


def entity_continuity(prev_entities: List[dict], cand_entities: List[dict], prev_cams, cand_cams,
                      new_evidence_ids: Sequence[str] = ()) -> Optional[dict]:
    """How the previous version's structure is carried by the candidate, in the PREVIOUS frame.

    Geometry-aware (see spatial_continuity): footprint overlap of fitted planes where points were stored,
    centre distance otherwise; splits, merges, extensions and reductions are told apart from real
    additions and removals. None when there is no camera alignment or an older version stored no entities.
    The legacy keys (preserved / refined ints, removed list, new int) are kept for existing consumers."""
    from engine.pipeline import spatial_continuity

    al = _align(prev_cams, cand_cams) if prev_cams and cand_cams else None
    if al is None or not prev_entities or al["s"] <= 1e-12 or al["extent_p"] <= 1e-12:
        return None

    def to_prev(pts):  # candidate frame -> previous frame; rows are points
        return ((pts - al["mb"]) @ al["R"]) / al["s"] + al["ma"]

    out = spatial_continuity.reconcile(prev_entities, cand_entities, ext_prev=al["extent_p"], to_prev=to_prev,
                                       scale=al["s"], new_evidence_ids=new_evidence_ids)
    c = out["counts"]
    out.update(
        preserved=c["preserved"], refined=c["refined"], new=c["new"], ext_prev=al["extent_p"],
        removed=[{"type": r["type"], "provenance": r.get("provenance")} for r in out["relations"] if r["kind"] == "removed"],
    )
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
            "alignment": None, "not_measurable": [],
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
    new_input = sorted(set(cand["input_ids"]) - set(prev["input_ids"]))
    ents = None
    if prev.get("entities") and cand.get("entities") and cc is not None:
        ents = entity_continuity(prev["entities"], cand["entities"], prev["cameras"], cand["cameras"],
                                 new_evidence_ids=new_input)
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
    alignment = None
    if cc:
        al = _align(prev["cameras"], cand["cameras"])
        # kept so conflict hypotheses can be carried from the previous frame into the candidate's frame
        alignment = {"R": al["R"].tolist(), "s": al["s"], "ma": al["ma"].tolist(), "mb": al["mb"].tolist()}
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
            "new_input": new_input,
        },
        "level": {"before": prev.get("level"), "after": cand.get("level")},
        "points": {"before": prev.get("points"), "after": cand.get("points")},
        "structure": struct,
        "camera_consistency": cc,
        "entities": ents,
        "moved_cameras": moved,
        "alignment": alignment,
        "not_measurable": not_measurable,
    }


def _reconcile_camera_conflicts(prev_conflicts: Optional[List[dict]], delta: dict, tag: str) -> List[dict]:
    """Camera-pose conflicts. Carry conflicts forward and update them from the delta. NEVER silently drops one:
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


#: a candidate surface this close (x scene extent) to a conflict's current hypothesis is the same surface
GEOMETRY_MATCH_REL = 0.05


def _reconcile_geometry_conflicts(conflicts: List[dict], delta: dict, tag: str) -> List[dict]:
    """Geometry conflicts: a surface that MOVED beyond tolerance while no new photograph supports the move
    keeps BOTH positions (with the evidence and confidence each side really has) until a later version
    confirms one. Resolution needs new supporting evidence; time passing is not evidence."""
    import numpy as np

    ents = delta.get("entities") or {}
    rels = ents.get("relations", [])
    ext = float(ents.get("ext_prev") or 0.0)
    tol = GEOMETRY_MATCH_REL * ext
    out: List[dict] = []
    for c in conflicts:
        if c.get("kind") != "geometry" or c.get("status") != "unresolved":
            out.append(c)
            continue
        c = dict(c, history=list(c.get("history") or []), hypotheses=list(c.get("hypotheses") or []))
        cur = np.asarray(c["hypotheses"][-1]["position"], float)
        first = np.asarray(c["hypotheses"][0]["position"], float)
        near = []
        if ents and ext > 0 and not c.get("frame_lost"):
            for r in rels:
                if (r["type"] == c["subject_type"] and r["kind"] in ("preserved", "refined")
                        and r.get("prev_center") is not None
                        and float(np.linalg.norm(np.asarray(r["prev_center"], float) - cur)) <= tol):
                    near.append(r)
        if not near:
            c["history"].append({"version": tag, "event": "still_unresolved",
                                 "detail": "the surface could not be matched in this version (not measurable)"})
        else:
            r = min(near, key=lambda r: float(np.linalg.norm(np.asarray(r["prev_center"], float) - cur)))
            support = r.get("supporting_new_evidence") or []
            back = float(np.linalg.norm(np.asarray(r["cand_center"], float) - first))
            stay = float(np.linalg.norm(np.asarray(r["cand_center"], float) - cur))
            if r["kind"] == "preserved" and support:
                c["status"], c["resolved_to"] = "resolved", c["hypotheses"][-1]["source"]
                c["history"].append({"version": tag, "event": "resolved", "detail":
                                     f"the adopted position was preserved and {len(support)} new photograph(s) support it"})
            elif r["kind"] == "refined" and support and back <= tol and back < stay:
                c["status"], c["resolved_to"] = "resolved", "previous_version"
                c["history"].append({"version": tag, "event": "resolved", "detail":
                                     "the surface returned to its earlier position, supported by new photographs"})
            elif r["kind"] == "refined" and stay > tol:
                c["hypotheses"].append({"source": tag, "position": r["cand_center"], "entity": r["cand"][0],
                                        "confidence": r.get("cand_confidence"), "provenance": r.get("cand_evidence")})
                c["history"].append({"version": tag, "event": "moved_again",
                                     "detail": f"moved another {r['gap_rel']:.0%} of scene extent"})
            else:
                c["history"].append({"version": tag, "event": "still_unresolved",
                                     "detail": "unchanged, and no new photograph supports either position yet"})
        out.append(c)

    open_now = [c for c in out if c.get("kind") == "geometry" and c.get("status") == "unresolved"]
    for r in rels:
        if not r.get("unsupported_move"):
            continue
        centre = np.asarray(r["prev_center"], float)
        if any(c["subject_type"] == r["type"]
               and float(np.linalg.norm(np.asarray(c["hypotheses"][-1]["position"], float) - centre)) <= tol
               for c in open_now):
            continue
        out.append({
            "id": f"conflict-geometry-{r['type']}-{len([c for c in out if c.get('kind') == 'geometry']) + 1}-{tag}",
            "kind": "geometry", "subject_type": r["type"], "subject": f"{r['type']}:{r['prev'][0]}",
            "status": "unresolved",
            "summary": (f"A {r['type']} surface moved or tilted more than the matching tolerance "
                        "and no new photograph supports the change; both positions are kept."),
            "hypotheses": [
                {"source": "previous_version", "position": r["prev_center"], "entity": r["prev"][0],
                 "confidence": r.get("prev_confidence"), "provenance": r.get("prev_evidence")},
                {"source": tag, "position": r["cand_center"], "entity": r["cand"][0],
                 "confidence": r.get("cand_confidence"), "provenance": r.get("cand_evidence")},
            ],
            "displacement_rel": r["gap_rel"], "angle_deg": r.get("angle_deg"), "opened_in": tag,
            "history": [{"version": tag, "event": "opened",
                         "detail": f"offset {r['gap_rel']:.0%} of scene extent, tilt {r.get('angle_deg') or 0:.1f} deg"}],
        })
    return out


def _carry_to_candidate_frame(conflicts: List[dict], delta: dict) -> List[dict]:
    """Hypothesis positions are recorded in the frame of the version being replaced. The candidate has its
    own frame, so every UNRESOLVED conflict's positions are mapped into it (similarity from the shared
    cameras); without an alignment the conflict is kept and marked ``frame_lost`` rather than dropped."""
    import numpy as np

    al = delta.get("alignment")
    out: List[dict] = []
    for c in conflicts:
        if c.get("status") != "unresolved" or not c.get("hypotheses"):
            out.append(c)
            continue
        c = dict(c, hypotheses=[dict(h) for h in c["hypotheses"]])
        if al is None:
            c["frame_lost"] = True
        else:
            R, s = np.asarray(al["R"], float), float(al["s"])
            ma, mb = np.asarray(al["ma"], float), np.asarray(al["mb"], float)
            for h in c["hypotheses"]:
                if h.get("position") is not None:
                    h["position"] = [round(float(v), 5) for v in s * ((np.asarray(h["position"], float) - ma) @ R.T) + mb]
            c.pop("frame_lost", None)
        out.append(c)
    return out


def reconcile_conflicts(prev_conflicts: Optional[List[dict]], delta: dict, tag: str) -> List[dict]:
    """All conflict kinds (camera pose, geometry): carried forward, opened, resolved -- never dropped.
    The returned positions are expressed in the CANDIDATE's frame, ready to be the next version's 'previous'."""
    conflicts = _reconcile_camera_conflicts(prev_conflicts, delta, tag)
    conflicts = _reconcile_geometry_conflicts(conflicts, delta, tag)
    return _carry_to_candidate_frame(conflicts, delta)


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
        # Splits, merges, extensions and refinements are CHANGES OF REPRESENTATION and keep the structure;
        # only surfaces nothing in the candidate overlaps ("removed") or that are only partly reproduced
        # ("reduced") count against it.
        prev_struct = sum(len(r["prev"]) for r in ents["relations"] if r["type"] in STRUCTURAL)
        removed = sum(len(r["prev"]) for r in ents["relations"] if r["kind"] == "removed" and r["type"] in STRUCTURAL)
        reduced = sum(len(r["prev"]) for r in ents["relations"] if r["kind"] == "reduced" and r["type"] in STRUCTURAL)
        ambiguous = sum(1 for r in ents["relations"] if r["kind"] == "ambiguous" and r["type"] in STRUCTURAL)
        if prev_struct >= 3 and removed / prev_struct >= REGRESSION_REMOVED_FRACTION and not gained:
            reject.append(f"{removed} of {prev_struct} previously established structural surfaces are not "
                          "reproduced and no additional photograph was placed to justify the loss "
                          f"(matched by {ents['matched_by']})")
        elif prev_struct >= 3 and removed / prev_struct >= UNCERTAIN_REMOVED_FRACTION:
            unc.append(f"{removed} of {prev_struct} previously established structural surfaces are not "
                       f"reproduced (matched by {ents['matched_by']})")
        if reduced:
            unc.append(f"{reduced} previously established surface(s) are only partly reproduced")
        if ambiguous:
            unc.append(f"{ambiguous} structural change(s) overlap in a pattern that cannot be classified as "
                       "a split, merge or move")
    fresh = [c for c in (conflicts or []) if c.get("status") == "unresolved"
             and (c.get("history") or [{}])[-1].get("event") in ("opened", "moved_again")]
    if fresh:
        unc.append(f"{len(fresh)} photo position(s) now conflict with the previous version and stay unresolved")
    verdict = REJECT if reject else (ACCEPT_WITH_UNCERTAINTY if unc else ACCEPT)
    return {"verdict": verdict, "reasons": reject, "uncertainties": unc}


def _phrase(types: List[str]) -> str:
    """['wall', 'wall', 'floor'] -> '2 walls, 1 floor'."""
    return ", ".join(f"{n} {t}{'' if n == 1 else 's'}" for t, n in sorted(Counter(types).items()))


def physical_changes(delta: dict, conflicts: Optional[List[dict]] = None) -> dict:
    """What changed in the WORLD (not in the reconstruction's bookkeeping), derived from the geometric
    relations between the two versions. ``available`` is False, with the reason, when the relations
    could not be measured -- the caller then falls back to plain counts and says so."""
    ents = delta.get("entities")
    if not ents:
        return {"available": False,
                "reason": "; ".join(delta.get("not_measurable") or []) or "no previous structure to compare"}
    rels = ents["relations"]

    def of(kind):
        return [r for r in rels if r["kind"] == kind]

    new_input = delta["evidence"].get("new_input") or []
    out: dict = {"available": True, "signals": ents.get("signals", {})}
    # items are noun phrases: the label ("Added", "Extended", ...) carries the verb
    out["added"] = [_phrase([r["type"] for r in of("new")])] if of("new") else []
    out["extended"] = [f"{_phrase([r['type'] for r in of('extended')])} (now reach further than before)"] if of("extended") else []
    refined = of("refined")
    supported = [r for r in refined if r.get("supporting_new_evidence")]
    unsupported = [r for r in refined if r.get("unsupported_move")]
    out["refined"] = []
    if refined:
        out["refined"].append(f"{_phrase([r['type'] for r in refined])} (position or orientation adjusted)")
    if supported:
        out["refined"].append(f"{len(supported)} of these backed by new photographs")
    out["preserved"] = len(of("preserved"))
    differently = [r for r in rels if r["kind"] in ("split", "merge", "regrouped")]

    def _regroup_text(r):
        if r["kind"] == "split":
            return f"one {r['type']} is now {len(r['cand'])} fragments"
        if r["kind"] == "merge":
            return f"{len(r['prev'])} {r['type']}s are now one surface"
        return f"{len(r['prev'])} {r['type']}s are now {len(r['cand'])} {r['type']}s covering the same area"

    out["represented_differently"] = [_regroup_text(r) + " (same surface, different grouping)" for r in differently]
    lost = of("removed") + of("reduced")
    out["not_reproduced"] = [f"{_phrase([r['type'] for r in lost])} from the previous version"] if lost else []
    out["uncertain"] = [r["explanation"] for r in of("ambiguous")]
    out["unsupported_moves"] = len(unsupported)
    open_c = [c for c in (conflicts or []) if c.get("status") == "unresolved"]
    out["conflicts"] = {"unresolved": len(open_c), "kinds": dict(Counter(c.get("kind") for c in open_c))}
    out["regions"] = []
    for reg in ents["regions"]:
        touched = reg.get("affected_by_new_evidence")
        parts = [f"{n} {k}" for k, n in sorted(reg["relations"].items())]
        out["regions"].append({
            "id": reg["id"], "status": reg["status"], "types": reg["types"], "relations": reg["relations"],
            "affected_by_new_evidence": touched, "new_evidence_count": len(touched) if touched is not None else None,
            "summary": (f"{_phrase([t for t, n in reg['types'].items() for _ in range(n)])}: " + ", ".join(parts)
                        + (f"; touched by {len(touched)} of {len(new_input)} new photo(s)" if touched else "")),
        })
    return out


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
        phys = physical_changes(delta, conflicts) if structure else {"available": False}
        if phys["available"]:
            if phys["preserved"]:
                out.append(f"Preserved: {phys['preserved']} surface{s(phys['preserved'])} unchanged.")
            for label, key in (("Newly observed", "added"), ("Extended", "extended"), ("Refined", "refined"),
                               ("Represented differently", "represented_differently"),
                               ("Not reproduced", "not_reproduced"), ("Still uncertain", "uncertain")):
                for line in phys[key]:
                    out.append(f"{label}: {line}.")
            if phys["unsupported_moves"]:
                out.append(f"{phys['unsupported_moves']} surface move(s) are not supported by any new photograph.")
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
