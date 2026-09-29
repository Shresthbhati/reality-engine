"""Evidence guidance: tell the user what evidence would help MOST, from
measured uncertainty -- never a generic "upload more images".

Every guidance item is ``{kind, message, basis}``. ``basis`` states the
measurement the advice rests on, so a reader (or a test) can check that
the advice follows from data rather than boilerplate. Rules only fire on
things that were actually measured:

  * unregistered images (from the SfM result)
  * the largest empty gap between registered viewing directions
  * structure that touches the frame edge (single-view truncation)
  * roles never observed (no floor / no ceiling plane)
  * redundancy of the newest batch (from evidence contribution)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence


def camera_azimuths_deg(camera_poses: Sequence[tuple]) -> Dict[str, float]:
    """Horizontal viewing direction per camera. Poses are
    (evidence_id, position, (w,x,y,z) camera-to-world); the camera looks
    along its local +Z (COLMAP/OpenCV convention), world up is +Y."""
    out: Dict[str, float] = {}
    for eid, _pos, rot in camera_poses:
        w, x, y, z = rot
        # rotate (0,0,1) by q: standard quaternion-vector rotation
        dx = 2 * (x * z + w * y)
        dz = 1 - 2 * (x * x + y * y)
        if math.hypot(dx, dz) < 1e-6:
            continue  # looking straight up/down: no horizontal direction
        out[eid] = math.degrees(math.atan2(dx, dz)) % 360.0
    return out


def _largest_gap(az: Dict[str, float]):
    items = sorted(az.items(), key=lambda kv: kv[1])
    if len(items) < 2:
        return None
    best = None
    for i in range(len(items)):
        a, b = items[i], items[(i + 1) % len(items)]
        gap = (b[1] - a[1]) % 360.0
        if best is None or gap > best[0]:
            best = (gap, a[0], b[0])
    return best


def coverage_degrees(camera_poses: Sequence[tuple], scene_center: Optional[tuple]) -> Optional[dict]:
    """How much of the full circle the registered cameras cover.

    Object-centric captures (cameras look TOWARD the scene centre, e.g. a
    facade or a room corner from outside) are measured by the bearing of the
    camera POSITIONS around the centre. Interior captures (centre enclosed,
    cameras look outward) are measured by VIEWING DIRECTION. None when fewer
    than two cameras have a horizontal direction."""
    az = camera_azimuths_deg(camera_poses)
    if len(az) < 2:
        return None
    mode, angles = "viewing_directions", dict(az)
    if scene_center is not None:
        inward, bearings = 0.0, {}
        for eid, pos, _rot in camera_poses:
            if eid not in az:
                continue
            dx, dz = scene_center[0] - pos[0], scene_center[1] - pos[2]
            n = math.hypot(dx, dz)
            if n < 1e-9:
                continue
            vx, vz = math.sin(math.radians(az[eid])), math.cos(math.radians(az[eid]))
            inward += (vx * dx + vz * dz) / n
            bearings[eid] = math.degrees(math.atan2(-dx, -dz)) % 360.0
        if len(bearings) >= 2 and inward / len(bearings) > 0.5:
            mode, angles = "around_scene", bearings
    vals = sorted(angles.values())
    gap = _largest_gap(angles)
    return {"degrees": 360.0 - gap[0] if gap else 0.0, "mode": mode, "gap": gap,
            "count": len(vals)}


def build_guidance(
    *,
    level: int,
    entity_roles: Sequence[str],
    names: Dict[str, str],
    input_ids: Sequence[str],
    registered_ids: Sequence[str],
    camera_poses: Sequence[tuple],
    contribution_summary: Optional[dict],
    bootstrap_facts: Optional[dict],
    attempts: Sequence[dict],
    scene_center: Optional[tuple] = None,
) -> List[dict]:
    out: List[dict] = []

    def nm(eid: str) -> str:
        return names.get(eid, eid)

    # ---- single-view (bootstrap) advice, from what was measured in the frame ----
    if level == 0:
        n_fused = int((bootstrap_facts or {}).get("fused_views") or 0)
        if n_fused > 1:
            out.append({
                "kind": "add_overlap",
                "message": (f"We built a rough model from each of your {n_fused} photos, but could not place "
                            "them relative to each other. A photo that shows part of the same surface as a "
                            "neighbouring photo lets the engine join them."),
                "basis": "no photo pair could be registered; each photo is an independent single-view estimate",
            })
        else:
            out.append({
                "kind": "add_viewpoint",
                "message": ("We built a rough spatial model from what is visible. Add photos from a different "
                            "position so the engine can measure real depth instead of estimating it."),
                "basis": "one viewpoint: depth is a monocular estimate, scale is relative",
            })
        trunc = (bootstrap_facts or {}).get("edge_truncation") or {}
        for side in ("left", "right"):
            share = trunc.get(side, 0.0)
            if share >= 0.10:
                out.append({
                    "kind": "capture_direction",
                    "message": f"Structure continues past the {side} edge of the photo: add a photo aimed further {side}.",
                    "basis": f"{share:.0%} of a detected surface's supporting pixels touch the {side} frame border",
                })
        if trunc.get("top", 0.0) >= 0.10:
            out.append({
                "kind": "capture_direction",
                "message": "A surface is cut off at the top of the frame: tilt up or step back to capture it.",
                "basis": f"{trunc['top']:.0%} of a detected surface touches the top frame border",
            })
        roles = set(entity_roles)
        if "wall" in roles and "floor" not in roles:
            out.append({
                "kind": "unobserved",
                "message": "No ground/floor surface was visible; include the base of the scene in a photo.",
                "basis": "walls detected but no horizontal upward-facing plane",
            })
        out.append({
            "kind": "unobserved",
            "message": "Rear and side surfaces are unobserved (unknown), not guessed.",
            "basis": "a single view only sees the surfaces facing the camera",
        })

    # ---- multi-view advice ----
    else:
        unreg = [e for e in input_ids if e not in set(registered_ids)]
        if unreg:
            shown = ", ".join(nm(e) for e in unreg[:4]) + (" ..." if len(unreg) > 4 else "")
            out.append({
                "kind": "retake",
                "message": (f"{len(unreg)} image(s) could not be placed in the model ({shown}). "
                            "Retake them with more overlap with a neighbouring photo."),
                "basis": f"{len(registered_ids)} of {len(input_ids)} images registered",
            })
        cov = coverage_degrees(camera_poses, scene_center)
        if cov and cov["gap"] and cov["gap"][0] >= 90.0:
            gap_deg, ga, gb = cov["gap"]
            if cov["mode"] == "around_scene":
                out.append({
                    "kind": "capture_direction",
                    "message": (f"All photos were taken from one side: they cover {cov['degrees']:.0f} "
                                "degrees of the way around the scene. Walk around and add photos from "
                                "the sides and back; those surfaces are currently unobserved."),
                    "basis": (f"camera positions span {cov['degrees']:.0f} of 360 degrees of bearing "
                              f"around the scene centre (largest empty arc {gap_deg:.0f} degrees)"),
                })
            else:
                out.append({
                    "kind": "capture_direction",
                    "message": (f"No photo looks into a {gap_deg:.0f}-degree arc between "
                                f"{nm(ga)} and {nm(gb)}: add a photo aimed between them."),
                    "basis": "largest empty angular gap between registered camera viewing directions",
                })
        roles = set(entity_roles)
        if roles and "ceiling" not in roles and "floor" in roles:
            out.append({
                "kind": "unobserved",
                "message": "No ceiling was observed; tilt a photo upward to capture it.",
                "basis": "floor plane detected, no ceiling plane",
            })
        if roles and "floor" not in roles and "wall" in roles:
            out.append({
                "kind": "unobserved",
                "message": "No floor/ground was observed; include the base of the walls in a photo.",
                "basis": "wall planes detected, no floor plane",
            })
        if contribution_summary:
            new = contribution_summary.get("new_images", 0)
            red = contribution_summary.get("redundant", 0)
            if new and red / new >= 0.5:
                out.append({
                    "kind": "redundant",
                    "message": ("Most of the latest photos repeat viewpoints you already have; more images "
                                "from this direction are unlikely to add much. Move to a new position."),
                    "basis": f"{red} of {new} new images had near-identical viewpoints to existing ones",
                })

    # ---- what the engine tried and could not do ----
    for a in attempts:
        if a.get("outcome") == "failed" and a.get("level", 0) >= 1:
            out.append({
                "kind": "engine_note",
                "message": ("We couldn't establish reliable multi-view geometry from these photos, usually "
                            "because neighbouring photos share too little of the same scene. "
                            "Showing the strongest result that was possible; your photos are kept."),
                "basis": f"attempt at level {a['level']} ({a.get('name')}) failed: {a.get('detail', 'no detail')}"[:500],
            })
    if not out:
        out.append({
            "kind": "improve",
            "message": "Add views from angles you have not photographed to make the model more complete.",
            "basis": "no specific gap was measured",
        })
    return out
