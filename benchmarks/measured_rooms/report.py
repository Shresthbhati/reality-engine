"""Measured-room accuracy report (build-plan task 12.2).

    python -m benchmarks.measured_rooms.report <truth.json> <world.json> [--calibrate] [--json]

``truth.json`` is a tape/laser measurement of a real room (schema ``measured_room/1``, see README.md); ``world.json`` is the
engine's WorldIR for the same room (``GET /api/worlds/{id}/worldir``). Every element is compared against the plan's tolerances
(wall length +/-2 cm, room area +/-1 %, ceiling height +/-2 cm, opening width +/-2 cm: PROPOSED, to be confirmed with firms in H-0.1).

Honesty rules:
  * a wall the engine never produced is a MISS in the table, never silently skipped;
  * dimensions in model units are never treated as metres without saying so (``scale_calibrated`` False and a note on each row);
  * the engine's own stated +/- is checked against the truth (``uncertainty_honesty``): an over-confident engine is a finding;
  * ``--calibrate`` fits ONE scale factor from the measured walls so the remaining error is shape error, not scale error.
"""
from __future__ import annotations

import json
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: PROPOSED (docs/blueprint/33_TURNAROUND_PLAN.md section 1): confirm with 2-3 firms.
TOLERANCES = {"wall_length_m": 0.02, "room_area_rel": 0.01, "ceiling_height_m": 0.02, "opening_width_m": 0.02}


def _entities(world: Dict[str, Any]) -> List[Dict[str, Any]]:
    ents = world.get("entities") or {}
    return list(ents.values()) if isinstance(ents, dict) else list(ents)


def _dim(ent: Optional[Dict[str, Any]], key: str) -> Optional[Dict[str, Any]]:
    if ent is None:
        return None
    d = ((ent.get("custom_properties") or {}).get("dimensions") or {}).get(key)
    return d if isinstance(d, dict) and isinstance(d.get("value"), (int, float)) else None


def _is_model_units(dim: Dict[str, Any]) -> bool:
    return str(dim.get("unit") or "").startswith("model_units")


def _match(truth_vals: Sequence[float], engine_vals: Sequence[float]) -> List[Tuple[int, int]]:
    """Optimal one-to-one pairing (minimum total absolute difference) of truth indices to engine indices."""
    if not truth_vals or not engine_vals:
        return []
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    cost = np.abs(np.asarray(truth_vals)[:, None] - np.asarray(engine_vals)[None, :])
    r, c = linear_sum_assignment(cost)
    return [(int(i), int(j)) for i, j in zip(r, c)]


def _row(element: str, truth: float, engine: Optional[float], tol_abs: Optional[float], tol_rel: Optional[float],
         unc: Optional[float], unit_note: str, scale: float = 1.0, area: bool = False) -> Dict[str, Any]:
    if engine is None:
        return {"element": element, "truth": truth, "engine": None, "error": None,
                "tolerance": tol_abs if tol_abs is not None else tol_rel, "pass": False, "stated_uncertainty": None,
                "within_stated_uncertainty": None, "note": "not produced by the engine (missed)"}
    k = scale ** 2 if area else scale
    eng = engine * k
    unc_s = None if unc is None else unc * k
    err = eng - truth
    if tol_rel is not None:
        ok: Optional[bool] = abs(err) <= tol_rel * abs(truth)
        tol: Optional[float] = tol_rel
    elif tol_abs is not None:
        ok, tol = abs(err) <= tol_abs, tol_abs
    else:
        ok, tol = None, None
    within = None if unc_s is None else abs(err) <= unc_s
    return {"element": element, "truth": truth, "engine": eng, "error": err, "tolerance": tol, "pass": ok,
            "stated_uncertainty": unc_s, "within_stated_uncertainty": within, "note": unit_note}


def evaluate(truth: Dict[str, Any], world: Dict[str, Any], tolerances: Optional[Dict[str, float]] = None,
             calibrate: bool = False) -> Dict[str, Any]:
    tol = {**TOLERANCES, **(tolerances or {})}
    ents = _entities(world)
    walls = [(e, _dim(e, "length_m")) for e in ents if e.get("type") == "wall"]
    walls = [(e, d) for e, d in walls if d is not None]
    rooms = [e for e in ents if e.get("type") == "room"]
    room = rooms[0] if rooms else None
    windows = [e for e in ents if e.get("type") == "window"]
    doors = [e for e in ents if e.get("type") == "door"]

    any_model_units = any(_is_model_units(d) for _, d in walls) or any(
        _is_model_units(d) for r in rooms for d in (_dim(r, "height_m"), _dim(r, "floor_area_m2")) if d)
    scale_state = str(((world.get("metadata") or {}).get("scale") or {}).get("state") or "").lower()
    uncalibrated = any_model_units or scale_state in ("relative", "unknown", "")

    t_walls = [float(w["length_m"]) for w in truth.get("walls", [])]
    e_walls = [float(d["value"]) for _, d in walls]
    scale = 1.0
    pairs = _match(t_walls, e_walls)
    if calibrate and pairs:
        num = sum(t_walls[i] * e_walls[j] for i, j in pairs)
        den = sum(e_walls[j] ** 2 for _, j in pairs)
        scale = num / den if den else 1.0
        pairs = _match(t_walls, [v * scale for v in e_walls])
    note = ("model units treated as metres: the scale is not calibrated" if (uncalibrated and not calibrate) else
            (f"after one fitted scale factor ({scale:.4f})" if calibrate else ""))
    rows: List[Dict[str, Any]] = []
    matched_t = {i: j for i, j in pairs}
    for i, tv in enumerate(t_walls):
        j = matched_t.get(i)
        d = walls[j][1] if j is not None else None
        rows.append(_row("wall length", tv, None if d is None else float(d["value"]), tol["wall_length_m"], None,
                         None if d is None else d.get("uncertainty_m"), note, scale))

    if truth.get("floor_area_m2") is not None:
        d = _dim(room, "floor_area_m2")
        r = _row("room area", float(truth["floor_area_m2"]), None if d is None else float(d["value"]), None, tol["room_area_rel"],
                 None if d is None else d.get("uncertainty_m"), note, scale, area=True)
        if d is not None and (room.get("custom_properties") or {}).get("visible_part_only"):
            r["note"] = (r["note"] + "; " if r["note"] else "") + "engine area is the visible part only (a lower bound)"
        rows.append(r)
    if truth.get("ceiling_height_m") is not None:
        d = _dim(room, "height_m")
        rows.append(_row("ceiling height", float(truth["ceiling_height_m"]), None if d is None else float(d["value"]),
                         tol["ceiling_height_m"], None, None if d is None else d.get("uncertainty_m"), note, scale))

    for kind, pool in (("window", windows), ("door", doors)):
        t_ops = [o for o in truth.get("openings", []) if o.get("kind") == kind]
        e_w = [(e, _dim(e, "width_m")) for e in pool]
        e_w = [(e, d) for e, d in e_w if d is not None]
        pairing = dict(_match([float(o["width_m"]) for o in t_ops], [float(d["value"]) * scale for _, d in e_w])) if t_ops else {}
        for i, o in enumerate(t_ops):
            j = pairing.get(i)
            ent = e_w[j][0] if j is not None else None
            for key, label, tolerance in (("width_m", f"{kind} width", tol["opening_width_m"]), ("height_m", f"{kind} height", None),
                                          ("sill_m", f"{kind} sill", None)):
                if o.get(key) is None:
                    continue
                d = _dim(ent, key)
                rows.append(_row(label, float(o[key]), None if d is None else float(d["value"]), tolerance, None,
                                 None if d is None else d.get("uncertainty_m"), note, scale))

    evaluated = [r for r in rows if r["pass"] is not None]
    with_unc = [r for r in rows if r["within_stated_uncertainty"] is not None]
    summary: Dict[str, Any] = {
        "room": truth.get("room"), "method": truth.get("method"),
        "elements_evaluated": len(evaluated), "passed": sum(1 for r in evaluated if r["pass"]),
        "failed": sum(1 for r in evaluated if not r["pass"]),
        "walls_missed": max(0, len(t_walls) - len(pairs)), "walls_extra": max(0, len(e_walls) - len(pairs)),
        "scale_calibrated": (not uncalibrated) or calibrate,
        "scale_factor": scale if calibrate else None,
        "uncertainty_honesty": (sum(1 for r in with_unc if r["within_stated_uncertainty"]) / len(with_unc)) if with_unc else None,
        "tolerances": tol,
    }
    return {"rows": rows, "summary": summary}


def format_markdown(report: Dict[str, Any]) -> str:
    s = report["summary"]
    lines = [f"# Accuracy: {s.get('room') or 'room'} ({s.get('method') or 'method not recorded'})", "",
             f"{s['passed']} of {s['elements_evaluated']} elements inside tolerance; {s['walls_missed']} wall(s) missed, "
             f"{s['walls_extra']} extra.", ""]
    if not s["scale_calibrated"]:
        lines += ["**Scale is not calibrated: the engine's dimensions are in model units and are compared here as if they were metres. "
                  "Re-run with --calibrate to separate scale error from shape error.**", ""]
    if s["uncertainty_honesty"] is not None:
        lines += [f"The truth fell inside the engine's own stated +/- for {s['uncertainty_honesty']:.0%} of the elements that stated one.", ""]
    lines += ["| Element | Truth | Engine | Error | Tolerance | Result | Stated +/- | Inside it |", "|---|---|---|---|---|---|---|---|"]

    def f(v, p=3):
        return "n/a" if v is None else f"{v:.{p}f}"

    for r in report["rows"]:
        res = "MISS" if r["engine"] is None else ("n/a" if r["pass"] is None else ("PASS" if r["pass"] else "FAIL"))
        tol = "n/a" if r["tolerance"] is None else (f"{r['tolerance']:.0%}" if r["element"] == "room area" else f"{r['tolerance']:.3f}")
        ins = "n/a" if r["within_stated_uncertainty"] is None else ("yes" if r["within_stated_uncertainty"] else "NO")
        lines.append(f"| {r['element']} | {f(r['truth'])} | {f(r['engine'])} | {f(r['error'])} | {tol} | {res} | "
                     f"{f(r['stated_uncertainty'])} | {ins} |")
    lines += ["", "Tolerances are PROPOSED (wall length, ceiling height, opening width: +/-2 cm; room area: +/-1 %) and await "
              "confirmation from the firms (H-0.1)."]
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    flags = {a for a in argv if a.startswith("--")}
    paths = [a for a in argv if not a.startswith("--")]
    if len(paths) != 2:
        print(__doc__)
        return 2
    truth = json.load(open(paths[0], encoding="utf-8"))
    world = json.load(open(paths[1], encoding="utf-8"))
    report = evaluate(truth, world, calibrate="--calibrate" in flags)
    print(json.dumps(report, indent=1, default=str) if "--json" in flags else format_markdown(report))
    return 0 if report["summary"]["failed"] == 0 and report["summary"]["walls_missed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
