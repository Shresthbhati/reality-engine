#!/usr/bin/env python3
"""Room-fidelity scorecard: compile a fixture through the REAL canonical pipeline and compare the WorldIR with ground truth.

    python scripts/room_fidelity_report.py                                  # text table (synthetic fixture)
    python scripts/room_fidelity_report.py --json out.json                  # also write the full machine-readable report
    python scripts/room_fidelity_report.py --write-baseline docs/reconstruction/room_fidelity_baseline.json
    python scripts/room_fidelity_report.py --baseline docs/reconstruction/room_fidelity_baseline.json --fail-on-regression

Pipeline under test: ``synthetic.indoor.fidelity_room`` -> RGB-D ``observe`` (per-point colour) ->
``engine.compiler.compile_reconstruction_to_world`` (the same entry the vertical slice uses; production CompileOptions
unless ``--promote-building``). Headless, deterministic, no COLMAP, ~10 s.

Honesty rules (Decision 023 and the plan, section 0): a metric that cannot be computed prints ``not available`` with the
reason and is never replaced by a number. Errors in cm are only meaningful when the world scale is metric; the synthetic
fixture is metric by construction, but the compiler records no scale state for it, so that is reported as not available.

The real-room fixture needs the user's photos and tape measurements (see datasets/README.md): ``--fixture real`` reports
every metric as not available until ``datasets/real_room/ground_truth.json`` and a reconstruction exist.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SCHEMA = "room_fidelity_report/1"
UP = (0.0, 0.0, 1.0)
NA = None  # a metric value of None == "not available"

#: direction: "lower" = smaller is better, "higher" = larger is better. Used only by --fail-on-regression.
LOWER, HIGHER = "lower", "higher"


# ------------------------------------------------------------------------------------------------ colour (CIE76)

def _srgb_to_lab(rgb) -> np.ndarray:
    c = np.asarray(rgb, dtype=float) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    m = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = (m @ lin) / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def delta_e76(a, b) -> float:
    """CIE76 colour difference between two sRGB (0-255) triples."""
    return float(np.linalg.norm(_srgb_to_lab(a) - _srgb_to_lab(b)))


# ------------------------------------------------------------------------------------------------ build + compile

def build_and_compile(colors: str = "albedo", voxel_m: float = 0.08, promote_building: bool = False,
                      cameras: str = "all"):
    """Fixture -> observation -> real compile. Returns ``(truth, observation, world, diagnostics)``.

    ``colors``: ``albedo`` | ``rendered`` | ``none`` (no per-point colour). ``cameras``: ``all`` (3 standing positions) or
    ``one`` (a single standing position: partial wall coverage)."""
    from engine.compiler import CompileOptions, compile_reconstruction_to_world
    from synthetic.indoor import fidelity_ground_truth, fidelity_room, observe

    import time as _time

    _t0 = _time.perf_counter()
    ind = fidelity_room()
    truth = fidelity_ground_truth(ind)
    cams = ind.cameras if cameras == "all" else ind.cameras[:24]
    _t1 = _time.perf_counter()
    obs = observe(ind, colors=None if colors == "none" else colors, voxel_m=voxel_m, cameras=cams)
    _t2 = _time.perf_counter()
    options = CompileOptions(seed=42, up=UP, promote_building=promote_building)
    world, diag = compile_reconstruction_to_world(obs.result, options)
    _t3 = _time.perf_counter()
    # F10: per-stage end-to-end timings live beside the metrics (not compared by the gate: they vary by machine).
    world.metadata["timings"] = {
        "sensor_fixture_s": round(_t1 - _t0, 3), "observe_s": round(_t2 - _t1, 3),
        "compile_s": round(_t3 - _t2, 3), "total_s": round(_t3 - _t0, 3),
    }
    return truth, obs, world, diag


# ------------------------------------------------------------------------------------------------ world accessors

def _of(world, kind: str) -> list:
    return [e for e in world.entities.values() if e.type.value == kind]


def _bounds(world, entity):
    """(min, max) numpy arrays from the entity's geometry, else from bounds_min/bounds_max custom properties."""
    for gid in entity.geometry_ids:
        g = world.geometries.get(gid)
        if g is not None and g.bounds_min is not None and g.bounds_max is not None:
            return (np.array([g.bounds_min.x, g.bounds_min.y, g.bounds_min.z]),
                    np.array([g.bounds_max.x, g.bounds_max.y, g.bounds_max.z]))
    cp = entity.custom_properties
    if cp.get("bounds_min") is not None and cp.get("bounds_max") is not None:
        return np.array(cp["bounds_min"], float), np.array(cp["bounds_max"], float)
    return None


def _na(reason: str) -> dict:
    return {"value": NA, "status": "not available", "reason": reason}


def _m(value: float, direction: str, unit: str = "", **extra) -> dict:
    return {"value": float(value), "status": "ok", "direction": direction, "unit": unit, **extra}


# ------------------------------------------------------------------------------------------------ metrics

def primary_walls(world) -> list:
    """The walls that represent the room, not the plane evidence behind them.

    Since plan S3 a solved room carries solid slabs (entities labelled ``wall_slab``) in addition to the reconstructed
    plane entities, which are kept as evidence. Counting both would double the wall count; count ONLY the entities
    labelled ``wall_slab`` when the pipeline produced them and fall back to plane walls for a world without a solved
    layout."""
    walls = _of(world, "wall")
    slabs = [w for w in walls if "wall_slab" in (w.semantic_labels or [])]
    return slabs or walls


def wall_metrics(world, truth) -> Dict[str, dict]:
    gt_walls = truth["walls"]
    walls = primary_walls(world)
    out = {"wall_count": _m(len(walls), HIGHER, "walls", ground_truth=len(gt_walls)),
           "wall_count_error": _m(abs(len(walls) - len(gt_walls)), LOWER, "walls")}
    boxes = [b for b in (_bounds(world, w) for w in walls) if b is not None]
    if not boxes:
        out["wall_perimeter_coverage"] = _na("no wall entity carries geometry bounds")
        return out
    tol, covered, total = 0.15, 0.0, 0.0
    for gw in gt_walls:
        a, b = np.array(gw["start"], float), np.array(gw["end"], float)
        n = max(2, int(round(gw["length_m"] / 0.05)))
        pts = a + (b - a) * (np.arange(n) + 0.5)[:, None] / n
        hit = np.zeros(n, bool)
        for lo, hi in boxes:
            hit |= ((pts[:, 0] >= lo[0] - tol) & (pts[:, 0] <= hi[0] + tol) &
                    (pts[:, 1] >= lo[1] - tol) & (pts[:, 1] <= hi[1] + tol))
        covered += gw["length_m"] * hit.mean()
        total += gw["length_m"]
    out["wall_perimeter_coverage"] = _m(covered / total, HIGHER, "fraction",
                                        basis=f"a wall entity's plan bounds (+{tol} m) overlaps the GT perimeter")
    return out


def room_metrics(world, truth) -> Dict[str, dict]:
    rooms = _of(world, "room")
    gt = truth["room"]
    out: Dict[str, dict] = {}
    if not rooms:
        reason = "no room entity was produced"
        for k in ("room_length_error_pct", "room_width_error_pct", "room_height_error_pct", "footprint_area_error_pct"):
            out[k] = _na(reason)
        return out
    cp = rooms[0].custom_properties
    u, v = cp.get("floor_dimension_u_m"), cp.get("floor_dimension_v_m")
    if u is None or v is None:
        out["room_length_error_pct"] = out["room_width_error_pct"] = _na("room carries no floor_dimension_u/v_m")
    else:
        length, width = max(u, v), min(u, v)  # a room's two plan axes; GT length is the longer one
        for name, meas, ref in (("length", length, gt["length_m"]), ("width", width, gt["width_m"])):
            out[f"room_{name}_error_pct"] = _m(abs(meas - ref) / ref * 100, LOWER, "%", measured_m=meas,
                                               ground_truth_m=ref, error_cm=abs(meas - ref) * 100)
    h = cp.get("height_m")
    if h is None:
        out["room_height_error_pct"] = _na("room carries no height_m")
    else:
        out["room_height_error_pct"] = _m(abs(h - gt["height_m"]) / gt["height_m"] * 100, LOWER, "%", measured_m=h,
                                          ground_truth_m=gt["height_m"], error_cm=abs(h - gt["height_m"]) * 100)
    area = cp.get("floor_area_m2")
    if area is None:
        out["footprint_area_error_pct"] = _na("room carries no floor_area_m2")
    else:
        out["footprint_area_error_pct"] = _m(abs(area - gt["area_m2"]) / gt["area_m2"] * 100, LOWER, "%",
                                             measured_m2=area, ground_truth_m2=gt["area_m2"],
                                             bbox_floor_area_m2=cp.get("bbox_floor_area_m2"))
    # D3: dimensions from the regularised layout (not the axis-aligned box), plus the consistency verdict
    cons = cp.get("dimensional_consistency") or {}
    if cons.get("length_m") is None:
        out["dimensional_consistency"] = _na("no solved layout")
        for k in ("layout_length_error_pct", "layout_width_error_pct"):
            out[k] = _na("no solved layout")
    else:
        for name, key in (("length", "length_m"), ("width", "width_m")):
            meas, ref = cons[key], gt[f"{name}_m"]
            out[f"layout_{name}_error_pct"] = _m(abs(meas - ref) / ref * 100, LOWER, "%", measured_m=meas,
                                                 ground_truth_m=ref, error_cm=abs(meas - ref) * 100)
        mismatches = cons.get("mismatches") or []
        out["dimensional_consistency"] = _m(1.0 if cons.get("ok") else 0.0, HIGHER, "bool",
                                            mismatches=len(mismatches), height_layers_m=cons.get("height_layers_m"))
    return out


def _match_by_distance(dets: List[np.ndarray], refs: List[np.ndarray], max_dist: float):
    """Greedy nearest-first one-to-one matching. Returns list of (det_idx, ref_idx, dist)."""
    pairs = sorted(((float(np.linalg.norm(d - r)), i, j) for i, d in enumerate(dets) for j, r in enumerate(refs)))
    used_d, used_r, out = set(), set(), []
    for dist, i, j in pairs:
        if dist <= max_dist and i not in used_d and j not in used_r:
            used_d.add(i), used_r.add(j)
            out.append((i, j, dist))
    return out


def opening_metrics(world, truth) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for kind in ("door", "window"):
        gt = [o for o in truth["openings"] if o["kind"] == kind]
        dets = _of(world, kind)
        centres = []
        for e in dets:
            c = e.custom_properties.get("position")
            centres.append(np.array(c, float) if c is not None else None)
        if any(c is None for c in centres):
            for k in ("recall", "precision", "width_error_cm"):
                out[f"{kind}_{k}"] = _na(f"a detected {kind} carries no position")
            continue
        matches = _match_by_distance(centres, [np.array(o["center"], float) for o in gt], 0.6)
        out[f"{kind}_recall"] = _m(len(matches) / len(gt) if gt else 1.0, HIGHER, "fraction", detected=len(dets),
                                   ground_truth=len(gt))
        out[f"{kind}_precision"] = _m(len(matches) / len(dets) if dets else 1.0 if not gt else 0.0, HIGHER, "fraction")
        if matches:
            werr = [abs(dets[i].custom_properties.get("width_m", math.nan) - gt[j]["width_m"]) for i, j, _ in matches]
            herr = [abs(dets[i].custom_properties.get("height_m", math.nan) - gt[j]["height_m"]) for i, j, _ in matches]
            if np.isfinite(werr).all():
                out[f"{kind}_width_error_cm"] = _m(max(werr) * 100, LOWER, "cm")
            else:
                out[f"{kind}_width_error_cm"] = _na(f"matched {kind} carries no width_m")
            if np.isfinite(herr).all():
                out[f"{kind}_height_error_cm"] = _m(max(herr) * 100, LOWER, "cm")
            else:
                out[f"{kind}_height_error_cm"] = _na(f"matched {kind} carries no height_m")
        else:
            out[f"{kind}_width_error_cm"] = _na(f"no {kind} was matched to ground truth")
    return out


def furniture_metrics(world, truth) -> Dict[str, dict]:
    gt = truth["furniture"]
    dets = [e for e in world.entities.values() if "furniture_candidate" in e.semantic_labels]
    boxes = [_bounds(world, e) for e in dets]
    if any(b is None for b in boxes):
        return {"furniture_recall": _na("a furniture candidate carries no bounds")}
    centres = [(lo + hi) / 2 for lo, hi in boxes]
    refs = [(np.array(f["min"]) + np.array(f["max"])) / 2 for f in gt]
    matches = _match_by_distance(centres, refs, 0.5)
    out = {"furniture_recall": _m(len(matches) / len(gt), HIGHER, "fraction", detected=len(dets), ground_truth=len(gt)),
           "furniture_precision": _m(len(matches) / len(dets) if dets else 0.0, HIGHER, "fraction")}
    if matches:
        errs = [np.abs((boxes[i][1] - boxes[i][0]) - np.array(gt[j]["size_m"])) for i, j, _ in matches]
        out["furniture_size_error_cm_max"] = _m(float(np.max(errs)) * 100, LOWER, "cm",
                                                per_item={gt[j]["ref"]: (np.round((boxes[i][1] - boxes[i][0]) * 100, 1)
                                                                          ).tolist() for i, j, _ in matches})
    return out


def colour_metrics(world, truth, furniture_pairs: Optional[list] = None) -> Dict[str, dict]:
    out: Dict[str, dict] = {}

    def entry(name: str, ent, gt_rgb):
        got = None if ent is None else ent.custom_properties.get("mean_color_rgb")
        if got is None:
            out[f"deltaE76_{name}"] = _na("no mean_color_rgb on the matching entity" if ent is not None
                                          else "no matching entity")
        else:
            out[f"deltaE76_{name}"] = _m(delta_e76(got, gt_rgb), LOWER, "dE76", measured=list(got),
                                         ground_truth=list(gt_rgb))

    for kind in ("floor", "ceiling"):
        ents = _of(world, kind)
        entry(kind, ents[0] if ents else None, truth[kind]["color_rgb"])
    walls = [(w, _bounds(world, w)) for w in _of(world, "wall")]
    for gw in truth["walls"]:
        a, b = np.array(gw["start"]), np.array(gw["end"])
        mid = (a + b) / 2
        best = None
        for w, bb in walls:
            if bb is None:
                continue
            lo, hi = bb
            d = float(np.linalg.norm(np.maximum(np.maximum(lo[:2] - mid, mid - hi[:2]), 0)))
            if best is None or d < best[0]:
                best = (d, w)
        entry(f"wall_{gw['side']}", best[1] if best and best[0] < 0.3 else None, gw["color_rgb"])
    # surface colour present on the room itself
    rooms = _of(world, "room")
    for key in ("floor_color_rgb", "wall_color_rgb"):
        present = bool(rooms and rooms[0].custom_properties.get(key))
        out[f"room_{key}_present"] = _m(1.0 if present else 0.0, HIGHER, "bool")
    return out


def dimension_state_metric(world) -> Dict[str, dict]:
    """Share of dimension-like properties (numeric ``*_m`` / ``*_m2``) that carry a measurement state.

    A state is recognised as ``<key>_state``, ``<key>_measurement_state`` or an entry in ``dimension_states`` on the same
    entity, or a non-None ``Entity.statement_state``. The D1 contract (plan task D1) will fix the real shape."""
    total = stated = 0
    for e in world.entities.values():
        cp = e.custom_properties
        states = cp.get("dimension_states") or {}
        dims = cp.get("dimensions") if isinstance(cp.get("dimensions"), dict) else {}
        for k, v in cp.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool) and k.endswith(("_m", "_m2")):
                total += 1
                if (isinstance(dims.get(k), dict) and dims[k].get("state")) or k in states \
                        or f"{k}_state" in cp or f"{k}_measurement_state" in cp:
                    stated += 1
    if not total:
        return {"dimensions_with_state_pct": _na("no dimension-like properties in the world")}
    return {"dimensions_with_state_pct": _m(stated / total * 100, HIGHER, "%", dimensions=total, with_state=stated)}


def scale_metric(world) -> Dict[str, dict]:
    st = (world.metadata.get("scale") or {}).get("state")
    if st is None:
        return {"scale_state": {"value": NA, "status": "not available",
                                "reason": "compile_reconstruction_to_world records no scale; engine/pipeline/vertical_slice "
                                          "sets world.metadata['scale']. Treat as RELATIVE (Decision 023)"}}
    return {"scale_state": {"value": st, "status": "ok", "direction": None}}


def room_count_metrics(world, truth) -> Dict[str, dict]:
    rooms = _of(world, "room")
    ids = [r.id for r in rooms]
    return {"room_count_error": _m(abs(len(rooms) - 1), LOWER, "rooms", detected=len(rooms), ground_truth=1),
            "duplicate_room_ids": _m(len(ids) - len(set(ids)), LOWER, "ids"),
            "duplicate_room_entities": _m(_duplicate_room_pairs(rooms), LOWER, "pairs")}


def _duplicate_room_pairs(rooms) -> int:
    """Room pairs whose CONTAINS targets overlap (Jaccard >= 0.5), i.e. two entities for one physical room."""
    def parts(r):
        return {x.target_id for x in r.relationships if x.kind.value == "contains"}
    n = 0
    for a, b in combinations(rooms, 2):
        pa, pb = parts(a), parts(b)
        if pa and pb and len(pa & pb) / len(pa | pb) >= 0.5:
            n += 1
    return n


# ------------------------------------------------------------------------------------------------ stage trace (T0.3)

def stage_trace(obs, world, diag) -> Dict[str, Any]:
    pts = obs.result.points
    rooms = _of(world, "room")
    rcp = rooms[0].custom_properties if rooms else {}
    furn = [e for e in world.entities.values() if "furniture_candidate" in e.semantic_labels]
    return {
        "points_total": len(pts),
        "points_with_colour": sum(1 for p in pts if p.color is not None),
        "points_in_planes": diag.points_in_planes,
        "planes_total": diag.planes_total,
        "planes_by_role": dict(diag.planes_by_role),
        "planes_unpromoted": len(diag.planes_unpromoted),
        "walls_promoted": len(_of(world, "wall")),
        "floors_promoted": len(_of(world, "floor")),
        "ceilings_promoted": len(_of(world, "ceiling")),
        "rooms": len(rooms),
        "duplicate_room_pairs": _duplicate_room_pairs(rooms),
        "footprint_polygon_present": bool(rcp.get("footprint_polygon_m")),
        "footprint_vertices": len(rcp.get("footprint_polygon_m") or []),
        "doors": len(_of(world, "door")),
        "windows": len(_of(world, "window")),
        "openings_generic": len(_of(world, "opening")),
        "furniture_candidates": len(furn),
        "entities_with_mean_color_rgb": sum(1 for e in world.entities.values() if e.custom_properties.get("mean_color_rgb")),
        "room_floor_color_rgb_present": bool(rcp.get("floor_color_rgb")),
        "room_wall_color_rgb_present": bool(rcp.get("wall_color_rgb")),
        "ceiling_layers": rcp.get("ceiling_layers"),
        "scale_state": (world.metadata.get("scale") or {}).get("state", "not recorded by compile"),
        "interior_warnings": list(diag.interior_warnings),
        "validation_issues": list(diag.validation_issues),
    }


# ------------------------------------------------------------------------------------------------ report + gate

def build_report(colors: str = "albedo", promote_building: bool = False, voxel_m: float = 0.08,
                 cameras: str = "all") -> dict:
    truth, obs, world, diag = build_and_compile(colors, voxel_m, promote_building, cameras)
    metrics: Dict[str, dict] = {}
    for part in (room_count_metrics(world, truth), wall_metrics(world, truth), room_metrics(world, truth),
                 opening_metrics(world, truth), furniture_metrics(world, truth), colour_metrics(world, truth),
                 scale_metric(world), dimension_state_metric(world)):
        metrics.update(part)
    return {"schema": SCHEMA, "fixture": truth["fixture"], "source": truth["source"],
            "config": {"colors": colors, "promote_building": promote_building, "voxel_m": voxel_m, "seed": 42,
                       "cameras": cameras},
            "metrics": metrics, "stage_trace": stage_trace(obs, world, diag),
            "timings": dict(world.metadata.get("timings") or {})}


def real_room_report() -> dict:
    gt = ROOT / "datasets" / "real_room" / "ground_truth.json"
    reason = ("no real-room fixture: it needs the user's photo set plus tape-measured L/W/H, door/window sizes and one "
              "camera-to-camera baseline (datasets/README.md). Not fabricated.")
    if gt.exists():
        reason = f"{gt} exists but no reconstruction runner is wired for it yet (needs COLMAP; use scripts/run_lroom_e2e.py style)"
    return {"schema": SCHEMA, "fixture": "real_room", "source": "real", "config": {},
            "metrics": {k: _na(reason) for k in ("wall_perimeter_coverage", "room_length_error_pct", "door_recall")},
            "stage_trace": {}}


def text_table(rep: dict) -> str:
    lines = [f"room fidelity report  fixture={rep['fixture']}  config={rep['config']}", "-" * 78,
             f"{'metric':<36}{'value':>14}  detail", "-" * 78]
    for name, m in rep["metrics"].items():
        if m["status"] != "ok":
            lines.append(f"{name:<36}{'not available':>14}  {m['reason']}")
            continue
        v = m["value"]
        shown = f"{v:.3f}" if isinstance(v, float) else str(v)
        detail = m.get("unit", "")
        for k in ("measured_m", "ground_truth_m", "error_cm", "detected", "ground_truth", "measured_m2", "measured"):
            if k in m:
                detail += f"  {k}={round(m[k], 3) if isinstance(m[k], float) else m[k]}"
        lines.append(f"{name:<36}{shown:>14}  {detail}")
    lines.append("-" * 78)
    lines.append("stage trace: " + json.dumps({k: v for k, v in rep["stage_trace"].items()
                                                if k not in ("interior_warnings", "validation_issues")}, default=str))
    for w in rep["stage_trace"].get("interior_warnings", []):
        lines.append(f"interior warning: {w}")
    return "\n".join(lines)


SWEEP_KEYS = ("walls_promoted", "floors_promoted", "doors", "windows", "furniture_candidates", "rooms")
SWEEP_METRICS = ("door_recall", "window_recall", "furniture_recall", "footprint_area_error_pct",
                 "room_length_error_pct", "room_height_error_pct", "room_wall_color_rgb_present")


def sweep_table(voxels: List[float], colors: str, promote_building: bool, cameras: str) -> str:
    """Key metrics per point density: shows how much the result depends on the cloud's voxel size."""
    heads = ["voxel_m", "points"] + list(SWEEP_KEYS) + list(SWEEP_METRICS)
    rows = [heads]
    for v in voxels:
        r = build_report(colors, promote_building, v, cameras)
        t, m = r["stage_trace"], r["metrics"]
        rows.append([f"{v:g}", str(t["points_total"])] + [str(t[k]) for k in SWEEP_KEYS] +
                    [f"{m[k]['value']:.3g}" if m[k]["status"] == "ok" else "n/a" for k in SWEEP_METRICS])
    w = [max(len(r[i]) for r in rows) for i in range(len(heads))]
    return "\n".join("  ".join(c.rjust(w[i]) for i, c in enumerate(r)) for r in rows)


def compare_to_baseline(rep: dict, base: dict, rel_tol: float = 0.02, abs_tol: float = 1e-3) -> List[str]:
    """Regressions of ``rep`` against ``base``: a metric that got worse beyond tolerance, or became not available."""
    problems = []
    for name, b in base["metrics"].items():
        if b["status"] != "ok" or b.get("direction") not in (LOWER, HIGHER) or not isinstance(b["value"], (int, float)):
            continue
        cur = rep["metrics"].get(name)
        if cur is None or cur["status"] != "ok":
            problems.append(f"{name}: was {b['value']:.4g}, now not available")
            continue
        slack = max(abs_tol, rel_tol * abs(b["value"]))
        worse = cur["value"] > b["value"] + slack if b["direction"] == LOWER else cur["value"] < b["value"] - slack
        if worse:
            problems.append(f"{name}: baseline {b['value']:.4g} -> {cur['value']:.4g} ({b['direction']} is better)")
    return problems


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fixture", choices=("synthetic", "real"), default="synthetic")
    ap.add_argument("--colors", choices=("albedo", "rendered", "none"), default="albedo",
                    help="point colour source: albedo = surface base colour, rendered = shaded pixel, none = uncoloured")
    ap.add_argument("--cameras", choices=("all", "one"), default="all",
                    help="all = 3 standing positions, one = a single position (partial coverage)")
    ap.add_argument("--promote-building", action="store_true", help="compile with promote_building=True (as the interior tests do)")
    ap.add_argument("--voxel", type=float, default=0.08,
                    help="point-cloud voxel size in m. The result DEPENDS on it (see ROOM_FIDELITY_BASELINE.md sweep)")
    ap.add_argument("--sweep", help="comma-separated voxel sizes: print key metrics per density and exit (no gate)")
    ap.add_argument("--json", help="write the full report here")
    ap.add_argument("--baseline", help="committed baseline JSON to compare against")
    ap.add_argument("--write-baseline", help="write this run as the baseline JSON")
    ap.add_argument("--fail-on-regression", action="store_true", help="exit 1 if any metric is worse than --baseline")
    ap.add_argument("--rel-tol", type=float, default=0.02)
    a = ap.parse_args(argv)

    if a.sweep:
        print(sweep_table([float(v) for v in a.sweep.split(",")], a.colors, a.promote_building, a.cameras))
        return 0
    rep = real_room_report() if a.fixture == "real" else build_report(a.colors, a.promote_building, a.voxel, a.cameras)
    print(text_table(rep))
    for path, label in ((a.json, "report"), (a.write_baseline, "baseline")):
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(json.dumps(rep, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
            print(f"wrote {label}: {path}")
    if a.fail_on_regression:
        if not a.baseline:
            print("--fail-on-regression needs --baseline", file=sys.stderr)
            return 2
        base = json.loads(Path(a.baseline).read_text(encoding="utf-8"))
        if base.get("config") != rep.get("config"):
            print(f"baseline config {base.get('config')} != run config {rep.get('config')}", file=sys.stderr)
            return 2
        problems = compare_to_baseline(rep, base, a.rel_tol)
        for p in problems:
            print("REGRESSION", p)
        print("regression gate:", "FAIL" if problems else "pass")
        return 1 if problems else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
