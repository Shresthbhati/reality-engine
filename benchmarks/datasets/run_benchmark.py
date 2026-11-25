"""Benchmark runner: SceneManifest -> the PRODUCTION reconstruction entry -> per-scene + aggregate machine-readable JSON.

Production entry: ``engine.pipeline.vertical_slice.vertical_slice(evidence_items, VerticalSliceOptions())`` -- the function the API job's multi-view
rung calls (apps/api/jobs.py: ``vertical_slice_fn=lambda evidence, vs_opts: _vs.vertical_slice(...)``), fed ``EvidenceItem(id, PHOTO,
source_uri=path.resolve().as_uri(), sha256)`` exactly as the job builds them. NOT exercised: the progressive ladder's lower rungs, the API's
open-vocabulary detectors/dense options, evidence-quality gating, persistence. GT intrinsics are NOT given to the pipeline (no leakage).

Status per scene (never inferred, never defaulted):
  PASSED   the pipeline ran and returned a world with >= 1 registered camera AND at least one metric was computed against GT. This means
           "processed and measured", NOT "accurate": there are no accuracy thresholds here.
  BLOCKED  not scored; ``reason`` says why (invalid/missing data, too few images, pipeline unavailable, or processed but nothing scorable).
  FAILED   the pipeline raised or returned nothing usable (its honest refusals included).
Metrics a run cannot compute are reported ``{"status": "BLOCKED", "reason": ...}``, never as 0 or a default.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.metrics import geometry as G

from .manifest import ImageRecord, SceneManifest, load_scene_manifest, pose_c2w_opencv, validate_scene

PIPELINE_ENTRY = "engine.pipeline.vertical_slice.vertical_slice"
MIN_IMAGES = 3
MAX_POINTS = 50_000


class PipelineUnavailable(RuntimeError):
    pass


@dataclass
class PipelineHandle:
    entry: str
    production: bool
    run: Callable[[list[Any]], Any]      # evidence items -> VerticalSliceResult-like


def production_pipeline() -> PipelineHandle:
    """Resolve the real entry lazily (importing it pulls torch; ~20 s)."""
    try:
        from engine.pipeline.vertical_slice import VerticalSliceOptions, vertical_slice
    except Exception as exc:  # noqa: BLE001 - optional heavy deps may be absent
        raise PipelineUnavailable(f"{PIPELINE_ENTRY} cannot be imported: {type(exc).__name__}: {exc}") from exc
    return PipelineHandle(PIPELINE_ENTRY, True, lambda items: vertical_slice(items, VerticalSliceOptions()))


def _evidence_items(images: Sequence[ImageRecord], root: Path) -> list[Any]:
    from evidence.session import EvidenceItem, EvidenceKind
    return [EvidenceItem(id=i.id, kind=EvidenceKind.PHOTO, source_uri=(root / i.path).resolve().as_uri(), sha256=i.sha256) for i in images]


def _quat_wxyz_to_r(q: Sequence[float]) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _estimated_poses(result: Any) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for p in getattr(result, "camera_poses", ()) or ():
        eid, pos, rot = (p if isinstance(p, tuple) else (p.evidence_id, p.position, p.rotation))
        m = np.eye(4)
        m[:3, :3], m[:3, 3] = _quat_wxyz_to_r(rot), [float(v) for v in pos]     # camera-to-world, quaternion (w, x, y, z)
        out[str(eid)] = m
    return out


def _blocked(reason: str) -> dict[str, Any]:
    return {"status": "BLOCKED", "reason": reason}


def _load_gt_points(sm: SceneManifest, root: Path, n: int = 20_000) -> tuple[np.ndarray | None, dict[str, Any]]:
    """GT geometry as metre points: a measured point cloud if present, else a mesh sampled by area. Returns (points | None, info)."""
    cands = sorted([a for a in sm.annotations if a.kind in ("point_cloud", "mesh") and a.path],
                   key=lambda a: (a.kind != "point_cloud" or a.provenance_class != "measured", a.provenance_class != "measured", a.id))
    if not cands:
        return None, {"reason": "no ground-truth mesh or point cloud in the manifest"}
    a = cands[0]
    try:
        import trimesh
        g = trimesh.load(root / a.path, force="mesh") if a.kind == "mesh" else trimesh.load(root / a.path)
        if a.kind == "mesh":
            if len(g.faces) == 0:
                return None, {"reason": f"GT mesh {a.path} has no faces"}
            pts = G.sample_surface((np.asarray(g.vertices, float), np.asarray(g.faces, int)), n, seed=0)
        else:
            pts = np.asarray(g.vertices, float)
            pts = pts[:: max(1, len(pts) // n)]
    except Exception as exc:  # noqa: BLE001 - unreadable GT must block the metric, not crash the run
        return None, {"reason": f"GT geometry {a.path} could not be read: {type(exc).__name__}: {exc}"}
    return pts * sm.frame.unit_to_metre, {"annotation": a.id, "kind": a.kind, "provenance_class": a.provenance_class}


def _score(sm: SceneManifest, root: Path, result: Any, structure_extractor: Callable | None, dimension_extractor: Callable | None) -> dict[str, Any]:
    m: dict[str, Any] = {}
    est = _estimated_poses(result)
    gt = {i.id: p for i in sm.images if (p := pose_c2w_opencv(i, sm.frame)) is not None}
    pose_prov = sorted({i.pose_provenance for i in sm.images if i.pose_c2w is not None})
    if not gt:
        m["camera_registration"] = _blocked("no ground-truth camera poses in the manifest")
    else:
        reg = G.camera_registration(est, gt)
        reg["gt_pose_provenance"] = pose_prov
        reg["n_input_images"] = len(sm.images)
        if sm.frame.camera_axes == "unknown":
            for k in ("rotation_err_mean_deg", "rotation_err_max_deg"):
                reg.pop(k, None)
            reg["rotation"] = "NOT_REPORTED: GT camera-axes convention is unknown/UNVERIFIED for this dataset (centre error is axes-independent)"
        m["camera_registration"] = reg
    reg = m["camera_registration"]
    gt_pts, info = _load_gt_points(sm, root)
    pts = np.asarray(getattr(result, "points", ()) or (), float)
    if reg.get("status") != "OK":
        m["geometry"] = _blocked("needs a Sim(3) alignment from camera registration, which is not available: " + str(reg.get("reason", "no GT poses")))
    elif gt_pts is None:
        m["geometry"] = _blocked(info["reason"])
    elif len(pts) == 0:
        m["geometry"] = _blocked("the pipeline returned no points")
    else:
        pts = pts[:: max(1, len(pts) // MAX_POINTS)]
        a = reg["alignment"]
        aligned = a["s"] * pts @ np.asarray(a["R"]).T + np.asarray(a["t"])
        m["geometry"] = {"status": "OK", **G.chamfer(aligned, gt_pts), "fscore": G.fscore(aligned, gt_pts), "gt": info,
                         "n_pred_points": int(len(aligned)), "n_gt_points": int(len(gt_pts)),
                         "caveat": "points aligned with the camera-centre Sim(3); GT geometry assumed to share the GT pose frame (UNVERIFIED per dataset); "
                                   "sampled, not exact surface distance"}
    m["mesh_accuracy_completeness"] = _blocked("predicted-mesh extraction from the pipeline artifact store is not wired (point-cloud metrics above are used)")
    gt_struct = [e for a in sm.annotations_of("structure") for e in a.data.get("elements", [])]
    if structure_extractor is None:
        m["structure_completeness"] = _blocked("no extractor for WorldIR planar structure is wired")
        m["opening_completeness"] = _blocked("no extractor for WorldIR openings is wired")
    elif not gt_struct:
        m["structure_completeness"] = _blocked("no bounded ground-truth structural elements in the manifest")
        m["opening_completeness"] = _blocked("no ground-truth openings in the manifest")
    else:
        pred = structure_extractor(result)
        m["structure_completeness"] = {"status": "OK", **G.structural_completeness([e for e in pred if e["kind"] in ("wall", "floor", "ceiling")],
                                                                                   [e for e in gt_struct if e["kind"] in ("wall", "floor", "ceiling")])}
        m["opening_completeness"] = G.opening_completeness([e for e in pred if e["kind"] in ("door", "window")],
                                                           [e for e in gt_struct if e["kind"] in ("door", "window")])
    dims = [{"name": a.data.get("name", a.id), "value_m": a.data.get("value_m"), "uncertainty_m": a.uncertainty_m, "provenance_class": a.provenance_class}
            for a in sm.annotations_of("dimension")]
    if not dims:
        m["dimension_error"] = _blocked("no independent dimension measurements exist for this scene")
    elif dimension_extractor is None:
        m["dimension_error"] = _blocked("no extractor for named WorldIR dimensions is wired")
    else:
        m["dimension_error"] = G.dimension_errors(dimension_extractor(result), dims)
    return m


def run_scene(sm: SceneManifest, root: Path, pipeline: PipelineHandle | None, *, structure_extractor: Callable | None = None,
              dimension_extractor: Callable | None = None, pipeline_loader: Callable[[], PipelineHandle] = production_pipeline) -> dict[str, Any]:
    base: dict[str, Any] = {"scene_id": sm.scene_id, "dataset": sm.dataset, "group_id": sm.group_id, "split": sm.split,
                            "data_kinds": sorted(sm.data_kinds), "data_kind_key": sm.data_kind_key(), "n_images": len(sm.images),
                            "gt_provenance_classes": sorted({a.provenance_class for a in sm.annotations}), "notes": sm.notes, "processed": False}

    def blocked(reason: str, **kw: Any) -> dict[str, Any]:
        return {**base, "status": "BLOCKED", "reason": reason, **kw}
    problems = validate_scene(sm, root)
    if problems:
        return blocked("scene manifest/data invalid: " + "; ".join(problems[:5]) + (f" (+{len(problems) - 5} more)" if len(problems) > 5 else ""))
    if len(sm.images) < MIN_IMAGES:
        return blocked(f"{len(sm.images)} image(s); the pipeline needs >= {MIN_IMAGES}. " + " ".join(n for n in sm.notes if "needs" in n or "no " in n))
    try:
        handle = pipeline or pipeline_loader()
    except PipelineUnavailable as exc:
        return blocked(str(exc))
    base["pipeline"] = {"entry": handle.entry, "production": handle.production}
    try:
        result = handle.run(_evidence_items(sm.images, root))
    except Exception as exc:  # noqa: BLE001 - any pipeline failure is a FAILED scene, recorded with its traceback tail
        return {**base, "status": "FAILED", "reason": f"pipeline raised {type(exc).__name__}: {exc}", "traceback_tail": traceback.format_exc().splitlines()[-6:]}
    registered = len(getattr(result, "camera_poses", ()) or ())
    if result is None or getattr(result, "world", None) is None or registered < 1:
        return {**base, "status": "FAILED", "reason": f"pipeline returned no usable world (world={getattr(result, 'world', None) is not None}, "
                                                      f"registered cameras={registered})"}
    base.update(processed=True, cameras_registered=registered, cameras_input=len(sm.images),
                registration_status=getattr(result, "registration_status", None), scale_state=getattr(result, "scale_state", None))
    metrics = _score(sm, root, result, structure_extractor, dimension_extractor)
    base["metrics"] = metrics
    if not any(v.get("status") == "OK" for v in metrics.values()):
        return {**base, "status": "BLOCKED", "reason": "the pipeline processed this scene but no metric could be computed: " +
                "; ".join(f"{k}: {v.get('reason')}" for k, v in metrics.items())}
    return {**base, "status": "PASSED"}


def _numeric(d: dict[str, Any], prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in d.items():
        if isinstance(v, bool) or k in ("alignment", "gt", "gt_pose_provenance") or k.startswith("n_"):
            continue
        if isinstance(v, (int, float)) and math.isfinite(v):
            out[prefix + k] = float(v)
        elif isinstance(v, dict) and v.get("status", "OK") == "OK":
            out.update(_numeric(v, f"{prefix}{k}."))
    return out


def aggregate(results: Sequence[dict[str, Any]], *, production: bool) -> dict[str, Any]:
    """Per split, per data-kind combination, per GT provenance -- never one blended number. Only PASSED scenes contribute to means."""
    out: dict[str, Any] = {"production_pipeline": production, "pipeline_entry": PIPELINE_ENTRY, "n_scenes": len(results),
                           "status_counts": {s: sum(1 for r in results if r["status"] == s) for s in ("PASSED", "BLOCKED", "FAILED")},
                           "blocked": [{"scene_id": r["scene_id"], "reason": r["reason"]} for r in results if r["status"] != "PASSED"], "groups": {}}
    acc: dict[str, dict[str, list[float]]] = {}
    for r in results:
        if r["status"] != "PASSED":
            continue
        key = f"split={r['split']}|data_kinds={r['data_kind_key']}|gt_provenance={','.join(r['gt_provenance_classes'])}"
        for k, v in _numeric({mk: mv for mk, mv in r["metrics"].items() if mv.get("status") == "OK"}).items():
            acc.setdefault(key, {}).setdefault(k, []).append(v)
        acc.setdefault(key, {}).setdefault("_n_scenes", []).append(1.0)
    for key, ms in sorted(acc.items()):
        out["groups"][key] = {"n_scenes": int(sum(ms.pop("_n_scenes"))), "mean": {k: float(np.mean(v)) for k, v in sorted(ms.items())},
                              "n_per_metric": {k: len(v) for k, v in sorted(ms.items())}}
    if not production:
        out["warning"] = "NOT the production pipeline: these numbers validate the harness only and must not be quoted as reconstruction results"
    return out


def run_benchmark(scenes: Sequence[SceneManifest], root: str | Path, out_dir: str | Path, *, splits: Sequence[str] = ("development", "validation", "test"),
                  pipeline: PipelineHandle | None = None, **kw: Any) -> dict[str, Any]:
    root, out = Path(root), Path(out_dir)
    (out / "scenes").mkdir(parents=True, exist_ok=True)
    selected = [s for s in scenes if s.split in splits]
    results = []
    for sm in selected:
        r = run_scene(sm, root, pipeline, **kw)
        results.append(r)
        (out / "scenes" / f"{sm.scene_id.replace('/', '_')}.json").write_text(json.dumps(r, indent=2, sort_keys=True, default=str), encoding="utf-8")
    used = [r["pipeline"]["production"] for r in results if "pipeline" in r]
    agg = aggregate(results, production=bool(used) and all(used))
    skipped = [s.scene_id for s in scenes if s.split not in splits]
    agg["skipped_not_in_selected_splits"] = skipped
    (out / "aggregate.json").write_text(json.dumps(agg, indent=2, sort_keys=True), encoding="utf-8")
    return agg


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifests", required=True, help="directory of SceneManifest *.json files (splits already assigned)")
    ap.add_argument("--root", required=True, help="dataset root the manifest paths are relative to")
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", nargs="+", default=["test"])
    a = ap.parse_args(argv)
    scenes = [load_scene_manifest(p) for p in sorted(Path(a.manifests).glob("*.json"))]
    agg = run_benchmark(scenes, a.root, a.out, splits=a.splits)
    print(json.dumps(agg["status_counts"]))
    return 0 if agg["status_counts"]["FAILED"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
