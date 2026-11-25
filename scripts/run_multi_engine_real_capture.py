#!/usr/bin/env python3
"""Real-image multi-engine run: per-engine measurements, world lineage, reload proof.

What this script does, in order, on a REAL photograph capture only:

  Stage 1  (Day 4 geometry)
      Builds a reproducible pair set (sequential + skip-2: overlap falls with stride,
      no ground truth is used to choose pairs), then runs the matching engines
      (SIFT, LightGlue-ALIKED) through the MultiEngineOrchestrator with the camera
      intrinsics. Every pair is geometrically verified (fundamental matrix -> RANSAC
      inliers -> Sampson residuals -> essential matrix -> relative pose). The camera
      connectivity graph, its components, weak cameras and triangle cycle
      consistency are derived from the VERIFIED pairs. The reference COLMAP model's
      co-visibility is used afterwards only to score the pair graph.

  Stage 2  (production path)
      Runs the real evidence-world job (``apps.api.evidence_world_stage.run_for_job``)
      on the same images. That path runs the perception flow, then the multi-engine
      pass over EVERY registered engine, imports the engine evidence into the run's
      canonical Evidence Matrix, and persists the WorldIR version with the evidence
      lineage in its metadata.

  Stage 3  (fresh-process reload)
      A NEW python process opens the WorldStore, verifies the version, loads it, and
      compares ``world.metadata["multi_engine_evidence"]`` with the lineage file the
      run wrote. The parent process compares that child's answer with its own.

Everything is measured or reported as unavailable. Nothing is substituted for an
engine that failed, and no synthetic dataset is accepted as real evidence.

Usage:
  python scripts/run_multi_engine_real_capture.py [--out DIR] [--max-images N]
                                                 [--pairs sequential_skip2|sequential]
                                                 [--min-inliers 15] [--allow-synthetic]
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

DEFAULT_DATASET = REPO / "datasets" / "south_building"

#: Two images are "connected" in the camera graph when their pair verified with at least
#: this many RANSAC inliers. 15 inliers is a weak but real two-view geometry (the minimum
#: for a stable 7-point fundamental matrix is 7); below it the pair is measured but not
#: counted as registration evidence.
DEFAULT_MIN_INLIERS = 15


# --------------------------------------------------------------------------- dataset
def _dataset_manifest(dataset: Path) -> Dict[str, Any]:
    for name in ("MANIFEST.json", "manifest.json"):
        p = dataset / name
        if p.is_file():
            return json.loads(p.read_text(encoding="utf-8"))
    return {}


def _is_synthetic(manifest: Dict[str, Any]) -> bool:
    text = " ".join(str(manifest.get(k, "")) for k in ("dataset", "description", "identity")).lower()
    return "synthetic" in text


def _camera_model(parts: List[str]) -> Optional[Dict[str, float]]:
    """One cameras.txt row as {fx, fy, cx, cy, width, height, model} (or None if unsupported)."""
    if len(parts) < 5:
        return None
    model, width, height = parts[1], int(parts[2]), int(parts[3])
    params = [float(v) for v in parts[4:]]
    row: Dict[str, float] = {"width": width, "height": height}
    if model in ("PINHOLE", "OPENCV") and len(params) >= 4:
        row.update(fx=params[0], fy=params[1], cx=params[2], cy=params[3])
        if model == "OPENCV" and len(params) >= 8:
            row["dist"] = [params[4], params[5], params[6], params[7]]  # type: ignore[assignment]
    elif model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL") and len(params) >= 3:
        row.update(fx=params[0], fy=params[0], cx=params[1], cy=params[2])
        if model == "SIMPLE_RADIAL" and len(params) >= 4:
            row["dist"] = [params[3], 0.0, 0.0, 0.0]  # type: ignore[assignment]
        elif model == "RADIAL" and len(params) >= 5:
            row["dist"] = [params[3], params[4], 0.0, 0.0]  # type: ignore[assignment]
    else:
        return None
    row["model"] = model  # type: ignore[assignment]
    return row


def _image_size(path: str) -> Optional[Tuple[int, int]]:
    try:
        from PIL import Image

        with Image.open(path) as im:
            return im.size
    except Exception:
        return None


def _colmap_intrinsics(dataset: Path, images: Sequence[str] = ()) -> Tuple[Dict[str, Dict[str, float]],
                                                                          Dict[str, List[int]], Dict[str, Any]]:
    """Camera intrinsics and reference co-visibility, with explicit provenance.

    Two cases, and they are NOT the same evidence:

    * the reference model describes the very images being measured -> its per-image
      camera and point co-visibility are used directly;
    * the model describes a different image set (this repository's south_building
      subset is disjoint from the shipped sparse model) -> intrinsics are taken from
      the single calibrated camera and RESCALED to each image's actual size, and
      co-visibility is reported unavailable. Rescaling a calibrated pinhole camera by
      the image-size ratio is exact for a pure resize; it is not a new measurement.
    """
    cam_file = dataset / "sparse" / "cameras.txt"
    img_file = dataset / "sparse" / "images.txt"
    cams: Dict[int, Dict[str, float]] = {}
    intrinsics: Dict[str, Dict[str, float]] = {}
    covisibility: Dict[str, List[int]] = {}
    provenance: Dict[str, Any] = {"source": "dataset sparse model", "overlap_with_reference_model": 0}
    if cam_file.is_file():
        for line in cam_file.read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.split()
            if not parts[0].isdigit():
                continue
            row = _camera_model(parts)
            if row:
                cams[int(parts[0])] = row

    model_names: List[str] = []
    if img_file.is_file():
        lines = img_file.read_text(encoding="utf-8").splitlines()
        for i in range(0, len(lines) - 1, 2):
            parts = lines[i].split()
            if len(parts) < 10 or not parts[0].isdigit():
                continue
            cam_id, name = int(parts[8]), parts[9]
            model_names.append(name)
            if cam_id in cams:
                intrinsics[name] = dict(cams[cam_id])
            observed = lines[i + 1].split()
            ids: List[int] = []
            for j in range(2, len(observed), 3):
                try:
                    pid = int(observed[j])
                except (IndexError, ValueError):
                    continue
                if pid > 0:
                    ids.append(pid)
            covisibility[name] = ids

    wanted = {Path(p).name for p in images}
    overlap = len(wanted & set(model_names))
    provenance["overlap_with_reference_model"] = overlap
    provenance["reference_model_images"] = len(model_names)
    if overlap:
        # A name match is NOT enough: the reference model calibrates the ORIGINAL
        # frames (3072x2304 here) while the committed images are the resized
        # derivatives (1024x768). Using the model's pixels unchanged put the
        # principal point at (1536, 1152) -- outside a 1024x768 image -- which made
        # every essential matrix (and therefore every recovered rotation) wrong
        # while the pixel-space epipolar residual stayed small. Rescaling by the
        # image-size ratio is exact for a pure resize.
        rescaled: Dict[str, Dict[str, float]] = {}
        n_rescaled = 0
        model_sizes: List[str] = []
        no_size: List[str] = []
        for path in images:
            name = Path(path).name
            cam = intrinsics.get(name)
            if not cam:
                continue
            size = _image_size(path)
            if not size:
                no_size.append(name)
                rescaled[path] = cam
                continue
            w, h = size
            if (w, h) != (int(cam["width"]), int(cam["height"])):
                model_sizes.append(f"{int(cam['width'])}x{int(cam['height'])}")
                sx, sy = w / float(cam["width"]), h / float(cam["height"])
                # distortion coefficients are in normalised units: unchanged by a resize
                dist = cam.get("dist")
                cam = {"fx": cam["fx"] * sx, "fy": cam["fy"] * sy,
                       "cx": cam["cx"] * sx, "cy": cam["cy"] * sy,
                       "width": w, "height": h, "model": cam["model"]}
                if dist is not None:
                    cam["dist"] = dist
                n_rescaled += 1
            rescaled[path] = cam
        sizes_note = (f"; rescaled from the model's {sorted(set(model_sizes))[0]} to each image's "
                      f"actual size ({n_rescaled} images, pure-resize exactness; not a new measurement)"
                      if n_rescaled else "")
        provenance["intrinsics"] = ("reference model per-image camera (names matched)" + sizes_note)
        provenance["images_rescaled_to_actual_size"] = n_rescaled
        provenance["images_without_readable_size"] = no_size
        provenance["covisibility"] = "reference model"
        return rescaled, covisibility, provenance

    # Disjoint reference model: rescale the calibrated camera to each image's size.
    rescale: Dict[str, Dict[str, float]] = {}
    notes: List[str] = []
    if len(cams) == 1:
        cam = next(iter(cams.values()))
        for path in images:
            size = _image_size(path)
            if not size:
                continue
            w, h = size
            sx, sy = w / float(cam["width"]), h / float(cam["height"])
            rescale[path] = {"fx": cam["fx"] * sx, "fy": cam["fy"] * sy,
                             "cx": cam["cx"] * sx, "cy": cam["cy"] * sy,
                             "width": w, "height": h}
        notes.append(f"calibrated camera model {cam['model']} at {int(cam['width'])}x{int(cam['height'])} "
                     f"rescaled by image size (pure-resize exactness; no new measurement)")
    provenance["intrinsics"] = ("; ".join(notes) if rescale else
                                "unavailable: no camera could be attributed to these images")
    provenance["covisibility"] = ("unavailable: the reference model does not contain these images"
                                  if not overlap else "reference model")
    return rescale, {}, provenance


def _pair_strategy(paths: Sequence[str], strategy: str) -> List[Tuple[str, str]]:
    """A reproducible, content-only pair set (no ground truth consulted)."""
    pairs: List[Tuple[str, str]] = []
    stride = 2 if strategy == "sequential_skip2" else 1
    for step in range(1, stride + 1):
        for i in range(len(paths) - step):
            pairs.append((paths[i], paths[i + step]))
    return pairs


# --------------------------------------------------------------------------- camera graph
class _UnionFind:
    def __init__(self, nodes: Sequence[str]) -> None:
        self.parent = {n: n for n in nodes}

    def find(self, a: str) -> str:
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _camera_graph(nodes: Sequence[str], verified_pairs: List[Dict[str, Any]],
                  min_inliers: int) -> Dict[str, Any]:
    """Connectivity, weak cameras and triangle cycle consistency from VERIFIED pairs."""
    import numpy as np

    from perception.camera_pose_graph import cycle_residuals

    uf = _UnionFind(nodes)
    edges: List[Tuple[str, str]] = []
    for row in verified_pairs:
        if row["inliers"] >= min_inliers:
            a, b = row["a"], row["b"]
            uf.union(a, b)
            edges.append((a, b))
    degree: Dict[str, int] = {n: 0 for n in nodes}
    for a, b in edges:
        degree[a] += 1
        degree[b] += 1
    components: Dict[str, List[str]] = {}
    for n in nodes:
        components.setdefault(uf.find(n), []).append(n)

    # Cycle consistency: on every triangle whose three pairs were verified, composing
    # a->b then b->c must equal a->c, i.e. angle(R_bc @ R_ab @ R_ac.T).  The product
    # order is NOT symmetric: the previous version multiplied R_ab @ R_bc, which is a
    # similarity transform of the closure, not the closure, so it reported tens of
    # degrees on perfectly consistent rotations (pinned by
    # tests/test_camera_pose_graph.py::test_cycle_residual_is_zero_for_consistent_poses).
    triangles = cycle_residuals(nodes, [(row["a"], row["b"], row["relative_pose"]["R"])
                                        for row in verified_pairs if row.get("relative_pose")])
    residuals = sorted(t["residual_deg"] for t in triangles)
    return {
        "nodes": len(nodes),
        "edges_min_inliers": len(edges),
        "components": sorted((sorted(c) for c in components.values()), key=len, reverse=True),
        "n_components": len(components),
        "isolated_cameras": sorted(n for n in nodes if degree[n] == 0),
        "weakly_connected_cameras": sorted(n for n in nodes if degree[n] == 1),
        "triangles_checked": len(residuals),
        "cycle_closure": "R_bc @ R_ab @ R_ac.T == I (a->b then b->c equals a->c)",
        "cycle_residual_deg": {
            "median": round(float(np.median(residuals)), 4) if residuals else None,
            "p90": round(float(np.percentile(residuals, 90)), 4) if residuals else None,
            "max": round(float(max(residuals)), 4) if residuals else None,
            "n_over_5_deg": int(sum(1 for r in residuals if r > 5.0)),
        },
        "worst_triangles": [{"a": Path(t["a"]).name, "b": Path(t["b"]).name,
                             "c": Path(t["c"]).name, "residual_deg": t["residual_deg"]}
                            for t in triangles[:5]],
        "note": ("cycle residuals are rotation-only (two-view translation is scale-free); a large "
                 "residual means the pair poses disagree around the loop, not that a camera is wrong"),
    }


def _gt_covisibility_score(nodes: Sequence[str], engine_pairs: List[Tuple[str, str]],
                           covisibility: Dict[str, List[int]], min_shared: int = 50) -> Dict[str, Any]:
    """How well the engine pair graph covers the reference model's co-visibility graph."""
    if not covisibility:
        return {"status": "unavailable", "reason": "no sparse model co-visibility in this dataset"}
    by_name = {Path(n).name: n for n in nodes}
    names = [Path(n).name for n in nodes]
    ids = {Path(n).name: set(covisibility.get(Path(n).name, [])) for n in names}
    gt: set[Tuple[str, str]] = set()
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            if len(ids[a] & ids[b]) >= min_shared:
                gt.add((a, b))
    engine = {(Path(a).name, Path(b).name) if Path(a).name < Path(b).name else (Path(b).name, Path(a).name)
              for a, b in engine_pairs}
    gt_norm = {(a, b) if a < b else (b, a) for a, b in gt}
    covered = engine & gt_norm
    return {
        "status": "measured",
        "min_shared_points3d": min_shared,
        "n_gt_pairs": len(gt_norm),
        "n_engine_pairs": len(engine),
        "n_covered": len(covered),
        "coverage_of_reference_pairs": round(len(covered) / max(1, len(gt_norm)), 4),
        "engine_pairs_not_covisible_in_reference": sorted(f"{a}|{b}" for a, b in engine - gt_norm)[:50],
        "note": ("the reference model's co-visibility is an upper bound from a different reconstruction; "
                 "coverage measures whether the pair set could support that topology at all"),
    }


# --------------------------------------------------------------------------- stages
def _stage1_registration(images: Sequence[str], intrinsics: Dict[str, Dict[str, float]],
                         pairs: List[Tuple[str, str]], min_inliers: int) -> Dict[str, Any]:
    from perception.engine_adapters import register_all_engines
    from perception.engine_orchestrator import MultiEngineOrchestrator, list_available_engines

    register_all_engines()
    available = sorted(e.engine_id for e in list_available_engines())
    matching = [e for e in ("sift", "lightglue_aliked") if e in available]
    if not matching:
        return {"status": "unavailable", "reason": "neither SIFT nor LightGlue is available",
                "available_engines": available}

    from perception.engine_orchestrator import merge_engine_outputs

    intr = {str(p): (intrinsics.get(str(p)) or intrinsics[Path(str(p)).name])
            for p in images if str(p) in intrinsics or Path(str(p)).name in intrinsics}
    t0 = time.perf_counter()
    orch = MultiEngineOrchestrator()
    outputs = orch.run_wave(list(images),
                            {"run_id": "registration", "pairs": pairs, "intrinsics": intr},
                            engine_ids=matching)
    wall = time.perf_counter() - t0
    merged = merge_engine_outputs(outputs)
    executions = [e.to_dict() for e in orch._executions]  # noqa: SLF001 -- the measured execution records
    return {"status": "measured", "wall_s": round(wall, 2), "matching_engines": matching,
            "available_engines": available, "engine_executions": executions,
            "n_observations": len(merged.observations), "merged": merged}


def _pair_rows(merged: Any, min_inliers: int) -> List[Dict[str, Any]]:
    """One measured row per engine pair, straight from the engine's own observation."""
    rows: List[Dict[str, Any]] = []
    for obs in merged.observations:
        m = dict(obs.metadata)
        engine_id = obs.model_id.architecture_id
        rows.append({
            "engine": engine_id,
                "a": m.get("a_path"),
                "b": m.get("b_path"),
                "keypoints_a": m.get("keypoints_a"),
                "keypoints_b": m.get("keypoints_b"),
                "putative_matches": m.get("matches"),
                "inliers": m.get("inliers"),
                "inlier_ratio": m.get("inlier_ratio"),
                "epipolar_residual_median_px": m.get("epipolar_residual_median_px"),
                "epipolar_residual_p90_px": m.get("epipolar_residual_p90_px"),
                "relative_pose": m.get("relative_pose"),
                "runtime_s": m.get("runtime_s"),
                "verified": bool(m.get("verified")),
                "counted_as_connected": bool((m.get("inliers") or 0) >= min_inliers),
            })
    return rows


_RELOAD_SNIPPET = r"""
import json, os, sys
from pathlib import Path
from worldstore.store import WorldStore

root, version, lineage_file = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
store = WorldStore(root / "worldstore")
problems = store.verify_version(version)
world = store.load_version(version)
block = (world.metadata or {}).get("multi_engine_evidence")
on_disk = json.loads(lineage_file.read_text(encoding="utf-8")) if lineage_file.is_file() else None
print(json.dumps({
    "pid": os.getpid(),
    "world_id": world.id,
    "problems": problems,
    "has_lineage": block is not None,
    "matches_lineage_file": block == on_disk,
    "engine_statuses": (block or {}).get("engine_statuses"),
    "observation_content_ids_by_engine": {
        k: len(v) for k, v in ((block or {}).get("observation_content_ids_by_engine") or {}).items()
    },
}))
"""


def _reference_poses(dataset: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Per-image poses from the dataset's reference COLMAP text model, or ({}, reason)."""
    from perception.camera_pose_graph import read_colmap_text_model

    try:
        poses = read_colmap_text_model(dataset / "sparse")
    except (FileNotFoundError, ValueError) as exc:
        return {}, {"status": "unavailable", "reason": str(exc)}
    if not poses:
        return {}, {"status": "unavailable", "reason": "reference model parsed but contains no images"}
    return poses, {"status": "measured", "source": str(dataset / "sparse" / "images.txt"),
                   "images": len(poses)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--pairs", choices=["sequential_skip2", "sequential"], default="sequential_skip2")
    ap.add_argument("--min-inliers", type=int, default=DEFAULT_MIN_INLIERS)
    ap.add_argument("--allow-synthetic", action="store_true",
                    help="accept a dataset whose own manifest says it is synthetic (it is then labelled)")
    ap.add_argument("--skip-production", action="store_true", help="run Stage 1 only")
    args = ap.parse_args()

    dataset = args.dataset.resolve()
    manifest = _dataset_manifest(dataset)
    synthetic = _is_synthetic(manifest)
    if synthetic and not args.allow_synthetic:
        print(f"REFUSED: {dataset} describes itself as synthetic ({manifest.get('description')!r}). "
              "This script exists to measure REAL captures; re-run with --allow-synthetic to label it as such.",
              file=sys.stderr)
        return 2

    images = sorted(str(p) for p in (dataset / "images").glob("*.[jJ][pP][gG]"))
    if args.max_images:
        images = images[: args.max_images]
    if len(images) < 2:
        print(f"REFUSED: {len(images)} usable images in {dataset}/images", file=sys.stderr)
        return 2

    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    out_dir = (args.out or dataset / "runs" / f"multi_engine_{stamp}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch

    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    vram_total = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 2), 1) if torch.cuda.is_available() else None
    report: Dict[str, Any] = {
        "dataset": {"path": str(dataset), "identity": manifest.get("identity") or manifest.get("dataset"),
                    "declared_synthetic": synthetic, "image_count": len(images),
                    "pair_strategy": args.pairs, "min_inliers": args.min_inliers},
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "torch_cuda": torch.version.cuda, "cuda_available": bool(torch.cuda.is_available()),
                        "device": device, "vram_total_mb": vram_total},
    }

    intrinsics, covisibility, camera_provenance = _colmap_intrinsics(dataset, images)
    report["dataset"]["intrinsics_source"] = ("dataset sparse/cameras.txt" if intrinsics else "unavailable")
    report["dataset"]["camera_provenance"] = camera_provenance
    report["dataset"]["intrinsics_known_for"] = len([p for p in images
                                                     if p in intrinsics or Path(p).name in intrinsics])

    pairs = _pair_strategy(images, args.pairs)
    print(f"[stage 1] {len(images)} images, {len(pairs)} pairs ({args.pairs})", flush=True)
    stage1 = _stage1_registration(images, intrinsics, pairs, args.min_inliers)
    if stage1.get("status") != "measured":
        report["stage1_registration"] = stage1
        out = out_dir / "report.json"
        out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(json.dumps(stage1, indent=2)[:2000])
        return 3

    merged = stage1.pop("merged")
    # Observations from every engine are merged into one report; each row names the engine
    # that produced it (its architecture id), so the per-engine statistics stay separate.
    rows = _pair_rows(merged, args.min_inliers)
    by_engine: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_engine.setdefault(row["engine"], []).append(row)
    verified_pairs = [r for r in rows if r["verified"] and r["inliers"] and r["inliers"] >= args.min_inliers]
    # one row per image pair (prefer the engine with more inliers, record both)
    best: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for r in verified_pairs:
        key = tuple(sorted((r["a"], r["b"])))
        if key not in best or (r["inliers"] or 0) > (best[key]["inliers"] or 0):
            best[key] = r
    graph = _camera_graph(images, list(best.values()), args.min_inliers)
    gt = _gt_covisibility_score(images, [(r["a"], r["b"]) for r in best.values()], covisibility)

    # Is the cycle disagreement a property of the pair poses or of the estimator? A
    # relative rotation does not depend on the world gauge, so the reference model can
    # score every pair directly.
    from perception.camera_pose_graph import compare_pair_rotations, solve_pose_graph

    ref_poses, ref_provenance = _reference_poses(dataset)
    ref_compare = (compare_pair_rotations(list(best.values()), ref_poses) if ref_poses
                   else {"status": "unavailable", "reason": ref_provenance.get("reason")})
    pose_edges = [{"a": r["a"], "b": r["b"], "R": r["relative_pose"]["R"],
                   "t_unit": r["relative_pose"].get("t_unit"), "inliers": r["inliers"]}
                  for r in best.values() if r.get("relative_pose")]
    pose_graph = solve_pose_graph(images, pose_edges, reference=ref_poses or None)
    report["stage1_registration"] = {
        "status": "measured",
        "wall_s": stage1["wall_s"],
        "matching_engines": stage1["matching_engines"],
        "available_engines": stage1["available_engines"],
        "engine_executions": stage1["engine_executions"],
        "pairs_requested": len(pairs),
        "per_engine": {eid: {
            "n_verified_pairs": len(v),
            "median_inliers": sorted(r["inliers"] for r in v)[len(v) // 2] if v else None,
            "median_inlier_ratio": round(sorted(r["inlier_ratio"] for r in v)[len(v) // 2], 4) if v else None,
            "median_epipolar_residual_px": (round(sorted(r["epipolar_residual_median_px"] for r in v)[len(v) // 2], 4)
                                            if v else None),
            "pairs": v,
        } for eid, v in sorted(by_engine.items())},
        "verified_image_pairs": len(best),
        "camera_graph": graph,
        "gt_covisibility": gt,
        "reference_model": ref_provenance,
        "reference_pose_comparison": ref_compare,
        "pose_graph": pose_graph,
    }
    print(f"[stage 1] verified {len(best)}/{len(pairs)} pairs; components={graph['n_components']}", flush=True)
    if ref_compare.get("status") == "measured":
        print(f"[stage 1] pair rotation vs reference: median "
              f"{ref_compare['median_rotation_error_deg']} deg over {ref_compare['pairs_compared']} pairs", flush=True)
    if pose_graph.get("status") == "optimised":
        pc = pose_graph.get("reference_comparison") or {}
        print(f"[stage 1] pose graph: {pose_graph['edges_accepted']}/{pose_graph['edges_given']} edges, "
              f"camera rotation error median {(pc.get('rotation_error_deg') or {}).get('median')} deg", flush=True)

    if not args.skip_production:
        print("[stage 2] production evidence-world run (all engines + matrix + world)", flush=True)
        from apps.api.evidence_world_stage import evidence_root, MULTI_ENGINE_EVIDENCE_FILE, run_for_job

        base = out_dir
        job_id = f"real-{stamp}"
        t0 = time.perf_counter()
        facts = run_for_job(team_id="local", world_id="south-building", job_id=job_id,
                            sources=images, base=base, detection="legacy")
        run_dir = evidence_root("local", "south-building", job_id, base=base)
        me = facts.get("multi_engine") or {}
        report["stage2_production_run"] = {
            "job_id": job_id,
            "run_dir": str(run_dir),
            "wall_s": round(time.perf_counter() - t0, 2),
            "status": facts.get("status"),
            "world_ir": facts.get("world_ir"),
            "multi_engine_status": me.get("status"),
            "engines_run": me.get("engines_run"),
            "engine_executions": me.get("engine_executions"),
            "vram_peak_mb": me.get("vram_peak_mb"),
            "n_observations": me.get("n_observations"),
            "matrix_import": {k: v for k, v in (me.get("matrix_import") or {}).items()
                              if k in ("status", "n_observations_imported", "n_observations_already_present",
                                       "n_run_records", "engine_statuses")},
            "lineage": facts.get("multi_engine_evidence"),
            "detection_pipeline": (facts.get("detection_pipeline") or {}).get("mode"),
        }
        print(f"[stage 2] status={facts.get('status')} world={((facts.get('world_ir') or {}).get('status'))}", flush=True)

        world = facts.get("world_ir") or {}
        if world.get("status") == "persisted":
            print("[stage 3] fresh-process reload", flush=True)
            child = subprocess.run([sys.executable, "-c", _RELOAD_SNIPPET, str(run_dir),
                                    world["version_id"], str(run_dir / MULTI_ENGINE_EVIDENCE_FILE)],
                                   cwd=str(REPO), capture_output=True, text=True, timeout=300)
            child_ok = child.returncode == 0
            parsed: Dict[str, Any] = {}
            if child_ok:
                try:
                    parsed = json.loads(child.stdout.strip().splitlines()[-1])
                except (ValueError, IndexError):
                    child_ok = False
            report["stage3_reload"] = {
                "status": "verified" if (child_ok and parsed.get("matches_lineage_file")
                                         and parsed.get("has_lineage") and not parsed.get("problems")) else "failed",
                "fresh_process": True,
                "returncode": child.returncode,
                "stderr_tail": child.stderr.strip().splitlines()[-5:] if child.stderr else [],
                "child": parsed,
                "parent_lineage_engine_statuses": (facts.get("multi_engine_evidence") or {}).get("engine_statuses"),
            }
        else:
            report["stage3_reload"] = {"status": "not_run",
                                       "reason": f"no world was persisted ({world.get('status')}: {world.get('reason')})"}

    acc = report["stage1_registration"]
    report["acceptance"] = {
        "real_dataset": not synthetic,
        "real_images": len(images),
        "verified_pair_graph": acc["verified_image_pairs"] > 0,
        "single_connected_camera_group": acc["camera_graph"]["n_components"] == 1,
        "cycle_consistency_measured": acc["camera_graph"]["triangles_checked"] > 0,
        "pair_rotations_compared_to_reference": acc["reference_pose_comparison"].get("status") == "measured",
        "pose_graph_solved": acc["pose_graph"].get("status") == "optimised",
        "pose_graph_camera_error_measured": ((acc["pose_graph"].get("reference_comparison") or {})
                                             .get("status") == "measured"),
        "world_lineage_reload": (report.get("stage3_reload") or {}).get("status") == "verified",
        "not_claimed": ["metric scale", "survey accuracy", "depth accuracy (no ground-truth depth)",
                        "architectural entity accuracy (no surveyed ground truth)"],
    }
    out = out_dir / "report.json"
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("dataset", "acceptance")}, indent=2, default=str))
    print("report:", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
