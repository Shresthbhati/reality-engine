"""Executable capability audit (Master Directive Rule 1).

For each capability row it records the implementing modules, whether they are REACHABLE from the live API app through
a static import graph (lazy imports included), and the outcome of the row's own tests, which this tool runs. A status is
computed from evidence, never asserted:

  Existing  modules exist, are reachable from the API, tests exist and all pass
  Partial   modules exist but are not reachable from the API, or tests are missing / failing, or some modules are absent
  Missing   no implementing module exists

"Verified" means the row's tests were executed and passed in this run. It does NOT mean accuracy on real photographs:
that needs ground truth (see benchmarks/reality_benchmark). ``real_data_notes`` records what was actually observed.

  python scripts/capability_audit.py [--no-tests] [--out docs/engineering]
"""

from __future__ import annotations

import argparse
import ast
import json
import pathlib
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Sequence, Set

ROOT = pathlib.Path(__file__).resolve().parents[1]
LIVE_ROOTS = ("apps.api.main", "apps.api.jobs")
SKIP_DIRS = {".venv", "tests", "node_modules", ".claude", "__pycache__", ".git", "docs", "frontend", "web"}

# row -> implementing modules and the tests that exercise them
ROWS: Dict[str, dict] = {
    "Image ingestion": dict(
        modules=["evidence/importers.py", "evidence/session.py", "evidence/image_check.py", "evidence/dedup.py"],
        tests=["tests/test_universal_ingestion_extensions.py", "tests/test_task_13_3_extended_ingestion.py", "tests/test_evidence_quality.py"]),
    "Video ingestion": dict(
        modules=["reconstruction/video_frames.py", "evidence/video.py"],
        tests=["tests/test_video_frames.py", "tests/test_video_ingest_task9_2.py", "tests/test_video_keyframe_extraction.py"]),
    "Image quality analysis": dict(
        modules=["reconstruction/capture_quality.py", "evidence/quality_checks.py"],
        tests=["tests/test_capture_quality.py", "tests/test_evidence_quality.py"]),
    "Camera calibration": dict(
        modules=["reconstruction/calibration/camera.py", "reconstruction/colmap_intrinsics.py", "trajectories/interpolation.py"],
        tests=["tests/test_camera_calibration.py", "tests/test_calibration_transforms.py", "tests/test_pose_adopts_solved_intrinsics.py", "tests/test_trajectory_interpolation.py"]),
    "Object detection": dict(
        modules=["perception/detection/grounded_scene.py", "engine/pipeline/open_vocab_furniture.py", "perception/detection/maskrcnn_backend.py"],
        tests=["tests/test_grounded_scene_segmenter.py", "tests/test_open_vocab_furniture.py", "tests/test_detection_backends.py"]),
    "Instance segmentation": dict(
        modules=["perception/segmentation/sam2_backend.py", "perception/segmentation/sam_backend.py", "perception/detection/masks.py"],
        tests=["tests/test_detection_masks.py"]),
    "Small-object detection": dict(
        modules=["perception/entities/small_objects.py", "perception/detail/crop_pass.py", "perception/fine_grained/pipeline.py", "perception/fine_grained/estimators.py"],
        tests=["tests/test_fine_grained_perception_rule5.py", "tests/test_fine_grained_entity_attributes.py", "tests/test_prelabel_images.py", "tests/test_fine_grained_estimators.py"]),
    "Object tracking": dict(
        modules=["perception/instances/track_backend.py", "perception/tracking"],
        tests=["tests/test_temporal_tracking.py", "tests/test_track_backend.py", "tests/test_tracking_2d.py"]),
    "Depth estimation": dict(
        modules=["perception/depth/depth_anything_backend.py", "perception/depth/midAS_backend.py", "reconstruction/depth_to_points.py"],
        tests=["tests/test_depth_anything_backend.py", "tests/test_depth_to_points.py", "tests/test_depth_scale.py", "tests/test_stage_depth.py"]),
    "SfM": dict(
        modules=["reconstruction/backend/colmap_backend.py", "reconstruction/colmap_session.py", "reconstruction/sfm/core_pipeline.py"],
        tests=["tests/test_colmap_backend.py", "tests/test_colmap_session.py", "tests/test_mapper_retry.py", "tests/test_sfm_pipeline_rule4.py"]),
    "MVS": dict(
        modules=["reconstruction/mvs/dense_cloud.py", "reconstruction/dense_pipeline.py", "reconstruction/backend/plane_sweep.py"],
        tests=["tests/test_dense_mvs_stage.py", "tests/test_mvs_point_cloud_ops.py", "tests/test_colmap_backend_dense.py"]),
    "Point-cloud fusion": dict(
        modules=["reconstruction/fusion/fusion.py", "reconstruction/fusion/multi_source.py"],
        tests=["tests/test_fusion_pipeline.py", "tests/test_fusion_stage.py", "tests/test_dense_cloud_fusion.py"]),
    "Mesh reconstruction": dict(
        modules=["reconstruction/meshing/mesh.py", "reconstruction/meshing/surface.py", "reconstruction/meshing/room_mesh.py"],
        tests=["tests/test_meshing_pipeline.py", "tests/test_mesh_validation.py", "tests/test_mesh_repair_and_completeness.py", "tests/test_room_mesh.py"]),
    "Multi-view identity": dict(
        modules=["perception/instances/association.py", "perception/instances/object_resolution.py", "reconstruction/scene_identity.py"],
        tests=["tests/test_association.py", "tests/test_scene_identity.py", "tests/test_multiview_identity_appearance.py", "tests/test_identity_carry.py"]),
    "Room segmentation": dict(
        modules=["perception/architecture/room_semantics.py", "perception/architecture/room_graph.py", "perception/architecture/room_layout.py"],
        tests=["tests/test_room_building_graph.py", "tests/test_non_rectangular_rooms.py", "tests/test_room_fidelity_fixture.py",
               "tests/test_horizontal_consolidation.py", "tests/test_vertical_consolidation.py"]),
    "Architectural topology": dict(
        modules=["perception/architecture/topology.py", "perception/architecture/space_graph.py", "perception/architecture/openings.py", "perception/architecture/stairs.py", "perception/architecture/multi_storey_validator.py"],
        tests=["tests/test_topology_coherence.py", "tests/test_vertical_topology_graph.py", "tests/test_multi_storey_validator.py"]),
    "Material recognition": dict(
        modules=["perception/materials/classifier.py", "perception/materials/material_hypotheses.py", "perception/materials/room_surfaces.py"],
        tests=["tests/test_material_classification.py", "tests/test_material_hypotheses.py", "tests/test_pbr_material_classification.py"]),
    "Uncertainty estimation": dict(
        modules=["engine/quality/report.py", "reconstruction/consistency.py", "provenance"],
        tests=["tests/test_uncertainty_propagation.py", "tests/test_depth_consistency.py", "tests/test_quality_report.py"]),
    "WorldIR": dict(
        modules=["world_ir"],
        tests=["tests/test_arch_worldir.py", "tests/test_commit_validation_contract.py", "tests/test_cross_entity_validation.py"]),
    "Accuracy validation": dict(
        modules=["reconstruction/evaluation.py", "reconstruction/validation.py", "perception/architecture/accuracy_report.py",
                 "benchmarks/reality_benchmark/scorer.py", "benchmarks/reality_benchmark/inventory.py", "benchmarks/measured_rooms/run_accuracy_benchmark.py"],
        tests=["tests/test_reconstruction_validation.py", "tests/test_measured_room_accuracy_report.py", "tests/test_reality_benchmark.py",
               "tests/test_reality_inventory.py", "tests/test_world_accuracy_report.py", "tests/test_run_accuracy_benchmark.py"]),
    "Active capture guidance": dict(
        modules=["reconstruction/capture_quality.py", "reconstruction/completeness.py"],
        tests=["tests/test_capture_planner.py", "tests/test_capture_requests_task6_1.py", "tests/test_capture_feedback_wiring.py"]),
    "Export and persistence": dict(
        modules=["exporters/gltf", "exporters/ifc_bridge.py", "worldstore/store.py"],
        tests=["tests/test_gltf_exporter.py", "tests/test_export_pipeline_e2e.py", "tests/test_worldstore_concurrency.py",
               "tests/test_worldstore_retention.py", "tests/test_ifc_bridge.py"]),
}

#: what was actually observed on real photographs (the 10-photo living room, this project); empty when nothing was
REAL_DATA_NOTES: Dict[str, str] = {
    "Object detection": "Grounding DINO coarse pass inspected on one 1672x941 interior: vents, AC, doors, windows, shelves, plants, picture frame boxed sensibly.",
    "Small-object detection": "Tiled pass on 10 images: low precision (railings as 'visible wiring', texture as 'drain'); 678 unreviewed drafts, no ground truth.",
    "SfM": "10 photos: 9/10 registered after seed pinning; sparse geometry differed between runs (995 vs 1,423 points).",
    "Depth estimation": "DPT_Hybrid needs a scale per view; a metric model (Depth Anything V2 Metric-Indoor Small) fits SfM better (0.117 vs 0.199 relative residual) with ONE scale, but did not change floor/ceiling structure in a controlled A/B (docs/engineering/TECHNOLOGY_EVALUATION.md 3a).",
    "Room segmentation": "Floor/wall consolidation cut replayed wall counts 25->8 and 21->6; footprint remains a ragged polygon.",
}


def module_name(path: pathlib.Path, root: pathlib.Path = ROOT) -> str:
    parts = list(path.relative_to(root).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def build_import_graph(root: pathlib.Path = ROOT) -> Dict[str, Set[str]]:
    """module -> modules it imports (including imports nested in functions), restricted to modules in the repo."""
    files = {}
    for p in root.rglob("*.py"):
        if SKIP_DIRS & set(p.relative_to(root).parts[:-1]):
            continue
        files[module_name(p, root)] = p
    graph: Dict[str, Set[str]] = {m: set() for m in files}
    trees: Dict[str, ast.AST] = {}
    for mod, p in files.items():
        try:
            trees[mod] = ast.parse(p.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            pass

    def resolve(mod: str, p: pathlib.Path, n: ast.ImportFrom) -> str:
        base = n.module or ""
        if n.level:
            pkg = mod if p.name == "__init__.py" else mod.rpartition(".")[0]
            up = pkg.split(".") if pkg else []
            up = up[: len(up) - (n.level - 1)] if n.level > 1 else up
            base = ".".join([*up, *([base] if base else [])])
        return base

    # package -> {re-exported name: the submodule it comes from}: ``from pkg import Name`` is a use of pkg.sub, not of all of pkg
    reexport: Dict[str, Dict[str, str]] = {}
    for mod, p in files.items():
        if p.name == "__init__.py" and mod in trees:
            for n in ast.walk(trees[mod]):
                if isinstance(n, ast.ImportFrom):
                    src = resolve(mod, p, n)
                    for a in n.names:
                        reexport.setdefault(mod, {})[a.asname or a.name] = src if f"{src}.{a.name}" not in files else f"{src}.{a.name}"
    for mod, p in files.items():
        if p.name == "__init__.py" or mod not in trees:
            continue          # a package __init__ eagerly imports its re-exports; that is not a use of them
        for n in ast.walk(trees[mod]):
            if isinstance(n, ast.Import):
                targets = [a.name for a in n.names]
            elif isinstance(n, ast.ImportFrom):
                base = resolve(mod, p, n)
                targets = [base] + [f"{base}.{a.name}" for a in n.names]
                targets += [reexport[base][a.name] for a in n.names if a.name in reexport.get(base, {})]
            else:
                continue
            for t in targets:
                while t:                                   # an import of a.b.c also loads a.b and a
                    if t in graph and t != mod:
                        graph[mod].add(t)
                    t = t.rpartition(".")[0]
    return graph


def reach_parents(graph: Dict[str, Set[str]], roots: Sequence[str] = LIVE_ROOTS) -> Dict[str, Optional[str]]:
    """Breadth-first from the live roots: module -> the module that first imported it (None for a root)."""
    parent: Dict[str, Optional[str]] = {r: None for r in roots if r in graph}
    queue = list(parent)
    while queue:
        m = queue.pop(0)
        for t in sorted(graph.get(m, ())):
            if t not in parent:
                parent[t] = m
                queue.append(t)
    return parent


def reachable(graph: Dict[str, Set[str]], roots: Sequence[str] = LIVE_ROOTS) -> Set[str]:
    return set(reach_parents(graph, roots))


def chain(parent: Dict[str, Optional[str]], mod: str) -> List[str]:
    out = [mod]
    while parent.get(out[-1]):
        out.append(parent[out[-1]])
    return out[::-1]


def expand_modules(spec: List[str], root: pathlib.Path = ROOT) -> List[pathlib.Path]:
    out: List[pathlib.Path] = []
    for s in spec:
        p = root / s
        if p.is_dir():
            out += [q for q in p.rglob("*.py") if "__pycache__" not in q.parts]
        elif p.exists():
            out.append(p)
    return out


#: rows that are offline tools by design (never on the API path); they need modules + passing tests, not reachability
OFFLINE_ROWS = {"Accuracy validation"}


def classify(n_modules_expected: int, n_modules_found: int, n_entries_reachable: int, n_tests: int, tests_failed: int, tests_run: int,
             offline: bool = False) -> str:
    """Existing = every implementing entry exists and (for live rows) is reachable from the API, tests exist and all pass."""
    if n_modules_found == 0:
        return "Missing"
    live_ok = offline or n_entries_reachable == n_modules_expected
    if n_modules_found == n_modules_expected and live_ok and n_tests > 0 and tests_run > 0 and tests_failed == 0:
        return "Existing"
    return "Partial"


def run_tests(files: List[str], timeout_s: int = 1800) -> Dict[str, dict]:
    """Run pytest once over ``files``; returns file -> {passed, failed, skipped, errors}."""
    if not files:
        return {}
    with tempfile.TemporaryDirectory() as td:
        xml = pathlib.Path(td) / "r.xml"
        cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--tb=no", f"--junitxml={xml}", *files]
        try:
            subprocess.run(cmd, cwd=ROOT, timeout=timeout_s, capture_output=True, text=True)
        except subprocess.TimeoutExpired:
            return {f: {"passed": 0, "failed": 0, "skipped": 0, "errors": 1, "note": f"timed out after {timeout_s}s"} for f in files}
        if not xml.exists():
            return {f: {"passed": 0, "failed": 0, "skipped": 0, "errors": 1, "note": "pytest produced no report"} for f in files}
        res: Dict[str, dict] = {}
        for tc in ET.parse(xml).getroot().iter("testcase"):
            parts = (tc.get("classname") or "").split(".")
            mod = "/".join(p for p in parts if not p[:1].isupper()) + ".py"          # tests.test_x.TestY -> tests/test_x.py
            r = res.setdefault(mod, {"passed": 0, "failed": 0, "skipped": 0, "errors": 0, "failing": []})
            if tc.find("failure") is not None:
                r["failed"] += 1
                r["failing"].append(f"{mod}::{tc.get('name')}")
            elif tc.find("error") is not None:
                r["errors"] += 1
                r["failing"].append(f"{mod}::{tc.get('name')}")
            elif tc.find("skipped") is not None:
                r["skipped"] += 1
            else:
                r["passed"] += 1
        return res


def audit(run: bool = True) -> dict:
    graph = build_import_graph()
    parent = reach_parents(graph)
    live = set(parent)
    test_files = sorted({t for r in ROWS.values() for t in r["tests"] if (ROOT / t).exists()})
    results = run_tests(test_files) if run else {}
    rows = []
    for name, spec in ROWS.items():
        mods = expand_modules(spec["modules"])
        found_specs = [m for m in spec["modules"] if (ROOT / m).exists()]
        names = [module_name(p) for p in mods]
        n_live = sum(1 for n in names if n in live)
        live_names = sorted((n for n in names if n in live), key=lambda n: len(chain(parent, n)))
        live_path = chain(parent, live_names[0]) if live_names else []
        not_live = [n for n in names if n not in live]
        tests = [t for t in spec["tests"] if (ROOT / t).exists()]
        agg = {"passed": 0, "failed": 0, "skipped": 0, "errors": 0}
        failing: List[str] = []
        for t in tests:
            for k in agg:
                agg[k] += results.get(t, {}).get(k, 0)
            failing += results.get(t, {}).get("failing", [])
        ran = agg["passed"] + agg["failed"] + agg["errors"]
        bad = agg["failed"] + agg["errors"]
        entries_live = [m for m in spec["modules"] if any(module_name(p) in live for p in expand_modules([m]))]
        entries_dead = [m for m in found_specs if m not in entries_live]
        status = classify(len(spec["modules"]), len(found_specs), len(entries_live), len(tests), bad, ran if run else 0,
                          offline=name in OFFLINE_ROWS)
        verified = "not run" if not run else ("no tests" if not tests else "FAILING" if bad else "yes (unit tests)" if ran else "no tests ran")
        rows.append({"capability": name, "status": status, "verified": verified,
                     "modules": [str(p.relative_to(ROOT)).replace("\\", "/") for p in mods][:12],
                     "modules_missing": [m for m in spec["modules"] if m not in found_specs], "modules_total": len(mods),
                     "reachable_from_api": n_live, "shortest_live_import_chain": live_path, "not_reachable_from_api": not_live[:12],
                     "entries_not_reachable_from_api": entries_dead, "offline_tool": name in OFFLINE_ROWS, "failing_tests": failing,
                     "tests": tests, "tests_missing": [t for t in spec["tests"] if t not in tests],
                     "test_counts": agg, "real_data_notes": REAL_DATA_NOTES.get(name, "")})
    return {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "live_roots": list(LIVE_ROOTS), "ran_tests": run,
            "note": "Verified = the row's unit tests passed in this run; it is not accuracy on real photographs.", "rows": rows}


def to_markdown(report: dict) -> str:
    lines = ["# Capability matrix", "", report["note"], "", f"Live roots: {', '.join(report['live_roots'])}. Generated {report['generated_at']}.", "",
             "| Capability | Status | Verified | Modules (reachable from API / total) | Tests passed/failed | Real-data evidence |", "|---|---|---|---|---|---|"]
    for r in report["rows"]:
        c = r["test_counts"]
        lines.append(f"| {r['capability']} | {r['status']} | {r['verified']} | {r['reachable_from_api']} / {r['modules_total']} | "
                     f"{c['passed']}/{c['failed'] + c['errors']} | {r['real_data_notes'] or '-'} |")
    dead = [r for r in report["rows"] if r["entries_not_reachable_from_api"] and not r["offline_tool"]]
    if dead:
        lines += ["", "## Implemented but not reachable from the API", ""]
        lines += [f"* {r['capability']}: {', '.join(r['entries_not_reachable_from_api'])}" for r in dead]
    fail = [r for r in report["rows"] if r["failing_tests"]]
    if fail:
        lines += ["", "## Failing tests (this run)", ""]
        lines += [f"* {r['capability']}: {', '.join(r['failing_tests'])}" for r in fail]
    miss = [r for r in report["rows"] if r["modules_missing"] or r["tests_missing"]]
    if miss:
        lines += ["", "## Gaps in the audit itself", ""]
        lines += [f"* {r['capability']}: modules not found {r['modules_missing'] or '-'}; tests not found {r['tests_missing'] or '-'}" for r in miss]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-tests", action="store_true")
    ap.add_argument("--out", default="docs/engineering")
    args = ap.parse_args(argv)
    report = audit(run=not args.no_tests)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "capability_matrix.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    (out / "capability_matrix.md").write_text(to_markdown(report), encoding="utf-8")
    print(to_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
