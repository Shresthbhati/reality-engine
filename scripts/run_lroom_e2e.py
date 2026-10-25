#!/usr/bin/env python3
"""End-to-end: synthetic L-shaped room (false ceiling, window, furniture) -> real images -> scene gate -> REAL COLMAP ->
production vertical slice (compile + interior semantics) -> measured comparison with the known ground truth.

    python scripts/run_lroom_e2e.py --work <dir> --report <report.json> [--views 72]

Reports what the pipeline actually produced: rooms, footprint area vs 24.0 m^2, ceiling layers, window, furniture
(matched to ground truth through the camera-pose gauge), colours. Nothing is asserted that was not measured.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.pipeline.vertical_slice import VerticalSliceOptions, vertical_slice  # noqa: E402
from evidence.session import EvidenceItem, EvidenceKind  # noqa: E402
from reconstruction.backend.interface import ReconstructedCameraPose  # noqa: E402
from reconstruction.evaluation import _matrix_to_quat, evaluate_absolute_poses  # noqa: E402
from reconstruction.scale import ScaleReference  # noqa: E402
from reconstruction.scene_identity import SiftPairVerifier, group_images  # noqa: E402
from synthetic import lroom as L  # noqa: E402


def render_all(work: Path, views: int):
    img_dir = work / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    quads = L.build_scene()
    cams = L.trajectory()[:views]
    gt, paths = {}, {}
    for c in cams:
        R = L.look_at(c["position"], c["look_at"])
        t = -R @ c["position"]
        p = img_dir / f"{c['id']}.jpg"
        if not p.exists():
            Image.fromarray(L.render(quads, R, t)).save(p, quality=92)
        w, x, y, z = _matrix_to_quat(R.tolist())
        gt[c["id"]] = (w, x, y, z, *[float(v) for v in t])
        paths[c["id"]] = p
    (work / "lroom_gt.json").write_text(json.dumps({"poses": gt, "scene": L.ground_truth(),
                                                    "positions": {c["id"]: c["position"].tolist() for c in cams}},
                                                   default=str))
    return paths, gt, {c["id"]: c["position"] for c in cams}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--views", type=int, default=72)
    a = ap.parse_args()
    work = Path(a.work)
    t0 = time.time()
    report: dict = {"cmd": " ".join(sys.argv)}
    paths, gt, positions = render_all(work, a.views)
    report["render_s"] = round(time.time() - t0, 1)
    ids = sorted(paths)

    # scene gate: all views must form one verified group
    t1 = time.time()
    grouping = group_images(ids, SiftPairVerifier(paths))
    report["grouping"] = {"groups": [len(g.members) for g in grouping.groups], "unlinked": grouping.unlinked,
                          "verified_pairs": grouping.verified_pairs, "s": round(time.time() - t1, 1)}
    print("groups:", report["grouping"], flush=True)
    members = grouping.primary.members if grouping.primary else ids

    evidence = [EvidenceItem(id=i, kind=EvidenceKind.PHOTO, source_uri=paths[i].resolve().as_uri()) for i in members]
    # operator-measured baselines (what a tape measure gives): pairs of cameras far apart
    pairs = [(members[0], members[-1]), (members[8], members[40 if len(members) > 40 else -2]),
             (members[16], members[32 if len(members) > 32 else -3])]
    baselines = tuple(ScaleReference(x, y, float(np.linalg.norm(positions[x] - positions[y])), "tape_measure")
                      for x, y in pairs if x in positions and y in positions and x != y)
    t2 = time.time()
    vs = vertical_slice(evidence, VerticalSliceOptions(
        depth_model=None, perception_model=None, mesh_enabled=False, detail_enabled=False, dense_auto=False,
        measured_baselines=baselines, image_size=(L.W, L.H)))
    report["pipeline_s"] = round(time.time() - t2, 1)
    world = vs.world
    report["registration"] = {"registered": vs.cameras_registered, "input": vs.cameras_input, "points": vs.points_total,
                              "status": vs.registration_status, "scale_state": vs.scale_state}
    print("registered", vs.cameras_registered, "/", vs.cameras_input, "points", vs.points_total, "scale", vs.scale_state,
          flush=True)

    # keep the canonical-frame cloud + cameras so interior analysis can be iterated without re-running COLMAP
    np.savez_compressed(work / "lroom_cloud.npz", points=np.asarray(vs.points, float),
                        cam_ids=np.array([c[0] for c in vs.camera_poses]),
                        cam_pos=np.array([c[1] for c in vs.camera_poses], float),
                        cam_rot=np.array([c[2] for c in vs.camera_poses], float))
    types: dict = {}
    for e in world.entities.values():
        types[e.type.value] = types.get(e.type.value, 0) + 1
    report["entities"] = types
    rooms = [e for e in world.entities.values() if e.type.value == "room"]
    report["rooms"] = [{"id": r.id, **{k: r.custom_properties.get(k) for k in (
        "floor_area_m2", "bbox_floor_area_m2", "dimensions_m", "ceiling_layers", "ceiling_height_min_m",
        "ceiling_height_max_m", "footprint_method")},
        "footprint_vertices": len(r.custom_properties.get("footprint_polygon_m") or [])} for r in rooms]
    report["windows"] = [{k: e.custom_properties.get(k) for k in ("width_m", "height_m", "sill_height_m", "host_wall_id")}
                         for e in world.entities.values() if e.type.value == "window"]
    report["ceilings"] = [{"id": e.id, "labels": e.semantic_labels, "role": e.custom_properties.get("ceiling_role"),
                           "height_above_floor_m": e.custom_properties.get("height_above_floor_m")}
                          for e in world.entities.values() if e.type.value == "ceiling"]
    furn = [e for e in world.entities.values() if "furniture_candidate" in e.semantic_labels]
    report["furniture"] = [{"id": e.id, "size_m": e.custom_properties.get("plan_obb_size_m"),
                            "height_m": e.custom_properties.get("height_m"), "n_points": e.custom_properties.get("n_points"),
                            "color": e.custom_properties.get("mean_color_rgb")} for e in furn]
    report["semantics"] = world.metadata.get("interior_semantics")
    report["colours_on_structure"] = sum(1 for e in world.entities.values() if e.custom_properties.get("mean_color_rgb"))

    # pose accuracy + furniture vs GT through the pose gauge
    poses = [ReconstructedCameraPose(i, tuple(p), tuple(q)) for i, p, q in vs.camera_poses]
    ev = evaluate_absolute_poses(poses, gt)
    report["pose_vs_gt"] = {k: ev.get(k) for k in ("common_cameras", "rotation_deg_median", "rotation_deg_max",
                                                  "center_error_median", "center_error_max", "scale_ratio", "refused")}
    if not ev.get("refused") and ev.get("gauge_transform"):
        Rg, tg, sg = ev["gauge_transform"]
        Rg = np.asarray(Rg)
        match = []
        for name, box in L.FURNITURE.items():
            gc = (np.array(box[0]) + np.array(box[1])) / 2
            best = None
            for e in furn:
                p = e.transform["position"]
                c = sg * (Rg @ np.array([p["x"], p["y"], p["z"]])) + np.asarray(tg)
                d = float(np.linalg.norm(c - gc))
                if best is None or d < best[0]:
                    best = (d, e.id)
            match.append({"gt": name, "nearest_entity": best[1] if best else None,
                          "centre_error_m_in_gt_units": round(best[0], 3) if best else None})
        report["furniture_match"] = match
    report["total_s"] = round(time.time() - t0, 1)
    Path(a.report).write_text(json.dumps(report, indent=1, default=str))
    try:   # keep the compiled world for the viewer / export check
        (work / "lroom_world.json").write_text(json.dumps(world.to_dict(), default=str))
    except Exception as exc:  # noqa: BLE001
        print("world export skipped:", exc)
    print(json.dumps({k: report[k] for k in ("registration", "entities", "rooms", "windows", "ceilings", "furniture",
                                             "furniture_match", "pose_vs_gt") if k in report}, indent=1, default=str))


if __name__ == "__main__":
    main()
