#!/usr/bin/env python3
"""Run the real capture -> WorldIR vertical slice on a dataset folder.

The P0 flagship path:

    python scripts/run_vertical_slice.py datasets/room_capture

Reads the dataset manifest (images + trusted intrinsics + operator-measured
baselines), ingests every image as EvidenceItem, then executes the vertical
slice: real COLMAP SfM -> metric scale anchoring -> frame canonicalization
-> optional MiDaS depth stage -> world compiler -> validated WorldIR.

Artifacts (written next to --out, default <dataset>/pipeline_out):

  worldir.json   serialized canonical WorldIR (with scale/frame/depth metadata)
  report.json    observed run facts: stage statuses, counts, errors -- no fiction

Exit code 0 on any honest outcome (success OR partial-with-WorldIR), 1 when
a stage refuses (nothing usable was produced -- the report still says why).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.pipeline.artifacts import (  # noqa: E402
    write_cameras_json,
    write_mesh_ply_from_artifact,
    write_points_ply,
)
from engine.pipeline.dataset import load_capture_dataset  # noqa: E402
from engine.pipeline.vertical_slice import (  # noqa: E402
    VerticalSliceError,
    VerticalSliceOptions,
    vertical_slice,
)
from world_ir.artifact_store import FileArtifactStore  # noqa: E402


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    dataset = Path(sys.argv[1]).resolve()
    out = (
        Path(sys.argv[2]).resolve()
        if len(sys.argv) > 2
        else dataset / "pipeline_out"
    )
    out.mkdir(parents=True, exist_ok=True)

    items, refs, intrinsics, image_size = load_capture_dataset(dataset)
    report: dict = {
        "dataset": str(dataset),
        "images_ingested": len(items),
        "scale_references": len(refs),
        "stages": {},
    }
    print(f"ingested {len(items)} images, {len(refs)} measured baselines")

    options = VerticalSliceOptions(
        measured_baselines=refs,
        intrinsics=intrinsics,
        image_size=image_size,
        colmap_binary=os.environ.get("REALITY_COLMAP", "colmap"),
        # env knob: REALITY_DEPTH_MODEL="" disables the depth stage
        depth_model=os.environ.get("REALITY_DEPTH_MODEL", "DPT_Hybrid") or None,
        depth_stride=int(os.environ.get("REALITY_DEPTH_STRIDE", "16")),
        # env knobs: REALITY_MESH="" disables surface reconstruction;
        # REALITY_MESH_VOXEL_M / REALITY_MESH_DEPTH tune it
        mesh_enabled=bool(os.environ.get("REALITY_MESH", "1")),
        mesh_voxel_size_m=float(os.environ.get("REALITY_MESH_VOXEL_M", "0.02")),
        mesh_poisson_depth=int(os.environ.get("REALITY_MESH_DEPTH", "10")),
        artifact_store=FileArtifactStore(out / "artifacts"),
    )

    started = time.perf_counter()
    try:
        result = vertical_slice(items, options)
    except VerticalSliceError as exc:
        report["status"] = "FAILED"
        report["error"] = str(exc)
        report["runtime_s"] = round(time.perf_counter() - started, 2)
        (out / "report.json").write_text(json.dumps(report, indent=2))
        print(f"FAILED: {exc}")
        return 1
    runtime = round(time.perf_counter() - started, 2)

    report["status"] = (
        "SUCCESS" if result.registration_status == "success" else "PARTIAL_SUCCESS"
    )
    report["runtime_s"] = runtime
    report["stages"] = {
        "reconstruction": {
            "backend": result.stage_facts.get("backend"),
            "cameras_registered": result.cameras_registered,
            "cameras_input": result.cameras_input,
            "registration_status": result.registration_status,
            "sparse_points": result.points_total,
        },
        "scale": {
            "state": result.scale_state,
            "meters_per_unit": result.meters_per_unit,
            "note": result.scale_note,
        },
        "depth": result.stage_facts.get("depth"),
        "perception": result.stage_facts.get("perception"),
        "mesh": result.stage_facts.get("mesh"),
        "compile": {
            "entities": len(result.world.entities),
            "measurements": result.compile.measurements_count,
            "relationships": result.compile.relationships_count,
        },
    }
    report["world_id"] = result.world_id

    world_dict = result.world.to_dict()
    (out / "worldir.json").write_text(json.dumps(world_dict, indent=2))
    (out / "report.json").write_text(json.dumps(report, indent=2))

    # Inspectable geometry artifacts: the real reconstructed points and
    # camera positions (Phase-20 outputs), consumed by the Studio viewer.
    write_points_ply(out / "points.ply", result.points)
    mesh_facts = result.stage_facts.get("mesh") or {}
    if mesh_facts.get("status") == "ran" and result.world is not None:
        from reconstruction.meshing.mesh import MeshData
        geom = result.world.geometries.get("geom-mesh-room")
        if geom is not None and geom.data_uri:
            mesh_bytes = options.artifact_store.get(geom.data_uri)
            (out / "mesh.ply").write_bytes(
                MeshData.from_bytes(mesh_bytes).to_ply_bytes()
            )
    write_cameras_json(
        out / "cameras.json", result.camera_poses, result.scale_state,
        options.image_size,
    )
    from engine.pipeline.artifacts import write_exports

    write_exports(world_dict, options.artifact_store, out)

    print(result.summary_text())
    print(f"artifacts -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
