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

from engine.pipeline.vertical_slice import (  # noqa: E402
    VerticalSliceError,
    VerticalSliceOptions,
    vertical_slice,
)
from evidence.session import EvidenceKind, EvidenceItem  # noqa: E402
from provenance import Provenance  # noqa: E402
from reconstruction.scale import ScaleReference  # noqa: E402


def _uri(path: Path) -> str:
    return path.resolve().as_uri()


def load_evidence(dataset: Path):
    """Dataset folder -> EvidenceItems + scale references + intrinsics."""
    manifest = json.loads((dataset / "manifest.json").read_text())
    images_dir = dataset / "images"

    items = []
    for entry in manifest["images"]:
        img_path = images_dir / entry["file"]
        data = img_path.read_bytes()
        items.append(
            EvidenceItem(
                id=entry["file"].rsplit(".", 1)[0],
                kind=EvidenceKind.PHOTO,
                source_uri=_uri(img_path),
                sha256=hashlib.sha256(data).hexdigest(),
                metadata={
                    "file": entry["file"],
                    "dataset": manifest.get("dataset", "unknown"),
                },
                provenance=Provenance.OBSERVED,
            )
        )

    refs = [
        ScaleReference(
            evidence_id_a=b["evidence_id_a"],
            evidence_id_b=b["evidence_id_b"],
            distance_m=float(b["distance_m"]),
            method=manifest.get("scale_reference", {}).get(
                "method", "manual_measurement"
            ),
        )
        for b in manifest.get("measured_baselines", [])
    ]

    intr = manifest.get("intrinsics_px")
    intrinsics = (
        tuple(float(intr[k]) for k in ("fx", "fy", "cx", "cy"))
        if intr
        else None
    )
    size = manifest.get("image_size", [1280, 960])
    return items, refs, intrinsics, (int(size[0]), int(size[1]))


def _write_ply(path: Path, points) -> None:
    """Minimal binary PLY writer (float32 xyz): no dependencies, every
    real point preserved."""
    import struct

    n = len(points)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "end_header\n"
    ).encode("ascii")
    buf = bytearray(header)
    for p in points:
        buf += struct.pack("<3f", p[0], p[1], p[2])
    path.write_bytes(bytes(buf))


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

    items, refs, intrinsics, image_size = load_evidence(dataset)
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
    _write_ply(out / "points.ply", result.points)
    (out / "cameras.json").write_text(json.dumps({
        "frame": "world (meters, +Y up after frame canonicalization)",
        "scale_state": result.scale_state,
        "rotation_convention": "camera-to-world quaternion (w, x, y, z)",
        "image_size": list(options.image_size),
        "cameras": [
            {"evidence_id": eid, "position_m": list(pos), "rotation_wxyz": list(rot)}
            for eid, pos, rot in result.camera_poses
        ],
    }, indent=2))

    print(result.summary_text())
    print(f"artifacts -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
