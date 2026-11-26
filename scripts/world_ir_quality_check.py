"""WorldIR quality validation check (KiloCode RCPAS Workstream 4).

Runs the production WorldIR validation gate (`validate_world_ir`) on real
reconstructions produced by `vertical_slice` and reports whether the world
passes, with every issue classified as ERROR (blocks commit) or WARNING
(reported, never swallowed).

This verifies WORKSTREAM 4 (WorldIR completeness and consistency): the
compiled world is internally coherent -- room/wall/floor/ceiling
connectivity, door/window placement, duplicate and conflicting entities,
impossible spatial overlaps, occlusion consistency, cross-view identity
contradictions, disconnected geometry -- all checked honestly and reported.

Run: python scripts/world_ir_quality_check.py [--images N] [--out report.json]
"""
from __future__ import annotations

import argparse, json, sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.pipeline.vertical_slice import VerticalSliceOptions, vertical_slice
from evidence.session import EvidenceItem, EvidenceKind
from reconstruction.scale import ScaleReference
from world_ir.validation import validate_world_ir
from worldstore.store import WorldStore


DATASET = Path("datasets/room_capture")
IMAGES = DATASET / "images"
MANIFEST = DATASET / "manifest.json"


def load_manifest():
    return json.loads(MANIFEST.read_text())


def build_evidence(limit=None):
    imgs = sorted(IMAGES.glob("*.jpg"))
    if limit:
        imgs = imgs[:limit]
    return [EvidenceItem(
        id=p.name, kind=EvidenceKind.PHOTO,
        source_uri=p.resolve().as_posix(),
        metadata={"size": [1280, 960]},
    ) for p in imgs]


def main():
    ap = argparse.ArgumentParser(description="WorldIR quality validation check")
    ap.add_argument("--images", type=int, default=None, help="Number of images to use (default: all)")
    ap.add_argument("--out", type=str, default=None, help="Write JSON report to this file")
    args = ap.parse_args()

    manifest = load_manifest()
    items = build_evidence(limit=args.images)

    opts = VerticalSliceOptions(
        colmap_binary="colmap",
        reconstruction_backend=None,
        depth_model=None,
        perception_model=None,
        mesh_enabled=False,
        detail_enabled=False,
        dense_mvs_enabled=False,
        image_size=[1280, 960],
        scene_gate=False,
        measured_baselines=[
            ScaleReference(evidence_id_a=b["evidence_id_a"],
                           evidence_id_b=b["evidence_id_b"],
                           distance_m=b["distance_m"],
                           method="manual_measurement")
            for b in manifest["measured_baselines"]
        ],
    )

    res = vertical_slice(items, opts)
    report = validate_world_ir(res.world)

    errors = [i for i in report.issues if i.severity.value == "error"]
    warnings = [i for i in report.issues if i.severity.value == "warning"]

    out = {
        "dataset": manifest["dataset"],
        "evidence_kind": "real_photograph",
        "images": len(items),
        "cameras_registered": res.cameras_registered,
        "points_total": res.points_total,
        "registration_status": res.registration_status,
        "world_entities": len(res.world.entities),
        "world_valid": report.is_valid(),
        "total_issues": len(report.issues),
        "errors": len(errors),
        "warnings": len(warnings),
        "error_details": [str(i) for i in errors],
        "warning_details": [str(i) for i in warnings],
    }
    print(json.dumps(out, indent=2))

    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"\nWrote JSON report to {args.out}")

    return 0 if report.is_valid() else 1


if __name__ == "__main__":
    sys.exit(main())
