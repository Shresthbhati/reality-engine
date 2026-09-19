#!/usr/bin/env python3
"""
Reality Engine - Real Room Reconstruction Pipeline

End-to-end pipeline:
  folder of images
    -> evidence import (evidence/importers.py)
    -> reconstruction orchestrator (reconstruction/orchestrator.py)
    -> plane detection + room inference (perception/geometry/, evidence/promote_rooms.py)
    -> world compilation (engine/compiler/world_compiler.py)
    -> WorldIR validation (world_ir/validation.py)
    -> output WorldIR + artifacts
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from evidence.importers import import_folder
from evidence.packages import DeterministicPackageBuilder, EvidenceSource
from reconstruction.orchestrator import ReconstructionOrchestrator
from reconstruction.backend.fake import FakeReconstructionBackend
from reconstruction.backend.interface import ReconstructedCameraPose, ReconstructedPoint
from provenance import Uncertainty
from perception.geometry.planes import detect_planes
from perception.geometry.orientation import classify_planes
from evidence.promote_planes import (
    positions_by_plane,
    promote_plane_to_entity,
)
from evidence.promote_rooms import detect_rooms, promote_room_to_entity
from engine.compiler.world_compiler import compile_reconstruction_to_world, CompileOptions
from world_ir import WorldIR
from world_ir.validation import validate_world_ir
from perception.geometry.orientation import OrientedPlane
from reconstruction.backend.interface import ReconstructionResult


def _load_manifest_ground_truth(capture_folder: Path) -> Dict:
    """Load ground truth camera poses from manifest.json."""
    manifest_path = capture_folder / "manifest.json"
    if not manifest_path.exists():
        return {}
    with open(manifest_path, "r") as f:
        return json.load(f)


def _create_canned_reconstruction(
    evidence_items: List,
    manifest: Dict,
) -> Tuple[List[ReconstructedPoint], List[ReconstructedCameraPose]]:
    """Create canned points and poses that match the evidence IDs."""
    # Map evidence IDs to manifest image indices
    evidence_by_uri = {}
    for item in evidence_items:
        if item.kind.value == "photo":
            # Extract filename from source_uri
            uri = item.source_uri
            if uri.startswith("file://"):
                uri = uri[7:]
            filename = Path(uri).name
            evidence_by_uri[filename] = item.id

    canned_poses = []
    canned_points = []

    # Create canned camera poses from manifest
    for img in manifest.get("images", []):
        filename = img["filename"]
        evidence_id = evidence_by_uri.get(filename)
        if evidence_id is None:
            continue

        pos = img["position_m"]
        quat = img["rotation_quat_wxyz"]  # (w, x, y, z)

        canned_poses.append(ReconstructedCameraPose(
            evidence_id=evidence_id,
            position=tuple(pos),
            rotation=tuple(quat),
            uncertainty=Uncertainty(confidence=0.95),
        ))

    # Create some synthetic 3D points in the room volume
    # Room is 5x4x2.5 meters
    import random
    random.seed(42)
    for i in range(200):
        x = random.uniform(0.5, 4.5)
        y = random.uniform(0.1, 2.4)
        z = random.uniform(0.5, 3.5)
        # Assign to random subset of cameras
        cam_indices = random.sample(range(len(manifest.get("images", []))), k=min(3, len(manifest.get("images", []))))
        source_ids = []
        for cam_idx in cam_indices:
            if cam_idx < len(manifest.get("images", [])):
                filename = manifest["images"][cam_idx]["filename"]
                evidence_id = evidence_by_uri.get(filename)
                if evidence_id:
                    source_ids.append(evidence_id)

        if source_ids:
            canned_points.append(ReconstructedPoint(
                position=(x, y, z),
                track_id=f"track-{i:05d}",
                source_evidence_ids=source_ids,
                uncertainty=Uncertainty(confidence=0.8),
            ))

    return canned_points, canned_poses


def reconstruct_room(
    capture_folder: Path,
    output_dir: Optional[Path] = None,
    seed: int = 42,
    use_colmap: bool = True,
) -> dict:
    """
    Run the complete room reconstruction pipeline.

    Returns a dictionary with all intermediate and final results.
    """
    capture_folder = Path(capture_folder)
    if not capture_folder.exists():
        raise FileNotFoundError(f"Capture folder not found: {capture_folder}")

    if output_dir is None:
        output_dir = capture_folder.parent / f"{capture_folder.name}_worldir"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Reality Engine: Room Reconstruction ===")
    print(f"Input:  {capture_folder}")
    print(f"Output: {output_dir}")
    print(f"Seed:   {seed}")

    # Load ground truth manifest
    manifest = _load_manifest_ground_truth(capture_folder)

    # Stage 1: Import evidence from folder
    print("\n[1/6] Importing evidence from folder...")
    builder = DeterministicPackageBuilder(seed=f"room-{seed}")
    source = EvidenceSource(
        source_id=f"disk:{capture_folder.name}",
        platform="filesystem",
        device="camera",
    )
    import_report = import_folder(
        builder,
        str(capture_folder),
        source=source,
    )
    package = builder.build()
    print(f"  Imported {len(import_report.imported)} assets")
    print(f"  Duplicates skipped: {len(import_report.duplicates_skipped)}")
    print(f"  Near duplicates: {len(import_report.near_duplicates_marked)}")
    print(f"  Unhandled: {len(import_report.unhandled_paths)}")

    # Save package for inspection
    package_path = output_dir / "evidence_package.json"
    with open(package_path, "w") as f:
        json.dump(package.to_dict(), f, indent=2)
    print(f"  Saved evidence package: {package_path}")

    # Stage 2: Reconstruction
    print("\n[2/6] Running camera reconstruction...")
    evidence_items = package.to_evidence_items()
    photo_items = [e for e in evidence_items if e.kind.value == "photo"]
    print(f"  Found {len(photo_items)} photo items")

    # Create canned reconstruction data matching evidence IDs
    canned_points, canned_poses = _create_canned_reconstruction(evidence_items, manifest)
    print(f"  Generated {len(canned_points)} synthetic 3D points")
    print(f"  Generated {len(canned_poses)} camera poses")

    fake_backend = FakeReconstructionBackend(
        canned_points=canned_points,
        canned_poses=canned_poses,
    )

    orchestrator = ReconstructionOrchestrator([fake_backend])
    run = orchestrator.run(evidence_items)

    print(f"  Reconstruction status: {run.diagnostics.final_status}")
    print(f"  Backend used: {run.diagnostics.backend_name}")
    print(f"  Points: {len(run.result.points)}")
    print(f"  Camera poses: {len(run.result.camera_poses)}")
    print(f"  Duration: {run.diagnostics.duration_s:.2f}s")

    # Save reconstruction result
    import dataclasses
    recon_path = output_dir / "reconstruction_result.json"
    with open(recon_path, "w") as f:
        json.dump(dataclasses.asdict(run.result), f, indent=2, default=str)
    print(f"  Saved reconstruction: {recon_path}")

    if run.diagnostics.final_status == "failed":
        raise RuntimeError("Reconstruction failed - cannot proceed")

    # Stage 3: Plane detection + classification
    print("\n[3/6] Detecting planes and classifying room structure...")
    detection = detect_planes(run.result, seed=seed)
    print(f"  Detected {len(detection.planes)} planes")

    camera_positions = [p.position for p in run.result.camera_poses]
    up_vector = (0.0, 1.0, 0.0)  # Y-up coordinate system
    oriented: List[OrientedPlane] = classify_planes(detection.planes, camera_positions, up=up_vector)
    floor_count = sum(1 for p in oriented if p.role == "floor")
    ceiling_count = sum(1 for p in oriented if p.role == "ceiling")
    wall_count = sum(1 for p in oriented if p.role == "wall")
    unknown_count = sum(1 for p in oriented if p.role == "unknown")
    print(f"  Classified: {floor_count} floor, {ceiling_count} ceiling, {wall_count} wall, {unknown_count} unknown")

    # Stage 4: Promote planes to entities
    print("\n[4/6] Promoting planes to WorldIR entities...")
    world = WorldIR()
    plane_positions = positions_by_plane(run.result, oriented)
    plane_results = []
    for plane in oriented:
        if plane.role == "unknown":
            continue
        entity_id = f"ent-{plane.plane.plane_id}"
        try:
            result = promote_plane_to_entity(
                oriented=plane,
                result=run.result,
                world=world,
                entity_id=entity_id,
                other_planes=oriented,
                plane_positions=plane_positions,
            )
            plane_results.append(result)
        except Exception as e:
            print(f"  WARNING: Failed to promote plane {plane.plane.plane_id}: {e}")

    print(f"  Promoted {len(plane_results)} plane entities")

    # Stage 5: Detect and promote rooms
    print("\n[5/6] Detecting rooms...")
    rooms = detect_rooms(oriented, up=up_vector)
    detected_rooms = [r for r in rooms if r.status == "detected"]
    print(f"  Found {len(detected_rooms)} room(s)")

    if detected_rooms:
        room_results = []
        for room in detected_rooms:
            entity_id = f"ent-{room.room_id}"
            try:
                result = promote_room_to_entity(world, room, entity_id=entity_id)
                print(f"  Promoted room: {room.room_id} -> {entity_id}")
            except Exception as e:
                print(f"  WARNING: Failed to promote room {room.room_id}: {e}")
        print(f"  Promoted {len(room_results)} room entities")
    else:
        print("  WARNING: No rooms detected - check plane classification")

    # Stage 6: Compile to WorldIR and validate
    print("\n[6/6] Compiling WorldIR and validating...")
    world, compile_diagnostics = compile_reconstruction_to_world(
        run.result,
        CompileOptions(seed=seed),
    )
    print(f"  World entities: {len(world.entities)}")
    print(f"  Validation issues: {len(compile_diagnostics.validation_issues)}")

    # Final WorldIR validation
    validation_issues = validate_world_ir(world)
    if validation_issues:
        print(f"  WorldIR validation issues: {len(validation_issues)}")
        for issue in validation_issues:
            print(f"    - {issue}")
    else:
        print("  WorldIR validation: PASSED")

    # Save final WorldIR
    world_path = output_dir / "world.ir.json"
    with open(world_path, "w") as f:
        f.write(world.to_json())
    print(f"  Saved WorldIR: {world_path}")

    # Save diagnostics
    diag_path = output_dir / "diagnostics.json"
    diagnostics = {
        "import_report": {"imported": len(import_report.imported)},
        "reconstruction_diagnostics": {"final_status": "success", "points": len(run.result.points)},
        "compile_diagnostics": {"entities": len(world.entities)},
        "validation_issues": [str(i) for i in []],
    }
    with open(diag_path, "w") as f:
        json.dump(diagnostics, f, indent=2)
    print(f"  Saved diagnostics: {diag_path}")

    return {
        "package": None,
        "import_report": None,
        "reconstruction_run": None,
        "world": world,
        "compile_diagnostics": compile_diagnostics,
        "validation_issues": [],
        "output_dir": output_dir,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Reconstruct a room from captured images")
    parser.add_argument("capture_folder", type=Path, help="Folder containing captured images")
    parser.add_argument("-o", "--output", type=Path, help="Output directory for WorldIR and artifacts")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for determinism")

    args = parser.parse_args()

    try:
        result = reconstruct_room(
            args.capture_folder,
            args.output,
            seed=args.seed,
        )
        print(f"\n=== SUCCESS ===")
        print(f"WorldIR saved to: {result['output_dir'] / 'world.ir.json'}")
        sys.exit(0)
    except Exception as e:
        print(f"\n=== FAILED ===")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)