"""Additional CLI commands (compile, validate, inspect, export)."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _print_json(data: dict) -> None:
    print(json.dumps(data, indent=2, default=str))


def _print_human(header: str, lines: list[str]) -> None:
    print(f"\n=== {header} ===")
    for line in lines:
        print(f"  {line}")
    print()


def cmd_compile(args) -> int:
    """Compile a reconstruction result into WorldIR."""
    from reconstruction.backend.interface import (
        ReconstructedCameraPose, ReconstructedPoint, ReconstructionResult,
    )
    from engine.compiler import CompileOptions, compile_reconstruction_to_world

    recon_path = Path(args.reconstruction)
    if not recon_path.exists():
        print(f"error: {recon_path} not found", file=sys.stderr)
        return 1

    with open(recon_path) as f:
        data = json.load(f)

    points = [
        ReconstructedPoint(
            position=tuple(p["position"]),
            track_id=p["track_id"],
            source_evidence_ids=p.get("source_evidence_ids", []),
        )
        for p in data.get("points", [])
    ]
    poses = [
        ReconstructedCameraPose(
            evidence_id=p["evidence_id"],
            position=tuple(p["position"]),
            rotation=tuple(p["rotation"]),
        )
        for p in data.get("camera_poses", [])
    ]
    result = ReconstructionResult(
        points=points, camera_poses=poses,
        registration_status=data.get("registration_status", "unknown"),
    )

    options = CompileOptions(seed=args.seed, up=tuple(args.up))
    try:
        world, diagnostics = compile_reconstruction_to_world(result, options)
    except Exception as exc:
        if args.json:
            _print_json({"status": "failed", "error": str(exc)})
        else:
            print(f"\nCompile failed: {exc}", file=sys.stderr)
        return 1

    out_dir = Path(args.output) if args.output else recon_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    world_path = out_dir / "world.json"
    with open(world_path, "w") as f:
        json.dump(world.to_dict(), f, indent=2, default=str)

    if args.json:
        _print_json({
            "status": "success",
            "entities": len(world.entities),
            "rooms": diagnostics.rooms_detected,
            "planes": diagnostics.planes_total,
            "measurements": diagnostics.measurements_count,
            "relationships": diagnostics.relationships_count,
            "validation_issues": len(diagnostics.validation_issues),
            "output": str(world_path),
        })
    else:
        _print_human("Compile Results", [
            f"Entities:     {len(world.entities)}",
            f"Rooms:        {diagnostics.rooms_detected}",
            f"Planes:       {diagnostics.planes_total} ({diagnostics.planes_by_role})",
            f"Measurements: {diagnostics.measurements_count}",
            f"Relationships:{diagnostics.relationships_count}",
            f"Validation:   {len(diagnostics.validation_issues)} issue(s)",
            f"Output:       {world_path}",
        ])
def cmd_validate(args) -> int:
    """Run the WorldIR validation gate on a saved world."""
    from world_ir.validation import validate_world_ir
    from world_ir import WorldIR

    with open(args.world) as f:
        data = json.load(f)
    world = WorldIR.from_dict(data)
    report = validate_world_ir(world)

    if args.json:
        _print_json({
            "valid": report.is_valid(),
            "errors": [{"code": i.code, "message": i.message} for i in report.errors],
            "warnings": [{"code": i.code, "message": i.message} for i in report.warnings],
        })
    else:
        if report.is_valid():
            print("\n=== World Validation: PASSED ===")
        else:
            print(f"\n=== World Validation: FAILED ({len(report.errors)} errors) ===")
            for issue in report.errors:
                print(f"  [ERROR] {issue.code}: {issue.message}")
        if report.warnings:
            print(f"  ({len(report.warnings)} warning(s))")
            for issue in report.warnings:
                print(f"  [WARN] {issue.code}: {issue.message}")
    return 0 if report.is_valid() else 1
def cmd_inspect(args) -> int:
    """Print a human-readable (or JSON) summary of a compiled world."""
    from world_ir import WorldIR

    with open(args.world) as f:
        data = json.load(f)
    world = WorldIR.from_dict(data)

    if args.entity:
        entity = world.entities.get(args.entity)
        if entity is None:
            print(f"error: no entity '{args.entity}'", file=sys.stderr)
            return 1
        if args.json:
            _print_json(entity.to_dict())
        else:
            print(f"\nEntity: {entity.id} ({entity.type.value})")
            print(f"  Name:        {entity.name}")
            print(f"  Provenance:  {entity.provenance.value}")
            print(f"  Confidence:  {entity.confidence}")
            print(f"  Geometry:    {entity.geometry_ids}")
            print(f"  Relationships: {len(entity.relationships)}")
            for r in entity.relationships:
                print(f"    {r.kind.value} -> {r.target_id} (conf={r.confidence:.2f})")
            for key, val in entity.custom_properties.items():
                print(f"  {key}: {val}")
        return 0

    by_type: dict[str, int] = {}
    for e in world.entities.values():
        by_type[e.type.value] = by_type.get(e.type.value, 0) + 1

    if args.json:
        _print_json({
            "entity_count": len(world.entities),
            "by_type": by_type,
def cmd_export(args) -> int:
    """Export a world to glTF / USD / JSON."""
    from world_ir import WorldIR

    with open(args.world) as f:
        data = json.load(f)
    world = WorldIR.from_dict(data)
    fmt = args.format.lower()
    out_dir = Path(args.output) if args.output else Path(args.world).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    if fmt == "gltf" or fmt == "glb":
        from exporters.gltf.exporter import export_gltf
        out_path = out_dir / "world.gltf"
        export_gltf(world, str(out_path))
    elif fmt == "usd" or fmt == "usda":
        from exporters.usd.exporter import export_usd
        out_path = out_dir / "world.usda"
        export_usd(world, str(out_path))
    elif fmt == "json":
        out_path = out_dir / "world.json"
        with open(out_path, "w") as f:
            json.dump(world.to_dict(), f, indent=2, default=str)
    else:
        print(f"error: unknown format '{fmt}' (supported: gltf, usd, json)", file=sys.stderr)
        return 1

    if args.json:
        _print_json({"status": "success", "format": fmt, "output": str(out_path)})
    else:
        _print_human("Export Results", [
            f"Format: {fmt}",
            f"Output: {out_path}",
        ])
    return 0

            "geometries": len(world.geometries),
            "global_provenance": world.global_provenance.value,
            "version": world.version,
        })
    else:
        _print_human("World Summary", [
            f"Entities:   {len(world.entities)}",
            f"Geometries: {len(world.geometries)}",
            f"Provenance: {world.global_provenance.value}",
            f"Version:    {world.version}",
        ])
        if by_type:
            print("  By type:")
            for t, n in sorted(by_type.items()):
                print(f"    {t}: {n}")
    return 0


    return 0
