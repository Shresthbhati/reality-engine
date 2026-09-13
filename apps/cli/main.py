"""``reality`` command-line interface -- a thin client of `sdk.reality`.

This closes the biggest named gap from the studio campaign (Phase 17,
"headless workflow"): before this module, every capability in the SDK
(compile, validate, diff, physics, export) was reachable only from
Python. This gives the real end-to-end vertical slice

    photos on disk -> evidence package -> reconstruction -> WorldIR
    -> validate / diff / export / physics

a command-line entry point, with every subcommand a direct call into
`sdk.reality` / `evidence.packages` / `reconstruction.orchestrator` --
no separate reimplementation, no fake runtime.

Honesty rules carried over from the engine: a reconstruction backend
that has no real geometry to offer (no COLMAP install, no canned fake
data) FAILS loudly (`reconstruct` exits non-zero with the orchestrator's
full attempt log) rather than fabricating a world. `compile` refuses the
same way for an empty/failed reconstruction. Nothing here invents
geometry the underlying layers didn't produce.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from evidence.importers import import_folder
from evidence.packages import DeterministicPackageBuilder, EvidencePackage
from reconstruction.backend.colmap_backend import ColmapReconstructionBackend
from reconstruction.backend.fake import FakeReconstructionBackend
from reconstruction.orchestrator import (
    EvidenceValidationError,
    ReconstructionOrchestrationError,
    ReconstructionOrchestrator,
)
from sdk import reality
from world_ir.artifact_store import FileArtifactStore
from world_ir.world_v1 import WorldIR


def _eprint(*args: object) -> None:
    print(*args, file=sys.stderr)


def _load_world(path: str) -> WorldIR:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return WorldIR.from_dict(data)


def _save_world(world: WorldIR, path: str) -> None:
    Path(path).write_text(json.dumps(world.to_dict(), indent=2, sort_keys=True), encoding="utf-8")


def _artifacts_dir_for(world_path: str) -> Path:
    """Convention: `world.json` -> `world.json.artifacts/`, alongside it."""
    return Path(f"{world_path}.artifacts")


def _default_backends(colmap_binary: str, use_gpu: bool) -> List:
    """Real backend first, honest-failure fake second.

    The fake backend has no canned data here, so it always declines/fails
    for real photo folders -- it exists only so the orchestrator's
    fallback path is exercised the same way tests exercise it, not as a
    disguised source of invented geometry.
    """
    return [
        ColmapReconstructionBackend(colmap_binary=colmap_binary, use_gpu=use_gpu),
        FakeReconstructionBackend(),
    ]


def cmd_ingest(args: argparse.Namespace) -> int:
    builder = DeterministicPackageBuilder(seed=args.seed)
    report = import_folder(builder, args.folder)
    package = builder.build(package_id=args.package_id or f"pkg-{args.seed}")
    Path(args.output).write_text(json.dumps(package.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    print(f"ingested {len(report.imported)} asset(s), {len(report.unhandled_paths)} unhandled path(s)")
    if report.unhandled_paths:
        for path in report.unhandled_paths:
            _eprint(f"  unhandled: {path}")
    print(f"wrote package '{package.package_id}' -> {args.output}")
    return 0


def cmd_reconstruct(args: argparse.Namespace) -> int:
    package = EvidencePackage.from_dict(json.loads(Path(args.package).read_text(encoding="utf-8")))
    evidence = package.to_evidence_items()

    orchestrator = ReconstructionOrchestrator(_default_backends(args.colmap_binary, args.gpu))
    try:
        run = orchestrator.run(evidence)
    except (EvidenceValidationError, ReconstructionOrchestrationError) as exc:
        _eprint(f"reconstruction refused: {exc}")
        if isinstance(exc, ReconstructionOrchestrationError):
            for attempt in exc.attempts:
                _eprint(f"  {attempt.backend_name}: {attempt.outcome} - {attempt.detail or attempt.error}")
        return 1

    compile_options = None
    if not args.no_real_geometry:
        store_root = _artifacts_dir_for(args.output)
        compile_options = reality.CompileOptions(artifact_store=FileArtifactStore(store_root))

    try:
        world, diagnostics = reality.compile_world_from_reconstruction(run.result, compile_options)
    except (reality.CompileInputError, reality.WorldValidationGateError) as exc:
        _eprint(f"compile refused: {exc}")
        return 1

    _save_world(world, args.output)
    print(diagnostics.summary_text())
    print(f"wrote world '{world.id}' -> {args.output}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    world = _load_world(args.world)
    report = reality.validate(world)
    for issue in report.issues:
        print(f"[{issue.severity.name}] {issue.message}")
    print(f"valid: {report.is_valid()} ({len(report.issues)} issue(s))")
    return 0 if report.is_valid() else 1


def cmd_diff(args: argparse.Namespace) -> int:
    before = _load_world(args.before)
    after = _load_world(args.after)
    world_diff = reality.diff(before, after)
    print(json.dumps(world_diff.to_dict(), indent=2, sort_keys=True))
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    world = _load_world(args.world)
    artifact_store = None
    if args.format == "gltf":
        store_root = _artifacts_dir_for(args.world)
        if store_root.is_dir():
            artifact_store = FileArtifactStore(store_root)
    try:
        content, report = reality.export(world, args.format, artifact_store=artifact_store)
    except reality.UnsupportedExportFormatError as exc:
        _eprint(str(exc))
        return 1
    if isinstance(content, dict):
        text = json.dumps(content, indent=2, sort_keys=True)
    else:
        text = content
    mode = "w" if isinstance(text, str) else "wb"
    with open(args.output, mode, encoding=None if mode == "wb" else "utf-8") as f:
        f.write(text)
    print(f"exported {args.format} -> {args.output} "
          f"({len(report.entities_exported)} entities, {len(report.entities_skipped)} skipped, "
          f"hash={report.content_hash[:12]})")
    for entity_id, reason in zip(report.entities_skipped, report.skip_reasons):
        _eprint(f"  skipped {entity_id}: {reason}")
    return 0


def cmd_physics(args: argparse.Namespace) -> int:
    world = _load_world(args.world)
    diagnostics = reality.compile_physics(world)
    print(json.dumps(diagnostics.to_dict(), indent=2, sort_keys=True))
    return 0


def cmd_query_nearest(args: argparse.Namespace) -> int:
    world = _load_world(args.world)
    index = reality.spatial_index(world)
    hits = index.nearest((args.x, args.y, args.z), k=args.k)
    if not hits:
        print("no localized entities found")
        return 1
    for entity, distance in hits:
        print(f"{entity.id}\t{distance:.4f}m\t{entity.name or entity.type.value}")
    return 0


def cmd_query_contents(args: argparse.Namespace) -> int:
    world = _load_world(args.world)
    graph = reality.scene_graph(world)
    if args.world_entity_id not in world.entities:
        _eprint(f"unknown entity id: {args.world_entity_id!r}")
        return 1
    contents = graph.contents_of(args.world_entity_id)
    if not contents:
        print(f"{args.world_entity_id} has no known contents")
        return 0
    for entity in sorted(contents, key=lambda e: e.id):
        print(f"{entity.id}\t{entity.name or entity.type.value}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="reality", description="Reality Engine command-line interface")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="import a folder of photos/video/LAS into an evidence package")
    p_ingest.add_argument("folder")
    p_ingest.add_argument("-o", "--output", required=True, help="package JSON output path")
    p_ingest.add_argument("--seed", default="pkg", help="deterministic id seed (default: pkg)")
    p_ingest.add_argument("--package-id", default="", help="explicit package id (default: pkg-<seed>)")
    p_ingest.set_defaults(func=cmd_ingest)

    p_recon = sub.add_parser("reconstruct", help="evidence package -> reconstruction -> compiled WorldIR")
    p_recon.add_argument("package", help="package JSON produced by `ingest`")
    p_recon.add_argument("-o", "--output", required=True, help="world JSON output path")
    p_recon.add_argument("--colmap-binary", default="colmap")
    p_recon.add_argument("--gpu", action="store_true")
    p_recon.add_argument(
        "--no-real-geometry", action="store_true",
        help="skip storing real plane point-cloud geometry (no <output>.artifacts/ dir written); "
             "produces a smaller world.json with no Geometry.data_uri set, matching pre-artifact-store behavior",
    )
    p_recon.set_defaults(func=cmd_reconstruct)

    p_validate = sub.add_parser("validate", help="validate a compiled WorldIR")
    p_validate.add_argument("world", help="world JSON path")
    p_validate.set_defaults(func=cmd_validate)

    p_diff = sub.add_parser("diff", help="structural diff between two WorldIR snapshots")
    p_diff.add_argument("before")
    p_diff.add_argument("after")
    p_diff.set_defaults(func=cmd_diff)

    p_export = sub.add_parser(
        "export",
        help="export a WorldIR to gltf/usda/blender (gltf reconnects to <world>.artifacts/ "
             "for real geometry, if it exists)",
    )
    p_export.add_argument("world")
    p_export.add_argument("--format", required=True, choices=["gltf", "usda", "blender"])
    p_export.add_argument("-o", "--output", required=True)
    p_export.set_defaults(func=cmd_export)

    p_physics = sub.add_parser("physics", help="compile a WorldIR into physics bodies + diagnostics")
    p_physics.add_argument("world")
    p_physics.set_defaults(func=cmd_physics)

    p_query = sub.add_parser("query", help="spatial/relationship queries over a WorldIR")
    query_sub = p_query.add_subparsers(dest="query_command", required=True)

    p_nearest = query_sub.add_parser("nearest", help="k nearest localized entities to a point")
    p_nearest.add_argument("world")
    p_nearest.add_argument("x", type=float)
    p_nearest.add_argument("y", type=float)
    p_nearest.add_argument("z", type=float)
    p_nearest.add_argument("--k", type=int, default=1)
    p_nearest.set_defaults(func=cmd_query_nearest)

    p_contents = query_sub.add_parser("contents", help="entities contained in/part of a container entity")
    p_contents.add_argument("world")
    p_contents.add_argument("world_entity_id", metavar="entity-id")
    p_contents.set_defaults(func=cmd_query_contents)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
