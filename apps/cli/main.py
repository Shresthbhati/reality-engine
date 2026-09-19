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
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

from evidence.importers import import_folder
from evidence.multi_source import MultiSourceSession
from evidence.packages import DeterministicPackageBuilder, EvidencePackage
from reconstruction.backend.colmap_backend import ColmapReconstructionBackend
from reconstruction.backend.fake import FakeReconstructionBackend
from reconstruction.orchestrator import (
    EvidenceValidationError,
    ReconstructionOrchestrationError,
    ReconstructionOrchestrator,
)
from registration import (
    RegistrationEngine,
    RegistrationResult,
)
from engine.math import Vec3
from reconstruction.calibration.transforms import RigidTransform
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


def _session_path(session_dir: str) -> Path:
    return Path(session_dir) / "session.json"


def _load_session(session_dir: str) -> MultiSourceSession:
    path = _session_path(session_dir)
    if not path.is_file():
        raise FileNotFoundError(f"no session.json under {session_dir!r} -- run `reality session create` first")
    return MultiSourceSession.from_dict(json.loads(path.read_text(encoding="utf-8")))


def _save_session(session: MultiSourceSession, session_dir: str) -> None:
    Path(session_dir).mkdir(parents=True, exist_ok=True)
    _session_path(session_dir).write_text(
        json.dumps(session.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )


def cmd_session_create(args: argparse.Namespace) -> int:
    path = _session_path(args.session_dir)
    if path.is_file():
        _eprint(f"session already exists at {path} -- use `session add-source` to add evidence to it")
        return 1
    session = MultiSourceSession(session_id=args.session_id, name=args.name or args.session_id)
    _save_session(session, args.session_dir)
    print(f"created session '{session.session_id}' -> {path}")
    return 0


def cmd_source_add(args: argparse.Namespace) -> int:
    session = _load_session(args.session_dir)
    record = session.add_source(args.path, capture_type=args.capture_type)
    _save_session(session, args.session_dir)
    print(f"{record.status.value}\t{record.source_id}\t{record.source_type.value}\t{args.path}")
    if record.error:
        _eprint(f"  {record.error}")
    if record.unhandled_paths:
        for unhandled in record.unhandled_paths:
            _eprint(f"  unhandled: {unhandled}")
    return 0 if record.status.value in ("ingested", "already_ingested") else 1


def cmd_source_list(args: argparse.Namespace) -> int:
    session = _load_session(args.session_dir)
    for record in session.sources():
        print(f"{record.source_id}\t{record.status.value}\t{record.source_type.value}\t"
              f"{len(record.asset_ids)} asset(s)\t{record.original_path}")
    print(f"{len(session.sources())} source(s), {len(session.package.all_assets())} asset(s) total")
    return 0


def cmd_session_inspect(args: argparse.Namespace) -> int:
    session = _load_session(args.session_dir)
    summary = session.evidence_summary()
    print(f"session '{session.session_id}' ({session.name})")
    print(f"  sources: {summary['source_count']}")
    for kind, count in sorted(summary["asset_counts"].items()):
        print(f"    {kind}: {count}")
    print(f"  assets with GPS: {summary['gps_asset_count']}")
    print(f"  ready for reconstruction: {'YES' if summary['ready_for_reconstruction'] else 'NO'}")
    for issue in summary["readiness_issues"]:
        _eprint(f"    - {issue}")
    return 0


def cmd_source_inspect(args: argparse.Namespace) -> int:
    session = _load_session(args.session_dir)
    matches = [s for s in session.sources() if s.source_id == args.source_id]
    if not matches:
        _eprint(f"unknown source id: {args.source_id!r}")
        return 1
    record = matches[0]
    print(f"source '{record.source_id}'")
    print(f"  path: {record.original_path}")
    print(f"  type: {record.source_type.value}")
    print(f"  status: {record.status.value}")
    print(f"  content hash: {record.content_hash}")
    print(f"  registration: {record.registration['status']}")
    print(f"  asset(s): {len(record.asset_ids)}")
    for asset_id in record.asset_ids:
        print(f"    {asset_id}")
    if record.components:
        print(f"  component(s): {len(record.components)}")
        for component, paths in sorted(record.components.items()):
            print(f"    {component}: {len(paths)} file(s)")
    if record.unhandled_paths:
        print(f"  unhandled path(s): {len(record.unhandled_paths)}")
        for path in record.unhandled_paths:
            print(f"    {path}")
    if record.error:
        print(f"  error: {record.error}")
    return 0


def cmd_session_export_package(args: argparse.Namespace) -> int:
    session = _load_session(args.session_dir)
    Path(args.output).write_text(
        json.dumps(session.package.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"exported package '{session.package.package_id}' "
          f"({len(session.package.all_assets())} asset(s)) -> {args.output}")
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


def _resolve_test_backend():
    """REALITY_TEST_BACKEND="module.path:ClassName" -> backend instance.

    Deterministic-backend seam for offline tests of `compile`; unset in
    production, where the real COLMAP backend is always used. Raises a
    clear error if set but unresolvable -- never silently ignored.
    """
    spec = os.environ.get("REALITY_TEST_BACKEND", "").strip()
    if not spec:
        return None
    module_name, _, class_name = spec.partition(":")
    if not module_name or not class_name:
        raise ValueError(
            f"REALITY_TEST_BACKEND must be 'module:Class', got {spec!r}"
        )
    import importlib

    module = importlib.import_module(module_name)
    return getattr(module, class_name)()


def cmd_compile(args: argparse.Namespace) -> int:
    """One-command mapping path: capture dataset -> full vertical slice.

    Runs the same pipeline as the flagship runner (SfM -> metric scale
    -> frame canonicalization -> depth -> perception -> mesh -> WorldIR
    -> validation gate), writes the machine-readable output set
    (worldir.json / report.json / points.ply / mesh.ply / cameras.json
    / artifacts/ / exports/scene.gltf), and prints the per-stage report.
    Exit 0 on honest success OR partial-with-world; 1 when a stage
    refuses -- the report still says why.
    """
    from engine.pipeline.artifacts import (
        write_cameras_json,
        write_exports,
        write_mesh_ply_from_artifact,
        write_points_ply,
    )
    from engine.pipeline.dataset import DatasetError, load_capture_dataset
    from engine.pipeline.vertical_slice import (
        VerticalSliceError,
        VerticalSliceOptions,
        vertical_slice,
    )

    dataset = Path(args.dataset).resolve()
    out = Path(args.output).resolve() if args.output else dataset / "pipeline_out"
    out.mkdir(parents=True, exist_ok=True)

    try:
        items, refs, intrinsics, image_size = load_capture_dataset(dataset)
    except DatasetError as exc:
        _eprint(f"dataset rejected: {exc}")
        return 1

    store = FileArtifactStore(out / "artifacts")
    backend = _resolve_test_backend()
    options = VerticalSliceOptions(
        measured_baselines=refs,
        intrinsics=intrinsics,
        image_size=image_size,
        colmap_binary=args.colmap_binary,
        depth_model=None if args.no_depth else "DPT_Hybrid",
        mesh_enabled=not args.no_mesh,
        artifact_store=store,
        reconstruction_backend=backend,
    )

    report: dict = {
        "dataset": str(dataset),
        "images_ingested": len(items),
        "scale_references": len(refs),
        "stages": {},
    }
    started = time.perf_counter()
    try:
        result = vertical_slice(items, options)
    except VerticalSliceError as exc:
        report["status"] = "FAILED"
        report["error"] = str(exc)
        report["runtime_s"] = round(time.perf_counter() - started, 2)
        (out / "report.json").write_text(json.dumps(report, indent=2))
        _eprint(f"compile failed: {exc}")
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
            "points": result.points_total,
        },
        "scale": {
            "state": result.scale_state,
            "meters_per_unit": result.meters_per_unit,
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
    report["outputs"] = {
        "world": str(out / "worldir.json"),
        "report": str(out / "report.json"),
        "points_ply": str(out / "points.ply"),
        "cameras": str(out / "cameras.json"),
        "artifacts": str(out / "artifacts"),
        "exports": str(out / "exports" / "scene.gltf"),
    }

    world_dict = result.world.to_dict()
    (out / "worldir.json").write_text(json.dumps(world_dict, indent=2))
    (out / "report.json").write_text(json.dumps(report, indent=2))
    write_points_ply(out / "points.ply", result.points)
    write_mesh_ply_from_artifact(out, store, result.stage_facts.get("mesh"))
    write_cameras_json(
        out / "cameras.json", result.camera_poses, result.scale_state,
        options.image_size,
    )
    exports_path = write_exports(world_dict, store, out)

    print(result.summary_text())
    print(f"world -> {out / 'worldir.json'}")
    print(f"exports -> {exports_path}")
    print(f"report -> {out / 'report.json'}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    world = _load_world(args.world)
    report = reality.validate(world)
    for issue in report.issues:
        print(f"[{issue.severity.name}] {issue.message}")
    print(f"valid: {report.is_valid()} ({len(report.issues)} issue(s))")
    return 0 if report.is_valid() else 1


def cmd_diff(args: argparse.Namespace) -> int:
    if getattr(args, "store", None):
        from worldstore.store import WorldStore
        store = WorldStore(args.store)
        before = store.load_version(args.before)
        after = store.load_version(args.after)
    else:
        before = _load_world(args.before)
        after = _load_world(args.after)
    world_diff = reality.diff(before, after)
    print(json.dumps(world_diff.to_dict(), indent=2, sort_keys=True))
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    world = _load_world(args.world)
    # All three exporters (gltf/usda/blender) now accept artifact_store
    # uniformly, so reconnection to a real geometry store is no longer
    # format-specific.
    artifact_store = None
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


def _load_point_cloud(path: str) -> list[Vec3]:
    """Load point cloud from PLY or JSON."""
    import numpy as np
    path_obj = Path(path)
    if path_obj.suffix.lower() == ".ply":
        from reconstruction.backend.dense_output import parse_fused_ply
        points, _ = parse_fused_ply(path_obj.read_bytes(), source_evidence_ids=[])
        return [Vec3(p.position[0], p.position[1], p.position[2]) for p in points]
    else:
        # JSON format: list of [x, y, z]
        data = json.loads(path_obj.read_text(encoding="utf-8"))
        return [Vec3(float(p[0]), float(p[1]), float(p[2])) for p in data]


def _load_anchors(path: str) -> list[tuple[Vec3, Vec3]]:
    """Load anchor pairs from JSON: {"source": [[x,y,z],...], "target": [[x,y,z],...]}."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    src = [Vec3(float(p[0]), float(p[1]), float(p[2])) for p in data.get("source", [])]
    tgt = [Vec3(float(p[0]), float(p[1]), float(p[2])) for p in data.get("target", [])]
    if len(src) != len(tgt):
        raise ValueError(f"anchor source/target length mismatch: {len(src)} vs {len(tgt)}")
    return list(zip(src, tgt))


def cmd_register(args: argparse.Namespace) -> int:
    """Register two point clouds using the RegistrationEngine.

    Reads source/target point clouds (PLY or JSON), optional anchor pairs,
    and optional initial transform. Outputs a RegistrationResult JSON
    with the transform, covariance, and attempt log.
    """
    source_cloud = _load_point_cloud(args.source)
    target_cloud = _load_point_cloud(args.target)

    anchor_source = None
    anchor_target = None
    if args.anchors:
        pairs = _load_anchors(args.anchors)
        anchor_source, anchor_target = zip(*pairs) if pairs else ([], [])

    initial_transform = None
    if args.initial_transform:
        data = json.loads(Path(args.initial_transform).read_text(encoding="utf-8"))
        if "transform" in data:
            data = data["transform"]
        from reconstruction.calibration.transforms import RigidTransform
        initial_transform = RigidTransform.from_dict(data)

    engine = RegistrationEngine()
    try:
        result = engine.register(
            source_cloud=source_cloud,
            target_cloud=target_cloud,
            from_frame=args.from_frame,
            to_frame=args.to_frame,
            anchor_source=list(anchor_source) if anchor_source else None,
            anchor_target=list(anchor_target) if anchor_target else None,
            initial_transform=initial_transform,
            min_overlap=args.min_overlap,
        )
    except Exception as exc:
        _eprint(f"registration failed: {exc}")
        return 1

    out_data = result.to_dict()
    Path(args.output).write_text(json.dumps(out_data, indent=2, sort_keys=True), encoding="utf-8")
    status = "ACCEPTED" if result.status == "accepted" else "BLOCKED"
    print(f"registration {status}: {result.method}")
    print(f"  rmse: {result.rmse:.6f}  inlier_fraction: {result.inlier_fraction:.3f}")
    print(f"  attempts: {len(result.attempts)}")
    if result.covariance:
        print(f"  translation_sigma_m: {result.covariance.translation_sigma_m:.6f}")
        print(f"  rotation_sigma_rad: {result.covariance.rotation_sigma_rad:.6f}")
    print(f"wrote result -> {args.output}")
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


def cmd_store_save(args: argparse.Namespace) -> int:
    """Persist a compiled WorldIR into a versioned WorldStore."""
    from worldstore.store import WorldStore

    world = _load_world(args.world)
    store = WorldStore(args.store)
    try:
        stored = store.save_version(world, parent=args.parent, version_id=args.version_id or None)
    except Exception as exc:
        _eprint(f"store save failed: {exc}")
        return 1
    print(f"saved version {stored.version_id} (world={stored.world_id} parent={stored.parent})")
    return 0


def cmd_store_load(args: argparse.Namespace) -> int:
    """Load one WorldStore version back to a WorldIR JSON file."""
    from worldstore.store import WorldStore

    store = WorldStore(args.store)
    try:
        world = store.load_version(args.version)
    except Exception as exc:
        _eprint(f"store load failed: {exc}")
        return 1
    _save_world(world, args.output)
    print(f"loaded version {args.version} -> {args.output}")
    return 0


def cmd_store_list(args: argparse.Namespace) -> int:
    """List WorldStore versions in save order with lineage."""
    from worldstore.store import WorldStore

    store = WorldStore(args.store)
    for record in store.list_versions():
        print(f"{record.version_id}\tworld={record.world_id}\tparent={record.parent or '-'}")
    return 0


def cmd_store_verify(args: argparse.Namespace) -> int:
    """Verify one WorldStore version's artifact integrity."""
    from worldstore.store import WorldStore

    store = WorldStore(args.store)
    try:
        failures = store.verify_version(args.version)
    except Exception as exc:
        _eprint(f"store verify failed: {exc}")
        return 1
    if failures:
        for failure in failures:
            _eprint(f"FAILED {failure.get('version')}: {failure.get('reason')}")
        return 1
    print(f"version {args.version} verifies OK")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    """Inspect one world: entity summary, one entity, or provenance detail."""
    world = _load_world(args.world)
    if args.entity:
        entity = world.entities.get(args.entity)
        if entity is None:
            _eprint(f"unknown entity id: {args.entity!r}")
            return 1
        payload = {
            "id": entity.id,
            "type": entity.type.value,
            "name": entity.name,
            "provenance": entity.provenance.value,
            "confidence": entity.confidence,
            "geometry_ids": list(entity.geometry_ids),
            "relationships": [
                {
                    "kind": rel.kind.value,
                    "target": rel.target_id,
                    "confidence": rel.confidence,
                    "provenance": rel.provenance.value,
                }
                for rel in entity.relationships
            ],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    by_type: dict = {}
    for entity in world.entities.values():
        by_type[entity.type.value] = by_type.get(entity.type.value, 0) + 1
    payload = {
        "world_id": world.id,
        "entities": len(world.entities),
        "geometries": len(world.geometries),
        "by_type": by_type,
        "global_provenance": world.global_provenance.value,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def cmd_viewer(args: argparse.Namespace) -> int:
    """Build the self-contained offline viewer HTML for pipeline artifacts."""
    from apps.viewer.build_viewer import main as build_viewer_main

    argv: List[str] = ["--out", args.output]
    if args.worldir:
        argv += ["--worldir", args.worldir]
    if args.points:
        argv += ["--points", args.points]
    if args.cameras:
        argv += ["--cameras", args.cameras]
    if args.mesh:
        argv += ["--mesh", args.mesh]
    try:
        return int(build_viewer_main(argv))
    except SystemExit as exc:
        # build_viewer uses argparse: surface its exit code, never crash.
        return int(exc.code or 0)
    except Exception as exc:
        _eprint(f"viewer build failed: {exc}")
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="reality", description="Reality Engine command-line interface")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="import a folder of photos/video/LAS into an evidence package")
    p_ingest.add_argument("folder")
    p_ingest.add_argument("-o", "--output", required=True, help="package JSON output path")
    p_ingest.add_argument("--seed", default="pkg", help="deterministic id seed (default: pkg)")
    p_ingest.add_argument("--package-id", default="", help="explicit package id (default: pkg-<seed>)")
    p_ingest.set_defaults(func=cmd_ingest)

    p_session = sub.add_parser("session", help="multi-source ingestion session: create/add-source/list/export")
    session_sub = p_session.add_subparsers(dest="session_command", required=True)

    p_session_create = session_sub.add_parser("create", help="create a new multi-source session")
    p_session_create.add_argument("session_id")
    p_session_create.add_argument("-o", "--session-dir", required=True, help="directory to hold session.json")
    p_session_create.add_argument("--name", default="", help="human-readable session name")
    p_session_create.set_defaults(func=cmd_session_create)

    p_source_add = session_sub.add_parser(
        "add-source", help="ingest one file/folder into an existing session (incremental; dedups by content)"
    )
    p_source_add.add_argument("session_dir")
    p_source_add.add_argument("path", help="file or folder to ingest")
    p_source_add.add_argument(
        "--capture-type", choices=["phone", "drone"], default=None,
        help="declare a composite folder (rgb/video + gps/imu/depth/calibration/telemetry subfolders) "
             "as a phone or drone capture; ignored for plain files/folders",
    )
    p_source_add.set_defaults(func=cmd_source_add)

    p_source_list = session_sub.add_parser("list", help="list sources and asset counts in a session")
    p_source_list.add_argument("session_dir")
    p_source_list.set_defaults(func=cmd_source_list)

    p_session_inspect = session_sub.add_parser(
        "inspect", help="session-level evidence summary + reconstruction readiness"
    )
    p_session_inspect.add_argument("session_dir")
    p_session_inspect.set_defaults(func=cmd_session_inspect)

    p_source_inspect = session_sub.add_parser("inspect-source", help="full detail for one source in a session")
    p_source_inspect.add_argument("session_dir")
    p_source_inspect.add_argument("source_id")
    p_source_inspect.set_defaults(func=cmd_source_inspect)

    p_session_export = session_sub.add_parser(
        "export-package", help="write the session's accumulated EvidencePackage as package JSON (for `reconstruct`)"
    )
    p_session_export.add_argument("session_dir")
    p_session_export.add_argument("-o", "--output", required=True)
    p_session_export.set_defaults(func=cmd_session_export_package)

    p_compile = sub.add_parser(
        "compile",
        help="capture dataset -> full mapping pipeline -> WorldIR + artifacts + exports",
    )
    p_compile.add_argument("dataset", help="dataset folder with manifest.json + images/")
    p_compile.add_argument(
        "-o", "--output", default=None,
        help="output directory (default: <dataset>/pipeline_out)",
    )
    p_compile.add_argument("--colmap-binary", default="colmap")
    p_compile.add_argument("--no-depth", action="store_true", help="skip the depth stage")
    p_compile.add_argument("--no-mesh", action="store_true", help="skip surface reconstruction")
    p_compile.set_defaults(func=cmd_compile)

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

    p_diff = sub.add_parser("diff", help="structural diff between two WorldIR snapshots or versions")
    p_diff.add_argument("before", help="before world JSON path or version ID")
    p_diff.add_argument("after", help="after world JSON path or version ID")
    p_diff.add_argument("--store", default=None, help="optional WorldStore root directory when comparing version IDs")
    p_diff.set_defaults(func=cmd_diff)

    p_export = sub.add_parser(
        "export",
        help="export a WorldIR to gltf/usda/blender (reconnects to <world>.artifacts/ "
             "for real geometry, if it exists)",
    )
    p_export.add_argument("world")
    p_export.add_argument("--format", required=True, choices=["gltf", "usda", "blender"])
    p_export.add_argument("-o", "--output", required=True)
    p_export.set_defaults(func=cmd_export)

    p_register = sub.add_parser(
        "register",
        help="register two point clouds (cross-source alignment via GNSS anchors / ICP)",
    )
    p_register.add_argument("source", help="source point cloud (PLY or JSON)")
    p_register.add_argument("target", help="target point cloud (PLY or JSON)")
    p_register.add_argument("-o", "--output", required=True, help="output JSON path")
    p_register.add_argument("--from-frame", required=True, help="source frame name")
    p_register.add_argument("--to-frame", required=True, help="target frame name")
    p_register.add_argument("--anchors", help="JSON file with anchor pairs: {\"source\": [...], \"target\": [...]}")
    p_register.add_argument("--initial-transform", help="JSON file with initial RigidTransform")
    p_register.add_argument("--min-overlap", type=float, default=0.5, help="minimum inlier fraction (default: 0.5)")
    p_register.set_defaults(func=cmd_register)

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

    p_store = sub.add_parser("store", help="versioned WorldStore persistence: save/load/list/verify")
    store_sub = p_store.add_subparsers(dest="store_command", required=True)

    p_store_save = store_sub.add_parser("save", help="persist a world JSON as a new immutable version")
    p_store_save.add_argument("world", help="world JSON path")
    p_store_save.add_argument("--store", required=True, help="WorldStore root directory")
    p_store_save.add_argument("--parent", default=None, help="parent version id (lineage)")
    p_store_save.add_argument("--version-id", default=None, help="explicit version id")
    p_store_save.set_defaults(func=cmd_store_save)

    p_store_load = store_sub.add_parser("load", help="load one version back to a world JSON file")
    p_store_load.add_argument("--store", required=True, help="WorldStore root directory")
    p_store_load.add_argument("--version", required=True, help="version id to load")
    p_store_load.add_argument("-o", "--output", required=True, help="world JSON output path")
    p_store_load.set_defaults(func=cmd_store_load)

    p_store_list = store_sub.add_parser("list", help="list versions in save order")
    p_store_list.add_argument("--store", required=True, help="WorldStore root directory")
    p_store_list.set_defaults(func=cmd_store_list)

    p_store_verify = store_sub.add_parser("verify", help="verify one version's artifact integrity")
    p_store_verify.add_argument("--store", required=True, help="WorldStore root directory")
    p_store_verify.add_argument("--version", required=True, help="version id to verify")
    p_store_verify.set_defaults(func=cmd_store_verify)

    p_inspect = sub.add_parser("inspect", help="inspect a world summary or one entity's provenance")
    p_inspect.add_argument("world", help="world JSON path")
    p_inspect.add_argument("--entity", default=None, help="entity id for full detail")
    p_inspect.set_defaults(func=cmd_inspect)

    p_viewer = sub.add_parser("viewer", help="build the self-contained offline viewer HTML")
    p_viewer.add_argument("--worldir", default=None, help="worldir.json artifact path")
    p_viewer.add_argument("--points", default=None, help="points.ply artifact path")
    p_viewer.add_argument("--cameras", default=None, help="cameras.json artifact path")
    p_viewer.add_argument("--mesh", default=None, help="mesh.ply artifact path")
    p_viewer.add_argument("-o", "--output", required=True, help="viewer HTML output path")
    p_viewer.set_defaults(func=cmd_viewer)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
