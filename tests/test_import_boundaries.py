"""Import-boundary test (blueprint task 0.6): enforces the layer table in docs/blueprint/10_SYSTEM_ARCHITECTURE.md.

Imports are read with ``ast`` (nothing is executed). Lazy imports inside functions count. A violation is an
(importing module, imported module) pair; existing ones live in KNOWN_VIOLATIONS so the test fails on any NEW one,
and a stale allow-list entry (violation gone) also fails so the list shrinks as the code is fixed.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CORE = ("engine.core", "uncertainty", "provenance", "events")
SIBLINGS = ("perception", "reconstruction", "evidence", "registration")
ALL = (*CORE, "world_ir", *SIBLINGS, "worldstore", "engine.pipeline", "exporters", "apps.api")


def _others(*keep_out: str) -> tuple[str, ...]:
    return tuple(p for p in ALL if p not in keep_out)


# package -> packages it MUST NOT import (the "Must NOT import" column of the layer table).
FORBIDDEN: dict[str, tuple[str, ...]] = {
    # core layer: stdlib + numpy only -> nothing "above" (every other layer package)
    **{p: tuple(x for x in ALL if x not in CORE) for p in CORE},
    "world_ir": ("perception", "reconstruction", "apps.api"),
    # perception/reconstruction/evidence/registration: not worldstore, apps, or each other's internals
    **{p: ("worldstore", "apps.api", *(s for s in SIBLINGS if s != p)) for p in SIBLINGS},
    "worldstore": ("perception", "reconstruction", "apps.api"),
    "engine.pipeline": ("apps.api",),
    "exporters": ("perception", "reconstruction"),
    "apps.api": ("perception", "reconstruction"),  # "model backends directly"
}
# frontend is TypeScript talking HTTP over the API: it has no Python imports to check, so it is not in this table.

# (importing module, imported module) -> why it exists today. Filled from the first run; do not add casually.
KNOWN_VIOLATIONS: dict[tuple[str, str], str] = {
    ("apps.api.jobs", "perception.materials.room_surfaces"): "API imports perception directly instead of via engine/pipeline",
    ("apps.api.jobs", "reconstruction.colmap_session"): "API imports reconstruction directly instead of via engine/pipeline",
    ("apps.api.jobs", "reconstruction.proc"): "API imports reconstruction directly instead of via engine/pipeline",
    ("apps.api.jobs", "reconstruction.scale"): "API imports reconstruction directly instead of via engine/pipeline",
    ("apps.api.jobs", "reconstruction.scene_identity"): "API imports reconstruction directly instead of via engine/pipeline",
    ("apps.api.jobs", "reconstruction.structure_points"): "API imports reconstruction directly instead of via engine/pipeline",
    ("apps.api.main", "reconstruction.proc"): "API imports reconstruction directly instead of via engine/pipeline",
    ("apps.api.routes_jobs", "reconstruction.proc"): "API imports reconstruction directly instead of via engine/pipeline",
    ("apps.api.routes_sessions", "reconstruction.orchestrator"): "API imports reconstruction directly instead of via engine/pipeline",
    ("apps.api.worldstore_service", "reconstruction.scale"): "API imports reconstruction directly instead of via engine/pipeline",
    ("engine.core.verification", "exporters.citydb"): "engine.core.verification probes exporters.citydb",
    ("evidence.multi_source", "reconstruction.orchestrator"): "evidence promoters depend on reconstruction result/validation types",
    ("evidence.promote_objects", "perception.instances.measurement"): "evidence promoters depend on perception types",
    ("evidence.promote_objects", "perception.instances.object_resolution"): "evidence promoters depend on perception types",
    ("evidence.promote_planes", "perception.geometry.orientation"): "evidence promoters depend on perception types",
    ("evidence.promote_planes", "reconstruction.backend.interface"): "evidence promoters depend on reconstruction result/validation types",
    ("evidence.promote_reconstruction", "reconstruction.backend.interface"): "evidence promoters depend on reconstruction result/validation types",
    ("evidence.promote_reconstruction", "reconstruction.validation"): "evidence promoters depend on reconstruction result/validation types",
    ("evidence.sensors", "perception.depth.interface"): "evidence promoters depend on perception types",
    ("perception.architecture.scene", "evidence.promote_planes"): "perception consumes evidence.session items / promote_planes (sibling coupling)",
    ("perception.architecture.segments", "reconstruction.backend.interface"): "perception consumes reconstruction result/camera types (shared types live in the wrong layer)",
    ("perception.contract", "reconstruction.backend.interface"): "perception consumes reconstruction result/camera types (shared types live in the wrong layer)",
    ("perception.contract", "reconstruction.orchestrator"): "perception consumes reconstruction result/camera types (shared types live in the wrong layer)",
    ("perception.depth.interface", "evidence.session"): "perception consumes evidence.session items / promote_planes (sibling coupling)",
    ("perception.depth.midAS_backend", "evidence.session"): "perception consumes evidence.session items / promote_planes (sibling coupling)",
    ("perception.detail.discovery", "reconstruction.backend.interface"): "perception consumes reconstruction result/camera types (shared types live in the wrong layer)",
    ("perception.detail.pipeline", "reconstruction.backend.interface"): "perception consumes reconstruction result/camera types (shared types live in the wrong layer)",
    ("perception.detection.interface", "evidence.session"): "perception consumes evidence.session items / promote_planes (sibling coupling)",
    ("perception.detection.maskrcnn_backend", "evidence.session"): "perception consumes evidence.session items / promote_planes (sibling coupling)",
    ("perception.geometry.planes", "reconstruction.backend.interface"): "perception consumes reconstruction result/camera types (shared types live in the wrong layer)",
    ("perception.quality.assessment", "reconstruction.backend.interface"): "perception consumes reconstruction result/camera types (shared types live in the wrong layer)",
    ("perception.segmentation.interface", "evidence.session"): "perception consumes evidence.session items / promote_planes (sibling coupling)",
    ("perception.segmentation.sam_backend", "evidence.session"): "perception consumes evidence.session items / promote_planes (sibling coupling)",
    ("provenance.graph", "world_ir.artifact_store"): "core-layer provenance imports world_ir.artifact_store",
    ("reconstruction.backend.colmap_backend", "evidence.session"): "reconstruction consumes evidence.session items (sibling coupling)",
    ("reconstruction.backend.fake", "evidence.session"): "reconstruction consumes evidence.session items (sibling coupling)",
    ("reconstruction.backend.interface", "evidence.session"): "reconstruction consumes evidence.session items (sibling coupling)",
    ("reconstruction.batch", "registration.cross_session"): "reconstruction batch/merge calls cross-session registration (sibling coupling)",
    ("reconstruction.consistency", "perception.depth.interface"): "reconstruction consumes perception depth/plane types (sibling coupling)",
    ("reconstruction.depth_to_points", "perception.depth.interface"): "reconstruction consumes perception depth/plane types (sibling coupling)",
    ("reconstruction.frame", "perception.geometry.planes"): "reconstruction consumes perception depth/plane types (sibling coupling)",
    ("reconstruction.merge", "registration.cross_session"): "reconstruction batch/merge calls cross-session registration (sibling coupling)",
    ("reconstruction.orchestrator", "evidence.session"): "reconstruction consumes evidence.session items (sibling coupling)",
    ("reconstruction.orchestrator_consistency", "perception.depth.interface"): "reconstruction consumes perception depth/plane types (sibling coupling)",
    ("reconstruction.robustness", "evidence.session"): "reconstruction consumes evidence.session items (sibling coupling)",
    ("registration.cross_session", "reconstruction.backend.interface"): "registration uses reconstruction camera/transform/result types (sibling coupling)",
    ("uncertainty.propagation", "reconstruction.calibration.transforms"): "core-layer uncertainty imports RigidTransform from reconstruction.calibration",
    ("perception.segmentation.oneformer_backend", "evidence.session"): "perception consumes evidence.session items (sibling coupling)",
    ("reconstruction.mvs.dense_cloud", "perception.depth.interface"): "reconstruction consumes perception depth/plane types (sibling coupling)",
    ("reconstruction.scale.metric_anchor", "evidence.session"): "reconstruction consumes evidence.session items (sibling coupling)",
    ("reconstruction.scale.metric_anchor", "evidence.session_grouping"): "reconstruction consumes evidence.session items (sibling coupling)",
}


def _package_of(module: str) -> str | None:
    best = None
    for p in ALL:
        if (module == p or module.startswith(p + ".")) and (best is None or len(p) > len(best)):
            best = p
    return best


def _module_name(path: Path) -> str:
    parts = list(path.relative_to(ROOT).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _resolve(imp: str) -> str:
    """Trim trailing names (classes/functions) so the import is reported as the module that actually exists."""
    parts = imp.split(".")
    while len(parts) > 1 and not ((ROOT.joinpath(*parts).with_suffix(".py")).exists() or ROOT.joinpath(*parts, "__init__.py").exists()):
        parts.pop()
    return ".".join(parts)


def _imports(path: Path, module: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    is_pkg = path.name == "__init__.py"
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:  # resolve relative import against this module's package
                pkg = module.split(".") if is_pkg else module.split(".")[:-1]
                pkg = pkg[: len(pkg) - (node.level - 1)]
                base = ".".join([*pkg, *([base] if base else [])])
            found.add(base)
            found.update(f"{base}.{a.name}" for a in node.names if a.name != "*")
    return found


def collect_violations() -> dict[tuple[str, str], str]:
    out: dict[tuple[str, str], str] = {}
    for pkg, banned in FORBIDDEN.items():
        for path in (ROOT / pkg.replace(".", "/")).rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            module = _module_name(path)
            for imp in {_resolve(i) for i in _imports(path, module)}:
                target = _package_of(imp)
                if target in banned and target != _package_of(module):
                    # report at module granularity: drop a trailing name that is not a module
                    out[(module, imp)] = f"{pkg} must not import {target}"
    return out


def test_no_new_boundary_violations():
    new = {k: v for k, v in collect_violations().items() if k not in KNOWN_VIOLATIONS}
    assert not new, "new forbidden imports (fix the import, do not allow-list it):\n" + "\n".join(
        f"  {m} -> {i}   [{why}]" for (m, i), why in sorted(new.items())
    )


def test_known_violations_not_stale():
    current = collect_violations()
    stale = sorted(k for k in KNOWN_VIOLATIONS if k not in current)
    assert not stale, "KNOWN_VIOLATIONS entries no longer occur; delete them:\n" + "\n".join(f"  {m} -> {i}" for m, i in stale)
