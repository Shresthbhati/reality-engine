"""Platform boundary enforcement (P0-02 rule R1).

The world-platform chain (world_ir, evidence, reconstruction, perception,
exporters, sdk) must not import simulation/application domains
(engine.physics fire/destruction/disasters, engine.environment,
engine.disasters, engine.{fire,weather,terrain,vegetation,audio}).
Applications consume the world; the world never consumes applications --
and simulation domains attach only at the explicit physics-compiler bridge
(engine/compiler/physics_compiler.py), which the chain does not import.

This is a structural guard, not pedantry: it is the mechanism that keeps
`reality-apps/` extraction possible later (copying core into an app must
never drag simulation or application code along), per
docs/implementation/PLATFORM_APPLICATION_SEPARATION.md.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.physics

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The world-platform chain: these packages define the platform itself.
PLATFORM_PACKAGES = (
    "world_ir",
    "evidence",
    "reconstruction",
    "perception",
    "exporters",
    "sdk",
)

#: Simulation / application domains that attach only at the simulation
#: boundary (engine/compiler/physics_compiler.py) or in application
#: packages -- never inside the platform chain.
FORBIDDEN_PREFIXES = (
    "engine.physics.fire",
    "engine.physics.destruction",
    "engine.physics.disasters",
    "engine.environment",
    "engine.disasters",
    "engine.fire",
    "engine.weather",
    "engine.terrain",
    "engine.vegetation",
    "engine.audio",
    # Application packages (future): reality-apps/*
    "reality_apps.",
    "reality-apps.",
)

#: Known, explicitly accepted couplings to basic utilities (math/logging),
#: documented in PLATFORM_APPLICATION_SEPARATION.md ss3 with follow-ups
#: F1-F3. Nothing on this list touches a simulation domain.
ALLOWLISTED_FILES = {
    # engine.core.logging (F2)
    "evidence/session.py",
    "evidence/dataset.py",
    # engine.core.rng (F1)
    "perception/geometry/planes.py",
    # engine.physics.math3 (F1): Quat/Vec3 basic algebra
    "reconstruction/calibration/camera.py",
    "perception/instances/lifting.py",
    "perception/instances/object_resolution.py",
    # engine.scene_graph.spatial_index (F3)
    "world_ir/entity_reid.py",
}


def _module_name(rel_path: Path) -> str:
    parts = list(rel_path.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _imports_of(tree: ast.AST) -> list[str]:
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def _py_files_under(package: str) -> list[Path]:
    pkg_dir = REPO_ROOT / package
    if not pkg_dir.is_dir():
        return []
    return sorted(pkg_dir.rglob("*.py"))


@pytest.mark.parametrize("package", PLATFORM_PACKAGES)
def test_platform_package_does_not_import_simulation_domains(package):
    pkg_dir = REPO_ROOT / package
    if not pkg_dir.is_dir():
        pytest.skip(f"package {package} not present")
    violations: list[str] = []
    for py in _py_files_under(package):
        rel = py.relative_to(REPO_ROOT).as_posix()
        tree = ast.parse(py.read_text(encoding="utf-8"), filename=rel)
        for imported in _imports_of(tree):
            for prefix in FORBIDDEN_PREFIXES:
                if imported == prefix or imported.startswith(prefix + "."):
                    if rel in ALLOWLISTED_FILES:
                        # Allowlisted warts are only tolerated for
                        # basic-utility prefixes; a forbidden simulation
                        # domain on the allowlist would still be a bug.
                        raise AssertionError(
                            f"allowlisted file {rel} imports a simulation domain "
                            f"({imported}) -- the allowlist is for basic "
                            "utilities only; see PLATFORM_APPLICATION_SEPARATION.md"
                        )
                    violations.append(f"{rel}: {imported}")
    assert not violations, (
        "platform chain must not import simulation/application domains "
        f"(rule R1, P0-02):\n  " + "\n  ".join(violations)
    )


def test_allowlist_is_minimal_and_documented():
    """Every allowlisted file must actually exist and import nothing
    beyond basic utilities (engine.core.*, engine.physics.math3,
    engine.scene_graph.spatial_index)."""
    allowed_prefixes = (
        "engine.core.",
        "engine.physics.math3",
        "engine.scene_graph.spatial_index",
    )
    assert ALLOWLISTED_FILES, "allowlist must not be silently emptied or padded"
    for rel in sorted(ALLOWLISTED_FILES):
        py = REPO_ROOT / rel
        assert py.is_file(), f"allowlist entry no longer exists: {rel}"
        tree = ast.parse(py.read_text(encoding="utf-8"), filename=rel)
        for imported in _imports_of(tree):
            if not imported.startswith(("engine.", "reality_apps.")):
                continue
            assert imported.startswith(allowed_prefixes) or imported in allowed_prefixes, (
                f"{rel} imports {imported} -- outside the documented basic-utility "
                "allowlist; either fix the coupling or update "
                "PLATFORM_APPLICATION_SEPARATION.md with justification"
            )


def test_simulation_bridge_is_reachable_only_through_the_sdk_facade():
    """The physics compiler bridge (engine.compiler.physics_compiler)
    is the single deliberate attachment point between the world and
    simulation domains. Within the platform packages, exactly one
    module may import it: the SDK facade (sdk/reality.py), the
    sanctioned surface applications call (`reality.compile_physics`)
    -- i.e. simulation compilation is invoked from the consumer side,
    through the facade.

    engine.compiler.world_compiler is NOT a simulation attachment: it
    is the platform's own WorldIR compilation step and may be imported
    freely by the chain."""
    bridge = REPO_ROOT / "engine" / "compiler" / "physics_compiler.py"
    assert bridge.is_file(), "physics compiler bridge missing"
    allowed = {
        ("sdk/reality.py", "engine.compiler.physics_compiler"),
    }
    seen: set[tuple[str, str]] = set()
    violations: list[str] = []
    for package in PLATFORM_PACKAGES:
        pkg_dir = REPO_ROOT / package
        if not pkg_dir.is_dir():
            continue
        for py in _py_files_under(package):
            rel = py.relative_to(REPO_ROOT).as_posix()
            tree = ast.parse(py.read_text(encoding="utf-8"), filename=rel)
            for imported in _imports_of(tree):
                if imported == "engine.compiler.physics_compiler" or imported.startswith(
                    "engine.compiler.physics_compiler."
                ):
                    seen.add((rel, imported))
                    if (rel, imported) not in allowed:
                        violations.append(
                            f"{rel} imports {imported} -- the simulation "
                            "bridge must be reached through the SDK "
                            "facade (sdk/reality.py), not imported around it"
                        )
    # The sanctioned facade must actually exist and stay wired: if it
    # silently disappears the boundary rule has rotted too.
    for rel, imported in allowed:
        assert (rel, imported) in seen, (
            f"expected {rel} to import {imported} -- the sanctioned "
            "simulation attachment point vanished; update this guard "
            "deliberately, not silently"
        )
    assert not violations, "\n  ".join([""] + violations)
