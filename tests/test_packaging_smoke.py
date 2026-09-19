"""Packaging smoke test: catches the class of bug where a top-level
package (registration/, trajectories/, worldstore/) exists on disk and
imports fine in-place, but was never added to
[tool.setuptools.packages.find].include in pyproject.toml -- so it
silently disappears from a built wheel/sdist and only breaks post-install.

Cross-checks the declared include globs against every top-level directory
in the repo that actually contains an __init__.py (i.e. is a real Python
package), rather than hardcoding the expected package list -- so this test
catches the *next* package that's added to the repo but forgotten here too.
"""

from __future__ import annotations

import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[import-not-found]

REPO_ROOT = Path(__file__).resolve().parent.parent

# Directories that are real Python packages but are deliberately excluded
# from the build (tests, tooling, non-package data/config directories).
_NOT_SHIPPED = {"tests", "scripts", "tools", "benchmarks", "datasets", "config", "database", "gpu", "shaders", "plugins", "agent_tasks"}


def _top_level_packages() -> set[str]:
    packages = set()
    for entry in REPO_ROOT.iterdir():
        if entry.is_dir() and (entry / "__init__.py").exists() and entry.name not in _NOT_SHIPPED:
            packages.add(entry.name)
    return packages


def _declared_include_prefixes() -> set[str]:
    with open(REPO_ROOT / "pyproject.toml", "rb") as f:
        data = tomllib.load(f)
    globs = data["tool"]["setuptools"]["packages"]["find"]["include"]
    # Each entry is "<name>*" -- strip the trailing glob star.
    return {g.rstrip("*") for g in globs}


def test_every_top_level_package_is_declared_for_packaging():
    packages = _top_level_packages()
    declared = _declared_include_prefixes()
    missing = packages - declared
    assert not missing, (
        f"top-level package(s) {sorted(missing)} exist on disk with an "
        f"__init__.py but are NOT in pyproject.toml's "
        f"[tool.setuptools.packages.find].include -- they will silently "
        f"disappear from any built wheel/sdist. Add '<name>*' to include."
    )


def test_registration_trajectories_worldstore_are_declared():
    """Named regression test for the specific known-bad state (P3 brief):
    these three packages existed on disk but were missing from the build
    include list."""
    declared = _declared_include_prefixes()
    for name in ("registration", "trajectories", "worldstore"):
        assert name in declared, f"{name}* missing from packages.find include"


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
