"""Dependency contract: what the packages import must match what ``pyproject.toml`` declares.

Three failure classes this guards (all of which shipped before):

1. an OPTIONAL dependency (torch, transformers, ...) imported unguarded at module level, so importing a package -- or
   COLLECTING a test file that imports it -- crashes on an install without the extra (the torch-free CI image);
2. a module the core code imports at module level that no ``dependencies`` / extra entry declares (it only worked because
   another package dragged it in);
3. a first-party package imported by shipped code but missing from the wheel (``world`` was).

No network, no heavy imports: (1) runs a child interpreter that BLOCKS the optional modules even if they are installed.
"""
from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

try:  # tomllib is 3.11+; the package supports 3.10 (tomli is a declared conditional dependency)
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

#: pip distribution name (normalised) -> import name, where they differ
_DIST_TO_MODULE = {
    "opencv-python-headless": "cv2", "pillow": "PIL", "pyyaml": "yaml", "python-multipart": "multipart",
    "usd-core": "pxr", "openexr": "OpenEXR", "segment-anything": "segment_anything",
}
#: modules that ship with another declared package (their parent is declared, they have no dist of their own)
_PROVIDED = {"starlette": "fastapi", "botocore": "boto3", "anyio": "httpx", "greenlet": "sqlalchemy"}
#: first-party modules that cannot be imported in this repo by design (downstream `reality-engine-child` physics repo)
_KNOWN_UNIMPORTABLE = {
    "engine.compiler.simulation_loop": "needs engine.compiler.physics_compiler (reality-engine-child)",
}
#: optional modules that must NEVER be needed to import a first-party package
OPTIONAL_BLOCKED = ("torch", "torchvision", "transformers", "kornia", "timm", "segment_anything", "open3d", "pycolmap",
                    "pxr", "manifold3d", "OpenEXR", "imageio", "psutil", "trimesh", "ifcopenshell", "ezdxf")
_NOT_PACKAGES = {"tests", "scripts", "benchmarks", "tools", "docs", "frontend", "desktop"}
#: shipped modules that import a NON-shipped directory at module level (an inverted dependency, tracked, never grown).
#: DECISION REQUIRED (perception owner): move `indoor_dataset_adapter` from benchmarks/perception/ into perception/, or
#: stop importing it from shipped code. Until then a wheel install cannot import this module.
_KNOWN_PACKAGING_GAPS = {"perception/intelligence/diagnostics_v2.py": {"benchmarks"}}


def _pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _dist_name(requirement: str) -> str:
    return re.split(r"[\s\[<>=!~;(]", requirement.strip(), maxsplit=1)[0].lower().replace("_", "-")


def _modules(requirements: list[str]) -> set[str]:
    out: set[str] = set()
    for r in requirements:
        d = _dist_name(r)
        if d != "reality-engine":
            out.add(_DIST_TO_MODULE.get(d, d.replace("-", "_")))
    return out


def _first_party_tops() -> list[str]:
    return [p.rstrip("*") for p in _pyproject()["tool"]["setuptools"]["packages"]["find"]["include"]]


def _declared() -> tuple[set[str], set[str]]:
    proj = _pyproject()["project"]
    core = _modules(proj["dependencies"])
    api = _modules(proj["optional-dependencies"]["api"])
    api |= {m for m, parent in _PROVIDED.items() if parent in api}
    return core, api


def _unguarded_module_level_imports(tree: ast.Module) -> list[tuple[str, int]]:
    """(top-level module, line) of imports executed on ``import`` and NOT inside try/except or TYPE_CHECKING."""
    out: list[tuple[str, int]] = []

    def walk(body: list[ast.stmt], guarded: bool) -> None:
        for n in body:
            if isinstance(n, ast.Import):
                if not guarded:
                    out.extend((a.name.split(".")[0], n.lineno) for a in n.names)
            elif isinstance(n, ast.ImportFrom):
                if n.level == 0 and n.module and not guarded:
                    out.append((n.module.split(".")[0], n.lineno))
            elif isinstance(n, ast.Try):
                walk(n.body, True)
                for h in n.handlers:
                    walk(h.body, guarded)
                walk(n.orelse, guarded)
                walk(n.finalbody, guarded)
            elif isinstance(n, ast.If):
                if "TYPE_CHECKING" in ast.unparse(n.test):
                    continue
                walk(n.body, guarded)
                walk(n.orelse, guarded)
            elif isinstance(n, ast.With):
                walk(n.body, guarded)

    walk(tree.body, False)
    return out


def test_module_level_third_party_imports_are_declared_dependencies():
    tops = _first_party_tops()
    core, api = _declared()
    known = set(sys.stdlib_module_names) | set(tops) | {"__future__"}
    bad: list[str] = []
    seen_gaps: set[tuple[str, str]] = set()
    for top in tops:
        for path in sorted((ROOT / top).rglob("*.py")):
            rel = path.relative_to(ROOT).as_posix()
            api_ok = rel.startswith(("apps/", "geo/"))
            allowed = core | (api if api_ok else set())
            for mod, line in _unguarded_module_level_imports(ast.parse(path.read_text(encoding="utf-8"))):
                if mod in _KNOWN_PACKAGING_GAPS.get(rel, ()):
                    seen_gaps.add((rel, mod))
                    continue
                if mod not in known and mod not in allowed:
                    bad.append(f"{rel}:{line} imports '{mod}' at module level but it is not a declared "
                               f"{'core/api' if api_ok else 'core'} dependency (declare it in pyproject.toml or guard it)")
    assert not bad, "\n".join(bad)
    stale = {(f, m) for f, mods in _KNOWN_PACKAGING_GAPS.items() for m in mods} - seen_gaps
    assert not stale, f"remove from _KNOWN_PACKAGING_GAPS (no longer imported): {sorted(stale)}"


def test_every_first_party_package_imported_by_shipped_code_is_in_the_wheel():
    tops = set(_first_party_tops())
    on_disk = {p.name for p in ROOT.iterdir() if p.is_dir() and not p.name.startswith((".", "_"))
               and (any(p.glob("*/__init__.py")) or (p / "__init__.py").exists())} - _NOT_PACKAGES
    missing: dict[str, str] = {}
    for top in sorted(tops):
        for path in (ROOT / top).rglob("*.py"):
            for mod, line in _unguarded_module_level_imports(ast.parse(path.read_text(encoding="utf-8"))):
                if mod in on_disk and mod not in tops and mod not in missing:
                    missing[mod] = f"{path.relative_to(ROOT).as_posix()}:{line}"
    assert not missing, f"imported by shipped code but not in [tool.setuptools.packages.find].include: {missing}"


_BLOCKED_IMPORT_PROBE = r"""
import importlib, importlib.abc, pkgutil, sys, json
sys.path.insert(0, {root!r})
BLOCK = set({block!r})
class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name.split('.')[0] in BLOCK:
            raise ModuleNotFoundError('blocked optional module ' + name, name=name)
sys.meta_path.insert(0, Blocker())
bad = {{}}
n = 0
for top in {tops!r}:
    pkg = importlib.import_module(top)
    names = [top] + [m.name for m in pkgutil.walk_packages(pkg.__path__, top + '.', onerror=lambda _x: None)]
    for name in names:
        n += 1
        try:
            importlib.import_module(name)
        except BaseException as e:  # report every failure kind; the parent decides
            bad[name] = type(e).__name__ + ': ' + str(e)[:160]
print(json.dumps({{'imported': n, 'failed': bad}}))
"""


def test_every_package_imports_with_all_optional_dependencies_blocked():
    """The torch-free CI image, reproduced anywhere: no optional module may be needed to IMPORT a first-party module."""
    code = _BLOCKED_IMPORT_PROBE.format(root=str(ROOT), block=list(OPTIONAL_BLOCKED), tops=_first_party_tops())
    proc = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-2000:]
    report = json.loads(proc.stdout.strip().splitlines()[-1])
    failed = {k: v for k, v in report["failed"].items() if k not in _KNOWN_UNIMPORTABLE}
    assert report["imported"] > 400, "import walk found suspiciously few modules"
    assert not failed, "modules that need an optional dependency (or a missing package) just to import:\n" + "\n".join(
        f"  {k}: {v}" for k, v in sorted(failed.items()))
    stale = sorted(k for k in _KNOWN_UNIMPORTABLE if k not in report["failed"])
    assert not stale, f"remove from _KNOWN_UNIMPORTABLE (they import fine now): {stale}"


def test_perception_extra_declares_the_modules_the_gpu_lane_needs():
    declared = _modules(_pyproject()["project"]["optional-dependencies"]["perception"])
    assert {"torch", "torchvision", "transformers", "kornia"} <= declared
