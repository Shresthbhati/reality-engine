"""Generate the CI test registry from real test discovery.

    python scripts/ci_test_registry.py                      # (re)write docs/engineering/CI_TEST_REGISTRY.md + .json
    python scripts/ci_test_registry.py --check              # exit 1 (naming added/removed/changed files) if stale
    python scripts/ci_test_registry.py --verify-collection  # cross-check the AST discovery against `pytest --collect-only`

Discovery parses ``tests/**/test_*.py`` (no imports, no network, no optional dependencies), so the registry is
deterministic on any machine. Per test file it records: owner, area, category, test count, slow / GPU / native-engine /
physical-dataset flags, declared optional-dependency requirements, the CI shard it runs in and the CI lanes that run it.
``--verify-collection`` proves the AST view equals what pytest really collects, so a file cannot silently be dropped.
"""
from __future__ import annotations

import argparse
import ast
import fnmatch
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ci_shard_tests  # noqa: E402

CI_YAML = ROOT / ".github/workflows/ci.yml"
CODEOWNERS = ROOT / ".github/CODEOWNERS"
OUT_MD = ROOT / "docs/engineering/CI_TEST_REGISTRY.md"
OUT_JSON = ROOT / "docs/engineering/ci_test_registry.json"
SHARDS = 8
_PATH = re.compile(r"tests/[\w./-]+\.py")

#: optional / heavy modules whose use makes a test depend on an extra
OPTIONAL_MODULES = ("torch", "torchvision", "transformers", "kornia", "timm", "segment_anything", "open3d", "pycolmap",
                    "pxr", "manifold3d", "OpenEXR", "imageio", "psutil", "trimesh", "ifcopenshell", "ezdxf", "boto3")
_NATIVE = re.compile(r"colmap_binary|shutil\.which\(\s*[\"']colmap|ColmapBackend|COLMAPBackend|colmap_backend|"
                     r"requires\([^)]*[\"']colmap[\"']|real COLMAP", re.I)
_GPU = re.compile(r"torch\.cuda|[\"']cuda[\"']|\bRTX\s?4050|gpu_lifecycle|vram", re.I)
_DATASET = re.compile(r"datasets[/\\]+(\w+)")
_PRIORITY = ("release-gate", "physics-excluded", "physical-data", "native", "gpu", "integration", "e2e", "unit")


def _physics_excluded() -> set[str]:
    src = (ROOT / "conftest.py").read_text(encoding="utf-8")
    block = re.search(r"PHYSICS_TEST_MODULES\s*=\s*\[(.*?)\]", src, re.S)
    return {f"tests/{n}" for n in re.findall(r"[\"']([\w.]+\.py)[\"']", block.group(1))} if block else set()


def _owners() -> list[tuple[str, str]]:
    rules = []
    for line in CODEOWNERS.read_text(encoding="utf-8").splitlines():
        parts = line.split("#")[0].split()
        if len(parts) >= 2:
            rules.append((parts[0], parts[1]))
    return rules


def _owner_of(path: str, rules: list[tuple[str, str]]) -> str:
    owner = "unowned"
    for pat, who in rules:  # CODEOWNERS: the last matching rule wins
        p = pat.lstrip("/")
        if pat == "*" or fnmatch.fnmatch(path, p) or path.startswith(p.rstrip("*")):
            owner = who
    return owner


def _marks(node: ast.AST) -> list[tuple[str, tuple]]:
    """(marker name, literal string args) of every ``pytest.mark.X(...)`` / ``pytestmark`` entry under ``node``."""
    out: list[tuple[str, tuple]] = []
    for n in ast.walk(node):
        target = n.func if isinstance(n, ast.Call) else n
        if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Attribute) and target.value.attr == "mark":
            args = tuple(a.value for a in getattr(n, "args", []) if isinstance(a, ast.Constant) and isinstance(a.value, str))
            out.append((target.attr, args))
    return out


def _is_test(n: ast.AST) -> bool:
    """A ``test*`` function that pytest collects: fixtures that merely START with ``test`` are not tests."""
    return (isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name.startswith("test")
            and not any("fixture" in ast.unparse(d) for d in n.decorator_list))


def _tests(tree: ast.Module) -> int:
    count = 0
    for n in tree.body:
        if _is_test(n):
            count += 1
        elif isinstance(n, ast.ClassDef) and n.name.startswith("Test"):
            count += sum(1 for m in n.body if _is_test(m))
    return count


def _imports(tree: ast.Module) -> set[str]:
    mods: set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods.update(a.name.split(".")[0] for a in n.names)
        elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
            mods.add(n.module.split(".")[0])
    return mods


def _first_party() -> set[str]:
    return {p.name for p in ROOT.iterdir() if p.is_dir() and not p.name.startswith(".")} - {"tests", "docs", "scripts"}


def analyse(rel: str, shard_of: dict[str, int], core: set[str], fidelity: set[str], physics: set[str],
            rules: list[tuple[str, str]], first_party: set[str]) -> dict:
    text = (ROOT / rel).read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(text)
    marks = _marks(tree)
    names = {m for m, _ in marks}
    requires = sorted({a for m, args in marks if m == "requires" for a in args}
                      | {m.group(1) for m in re.finditer(r"importorskip\(\s*[\"']([\w.]+)[\"']", text)})
    imported = _imports(tree)
    optional = sorted((set(OPTIONAL_MODULES) & imported) | {r for r in requires if r in OPTIONAL_MODULES})
    datasets = sorted(set(_DATASET.findall(text)))
    native = bool(_NATIVE.search(text)) or "colmap" in requires
    gpu = bool(_GPU.search(text)) or "cuda" in requires
    area_votes = Counter(m for m in imported if m in first_party)
    area = max(area_votes, key=lambda m: (area_votes[m], m)) if area_votes else "-"   # tie -> name: hash-seed independent
    slow = "slow" in names
    tags = {"release-gate": "release_gate" in rel, "physics-excluded": rel in physics, "physical-data": bool(datasets),
            "native": native, "gpu": gpu, "integration": rel.startswith("tests/integration/") or "integration" in names,
            "e2e": "e2e" in names, "unit": True}
    category = next(c for c in _PRIORITY if tags[c])
    lanes = []
    if rel in core:
        lanes.append("core")
    if rel in fidelity:
        lanes.append("fidelity-gate")
    lanes.append("full")
    if slow:
        lanes.append("nightly-slow")
    return {"file": rel, "owner": _owner_of(rel, rules), "area": area, "category": category, "tests": _tests(tree),
            "slow": slow, "gpu": gpu, "native_engine": native, "datasets": datasets, "optional_deps": optional,
            "requires": requires, "shard": shard_of[rel], "lanes": lanes,
            "collected_in_default_run": rel not in physics}


def _ci_lists() -> tuple[set[str], set[str]]:
    jobs = yaml.safe_load(CI_YAML.read_text(encoding="utf-8"))["jobs"]
    core: list[str] = []
    for step in jobs["core"]["steps"]:
        if step.get("name") == "Run core test suite":
            core = _PATH.findall(step["run"])
    fidelity = [p for s in jobs["fidelity-gate"]["steps"] for p in _PATH.findall(s.get("run", ""))]
    return set(core), set(fidelity)


def build_registry(ci_yaml: Path = CI_YAML, tests_dir: Path = ROOT / "tests") -> dict:
    """Backward-compatible keys (core / fidelity_gate / unlisted / ...) plus the per-file ``entries``."""
    core, fidelity = _ci_lists()
    base = tests_dir.parent
    files = sorted(p.relative_to(base).as_posix() for p in tests_dir.rglob("test_*.py"))   # == ci_shard_tests.all_test_files
    parts = ci_shard_tests.shards(files, SHARDS, base)
    shard_of = {f: i + 1 for i, p in enumerate(parts) for f in p}
    physics, rules, fp = _physics_excluded(), _owners(), _first_party()
    entries = [analyse(f, shard_of, core, fidelity, physics, rules, fp) for f in files]
    return {
        "core": sorted(core),
        "fidelity_gate": sorted(fidelity),
        "collect_only": ["tests/ (collection only, nothing executed)"],
        "unlisted": sorted(f for f in files if f not in core | fidelity),
        "controlled_baseline": ["scripts/ci_static_analysis_gate.py --fail-on-regression"],
        "entries": entries,
    }


def _cell(v) -> str:
    if isinstance(v, bool):
        return "yes" if v else ""
    if isinstance(v, list):
        return ", ".join(v)
    return str(v)


def render(registry: dict) -> str:
    e = registry["entries"]
    cat = Counter(x["category"] for x in e)
    lines = [
        "# CI Test Registry", "",
        "Generated by `scripts/ci_test_registry.py` from real test discovery. **Do not edit by hand**; run the script "
        "(`--check` is enforced by `tests/test_ci_test_registry.py`; `--verify-collection` proves it equals what pytest "
        "collects).", "",
        f"- Test files: **{len(e)}**  |  every file runs in exactly one of **{SHARDS}** deterministic size-weighted `full` shards "
        f"(`python scripts/ci_shard_tests.py --shard N --of {SHARDS}`; per-file counts and shard numbers are NOT committed: they churn on every edit)",
        f"- Slow (nightly lane only): {sum(x['slow'] for x in e)}  |  GPU: {sum(x['gpu'] for x in e)}  |  native engine "
        f"(COLMAP): {sum(x['native_engine'] for x in e)}  |  physical dataset: {sum(bool(x['datasets']) for x in e)}  |  "
        f"declare optional deps: {sum(bool(x['optional_deps'] or x['requires']) for x in e)}",
        f"- Files named by the `core` allowlist job: {len(registry['core'])}  |  by the fidelity gate: "
        f"{len(registry['fidelity_gate'])}  |  every file also runs in its `full` shard (core/fidelity are extra gates, "
        "not the only coverage).",
        "- Excluded from default collection (downstream physics repo, `conftest.py::PHYSICS_TEST_MODULES`): "
        f"{sum(not x['collected_in_default_run'] for x in e)}", "",
        "## Categories", "", "| category | files |", "|---|---|", *[f"| {c} | {cat[c]} |" for c in _PRIORITY if cat[c]], "",
        "Category priority: release-gate > physics-excluded > physical-data > native > gpu > integration > e2e > unit. "
        "`requires` lists declared `@pytest.mark.requires(...)` capabilities and `importorskip` modules (see "
        "`tests/_capabilities.py`); under `RE_STRICT_REQUIREMENTS=1` an unmet requirement FAILS instead of skipping.", "",
        "## Files", "",
        "| file | owner | area | category | slow | gpu | native | datasets | optional deps | requires | lanes |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for x in e:
        lines.append("| `{file}` | {o} | {a} | {c} | {s} | {g} | {n} | {d} | {od} | {r} | {l} |".format(
            file=x["file"], o=x["owner"], a=x["area"], c=x["category"], s=_cell(x["slow"]), g=_cell(x["gpu"]),
            n=_cell(x["native_engine"]), d=_cell(x["datasets"]), od=_cell(x["optional_deps"]), r=_cell(x["requires"]),
            l=_cell(x["lanes"])))
    return "\n".join(lines) + "\n"


def _dump_json(registry: dict) -> str:
    return json.dumps({"shards": SHARDS, "entries": registry["entries"]}, indent=1, sort_keys=True) + "\n"


def _diff(old: dict, new: dict) -> list[str]:
    a, b = {x["file"]: x for x in old.get("entries", [])}, {x["file"]: _stable(x) for x in new["entries"]}
    out = [f"  added:   {f}" for f in sorted(b.keys() - a.keys())] + [f"  removed: {f}" for f in sorted(a.keys() - b.keys())]
    out += [f"  changed: {f}" for f in sorted(a.keys() & b.keys()) if a[f] != b[f]]
    return out


def verify_collection() -> int:
    """AST discovery must equal what pytest collects: no file may silently collect nothing.

    A module skipped AT IMPORT (``pytest.importorskip`` at module level) collects 0 tests by design but is reported by
    pytest as SKIPPED with its reason, so it is listed (visible, budgeted) instead of failing. Lower counts than the AST
    (duplicate names, conditionally defined tests) are printed as notes.
    """
    reg = build_registry()
    proc = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-rs", "-p", "no:cacheprovider", "tests"],
                          cwd=ROOT, capture_output=True, text=True)
    out = proc.stdout
    collected = Counter(line.split("::", 1)[0].replace("\\", "/") for line in out.splitlines() if "::" in line)
    skipped_at_import = {m.group(1).replace("\\", "/"): m.group(2) for m in
                         re.finditer(r"^SKIPPED \[\d+\] (\S+?):\d+: (.*)$", out, re.M)}
    problems, notes = [], []
    for x in reg["entries"]:
        f = x["file"]
        if not x["tests"] or not x["collected_in_default_run"]:
            continue
        if not collected[f]:
            if f in skipped_at_import:
                notes.append(f"  skipped at import: {f} ({skipped_at_import[f][:90]})")
            else:
                problems.append(f"  {f}: {x['tests']} test functions in the AST, 0 collected and not reported skipped")
        elif collected[f] < x["tests"]:
            notes.append(f"  fewer collected than AST functions: {f} ({collected[f]} < {x['tests']})")
    known = {x["file"] for x in reg["entries"]}
    problems += [f"  {f}: collected but absent from the registry" for f in sorted(set(collected) - known)]
    problems += [f"  collection {line}" for line in out.splitlines() if line.startswith("ERROR ")]
    for n in notes:
        print(n)
    if problems:
        print("collection does not match the registry:\n" + "\n".join(problems), file=sys.stderr)
        return 1
    print(f"OK: {len(reg['entries'])} files, {sum(collected.values())} collected tests, {len(notes)} notes, no collection errors")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--verify-collection", action="store_true")
    ap.add_argument("--json", action="store_true", help="print the FULL registry (with test counts and shard numbers) to stdout")
    args = ap.parse_args(argv)
    if args.verify_collection:
        return verify_collection()
    reg = build_registry()
    if args.json:
        sys.stdout.write(_dump_json(reg, full=True))
        return 0
    md, js = render(reg), _dump_json(reg)
    if args.check:
        cur_md = OUT_MD.read_text(encoding="utf-8") if OUT_MD.exists() else None
        cur_js = OUT_JSON.read_text(encoding="utf-8") if OUT_JSON.exists() else None
        if cur_md != md or cur_js != js:
            old = json.loads(cur_js) if cur_js else {}
            print(f"{OUT_MD} is out of date; run `python scripts/ci_test_registry.py`.\n" + "\n".join(_diff(old, reg)[:40]),
                  file=sys.stderr)
            return 1
        return 0
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(md, encoding="utf-8", newline="\n")
    OUT_JSON.write_text(js, encoding="utf-8", newline="\n")
    print(f"wrote {OUT_MD.relative_to(ROOT)} and {OUT_JSON.relative_to(ROOT)} ({len(reg['entries'])} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
