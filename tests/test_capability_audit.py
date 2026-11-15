"""The capability audit decides from evidence: use-based reachability (a package __init__ re-export is not a use) and a
status that needs every live entry reachable plus passing tests."""

from __future__ import annotations

import pathlib

from scripts.capability_audit import build_import_graph, chain, classify, reach_parents


def _tree(root: pathlib.Path, files: dict) -> None:
    for rel, src in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(src, encoding="utf-8")


def test_reexport_through_an_init_is_a_use_of_the_defining_submodule_only(tmp_path):
    _tree(tmp_path, {
        "app/__init__.py": "", "app/main.py": "from lib import used_fn\n",
        "lib/__init__.py": "from .a import used_fn\nfrom .b import unused_fn\n",
        "lib/a.py": "def used_fn(): ...\n", "lib/b.py": "def unused_fn(): ...\n"})
    parent = reach_parents(build_import_graph(tmp_path), roots=("app.main",))
    assert "lib.a" in parent and "lib.b" not in parent                 # importing the package does not make b a live capability


def test_lazy_imports_inside_functions_count_and_the_chain_is_reported(tmp_path):
    _tree(tmp_path, {
        "app/__init__.py": "", "app/main.py": "def run():\n    from svc import go\n    return go()\n",
        "svc/__init__.py": "", "svc/go.py": "import deep.core\n", "deep/__init__.py": "", "deep/core.py": "x = 1\n",
        "orphan/__init__.py": "", "orphan/tool.py": "y = 2\n"})
    parent = reach_parents(build_import_graph(tmp_path), roots=("app.main",))
    assert chain(parent, "deep.core")[0] == "app.main" and chain(parent, "deep.core")[-1] == "deep.core"
    assert "orphan.tool" not in parent


def test_relative_imports_resolve(tmp_path):
    _tree(tmp_path, {"app/__init__.py": "", "app/main.py": "from .helpers import h\n", "app/helpers.py": "def h(): ...\n"})
    assert "app.helpers" in reach_parents(build_import_graph(tmp_path), roots=("app.main",))


def test_status_needs_every_live_entry_reachable_and_passing_tests():
    assert classify(3, 3, 3, 2, 0, 10) == "Existing"
    assert classify(3, 3, 2, 2, 0, 10) == "Partial"                    # one implementing entry is not on the live path
    assert classify(3, 3, 3, 2, 1, 10) == "Partial"                    # a failing test
    assert classify(3, 3, 3, 0, 0, 0) == "Partial"                     # no tests ran: not verified
    assert classify(3, 2, 2, 2, 0, 10) == "Partial"                    # a module is absent
    assert classify(3, 0, 0, 2, 0, 10) == "Missing"
    assert classify(5, 5, 1, 2, 0, 10, offline=True) == "Existing"     # an offline tool is not judged by API reachability
