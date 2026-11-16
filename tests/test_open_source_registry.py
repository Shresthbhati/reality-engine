"""The open-source technology registry: schema, licence rules, honesty rules, and sync with its generated markdown view."""
from __future__ import annotations

import importlib.util
import pathlib

from engine.technologies.open_source_registry import by_id, load_registry, problems

ROOT = pathlib.Path(__file__).resolve().parents[1]
#: restrictive licence, yet usable: the reason is part of the test so it cannot be added silently
EXCEPTIONS = {"blender": "GPL-3.0 tool is never linked: exporters/blender generates a script run as a separate process"}


def _builder():
    spec = importlib.util.spec_from_file_location("build_open_source_registry", ROOT / "scripts" / "build_open_source_registry.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_registry_has_no_rule_violations():
    assert problems(load_registry(), EXCEPTIONS) == []


def test_restrictive_licences_are_never_classified_usable():
    reg = by_id(load_registry())
    for rid in ("openmvs", "ultralytics", "libigl", "xeokit", "opensplat", "odm", "pyslam"):
        assert reg[rid]["classification"] == "REJECTED", rid
    for rid in ("vggt", "dust3r", "mast3r", "cotracker", "depth_pro", "graphdeco_3dgs", "depth_anything_v2_base_large"):
        assert reg[rid]["classification"] == "RESEARCH_ONLY", rid
    # an undeclared weights licence is a review item, not an approval
    assert reg["depth_anything_v2_metric_indoor"]["classification"] == "LICENCE_REVIEW_REQUIRED"


def test_the_checker_actually_catches_a_bad_record():
    reg = load_registry()
    bad = dict(by_id(reg)["sam2"], id="sam2_bad", repo_licence="AGPL-3.0")
    found = problems({**reg, "records": reg["records"] + [bad]}, EXCEPTIONS)
    assert any("sam2_bad" in p and "AGPL" in p for p in found)


def test_no_record_claims_real_data_validation_or_production_readiness():
    assert all(r["maturity"] not in ("Real-data validated", "Production ready") for r in load_registry()["records"])


def test_every_record_has_provenance_and_a_commit_where_a_repo_exists():
    for r in load_registry()["records"]:
        assert r["provenance"]["probed_on"] and r["provenance"]["probe_file"]
        if r["repository"]:
            assert r["commit_sha"] and len(r["commit_sha"]) == 40, r["id"]


def test_ten_opportunities_each_point_at_a_registered_technology():
    reg = load_registry()
    ids = set(by_id(reg))
    assert [o["rank"] for o in reg["top_opportunities"]] == list(range(1, 11))
    assert all(o["technology_id"] in ids for o in reg["top_opportunities"])


def test_generated_files_are_in_sync_with_the_builder():
    b = _builder()
    reg = b.build()
    assert reg == load_registry(), "run scripts/build_open_source_registry.py"
    md = (ROOT / "docs" / "engineering" / "OPEN_SOURCE_TECHNOLOGY_REGISTRY.md").read_text(encoding="utf-8")
    assert md == b.render_md(reg), "run scripts/build_open_source_registry.py"
