"""Run manifest: records what was run, verifies against the disk, and detects tampering. Synthetic engines; the manifest, hashing and
world-verification code is real."""

from __future__ import annotations

import hashlib
import json
import os

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")
pytest.importorskip("aiosqlite")

from apps.api import evidence_world_stage as ew  # noqa: E402
from engine.multiengine import run_manifest  # noqa: E402
from tests.test_lift_stage_e2e import DX, _camera, _Depth, _inputs, _per_view, _points, photos  # noqa: E402, F401


def _run(tmp_path, photo_paths):
    cams = {p: _camera(DX[os.path.basename(p)]) for p in photo_paths}
    return ew.run_for_job(team_id="t1", world_id="w1", job_id="job-1", sources=photo_paths, base=tmp_path / "ws", flow_inputs=_inputs(_Depth(_per_view())),
                          depth_engines=["fake_depth"], cameras=cams, points3d=_points())


def _root(tmp_path):
    return tmp_path / "ws" / "evidence-world" / "t1" / "w1" / "job-1"


def test_the_manifest_records_inputs_engines_parameters_environment_and_the_world_hash(tmp_path, photos):  # noqa: F811
    facts = _run(tmp_path, photos)
    assert facts["manifest"]["file"] == "manifest.json"
    m = json.loads((_root(tmp_path) / "manifest.json").read_text())
    assert m["schema"] == run_manifest.SCHEMA and m["run_id"] == "job-1" and m["content_sha256"] == facts["manifest"]["content_sha256"]
    assert {i["path"]: i["sha256"] for i in m["inputs"]} == {p: hashlib.sha256(open(p, "rb").read()).hexdigest() for p in photos}
    assert m["engines"] and all("engine_id" in e for e in m["engines"])  # every engine execution, failed ones included
    assert m["depth"][0]["engine_id"] == "fake_depth" and "cache" in m["depth"][0]
    assert m["lifting"]["engine"] == "spatial_intelligence.reconstruct_spatial_scene"
    p = m["parameters"]
    assert p["depth_alignment"]["max_held_out_relative_rmse"] > 0 and p["spatial_stage"]["ray_fraction_of_median_depth"] > 0 and "none was tuned" in p["note"]
    assert m["environment"]["python"] and m["environment"]["numpy"] and "NOT_MEASURED" in m["scale"] and "bit-identical" in m["reproducibility"]
    assert m["result"]["world"]["sha256"] == facts["world_ir"]["sha256"] and m["result"]["world"]["status"] == "persisted"
    assert m["code"]["commit"] is None or len(m["code"]["commit"]) == 40


def test_verify_passes_on_an_untouched_run_and_names_what_changed_otherwise(tmp_path, photos):  # noqa: F811
    _run(tmp_path, photos)
    root = _root(tmp_path)
    ok = run_manifest.verify(root)
    assert ok["ok"] is True and ok["problems"] == []
    with open(photos[0], "ab") as fh:  # an input changes after the run
        fh.write(b"x")
    bad = run_manifest.verify(root)
    assert not bad["ok"] and any("changed since the run" in x for x in bad["problems"])
    m = json.loads((root / "manifest.json").read_text())
    m["environment"]["python"] = "0.0.0"  # the manifest is edited
    (root / "manifest.json").write_text(json.dumps(m))
    assert any("modified after it was written" in x for x in run_manifest.verify(root)["problems"])


def test_verify_detects_a_missing_manifest_and_a_damaged_world(tmp_path, photos):  # noqa: F811
    assert not run_manifest.verify(tmp_path)["ok"]
    _run(tmp_path, photos)
    root = _root(tmp_path)
    vfile = next((root / "worldstore" / "versions").glob("*.json"))
    vfile.write_text("{}")  # the persisted version record is damaged
    out = run_manifest.verify(root)
    assert not out["ok"] and any("world version" in x for x in out["problems"])
