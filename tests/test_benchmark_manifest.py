"""Benchmark manifest rules: hashes, split leakage, ground-truth provenance, and BLOCKED-instead-of-a-score. The manifests and files here
are SYNTHETIC test fixtures; passing them says nothing about any real interior benchmark (none exists in the repository yet)."""

from __future__ import annotations

import copy
import hashlib

import pytest

from engine.multiengine.benchmark_manifest import (
    METRIC_REQUIREMENTS,
    ManifestError,
    benchmark_status,
    load_manifest,
    validate_manifest,
)

GT_ITEM = {"provenance": "laser distance meter, 3 repeats", "items": [{"name": "north wall", "value_m": 3.2, "uncertainty_m": 0.005}]}


def _img(tmp_path, name, payload):
    p = tmp_path / name
    p.write_bytes(payload)
    return {"path": name, "sha256": hashlib.sha256(payload).hexdigest()}


def _manifest(tmp_path, gt_keys=()):
    gt = {k: ({"ref": "x", "provenance": "annotator A"} if k not in ("room_dimensions", "object_dimensions") else copy.deepcopy(GT_ITEM))
          for k in gt_keys}  # a copy per scene: tests mutate their manifest and must not poison the shared constant
    return {"benchmark_id": "fixture", "scenes": [
        {"scene_id": "dev1", "split": "development", "kind": "interior", "images": [_img(tmp_path, "a.jpg", b"A")], "ground_truth": {}},
        {"scene_id": "val1", "split": "validation", "kind": "interior", "images": [_img(tmp_path, "b.jpg", b"B")], "ground_truth": {}},
        {"scene_id": "test1", "split": "test", "kind": "interior", "images": [_img(tmp_path, "c.jpg", b"C")], "ground_truth": gt},
    ]}


def test_a_well_formed_manifest_has_no_problems(tmp_path):
    assert validate_manifest(_manifest(tmp_path), tmp_path) == []


def test_image_level_leakage_across_splits_is_detected(tmp_path):
    m = _manifest(tmp_path)
    m["scenes"][2]["images"] = [m["scenes"][0]["images"][0]]  # the same photograph is in development AND test
    probs = validate_manifest(m, tmp_path)
    assert any("leakage across splits" in p and "dev1" in p and "test1" in p for p in probs)


def test_the_same_image_in_two_scenes_of_one_split_is_also_leakage(tmp_path):
    m = _manifest(tmp_path)
    m["scenes"][1]["split"] = "development"
    m["scenes"][1]["images"] = [m["scenes"][0]["images"][0]]
    assert any("between scenes" in p for p in validate_manifest(m, tmp_path))


def test_a_modified_or_missing_original_is_detected(tmp_path):
    m = _manifest(tmp_path)
    (tmp_path / "a.jpg").write_bytes(b"edited")  # the original must never change
    (tmp_path / "b.jpg").unlink()
    probs = validate_manifest(m, tmp_path)
    assert any("does not match its recorded sha256" in p for p in probs) and any("does not exist" in p for p in probs)


@pytest.mark.parametrize("mut,needle", [
    (lambda s: s.update(split="holdout"), "is not one of"),
    (lambda s: s.update(kind="outdoor"), "interior or exterior"),
    (lambda s: s.update(images=[]), "no images"),
    (lambda s: s["images"][0].update(sha256="zz"), "no valid sha256"),
])
def test_malformed_scenes_are_reported(tmp_path, mut, needle):
    m = _manifest(tmp_path)
    mut(m["scenes"][2])
    assert any(needle in p for p in validate_manifest(m, tmp_path))


def test_ground_truth_without_provenance_or_uncertainty_is_not_ground_truth(tmp_path):
    m = _manifest(tmp_path, gt_keys=("room_dimensions",))
    del m["scenes"][2]["ground_truth"]["room_dimensions"]["provenance"]
    assert any("without provenance" in p for p in validate_manifest(m, tmp_path))
    m = _manifest(tmp_path, gt_keys=("room_dimensions",))
    m["scenes"][2]["ground_truth"]["room_dimensions"] = {**GT_ITEM, "items": [{"name": "w", "value_m": 3.2}]}
    assert any("uncertainty_m" in p for p in validate_manifest(m, tmp_path))
    m = _manifest(tmp_path, gt_keys=("room_dimensions",))
    m["scenes"][2]["ground_truth"]["room_dimensions"] = {**GT_ITEM, "items": [{"name": "w", "value_m": float("nan"), "uncertainty_m": 0.01}]}
    assert any("non-positive or non-finite" in p for p in validate_manifest(m, tmp_path))


def test_missing_ground_truth_is_blocked_never_a_score(tmp_path):
    st = benchmark_status(_manifest(tmp_path), tmp_path)
    assert st["valid"] and not st["all_ready"]
    assert set(st["metrics"]) == set(METRIC_REQUIREMENTS)
    assert all(v["status"] == "BLOCKED" and "no " in v["reason"] for v in st["metrics"].values())
    assert not any(isinstance(v, (int, float)) for m in st["metrics"].values() for v in m.values())  # there is no number anywhere


def test_only_the_metrics_whose_ground_truth_exists_become_ready(tmp_path):
    st = benchmark_status(_manifest(tmp_path, gt_keys=("room_dimensions",)), tmp_path)
    assert st["metrics"]["room_geometry_error"]["status"] == "READY"
    assert st["metrics"]["object_dimension_error"]["status"] == "BLOCKED" and not st["all_ready"]
    assert "not yet evaluated" in st["metrics"]["room_geometry_error"]["reason"]  # READY is not a result


def test_no_held_out_test_scene_blocks_everything(tmp_path):
    m = _manifest(tmp_path, gt_keys=tuple(k for k, _ in METRIC_REQUIREMENTS.values()))
    m["scenes"] = [s for s in m["scenes"] if s["split"] != "test"]
    st = benchmark_status(m, tmp_path)
    assert all(v["status"] == "BLOCKED" and "no held-out test" in v["reason"] for v in st["metrics"].values())


def test_only_the_requested_kind_counts(tmp_path):
    m = _manifest(tmp_path, gt_keys=tuple(k for k, _ in METRIC_REQUIREMENTS.values()))
    assert benchmark_status(m, tmp_path, kind="interior")["all_ready"] is True
    assert benchmark_status(m, tmp_path, kind="exterior")["all_ready"] is False  # an interior benchmark is not an exterior one


def test_an_invalid_manifest_blocks_every_metric(tmp_path):
    m = _manifest(tmp_path, gt_keys=tuple(k for k, _ in METRIC_REQUIREMENTS.values()))
    bad = copy.deepcopy(m)
    bad["scenes"][2]["images"] = [bad["scenes"][0]["images"][0]]
    st = benchmark_status(bad, tmp_path)
    assert not st["valid"] and all(v["status"] == "BLOCKED" and "invalid" in v["reason"] for v in st["metrics"].values())


def test_load_manifest_refuses_garbage(tmp_path):
    (tmp_path / "m.json").write_text("{ nope", encoding="utf-8")
    with pytest.raises(ManifestError):
        load_manifest(tmp_path / "m.json")
    (tmp_path / "n.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ManifestError):
        load_manifest(tmp_path / "n.json")


def test_the_repository_ships_no_interior_benchmark_so_the_real_status_is_blocked():
    """The honest state of the repository: there is no furnished-interior manifest, so nothing is READY."""
    st = benchmark_status({"benchmark_id": "none", "scenes": []})
    assert st["test_scenes"] == 0 and all(v["status"] == "BLOCKED" for v in st["metrics"].values())
