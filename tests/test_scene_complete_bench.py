"""Runner and label checker for the scene-complete benchmark (build-plan 8.6 / 8.7).

The runner is tested with a stub detector: this proves the PLUMBING (photo -> prediction file -> scorer) and the honesty
rules (unknown classes are counted not hidden, a failed photo is a miss not a skip). It says nothing about detection quality:
that needs the real detector on labelled photographs (benchmarks/scene_complete/README.md).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.scene_complete.check import check
from benchmarks.scene_complete.run import run
from benchmarks.scene_complete.score import score

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402


def _img(path: Path, size=(200, 100)) -> Path:
    Image.new("RGB", size, (90, 90, 90)).save(path)
    return path


def _label(d: Path, stem: str, objects, source="real"):
    (d / f"{stem}.json").write_text(json.dumps({"image": f"{stem}.jpg", "source": source, "objects": objects}))


# ------------------------------------------------------------------------------------------------ runner


def test_runner_maps_coco_names_drops_unknown_classes_visibly_and_scores_end_to_end(tmp_path):
    imgs, out, labels = tmp_path / "i", tmp_path / "p", tmp_path / "l"
    imgs.mkdir(); labels.mkdir()
    a = _img(imgs / "a.jpg")

    def detect(path):
        return [("couch", 0.9, (10.0, 20.0, 110.0, 90.0)), ("person", 0.99, (0.0, 0.0, 5.0, 5.0)),
                ("tv", 0.8, (150.0, 10.0, 190.0, 40.0)), ("cup", 0.7, (1.0, 1.0, 2.0, 2.0))]

    report = run([a], out, detect)
    assert report["images_written"] == 1 and report["objects_kept"] == 2
    assert report["dropped_labels_not_in_vocabulary"] == {"person": 1, "cup": 1}      # visible, not silently lost
    doc = json.loads((out / "a.json").read_text())
    assert [(o["class"], o["tier"]) for o in doc["objects"]] == [("sofa", "furniture"), ("tv", "appliance")]
    assert doc["objects"][0]["box"] == [10.0, 20.0, 110.0, 90.0]                       # original pixel coordinates

    _label(labels, "a", [{"class": "sofa", "tier": "furniture", "box": [10, 20, 110, 90]},
                         {"class": "chair", "tier": "furniture", "box": [120, 50, 150, 90]}])
    r = score(labels, out)
    assert r["tiers"]["furniture"]["tp"] == 1 and r["tiers"]["furniture"]["fn"] == 1   # the chair was missed
    assert r["tiers"]["appliance"]["fp"] == 1                                          # the tv was not labelled


def test_a_photo_the_detector_cannot_read_gets_no_prediction_file_and_scores_as_a_miss(tmp_path):
    imgs, out, labels = tmp_path / "i", tmp_path / "p", tmp_path / "l"
    imgs.mkdir(); labels.mkdir()
    good, bad = _img(imgs / "good.jpg"), _img(imgs / "bad.jpg")

    def detect(path):
        if path.stem == "bad":
            raise RuntimeError("detector could not decode bad.jpg")
        return [("chair", 0.9, (0.0, 0.0, 10.0, 10.0))]

    report = run([good, bad], out, detect)
    assert report["images_failed"] == 1 and report["failures"][0]["image"] == "bad.jpg"
    assert (out / "good.json").exists() and not (out / "bad.json").exists()
    for stem in ("good", "bad"):
        _label(labels, stem, [{"class": "chair", "tier": "furniture", "box": [0, 0, 10, 10]}])
    r = score(labels, out)
    assert r["tiers"]["furniture"]["tp"] == 1 and r["tiers"]["furniture"]["fn"] == 1   # missing run = miss, not skip


def test_runner_writes_an_empty_prediction_for_a_photo_with_nothing_detected(tmp_path):
    img = _img(tmp_path / "x.jpg")
    run([img], tmp_path / "p", lambda p: [])
    assert json.loads((tmp_path / "p" / "x.json").read_text())["objects"] == []


# ------------------------------------------------------------------------------------------------ checker


def test_checker_accepts_a_clean_label(tmp_path):
    imgs, labels = tmp_path / "i", tmp_path / "l"
    imgs.mkdir(); labels.mkdir()
    _img(imgs / "r1.jpg")
    _label(labels, "r1", [{"class": "sofa", "tier": "furniture", "box": [10, 10, 100, 80], "support": "floor"}])
    res = check(labels, imgs)
    assert res["errors"] == [] and res["warnings"] == []


@pytest.mark.parametrize("obj, fragment", [
    ({"class": "Sofa", "tier": "furniture", "box": [10, 10, 50, 50]}, "lowercase"),
    ({"class": "wine rack", "tier": "furniture", "box": [10, 10, 50, 50]}, "lowercase"),
    ({"class": "sofa", "tier": "forniture", "box": [10, 10, 50, 50]}, "tier must be"),
    ({"class": "sofa", "tier": "furniture", "box": [50, 50, 10, 10]}, "no area"),
    ({"class": "sofa", "tier": "furniture", "box": [10, 10, 500, 50]}, "outside the 200x100 image"),
    ({"class": "sofa", "tier": "furniture", "box": [10, 10, 50]}, "box must be"),
    ({"class": "sofa", "tier": "furniture", "box": [10, 10, 50, 50], "support": "ceilng"}, "support must be"),
])
def test_checker_catches_each_labelling_mistake(tmp_path, obj, fragment):
    imgs, labels = tmp_path / "i", tmp_path / "l"
    imgs.mkdir(); labels.mkdir()
    _img(imgs / "r1.jpg")
    _label(labels, "r1", [obj])
    errors = check(labels, imgs)["errors"]
    assert any(fragment in e for e in errors), errors


def test_checker_catches_duplicates_orphans_bad_source_and_bad_json(tmp_path):
    imgs, labels = tmp_path / "i", tmp_path / "l"
    imgs.mkdir(); labels.mkdir()
    _img(imgs / "has_label.jpg"); _img(imgs / "no_label.jpg")
    o = {"class": "chair", "tier": "furniture", "box": [1, 1, 20, 20]}
    _label(labels, "has_label", [o, dict(o)], source="photo")
    _label(labels, "orphan", [])
    (labels / "broken.json").write_text("{not json")
    errors = " | ".join(check(labels, imgs)["errors"])
    for fragment in ("duplicate", "no image named orphan", "no label file no_label.json", "source must be", "not valid JSON"):
        assert fragment in errors, (fragment, errors)


def test_shipped_demo_label_passes_the_checker():
    base = Path(__file__).resolve().parent.parent / "benchmarks" / "scene_complete"
    assert check(base / "labels", None)["errors"] == []
