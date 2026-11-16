"""Florence-2 second opinion: agreement between two detectors is reported as such, a missing model is reported not faked."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from engine.pipeline.vertical_slice import _second_opinion_facts
from perception.detection import florence2_detector as f2
from perception.detection.florence2_detector import Florence2Unavailable, box_agreement

PRIMARY = [{"label": "sofa", "bbox_px": [100.0, 100.0, 300.0, 200.0]}, {"label": "rug", "bbox_px": [0.0, 300.0, 600.0, 470.0]}]
SECOND = [{"label": "couch", "box_xyxy": (105.0, 98.0, 298.0, 205.0)}, {"label": "lamp", "box_xyxy": (500.0, 10.0, 540.0, 90.0)}]


def test_box_agreement_pairs_overlapping_boxes_and_lists_the_rest():
    r = box_agreement(PRIMARY, SECOND)
    assert [(m["primary"], m["second"]) for m in r["matched"]] == [("sofa", "couch")]
    assert [p["label"] for p in r["primary_only"]] == ["rug"] and [s["label"] for s in r["second_only"]] == ["lamp"]


def test_one_second_box_cannot_match_two_primary_boxes():
    r = box_agreement([{"label": "a", "bbox_px": [0, 0, 10, 10]}, {"label": "b", "bbox_px": [0, 0, 10, 10]}],
                      [{"label": "x", "box_xyxy": (0, 0, 10, 10)}])
    assert len(r["matched"]) == 1 and len(r["primary_only"]) == 1


def _photo(tmp_path):
    p = tmp_path / "room.png"
    Image.fromarray(np.full((480, 640, 3), 120, np.uint8)).save(p)
    return SimpleNamespace(id="e1", source_uri=str(p))


class _FakeDetector:
    def detect(self, image):
        return [{"label": "couch", "box_xyxy": (105.0, 98.0, 298.0, 205.0)}]

    def unload(self):
        pass


def test_disabled_records_nothing(tmp_path):
    assert _second_opinion_facts(False, [_photo(tmp_path)], {"e1": PRIMARY}) is None


def test_second_opinion_facts_report_agreement_per_photo(tmp_path, monkeypatch):
    monkeypatch.setattr(f2, "Florence2Detector", _FakeDetector)
    facts = _second_opinion_facts(True, [_photo(tmp_path)], {"e1": PRIMARY})
    assert facts["status"] == "ran" and facts["detector"] == f2.MODEL_ID
    assert facts["totals"] == {"matched": 1, "primary_only": 1, "second_only": 0}
    assert "not accuracy" in facts["note"] and facts["per_view"]["e1"]["matched"][0]["second"] == "couch"


def test_missing_checkpoint_is_a_recorded_status(tmp_path, monkeypatch):
    class _Down:
        def detect(self, image):
            raise Florence2Unavailable("not cached; fetch once with ...")

        def unload(self):
            pass

    monkeypatch.setattr(f2, "Florence2Detector", _Down)
    facts = _second_opinion_facts(True, [_photo(tmp_path)], {"e1": PRIMARY})
    assert facts["status"] == "unavailable" and "fetch once" in facts["note"]


def test_real_checkpoint_loads_or_says_how_to_fetch():
    try:
        f2.Florence2Detector()._load()
    except Florence2Unavailable as exc:
        assert "snapshot_download" in str(exc)
    else:
        pytest.skip("Florence-2 is cached here")
