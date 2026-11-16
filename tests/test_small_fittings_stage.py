"""The opt-in small-fittings pass records unverified 2D detections as facts; it never promotes them and never crashes the stage."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from PIL import Image

from engine.pipeline.vertical_slice import _small_fittings_facts
from perception.detection import grounded_small
from perception.detection.grounded_small import RawDetection


def _photo(tmp_path):
    p = tmp_path / "room.png"
    Image.fromarray(np.full((960, 1280, 3), 120, np.uint8)).save(p)
    return SimpleNamespace(id="e1", source_uri=str(p))


class _Fake:
    threshold = 0.45

    def available(self):
        return True, ""

    def detect(self, image_rgb, queries):
        # one socket per call, in whichever tile or coarse pass it is asked
        return [RawDetection("wall socket", 0.8, (10.0, 10.0, 50.0, 50.0))] if "wall socket" in queries else []


def test_disabled_records_nothing(tmp_path):
    assert _small_fittings_facts(False, [_photo(tmp_path)], {"e1": object()}) is None


def test_detections_are_recorded_as_facts_and_never_promoted(tmp_path, monkeypatch):
    monkeypatch.setattr(grounded_small, "GroundedSmallDetector", _Fake)
    facts = _small_fittings_facts(True, [_photo(tmp_path)], {"e1": object()})
    assert facts["status"] == "ran" and facts["promoted_to_entities"] is False
    assert facts["detections_by_class"].get("electrical.switchboard.module.socket", 0) >= 1
    assert facts["tiles_run"] > 0 and facts["per_view"]["e1"] and facts["detections"]
    assert "no accuracy claim" in facts["note"]


def test_a_photograph_without_depth_is_skipped(tmp_path, monkeypatch):
    monkeypatch.setattr(grounded_small, "GroundedSmallDetector", _Fake)
    facts = _small_fittings_facts(True, [_photo(tmp_path)], {})
    assert facts["status"] == "ran" and facts["detections_by_class"] == {} and facts["tiles_run"] == 0


def test_unavailable_detector_is_reported_not_faked(tmp_path, monkeypatch):
    class _Down(_Fake):
        def available(self):
            return False, "Grounding DINO is not available locally: test"

    monkeypatch.setattr(grounded_small, "GroundedSmallDetector", _Down)
    facts = _small_fittings_facts(True, [_photo(tmp_path)], {"e1": object()})
    assert facts["status"] == "unavailable" and facts["promoted_to_entities"] is False and "not available" in facts["note"]


def test_a_failure_is_a_recorded_status_not_a_crash(tmp_path, monkeypatch):
    class _Boom(_Fake):
        def detect(self, image_rgb, queries):
            raise RuntimeError("cuda out of memory")

    monkeypatch.setattr(grounded_small, "GroundedSmallDetector", _Boom)
    facts = _small_fittings_facts(True, [_photo(tmp_path)], {"e1": object()})
    assert facts["status"] == "failed" and "out of memory" in facts["note"] and facts["promoted_to_entities"] is False
