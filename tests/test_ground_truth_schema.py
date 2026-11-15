"""E205: the measured-room ground-truth schema accepts the synthetic fixture and rejects bad files."""

from __future__ import annotations

import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "benchmarks" / "measured_rooms" / "ground_truth.schema.json"
SYNTHETIC_PATH = ROOT / "synthetic" / "fidelity_room" / "ground_truth.json"

CAPTURE = {
    "room_id": "room-a",
    "date": "2026-11-14",
    "device_model": "Bosch GLM 50-27 C",
    "measured_by": "tester-1",
    "method": "laser",
    "tolerance_m": 0.002,
    "consent_reference": "consent-0001",
    "lighting": "daylight, overcast",
    "notes": "",
}


def _validator():
    from jsonschema import Draft202012Validator  # dev dependency; imported here like the other schema tests

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _errors(doc: dict) -> list[str]:
    return [e.message for e in _validator().iter_errors(doc)]


def _synthetic() -> dict:
    return json.loads(SYNTHETIC_PATH.read_text(encoding="utf-8"))


def _measured() -> dict:
    doc = _synthetic()
    doc["source"] = "measured"
    doc["capture"] = copy.deepcopy(CAPTURE)
    return doc


def test_synthetic_fixture_validates_unchanged_without_capture():
    doc = _synthetic()
    assert "capture" not in doc
    assert _errors(doc) == []


def test_synthetic_true_flag_also_waives_capture():
    doc = _synthetic()
    del doc["source"]
    doc["synthetic"] = True
    assert _errors(doc) == []


def test_complete_measured_file_validates():
    assert _errors(_measured()) == []


def test_missing_walls_fails():
    doc = _synthetic()
    del doc["walls"]
    assert any("'walls' is a required property" in m for m in _errors(doc))


def test_unknown_opening_kind_fails():
    doc = _synthetic()
    doc["openings"][0]["kind"] = "portal"
    assert any("'portal' is not one of" in m for m in _errors(doc))


def test_measured_file_without_capture_fails():
    doc = _measured()
    del doc["capture"]
    assert any("'capture' is a required property" in m for m in _errors(doc))


def test_measured_file_without_consent_reference_fails():
    doc = _measured()
    del doc["capture"]["consent_reference"]
    assert any("'consent_reference' is a required property" in m for m in _errors(doc))


def test_empty_consent_reference_and_bad_method_and_bad_date_fail():
    for key, bad in (("consent_reference", ""), ("method", "eyeball"), ("date", "14/11/2026")):
        doc = _measured()
        doc["capture"][key] = bad
        assert _errors(doc), key


def test_zero_openings_is_valid_but_a_missing_openings_key_is_not():
    doc = _synthetic()
    doc["openings"] = []
    assert _errors(doc) == []
    del doc["openings"]
    assert _errors(doc)


def test_malformed_geometry_fails():
    doc = _synthetic()
    doc["walls"][0]["start"] = [0]  # not a 2D point
    doc["openings"][0]["center"] = [1, 2]  # not a 3D point
    assert len(_errors(doc)) >= 2
