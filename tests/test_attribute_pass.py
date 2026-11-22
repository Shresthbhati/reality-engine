"""B05 live path: the attribute pass fills attrs_json, combines views per entity, respects budget and small crops."""
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest

from perception.detection.attribute_pass import attribute_phrase, run_attribute_pass
from world_ir.observation_table import ObservationTable

LOCK = NS(name="lock", label="Lock", attributes={"lock_type": ("key", "numpad", "unknown")})
ONT = NS(get=lambda path: {"fix/lock": LOCK}.get(path))


def table(rows):
    t = ObservationTable.empty(len(rows))
    for i, (cls, box, ent) in enumerate(rows):
        t.obs_id[i], t.image_id[i], t.class_path[i], t.bbox_px[i], t.entity_row[i] = f"o{i}", "img", cls, box, ent
    return t


IMG = lambda _id: np.zeros((300, 300, 3), np.uint8)
numpad_wins = lambda crop, phrase: 0.9 if phrase == attribute_phrase("lock_type", "numpad") else 0.1
BOX = (50, 50, 150, 150)


def test_rows_are_classified_and_written_to_attrs_json():
    t = table([("fix/lock", BOX, 0), ("fix/lock", BOX, 0)])
    out = run_attribute_pass(t, IMG, numpad_wins, ontology=ONT)
    rec = json.loads(str(t.attrs_json[0]))
    assert rec["lock_type"]["value"] == "numpad" and rec["lock_type"]["basis"] == "visual"
    assert out["rows_classified"] == 2 and out["attributes_decided"] == 2 and out["detector_calls"] == 4
    assert out["entity_attributes"][0]["lock_type"]["value"] == "numpad"        # two views of entity 0 combined


def test_unknown_node_and_small_crop_are_counted_not_classified():
    t = table([("fix/unknown", BOX, -1), ("fix/lock", (0, 0, 20, 20), -1)])
    out = run_attribute_pass(t, IMG, numpad_wins, ontology=ONT)
    assert out["no_attributes"] == 1 and out["crop_too_small"] == 1 and "rows_classified" not in out
    assert str(t.attrs_json[0]) == "" and str(t.attrs_json[1]) == ""


def test_budget_stops_detector_calls():
    t = table([("fix/lock", BOX, -1)] * 3)
    out = run_attribute_pass(t, IMG, numpad_wins, ontology=ONT, budget_calls=4)
    assert out["detector_calls"] == 4 and out["rows_classified"] == 2 and out["budget_exhausted"] == 1


def test_ambiguous_scores_stay_unknown():
    t = table([("fix/lock", BOX, -1)])
    out = run_attribute_pass(t, IMG, lambda crop, phrase: 0.5, ontology=ONT)
    assert json.loads(str(t.attrs_json[0]))["lock_type"]["value"] is None and out["attributes_unknown"] == 1


def test_attributes_without_vetted_phrases_are_unknown_with_a_reason_and_cost_no_calls():
    door = NS(name="door", label="Door", attributes={"swing": ("left", "right", "unknown"), "pane_count": ("1", "2", "unknown")})
    t = table([("arch/door", BOX, -1)])
    calls = []
    out = run_attribute_pass(t, IMG, lambda c, p: calls.append(p) or 0.9, ontology=NS(get=lambda path: door))
    rec = json.loads(str(t.attrs_json[0]))
    assert rec["swing"] == {"value": None, "p": 0.0, "basis": "none", "obs": [], "reason": "no vetted detector phrase"} and calls == []
    assert out["no_vetted_phrase"] == 2 and out["detector_calls"] == 0 and "attributes_decided" not in out


def test_one_vetted_phrase_cannot_discriminate_so_the_attribute_is_not_attempted():
    odd = NS(name="odd", label="Odd", attributes={"lock_type": ("key", "none", "unknown")})   # 'none' has no phrase
    t = table([("x", BOX, -1)])
    out = run_attribute_pass(t, IMG, numpad_wins, ontology=NS(get=lambda path: odd))
    assert out["no_vetted_phrase"] == 1 and out["detector_calls"] == 0


def test_a_scorer_with_many_is_called_once_per_attribute_not_once_per_value():
    batches, singles = [], []

    def one(crop, phrase):
        singles.append(phrase)
        return 0.1

    one.many = lambda crop, phrases: batches.append(list(phrases)) or [0.9 if p == attribute_phrase("lock_type", "numpad") else 0.1 for p in phrases]
    t = table([("fix/lock", BOX, -1)])
    out = run_attribute_pass(t, IMG, one, ontology=ONT)
    assert singles == [] and len(batches) == 1 and len(batches[0]) == 2
    assert json.loads(str(t.attrs_json[0]))["lock_type"]["value"] == "numpad" and out["detector_calls"] == 2


@pytest.mark.slow
def test_real_grounded_scorer_many_returns_one_score_per_phrase():
    import pytest

    from perception.detection.attribute_pass import grounded_scorer

    try:
        got = grounded_scorer().many(np.full((120, 120, 3), 200, np.uint8), ["keyhole lock", "padlock", "rocker light switch"])
    except Exception as exc:  # noqa: BLE001 -- model not cached / no GPU
        pytest.skip(f"Grounding DINO unavailable: {exc}")
    assert len(got) == 3 and all(isinstance(s, float) and 0.0 <= s <= 1.0 for s in got)
