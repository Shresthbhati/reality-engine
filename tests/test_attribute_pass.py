"""B05 live path: the attribute pass fills attrs_json, combines views per entity, respects budget and small crops."""
import json
from types import SimpleNamespace as NS

import numpy as np

from perception.detection.attribute_pass import attribute_phrase, run_attribute_pass
from world_ir.observation_table import ObservationTable

LOCK = NS(name="lock", label="Lock", attributes={"type": ("key", "numpad", "unknown")})
ONT = NS(get=lambda path: {"fix/lock": LOCK}.get(path))


def table(rows):
    t = ObservationTable.empty(len(rows))
    for i, (cls, box, ent) in enumerate(rows):
        t.obs_id[i], t.image_id[i], t.class_path[i], t.bbox_px[i], t.entity_row[i] = f"o{i}", "img", cls, box, ent
    return t


IMG = lambda _id: np.zeros((300, 300, 3), np.uint8)
numpad_wins = lambda crop, phrase: 0.9 if phrase == attribute_phrase(LOCK, "numpad") else 0.1
BOX = (50, 50, 150, 150)


def test_rows_are_classified_and_written_to_attrs_json():
    t = table([("fix/lock", BOX, 0), ("fix/lock", BOX, 0)])
    out = run_attribute_pass(t, IMG, numpad_wins, ontology=ONT)
    rec = json.loads(str(t.attrs_json[0]))
    assert rec["type"]["value"] == "numpad" and rec["type"]["basis"] == "visual"
    assert out["rows_classified"] == 2 and out["attributes_decided"] == 2 and out["detector_calls"] == 4
    assert out["entity_attributes"][0]["type"]["value"] == "numpad"        # two views of entity 0 combined


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
    assert json.loads(str(t.attrs_json[0]))["type"]["value"] is None and out["attributes_unknown"] == 1
