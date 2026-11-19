"""WorldIR optional scalar confidence (agreed contract): None = "not measured". This is the CONSUMER AUDIT as a test: a world holding
legacy (default 0.5), measured and None confidences is pushed through serialization, the world store, validation, every consumer that
reads entity confidence, and every exporter. SYNTHETIC world; proves compatibility, not any accuracy."""

from __future__ import annotations

import json

import pytest

from engine.inspector.inspector import Inspector
from engine.quality.report import mean_confidence
from engine.scene_graph.query import WorldQuery
from provenance import Provenance
from sdk.reality import EXPORT_FORMATS, export
from tests.test_multiengine_worldir_bridge import Builder, _canon, _chair
from world_ir.diff import diff_worlds
from world_ir.schema_v1 import Entity, EntityType
from world_ir.validation import validate_world_ir
from world_ir.world_v1 import WorldIR
from worldstore.store import WorldStore

BASIS = {"kind": "calibrated", "dataset": "synthetic-v1", "method": "isotonic", "version": "1.0"}


def _bridge_world() -> WorldIR:
    from engine.multiengine.worldir_bridge import convert

    b = Builder()
    res = convert([_canon(b, _chair(b), "chair")], b.m)
    w = WorldIR(id="world-opt", name="opt")
    res.apply_to(w)
    return w


def _with_others(w: WorldIR) -> WorldIR:
    w.entities["ent-legacy"] = Entity(id="ent-legacy", type=EntityType.FURNITURE, provenance=Provenance.OBSERVED)  # default 0.5
    w.entities["ent-measured"] = Entity(
        id="ent-measured", type=EntityType.FURNITURE, provenance=Provenance.OBSERVED, confidence=0.8,
        confidence_basis=dict(BASIS),
    )
    return w


def _none_ids(w):
    return [i for i, e in w.entities.items() if e.confidence is None]


def test_none_and_basis_round_trip_and_legacy_worlds_load_unchanged():
    w = _with_others(_bridge_world())
    assert len(_none_ids(w)) == 1
    again = WorldIR.from_dict(json.loads(json.dumps(w.to_dict(), sort_keys=True)))
    assert again.to_dict() == w.to_dict()
    assert again.entities["ent-measured"].confidence_basis == BASIS and again.entities["ent-legacy"].confidence == 0.5
    d = w.entities["ent-legacy"].to_dict()
    assert "confidence_basis" not in d and d["confidence"] == 0.5  # legacy entities serialize byte-identically
    none_d = w.entities[_none_ids(w)[0]].to_dict()
    assert none_d["confidence"] is None and "confidence_basis" not in none_d


def test_migration_a_world_saved_before_the_field_existed_still_loads():
    legacy = Entity(id="e", type=EntityType.FURNITURE).to_dict()
    assert Entity.from_dict(legacy).confidence == 0.5 and Entity.from_dict(legacy).confidence_basis is None
    no_key = dict(legacy)
    del no_key["confidence"]  # an even older record with no confidence key at all
    assert Entity.from_dict(no_key).confidence == 0.5  # a MISSING key keeps the old default...
    assert Entity.from_dict(dict(legacy, confidence=None)).confidence is None  # ...an explicit null means "not measured"


def test_world_store_saves_verifies_and_reloads_none_confidence(tmp_path):
    w = _with_others(_bridge_world())
    store = WorldStore(tmp_path)
    v = store.save_version(w, parent=None)
    assert store.verify_version(v.version_id) == []
    back = store.load_version(v.version_id)
    assert back.entities[_none_ids(w)[0]].confidence is None
    assert back.entities["ent-measured"].confidence_basis == BASIS


def test_validation_accepts_none_but_still_rejects_bad_numbers():
    w = _with_others(_bridge_world())
    assert "confidence_out_of_range" not in [i.code for i in validate_world_ir(w).issues]
    w.entities["ent-legacy"].confidence = 1.5
    w.entities["ent-measured"].confidence = "high"  # type: ignore[assignment]
    bad = [i for i in validate_world_ir(w).issues if i.code == "confidence_out_of_range"]
    assert len(bad) == 2  # None is valid; out-of-range and non-numeric are still ERRORs


def test_consumers_treat_none_as_not_measured():
    w = _with_others(_bridge_world())
    nid = _none_ids(w)[0]
    assert nid in {e.id for e in Inspector(w).low_confidence_entities(0.5)}  # unmeasured is listed, never passes
    assert mean_confidence(w) == pytest.approx((0.5 + 0.8) / 2)  # measured only; None is not counted as 0 or 0.5
    q = WorldQuery(w)
    assert nid not in {e.id for e in q.by_confidence(0.1)}  # cannot be shown to meet a threshold
    assert nid in {e.id for e in q.by_confidence(0.0)}  # no threshold: nothing to fail
    only_none = WorldIR(id="n", name="n")
    only_none.entities[nid] = w.entities[nid]
    assert mean_confidence(only_none) is None


def test_diff_and_change_assessment_handle_none_on_either_side():
    from worldstore.proposals import _assess

    base = _with_others(_bridge_world())
    cand = WorldIR.from_dict(base.to_dict())
    nid = _none_ids(base)[0]
    cand.entities[nid].confidence = 0.6  # a measured value appears
    cand.entities[nid].transform = {"position": {"x": 99.0, "y": 0.0, "z": 0.0}, "frame_id": "world"}
    assert _assess(base, cand, diff_worlds(base, cand))  # None -> number, with moved geometry: no TypeError
    assert _assess(cand, base, diff_worlds(cand, base))  # number -> None
    added = WorldIR.from_dict(base.to_dict())
    added.entities["ent-new"] = Entity(id="ent-new", type=EntityType.FURNITURE, confidence=None, provenance=Provenance.OBSERVED)
    res = _assess(base, added, diff_worlds(base, added))
    assert [a for a in res if a.entity_id == "ent-new"][0].verdict != "supported"  # no observation + no confidence: review


@pytest.mark.parametrize("fmt", sorted(EXPORT_FORMATS))
def test_every_exporter_survives_none_confidence(fmt):
    """An exporter that works on the same world with numeric confidences must not break (or write 'None' into a numeric field)
    when an entity's confidence is None."""
    withnone = _with_others(_bridge_world())
    numeric = WorldIR.from_dict(withnone.to_dict())
    for e in numeric.entities.values():
        if e.confidence is None:
            e.confidence = 0.5
    try:
        export(numeric, fmt)
    except Exception as exc:  # noqa: BLE001 -- the control must work, otherwise the failure is not about confidence
        pytest.skip(f"{fmt} cannot export this synthetic world even with numeric confidences: {type(exc).__name__}")
    out = export(withnone, fmt)  # must not raise
    blob = out if isinstance(out, (str, bytes)) else json.dumps(out, default=str)
    text = blob.decode("utf-8", "ignore") if isinstance(blob, bytes) else blob
    if fmt == "citygml":
        assert "<gen:value>None</gen:value>" not in text
