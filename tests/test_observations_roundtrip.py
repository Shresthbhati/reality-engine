"""Blueprint 3.1: the observation ledger is saved inside each WorldIR version and round-trips."""
from __future__ import annotations

from world_ir import WorldIR
from world_ir.observations import (
    AssociationDecision, AssociationState, DepthSource, Observation, Visibility, attach_ledger, ledger_of,
)
from worldstore.store import WorldStore


def _full_obs() -> Observation:
    return Observation(
        observation_id="o1", image_id="img1", label="door", world_pos=(1.0, 2.0, 3.0), sigma_m=0.1,
        depth_source=DepthSource.RELATIVE, model="maskrcnn@1", license="BSD-3-Clause", normal=(0.0, 0.0, 1.0),
        bounds_min=(0.0, 0.0, 0.0), bounds_max=(1.0, 2.0, 0.1), confidence=0.7, mask_uri="artifact://m1",
        bbox_px=(1.0, 2.0, 30.0, 40.0), visibility=Visibility.PARTIAL, embedding_ref="emb://o1",
        session_id="sess-1", created_at="2026-01-01T00:00:00Z")


def _decision() -> AssociationDecision:
    return AssociationDecision("o1", AssociationState.AMBIGUOUS, "ent-1", 0.4, "tied", 0.2, 1.5, 3.0, 0.5, None,
                               (("ent-1", 0.4), ("ent-2", 0.39)))


def test_ledger_round_trips_through_world_json():
    w = WorldIR(id="w1", name="w")
    attach_ledger(w, [_full_obs()], [_decision()])
    w2 = WorldIR.from_json(w.to_json())
    obs, dec = ledger_of(w2)
    assert obs == (_full_obs(),)
    assert dec == (_decision(),)
    assert obs[0].visibility is Visibility.PARTIAL and obs[0].bbox_px == (1.0, 2.0, 30.0, 40.0)


def test_ledger_is_saved_inside_each_stored_version(tmp_path):
    store = WorldStore(tmp_path)
    w = WorldIR(id="w1", name="w")
    attach_ledger(w, [_full_obs()], [_decision()])
    v = store.save_version(w, parent=None, version_id="v-1")
    obs, dec = ledger_of(store.load_version(v.version_id))
    assert obs == (_full_obs(),) and dec == (_decision(),)


def test_unmeasured_optional_fields_stay_none():
    o = Observation("o2", "img", "chair", (0.0, 0.0, 0.0), 0.1, DepthSource.MVS)
    d = o.to_dict()
    assert all(d[k] is None for k in ("mask_uri", "bbox_px", "visibility", "embedding_ref", "session_id", "created_at"))
    assert Observation.from_dict(d) == o


def test_version_written_before_the_ledger_existed_still_loads():
    data = WorldIR(id="old", name="old").to_dict()
    assert "observation_ledger" not in data["metadata"]
    obs, dec = ledger_of(WorldIR.from_dict(data))
    assert obs == () and dec == ()
