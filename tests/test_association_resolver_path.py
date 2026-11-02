"""Blueprint 3.2: association is the only resolver path; every decision is stored in the ledger."""
from __future__ import annotations

import pytest

from engine.math import Vec3
from perception.instances.lifting import ObjectHypothesis3D
from perception.instances.object_resolution import merge_hypotheses, resolve_objects
from provenance import Uncertainty
from world_ir import WorldIR
from world_ir.observations import DepthSource, ledger_of


def _hyp(region_id, evidence_id, x, label="door"):
    return ObjectHypothesis3D(
        region_id=region_id, evidence_id=evidence_id, label=label, position=Vec3(x, 0, 0),
        bounds_min=Vec3(x - 0.4, -0.1, 0), bounds_max=Vec3(x + 0.4, 0.1, 2.0), point_count=50, mask_pixel_count=60,
        confidence=0.8, uncertainty=Uncertainty(confidence=0.8))


def _two_doors():
    # two similar doors 3 m apart, each seen in 4 images with small jitter (A4)
    return [_hyp(f"{k}{i}", f"img{i}", x + 0.01 * i) for i in range(4) for k, x in (("a", 0.0), ("b", 3.0))]


def test_default_resolver_keeps_two_distinct_doors_two_and_stores_every_decision():
    world = WorldIR(id="w", name="w")
    hyps = _two_doors()
    candidates, result = resolve_objects(hyps, depth_source=DepthSource.METRIC, world=world)
    assert len(candidates) == 2 and len(result.entities) == 2
    obs, decisions = ledger_of(WorldIR.from_json(world.to_json()))  # stored in the version, survives serialisation
    assert len(obs) == len(hyps)
    assert {d.observation_id for d in decisions} == {o.observation_id for o in obs}  # one decision per observation
    assert all(d.reason for d in decisions)


def test_legacy_union_find_is_explicit_and_warns():
    with pytest.warns(DeprecationWarning, match="legacy union-find"):
        merge_hypotheses(_two_doors()[:2])


def test_no_resolver_default_is_union_find():
    from engine.pipeline.vertical_slice import VerticalSliceOptions
    from perception.instances.track_backend import MultiViewIdentityTrackBackend
    import inspect

    assert VerticalSliceOptions().object_resolver == "association"
    assert "merge_hypotheses" not in inspect.getsource(MultiViewIdentityTrackBackend)
