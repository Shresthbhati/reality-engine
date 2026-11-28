"""P0 tests for PR #467: pipeline cache integrity.

A cached StageResult may be reused ONLY when declared inputs, version,
params (model identity/frame), producer/consumer lineage and required
outputs all validate. An invalid hit must be discarded, never dereferenced.
"""
from __future__ import annotations

from datetime import datetime, timezone

from engine.pipeline.stages.base import RunManifest, StageContext, StageResult, StageSpec, compute_inputs_hash
from engine.pipeline.stages.runner import _cache_result_valid, run_stages
from engine.pipeline.stages.lineage import LineageGraph


def _now():
    return datetime.now(timezone.utc)


def _result(name="stage", status="succeeded", inputs_hash="h", outputs=None, requires=()):
    return StageResult(
        name=name, status=status, inputs_hash=inputs_hash,
        outputs=outputs if outputs is not None else {f"{name}_out": "hash"},
        metrics={}, warnings=[], error=None, started_at=_now(), ended_at=_now(), peak_rss_mb=None,
    )


def _spec(name="stage", inputs=("a",), requires=("stage_out",), version="1"):
    return StageSpec(name=name, version=version, resource_class="cpu", required=True,
                     inputs=inputs, requires_outputs=requires)


def _ctx(**artifacts):
    return StageContext(job_id="j1", workdir="/tmp", artifacts=dict(artifacts),
                        params={"model": "m1", "frame": "world"}, state={})


def _manifest():
    return RunManifest(pipeline_profile="p", pipeline_version="v", stages=[])


def test_valid_cache_reused():
    spec, ctx, man = _spec(), _ctx(a="hash-a"), _manifest()
    h = compute_inputs_hash("1", {"a": "hash-a"}, ctx.params)
    hit = _result(inputs_hash=h, outputs={"stage_out": "out-hash"})
    assert _cache_result_valid(hit, spec, ("a",), ctx, h, man) is True


def test_changed_input_invalidates():
    """A changed input produces a different deterministic cache key -> checkpoint miss."""
    spec, ctx, man = _spec(), _ctx(a="hash-a"), _manifest()
    h_old = compute_inputs_hash("1", {"a": "hash-a"}, ctx.params)
    hit = _result(inputs_hash=h_old)  # cached under the OLD input hash
    ctx.artifacts["a"] = "hash-b"    # input changed
    h_new = compute_inputs_hash("1", {"a": "hash-b"}, ctx.params)
    assert h_new != h_old
    # The old hit fails validation against the NEW computed hash -> not reused.
    assert _cache_result_valid(hit, spec, ("a",), ctx, h_new, man) is False


def test_missing_declared_input_invalidates():
    spec, ctx, man = _spec(), _ctx(), _manifest()  # no 'a'
    h = compute_inputs_hash("1", {}, ctx.params)
    hit = _result(inputs_hash=h)
    assert _cache_result_valid(hit, spec, ("a",), ctx, h, man) is False


def test_model_or_frame_change_invalidates():
    ctx = _ctx(a="hash-a")
    h = compute_inputs_hash("1", {"a": "hash-a"}, ctx.params)
    hit = _result(inputs_hash=h)
    ctx.params["model"] = "m2"  # model identity changed
    new_h = compute_inputs_hash("1", {"a": "hash-a"}, ctx.params)
    assert new_h != h
    assert _cache_result_valid(hit, _spec(), ("a",), ctx, new_h, _manifest()) is False


def test_wrong_inputs_hash_invalidates():
    ctx = _ctx(a="hash-a")
    hit = _result(inputs_hash="not-the-hash")
    assert _cache_result_valid(hit, _spec(), ("a",), ctx,
                               compute_inputs_hash("1", {"a": "hash-a"}, ctx.params), _manifest()) is False


def test_required_output_missing_invalidates():
    ctx = _ctx(a="hash-a")
    h = compute_inputs_hash("1", {"a": "hash-a"}, ctx.params)
    hit = _result(inputs_hash=h, outputs={})  # required output missing
    assert _cache_result_valid(hit, _spec(requires=("stage_out",)), ("a",), ctx, h, _manifest()) is False


def test_different_producer_invalidates():
    ctx = _ctx(a="hash-a")
    h = compute_inputs_hash("1", {"a": "hash-a"}, ctx.params)
    hit = _result(inputs_hash=h)
    man = _manifest()
    man.lineage.producer["stage_out"] = "other-stage"  # another stage owns this artifact
    assert _cache_result_valid(hit, _spec(), ("a",), ctx, h, man) is False


def test_failed_status_never_reused():
    ctx = _ctx(a="hash-a")
    h = compute_inputs_hash("1", {"a": "hash-a"}, ctx.params)
    hit = _result(inputs_hash=h, status="failed")
    assert _cache_result_valid(hit, _spec(), ("a",), ctx, h, _manifest()) is False


def test_none_hit_is_safe():
    ctx = _ctx(a="hash-a")
    assert _cache_result_valid(None, _spec(), ("a",), ctx, "h", _manifest()) is False


def test_checkpoint_store_never_sees_invalid_reuse():
    """run_stages must not reuse an invalid hit even if the store returns one."""
    class Store:
        def __init__(self, hit): self.hit = hit; self.lookups = 0
        def lookup(self, job, name, ih): self.lookups += 1; return self.hit
        def record(self, job, result): pass

    ctx = _ctx(a="hash-a")
    bad_hit = _result(inputs_hash="wrong-hash", outputs={"stage_out": "h"})
    store = Store(bad_hit)
    ran = []
    class Stage:
        spec = _spec()
        def run(self, c):
            ran.append("executed")
            return _result(inputs_hash="x", outputs={"stage_out": "fresh"})
    man = run_stages([Stage()], ctx, checkpoints=store)
    assert ran == ["executed"], "invalid cache hit must not suppress execution"
    assert man.stages[0].outputs["stage_out"] == "fresh"
