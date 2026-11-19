"""Multi-engine foundation: registry, observations, adapter contract, execution manager, evidence store.

Every adapter here is a TEST DOUBLE (``FakeAdapter``) with scripted behaviour. These tests prove the framework's rules
(no fake availability, no silent fallback, no overwritten evidence, failures never read as success). They say nothing about
any real model's accuracy; the last test checks the default registry against the real environment's probes.
"""
from __future__ import annotations

import json
from dataclasses import replace

import pytest

from engine.multiengine.contract import (
    Availability, AvailabilityStatus, Device, EngineAdapter, EngineError, EngineErrorKind, EngineRequest, EngineRunStatus,
    EngineSpec, HardwareRequirement, Task,
)
from engine.multiengine.evidence_store import EvidenceStore, EvidenceStoreError
from engine.multiengine.executor import ExecutionManager, Policy, ReportStatus
from engine.multiengine.observation import ObservationDraft, ObservationError, make_observation
from engine.multiengine.registry import EngineRegistry, RegistryError, default_registry

OK = Availability(AvailabilityStatus.AVAILABLE, "verified in test", evidence=("test double",))


def blocked(reason="weights missing: checkpoint not cached"):
    return Availability(AvailabilityStatus.BLOCKED, reason, "download it")


def spec(eid, *, task=Task.DEPTH, probe=None, priority=100, family="", commercial=True, hw=None, version="1"):
    return EngineSpec(engine_id=eid, name=eid, version=version, tasks=frozenset({task}), input_schema="img", output_schema="d",
                      probe=probe or (lambda: OK), family=family, priority=priority, commercial_ok=commercial,
                      licence="Apache-2.0" if commercial else "CC-BY-NC-4.0", hardware=hw or HardwareRequirement())


def draft(value=1.0, frame=None, cs="cam:img1"):
    return ObservationDraft(task="depth", quantity="depth_value", source_id="img1", coordinate_system=cs,
                            predicted={"v": value}, frame=frame)


class FakeAdapter(EngineAdapter):
    """TEST DOUBLE. ``script`` is a list of behaviours consumed per ``infer`` call: a float value, or an Exception to raise."""

    def __init__(self, sp, script=None, value=1.0, validate_fail=False):
        self.spec = sp
        self.script = list(script or [])
        self.value = value
        self.validate_fail = validate_fail
        self.loaded = 0
        self.unloaded = 0
        self.devices = []

    def validate_input(self, request):
        if self.validate_fail:
            raise EngineError(EngineErrorKind.INVALID_INPUT, "needs at least two sources")

    def load(self, device):
        self.loaded += 1
        self.devices.append(str(device))

    def unload(self):
        self.unloaded += 1

    def infer(self, request):
        step = self.script.pop(0) if self.script else self.value
        if isinstance(step, BaseException):
            raise step
        return [draft(step)]


REQ = EngineRequest(sources=("img1",))


def manager(specs_and_adapters, **kw):
    reg = EngineRegistry()
    adapters = {}
    for sp, ad in specs_and_adapters:
        reg.register(sp)
        if ad is not None:
            adapters[sp.engine_id] = ad
    return ExecutionManager(reg, adapters, **kw), reg


# ------------------------------------------------------------------------------------------------ registry

def test_available_requires_evidence_and_blocked_requires_a_reason():
    with pytest.raises(ValueError):
        Availability(AvailabilityStatus.AVAILABLE, "an import worked")
    with pytest.raises(ValueError):
        Availability(AvailabilityStatus.BLOCKED, "")


def test_a_vram_requirement_must_cite_its_measurement():
    with pytest.raises(ValueError):
        spec("e", hw=HardwareRequirement(min_vram_mb=900.0))
    assert spec("e", hw=HardwareRequirement(min_vram_mb=900.0, measured_source="benchmarks/x.json")).hardware.min_vram_mb == 900.0


def test_duplicate_and_unknown_engines_are_refused():
    reg = EngineRegistry()
    reg.register(spec("a"))
    with pytest.raises(RegistryError):
        reg.register(spec("a"))
    with pytest.raises(RegistryError):
        reg.get("nope")


def test_a_probe_that_raises_is_blocked_with_the_real_error_never_available():
    def boom():
        raise RuntimeError("driver exploded")

    reg = EngineRegistry()
    reg.register(spec("a", probe=boom))
    a = reg.availability("a")
    assert not a.ok and "RuntimeError" in a.reason and "driver exploded" in a.reason
    assert reg.available(Task.DEPTH) == []


def test_candidates_report_unavailable_engines_with_their_reason():
    reg = EngineRegistry()
    reg.register(spec("up"))
    reg.register(spec("down", probe=lambda: blocked("checkpoint missing")))
    assert [s.engine_id for s in reg.available(Task.DEPTH)] == ["up"]
    assert [(s.engine_id, a.reason) for s, a in reg.unavailable(Task.DEPTH)] == [("down", "checkpoint missing")]


def test_engines_sharing_a_family_are_grouped_so_consensus_can_treat_them_as_correlated():
    reg = EngineRegistry()
    for eid, fam in (("a", "dino"), ("b", "dino"), ("c", "")):
        reg.register(spec(eid, family=fam))
    assert reg.families(["a", "b", "c"]) == {"dino": ["a", "b"], "c": ["c"]}


# ------------------------------------------------------------------------------------------------ observations

def test_an_observation_needs_a_coordinate_system_a_source_and_a_sane_confidence():
    kw = dict(engine_id="e", model_version="1", execution_config={})
    with pytest.raises(ObservationError):
        make_observation(draft(cs="unknown"), **kw)
    with pytest.raises(ObservationError):
        make_observation(ObservationDraft("depth", "q", "", "cam", {"v": 1}), **kw)
    with pytest.raises(ObservationError):
        make_observation(ObservationDraft("depth", "q", "s", "cam", {"v": 1}, confidence=1.5), **kw)
    with pytest.raises(ObservationError):
        make_observation(ObservationDraft("depth", "q", "s", "cam", {"v": float("nan")}), **kw)
    with pytest.raises(ObservationError):
        make_observation(ObservationDraft("depth", "q", "s", "cam", {"v": 1}, image_region=(5, 5, 2, 9)), **kw)


def test_observation_ids_are_content_hashes_and_engines_never_share_one():
    kw = dict(model_version="1", execution_config={}, processed_at="t")
    a = make_observation(draft(1.0), engine_id="a", **kw)
    a2 = make_observation(draft(1.0), engine_id="a", **kw)
    b = make_observation(draft(1.0), engine_id="b", **kw)
    c = make_observation(draft(1.0001), engine_id="a", **kw)
    assert a.observation_id == a2.observation_id
    assert len({a.observation_id, b.observation_id, c.observation_id}) == 3
    assert not a.calibrated and a.confidence is None      # nothing is invented


# ------------------------------------------------------------------------------------------------ adapter contract

def test_the_runner_stamps_engine_identity_so_an_adapter_cannot_misreport_it():
    ad = FakeAdapter(spec("midas", version="v7"))
    run = ad.run(REQ, Device("cpu"))
    assert run.ok and run.observations[0].engine_id == "midas" and run.observations[0].model_version == "v7"
    assert run.observations[0].execution_config["device"] == "cpu"
    assert ad.loaded == ad.unloaded == 1


def test_a_missing_checkpoint_is_unavailable_not_a_crash():
    ad = FakeAdapter(spec("m", probe=lambda: blocked("weights missing: checkpoint not cached")))
    run = ad.run(REQ, Device("cpu"))
    assert run.status is EngineRunStatus.UNAVAILABLE and run.error_kind == EngineErrorKind.MISSING_CHECKPOINT.value
    assert ad.loaded == 0 and run.observations == ()


def test_invalid_input_model_crash_and_oom_are_failures_with_the_real_message_and_the_model_is_unloaded():
    bad = FakeAdapter(spec("a"), validate_fail=True)
    r = bad.run(REQ, Device("cpu"))
    assert r.status is EngineRunStatus.FAILED and r.error_kind == "invalid_input" and "two sources" in r.error and bad.loaded == 0
    crash = FakeAdapter(spec("b"), script=[ValueError("tensor shape mismatch")])
    r = crash.run(REQ, Device("cpu"))
    assert r.status is EngineRunStatus.FAILED and r.error_kind == "model_error" and "tensor shape mismatch" in r.error
    assert crash.unloaded == 1
    oom = FakeAdapter(spec("c"), script=[MemoryError("CUDA out of memory")])
    r = oom.run(REQ, Device("cuda"))
    assert r.error_kind == "out_of_memory" and oom.unloaded == 1


# ------------------------------------------------------------------------------------------------ execution manager: policies

def test_maximum_fidelity_runs_every_approved_engine_and_reports_the_one_that_could_not_run():
    m, _ = manager([(spec("a"), FakeAdapter(spec("a"))), (spec("b"), FakeAdapter(spec("b"))),
                    (spec("c", probe=lambda: blocked("checkpoint missing")), FakeAdapter(spec("c")))])
    rep = m.run(Task.DEPTH, REQ)
    assert rep.status is ReportStatus.PARTIAL
    by = {r.engine_id: r for r in rep.runs}
    assert by["a"].ok and by["b"].ok and by["c"].status is EngineRunStatus.UNAVAILABLE and "checkpoint missing" in by["c"].error
    assert "unknown to the manager" in rep.memory_verification


def test_maximum_fidelity_is_complete_only_when_every_required_engine_succeeded():
    m, _ = manager([(spec("a"), FakeAdapter(spec("a"))), (spec("b"), FakeAdapter(spec("b")))])
    assert m.run(Task.DEPTH, REQ).status is ReportStatus.COMPLETE
    m2, _ = manager([(spec("a"), FakeAdapter(spec("a"))), (spec("b"), FakeAdapter(spec("b"), script=[ValueError("x")]))])
    assert m2.run(Task.DEPTH, REQ).status is ReportStatus.PARTIAL


def test_accurate_never_substitutes_a_weaker_engine_for_an_unavailable_primary():
    weak = FakeAdapter(spec("weak", priority=50))
    m, _ = manager([(spec("primary", priority=10, probe=lambda: blocked()), FakeAdapter(spec("primary"))),
                    (spec("weak", priority=50), weak)], policy=Policy.ACCURATE)
    rep = m.run(Task.DEPTH, REQ)
    assert rep.status is ReportStatus.FAILED and weak.loaded == 0
    assert {e.engine_id: e.state for e in rep.plan.entries} == {"primary": "unavailable", "weak": "not_primary"}


def test_fast_runs_one_engine_and_says_in_words_that_the_result_is_reduced_fidelity():
    m, _ = manager([(spec("a", priority=1), FakeAdapter(spec("a"))), (spec("b", priority=2), FakeAdapter(spec("b")))],
                   policy=Policy.FAST)
    rep = m.run(Task.DEPTH, REQ)
    assert len(rep.observations()) == 1 and "REDUCED FIDELITY" in rep.quality_claim and rep.warnings


def test_engines_with_a_restricted_or_unestablished_licence_are_excluded_and_reported():
    m, _ = manager([(spec("ok"), FakeAdapter(spec("ok"))), (spec("nc", commercial=False), FakeAdapter(spec("nc"))),
                    (spec("unk", commercial=None), FakeAdapter(spec("unk")))])
    rep = m.run(Task.DEPTH, REQ)
    assert {e.engine_id: e.state for e in rep.plan.entries} == {"ok": "planned", "nc": "not_approved", "unk": "not_approved"}
    assert [r.engine_id for r in rep.runs] == ["ok"]


def test_an_engine_with_no_adapter_is_reported_unavailable_not_silently_skipped():
    m, _ = manager([(spec("a"), None)])
    rep = m.run(Task.DEPTH, REQ)
    assert rep.status is ReportStatus.FAILED and rep.runs[0].status is EngineRunStatus.UNAVAILABLE
    assert "no adapter" in rep.runs[0].error


# ------------------------------------------------------------------------------------------------ memory, devices, retries

class Probe:
    def __init__(self, gpu, free):
        self.gpu, self.free = gpu, free

    def gpu_available(self):
        return self.gpu

    def free_vram_mb(self):
        return self.free


HW = HardwareRequirement(min_vram_mb=4000.0, measured_source="benchmarks/hardware/x.json", cpu_supported=True)


def test_insufficient_vram_falls_back_to_cpu_and_records_the_fallback_on_every_observation():
    ad = FakeAdapter(spec("a", hw=HW))
    m, _ = manager([(spec("a", hw=HW), ad)], probe=Probe(True, 1000.0), memory_wait_s=0.0)
    rep = m.run(Task.DEPTH, REQ)
    o = rep.observations()[0]
    assert ad.devices == ["cpu"] and o.fallback["to"] == "cpu" and "VRAM" in o.fallback["reason"]
    assert rep.status is ReportStatus.COMPLETE


def test_no_cpu_path_and_not_enough_vram_is_a_failed_run_never_a_skipped_one():
    hw = HardwareRequirement(min_vram_mb=4000.0, measured_source="x", cpu_supported=False, gpu_required=True)
    ad = FakeAdapter(spec("a", hw=hw))
    m, _ = manager([(spec("a", hw=hw), ad)], probe=Probe(True, 1000.0))
    rep = m.run(Task.DEPTH, REQ)
    assert rep.status is ReportStatus.FAILED and rep.runs[0].error_kind == "out_of_memory" and ad.loaded == 0


def test_a_required_gpu_that_is_absent_or_unverifiable_fails_instead_of_running_on_cpu():
    hw = HardwareRequirement(gpu_required=True, cpu_supported=False)
    for gpu in (False, None):
        m, _ = manager([(spec("a", hw=hw), FakeAdapter(spec("a", hw=hw)))], probe=Probe(gpu, None))
        rep = m.run(Task.DEPTH, REQ)
        assert rep.status is ReportStatus.FAILED and "requires a GPU" in rep.runs[0].error


def test_gpu_memory_exhaustion_at_runtime_is_retried_once_then_falls_back_to_cpu_with_a_record():
    ad = FakeAdapter(spec("a"), script=[MemoryError("oom"), MemoryError("oom")])
    m, _ = manager([(spec("a"), ad)], probe=Probe(True, 8000.0))
    rep = m.run(Task.DEPTH, REQ)
    assert ad.devices == ["cuda:0", "cuda:0", "cpu"]
    assert rep.status is ReportStatus.COMPLETE and rep.observations()[0].fallback["to"] == "cpu"
    assert ad.unloaded == 3                       # unloaded after every attempt


def test_oom_everywhere_ends_as_a_failure():
    ad = FakeAdapter(spec("a"), script=[MemoryError("oom")] * 4)
    m, _ = manager([(spec("a"), ad)], probe=Probe(True, 8000.0))
    rep = m.run(Task.DEPTH, REQ)
    assert rep.status is ReportStatus.FAILED and rep.runs[0].error_kind == "out_of_memory"


def test_transient_errors_are_retried_with_backoff_and_other_errors_are_not():
    waits = []
    flaky = FakeAdapter(spec("a"), script=[EngineError(EngineErrorKind.TRANSIENT, "rate limit"), EngineError(EngineErrorKind.TRANSIENT, "x")])
    m, _ = manager([(spec("a"), flaky)], sleep=waits.append)
    rep = m.run(Task.DEPTH, REQ)
    assert rep.runs[0].ok and rep.runs[0].attempts == 3 and waits == [1.0, 2.0]
    hard = FakeAdapter(spec("b"), script=[ValueError("bug"), 1.0])
    m2, _ = manager([(spec("b"), hard)], sleep=waits.append)
    assert m2.run(Task.DEPTH, REQ).runs[0].attempts == 1


def test_parallel_execution_only_when_memory_fit_is_proven_otherwise_sequential_with_a_warning():
    hw = HardwareRequirement(min_vram_mb=1000.0, measured_source="x", parallel_safe=True)

    def pair():
        return [(spec("a", hw=hw), FakeAdapter(spec("a", hw=hw))), (spec("b", hw=hw), FakeAdapter(spec("b", hw=hw)))]

    m, _ = manager(pair(), probe=Probe(True, 8000.0), max_parallel=2)
    rep = m.run(Task.DEPTH, REQ)
    assert rep.status is ReportStatus.COMPLETE and not any("sequentially" in w for w in rep.warnings)
    m2, _ = manager(pair(), probe=Probe(True, 1500.0), max_parallel=2)     # 2 x 1000 MB does not fit in 1500 MB
    assert any("sequentially" in w for w in m2.run(Task.DEPTH, REQ).warnings)
    m3, _ = manager([(spec("a"), FakeAdapter(spec("a"))), (spec("b"), FakeAdapter(spec("b")))], max_parallel=2)  # unknown memory
    assert any("sequentially" in w for w in m3.run(Task.DEPTH, REQ).warnings)


def test_a_requested_resolution_cap_is_recorded_and_warned_about_never_applied_silently():
    m, _ = manager([(spec("a"), FakeAdapter(spec("a")))])
    rep = m.run(Task.DEPTH, EngineRequest(sources=("img1",), max_resolution=512))
    assert rep.observations()[0].execution_config["max_resolution"] == 512 and any("512" in w for w in rep.warnings)
    full = m.run(Task.DEPTH, REQ)
    assert full.observations()[0].execution_config["max_resolution"] is None


def test_only_successful_runs_reach_the_evidence_store():
    store = EvidenceStore()
    m, _ = manager([(spec("a"), FakeAdapter(spec("a"))), (spec("b"), FakeAdapter(spec("b"), script=[ValueError("x")]))], store=store)
    m.run(Task.DEPTH, REQ)
    assert [o.engine_id for o in store.query()] == ["a"]


# ------------------------------------------------------------------------------------------------ evidence store

def obs(engine="a", value=1.0, frame=None):
    return make_observation(draft(value, frame), engine_id=engine, model_version="1", execution_config={}, processed_at="t")


def test_store_is_append_only_idempotent_and_never_overwrites_another_engines_output():
    s = EvidenceStore()
    a, b = obs("a", 1.0), obs("b", 9.0)
    assert s.add(a) and s.add(b) and not s.add(a)
    assert len(s) == 2 and s.get(a.observation_id).predicted == {"v": 1.0} and s.get(b.observation_id).predicted == {"v": 9.0}
    assert not any(hasattr(s, n) for n in ("update", "delete", "remove", "overwrite", "replace"))


def test_an_observation_whose_id_does_not_match_its_content_is_refused():
    s = EvidenceStore()
    with pytest.raises(EvidenceStoreError):
        s.add(replace(obs("a", 1.0), predicted={"v": 999.0}))


def test_persistence_round_trips_and_detects_tampering_and_corruption(tmp_path):
    p = tmp_path / "ev.jsonl"
    s = EvidenceStore(p)
    a, b = obs("a", 1.0), obs("b", 2.0)
    s.add_many([a, b])
    s.link(b.observation_id, a.observation_id, "conflicts_with")
    again = EvidenceStore(p)
    assert len(again) == 2 and again.relations_of(a.observation_id) == [(b.observation_id, a.observation_id, "conflicts_with")]
    lines = p.read_text(encoding="utf-8").splitlines()
    tampered = json.loads(lines[0])
    tampered["predicted"] = {"v": 42.0}
    p.write_text("\n".join([json.dumps(tampered)] + lines[1:]) + "\n", encoding="utf-8")
    with pytest.raises(EvidenceStoreError, match="content-hash"):
        EvidenceStore(p)
    p.write_text("{not json\n", encoding="utf-8")
    with pytest.raises(EvidenceStoreError, match="corrupt"):
        EvidenceStore(p)


def test_relations_preserve_originals_and_lineage_follows_refinement_chains():
    s = EvidenceStore()
    first, second, third = obs("a", 1.0), obs("a", 1.1), obs("a", 1.2)
    s.add_many([first, second, third])
    s.link(second.observation_id, first.observation_id, "refines")
    s.link(third.observation_id, second.observation_id, "refines")
    assert s.lineage(third.observation_id) == [second.observation_id, first.observation_id]
    assert s.get(first.observation_id).predicted == {"v": 1.0}            # the original is untouched
    with pytest.raises(EvidenceStoreError):
        s.link(first.observation_id, "obs_missing", "refines")
    with pytest.raises(EvidenceStoreError):
        s.link(first.observation_id, second.observation_id, "not_a_kind")


def test_queries_answer_who_said_what_about_the_same_thing():
    s = EvidenceStore()
    s.add_many([obs("a", 1.0, frame=3), obs("b", 2.0, frame=3), obs("a", 5.0, frame=4)])
    who = s.engines_for("depth_value", "img1", frame=3)
    assert sorted(who) == ["a", "b"] and all(len(v) == 1 for v in who.values())
    assert len(s.query(engine_id="a")) == 2 and len(s.query(frame=4)) == 1


# ------------------------------------------------------------------------------------------------ default registry (real environment)

def test_default_registry_availability_matches_the_real_probes_and_never_claims_an_unverified_engine():
    from perception.availability import real_models_by_name

    reg = default_registry()
    real = real_models_by_name()
    assert {s.engine_id for s in reg.all()} == {"colmap", "midas", "depth_anything", "owlvit", "sam", "mask_rcnn"}
    for s in reg.all():
        a = reg.availability(s.engine_id)
        assert a.ok == (real[s.engine_id].status == "available"), s.engine_id
        if a.ok:
            assert a.evidence                         # never available without stating what was verified
        assert s.hardware.min_vram_mb is None         # no uncited VRAM numbers
    assert {s.engine_id: s.integration for s in reg.all()}["sam"] == "registry-only"
