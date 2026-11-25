"""D3 (VRAM released after the scene gate) and D4 (indoor/outdoor gate) of docs/reconstruction/REAL_RUN_STAGE_REPORT.md."""
from __future__ import annotations

import numpy as np
import pytest

from perception import scene_type as st


def _fr(outdoor, indoor):
    return {"outdoor": outdoor, "indoor": indoor, "top_outdoor": {}, "top_indoor": {}}


# ---------------------------------------------------------------- D4: decision rule
def test_indoor_scene_passes():
    res = st.classify_from_fractions([_fr(0.02, 0.6)] * 5)
    gate = st.decide_scene_gate(res)
    assert res.scene_type == "indoor" and gate.allow_indoor and gate.state == "passed"


def test_exterior_blocked_with_recorded_reason_and_evidence():
    res = st.classify_from_fractions([_fr(0.7, 0.0)] * 5)
    gate = st.decide_scene_gate(res)
    assert res.scene_type == "outdoor" and not gate.allow_indoor and gate.state == "skipped"
    assert "outdoor" in gate.reason and "skipped" in gate.reason
    assert gate.to_dict()["scene"]["views_measured"] == 5 and gate.to_dict()["scene"]["score"] == -1.0


def test_mixed_evidence_is_unknown_and_blocked():
    res = st.classify_from_fractions([_fr(0.3, 0.3)] * 4)          # score 0 -> neither
    gate = st.decide_scene_gate(res)
    assert res.scene_type == "unknown" and not gate.allow_indoor and "mixed" in gate.reason


def test_unmeasured_fails_open_but_is_recorded_degraded():
    res = st.classify_from_fractions([_fr(0.01, 0.02)] * 3)         # too little labelled evidence per view
    gate = st.decide_scene_gate(res)
    assert res.scene_type == "unmeasured" and gate.allow_indoor and gate.state == "degraded"
    assert "could not be measured" in gate.reason


def test_threshold_edge_is_inclusive_on_both_sides():
    t = st.CONFIDENCE_THRESHOLD                                       # score = (i - o)/(i + o) = 0.3  <=>  i = 0.65, o = 0.35
    at = st.classify_from_fractions([_fr(0.35, 0.65)], threshold=t)
    just_below = st.classify_from_fractions([_fr(0.351, 0.649)], threshold=t)
    assert at.scene_type == "indoor" and just_below.scene_type == "unknown"
    out_at = st.classify_from_fractions([_fr(0.65, 0.35)], threshold=t)
    assert out_at.scene_type == "outdoor"


def test_image_fractions_counts_only_listed_classes_and_ignores_wall():
    id2label = {0: "wall", 1: "sky", 2: "ceiling, roof", 3: "building, edifice"}
    lm = np.array([[0, 0, 1, 1], [2, 3, 3, 0]])
    f = st.image_fractions(lm, id2label)
    assert f["outdoor"] == pytest.approx(4 / 8) and f["indoor"] == pytest.approx(1 / 8)


def test_classify_scene_type_with_fake_segmenter_and_unavailable_segmenter(tmp_path):
    id2label = {0: "sky", 1: "floor"}
    paths = [tmp_path / f"{i}.png" for i in range(12)]
    indoor = st.classify_scene_type(paths, models=lambda p: (np.ones((4, 4), np.int32), id2label))
    outdoor = st.classify_scene_type(paths, models=lambda p: (np.zeros((4, 4), np.int32), id2label))
    assert indoor.scene_type == "indoor" and outdoor.scene_type == "outdoor"
    assert indoor.views_total == st.MAX_VIEWS                         # bounded cost on big captures

    def boom(p):
        raise OSError("unreadable")
    broken = st.classify_scene_type(paths[:2], models=boom)
    assert broken.scene_type == "unmeasured" and broken.evidence["error"]


# ---------------------------------------------------------------- D4: wiring (layout + perception)
def test_compile_without_structure_yields_no_walls_ceiling_or_rooms():
    from engine.compiler import CompileOptions, compile_reconstruction_to_world
    from reconstruction.backend.interface import ReconstructedCameraPose
    from tests.test_room_inference import _CAMS, _two_room_scene
    from world_ir import EntityType

    def run(**kw):
        result = _two_room_scene()
        result.camera_poses.extend(ReconstructedCameraPose(evidence_id=f"ev-{i}", position=p, rotation=(1.0, 0.0, 0.0, 0.0))
                                   for i, p in enumerate(_CAMS))
        return compile_reconstruction_to_world(result, CompileOptions(seed=42, **kw))

    on_world, _ = run()
    off_world, off_diag = run(structure_enabled=False)
    kinds = {EntityType.WALL, EntityType.CEILING, EntityType.FLOOR, EntityType.ROOM, EntityType.WINDOW}
    assert any(e.type in kinds for e in on_world.entities.values())             # the indoor path is unchanged
    assert not any(e.type in kinds for e in off_world.entities.values())
    assert off_diag.planes_total == 0 and off_diag.rooms_detected == 0


def test_perception_stage_skips_with_gate_reason_when_scene_blocked():
    from engine.pipeline import vertical_slice as vs
    from types import SimpleNamespace

    reason = "scene is outdoor (confidence 1.00): ..."
    world = SimpleNamespace(metadata={"scene_gate": {"allow_indoor": False, "reason": reason, "state": "skipped"}})
    facts = vs._perception_stage(None, world, [], [], vs.VerticalSliceOptions(perception_model="grounded"))
    assert facts["status"] == "skipped" and facts["note"] == reason and facts["scene_gate"]["state"] == "skipped"


def test_layout_stage_records_blocked_gate_as_degraded_with_reason(monkeypatch):
    from engine.pipeline import vertical_slice as vs
    from perception.scene_type import SceneTypeResult, decide_scene_gate

    gate = decide_scene_gate(SceneTypeResult("outdoor", 1.0, -1.0, 8, 8, {}))
    monkeypatch.setattr(vs, "_scene_gate_decision", lambda items, options: gate)
    from reconstruction.backend.interface import ReconstructedCameraPose
    from tests.test_room_inference import _CAMS, _two_room_scene

    result = _two_room_scene()
    result.camera_poses.extend(ReconstructedCameraPose(evidence_id=f"ev-{i}", position=p, rotation=(1.0, 0.0, 0.0, 0.0))
                               for i, p in enumerate(_CAMS))
    out = vs.run_layout(result, [], vs.VerticalSliceOptions(), backend=object())
    assert out.layout_facts["scene_gate"]["state"] == "skipped" and out.layout_facts["status"] == "skipped"
    assert out.world.metadata["scene_gate"]["allow_indoor"] is False
    assert out.diagnostics.planes_total == 0


# ---------------------------------------------------------------- D3: VRAM released after the scene gate
class _Ev:
    def __init__(self, i):
        self.item = type("I", (), {"id": i})()
        self.facts = type("F", (), {"geometry_eligible": True})()
        self.path = f"{i}.png"


@pytest.fixture
def gate_env(monkeypatch, tmp_path):
    import engine.core.device as dev
    from apps.api import worldstore_service
    calls = {"released": 0}
    monkeypatch.setattr(dev, "release_vram", lambda: calls.__setitem__("released", calls["released"] + 1))
    monkeypatch.setattr(worldstore_service, "worldstore_root", lambda: tmp_path)
    return calls


def test_scene_gate_releases_vram_on_success(monkeypatch, gate_env):
    import reconstruction.scene_identity as si
    from apps.api import jobs

    monkeypatch.setattr(si, "make_pair_verifier", lambda paths, matcher_name="auto": object())
    monkeypatch.setattr(si, "group_images", lambda ids, verifier, declared_sessions=None: si.SceneGrouping(
        groups=[], unlinked=list(ids), uncertain_links=[], declared_conflicts=[], pairs_checked=0, verified_pairs=0,
        thresholds=si.GroupingThresholds()))
    out = jobs._scene_gate("w1", [_Ev("a"), _Ev("b")], {})
    assert out["state"] == "ok" and gate_env["released"] == 1


def test_scene_gate_releases_vram_when_the_verifier_fails(monkeypatch, gate_env):
    import reconstruction.scene_identity as si
    from apps.api import jobs

    def boom(paths, matcher_name="auto"):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(si, "make_pair_verifier", boom)
    out = jobs._scene_gate("w1", [_Ev("a")], {})
    assert out["state"] == "unavailable" and gate_env["released"] == 1


def test_scene_gate_releases_vram_when_grouping_raises_after_the_verifier_exists(monkeypatch, gate_env):
    import reconstruction.scene_identity as si
    from apps.api import jobs

    monkeypatch.setattr(si, "make_pair_verifier", lambda paths, matcher_name="auto": object())

    def boom(ids, verifier, declared_sessions=None):
        raise RuntimeError("matcher died")

    monkeypatch.setattr(si, "group_images", boom)
    out = jobs._scene_gate("w1", [_Ev("a")], {})
    assert out["state"] == "unavailable" and gate_env["released"] == 1
