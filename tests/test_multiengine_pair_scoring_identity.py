"""Cross-view pair scores (real epipolar geometry + colour appearance) and stable entity identity. SYNTHETIC scenes with exact known
camera geometry: they prove the scoring/identity logic, not detector or association accuracy on real photographs."""

from __future__ import annotations

import math
import random

import pytest

from engine.math import Quat, Vec3
from engine.multiengine.association import AssociatedEntity, Candidate, PairScore, associate
from engine.multiengine.evidence_matrix import EvidenceContext, EvidenceMatrix
from engine.multiengine.identity import IdentityRegistry
from engine.multiengine.observation import ObservationDraft, make_observation
from engine.multiengine.pair_scoring import ScorerConfig, make_pair_scorer
from perception.instances.appearance import compute_color_histogram
from world_ir.camera import CameraExtrinsics, CameraIntrinsics, PinholeCamera

K = CameraIntrinsics(fx=800, fy=800, cx=320, cy=240, width=640, height=480)
CFG = ScorerConfig(epipolar_tolerance_px=3.0, appearance_min=0.5)


def _cam(x=0.0, y=0.0, z=0.0, yaw_deg=0.0):
    h = math.radians(yaw_deg) / 2
    return PinholeCamera(K, CameraExtrinsics(Vec3(x, y, z), Quat(math.cos(h), 0.0, math.sin(h), 0.0)))


CAMS = {"imgA": _cam(0, 0, 0), "imgB": _cam(1.0, 0, 0), "imgC": _cam(0.0, 0.0, 0.0, yaw_deg=15)}


class Scene:
    def __init__(self):
        self.m = EvidenceMatrix()
        self.cands: list[Candidate] = []

    def detect(self, source, world_point, label="chair"):
        uv = CAMS[source].project(Vec3(*world_point))
        assert uv is not None
        return self.detect_px(source, uv, label)

    def detect_px(self, source, uv, label="chair"):
        u, v = uv
        box = (u - 20, v - 20, u + 20, v + 20)
        nb = [box[0] / 640, box[1] / 480, box[2] / 640, box[3] / 480]
        o = make_observation(
            ObservationDraft(task="detection", quantity="bbox", source_id=source, coordinate_system=f"image:{source}",
                             image_region=box, predicted={"label": label, "box_normalized": nb}, confidence=0.9),
            engine_id="det", model_version="v1", execution_config={},
            processed_at=f"2026-01-01T00:00:{len(self.cands):02d}+00:00",
        )
        self.m.add(o, EvidenceContext(o.observation_id, run_id="r"))
        c = Candidate(o.observation_id, source, label)
        self.cands.append(c)
        return c


def _hist(rgb):
    return compute_color_histogram([[rgb] * 4] * 4, [[True] * 4] * 4)


def test_true_correspondence_passes_with_a_known_small_residual():
    s = Scene()
    a, b = s.detect("imgA", (0.2, 0.1, 4.0)), s.detect("imgB", (0.2, 0.1, 4.0))
    (p,) = make_pair_scorer(CAMS, CFG)(s.cands, s.m)
    assert p.score > 0.9 and p.basis["epipolar_residual_px"] < 0.01 and p.basis["appearance"] is None
    assert p.basis["epipolar_tolerance_px"] == 3.0 and "necessary, not sufficient" in p.basis["limitation"]
    assert {p.a, p.b} == {a.observation_id, b.observation_id}


def test_geometrically_inconsistent_pair_is_refused_with_the_measured_numbers():
    s = Scene()
    s.detect("imgA", (0.2, 0.1, 4.0))
    s.detect("imgB", (0.2, 0.7, 4.0))  # a different height: far off the epipolar line of a horizontal baseline
    (p,) = make_pair_scorer(CAMS, CFG)(s.cands, s.m)
    assert p.score == 0.0 and p.basis["epipolar_residual_px"] > 100 and "exceeds 3.0 px" in p.basis["refused"]
    res = associate(s.cands, [p])  # the association ledger records WHY nothing merged
    assert len(res.entities) == 2 and res.decisions[0].reason == "non_positive_score"
    assert "refused" in res.decisions[0].basis


def test_appearance_gates_when_both_descriptors_exist_and_is_none_otherwise():
    s = Scene()
    a, b = s.detect("imgA", (0.2, 0.1, 4.0)), s.detect("imgB", (0.2, 0.1, 4.0))
    reds = {a.observation_id: _hist((250, 10, 10)), b.observation_id: _hist((250, 10, 10))}
    mixed = {a.observation_id: _hist((250, 10, 10)), b.observation_id: _hist((10, 10, 250))}
    ok = make_pair_scorer(CAMS, CFG, appearance=lambda c, m: reds[c.observation_id])(s.cands, s.m)[0]
    assert ok.score > 0.9 and ok.basis["appearance"] == 1.0
    bad = make_pair_scorer(CAMS, CFG, appearance=lambda c, m: mixed[c.observation_id])(s.cands, s.m)[0]
    assert bad.score == 0.0 and bad.basis["appearance"] == 0.0 and "below minimum" in bad.basis["refused"]
    partial = make_pair_scorer(CAMS, CFG, appearance=lambda c, m: reds[c.observation_id] if c is a else None)(s.cands, s.m)[0]
    assert partial.basis["appearance"] is None and partial.score > 0.9  # geometry only; similarity is never invented


def test_no_camera_or_same_image_means_no_score_is_offered():
    s = Scene()
    s.detect("imgA", (0.2, 0.1, 4.0))
    s.detect("imgA", (0.9, 0.1, 5.0))  # same image
    assert list(make_pair_scorer(CAMS, CFG)(s.cands, s.m)) == []
    uncalibrated = {"imgA": CAMS["imgA"]}  # imgB has no registered camera
    s.detect("imgB", (0.2, 0.1, 4.0))
    assert list(make_pair_scorer(uncalibrated, CFG)(s.cands, s.m)) == []


def test_two_candidates_on_one_epipolar_line_never_both_merge_into_one_entity():
    """Ambiguity the geometry cannot resolve (same epipolar line, different depth): same-image exclusion keeps ONE object per image."""
    s = Scene()
    a = s.detect("imgA", (0.2, 0.1, 4.0))
    u, v = CAMS["imgB"].project(Vec3(0.2, 0.1, 4.0))
    s.detect("imgB", (0.2, 0.1, 4.0))
    s.detect_px("imgB", (u - 120, v))
    scores = list(make_pair_scorer(CAMS, CFG)(s.cands, s.m))
    assert sum(1 for p in scores if p.score > 0) == 2  # both pass the (necessary) epipolar test
    res = associate(s.cands, scores)
    merged = [e for e in res.entities if len(e.members) == 2]
    assert len(merged) == 1 and a.observation_id in merged[0].members
    assert res.refused("same_image_conflict")
    assert all(len(e.sources) == len(e.members) for e in res.entities)  # one member per image per entity


def test_config_has_no_defaults_and_validates():
    with pytest.raises(TypeError):
        ScorerConfig()  # type: ignore[call-arg]
    with pytest.raises(ValueError):
        ScorerConfig(epipolar_tolerance_px=0, appearance_min=0.5)
    with pytest.raises(ValueError):
        ScorerConfig(epipolar_tolerance_px=3, appearance_min=1.5)


def test_no_entity_ever_holds_two_members_from_one_image_on_random_inputs():
    rng = random.Random(11)
    for _ in range(80):
        n = rng.randint(2, 16)
        cands = [Candidate(f"o{i}", f"img{rng.randint(0, 3)}", "chair") for i in range(n)]
        scores = [PairScore(f"o{a}", f"o{b}", rng.uniform(0.01, 1)) for a in range(n) for b in range(a + 1, n) if rng.random() < 0.7]
        for e in associate(cands, scores).entities:
            srcs = [c.source_id for c in cands if c.observation_id in e.members]
            assert len(srcs) == len(set(srcs))


# ------------------------------------------------------------------------------------------------ identity
def _ent(*members):
    return AssociatedEntity("tmp", tuple(sorted(members)), ("chair",), ("s",))


def test_identity_is_stable_when_evidence_is_added(tmp_path):
    reg = IdentityRegistry(tmp_path / "ids.json")
    (e1,), ev = reg.resolve([_ent("o1", "o2")])
    assert ev[0].kind == "created"
    (e2,), ev = reg.resolve([_ent("o1", "o2")])
    assert e2.entity_key == e1.entity_key and ev[0].kind == "unchanged"
    (e3,), ev = reg.resolve([_ent("o1", "o2", "o3")])  # a new photograph's detection joins
    assert e3.entity_key == e1.entity_key and ev[0].kind == "extended" and "1 observation" in ev[0].detail
    assert reg.members(e1.entity_key) == frozenset({"o1", "o2", "o3"})


def test_merge_keeps_the_larger_identity_and_retires_the_other(tmp_path):
    reg = IdentityRegistry(tmp_path / "ids.json")
    out, _ = reg.resolve([_ent("o1", "o2", "o3"), _ent("o4", "o5")])
    by = {frozenset(e.members): e.entity_key for e in out}
    a, b = by[frozenset({"o1", "o2", "o3"})], by[frozenset({"o4", "o5"})]
    (m,), ev = reg.resolve([_ent("o1", "o2", "o3", "o4", "o5")])  # new evidence links the two
    assert m.entity_key == a and ev[0].kind == "merged" and ev[0].related == (b,)
    assert reg.retired == {b: a}  # retired, never reused, never forgotten
    assert b not in reg.ids()
    (again,), _ = reg.resolve([_ent("o1", "o2", "o3", "o4", "o5", "o6")])
    assert again.entity_key == a


def test_split_keeps_the_id_with_the_largest_part_and_records_the_origin(tmp_path):
    reg = IdentityRegistry(tmp_path / "ids.json")
    (a,), _ = reg.resolve([_ent("o1", "o2", "o3", "o4")])
    out, ev = reg.resolve([_ent("o1", "o2", "o3"), _ent("o4")])
    keys = {frozenset(e.members): e.entity_key for e in out}
    assert keys[frozenset({"o1", "o2", "o3"})] == a.entity_key
    new = keys[frozenset({"o4"})]
    assert new != a.entity_key
    split = next(e for e in ev if e.kind == "split")
    assert split.stable_id == new and split.related == (a.entity_key,)


def test_identity_resolution_is_order_independent_and_survives_a_restart(tmp_path):
    batches = [[_ent("o1", "o2"), _ent("o3")], [_ent("o1", "o2", "o4"), _ent("o3", "o5")]]
    runs = []
    for order in (1, -1):
        reg = IdentityRegistry(tmp_path / f"ids{order}.json")
        keys = []
        for batch in batches:
            out, _ = reg.resolve(batch[::order])
            keys.append(sorted((tuple(e.members), e.entity_key) for e in out))
        runs.append(keys)
    assert runs[0] == runs[1]
    reopened = IdentityRegistry(tmp_path / "ids1.json")
    (e,), _ = reopened.resolve([_ent("o1", "o2", "o4", "o6")])
    assert e.entity_key == dict(runs[0][1])[("o1", "o2", "o4")]  # same name after a process restart
    assert any(ev["kind"] == "extended" for ev in reopened.events)


def test_entities_not_seen_in_this_call_are_kept_not_dropped():
    reg = IdentityRegistry()
    (_, b), _ = reg.resolve([_ent("o1"), _ent("o2")])
    reg.resolve([_ent("o1", "o3")])  # this call saw nothing of o2's entity
    assert b.entity_key in reg.ids()
