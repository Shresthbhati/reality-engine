"""Scene identity: grouping, diagnostics, bounded correction, persistence (deterministic, no COLMAP needed).

Geometry fixture: one shared 3D point cloud and pinhole cameras with KNOWN poses. The pair verifier reports the
number of 3D points BOTH cameras see (a genuine co-visibility count), so "overlap" comes from geometry, not from a
hand-written adjacency list. Real-image behaviour is covered by scripts/run_scene_identity_e2e.py.
"""

import math

import numpy as np
import pytest

from evidence.session import EvidenceItem, EvidenceKind
from reconstruction.backend.colmap_backend import ColmapReconstructionBackend
from reconstruction.backend.interface import (
    IReconstructionBackend, ReconstructedCameraPose, ReconstructedPoint, ReconstructionResult,
)
from reconstruction.scene_identity import (
    PairEvidence, ReconOutcome, SceneRegistry, diagnose, group_images, pose_agreement_deg, reconstruct_group,
)

RNG = np.random.default_rng(0)


# ------------------------------------------------------------------ geometry fixture
def _room_points(center, n=400, half=2.0):
    return np.asarray(center) + RNG.uniform(-half, half, (n, 3))


def _visible(points, cam_pos, look, fov_deg=70.0, max_d=9.0):
    f = np.asarray(look, float) - cam_pos
    f /= np.linalg.norm(f)
    d = points - cam_pos
    dist = np.linalg.norm(d, axis=1)
    cosang = (d @ f) / np.maximum(dist, 1e-9)
    return set(np.flatnonzero((cosang > math.cos(math.radians(fov_deg / 2))) & (dist < max_d)))


def make_scene(points, cams):
    """cams: {id: (pos, look)} -> covisibility verifier."""
    seen = {k: _visible(points, np.asarray(p, float), l) for k, (p, l) in cams.items()}

    def verify(a, b):
        shared = len(seen[a] & seen[b])
        return PairEvidence(a, b, shared, shared, 1.0 if shared else 0.0)

    return verify


def ring(prefix, center, n=6, radius=3.0):
    return {f"{prefix}{i}": (np.asarray(center) + [radius * math.cos(2 * math.pi * i / n),
                                                   radius * math.sin(2 * math.pi * i / n), 0.5], center)
            for i in range(n)}


A_PTS = _room_points([0, 0, 0])
B_PTS = _room_points([100, 0, 0])
CAMS_A = ring("A", [0, 0, 0])
CAMS_B = ring("B", [100, 0, 0])


def photos(ids):
    return [EvidenceItem(id=i, kind=EvidenceKind.PHOTO, source_uri=f"file:///fake/{i}.jpg") for i in ids]


def result(ids, n_points=200, err=0.6, spread=True, status="success", comps=None):
    poses = [ReconstructedCameraPose(i, (math.cos(k) * 3 if spread else 0.0, math.sin(k) * 3 if spread else 0.0, 0.5),
                                     (1.0, 0.0, 0.0, 0.0)) for k, i in enumerate(ids)]
    pts = [ReconstructedPoint((math.cos(j) * 0.3, math.sin(j) * 0.3, 0.1 * (j % 7)), f"t{j}", list(ids))
           for j in range(n_points)]
    stats = {"registered_images": len(ids), "mean_reprojection_error_px": err, "components": comps or [{}]}
    return ReconstructionResult(points=pts, camera_poses=poses, registration_status=status, registration_stats=stats)


class Scripted(IReconstructionBackend):
    def __init__(self, outcomes):
        self.outcomes = outcomes

    def reconstruct(self, evidence):
        o = self.outcomes.pop(0)
        if isinstance(o, Exception):
            raise o
        return o


GROUPING_A = group_images(list(CAMS_A), make_scene(A_PTS, CAMS_A))


def factory(*outcomes):
    q = list(outcomes)
    return lambda cfg: Scripted(q)


# ------------------------------------------------------------------ 1. compatible views -> one group
def test_compatible_views_form_one_scene_group():
    g = group_images(list(CAMS_A), make_scene(A_PTS, CAMS_A))
    assert len(g.groups) == 1 and sorted(g.groups[0].members) == sorted(CAMS_A) and g.unlinked == []


# ------------------------------------------------------------------ 2. repeated views: no duplicate world
def test_repeated_views_do_not_create_duplicate_groups(tmp_path):
    cams = dict(CAMS_A)
    cams["A0_dup"] = cams["A0"]                       # the same viewpoint uploaded twice
    reg = SceneRegistry(tmp_path)
    first = reg.record_grouping("w", group_images(list(CAMS_A), make_scene(A_PTS, CAMS_A)))
    second = reg.record_grouping("w", group_images(list(cams), make_scene(A_PTS, cams)))
    assert len(second) == 1 and second[0].group_id == first[0].group_id          # identity unchanged
    assert "A0_dup" in second[0].members and len(reg.load("w")["groups"]) == 1


# ------------------------------------------------------------------ 3. overlapping views share geometry
def test_overlapping_views_share_registered_geometry():
    ids = list(CAMS_A)
    trace = reconstruct_group(photos(ids), factory(result(ids)), grouping=group_images(ids, make_scene(A_PTS, CAMS_A)))
    r = trace.best_result
    assert trace.best.outcome == ReconOutcome.SUCCESS and trace.stop_reason == "success"
    assert len(r.points) == len({p.track_id for p in r.points})                      # no duplicated tracks
    assert all(len(p.source_evidence_ids) >= 2 for p in r.points)                    # each point seen by >=2 images
    assert {p.evidence_id for p in r.camera_poses} == set(ids)                       # one frame, all cameras


# ------------------------------------------------------------------ 4. disconnected groups detected
def test_disconnected_image_groups_are_detected():
    cams = {**CAMS_A, **CAMS_B}
    pts = np.vstack([A_PTS, B_PTS])
    g = group_images(list(cams), make_scene(pts, cams))
    assert len(g.groups) == 2
    assert {frozenset(x.members) for x in g.groups} == {frozenset(CAMS_A), frozenset(CAMS_B)}
    assert diagnose(result(list(CAMS_A)), 6, grouping=g, require_single_scene=True).outcome == ReconOutcome.INCOMPATIBLE


def test_declared_session_split_by_verification_is_a_conflict():
    cams = {**CAMS_A, **CAMS_B}
    g = group_images(list(cams), make_scene(np.vstack([A_PTS, B_PTS]), cams), declared_sessions={"upload-1": list(cams)})
    assert g.declared_conflicts and g.declared_conflicts[0]["session"] == "upload-1"
    assert len(g.groups) == 2                                                         # not forced together


# ------------------------------------------------------------------ 5. similar-but-incompatible scenes
def test_visually_similar_but_unverified_pairs_are_not_merged():
    ids = ["A0", "A1", "B0", "B1"]

    def verify(a, b):
        if a[0] == b[0]:
            return PairEvidence(a, b, 200, 190, 0.95)
        return PairEvidence(a, b, 120, 18, 0.15)            # plenty of look-alike matches, no epipolar consistency

    g = group_images(ids, verify)
    assert len(g.groups) == 2 and g.uncertain_links, "cross-scene look-alikes must be recorded, not merged"
    assert all(set(x.members) in ({"A0", "A1"}, {"B0", "B1"}) for x in g.groups)


# ------------------------------------------------------------------ 6. insufficient overlap -> honest result
def test_insufficient_overlap_is_unlinked_and_asks_for_more_evidence():
    cams = {"A0": CAMS_A["A0"], "A3": CAMS_A["A3"], "B0": CAMS_B["B0"]}
    g = group_images(list(cams), make_scene(np.vstack([A_PTS, B_PTS]), cams))
    assert "B0" in g.unlinked
    ids = ["A0", "A3", "B0"]
    thin = result(["A0", "A3"], n_points=40, err=0.8)
    trace = reconstruct_group(photos(ids), factory(thin), grouping=g)
    assert trace.best.outcome == ReconOutcome.PARTIAL                                 # 2/3, not dressed up as success
    assert trace.stop_reason == "additional_evidence_required" and trace.additional_evidence_needed == ["B0"]


# ------------------------------------------------------------------ 7. invalid camera estimates
@pytest.mark.parametrize("bad", ["nan", "coincident", "non_unit", "high_error", "few_points"])
def test_invalid_camera_estimates_never_succeed(bad):
    ids = list(CAMS_A)
    r = result(ids, spread=(bad != "coincident"), err=5.0 if bad == "high_error" else 0.5,
               n_points=5 if bad == "few_points" else 200)
    if bad == "nan":
        p0 = r.camera_poses[0]
        r = ReconstructionResult(r.points, [ReconstructedCameraPose(p0.evidence_id, (float("nan"), 0, 0), p0.rotation),
                                            *r.camera_poses[1:]], "success", registration_stats=r.registration_stats)
    if bad == "non_unit":
        p0 = r.camera_poses[0]
        r = ReconstructionResult(r.points, [ReconstructedCameraPose(p0.evidence_id, p0.position, (2.0, 0, 0, 0)),
                                            *r.camera_poses[1:]], "success", registration_stats=r.registration_stats)
    d = diagnose(r, len(ids))
    assert d.outcome == ReconOutcome.FAILED and d.failure_reason


def test_unmeasured_quality_is_reported_not_invented():
    base = result(["a", "b", "c"])
    d = diagnose(ReconstructionResult(base.points, base.camera_poses, "success"), 3)
    assert d.mean_reprojection_error_px is None and "mean_reprojection_error_px" in d.unmeasured


# ------------------------------------------------------------------ 8. persistence / restart
def test_registry_survives_restart_with_same_identity(tmp_path):
    g = group_images(list(CAMS_A), make_scene(A_PTS, CAMS_A))
    reg = SceneRegistry(tmp_path)
    gid = reg.record_grouping("w1", g)[0].group_id
    d = diagnose(result(list(CAMS_A)), 6)
    assert reg.commit_reconstruction("w1", gid, d, "v1", list(CAMS_A))[0]
    again = SceneRegistry(tmp_path).load("w1")
    assert [x["group_id"] for x in again["groups"]] == [gid]
    assert again["last_valid"]["version_id"] == "v1" and again["last_valid"]["diagnostics"]["outcome"] == "success"


def test_group_identity_is_stable_when_images_are_added(tmp_path):
    reg = SceneRegistry(tmp_path)
    small = {k: CAMS_A[k] for k in ("A0", "A1", "A2", "A3")}
    gid = reg.record_grouping("w", group_images(list(small), make_scene(A_PTS, small)))[0].group_id
    grown = reg.record_grouping("w", group_images(list(CAMS_A), make_scene(A_PTS, CAMS_A)))
    assert len(grown) == 1 and grown[0].group_id == gid and len(grown[0].members) == 6


# ------------------------------------------------------------------ 9. a failed retry never overwrites the last valid result
def test_failed_or_raising_retries_keep_the_last_valid_reconstruction():
    ids = list(CAMS_A)
    partial = result(ids[:4], n_points=120)                          # 4/6 = partial
    trace = reconstruct_group(photos(ids), factory(partial, RuntimeError("colmap crashed"), result(ids[:1], n_points=3)),
                              max_retries=2)
    assert trace.best_result is partial and trace.best.outcome == ReconOutcome.PARTIAL
    assert [a.accepted for a in trace.attempts] == [True, False, False]
    assert "RuntimeError" in trace.attempts[1].diagnostics.failure_reason


def test_retry_is_adopted_only_when_measured_quality_improves():
    ids = list(CAMS_A)
    trace = reconstruct_group(photos(ids), factory(result(ids[:4]), result(ids)), grouping=GROUPING_A, max_retries=1)
    assert trace.stop_reason == "success" and trace.best.registered_images == 6
    worse = reconstruct_group(photos(ids), factory(result(ids[:4]), result(ids[:3])), max_retries=1)
    assert worse.best.registered_images == 4 and worse.stop_reason == "budget_exhausted"
    regress = reconstruct_group(photos(ids), factory(result(ids[:4], err=0.5), result(ids, err=1.9)), grouping=GROUPING_A, max_retries=1)
    assert regress.best.registered_images == 4 and not regress.attempts[1].accepted


def test_guided_matching_gain_that_warps_cameras_is_rejected():
    ids = list(CAMS_A)
    base = result(ids[:4])
    full = result(ids)
    rot = math.radians(40)
    warped_poses = [ReconstructedCameraPose(p.evidence_id, p.position,
                                            (math.cos(rot / 2), 0, 0, math.sin(rot / 2)) if k % 2 else (1.0, 0, 0, 0))  # half twisted
                    for k, p in enumerate(full.camera_poses)]
    warped = ReconstructionResult(full.points, warped_poses, "success", registration_stats=full.registration_stats)
    assert pose_agreement_deg(base, warped) > 5.0
    trace = reconstruct_group(photos(ids), factory(base, warped), grouping=GROUPING_A, max_retries=1)
    assert trace.best_result is base and "warped" in trace.attempts[1].decision


def test_registry_keeps_last_valid_when_new_attempt_fails_or_regresses(tmp_path):
    reg = SceneRegistry(tmp_path)
    ok = diagnose(result(list(CAMS_A)), 6)
    assert reg.commit_reconstruction("w", "sg-1", ok, "v1", list(CAMS_A))[0]
    failed = diagnose(None, 6, error="boom")
    assert reg.commit_reconstruction("w", "sg-1", failed, "v2", [])[0] is False
    worse = diagnose(result(list(CAMS_A)[:3]), 6)
    assert reg.commit_reconstruction("w", "sg-1", worse, "v3", list(CAMS_A)[:3])[0] is False
    s = reg.load("w")
    assert s["last_valid"]["version_id"] == "v1" and len(s["history"]) == 3


# ------------------------------------------------------------------ backend measures real registration facts
def test_colmap_backend_records_measured_registration_stats(tmp_path, monkeypatch):
    (tmp_path / "images.txt").write_text(
        "# hdr\n1 1 0 0 0 0 0 5 1 a.jpg\n1.0 2.0 1\n2 1 0 0 0 1 0 5 1 b.jpg\n1.0 2.0 1\n")
    (tmp_path / "points3D.txt").write_text(
        "# hdr\n1 0.1 0.2 0.3 9 9 9 0.5 1 0 2 0\n2 0.4 0.5 0.6 9 9 9 1.5 1 1 2 1 1 2\n")
    b = ColmapReconstructionBackend()
    monkeypatch.setattr(b, "_run_step", lambda *a, **k: None)
    res = b._parse_model_dir(tmp_path, {}, n_input=3, env={})
    s = res.registration_stats
    assert (s["n_input"], s["registered_images"], s["points"]) == (3, 2, 2)
    assert s["mean_reprojection_error_px"] == pytest.approx(1.0) and s["mean_track_length"] == pytest.approx(2.5)
    assert res.registration_status == "partial"


def test_gained_camera_without_independent_verified_links_is_not_trusted():
    ids = list(CAMS_A)
    only_link = {"A4": ["A0"], "A5": ["A1"]}                 # gained cameras each linked to just ONE placed camera

    def verify(a, b):
        ok = (a in only_link and b in only_link[a]) or (b in only_link and a in only_link[b]) or (
            a in ids[:4] and b in ids[:4])
        return PairEvidence(a, b, 50 if ok else 0, 50 if ok else 0, 1.0 if ok else 0.0)

    g = group_images(ids, verify)
    trace = reconstruct_group(photos(ids), factory(result(ids[:4]), result(ids)), grouping=g, max_retries=1)
    assert trace.best.registered_images == 4 and "independently verified" in trace.attempts[1].decision
    no_grouping = reconstruct_group(photos(ids), factory(result(ids[:4]), result(ids)), max_retries=1)
    assert no_grouping.best.registered_images == 4          # unverifiable gain is never adopted blind
