"""A per-photograph depth-scale error moves a detection along that camera's viewing ray. absorb_scale_drift_duplicates links such a
single-photograph orphan to the confirmed object it is the same object as -- by where the confirmed object projects in the orphan's
photograph, not by 3D distance -- and never absorbs a genuine second instance."""
from __future__ import annotations

from engine.math import Quat, Vec3
from perception.instances.lifting import ObjectHypothesis3D
from perception.instances.object_resolution import MergedObjectCandidate, absorb_scale_drift_duplicates
from reconstruction.calibration.camera import CameraExtrinsics, CameraIntrinsics, PinholeCamera


def _camera() -> PinholeCamera:       # at the origin, looking down +Z
    return PinholeCamera(intrinsics=CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480),
                         extrinsics=CameraExtrinsics(position=Vec3.zero(), rotation=Quat.identity()))


def _hyp(view: str, label: str, cx: float, cz: float, rid: str) -> ObjectHypothesis3D:
    pts = tuple(Vec3(cx + dx, dy, cz) for dx in (-0.5, 0.0, 0.5) for dy in (-0.5, 0.0, 0.5))
    return ObjectHypothesis3D(region_id=rid, evidence_id=view, label=label, position=Vec3(cx, 0.0, cz),
                              bounds_min=Vec3(cx - 0.5, -0.5, cz), bounds_max=Vec3(cx + 0.5, 0.5, cz), point_count=len(pts),
                              mask_pixel_count=len(pts), confidence=0.8, points=pts)


def _cand(cid: str, label: str, x: float, z: float, views) -> MergedObjectCandidate:
    hyps = tuple(_hyp(v, label, x, z, f"{cid}-{v}") for v in views)
    return MergedObjectCandidate(candidate_id=cid, label=label, position=Vec3(x, 0.0, z), bounds_min=Vec3(x - 0.5, -0.5, z),
                                 bounds_max=Vec3(x + 0.5, 0.5, z), source_hypotheses=hyps, confidence=0.8)


CAMS = {"C": _camera()}


def test_an_orphan_displaced_along_its_own_viewing_ray_is_absorbed_without_moving_the_object():
    sofa = _cand("sofa", "sofa", 0.0, 5.0, ["A", "B"])
    drifted = _cand("sofa-c", "sofa", 0.0, 9.0, ["C"])            # same direction from camera C, 4 m too far
    out, n = absorb_scale_drift_duplicates([sofa, drifted], CAMS)
    assert n == 1 and [c.candidate_id for c in out] == ["sofa"]
    assert set(out[0].evidence_ids) == {"A", "B", "C"}
    assert (out[0].position.x, out[0].position.z) == (0.0, 5.0)    # the drifted points did not drag the object
    assert out[0].bounds_min == sofa.bounds_min and out[0].bounds_max == sofa.bounds_max


def test_an_orphan_looking_somewhere_else_is_left_alone():
    sofa = _cand("sofa", "sofa", 0.0, 5.0, ["A", "B"])
    elsewhere = _cand("sofa-c", "sofa", 3.0, 5.0, ["C"])          # projects ~300 px away from where the sofa projects
    out, n = absorb_scale_drift_duplicates([sofa, elsewhere], CAMS)
    assert n == 0 and len(out) == 2


def test_a_second_instance_in_a_photograph_that_already_sees_the_object_is_not_absorbed():
    sofa = _cand("sofa", "sofa", 0.0, 5.0, ["A", "B", "C"])       # the confirmed sofa already has its own detection in C
    second = _cand("sofa-c2", "sofa", 0.0, 9.0, ["C"])
    out, n = absorb_scale_drift_duplicates([sofa, second], CAMS)
    assert n == 0 and len(out) == 2


def test_labels_must_match():
    tv = _cand("tv", "television", 0.0, 5.0, ["A", "B"])
    drifted_sofa = _cand("sofa-c", "sofa", 0.0, 9.0, ["C"])
    out, n = absorb_scale_drift_duplicates([tv, drifted_sofa], CAMS)
    assert n == 0 and len(out) == 2


def test_no_camera_for_the_orphans_photograph_means_no_guess():
    sofa = _cand("sofa", "sofa", 0.0, 5.0, ["A", "B"])
    orphan = _cand("sofa-d", "sofa", 0.0, 9.0, ["D"])
    out, n = absorb_scale_drift_duplicates([sofa, orphan], CAMS)
    assert n == 0 and len(out) == 2


def test_nothing_to_do_returns_the_input_unchanged():
    solo = _cand("a", "sofa", 0.0, 5.0, ["A", "B"])
    out, n = absorb_scale_drift_duplicates([solo], CAMS)
    assert n == 0 and out == [solo]
