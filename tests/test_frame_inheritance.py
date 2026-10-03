"""The world's visible coordinate frame must not wander between versions (reconstruction.frame.canonicalize_like_head).

Found by the canonical demo journey on real photographs: successive versions of ONE world were canonicalized
independently (each re-estimating 'up' from its own cameras and dominant plane), so V1 -> V2 -> V3 differed by
7.6 and 9.9 degrees about the vertical although COLMAP's own frame had been preserved -- the Studio said "frame kept"
while the world's frame had rotated. The rule under test: a candidate that DEMONSTRABLY shares HEAD's raw frame
(its shared cameras land where HEAD's are after HEAD's canonical rotation) reuses that rotation; a candidate in a
different frame (a full re-solve) is canonicalized on its own. Reuse needs evidence, never trust.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from engine.pipeline import candidate_selection as cs
from engine.pipeline import world_delta as wd
from provenance import Uncertainty
from reconstruction.backend.interface import ReconstructedCameraPose, ReconstructedPoint, ReconstructionResult
from reconstruction.frame import canonicalize_frame, canonicalize_like_head

CAMS = {"A": (0.0, 0.5, 0.0), "B": (1.0, 0.5, 0.3), "C": (-1.0, 0.5, 0.8), "D": (0.5, 0.5, -0.9)}


def _result(floor_n: int, wall_n: int, transform=None) -> ReconstructionResult:
    """A floor (y = -1) and a wall (z = 4) in the RAW frame, four cameras above the floor looking nowhere in
    particular (identical rotations carry no gravity information, so the dominant plane decides 'up')."""
    pts = []
    for i in range(floor_n):
        pts.append((-3 + 6 * (i % 9) / 8, -1.0, -3 + 6 * ((i // 9) % 9) / 8 + 0.01 * (i // 81)))
    for j in range(wall_n):
        pts.append((-3 + 6 * (j % 9) / 8, -1 + 4 * ((j // 9) % 7) / 6, 4.0 + 0.01 * (j // 63)))
    cams = dict(CAMS)
    if transform:
        pts = [transform(p) for p in pts]
        cams = {k: transform(v) for k, v in cams.items()}
    return ReconstructionResult(
        points=[ReconstructedPoint(position=tuple(float(c) for c in p), track_id=f"t{i:04d}",
                                   source_evidence_ids=["A", "B"], uncertainty=Uncertainty())
                for i, p in enumerate(pts)],
        camera_poses=[ReconstructedCameraPose(evidence_id=k, position=tuple(float(c) for c in v),
                                              rotation=(1.0, 0.0, 0.0, 0.0), uncertainty=Uncertainty())
                      for k, v in cams.items()],
        registration_status="success")


def _head():
    """HEAD: floor-dominant. Returns (canonical result, its frame record, the cameras it stored)."""
    res, rec = canonicalize_frame(_result(81, 20), seed=1)
    return res, rec, {p.evidence_id: list(p.position) for p in res.camera_poses}


def _positions(res):
    return {p.evidence_id: np.array(p.position) for p in res.camera_poses}


def _align_dict(prev, cand):
    al = wd._align(prev, cand)
    return {"R": al["R"].tolist(), "s": al["s"], "ma": al["ma"].tolist(), "mb": al["mb"].tolist()}


def test_the_premise_independent_canonicalization_moves_the_frame_when_the_dominant_plane_changes():
    _head_res, _rec, head_cams = _head()
    cand = _result(30, 63)                                             # same raw frame; the WALL is now dominant
    own, own_rec = canonicalize_frame(cand, seed=1)
    moved = max(np.linalg.norm(_positions(own)[k] - np.array(v)) for k, v in head_cams.items())
    assert own_rec.source_plane_id and moved > 1.0, "the premise: an independent estimate lands in another frame"


def test_a_candidate_in_heads_raw_frame_inherits_heads_canonical_rotation():
    _head_res, rec, head_cams = _head()
    cand = _result(30, 63)
    out, out_rec = canonicalize_like_head(cand, 1, rec.rotation, head_cams)
    assert out_rec.up_source == "inherited_from_head" and "NOT re-estimated" in out_rec.note
    assert np.allclose(out_rec.rotation, rec.rotation)
    for k, v in head_cams.items():
        assert np.linalg.norm(_positions(out)[k] - np.array(v)) < 1e-9, "the established cameras did not stay put"
    # the measured frame shift between HEAD and this candidate is zero (what the arbiter reads)
    cand_cams = {p.evidence_id: list(p.position) for p in out.camera_poses}
    fs = cs.frame_shift(_align_dict(head_cams, cand_cams), cand_cams)
    assert fs["changed"] is False and fs["rotation_deg"] < 1e-3


def test_a_candidate_in_a_different_frame_is_canonicalized_on_its_own():
    """A full re-solve lands in an arbitrary frame: HEAD's rotation does not apply and must not be forced on it."""
    _h, rec, head_cams = _head()
    th = math.radians(40.0)
    rot = np.array([[math.cos(th), 0, math.sin(th)], [0, 1, 0], [-math.sin(th), 0, math.cos(th)]])
    cand = _result(81, 20, transform=lambda p: 2.0 * rot @ np.asarray(p, float) + np.array([5.0, 1.0, -3.0]))
    out, out_rec = canonicalize_like_head(cand, 1, rec.rotation, head_cams)
    assert out_rec.up_source != "inherited_from_head" and out_rec.source_plane_id
    _alone, alone_rec = canonicalize_frame(cand, seed=1)
    assert np.allclose(out_rec.rotation, alone_rec.rotation)


@pytest.mark.parametrize("why,rotation,cams", [
    ("no frame was recorded for HEAD", None, "head"),
    ("HEAD stored no cameras", "rec", None),
    ("a malformed stored frame is never trusted", [[1, 2], [3, 4]], "head"),
    ("a stored 'rotation' that is not a rotation", [[2, 0, 0], [0, 2, 0], [0, 0, 2]], "head"),
])
def test_without_usable_evidence_the_frame_is_estimated_as_before(why, rotation, cams):
    _h, rec, head_cams = _head()
    cand = _result(30, 63)
    _out, out_rec = canonicalize_like_head(cand, 1, rec.rotation if rotation == "rec" else rotation,
                                           head_cams if cams == "head" else cams)
    assert out_rec.up_source != "inherited_from_head", why


def test_too_few_shared_cameras_cannot_demonstrate_a_shared_frame():
    _h, rec, head_cams = _head()
    only_two = {k: v for k, v in list(head_cams.items())[:2]}
    _out, out_rec = canonicalize_like_head(_result(30, 63), 1, rec.rotation, only_two)
    assert out_rec.up_source != "inherited_from_head"


def test_shared_cameras_that_moved_too_far_are_not_a_shared_frame():
    _h, rec, head_cams = _head()
    shifted = {k: [v[0] + 5.0, v[1], v[2]] for k, v in head_cams.items()}     # HEAD's cameras were elsewhere
    _out, out_rec = canonicalize_like_head(_result(30, 63), 1, rec.rotation, shifted)
    assert out_rec.up_source != "inherited_from_head"
