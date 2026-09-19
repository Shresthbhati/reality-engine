"""Tests for landmark registration (registration/landmarks.py) -- the
P4-01 open item: "landmark/correspondence method (needs cross-source
co-observation resolver from P7-01 multi-view identity ... remaining
work is wiring into RegistrationEngine)".

A LANDMARK is a named physical feature observed by >=2 sources: the
identity comes from the observation layer (track_id / provenance), NOT
from geometric proximity. This is the strongest registration evidence
short of GNSS: correspondences are declared, not discovered, so no
ICP-style local-minimum failure mode exists. Like everything in this
repo, the method must stay honest:

  - Ambiguity is represented, never silently resolved: a landmark
    observed N>=2 times by a source with NO track identity produces a
    named ambiguity record and is EXCLUDED from the solve.
  - Degenerate geometry refuses: <3 non-collinear/non-coplanar
    correspondences cannot determine a full 6-DOF transform (2 points
    -> rotation underdetermined about the connecting axis; all points
    collinear -> rotation about the line undetermined; coplanar ->
    roll about the plane normal undetermined). The refusal names the
    degeneracy and what would fix it.
  - Uncertainty is residual-derived: the returned RegistrationResult
    carries measured rmse, inlier fraction, and a residual-derived
    RegistrationCovariance, exactly like the other methods.
  - A declared correspondence that grossly disagrees with the consensus
    (outlier) is REJECTED and reported, not blended in -- a wrong
    landmark identity would otherwise corrupt the whole transform.

Fixtures are deterministic unit scenes; real cross-source landmark
runs are recorded as real in the ledgers.
"""

from __future__ import annotations

import math

import pytest

from engine.physics.math3 import Quat, Vec3
from registration.landmarks import (
    LandmarkAmbiguity,
    LandmarkCorrespondence,
    register_landmarks,
)
from registration.registration import RegistrationError


def _tri_non_degenerate():
    """Three correspondences spanning a non-degenerate basis."""
    return [
        LandmarkCorrespondence(landmark_id="L1",
                               source_position=(0.0, 0.0, 0.0),
                               target_position=(0.0, 0.0, 0.0)),
        LandmarkCorrespondence(landmark_id="L2",
                               source_position=(4.0, 0.0, 0.0),
                               target_position=(0.0, 4.0, 0.0)),
        LandmarkCorrespondence(landmark_id="L3",
                               source_position=(0.0, 0.0, 3.0),
                               target_position=(0.0, 0.0, 3.0)),
    ]


def test_identity_recovered_for_known_rotation_translation():
    """Round-trip: apply a known rigid transform to source points, then
    recover it from the correspondences."""
    half = math.radians(40.0) / 2.0
    quat = Quat(w=math.cos(half), x=0.0, y=0.0, z=math.sin(half)).normalized()
    t = Vec3(2.0, -1.0, 0.5)
    base = [(0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (0.0, 3.0, 0.0), (1.0, 1.0, 2.0)]
    corr = []
    for i, p in enumerate(base):
        q = quat.rotate(Vec3(*p))
        corr.append(LandmarkCorrespondence(
            landmark_id=f"L{i}",
            source_position=(q.x + t.x, q.y + t.y, q.z + t.z),
            target_position=p,
        ))
    result = register_landmarks(corr, from_frame="drone", to_frame="phone")
    assert result.status == "accepted"
    assert result.method == "landmark"
    tr = result.transform
    assert tr is not None
    for c, p in zip(corr, base):
        out = tr.rotation.rotate(Vec3(*c.source_position)) + tr.translation
        assert out.x == pytest.approx(p[0], abs=1e-9)
        assert out.y == pytest.approx(p[1], abs=1e-9)
        assert out.z == pytest.approx(p[2], abs=1e-9)
    assert result.rmse == pytest.approx(0.0, abs=1e-9)
    assert result.covariance is not None
    assert result.covariance.basis.startswith("residual-derived")


def test_two_correspondences_refuse_not_translate_only():
    """2 points fix translation but leave rotation about the connecting
    axis undetermined -- must refuse with that named, not silently
    return a translation-only transform (the GNSS method's model, not
    this one's)."""
    corr = [
        LandmarkCorrespondence("L1", (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        LandmarkCorrespondence("L2", (2.0, 0.0, 0.0), (3.0, 1.0, 1.0)),
    ]
    with pytest.raises(RegistrationError) as exc:
        register_landmarks(corr, from_frame="a", to_frame="b")
    assert "rotation" in str(exc.value).lower() or "degenerate" in str(exc.value).lower()


def test_collinear_correspondences_refuse():
    corr = [
        LandmarkCorrespondence("L1", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        LandmarkCorrespondence("L2", (1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        LandmarkCorrespondence("L3", (2.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
    ]
    with pytest.raises(RegistrationError):
        register_landmarks(corr, from_frame="a", to_frame="b")


def test_coplanar_correspondences_solve_with_conditioning_reported():
    """Coplanar-but-non-collinear DECLARED correspondences still pin all
    6 DOF through point identity (a scalene planar triangle admits no
    other rigid placement) -- so the solve proceeds, but the weaker
    conditioning is REPORTED via the covariance's degenerate_axes (the
    covariance scatter is rank-2 in-plane), never hidden."""
    half = math.radians(25.0) / 2.0
    quat = Quat(w=math.cos(half), x=0.0, y=0.0, z=math.sin(half)).normalized()
    base = [(0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (0.0, 3.0, 0.0)]  # all z=0
    corr = []
    for i, p in enumerate(base):
        q = quat.rotate(Vec3(*p))
        corr.append(LandmarkCorrespondence(
            landmark_id=f"L{i}",
            source_position=(q.x + 1.0, q.y - 2.0, q.z + 0.5),
            target_position=p,
        ))
    result = register_landmarks(corr, from_frame="a", to_frame="b")
    assert result.status == "accepted"
    assert result.covariance is not None
    assert result.covariance.degenerate_axes is not None
    assert "z" in result.covariance.degenerate_axes  # off-plane axis is the weak one


def test_outlier_correspondence_is_rejected_and_reported():
    """One corrupted identity (say, a duplicate track) must not corrupt
    the transform: it is rejected, named, and the rest solves exactly."""
    half = math.radians(90.0) / 2.0
    quat = Quat(w=math.cos(half), x=0.0, y=0.0, z=math.sin(half)).normalized()
    base = [(0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (0.0, 3.0, 0.0), (1.0, 1.0, 2.0)]
    corr = []
    for i, p in enumerate(base):
        q = quat.rotate(Vec3(*p))
        corr.append(LandmarkCorrespondence(
            landmark_id=f"L{i}",
            source_position=(q.x, q.y, q.z),
            target_position=p,
        ))
    # Corrupt L2's SOURCE position by +10 in x: a real identity error.
    s = corr[2].source_position
    corr[2] = LandmarkCorrespondence("L2", (s[0] + 10.0, s[1], s[2]),
                                     corr[2].target_position)
    result = register_landmarks(corr, from_frame="a", to_frame="b",
                                outlier_threshold_m=1.0)
    assert result.status == "accepted"
    tr = result.transform
    # The transform must still be the TRUE one (fit on inliers).
    for i in (0, 1, 3):
        c = corr[i]
        out = tr.rotation.rotate(Vec3(*c.source_position)) + tr.translation
        assert out.x == pytest.approx(base[i][0], abs=1e-6)
    # The outlier is reported, not blended.
    assert result.reason
    assert "L2" in result.reason


def test_ambiguous_landmark_refuses_when_it_leaves_too_few_and_solves_when_enough():
    """Ambiguous landmark ids are excluded from the solve and reported
    in the error/result -- never guessed into the consensus. With only
    2 unambiguous pairs left the solve REFUSES naming the exclusion;
    with >=3 it solves on the unambiguous subset."""
    corr = _tri_non_degenerate()
    corr.append(LandmarkCorrespondence("L1", (5.0, 5.0, 5.0), (0.0, 0.0, 0.0)))
    with pytest.raises(RegistrationError) as exc:
        register_landmarks(corr, from_frame="a", to_frame="b")
    assert "L1" in str(exc.value)

    # Four unique landmarks, one of them duplicated -> solve proceeds
    # on the unambiguous three... no: excluding the ambiguous one leaves
    # 3 usable pairs, which is exactly solvable.
    corr2 = _tri_non_degenerate() + [
        LandmarkCorrespondence("L4", (2.0, 2.0, 2.0), (2.0, 2.0, 2.0)),
        LandmarkCorrespondence("L4", (6.0, 6.0, 6.0), (2.0, 2.0, 2.0)),
    ]
    result = register_landmarks(corr2, from_frame="a", to_frame="b")
    assert result.status == "accepted"
    assert "L4" in result.reason  # the exclusion is recorded


def test_landmark_ambiguity_record_shape():
    amb = LandmarkAmbiguity(landmark_id="L9", n_source_observations=3,
                            reason="no track identity; cannot disambiguate")
    assert amb.landmark_id == "L9"
    d = amb.to_dict()
    assert d["n_source_observations"] == 3


def test_to_dict_roundtrip():
    result = register_landmarks(_tri_non_degenerate(),
                                from_frame="s", to_frame="t")
    d = result.to_dict()
    assert d["method"] == "landmark"
    assert d["status"] == "accepted"
    assert d["transform"]["from_frame"] == "s"


def test_registration_engine_prefers_landmarks_over_icp():
    """Engine wiring: with landmarks + overlapping-but-rotated clouds,
    the landmark method wins (confidence order), and its transform maps
    source onto target."""
    from registration.registration import RegistrationEngine
    half = math.radians(40.0) / 2.0
    quat = Quat(w=math.cos(half), x=0.0, y=0.0, z=math.sin(half)).normalized()
    base = [(0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (0.0, 3.0, 0.0), (1.0, 1.0, 2.0)]
    src, corr = [], []
    for i, p in enumerate(base):
        q = quat.rotate(Vec3(*p))
        src.append(Vec3(q.x + 2.0, q.y - 1.0, q.z + 0.5))
        corr.append(LandmarkCorrespondence(
            landmark_id=f"L{i}",
            source_position=(q.x + 2.0, q.y - 1.0, q.z + 0.5),
            target_position=p,
        ))
    tgt = [Vec3(*p) for p in base]
    engine = RegistrationEngine()
    result = engine.register(
        source_cloud=src, target_cloud=tgt,
        from_frame="drone", to_frame="phone",
        landmarks=corr,
    )
    assert result.method == "landmark"
    assert result.status == "accepted"
    assert result.transform is not None
    out = result.transform.rotation.rotate(src[3]) + result.transform.translation
    assert out.x == pytest.approx(base[3][0], abs=1e-6)


def test_registration_engine_falls_back_to_icp_on_bad_landmarks():
    """Degenerate landmarks (2 pairs) must NOT kill registration: the
    attempt is recorded as blocked and ICP still runs."""
    from registration.registration import RegistrationEngine
    corr = [
        LandmarkCorrespondence("L1", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        LandmarkCorrespondence("L2", (1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
    ]
    tgt = [Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 1, 0), Vec3(1, 1, 0)]
    src = [Vec3(p.x, p.y, p.z) for p in tgt]  # identity-aligned already
    engine = RegistrationEngine()
    result = engine.register(
        source_cloud=src, target_cloud=tgt,
        from_frame="a", to_frame="b",
        landmarks=corr,
    )
    methods = [a.method for a in result.attempts]
    assert "landmark" in methods
    assert any(a.method.startswith("icp") for a in result.attempts)
    assert result.status == "accepted"
