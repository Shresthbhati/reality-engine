"""Tests for cross-session registration (registration/cross_session.py).

Sessions from different capture runs are reconstructed independently --
their geometry lands in different arbitrary frames. Fusing them requires
a REAL rigid transform recovered from shared structure, with every
attempt recorded and every rejection explained. Nothing may silently
assume identity alignment; a refused session stays refused.
"""

from __future__ import annotations

import math

import pytest

from engine.physics.math3 import Quat, Vec3
from registration.cross_session import (
    CrossSessionReport,
    align_session,
    align_session_chain,
)


def _cloud_a():
    """Session A's structure: a 5x5 grid on z=0 -- a real 3D (planar) patch."""
    return [Vec3(x, y, 0.0) for x in range(5) for y in range(5)]


def _transform_b():
    """Session B's frame: A rotated 30 deg about z and translated."""
    half = math.radians(30.0) / 2.0
    quat = Quat(w=math.cos(half), x=0.0, y=0.0, z=math.sin(half)).normalized()
    return quat, Vec3(3.0, -1.5, 0.2)


def _cloud_b(quat, translation, noise=0.0):
    rng = math.sqrt(noise)
    out = []
    for i, p in enumerate(_cloud_a()):
        q = quat.rotate(p)
        j = (i * 37 % 11) * 1e-4 * (rng / 0.01 if noise else 0.0)
        out.append(Vec3(q.x + translation.x + j, q.y + translation.y - j, q.z + translation.z))
    return out


def test_identity_session_aligns_with_identity_transform():
    report = align_session(_cloud_a(), _cloud_a(), from_session="s2", to_session="s1")
    assert report.status == "accepted"
    assert report.method == "identity_verified"
    t = report.transform
    assert t is not None
    # The transform maps B points onto A points exactly.
    for pb, pa in zip(_cloud_a(), _cloud_a()):
        q = t.rotation.rotate(pb)
        assert math.isclose(q.x + t.translation.x, pa.x, abs_tol=1e-9)
        assert math.isclose(q.y + t.translation.y, pa.y, abs_tol=1e-9)


def test_transformed_session_recovers_the_real_transform():
    quat, translation = _transform_b()
    cloud_b = _cloud_b(quat, translation)
    report = align_session(cloud_b, _cloud_a(), from_session="s2", to_session="s1")
    assert report.status == "accepted"
    assert report.method == "icp"
    t = report.transform
    assert t is not None
    # Round-trip: transformed B points must land on their A counterparts.
    for pb, pa in zip(cloud_b, _cloud_a()):
        q = t.rotation.rotate(pb)
        assert math.isclose(q.x + t.translation.x, pa.x, abs_tol=0.15)
        assert math.isclose(q.y + t.translation.y, pa.y, abs_tol=0.15)
        assert math.isclose(q.z + t.translation.z, pa.z, abs_tol=0.15)
    assert report.rmse < 0.15


def test_disjoint_clouds_are_refused_not_identity_assumed():
    """B exists 100 units away with no overlap: no plausible contact."""
    cloud_b = [Vec3(p.x + 100.0, p.y + 100.0, p.z) for p in _cloud_a()]
    report = align_session(cloud_b, _cloud_a(), from_session="s2", to_session="s1")
    assert report.status == "refused"
    assert report.method == "none"
    assert report.transform is None
    assert report.reason  # explained, not silent


def test_too_few_points_is_refused():
    report = align_session([Vec3(0, 0, 0), Vec3(1, 0, 0)], _cloud_a(),
                           from_session="s2", to_session="s1")
    assert report.status == "refused"


def test_to_dict_roundtrip():
    report = align_session(_cloud_a(), _cloud_a(), from_session="s2", to_session="s1")
    d = report.to_dict()
    assert d["from_session"] == "s2"
    assert d["status"] == "accepted"


def test_chain_composes_transforms_and_labels_partial_paths():
    quat, translation = _transform_b()
    cloud_b = _cloud_b(quat, translation)
    # s1 -> s2 accepted; s3 has no shared structure with anything.
    report = align_session_chain(
        anchors={("s1", "s2"): align_session(cloud_b, _cloud_a(), "s2", "s1")},
        chain=["s1", "s2", "s3"],
    )
    assert isinstance(report, CrossSessionReport)
    assert report.transforms["s2"] is not None  # s2 -> s1 frame
    assert report.status_by_session["s1"] == "reference"
    assert report.status_by_session["s3"] == "unresolved"
    assert "s3" in report.reasons["s3"]


def test_chain_requires_reference_first():
    with pytest.raises(ValueError):
        align_session_chain(anchors={}, chain=[])
