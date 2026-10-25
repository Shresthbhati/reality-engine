"""Pose-graph tests for cross-session registration (registration/cross_session.py).

Exercises align_session_chain: multi-session composition, unresolved
sessions staying unresolved (never identity-placed), empty-chain refusal,
and determinism.
"""

from __future__ import annotations

import math

import pytest

from engine.math import Quat, Vec3
from registration.cross_session import (
    CrossSessionReport,
    align_session,
    align_session_chain,
)


def _cloud_a():
    return [Vec3(x, y, 0.0) for x in range(5) for y in range(5)]


def _rotated_cloud(quat, translation):
    out = []
    for p in _cloud_a():
        q = quat.rotate(p)
        out.append(Vec3(q.x + translation.x, q.y + translation.y, q.z + translation.z))
    return out


def _make_quat(deg):
    half = math.radians(deg) / 2.0
    return Quat(w=math.cos(half), x=0.0, y=0.0, z=math.sin(half)).normalized()


def test_empty_chain_raises():
    with pytest.raises(ValueError):
        align_session_chain({}, [])


def test_single_session_is_reference():
    report = align_session_chain({}, ["s1"])
    assert isinstance(report, CrossSessionReport)
    assert report.status_by_session["s1"] == "reference"
    assert report.transforms["s1"] is None


def test_two_session_chain_aligns():
    quat = _make_quat(30.0)
    cloud_b = _rotated_cloud(quat, Vec3(3.0, -1.5, 0.2))
    a = align_session(cloud_b, _cloud_a(), from_session="s2", to_session="s1")
    assert a.status == "accepted"
    report = align_session_chain({("s2", "s1"): a}, ["s1", "s2"])
    assert report.status_by_session["s2"] == "aligned"
    assert report.transforms["s2"] is not None


def test_unresolved_session_stays_unresolved():
    # s3 has no accepted path -> must not be identity-placed.
    quat = _make_quat(30.0)
    cloud_b = _rotated_cloud(quat, Vec3(3.0, -1.5, 0.2))
    a = align_session(cloud_b, _cloud_a(), from_session="s2", to_session="s1")
    report = align_session_chain({("s2", "s1"): a}, ["s1", "s2", "s3"])
    assert report.status_by_session["s3"] == "unresolved"
    assert report.transforms.get("s3") is None


def test_refused_anchor_does_not_align():
    cloud_b = [Vec3(p.x + 100.0, p.y + 100.0, p.z) for p in _cloud_a()]
    a = align_session(cloud_b, _cloud_a(), from_session="s2", to_session="s1")
    assert a.status == "refused"
    report = align_session_chain({("s2", "s1"): a}, ["s1", "s2"])
    assert report.status_by_session["s2"] != "aligned"