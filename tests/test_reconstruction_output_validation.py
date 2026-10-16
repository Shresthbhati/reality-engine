"""A reconstruction only counts as success when its OUTPUT is valid (release gate, Phase 2).

Every malformed backend output must make the vertical slice refuse (VerticalSliceError) so the
progressive ladder records a failed attempt instead of compiling it into a world version. The same
stub with a well-formed output must still compile, so the refusals are not a blanket failure.
"""
from __future__ import annotations

import dataclasses
import math

import pytest

from engine.pipeline.vertical_slice import VerticalSliceError, VerticalSliceOptions, vertical_slice
from evidence.session import EvidenceItem, EvidenceKind
from reconstruction.backend.interface import (
    IReconstructionBackend,
    ReconstructedCameraPose,
    ReconstructedPoint,
)
from tests.test_room_inference import _CAMS, _two_room_scene

ITEMS = [EvidenceItem(id=f"ev-{i}", kind=EvidenceKind.PHOTO, source_uri=f"file:///photo{i}.jpg") for i in range(3)]


def _poses(n=3):
    return [
        ReconstructedCameraPose(evidence_id=f"ev-{i}", position=_CAMS[i], rotation=(1.0, 0.0, 0.0, 0.0))
        for i in range(n)
    ]


class _Stub(IReconstructionBackend):
    def __init__(self, **overrides):
        self._overrides = overrides

    def reconstruct(self, evidence):
        base = _two_room_scene()
        fields = dict(camera_poses=_poses(), registration_status="success")
        fields.update(self._overrides)
        return dataclasses.replace(base, **fields)


def _run(**overrides):
    return vertical_slice(ITEMS, VerticalSliceOptions(reconstruction_backend=_Stub(**overrides)))


def test_a_well_formed_output_still_compiles():
    out = _run()
    assert out.cameras_registered == 3 and len(out.world.entities) > 0


@pytest.mark.parametrize(
    "overrides, why",
    [
        (dict(camera_poses=[], registration_status="partial"), "0 registered cameras"),
        (dict(camera_poses=_poses(1), registration_status="partial"), "1 registered camera"),
        (
            dict(camera_poses=[dataclasses.replace(_poses()[0], position=(math.nan, 0.0, 0.0))] + _poses()[1:]),
            "NaN camera position",
        ),
        (
            dict(camera_poses=[dataclasses.replace(_poses()[0], rotation=(1.0, 0.0, math.inf, 0.0))] + _poses()[1:]),
            "Inf camera rotation",
        ),
        (dict(points=[]), "empty point cloud"),
    ],
)
def test_a_malformed_output_is_refused_not_compiled(overrides, why):
    with pytest.raises(VerticalSliceError):
        _run(**overrides)


def test_a_non_finite_point_is_refused_not_compiled():
    pts = list(_two_room_scene().points)
    pts[0] = ReconstructedPoint(position=(math.inf, 0.0, 0.0), track_id="bad", source_evidence_ids=["ev-0"])
    with pytest.raises(VerticalSliceError, match="non-finite"):
        _run(points=pts)


def test_only_registered_photos_are_published_as_cameras():
    """A level-0 fusion lays unrelated single-view hypotheses out side by side for display; their 'poses' are not
    registrations and must not be served from /cameras as if COLMAP had placed those photographs."""
    from types import SimpleNamespace

    from apps.api.jobs import _registered_camera_poses

    poses = (
        ("ev-a", (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
        ("ev-b", (4.5, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
        ("ev-c", (9.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
    )
    fused = SimpleNamespace(camera_poses=poses, registered_ids=[])
    assert _registered_camera_poses(fused) == []
    multiview = SimpleNamespace(camera_poses=poses, registered_ids=["ev-a", "ev-b", "ev-c"])
    assert _registered_camera_poses(multiview) == list(poses)
    lone = SimpleNamespace(camera_poses=poses[:1], registered_ids=["ev-a"])
    assert [p[0] for p in _registered_camera_poses(lone)] == ["ev-a"]
    partial = SimpleNamespace(camera_poses=poses, registered_ids=["ev-b"])
    assert [p[0] for p in _registered_camera_poses(partial)] == ["ev-b"]
