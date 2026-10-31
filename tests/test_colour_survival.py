"""Plan M1: per-point colour must survive every stage between the photos and the wall/floor/ceiling entities.

Stages covered: dense points unprojected from a depth map (colour sampled from the SAME view's photo), COLMAP's
fused.ply (RGB vertex properties), the stages that rebuild points (rotation, provenance stamp), and the compiler.
A point with no measured colour carries ``color=None`` and the entity then has NO ``*_color_rgb`` -- never a default.
SYNTHETIC scene (synthetic/indoor.py); says nothing about a real room.
"""

from __future__ import annotations

import struct
from functools import lru_cache

import numpy as np
import pytest

from engine.compiler import CompileOptions, compile_reconstruction_to_world
from engine.math import Quat, Vec3
from perception.depth.interface import DepthMap
from reconstruction.backend.dense_output import parse_fused_ply
from reconstruction.backend.interface import ReconstructedPoint, ReconstructionResult
from reconstruction.calibration.camera import CameraExtrinsics, CameraIntrinsics, PinholeCamera, camera_from_pose
from reconstruction.depth_to_points import DepthToPointsError, depth_map_to_points, load_image_rgb
from scripts.room_fidelity_report import delta_e76
from synthetic.indoor import INDOOR_INTRINSICS, fidelity_ground_truth, fidelity_room, observe
from world_ir import EntityType

UP = (0.0, 0.0, 1.0)


# ------------------------------------------------------------------------------------------ depth -> points

def _cam(w=8, h=4):
    return PinholeCamera(intrinsics=CameraIntrinsics(fx=50.0, fy=50.0, cx=w / 2, cy=h / 2, width=w, height=h),
                         extrinsics=CameraExtrinsics(position=Vec3.zero(), rotation=Quat.identity()))


def _depth(w=8, h=4):
    return DepthMap(evidence_id="ev", width=w, height=h, values=[[3.0] * w for _ in range(h)], unit="meters")


def _two_tone(w=8, h=4):
    img = np.zeros((h, w, 3), np.uint8)
    img[:, : w // 2] = (200, 10, 10)
    img[:, w // 2:] = (10, 10, 200)
    return img


def test_dense_points_take_the_colour_of_the_pixel_they_came_from():
    pts = depth_map_to_points(_depth(), _cam(), image_rgb=_two_tone())
    assert len(pts) == 32 and all(p.color is not None for p in pts)
    for p in pts:
        assert p.color == ((200, 10, 10) if p.position[0] < 0 else (10, 10, 200))


def test_photo_with_a_different_resolution_is_sampled_not_misaligned():
    pts = depth_map_to_points(_depth(), _cam(), image_rgb=_two_tone(16, 8))
    assert {p.color for p in pts} == {(200, 10, 10), (10, 10, 200)}
    assert all((p.color == (200, 10, 10)) == (p.position[0] < 0) for p in pts)


def test_without_a_photo_colour_is_none_not_a_default():
    assert all(p.color is None for p in depth_map_to_points(_depth(), _cam()))


def test_photo_with_the_wrong_aspect_ratio_is_refused():
    with pytest.raises(DepthToPointsError):
        depth_map_to_points(_depth(), _cam(), image_rgb=np.zeros((8, 8, 3), np.uint8))


def test_load_image_rgb_reads_files_and_refuses_unreadable_ones(tmp_path):
    from PIL import Image

    path = tmp_path / "a.png"
    Image.fromarray(_two_tone()).save(path)
    assert np.array_equal(load_image_rgb(str(path)), _two_tone())
    assert np.array_equal(load_image_rgb(path.as_uri()), _two_tone())
    with pytest.raises(DepthToPointsError):
        load_image_rgb(str(tmp_path / "missing.png"))


# ------------------------------------------------------------------------------------------ fused.ply

def _ply(rows, with_rgb: bool) -> bytes:
    props = "property float x\nproperty float y\nproperty float z\n" + (
        "property uchar red\nproperty uchar green\nproperty uchar blue\n" if with_rgb else "")
    head = f"ply\nformat binary_little_endian 1.0\nelement vertex {len(rows)}\n{props}end_header\n".encode("ascii")
    fmt = "<fff" + ("BBB" if with_rgb else "")
    return head + b"".join(struct.pack(fmt, *r) for r in rows)


def test_fused_ply_rgb_is_kept():
    pts = parse_fused_ply(_ply([(0, 0, 1, 10, 20, 30), (1, 0, 1, 255, 0, 7)], True))
    assert [p.color for p in pts] == [(10, 20, 30), (255, 0, 7)]


def test_fused_ply_without_rgb_leaves_colour_unmeasured():
    pts = parse_fused_ply(_ply([(0, 0, 1), (1, 0, 1)], False))
    assert [p.color for p in pts] == [None, None]


# ------------------------------------------------------------------------------------------ rebuild stages

def test_stages_that_rebuild_points_keep_colour():
    from reconstruction.frame import _rotate_result
    from reconstruction.orchestrator import ReconstructionOrchestrator

    p = ReconstructedPoint(position=(1.0, 2.0, 3.0), track_id="t", source_evidence_ids=["e"], color=(1, 2, 3))
    q = ReconstructedPoint(position=(0.0, 0.0, 1.0), track_id="u", source_evidence_ids=["e"])
    r = ReconstructionResult(points=[p, q], camera_poses=[], registration_status="success")
    rot = _rotate_result(r, np.eye(3))
    assert {x.track_id: x.color for x in rot.points} == {"t": (1, 2, 3), "u": None}
    stamped = ReconstructionOrchestrator._stamp_provenance(r, "fake")
    assert [x.color for x in stamped.points] == [(1, 2, 3), None]


# ------------------------------------------------------------------------------------------ photos -> entities

@lru_cache(maxsize=None)
def _compiled(from_photos: bool):
    """Dense points from the fixture's depth + RGB frames, compiled by the real compiler. No other colour source."""
    ind = fidelity_room()
    obs = observe(ind, colors=None, voxel_m=0.08)
    f0 = obs.frames[0]
    i = INDOOR_INTRINSICS
    intr = CameraIntrinsics(fx=i.fx, fy=i.fy, cx=i.cx, cy=i.cy, width=f0.rgb.shape[1], height=f0.rgb.shape[0])
    poses = {p.evidence_id: p for p in obs.result.camera_poses}
    pts = []
    for f in obs.frames:
        eid = f"ev-{f.index:03d}"
        d = np.where(f.valid, f.depth_measured, 0.0)
        dm = DepthMap(evidence_id=eid, width=d.shape[1], height=d.shape[0], values=d.tolist(), unit="meters")
        pts += depth_map_to_points(dm, camera_from_pose(intr, poses[eid]), stride=8,
                                   image_rgb=f.rgb if from_photos else None)
    res = ReconstructionResult(points=pts, camera_poses=obs.result.camera_poses, registration_status="success")
    world, _ = compile_reconstruction_to_world(res, CompileOptions(seed=42, up=UP))
    return world, pts


def _structure(world):
    return [e for t in (EntityType.FLOOR, EntityType.CEILING, EntityType.WALL)
            for e in world.entities.values() if e.type == t]


def test_photo_sampled_colour_reaches_every_wall_floor_ceiling_and_the_room():
    world, pts = _compiled(True)
    assert pts and all(p.color is not None for p in pts)
    ents = _structure(world)
    # >= 6: 4 walls + floor + ceiling (a tabletop may additionally be promoted as a floor-type plane; not this test's concern)
    assert len(ents) >= 6
    assert all(e.custom_properties.get("mean_color_rgb") for e in ents)
    room = next(e for e in world.entities.values() if e.type == EntityType.ROOM)
    assert room.custom_properties.get("floor_color_rgb") and room.custom_properties.get("wall_color_rgb")


def test_measured_colours_are_the_right_colours_not_just_present():
    """Every ground-truth surface colour is recovered by some entity of the right type (shaded photos darken every
    surface, so the tolerance is loose; M2's exposure normalisation addresses the shading)."""
    gt = fidelity_ground_truth(fidelity_room())
    world, _ = _compiled(True)
    want = {EntityType.FLOOR: [gt["floor"]["color_rgb"]], EntityType.CEILING: [gt["ceiling"]["color_rgb"]],
            EntityType.WALL: [w["color_rgb"] for w in gt["walls"]]}
    for kind, colours in want.items():
        got = [e.custom_properties["mean_color_rgb"] for e in _structure(world) if e.type == kind]
        for c in colours:
            assert min(delta_e76(g, c) for g in got) < 25.0, (kind, c, got)


def test_unmeasured_colour_is_absent_never_a_default():
    world, pts = _compiled(False)
    assert all(p.color is None for p in pts)
    assert not any(e.custom_properties.get("mean_color_rgb") for e in world.entities.values())
    room = next(e for e in world.entities.values() if e.type == EntityType.ROOM)
    assert not room.custom_properties.get("floor_color_rgb") and not room.custom_properties.get("wall_color_rgb")
