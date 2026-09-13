"""End-to-end test of the full object-understanding chain built across
several sessions, now closed with WorldIR promotion:

  DepthMap + SegmentedRegion (two views of the same chair)
  -> lift_region_to_3d()          (perception/instances/lifting.py)
  -> merge_hypotheses()           (perception/instances/object_resolution.py)
  -> promote_object_to_entity()   (evidence/promote_objects.py)
  -> validated WorldIR

This is the real composition test proving the pipeline is one connected
system, not isolated units -- the same standard every other pipeline
stage in this repo (planes, rooms, physics) has already been held to.
"""

from __future__ import annotations

from engine.physics.math3 import Quat, Vec3
from evidence.promote_objects import promote_object_to_entity
from perception.depth.interface import DepthMap
from perception.instances.lifting import lift_region_to_3d
from perception.instances.object_resolution import merge_hypotheses
from perception.segmentation.interface import SegmentedRegion
from reconstruction.calibration.camera import CameraExtrinsics, CameraIntrinsics, PinholeCamera
from world_ir import WorldIR
from world_ir.validation import validate_world_ir

_WIDTH, _HEIGHT = 12, 10


def _camera(position: Vec3) -> PinholeCamera:
    intrinsics = CameraIntrinsics(fx=200.0, fy=200.0, cx=_WIDTH / 2, cy=_HEIGHT / 2, width=_WIDTH, height=_HEIGHT)
    return PinholeCamera(intrinsics=intrinsics, extrinsics=CameraExtrinsics(position=position, rotation=Quat.identity()))


def _rect_mask(row_lo, row_hi, col_lo, col_hi):
    return [[row_lo <= r < row_hi and col_lo <= c < col_hi for c in range(_WIDTH)] for r in range(_HEIGHT)]


def test_two_views_of_one_chair_produce_one_promoted_worldir_entity():
    # View 1: camera at the origin, chair mid-frame at depth 4m.
    camera_1 = _camera(Vec3.zero())
    depth_1 = DepthMap(evidence_id="photo-1", width=_WIDTH, height=_HEIGHT,
                        values=[[4.0] * _WIDTH for _ in range(_HEIGHT)], unit="meters")
    region_1 = SegmentedRegion(region_id="r1", evidence_id="photo-1", label="chair",
                                mask=_rect_mask(3, 7, 4, 8), confidence=0.85)

    # View 2: camera moved slightly (0.1m along x), same chair, same depth.
    camera_2 = _camera(Vec3(0.1, 0.0, 0.0))
    depth_2 = DepthMap(evidence_id="photo-2", width=_WIDTH, height=_HEIGHT,
                        values=[[4.0] * _WIDTH for _ in range(_HEIGHT)], unit="meters")
    region_2 = SegmentedRegion(region_id="r2", evidence_id="photo-2", label="chair",
                                mask=_rect_mask(3, 7, 4, 8), confidence=0.80)

    hyp_1 = lift_region_to_3d(region_1, depth_1, camera_1)
    hyp_2 = lift_region_to_3d(region_2, depth_2, camera_2)
    assert hyp_1 is not None and hyp_2 is not None

    candidates = merge_hypotheses([hyp_1, hyp_2], distance_threshold_m=0.5)
    assert len(candidates) == 1  # same object, two views -> one candidate
    candidate = candidates[0]
    assert candidate.observation_count == 2

    world = WorldIR()
    result = promote_object_to_entity(candidate, world, "ent-chair-1")

    assert result.entity.name == "chair"
    assert len(world.entities) == 1
    assert len(world.geometries) == 1

    report = validate_world_ir(world)
    assert report.is_valid(), [str(i) for i in report.issues]

    # Provenance trace intact end to end: the promoted entity's
    # observation names both source evidence photos.
    promotion_obs = next(o for o in result.entity.observations if o.sensor_type == "object_promotion")
    assert set(promotion_obs.metadata["evidence_ids"]) == {"photo-1", "photo-2"}


def test_two_different_objects_produce_two_promoted_entities():
    camera = _camera(Vec3.zero())
    depth = DepthMap(evidence_id="photo-1", width=_WIDTH, height=_HEIGHT,
                      values=[[4.0] * _WIDTH for _ in range(_HEIGHT)], unit="meters")
    chair_region = SegmentedRegion(region_id="r1", evidence_id="photo-1", label="chair",
                                    mask=_rect_mask(0, 3, 0, 3), confidence=0.8)
    table_region = SegmentedRegion(region_id="r2", evidence_id="photo-1", label="table",
                                    mask=_rect_mask(6, 9, 8, 11), confidence=0.8)

    chair_hyp = lift_region_to_3d(chair_region, depth, camera)
    table_hyp = lift_region_to_3d(table_region, depth, camera)
    candidates = merge_hypotheses([chair_hyp, table_hyp])
    assert len(candidates) == 2

    world = WorldIR()
    for i, candidate in enumerate(candidates):
        promote_object_to_entity(candidate, world, f"ent-{i}")

    assert len(world.entities) == 2
    assert {e.name for e in world.entities.values()} == {"chair", "table"}
    assert validate_world_ir(world).is_valid()
