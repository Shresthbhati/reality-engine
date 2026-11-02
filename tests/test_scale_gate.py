"""Blueprint 3.3: relative depth gives depth_source=relative + wide sigma, and no metric dimensions until scale is known."""
from __future__ import annotations

from engine.math import Quat, Vec3
from perception.depth.interface import DepthMap
from perception.instances.association import observation_from_hypothesis
from perception.instances.lifting import lift_region_to_3d
from perception.segmentation.interface import SegmentedRegion
from reconstruction.calibration.camera import CameraExtrinsics, CameraIntrinsics, PinholeCamera
from world_ir.observations import DepthSource, SIGMA_FLOOR_M

W, H = 8, 6


def _camera():
    return PinholeCamera(CameraIntrinsics(fx=100.0, fy=100.0, cx=W / 2, cy=H / 2, width=W, height=H),
                         CameraExtrinsics(position=Vec3.zero(), rotation=Quat.identity()))


def _lift(unit, scale_state="relative"):
    depth = DepthMap("ev", W, H, [[5.0] * W for _ in range(H)], unit=unit)
    mask = [[r < 4 and c < 4 for c in range(W)] for r in range(H)]
    region = SegmentedRegion(region_id="r", evidence_id="ev", label="chair", mask=mask, confidence=0.9)
    return lift_region_to_3d(region, depth, _camera(), scale_state=scale_state)


def test_relative_depth_is_flagged_relative_with_wide_sigma_and_no_metric_dimensions():
    rel, met = _lift("relative"), _lift("meters")
    assert rel.depth_source is DepthSource.RELATIVE
    assert rel.metric_extents_m is None
    assert rel.sigma_m > 5 * met.sigma_m  # 25% of range vs 2%
    assert rel.sigma_m > SIGMA_FLOOR_M


def test_relative_depth_gets_dimensions_only_when_the_world_scale_is_known():
    h = _lift("relative", scale_state="estimated")
    assert h.depth_source is DepthSource.RELATIVE and h.metric_extents_m is not None


def test_metric_depth_keeps_metric_dimensions_and_source():
    h = _lift("meters")
    assert h.depth_source is DepthSource.METRIC
    assert h.metric_extents_m is not None and all(v >= 0 for v in h.metric_extents_m)


def test_relative_hypothesis_enters_the_ledger_as_relative_whatever_the_caller_says():
    h = _lift("relative")
    o = observation_from_hypothesis(h, DepthSource.METRIC)  # caller wrongly claims metric
    assert o.depth_source is DepthSource.RELATIVE and o.sigma_m >= h.sigma_m * 0.99
