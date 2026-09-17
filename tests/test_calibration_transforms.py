"""Tests for reconstruction/calibration/transforms.py (P2-02
calibration/frame unification: RigidTransform + CalibrationChain with
explicit frames, WGS84 geodetic/ECEF/ENU conversions, declared
SensorRig, reprojection-residual quality statistics).

Known answers used: WGS84 equator/prime-meridian surface point is
exactly the semi-major axis from Earth's center; the pole is the
semi-minor axis; a +90-degree-about-z rotation maps +x to +y; a 1-pixel
u-bias in observations produces mean_px == mean_u == 1.0.
"""

import json
import math

import pytest

from engine.physics.math3 import Quat, Vec3
from reconstruction.calibration.camera import (
    CameraExtrinsics,
    CameraIntrinsics,
    PinholeCamera,
)
from reconstruction.calibration.transforms import (
    CalibrationChain,
    CalibrationEntry,
    CalibrationTransformError,
    ReprojectionResidualStats,
    RigidTransform,
    SensorRig,
    ecef_to_enu_transform,
    ecef_to_geodetic,
    geodetic_to_ecef,
    geodetic_to_enu,
    enu_to_geodetic,
    per_camera_reprojection_residual_stats,
    reprojection_residual_stats,
)


def _q_z90() -> Quat:
    """+90 degrees about +z: maps +x to +y."""
    s = math.sqrt(0.5)
    return Quat(s, 0.0, 0.0, s)


class TestRigidTransform:
    def test_identity_leaves_points_unchanged(self):
        T = RigidTransform(from_frame="a", to_frame="b")
        p = Vec3(1.5, -2.0, 3.25)
        out = T.apply(p)
        assert abs(out.x - p.x) < 1e-12 and abs(out.y - p.y) < 1e-12 and abs(out.z - p.z) < 1e-12

    def test_rotation_maps_known_vector(self):
        T = RigidTransform(from_frame="a", to_frame="b", rotation=_q_z90())
        out = T.apply_direction(Vec3(1.0, 0.0, 0.0))
        assert abs(out.x) < 1e-12 and abs(out.y - 1.0) < 1e-12

    def test_directions_ignore_translation_points_do_not(self):
        T = RigidTransform(from_frame="a", to_frame="b", translation=Vec3(10.0, 0.0, 0.0))
        d = T.apply_direction(Vec3(1.0, 0.0, 0.0))
        assert abs(d.x - 1.0) < 1e-12 and abs(d.y) < 1e-12  # no translation
        p = T.apply(Vec3(1.0, 0.0, 0.0))
        assert abs(p.x - 11.0) < 1e-12  # with translation

    def test_inverse_roundtrips_exactly(self):
        T = RigidTransform(
            from_frame="camera", to_frame="rig",
            rotation=_q_z90(), translation=Vec3(1.0, 2.0, 3.0),
        )
        p = Vec3(0.5, -0.5, 2.0)
        back = T.inverse().apply(T.apply(p))
        assert abs(back.x - p.x) < 1e-12 and abs(back.y - p.y) < 1e-12 and abs(back.z - p.z) < 1e-12
        assert T.inverse().from_frame == "rig" and T.inverse().to_frame == "camera"

    def test_compose_connects_frames_and_carries_translation(self):
        T1 = RigidTransform(from_frame="camera", to_frame="rig", rotation=_q_z90(), translation=Vec3(1.0, 0.0, 0.0))
        T2 = RigidTransform(from_frame="rig", to_frame="world", translation=Vec3(0.0, 0.0, 10.0))
        total = T1.compose(T2)
        assert total.from_frame == "camera" and total.to_frame == "world"
        expected = T2.apply(T1.apply(Vec3(1.0, 0.0, 0.0)))
        out = total.apply(Vec3(1.0, 0.0, 0.0))
        assert abs(out.x - expected.x) < 1e-12 and abs(out.z - expected.z) < 1e-12

    def test_compose_frame_mismatch_raises(self):
        T = RigidTransform(from_frame="a", to_frame="b")
        with pytest.raises(CalibrationTransformError, match="compose"):
            T.compose(RigidTransform(from_frame="x", to_frame="c"))

    def test_non_unit_rotation_rejected(self):
        with pytest.raises(CalibrationTransformError, match="unit"):
            RigidTransform(from_frame="a", to_frame="b", rotation=Quat(2.0, 0.0, 0.0, 0.0))

    def test_empty_frame_names_rejected(self):
        with pytest.raises(CalibrationTransformError):
            RigidTransform(from_frame="", to_frame="b")

    def test_non_finite_translation_rejected(self):
        with pytest.raises(CalibrationTransformError, match="finite"):
            RigidTransform(from_frame="a", to_frame="b", translation=Vec3(float("nan"), 0.0, 0.0))

    def test_roundtrip_dict(self):
        T = RigidTransform(from_frame="camera", to_frame="rig", rotation=_q_z90(), translation=Vec3(1.0, 2.0, 3.0))
        restored = RigidTransform.from_dict(T.to_dict())
        assert restored.from_frame == T.from_frame and restored.to_frame == T.to_frame
        out = restored.apply(Vec3(1.0, 0.0, 0.0))
        expected = T.apply(Vec3(1.0, 0.0, 0.0))
        assert abs(out.x - expected.x) < 1e-12 and abs(out.y - expected.y) < 1e-12


class TestCalibrationChain:
    def _three_step(self):
        return CalibrationChain(steps=(
            RigidTransform(from_frame="cam", to_frame="rig", rotation=_q_z90(), translation=Vec3(1.0, 0.0, 0.0)),
            RigidTransform(from_frame="rig", to_frame="vehicle", translation=Vec3(0.0, 1.0, 0.0)),
            RigidTransform(from_frame="vehicle", to_frame="enu@37.0,-122.0", translation=Vec3(10.0, 0.0, 0.0)),
        ))

    def test_connectivity_validated_at_construction(self):
        with pytest.raises(CalibrationTransformError, match="gap"):
            CalibrationChain(steps=(
                RigidTransform(from_frame="a", to_frame="b"),
                RigidTransform(from_frame="x", to_frame="c"),
            ))

    def test_empty_chain_rejected(self):
        with pytest.raises(CalibrationTransformError, match="at least one"):
            CalibrationChain(steps=())

    def test_frames_are_first_from_and_last_to(self):
        chain = self._three_step()
        assert chain.from_frame == "cam"
        assert chain.to_frame == "enu@37.0,-122.0"

    def test_total_equals_stepwise_application(self):
        chain = self._three_step()
        p = Vec3(1.0, 0.0, 1.0)
        stepwise = p
        for step in chain.steps:
            stepwise = step.apply(stepwise)
        out = chain.apply(p)
        assert abs(out.x - stepwise.x) < 1e-12 and abs(out.y - stepwise.y) < 1e-12 and abs(out.z - stepwise.z) < 1e-12

    def test_roundtrip_dict(self):
        chain = self._three_step()
        restored = CalibrationChain.from_dict(chain.to_dict())
        out = restored.apply(Vec3(1.0, 0.0, 1.0))
        expected = chain.apply(Vec3(1.0, 0.0, 1.0))
        assert abs(out.x - expected.x) < 1e-12


class TestGNSSFrames:
    def test_equator_prime_meridian_is_semi_major_axis(self):
        e = geodetic_to_ecef(0.0, 0.0, 0.0)
        assert abs(e.x - 6378137.0) < 1e-6
        assert abs(e.y) < 1e-9 and abs(e.z) < 1e-9

    def test_north_pole_is_semi_minor_axis(self):
        e = geodetic_to_ecef(90.0, 0.0, 0.0)
        assert abs(e.z - 6356752.314245) < 1e-3
        assert abs(e.x) < 1e-6 and abs(e.y) < 1e-6

    @pytest.mark.parametrize("lat,lon,h", [
        (37.421, -122.084, 15.2),
        (-33.868, 151.209, 250.0),
        (0.0, 0.0, 100.0),
        (89.9, 45.0, 5.0),
    ])
    def test_ecef_geodetic_roundtrip(self, lat, lon, h):
        back = ecef_to_geodetic(geodetic_to_ecef(lat, lon, h))
        assert abs(back[0] - lat) < 1e-9
        assert abs(back[1] - lon) < 1e-9
        assert abs(back[2] - h) < 1e-6

    def test_lat_range_validated(self):
        with pytest.raises(CalibrationTransformError, match="range"):
            geodetic_to_ecef(91.0, 0.0, 0.0)
        with pytest.raises(CalibrationTransformError, match="range"):
            ecef_to_enu_transform(91.0, 0.0, 0.0)

    def test_enu_origin_maps_to_zero(self):
        ref = (37.421, -122.084, 100.0)
        zero = geodetic_to_enu(*ref, *ref)
        assert abs(zero.x) < 1e-9 and abs(zero.y) < 1e-9 and abs(zero.z) < 1e-9

    def test_enu_roundtrip_east_north_up(self):
        ref = (37.421, -122.084, 100.0)
        lat, lon, h = enu_to_geodetic(10.0, -5.0, 3.0, *ref)
        back = geodetic_to_enu(lat, lon, h, *ref)
        assert abs(back.x - 10.0) < 1e-6 and abs(back.y - (-5.0)) < 1e-6 and abs(back.z - 3.0) < 1e-6

    def test_enu_up_is_ellipsoidal_height(self):
        ref = (37.421, -122.084, 100.0)
        lat, lon, h = enu_to_geodetic(0.0, 0.0, 10.0, *ref)
        assert abs(h - 110.0) < 1e-6  # ellipsoidal, not geoid -- documented

    def test_east_displacement_moves_east(self):
        ref = (37.421, -122.084, 100.0)
        lat0, lon0, _ = ref
        _, lon_east, _ = enu_to_geodetic(10.0, 0.0, 0.0, *ref)
        assert lon_east > lon0  # east is increasing longitude in this hemisphere

    def test_enu_frame_name_carries_its_reference(self):
        T = ecef_to_enu_transform(37.421, -122.084, 100.0)
        assert T.from_frame == "ecef"
        assert "37.421" in T.to_frame and "-122.084" in T.to_frame

    def test_enu_transform_inverse_roundtrips(self):
        T = ecef_to_enu_transform(37.421, -122.084, 100.0)
        p_ecef = geodetic_to_ecef(37.5, -122.1, 120.0)
        back = T.inverse().apply(T.apply(p_ecef))
        assert abs(back.x - p_ecef.x) < 1e-6 and abs(back.y - p_ecef.y) < 1e-6 and abs(back.z - p_ecef.z) < 1e-6


class TestSensorRig:
    def _rig(self):
        return SensorRig(rig_frame="rig", entries=(
            CalibrationEntry(
                sensor_id="cam0", sensor_type="camera",
                transform=RigidTransform(from_frame="cam0", to_frame="rig", rotation=_q_z90(), translation=Vec3(1.0, 0.0, 0.0)),
                source="declared",
            ),
            CalibrationEntry(sensor_id="imu0", sensor_type="imu", source="estimated"),  # no pose: uncalibrated
        ))

    def test_chain_maps_sensor_frame_to_rig_frame(self):
        chain = self._rig().chain("cam0")
        assert chain.from_frame == "cam0" and chain.to_frame == "rig"
        out = chain.apply(Vec3(1.0, 0.0, 0.0))
        assert abs(out.x - 1.0) < 1e-12 and abs(out.y - 1.0) < 1e-12  # rotated +x->+y, then +t

    def test_uncalibrated_member_raises_honestly(self):
        with pytest.raises(CalibrationTransformError, match="recorded state"):
            self._rig().chain("imu0")

    def test_unknown_sensor_raises_with_members(self):
        with pytest.raises(CalibrationTransformError, match="imu0"):
            self._rig().chain("ghost")

    def test_duplicate_sensor_ids_rejected(self):
        with pytest.raises(CalibrationTransformError, match="duplicate"):
            SensorRig(rig_frame="rig", entries=(
                CalibrationEntry(sensor_id="a", sensor_type="camera"),
                CalibrationEntry(sensor_id="a", sensor_type="imu"),
            ))

    def test_calibration_source_must_be_labeled(self):
        with pytest.raises(CalibrationTransformError, match="provenance"):
            CalibrationEntry(sensor_id="a", sensor_type="camera", source=" vibes")

    def test_self_mapped_transform_rejected(self):
        with pytest.raises(CalibrationTransformError, match="itself"):
            CalibrationEntry(
                sensor_id="a", sensor_type="camera",
                transform=RigidTransform(from_frame="a", to_frame="a"),
            )

    def test_quality_scalars_validated(self):
        with pytest.raises(CalibrationTransformError, match="finite scalar"):
            CalibrationEntry(sensor_id="a", sensor_type="camera", quality={"sigma": [1, 2]})

    def test_roundtrip_and_file_load(self, tmp_path):
        rig = self._rig()
        path = tmp_path / "rig.json"
        path.write_text(json.dumps(rig.to_dict()))
        restored = SensorRig.from_file(str(path))
        assert restored.rig_frame == "rig"
        assert restored.entry("cam0").transform is not None
        assert restored.entry("imu0").transform is None  # absent stays absent
        assert restored.entry("imu0").source == "estimated"  # provenance travels


class TestReprojectionResiduals:
    def _camera(self):
        intr = CameraIntrinsics(
            fx=1000.0, fy=1000.0, cx=320.0, cy=240.0, width=640, height=480,
            k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0,
        )
        extr = CameraExtrinsics(position=Vec3(0.0, 0.0, 0.0), rotation=Quat.identity())
        return PinholeCamera(intrinsics=intr, extrinsics=extr)

    def _grid(self):
        pts = [Vec3(x, y, 5.0) for x in (-1.0, 0.0, 1.0) for y in (-1.0, 0.0, 1.0)]
        pixels = [(320.0 + 1000.0 * p.x / 5.0, 240.0 + 1000.0 * p.y / 5.0) for p in pts]
        return pts, pixels

    def test_exact_observations_zero_residual(self):
        cam = self._camera()
        pts, pixels = self._grid()
        stats = reprojection_residual_stats("cam0", cam, list(zip(pts, pixels)))
        assert stats.n == 9 and stats.n_unprojectable == 0
        assert stats.mean_px < 1e-9 and stats.max_px < 1e-9

    def test_known_pixel_bias_appears_in_means(self):
        cam = self._camera()
        pts, pixels = self._grid()
        stats = reprojection_residual_stats("cam0", cam, [(p, (u + 1.0, v)) for p, (u, v) in zip(pts, pixels)])
        assert abs(stats.mean_px - 1.0) < 1e-9
        assert abs(stats.mean_u - 1.0) < 1e-9
        assert abs(stats.mean_v) < 1e-9

    def test_median_and_p95_nearest_rank(self):
        cam = self._camera()
        p = Vec3(0.0, 0.0, 5.0)
        base = self._camera().project(p)
        obs = [(base[0] + d, base[1]) for d in (1.0, 2.0, 3.0, 4.0)]
        stats = reprojection_residual_stats("cam0", cam, [(p, o) for o in obs])
        assert stats.n == 4
        assert abs(stats.median_px - 2.5) < 1e-12
        assert abs(stats.p95_px - 4.0) < 1e-12  # nearest-rank: ceil(0.95*4)=4
        assert abs(stats.max_px - 4.0) < 1e-12

    def test_unprojectable_counted_not_dropped(self):
        cam = self._camera()
        pts, pixels = self._grid()
        stats = reprojection_residual_stats(
            "cam0", cam, list(zip(pts + [Vec3(0.0, 0.0, -3.0)], pixels + [(320.0, 240.0)])),
        )
        assert stats.n == 9 and stats.n_unprojectable == 1

    def test_empty_observations_raise(self):
        with pytest.raises(CalibrationTransformError, match="empty"):
            reprojection_residual_stats("cam0", self._camera(), [])

    def test_all_unprojectable_raises_no_fabrication(self):
        cam = self._camera()
        with pytest.raises(CalibrationTransformError, match="unprojectable"):
            reprojection_residual_stats("cam0", cam, [(Vec3(0.0, 0.0, -5.0), (320.0, 240.0))])

    def test_per_camera_stats(self):
        cam = self._camera()
        pts, pixels = self._grid()
        stats = per_camera_reprojection_residual_stats(
            {"cam0": cam, "cam1": self._camera()},
            {"cam0": list(zip(pts, pixels))},
        )
        assert stats["cam0"].n == 9
        assert "cam1" not in stats  # no observations -> nothing reported, never zeros

    def test_observations_for_unknown_camera_raise(self):
        cam = self._camera()
        with pytest.raises(CalibrationTransformError, match="no calibration"):
            per_camera_reprojection_residual_stats(
                {"cam0": cam}, {"ghost": [(Vec3(0, 0, 5.0), (0.0, 0.0))]}
            )

    def test_camera_without_observations_absent_not_zero(self):
        cam = self._camera()
        result = per_camera_reprojection_residual_stats({"cam0": cam, "cam1": self._camera()}, {})
        assert result == {}  # nothing measured -> nothing reported, never zeros
