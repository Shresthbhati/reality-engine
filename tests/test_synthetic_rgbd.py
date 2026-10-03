"""SYNTHETIC RGB-D VERIFIED -- PHYSICAL DEVICE VERIFICATION PENDING.

A rendered RGB-D capture (RGB + 16-bit depth + intrinsics + a declared depth scale, with sensor noise, dropout holes,
glass, flying pixels, range limits and invalid pixels) pushed through the REAL RGB-D pipeline: the 16-bit PNG parser,
the sidecar depth stage, metric-scale gating and unprojection, then the vertical slice's world compile. Ground truth
comes from the scene, so accuracy is a measured error, not 'something came out'.

This proves the pipeline handles the RGB-D CONTRACT. It says nothing about a physical device: noise, holes and
range limits here are a plausible model, not a calibration of any sensor.
"""

from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path

import numpy as np
import pytest

from synthetic.indoor import INDOOR_INTRINSICS, INDOOR_SENSOR_NOISY, panorama, single_room
from synthetic.rgbd import (
    GroundTruthBackend, Intrinsics, SensorModel, look_at, render_frame, write_capture,
)
from synthetic.scene import Scene, add_wall, distance_to_scene

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _wall_scene() -> Scene:
    s = Scene()
    add_wall(s, (-10.0, 6.0), (10.0, 6.0), 0.0, 8.0, ref="back-wall")
    return s


def _clean(**kw) -> SensorModel:
    base = dict(sigma0_m=0.0, sigma_z2=0.0, hole_fraction=0.0, salt_fraction=0.0, edge_jump_m=1e9, rgb_noise_levels=0.0)
    base.update(kw)
    return dataclasses.replace(SensorModel(), **base)


# ------------------------------------------------------------------------------------------------ the sensor model

def test_ideal_sensor_returns_the_renderers_exact_depth():
    ind = single_room()
    f = render_frame(ind.scene, INDOOR_INTRINSICS, ind.cameras[0], SensorModel.ideal(), 0)
    hit = np.isfinite(f.depth_true)
    assert hit.any() and np.array_equal(np.isfinite(f.depth_measured), hit)
    assert np.array_equal(f.depth_measured[hit], f.depth_true[hit])


def test_depth_noise_follows_the_documented_quadratic_law():
    intr = Intrinsics.from_hfov(160, 120, 60.0)
    sensor = _clean(sigma0_m=0.002, sigma_z2=0.002, max_range_m=20.0)
    for distance in (1.0, 3.0, 5.0):
        eye = (0.0, 6.0 - distance, 1.0)
        f = render_frame(_wall_scene(), intr, look_at(eye, (0.0, 6.0, 1.0)), sensor, 0)
        center = np.s_[50:70, 70:90]
        err = (f.depth_measured[center] - f.depth_true[center]).ravel()
        # depth along the optical axis is the wall distance at the centre; sigma(z) = sigma0 + sigma_z2 z^2
        expected = 0.002 + 0.002 * distance ** 2
        assert err.std() == pytest.approx(expected, rel=0.35), (distance, err.std(), expected)
        assert abs(err.mean()) < expected * 0.6


def test_noise_is_seeded_and_the_seed_matters():
    ind = single_room()
    sensor = INDOOR_SENSOR_NOISY
    a = render_frame(ind.scene, INDOOR_INTRINSICS, ind.cameras[3], sensor, 3)
    b = render_frame(ind.scene, INDOOR_INTRINSICS, ind.cameras[3], sensor, 3)
    c = render_frame(ind.scene, INDOOR_INTRINSICS, ind.cameras[3], dataclasses.replace(sensor, seed=99), 3)
    assert np.array_equal(a.depth_measured, b.depth_measured, equal_nan=True) and np.array_equal(a.rgb, b.rgb)
    assert not np.array_equal(a.depth_measured, c.depth_measured, equal_nan=True)


def test_every_invalid_pixel_is_accounted_for_by_a_named_cause():
    ind = single_room()
    holes_seen = flying_seen = 0
    for i in (0, 5, 11, 20):
        f = render_frame(ind.scene, INDOOR_INTRINSICS, ind.cameras[i], INDOOR_SENSOR_NOISY, i)
        invalid_total = int((~f.valid).sum())
        accounted = sum(f.invalid_breakdown.values())
        assert accounted <= invalid_total
        assert invalid_total - accounted <= 0.001 * f.valid.size, (i, f.invalid_breakdown, invalid_total)
        holes_seen += f.invalid_breakdown["holes"]
        flying_seen += f.invalid_breakdown["flying"]
    assert holes_seen > 0 and flying_seen > 0, "the sensor model injected no holes / flying pixels"


def test_range_limits_make_far_and_near_surfaces_invalid():
    intr = Intrinsics.from_hfov(160, 120, 60.0)
    f = render_frame(_wall_scene(), intr, look_at((0.0, 3.0, 1.0), (0.0, 6.0, 1.0)),
                     _clean(max_range_m=2.0, min_range_m=0.3), 0)          # wall is 3 m away > 2 m max
    assert not f.valid.any()
    assert f.invalid_breakdown["out_of_range"] + f.invalid_breakdown["no_hit"] == f.valid.size   # no_hit: rays above the wall
    assert f.invalid_breakdown["out_of_range"] > 0.8 * f.valid.size
    g = render_frame(_wall_scene(), intr, look_at((0.0, 5.9, 1.0), (0.0, 6.0, 1.0)), _clean(min_range_m=0.3), 0)
    assert not g.valid[60, 80], "a surface nearer than min_range returned depth"


# ------------------------------------------------------------------------------- the capture + the real parser

@pytest.fixture(scope="module")
def capture(tmp_path_factory):
    ind = single_room()
    cams = panorama((1.3, 1.0, 1.4), yaws=4, pitches_deg=(-30, 0)) + panorama((3.7, 1.0, 1.4), yaws=4,
                                                                                pitches_deg=(-30, 0))
    frames = [render_frame(ind.scene, INDOOR_INTRINSICS, p, INDOOR_SENSOR_NOISY, i) for i, p in enumerate(cams)]
    root = write_capture(tmp_path_factory.mktemp("rgbd") / "capture", frames)
    return ind, frames, root


def test_capture_is_labelled_synthetic_and_ships_its_ground_truth(capture):
    import json

    _ind, frames, root = capture
    m = json.loads((root / "manifest.json").read_text())
    assert m["synthetic"]["label"] == "SYNTHETIC RGB-D VERIFIED"
    assert m["synthetic"]["pending"] == "PHYSICAL DEVICE VERIFICATION PENDING"
    assert (root / "ground_truth" / "poses.json").is_file() and (root / "ground_truth" / "depth_true.npz").is_file()
    assert len(list((root / "images").glob("*.jpg"))) == len(list((root / "depth").glob("*.png"))) == len(frames)


def test_the_real_parser_reads_back_the_rendered_depth_and_masks_invalid_pixels(capture):
    from PIL import Image

    from evidence.depth_frames import DepthSidecarManifest, parse_depth_frames

    _ind, frames, root = capture
    manifest = DepthSidecarManifest.load(str(root / "depth" / "manifest.json"))
    paths = sorted(str(p) for p in (root / "depth").glob("*.png"))
    parsed = parse_depth_frames(paths, manifest=manifest)
    assert [p.frame_id for p in parsed] == [f"img_{f.index:03d}" for f in frames]
    assert Image.open(paths[0]).mode in ("I;16", "I")
    for p, f in zip(parsed, frames):
        assert p.depth_dtype == "uint16" and p.depth_scale == pytest.approx(0.001) and p.units == "meter"
        assert p.invalid_pixel_count == int((~f.valid).sum()), "invalid pixels must be exactly the sensor's"
        rr, cc = np.nonzero(f.valid)
        pick = np.random.default_rng(0).integers(0, len(rr), 200)
        for k in pick:
            assert p.meters_at(int(rr[k]), int(cc[k])) == pytest.approx(float(f.depth_measured[rr[k], cc[k]]),
                                                                          abs=0.0006)
        ir, ic = np.nonzero(~f.valid)
        assert all(p.meters_at(int(ir[k]), int(ic[k])) is None for k in range(0, len(ir), max(1, len(ir) // 100))), \
            "an invalid pixel was read as a distance"


def test_depth_without_a_declared_scale_is_refused_not_guessed(tmp_path):
    from evidence.depth_frames import DepthScaleUnavailable, parse_depth_frames

    ind = single_room()
    frames = [render_frame(ind.scene, INDOOR_INTRINSICS, ind.cameras[0], INDOOR_SENSOR_NOISY, 0),
              render_frame(ind.scene, INDOOR_INTRINSICS, ind.cameras[9], INDOOR_SENSOR_NOISY, 1)]
    root = write_capture(tmp_path / "rel", frames, declare_depth_scale=False)
    assert not (root / "depth" / "manifest.json").exists()
    with pytest.raises(DepthScaleUnavailable):
        parse_depth_frames(sorted(str(p) for p in (root / "depth").glob("*.png")))


# ------------------------------------------------------------------------------- through the real pipeline stages

def _run(root: Path):
    from engine.pipeline.dataset import load_capture_dataset
    from engine.pipeline.vertical_slice import VerticalSliceOptions, vertical_slice
    from world_ir.artifact_store import MemoryArtifactStore

    items, refs, intr, size = load_capture_dataset(root)
    backend = GroundTruthBackend()
    sparse = len(backend.reconstruct(items).points)
    result = vertical_slice(items, VerticalSliceOptions(
        measured_baselines=refs, intrinsics=intr, image_size=size, depth_model=None, mesh_enabled=False,
        perception_model=None, detail_enabled=False, artifact_store=MemoryArtifactStore(),
        reconstruction_backend=backend))
    return result, sparse


def test_metric_capture_unprojects_every_valid_sampled_pixel_onto_the_true_surfaces(capture):
    ind, frames, root = capture
    result, sparse = _run(root)
    depth = result.stage_facts["depth"]
    assert result.scale_state == "metric"
    assert depth["status"] == "ran" and depth["source"] == "sidecar"
    assert depth["frames_matched"] == len(frames) and depth["failed_views"] == 0 and not depth["errors"]
    stride = depth["stride"]
    expected = sum(int(f.valid[::stride, ::stride].sum()) for f in frames)
    assert depth["dense_points"] == expected, "unprojected a different number of pixels than the sensor made valid"
    assert result.points_total - sparse == expected
    # the pipeline rotates the model into a gravity-canonical frame (rotation only, recorded in world.metadata["frame"]);
    # undo it to compare against the scene the frames were rendered from: x_old = R^T x_new
    rot = np.array(result.world.metadata["frame"]["rotation"], dtype=float)
    assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-9) and np.linalg.det(rot) == pytest.approx(1.0)
    dense = np.array(result.points[-expected:]) @ rot
    err = distance_to_scene(ind.scene, dense)
    assert np.median(err) < 0.02 and np.percentile(err, 95) < 0.07, (np.median(err), np.percentile(err, 95))


def test_the_compiled_world_is_valid_and_metric(capture):
    from world_ir.validation import validate_world_ir

    _ind, _frames, root = capture
    result, _ = _run(root)
    assert not validate_world_ir(result.world).errors
    assert result.meters_per_unit == pytest.approx(1.0, abs=1e-6)
    assert result.cameras_registered == result.cameras_input == 16


def test_relative_scale_world_refuses_device_metres(tmp_path):
    """No measured baseline -> the world stays RELATIVE -> metric depth must not be unprojected into it."""
    ind = single_room()
    frames = [render_frame(ind.scene, INDOOR_INTRINSICS, p, INDOOR_SENSOR_NOISY, i)
              for i, p in enumerate(panorama((1.3, 1.0, 1.4), yaws=4, pitches_deg=(0,)))]
    root = write_capture(tmp_path / "relworld", frames, measured_baseline=False)
    result, sparse = _run(root)
    depth = result.stage_facts["depth"]
    assert result.scale_state != "metric"
    assert depth["status"] == "skipped" and "non-metric" in depth["note"] and result.points_total == sparse


def test_undeclared_depth_scale_fails_the_depth_stage_honestly(tmp_path):
    ind = single_room()
    frames = [render_frame(ind.scene, INDOOR_INTRINSICS, p, INDOOR_SENSOR_NOISY, i)
              for i, p in enumerate(panorama((1.3, 1.0, 1.4), yaws=4, pitches_deg=(0,)) +
                                    panorama((3.7, 1.0, 1.4), yaws=2, pitches_deg=(0,)))]
    root = write_capture(tmp_path / "noscale", frames, declare_depth_scale=False)
    result, sparse = _run(root)
    depth = result.stage_facts["depth"]
    assert depth["status"] == "failed" and depth["frames_parsed"] == 0 and depth["dense_points"] == 0
    assert any("depth_scale" in e for e in depth["errors"]) and result.points_total == sparse


def test_the_pipeline_result_is_deterministic(capture):
    _ind, _frames, root = capture
    a, _ = _run(root)
    b, _ = _run(root)
    digest = lambda r: hashlib.sha256(repr(r.points).encode()).hexdigest()    # noqa: E731
    assert digest(a) == digest(b) and a.stage_facts["depth"] == b.stage_facts["depth"]
