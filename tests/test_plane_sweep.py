"""F3: the CPU plane sweep must recover the depth of a textured plane to sub-decimetre accuracy, and must refuse a
textureless one instead of inventing geometry."""

import numpy as np
import pytest

from reconstruction.backend import dense_registry, plane_sweep


class _K:
    def __init__(self, fx=400.0, fy=400.0, cx=320.0, cy=240.0):
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy


def _texture(world_xy, size=32):
    """Non-periodic smooth noise: a checkerboard repeats, which makes plane sweep ambiguous at several depths."""
    import cv2

    base = np.random.default_rng(7).integers(0, 256, (size, size)).astype(np.float32)
    up = cv2.resize(base, (512, 512), interpolation=cv2.INTER_CUBIC)
    u = (world_xy[..., 0] + 2.0) / 4.0
    v = (world_xy[..., 1] + 2.0) / 4.0
    idx = np.clip((u * 511).astype(int), 0, 511), np.clip((v * 511).astype(int), 0, 511)
    return np.clip(up[idx], 0, 255).astype(np.uint8)


def _render(center, k, w=640, h=480, plane_z=2.0, textured=True):
    ys, xs = np.mgrid[0:h, 0:w]
    rays = np.stack([(xs - k.cx) / k.fx, (ys - k.cy) / k.fy, np.ones_like(xs, float)], -1)
    t = (plane_z - center[2]) / rays[..., 2]
    pts = np.asarray(center, float) + rays * t[..., None]
    if not textured:
        img = np.full((h, w, 3), 180, np.uint8)
    else:
        img = np.repeat(_texture(pts[..., :2])[..., None], 3, axis=2)
    return img


def _views(textured=True):
    k = _K()
    return [
        {"K": k, "R": (1.0, 0.0, 0.0, 0.0), "C": (0.0, 0.0, 0.0), "image": _render((0, 0, 0), k, textured=textured)},
        {"K": k, "R": (1.0, 0.0, 0.0, 0.0), "C": (0.4, 0.0, 0.0), "image": _render((0.4, 0, 0), k, textured=textured)},
    ]


def test_plane_sweep_recovers_known_plane_depth():
    report = plane_sweep.sweep_view(_views()[0], _views()[1:], 1.2, 3.0, planes=60, stride=2)
    assert len(report["depth"]) > 500, "a textured plane should yield dense depth"
    assert np.median(np.abs(report["depth"] - 2.0)) < 0.06
    assert np.median(report["cost"]) < 10.0
    # backprojection must land on the plane, not just produce a plausible depth number
    assert np.median(np.abs(report["points_world"][:, 2] - 2.0)) < 0.07


def test_plane_sweep_refuses_a_textureless_plane():
    views = _views(textured=False)
    report = plane_sweep.sweep_view(views[0], views[1:], 1.2, 3.0, planes=60, stride=2)
    assert len(report["depth"]) == 0, "a flat wall gives no photometric evidence; it must be dropped, not filled"


def test_plane_sweep_needs_two_views():
    assert plane_sweep.dense_plane_sweep(_views()[:1], depth_min_m=1.0, depth_max_m=3.0) == []
    assert dense_registry.select_dense_backends() == ("plane_sweep_cpu",)


def test_plane_sweep_rejects_an_inverted_range():
    with pytest.raises(ValueError):
        plane_sweep.sweep_view(_views()[0], _views()[1:], 3.0, 1.0)


def test_pipeline_stage_adds_plane_sweep_points_in_a_metric_world(tmp_path):
    """The stage derives its depth range from the sparse cloud and appends measured dense points to the result."""
    from PIL import Image

    from engine.pipeline.vertical_slice import VerticalSliceOptions, _plane_sweep_stage
    from evidence.session import EvidenceItem, EvidenceKind
    from reconstruction.backend.interface import (
        ReconstructedCameraPose, ReconstructedPoint, ReconstructionResult,
    )

    items, poses = [], []
    for i, view in enumerate(_views()):
        path = tmp_path / f"img_{i}.png"
        Image.fromarray(view["image"]).save(path)
        items.append(EvidenceItem(id=f"img_{i}", kind=EvidenceKind.IMAGE, source_uri=path.as_uri()))
        poses.append(ReconstructedCameraPose(evidence_id=f"img_{i}", position=view["C"], rotation=view["R"]))
    pts = [ReconstructedPoint(position=(x, y, 2.0), track_id=f"s{k}", source_evidence_ids=["img_0"])
           for k, (x, y) in enumerate((x, y) for x in np.linspace(-1, 1, 20) for y in np.linspace(-0.7, 0.7, 20))]
    result = ReconstructionResult(points=pts, camera_poses=poses, registration_status="success")
    options = VerticalSliceOptions(intrinsics=(400.0, 400.0, 320.0, 240.0), image_size=(640, 480),
                                   plane_sweep_enabled=True, plane_sweep_max_views=2, plane_sweep_stride=4)

    facts = _plane_sweep_stage(result, items, options, "metric")

    assert facts["status"] == "ran" and facts["points_added"] > 100
    assert facts["depth_range_m"][0] <= 2.0 <= facts["depth_range_m"][1]
    added = [p for p in result.points if p.track_id.startswith("plane_sweep:")]
    assert len(added) == facts["points_added"] and all(p.color is not None for p in added)
    assert abs(float(np.median([p.position[2] for p in added])) - 2.0) < 0.1


def test_pipeline_stage_refuses_to_sweep_a_relative_world():
    from engine.pipeline.vertical_slice import VerticalSliceOptions, _plane_sweep_stage

    options = VerticalSliceOptions(plane_sweep_enabled=True, intrinsics=(400.0, 400.0, 320.0, 240.0))
    facts = _plane_sweep_stage(None, [], options, "relative")
    assert facts["status"] == "skipped" and "model units" in facts["note"]
    assert _plane_sweep_stage(None, [], VerticalSliceOptions(), "metric")["status"] == "disabled"
