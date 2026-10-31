"""Plan M2: image-projected surface appearance (perception/materials/surface_appearance.py).

SYNTHETIC cameras and images: a known textured wall is ray-cast into posed pinhole cameras, with per-camera exposure /
white-balance changes, a specular blob and an occluder painted into some photos. Every assertion is an error against the
known texture. Nothing here says anything about a real room.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from engine.math import Quat, Vec3
from perception.materials.surface_appearance import (
    LIGHTING_NOTE, PlanarSurface, SurfaceView, attach_appearance, linear_to_lab, linear_to_srgb,
    measure_surface_appearance, srgb_to_linear, views_from_poses,
)
from reconstruction.calibration.camera import CameraExtrinsics, CameraIntrinsics, PinholeCamera
from world_ir.artifact_store import MemoryArtifactStore

W, H = 160, 120
INTR = CameraIntrinsics(fx=110.0, fy=110.0, cx=W / 2, cy=H / 2, width=W, height=H)
BASE = (180, 90, 60)           # the wall's true colour
STRIPE = (60, 60, 160)         # a real feature: a vertical stripe 0.4 m wide at x in [2.0, 2.4]
WALL = PlanarSurface("wall-y0", np.array([[0, 0, 0], [4, 0, 0], [4, 0, 2.5], [0, 0, 2.5]], float), "wall",
                     interior_point=(2.0, 2.0, 1.2))


def _de(a, b):
    return float(np.linalg.norm(linear_to_lab(srgb_to_linear(np.array(a, float))) - linear_to_lab(srgb_to_linear(np.array(b, float)))))


def _look_at(pos, target, up=(0, 0, 1)):
    """Camera (x right, y down, z forward) at ``pos`` looking at ``target``."""
    pos, target = np.array(pos, float), np.array(target, float)
    z = target - pos
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=1)
    qx, qy, qz, qw = Rotation.from_matrix(R).as_quat()
    return PinholeCamera(INTR, CameraExtrinsics(Vec3(*pos), Quat(w=qw, x=qx, y=qy, z=qz)))


def _wall_colour(x, z):
    c = np.empty(x.shape + (3,))
    c[...] = BASE
    c[(x >= 2.0) & (x <= 2.4)] = STRIPE
    return c


def _render(cam: PinholeCamera, gain=(1.0, 1.0, 1.0), blob=None, occluder=None):
    """Ray-cast the plane y=0 (x in [0,4], z in [0,2.5]) into ``cam``. Returns (image uint8, z-depth float).
    ``gain`` multiplies LINEAR light per channel (exposure / white balance). ``blob`` = (u0,u1,v0,v1) saturated pixels;
    ``occluder`` = (u0,u1,v0,v1, depth) pixels that show a nearer grey object."""
    q = cam.extrinsics.rotation
    R = np.array([[1 - 2 * (q.y ** 2 + q.z ** 2), 2 * (q.x * q.y - q.w * q.z), 2 * (q.x * q.z + q.w * q.y)],
                  [2 * (q.x * q.y + q.w * q.z), 1 - 2 * (q.x ** 2 + q.z ** 2), 2 * (q.y * q.z - q.w * q.x)],
                  [2 * (q.x * q.z - q.w * q.y), 2 * (q.y * q.z + q.w * q.x), 1 - 2 * (q.x ** 2 + q.y ** 2)]])
    C = np.array([cam.extrinsics.position.x, cam.extrinsics.position.y, cam.extrinsics.position.z])
    uu, vv = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    d_cam = np.stack([(uu - INTR.cx) / INTR.fx, (vv - INTR.cy) / INTR.fy, np.ones_like(uu)], axis=-1)
    d_w = d_cam @ R.T
    with np.errstate(divide="ignore", invalid="ignore"):
        t = -C[1] / d_w[..., 1]
    P = C + d_w * t[..., None]
    hit = (t > 0) & (P[..., 0] >= 0) & (P[..., 0] <= 4) & (P[..., 2] >= 0) & (P[..., 2] <= 2.5)
    img = np.full((H, W, 3), 20.0)
    depth = np.zeros((H, W))
    lin = srgb_to_linear(_wall_colour(P[..., 0], P[..., 2])) * np.array(gain)
    img[hit] = linear_to_srgb(lin[hit])
    depth[hit] = t[hit]                                  # d_cam has z=1, so t is the z-depth
    if blob is not None:
        u0, u1, v0, v1 = blob
        img[v0:v1, u0:u1] = 255
    if occluder is not None:
        u0, u1, v0, v1, dz = occluder
        img[v0:v1, u0:u1] = (110, 110, 110)
        depth[v0:v1, u0:u1] = dz
    return np.clip(np.round(img), 0, 255).astype(np.uint8), depth


def _cams():
    # three interior viewpoints looking at the wall from different positions / angles
    return [_look_at((1.0, 3.0, 1.2), (1.2, 0, 1.2)), _look_at((2.0, 3.5, 1.3), (2.0, 0, 1.2)),
            _look_at((3.2, 2.8, 1.1), (2.8, 0, 1.2))]


def _views(gains=((1, 1, 1), (1, 1, 1), (1, 1, 1)), with_depth=False, **extra):
    out = []
    for i, (cam, g) in enumerate(zip(_cams(), gains)):
        img, dep = _render(cam, g, **extra.get(f"v{i}", {}))
        out.append(SurfaceView(f"v{i}", cam, img, dep if with_depth else None))
    return out


def test_recovers_the_true_wall_colour():
    rep = measure_surface_appearance([WALL], _views())
    a = rep.appearances["wall-y0"]
    assert a.color_state == "measured_approximate" and a.lighting == LIGHTING_NOTE
    assert _de(a.color_rgb, BASE) < 3.0, a.color_rgb
    assert a.views_used == ("v0", "v1", "v2")


def test_exposure_and_white_balance_are_normalised_across_views():
    gains = ((1.0, 1.0, 1.0), (0.5, 0.55, 0.45), (1.15, 1.0, 0.8))     # three different exposures / white balances
    views = _views(gains)
    raw = measure_surface_appearance([WALL], views, normalise=False).appearances["wall-y0"]
    fixed = measure_surface_appearance([WALL], views).appearances["wall-y0"]
    # gauge: the gains are relative (geometric mean 1), so the result is the colour under the AVERAGE exposure
    mean_gain = np.exp(np.log(np.array(gains)).mean(axis=0))
    expected = linear_to_srgb(srgb_to_linear(np.array(BASE)) * mean_gain)
    assert _de(fixed.color_rgb, expected) < 1.5, (fixed.color_rgb, expected)
    # without F5 the result is whatever the best-weighted photo happened to record
    assert _de(raw.color_rgb, expected) > 5.0 > _de(fixed.color_rgb, expected)


def test_view_gains_are_reported_and_undo_the_synthetic_exposure():
    gains = ((1.0, 1.0, 1.0), (0.5, 0.55, 0.45), (1.15, 1.0, 0.8))
    rep = measure_surface_appearance([WALL], _views(gains))
    assert all(rep.gain_estimated.values())
    g = np.array([rep.view_gains[f"v{i}"] for i in range(3)])
    true = np.array(gains) / np.exp(np.log(np.array(gains)).mean(axis=0))   # relative to the geometric mean
    # fitted corrective gain is the inverse of the relative exposure
    assert np.allclose(g * true, 1.0, atol=0.08), (g, true)


def test_specular_blob_in_one_photo_is_rejected():
    views = _views(v1={"blob": (40, 100, 30, 80)})
    a = measure_surface_appearance([WALL], views).appearances["wall-y0"]
    assert _de(a.color_rgb, BASE) < 4.0, a.color_rgb
    assert a.rejected_fraction > 0.0


def test_occluder_is_excluded_by_depth_when_depth_is_given():
    views = _views(with_depth=True, v1={"occluder": (30, 110, 20, 100, 1.5)})
    a = measure_surface_appearance([WALL], views).appearances["wall-y0"]
    assert _de(a.color_rgb, BASE) < 3.0


def test_atlas_matches_the_texture_and_marks_unobserved_texels():
    store = MemoryArtifactStore()
    # a single narrow-FOV photo sees only part of the 4 m wall
    cam = _look_at((1.0, 2.0, 1.2), (1.0, 0, 1.2))
    img, dep = _render(cam)
    a = measure_surface_appearance([WALL], [SurfaceView("only", cam, img, dep)], artifact_store=store,
                                   texel_m=0.05).appearances["wall-y0"]
    rgba = a.atlas_rgba
    assert rgba.shape[2] == 4 and a.atlas_size == (rgba.shape[1], rgba.shape[0])
    seen = rgba[..., 3] == 255
    assert 0.05 < a.coverage < 0.95 and seen.any() and (~seen).any()
    assert a.coverage == pytest.approx(seen.sum() / seen.size, abs=0.01)
    assert (rgba[~seen] == 0).all(), "unobserved texels must stay empty, not inpainted"
    # texel (row, col) -> wall (x, z): row 0 = top (z=2.5); seen from the interior (+y) x runs right-to-left, so col 0 = x=4
    h, w = rgba.shape[:2]
    cols_seen = np.flatnonzero(seen.any(axis=0))
    x_seen = 4.0 - (cols_seen + 0.5) / w * 4.0
    assert x_seen.min() > 0.0 and x_seen.max() < 4.0
    for c in cols_seen[:: max(1, len(cols_seen) // 12)]:
        r = np.flatnonzero(seen[:, c])[len(np.flatnonzero(seen[:, c])) // 2]
        x = 4.0 - (c + 0.5) / w * 4.0
        want = STRIPE if 2.03 < x < 2.37 else BASE if (x < 1.97 or x > 2.43) else None
        if want is not None:
            assert _de(rgba[r, c, :3], want) < 6.0, (x, rgba[r, c], want)
    assert any("not observed" in n for n in a.notes) and any("one photo" in n for n in a.notes)


def test_atlas_is_a_hash_addressed_deterministic_png_artifact():
    from PIL import Image
    import io

    store = MemoryArtifactStore()
    views = _views()
    a = measure_surface_appearance([WALL], views, artifact_store=store).appearances["wall-y0"]
    b = measure_surface_appearance([WALL], views, artifact_store=store).appearances["wall-y0"]
    assert a.atlas_uri == b.atlas_uri and a.atlas_hash == b.atlas_hash and a.atlas_uri.startswith("artifact://")
    png = store.get(a.atlas_uri)
    im = Image.open(io.BytesIO(png))
    assert im.mode == "RGBA" and im.size == a.atlas_size
    assert np.array_equal(np.asarray(im), a.atlas_rgba)
    prop = a.to_property()
    assert prop["texture"]["uri"] == a.atlas_uri and prop["texture"]["hash"] == a.atlas_hash
    assert prop["texture"]["media_type"] == "image/png" and prop["color_state"] == "measured_approximate"
    assert prop["lighting"] == LIGHTING_NOTE


def test_attach_appearance_writes_the_entity_reference():
    class _E:
        custom_properties: dict = {}

    e = _E()
    e.custom_properties = {}
    a = measure_surface_appearance([WALL], _views(), artifact_store=MemoryArtifactStore()).appearances["wall-y0"]
    attach_appearance(e, a)
    assert e.custom_properties["appearance"]["texture"]["hash"] == a.atlas_hash


def test_surface_nobody_saw_is_not_measured_not_guessed():
    behind = PlanarSurface("wall-y5", np.array([[0, 5, 0], [4, 5, 0], [4, 5, 2.5], [0, 5, 2.5]], float), "wall",
                           interior_point=(2.0, 2.0, 1.2))
    a = measure_surface_appearance([behind], _views()).appearances["wall-y5"]
    assert a.color_rgb is None and a.color_state == "not_measured" and a.coverage == 0.0
    assert a.to_property()["color_rgb"] is None and a.to_property()["texture"] is None
    assert a.atlas_rgba is not None and (a.atlas_rgba == 0).all()


def test_no_views_is_not_measured():
    a = measure_surface_appearance([WALL], []).appearances["wall-y0"]
    assert a.color_state == "not_measured" and a.color_rgb is None


def test_views_from_the_far_side_do_not_paint_the_interior_face():
    outside = [_look_at((1.0, -3.0, 1.2), (1.2, 0, 1.2))]
    img, _ = _render(outside[0])                                   # sees only the black background
    a = measure_surface_appearance([WALL], [SurfaceView("out", outside[0], img)]).appearances["wall-y0"]
    assert a.color_state == "not_measured"
    free = PlanarSurface("wall-any", WALL.polygon, "wall")          # no interior_point: either side accepted
    assert measure_surface_appearance([free], [SurfaceView("out", outside[0], img)]).appearances["wall-any"].views_used


def test_grazing_views_are_dropped():
    cam = _look_at((6.0, 0.15, 1.2), (3.0, 0, 1.2))                # nearly in the wall plane
    img, _ = _render(cam)
    strict = measure_surface_appearance([WALL], [SurfaceView("g", cam, img)], cos_min=0.5).appearances["wall-y0"]
    assert strict.color_state == "not_measured"


def test_non_planar_or_degenerate_polygons_are_refused():
    bad = PlanarSurface("bad", np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0.5], [0, 1, 0]], float))
    with pytest.raises(ValueError):
        measure_surface_appearance([bad], _views())
    with pytest.raises(ValueError):
        measure_surface_appearance([PlanarSurface("line", np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], float))], _views())


def test_polygon_with_a_cut_corner_leaves_the_cut_region_outside_the_atlas_mask():
    poly = np.array([[0, 0, 0], [4, 0, 0], [4, 0, 1.0], [2.0, 0, 2.5], [0, 0, 2.5]], float)   # sloped top-right corner
    s = PlanarSurface("cut", poly, "wall", interior_point=(2.0, 2.0, 1.2))
    a = measure_surface_appearance([s], _views()).appearances["cut"]
    assert a.color_state == "measured_approximate" and _de(a.color_rgb, BASE) < 3.0
    assert a.atlas_rgba[0, 0, 3] == 0, "the cut-off corner texel (seen from inside: top-left) lies outside the polygon and must stay empty"


def test_views_from_poses_skips_poses_without_photos():
    from reconstruction.backend.interface import ReconstructedCameraPose

    poses = [ReconstructedCameraPose("a", (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
             ReconstructedCameraPose("b", (1.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))]
    out = views_from_poses(poses, INTR, {"a": np.zeros((H, W, 3), np.uint8)})
    assert [v.view_id for v in out] == ["a"]
