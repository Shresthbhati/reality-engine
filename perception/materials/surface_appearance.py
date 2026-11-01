"""Image-projected surface appearance (plan M2): measured colour + planar texture atlas per wall/floor/ceiling.

Input is deliberately small and structure-agnostic: a PLANAR POLYGON in world coordinates (a footprint-derived wall,
a floor/ceiling outline, or a future solid-wall face) and a set of posed photos (``SurfaceView``). Nothing here reads
WorldIR entities, so it works for plane entities today and wall solids later.

What it does
  1. Rectifies the polygon into a texel grid (``texel_m``, capped at ``max_texels`` per side).
  2. Projects every texel centre into every camera that can see it: in front of the camera, inside the image, on the
     interior side (when ``interior_point`` is given), not grazing (|cos| >= ``cos_min``) and, when the view carries a
     depth map, not occluded. Pixels are bilinearly sampled in LINEAR light. Weight = cos(angle) / distance (more
     source pixels per texel).
  3. F5 - per-view white-balance/exposure normalisation: surface points seen by several views must agree, so a
     per-view per-channel gain is fitted in the log domain (alternating least squares, outlier-clipped, gauge:
     geometric-mean gain = 1). The ABSOLUTE illuminant is not recoverable from photos, so every colour is reported as
     ``approximate under capture lighting`` - never as albedo.
  4. Robust surface colour: weighted median in CIELAB over all samples after rejecting lightness outliers (specular
     highlights, shadows) and colour outliers (occluders).
  5. Texture atlas: per texel, a robust blend of the views that saw it (median-gated, weight-averaged). Texels no view
     saw stay alpha=0 and RGB=0 - they are never inpainted. Stored as PNG (RGBA) in the content-addressed
     ``ArtifactStore`` exactly like meshes; ``SurfaceAppearance.to_property()`` is the reference to store on the entity.

Honest limits: no occlusion test without a depth map (the median/outlier gates are the only defence then); lens
distortion uses the same Brown-Conrady model as ``PinholeCamera``; the gain gauge fixes relative, not absolute, colour;
the interior side needs ``interior_point`` - without it a view from behind the surface would also contribute.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from reconstruction.calibration.camera import PinholeCamera, quat_to_matrix

LIGHTING_NOTE = ("approximate under capture lighting: per-view white balance and exposure are normalised across the "
                 "photo set, but the absolute illuminant is unknown, so this is not the surface albedo")


# ------------------------------------------------------------------------------------------------ colour science

def srgb_to_linear(rgb255: np.ndarray) -> np.ndarray:
    c = np.asarray(rgb255, dtype=np.float64) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(lin: np.ndarray) -> np.ndarray:
    c = np.clip(np.asarray(lin, dtype=np.float64), 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055) * 255.0


_M_RGB2XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                       [0.2126729, 0.7151522, 0.0721750],
                       [0.0193339, 0.1191920, 0.9503041]])
_WHITE = np.array([0.95047, 1.0, 1.08883])


def linear_to_lab(lin: np.ndarray) -> np.ndarray:
    xyz = (np.asarray(lin, dtype=np.float64) @ _M_RGB2XYZ.T) / _WHITE
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], axis=-1)


def lab_to_linear(lab: np.ndarray) -> np.ndarray:
    lab = np.asarray(lab, dtype=np.float64)
    fy = (lab[..., 0] + 16) / 116
    fx, fz = fy + lab[..., 1] / 500, fy - lab[..., 2] / 200
    f = np.stack([fx, fy, fz], axis=-1)
    xyz = np.where(f ** 3 > 216 / 24389, f ** 3, (116 * f - 16) * 27 / 24389) * _WHITE
    return xyz @ np.linalg.inv(_M_RGB2XYZ).T


def lab_to_srgb255(lab: np.ndarray) -> np.ndarray:
    return linear_to_srgb(lab_to_linear(lab))


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Per-column weighted median of (n, k) values."""
    out = np.empty(values.shape[1])
    for j in range(values.shape[1]):
        order = np.argsort(values[:, j], kind="stable")
        cw = np.cumsum(weights[order])
        out[j] = values[order[min(int(np.searchsorted(cw, cw[-1] / 2.0)), len(order) - 1)], j]
    return out


# ------------------------------------------------------------------------------------------------ inputs / outputs

@dataclass(frozen=True, eq=False)
class SurfaceView:
    """One posed photo. ``depth_z`` (optional) is the metric z-depth per pixel (same size as the image); with it,
    occluded texels are excluded. Without it occlusion is NOT tested."""

    view_id: str
    camera: PinholeCamera
    image_rgb: np.ndarray
    depth_z: Optional[np.ndarray] = None


@dataclass(frozen=True, eq=False)
class PlanarSurface:
    """A planar polygon (>= 3 coplanar world vertices). ``interior_point`` (e.g. the room centroid) selects the side
    the photos must be taken from. ``up`` orients the atlas (row 0 = top) for non-horizontal surfaces.
    With ``interior_point`` the atlas is the surface as seen from the interior (u runs left-to-right for that viewer)."""

    surface_id: str
    polygon: np.ndarray
    kind: str = "wall"
    interior_point: Optional[Tuple[float, float, float]] = None
    up: Tuple[float, float, float] = (0.0, 0.0, 1.0)


@dataclass(frozen=True, eq=False)
class SurfaceAppearance:
    surface_id: str
    kind: str
    #: sRGB 0-255, None when no photo observed the surface (colour_state == "not_measured")
    color_rgb: Optional[Tuple[int, int, int]]
    color_lab: Optional[Tuple[float, float, float]]
    color_state: str
    lighting: str
    views_used: Tuple[str, ...]
    n_samples: int
    rejected_fraction: float
    #: median CIE76 distance of inlier samples from the surface colour (how uniform the observed surface is)
    spread_delta_e: Optional[float]
    #: fraction of the surface's texels at least one photo saw
    coverage: float
    notes: Tuple[str, ...] = ()
    atlas_uri: Optional[str] = None
    atlas_hash: Optional[str] = None
    atlas_size: Optional[Tuple[int, int]] = None
    texel_m: Optional[float] = None
    frame: Optional[dict] = None
    #: (H, W, 4) uint8 RGBA, alpha 0 = unobserved (never inpainted). Not serialised.
    atlas_rgba: Optional[np.ndarray] = None
    #: (H, W) uint8 number of photos behind each texel (0 = unobserved)
    n_obs: Optional[np.ndarray] = None

    def to_property(self) -> dict:
        """JSON-safe value for ``entity.custom_properties["appearance"]`` (no schema change needed)."""
        prop = {
            "color_rgb": list(self.color_rgb) if self.color_rgb else None,
            "color_state": self.color_state,
            "lighting": self.lighting,
            "views_used": list(self.views_used),
            "n_samples": self.n_samples,
            "rejected_fraction": round(self.rejected_fraction, 4),
            "spread_delta_e": None if self.spread_delta_e is None else round(self.spread_delta_e, 2),
            "coverage": round(self.coverage, 4),
            "notes": list(self.notes),
            "texture": None,
        }
        if self.atlas_uri:
            prop["texture"] = {"uri": self.atlas_uri, "hash": self.atlas_hash, "media_type": "image/png",
                               "width": self.atlas_size[0], "height": self.atlas_size[1], "texel_m": self.texel_m,
                               "frame": self.frame, "unobserved": "alpha=0 (never inpainted)",
                               "observed_fraction": round(self.coverage, 4)}
        return prop


@dataclass(frozen=True)
class AppearanceReport:
    appearances: Dict[str, SurfaceAppearance]
    #: per-view linear-light RGB gain applied (1,1,1 = unchanged / not estimated)
    view_gains: Dict[str, Tuple[float, float, float]]
    gain_estimated: Dict[str, bool]
    notes: Tuple[str, ...] = ()


# ------------------------------------------------------------------------------------------------ geometry

def _plane_frame(surface: PlanarSurface) -> dict:
    poly = np.asarray(surface.polygon, dtype=np.float64)
    if poly.ndim != 2 or poly.shape[1] != 3 or len(poly) < 3:
        raise ValueError(f"{surface.surface_id}: polygon must be (N>=3, 3), got {poly.shape}")
    # Newell normal
    n = np.zeros(3)
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        n += np.array([(a[1] - b[1]) * (a[2] + b[2]), (a[2] - b[2]) * (a[0] + b[0]), (a[0] - b[0]) * (a[1] + b[1])])
    if np.linalg.norm(n) < 1e-12:
        raise ValueError(f"{surface.surface_id}: degenerate polygon (zero area)")
    n /= np.linalg.norm(n)
    if surface.interior_point is not None and (np.asarray(surface.interior_point, float) - poly[0]) @ n < 0:
        n = -n                                         # the atlas is the surface as seen FROM the interior side
    off = np.abs((poly - poly[0]) @ n)
    if off.max() > 0.05 * max(np.ptp(poly, axis=0).max(), 1e-9) + 1e-6:
        raise ValueError(f"{surface.surface_id}: polygon is not planar (max offset {off.max():.3f})")
    up = np.asarray(surface.up, dtype=np.float64)
    v = up - (up @ n) * n
    if np.linalg.norm(v) < 0.2:                       # horizontal surface: orient by the first edge
        e = poly[1] - poly[0]
        v = e - (e @ n) * n
    v /= np.linalg.norm(v)
    u = np.cross(v, n)
    o = poly[0]
    uv = np.stack([(poly - o) @ u, (poly - o) @ v], axis=1)
    return {"poly": poly, "n": n, "u": u, "v": v, "origin": o, "uv": uv}


def _in_polygon(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Even-odd rule, vectorised. pts (m,2), poly (k,2)."""
    x, y = pts[:, 0], pts[:, 1]
    inside = np.zeros(len(pts), bool)
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi, xj, yj = poly[i, 0], poly[i, 1], poly[j, 0], poly[j, 1]
        cross = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / (yj - yi + 1e-300) + xi)
        inside ^= cross
        j = i
    return inside


def _texel_grid(fr: dict, texel_m: float, max_texels: int) -> dict:
    uv = fr["uv"]
    (u0, v0), (u1, v1) = uv.min(axis=0), uv.max(axis=0)
    texel = max(texel_m, max(u1 - u0, v1 - v0) / max_texels, 1e-4)
    nx, ny = max(1, int(math.ceil((u1 - u0) / texel))), max(1, int(math.ceil((v1 - v0) / texel)))
    cu = u0 + (np.arange(nx) + 0.5) * texel
    cv = v1 - (np.arange(ny) + 0.5) * texel           # row 0 = top (max v)
    gu, gv = np.meshgrid(cu, cv)
    pts2 = np.stack([gu.ravel(), gv.ravel()], axis=1)
    inside = _in_polygon(pts2, uv)
    world = fr["origin"] + np.outer(pts2[:, 0], fr["u"]) + np.outer(pts2[:, 1], fr["v"])
    return {"nx": nx, "ny": ny, "texel": texel, "inside": inside, "world": world,
            "u_range": (float(u0), float(u1)), "v_range": (float(v0), float(v1))}


# ------------------------------------------------------------------------------------------------ sampling

def _cam_arrays(cam: PinholeCamera) -> Tuple[np.ndarray, np.ndarray]:
    R = np.asarray(quat_to_matrix(cam.extrinsics.rotation), dtype=np.float64)
    e = cam.extrinsics.position
    return R, np.array([e.x, e.y, e.z], dtype=np.float64)


def _project(cam: PinholeCamera, P: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    R, C = _cam_arrays(cam)
    pc = (P - C) @ R                                   # world_to_camera = R^T (p - C)
    z = pc[:, 2]
    zs = np.where(z > 1e-6, z, 1.0)
    x, y = pc[:, 0] / zs, pc[:, 1] / zs
    k = cam.intrinsics
    r2 = x * x + y * y
    radial = 1.0 + k.k1 * r2 + k.k2 * r2 ** 2 + k.k3 * r2 ** 3
    xd = x * radial + 2.0 * k.p1 * x * y + k.p2 * (r2 + 2.0 * x * x)
    yd = y * radial + k.p1 * (r2 + 2.0 * y * y) + 2.0 * k.p2 * x * y
    return k.fx * xd + k.cx, k.fy * yd + k.cy, z


def _bilinear(img_lin: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    h, w = img_lin.shape[:2]
    x, y = u - 0.5, v - 0.5                            # pixel centres at +0.5
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    x0c, x1c = np.clip(x0, 0, w - 1), np.clip(x0 + 1, 0, w - 1)
    y0c, y1c = np.clip(y0, 0, h - 1), np.clip(y0 + 1, 0, h - 1)
    return ((img_lin[y0c, x0c] * (1 - fx) + img_lin[y0c, x1c] * fx) * (1 - fy)
            + (img_lin[y1c, x0c] * (1 - fx) + img_lin[y1c, x1c] * fx) * fy)


@dataclass
class _Obs:
    view_idx: int
    texel: np.ndarray       # flat texel indices (into the surface grid)
    lin: np.ndarray         # (m, 3) linear RGB as photographed (not yet gain-corrected)
    weight: np.ndarray


def _observe_surface(surface: PlanarSurface, fr: dict, grid: dict, views: Sequence[SurfaceView],
                     images_lin: List[np.ndarray], cos_min: float) -> List[_Obs]:
    idx = np.flatnonzero(grid["inside"])
    P, n = grid["world"][idx], fr["n"]
    side = None
    if surface.interior_point is not None:
        side = np.sign((np.asarray(surface.interior_point, float) - fr["origin"]) @ n)
    out: List[_Obs] = []
    for vi, view in enumerate(views):
        u, v, z = _project(view.camera, P)
        h, w = view.image_rgb.shape[:2]
        _, C = _cam_arrays(view.camera)
        to_cam = C - P
        dist = np.linalg.norm(to_cam, axis=1)
        cos = np.abs(to_cam @ n) / np.maximum(dist, 1e-9)
        ok = (z > 1e-3) & (u >= 0.5) & (u <= w - 0.5) & (v >= 0.5) & (v <= h - 0.5) & (cos >= cos_min)
        if side is not None and side != 0:
            ok &= np.sign(to_cam @ n) == side
        if view.depth_z is not None:
            dz = np.asarray(view.depth_z)
            d = dz[np.clip(v.astype(int), 0, dz.shape[0] - 1), np.clip(u.astype(int), 0, dz.shape[1] - 1)]
            valid = np.isfinite(d) & (d > 0)
            ok &= ~(valid & (d < z - np.maximum(0.05, 0.03 * z)))        # something nearer than the surface
        if not ok.any():
            continue
        out.append(_Obs(vi, idx[ok], _bilinear(images_lin[vi], u[ok], v[ok]), (cos[ok] / np.maximum(dist[ok], 1e-3))))
    return out


# ------------------------------------------------------------------------------------------------ F5 normalisation

def _fit_gains(all_obs: List[List[_Obs]], n_views: int, iters: int = 12) -> Tuple[np.ndarray, np.ndarray]:
    """Per-view per-channel log-gains so co-observed surface points agree. Returns (log_gain (V,3), estimated (V,))."""
    rows = []                                          # (key, view, log_rgb)
    for s, obs_list in enumerate(all_obs):
        for ob in obs_list:
            good = (ob.lin > 0.02).all(axis=1) & (ob.lin < 0.97).all(axis=1)       # drop black and clipped samples
            if good.any():
                rows.append((ob.texel[good] + s * 10_000_000, np.full(int(good.sum()), ob.view_idx),
                             np.log(ob.lin[good])))
    if not rows:
        return np.zeros((n_views, 3)), np.zeros(n_views, bool)
    key, view, logc = np.concatenate([r[0] for r in rows]), np.concatenate([r[1] for r in rows]), np.concatenate([r[2] for r in rows])
    uniq, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
    multi = counts[inv] >= 2                           # only points two or more photos saw carry exposure information
    if not multi.any():
        return np.zeros((n_views, 3)), np.zeros(n_views, bool)
    key_i, view, logc = inv[multi], view[multi], logc[multi]
    if len(key_i) > 400_000:                           # deterministic thinning
        sel = np.arange(0, len(key_i), math.ceil(len(key_i) / 400_000))
        key_i, view, logc = key_i[sel], view[sel], logc[sel]
    k = int(key_i.max()) + 1
    g = np.zeros((n_views, 3))
    keep = np.ones(len(key_i), bool)
    for it in range(iters):
        corrected = logc + g[view]
        w = keep.astype(float)
        cnt = np.maximum(np.bincount(key_i, weights=w, minlength=k), 1e-9)
        m = np.stack([np.bincount(key_i, weights=w * corrected[:, c], minlength=k) / cnt for c in range(3)], axis=1)
        resid = m[key_i] - logc                         # what each view's gain should have been, per sample
        w = keep.astype(float)
        vc = np.maximum(np.bincount(view, weights=w, minlength=n_views), 1e-9)
        g = np.stack([np.bincount(view, weights=w * resid[:, c], minlength=n_views) / vc for c in range(3)], axis=1)
        if it >= 1:                                     # clip outlier samples (occluders, specular) after a first pass
            err = np.abs(resid - g[view]).max(axis=1)
            keep = err <= max(0.35, 3 * 1.4826 * float(np.median(err)))
    estimated = np.bincount(view, minlength=n_views) > 0
    if estimated.any():
        g[estimated] -= g[estimated].mean(axis=0)       # gauge: geometric-mean gain = 1 (relative, not absolute)
    g[~estimated] = 0.0
    return g, estimated


# ------------------------------------------------------------------------------------------------ per-surface result

def _png_rgba(rgba: np.ndarray) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def _finish_surface(surface: PlanarSurface, fr: dict, grid: dict, obs_list: List[_Obs], views: Sequence[SurfaceView],
                    gains: np.ndarray, store, outlier_delta_e: float) -> SurfaceAppearance:
    total = int(grid["inside"].sum())
    frame = {"origin": [round(float(x), 6) for x in fr["origin"]], "u_axis": [round(float(x), 6) for x in fr["u"]],
             "v_axis": [round(float(x), 6) for x in fr["v"]], "normal": [round(float(x), 6) for x in fr["n"]],
             "u_range_m": [round(x, 6) for x in grid["u_range"]], "v_range_m": [round(x, 6) for x in grid["v_range"]],
             "row0": "max v (top)"}
    nx, ny = grid["nx"], grid["ny"]
    rgba = np.zeros((ny, nx, 4), np.uint8)
    n_obs = np.zeros(ny * nx, np.uint8)
    if not obs_list or total == 0:
        return SurfaceAppearance(surface.surface_id, surface.kind, None, None, "not_measured", LIGHTING_NOTE, (), 0, 0.0,
                                 None, 0.0, ("no photo observed this surface (outside every field of view, behind it, "
                                             "grazing, or occluded) - colour NOT measured",),
                                 frame=frame, atlas_rgba=rgba, n_obs=n_obs.reshape(ny, nx))
    tex = np.concatenate([o.texel for o in obs_list])
    lin = np.concatenate([o.lin * np.exp(gains[o.view_idx]) for o in obs_list])
    wt = np.concatenate([o.weight for o in obs_list])
    lab = linear_to_lab(np.clip(lin, 0.0, 1.5))

    # --- robust surface colour: reject lightness outliers (speculars, shadows) and colour outliers, weighted median Lab
    med = _weighted_median(lab, wt)
    mad_l = float(np.median(np.abs(lab[:, 0] - med[0]))) * 1.4826
    ok = np.abs(lab[:, 0] - med[0]) <= max(3.0 * mad_l, 8.0)
    ok &= np.linalg.norm(lab - med, axis=1) <= max(3.0 * 1.4826 * float(np.median(np.linalg.norm(lab - med, axis=1))), outlier_delta_e)
    if ok.sum() < max(3, 0.1 * len(ok)):
        ok = np.ones(len(ok), bool)                    # nothing consistent to gate against: keep all, spread says so
    surf_lab = _weighted_median(lab[ok], wt[ok])
    spread = float(np.median(np.linalg.norm(lab[ok] - surf_lab, axis=1)))
    rgb = tuple(int(round(v)) for v in np.clip(lab_to_srgb255(surf_lab), 0, 255))

    # --- atlas: per texel robust blend
    order = np.argsort(tex, kind="stable")
    tex_s, lab_s, wt_s, lin_s = tex[order], lab[order], wt[order], lin[order]
    starts = np.flatnonzero(np.r_[True, tex_s[1:] != tex_s[:-1]])
    ends = np.r_[starts[1:], len(tex_s)]
    flat = rgba.reshape(-1, 4)
    for a, b in zip(starts, ends):
        t, L, W = tex_s[a], lab_s[a:b], wt_s[a:b]
        n_obs[t] = min(b - a, 255)
        if b - a >= 3:
            m = np.median(L, axis=0)
            inl = np.linalg.norm(L - m, axis=1) <= outlier_delta_e
            L, W = (L[inl], W[inl]) if inl.any() else (m[None], np.ones(1))
        elif b - a == 2 and np.linalg.norm(L[0] - L[1]) > outlier_delta_e:
            L, W = L[[int(np.argmax(W))]], W[[int(np.argmax(W))]]          # two photos disagree: keep the better view
        flat[t, :3] = np.clip(lab_to_srgb255((L * W[:, None]).sum(axis=0) / W.sum()), 0, 255).round().astype(np.uint8)
        flat[t, 3] = 255
    coverage = float((n_obs[grid["inside"]] > 0).sum() / total)

    uri = h = None
    if store is not None:
        uri, h = store.put(_png_rgba(rgba))
    notes = []
    if coverage < 1.0:
        notes.append(f"{(1 - coverage) * 100:.0f}% of the surface was not observed by any photo; those texels are "
                     "transparent in the atlas (not inpainted)")
    if len({views[o.view_idx].view_id for o in obs_list}) == 1:
        notes.append("only one photo saw this surface: exposure/white balance could not be cross-checked")
    return SurfaceAppearance(
        surface.surface_id, surface.kind, rgb, tuple(round(float(x), 3) for x in surf_lab), "measured_approximate",
        LIGHTING_NOTE, tuple(sorted({views[o.view_idx].view_id for o in obs_list})), int(len(ok)),
        float(1.0 - ok.mean()), spread, coverage, tuple(notes), uri, h, (nx, ny), float(grid["texel"]), frame,
        rgba, n_obs.reshape(ny, nx))


# ------------------------------------------------------------------------------------------------ public API

def measure_surface_appearance(
    surfaces: Sequence[PlanarSurface],
    views: Sequence[SurfaceView],
    *,
    artifact_store=None,
    texel_m: float = 0.05,
    max_texels: int = 256,
    cos_min: float = 0.2,
    outlier_delta_e: float = 12.0,
    normalise: bool = True,
) -> AppearanceReport:
    """Measure colour + atlas for every surface from the photo set. ``normalise=False`` skips the F5 gain fit
    (kept for A/B comparison and for photo sets known to share one fixed exposure)."""
    views = list(views)
    images_lin = [srgb_to_linear(v.image_rgb[..., :3]) for v in views]
    frames = [_plane_frame(s) for s in surfaces]
    grids = [_texel_grid(f, texel_m, max_texels) for f in frames]
    observed = [_observe_surface(s, f, g, views, images_lin, cos_min) for s, f, g in zip(surfaces, frames, grids)]

    if normalise and views:
        log_g, est = _fit_gains(observed, len(views))
    else:
        log_g, est = np.zeros((len(views), 3)), np.zeros(len(views), bool)

    appearances = {s.surface_id: _finish_surface(s, f, g, o, views, log_g, artifact_store, outlier_delta_e)
                   for s, f, g, o in zip(surfaces, frames, grids, observed)}
    notes = []
    if normalise and views and not est.any():
        notes.append("no surface point was seen by two or more photos: exposure/white balance could not be normalised")
    return AppearanceReport(
        appearances,
        {v.view_id: tuple(round(float(x), 5) for x in np.exp(log_g[i])) for i, v in enumerate(views)},
        {v.view_id: bool(est[i]) for i, v in enumerate(views)}, tuple(notes))


def views_from_poses(poses: Sequence, intrinsics, images_by_evidence_id: Dict[str, np.ndarray],
                     depth_by_evidence_id: Optional[Dict[str, np.ndarray]] = None) -> List[SurfaceView]:
    """Build ``SurfaceView``s from backend camera poses (``ReconstructedCameraPose``) + decoded photos. Poses without a
    photo are skipped, never guessed."""
    from reconstruction.calibration.camera import camera_from_pose

    out = []
    for p in poses:
        img = images_by_evidence_id.get(p.evidence_id)
        if img is not None:
            out.append(SurfaceView(p.evidence_id, camera_from_pose(intrinsics, p), img,
                                   (depth_by_evidence_id or {}).get(p.evidence_id)))
    return out


def attach_appearance(entity, appearance: SurfaceAppearance) -> None:
    """Record the appearance reference on a WorldIR entity (``custom_properties["appearance"]``)."""
    entity.custom_properties["appearance"] = appearance.to_property()
