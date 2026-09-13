#!/usr/bin/env python3
"""Deterministic synthetic room-capture renderer (Prompt-1 vertical slice).

Renders REAL perspective images of the hand-computable 2.5 x 2 x 2.25 m
room (floor/ceiling/walls with doorway, table, chair, lamp) from known
poses, for reconstruction with REAL SfM:

    python scripts/render_room_dataset.py datasets/room_capture 12

Method (why COLMAP can actually reconstruct these images):

1.  The scene is rasterized into per-pixel WORLD-POSITION buffers --
    each covered pixel stores the exact 3D world point it sees (ray /
    plane intersection, z-buffered). Geometry is exact by construction.

2.  Pixels are shaded from their world position with multi-octave value
    noise: texture with natural statistics -- energy at every scale SIFT
    samples (centimeters to meters), uncorrelated between distant
    positions, identical across viewpoints for the same world point.
    Three measured failure modes motivated this (see
    docs/REAL_CAPTURE_VERTICAL_SLICE_AUDIT.md):
      - flat color regions: almost no features at all;
      - a pure checkerboard: every corner is descriptor-identical, ~50%
        of matches were cross-room confusions (verified with an
        independent 8-point RANSAC) and the reconstruction converged to
        a wrong-but-self-consistent 0.7 px shape;
      - cell-level random colors: too little energy at SIFT's scales.
    The noise is also brightness-normalization-friendly: structure
    comes from 2D noise GRADIENTS that differ at every position, not
    from absolute color.

3.  Ground truth (poses / intrinsics / measured baseline) goes to
    manifest.json for benchmarking; metric scale is anchored by the
    known camera baseline.

Deterministic: shading is a pure function of world position and fixed
seeds; identical invocations produce identical bytes.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Hand-computable room (same family as tests/test_room_inference.py).
ROOM_MIN = (0.0, 0.0, 0.0)
ROOM_MAX = (2.5, 2.0, 2.25)
DOORWAY = (0.5, 1.5, 0.0)      # x0, x1 on the z=0 front wall
DOORWAY_TOP = 1.8

IMAGE_W, IMAGE_H = 1280, 960
FX = FY = 1160.0
CX, CY = 640.0, 480.0

# Surface ids and their base albedos (modulated by world-position noise).
SURFACE_BASE = {
    0: (150, 150, 155),   # floor
    1: (230, 228, 222),   # ceiling
    2: (185, 178, 168),   # walls
    3: (96, 72, 52),      # door leaf
    4: (128, 96, 60),     # table
    5: (110, 82, 52),     # chair
    6: (235, 220, 140),   # lamp
}
FLOOR, CEILING, WALL, DOOR, TABLE, CHAIR, LAMP = range(7)

# Value-noise octaves: spatial frequencies in cycles per meter + seeds.
# 2/m mottles at ~0.5 m, 8/m at ~12 cm, 32/m at ~3 cm -- covering the
# scales SIFT descriptors integrate over at 1-3 m viewing distance.
NOISE_OCTAVES = ((2.0, 11), (8.0, 23), (32.0, 47))


def camera_trajectory(n: int):
    """Deterministic capture trajectory, two pitched-down rings.

    Measured failure mode of the first trajectory (3 rings, aimed at the
    room center): cameras nearly touched walls with a ~7 deg downward
    pitch, so frames contained a single oblique wall and NO floor, and
    adjacent cameras differed by ~110 deg yaw -- neighboring images
    shared almost nothing, so SfM initialization had no usable pair.
    Real indoor capture pitches the camera down 30-45 deg so every frame
    contains floor + walls + furniture, and steps yaw by ~30 deg for
    heavy overlap. This generator does exactly that:
      ring A: 12 cameras, h=1.4, r=0.85, yaw step 30 deg
      ring B: 12 cameras, h=0.9, r=0.85 (SAME radius as A -- standard
              multi-height capture stacks rings at equal radius so every
              ring-B frame sees the same floor/wall content ring A
              triangulated; a smaller inner radius looks mostly at floor
              content ring A never observed and fails to register),
              yaw step 30 deg, offset 15 deg
    both looking at (1.25, 0.5, 1.125) -- below center, producing the
    downward pitch. Total n = 24 (matches the spec's 20-50 images).

    Measured failure of different-look-at rings: ring B aiming at a
    farther target changes its PITCH as well as its height, so ring-B
    frames see ring-A content rescaled/rotated and ZERO A<->B feature
    pairs verify (measured: 0/568 verified pairs). Real multi-height
    capture keeps the view axis PARALLEL between rings (same pitch,
    same yaw grid, height-offset only) so every ring-B frame is a pure
    vertical shift of its ring-A twin. Ring B therefore looks at a
    target shifted down by exactly the height difference.
    """
    n_a = max(4, (n + 1) // 2)
    n_b = max(4, n - n_a)
    target = np.array([1.25, 0.5, 1.125])
    cams = []
    idx = 0
    for h, radius, count, offset in ((1.4, 0.85, n_a, 0.0),
                                     (0.9, 0.85, n_b, math.pi / 12)):
        for k in range(count):
            ang = 2.0 * math.pi * k / count + offset
            pos = np.array([
                1.25 + radius * math.cos(ang),
                h,
                1.125 + radius * math.sin(ang),
            ])
            # Parallel view axes across rings: drop the look-at target by
            # exactly the height difference so ring B's view direction is
            # identical to ring A's (pure vertical translation).
            look = target - np.array([0.0, 1.4 - h, 0.0])
            cams.append({"index": idx, "position": pos, "look_at": look})
            idx += 1
    # Wall-facing stations: square-on shots, one per wall, from a
    # DISTINCT position 0.9 m from its wall, pitched DOWN at the wall
    # base like the rings. Measured failure modes: (a) no wall shots ->
    # grazing-only wall views, never a closable room; (b) four shots
    # from one shared position -> zero baseline, degenerate duplicates;
    # (c) level (pitch-0) shots at 0.45 m or 0.9 m -> 0/4 registered:
    # a level frame is filled by mid-wall content the pitched rings
    # never see, exactly the ring-B rescaling failure. What worked for
    # ring B (parallel view axes, ~40 deg downward pitch shared with
    # the rings) is applied here: the wall station sees the wall base
    # + floor band the rings actually triangulated.
    wall_stations = (
        (np.array([0.90, 1.4, 1.125]), np.array([0.0, 0.3, 1.125])),    # x=0
        (np.array([1.60, 1.4, 1.125]), np.array([2.5, 0.3, 1.125])),    # x=2.5
        (np.array([1.25, 1.4, 0.90]),  np.array([1.25, 0.3, 0.0])),     # z=0
        (np.array([1.25, 1.4, 1.35]),  np.array([1.25, 0.3, 2.25])),    # z=2.25
    )
    for pos, wt in wall_stations:
        cams.append({"index": idx, "position": pos, "look_at": wt})
        idx += 1
    return cams


def look_at_matrix(pos: np.ndarray, target: np.ndarray):
    """World->camera rotation R with the camera looking along its +Z axis,
    Y down (standard CV convention, matching COLMAP's camera frame)."""
    forward = target - pos
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    R = np.stack([right, -up, forward], axis=0)  # rows: x_cam, y_cam, z_cam
    return R


def project_points(points_w: np.ndarray, R: np.ndarray, t: np.ndarray):
    """Project world points to pixels; returns (uv, depths, valid).

    Standard pinhole convention: x right, y down (u = +fx*x/z,
    v = +fy*y/z).
    """
    pc = (R @ points_w.T).T + t
    valid = pc[:, 2] > 1e-6
    uv = np.zeros((len(pc), 2))
    uv[valid, 0] = FX * pc[valid, 0] / pc[valid, 2] + CX
    uv[valid, 1] = FY * pc[valid, 1] / pc[valid, 2] + CY
    return uv, pc[:, 2], valid


# ---------------------------------------------------------------- noise --
def _hash01(ix: np.ndarray, iy: np.ndarray, iz: np.ndarray,
            seed: int) -> np.ndarray:
    """Deterministic pseudo-random [0,1) from integer lattice coords."""
    h = (ix * np.int64(374761393) + iy * np.int64(668265263)
         + iz * np.int64(1442695041) + np.int64(seed) * np.int64(1274126177))
    h &= np.int64(0xFFFFFFFF)
    h = ((h ^ (h >> np.int64(13))) * np.int64(1274126177)) & np.int64(0xFFFFFFFF)
    return ((h ^ (h >> np.int64(16))) & np.int64(0xFFFFFF)).astype(
        np.float64) / float(0x1000000)


def _value_noise(wx, wy, wz, freq: float, seed: int) -> np.ndarray:
    """Trilinear-interpolated value noise; freq in cycles per meter."""
    xs, ys, zs = wx * freq, wy * freq, wz * freq
    x0 = np.floor(xs).astype(np.int64)
    y0 = np.floor(ys).astype(np.int64)
    z0 = np.floor(zs).astype(np.int64)
    fx, fy, fz = xs - x0, ys - y0, zs - z0
    fx = fx * fx * (3 - 2 * fx)
    fy = fy * fy * (3 - 2 * fy)
    fz = fz * fz * (3 - 2 * fz)

    def c(dx, dy, dz):
        return _hash01(x0 + dx, y0 + dy, z0 + dz, seed)

    n00 = c(0, 0, 0) * (1 - fx) + c(1, 0, 0) * fx
    n10 = c(0, 1, 0) * (1 - fx) + c(1, 1, 0) * fx
    lower = n00 * (1 - fy) + n10 * fy
    n01 = c(0, 0, 1) * (1 - fx) + c(1, 0, 1) * fx
    n11 = c(0, 1, 1) * (1 - fx) + c(1, 1, 1) * fx
    upper = n01 * (1 - fy) + n11 * fy
    return lower * (1 - fz) + upper * fz


def _shade(wpos: np.ndarray, base_rgb: np.ndarray) -> np.ndarray:
    """Base color modulated by fractal world-anchored value noise."""
    lum = np.ones(len(wpos))
    for freq, seed in NOISE_OCTAVES:
        n = _value_noise(wpos[:, 0], wpos[:, 1], wpos[:, 2], freq, seed)
        lum *= (0.70 + 0.60 * n)          # ~+/-30% per octave
    return np.clip(base_rgb[None, :] * lum[:, None], 0, 255)


# ----------------------------------------------------------- rasterizer --
def render_image(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Software-render the room from this pose via world-position buffers.

    Quads rasterize into (zbuf, pbuf, sbuf); shading is a second pass
    over covered pixels, from each pixel's stored world position.
    """
    pbuf = np.zeros((IMAGE_H, IMAGE_W, 3))
    zbuf = np.full((IMAGE_H, IMAGE_W), np.inf)
    sbuf = np.full((IMAGE_H, IMAGE_W), -1, dtype=np.int32)

    cam_x, cam_y, cam_z = R[0], R[1], R[2]
    origin = -R.T @ t

    def draw_quad(p3d, surf_id: int) -> None:
        p3d = np.asarray(p3d, dtype=float)
        uv, _depth, valid = project_points(p3d, R, t)
        if not valid.all():
            return
        x0 = int(max(0, np.floor(uv[:, 0].min())))
        x1 = int(min(IMAGE_W - 1, np.ceil(uv[:, 0].max())))
        y0 = int(max(0, np.floor(uv[:, 1].min())))
        y1 = int(min(IMAGE_H - 1, np.ceil(uv[:, 1].max())))
        if x1 <= x0 or y1 <= y0:
            return

        # Point-in-quad: edge cross-product signs must agree (convex).
        ax, ay = uv[:, 0], uv[:, 1]
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1) + 0.5,
                             np.arange(y0, y1 + 1) + 0.5)
        signs = np.empty((4,) + xs.shape)
        for i in range(4):
            j = (i + 1) % 4
            ex, ey = ax[j] - ax[i], ay[j] - ay[i]
            signs[i] = ex * (ys - ay[i]) - ey * (xs - ax[i])
        inside = np.all(signs >= 0, axis=0) | np.all(signs <= 0, axis=0)
        if not inside.any():
            return

        # Exact world position: intersect the pixel ray with the quad's
        # plane (all quads are planar). Ray param tau == camera depth.
        a, b, c = p3d[0], p3d[1], p3d[2]
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-12:
            return
        n = n / nn
        d = float(n @ a)
        dx = (xs - CX) / FX
        dy = (ys - CY) / FY
        dw = (dx[..., None] * cam_x + dy[..., None] * cam_y + cam_z)
        denom = dw @ n
        safe = np.abs(denom) > 1e-12
        tau = np.where(safe, (d - float(n @ origin)) / np.where(safe, denom, 1.0),
                       np.inf)

        zreg = zbuf[y0:y1 + 1, x0:x1 + 1]
        preg = pbuf[y0:y1 + 1, x0:x1 + 1]
        sreg = sbuf[y0:y1 + 1, x0:x1 + 1]
        upd = inside & safe & (tau > 1e-6) & (tau < zreg)
        if not upd.any():
            return
        zreg[upd] = tau[upd]
        sreg[upd] = surf_id
        preg[upd] = origin + tau[upd, None] * dw[upd]

    def draw_box(lo, hi, surf_id: int) -> None:
        x0, y0, z0 = lo
        x1, y1, z1 = hi
        faces = [
            [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)],
            [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],
            [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],
            [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)],
            [(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)],
            [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)],
        ]
        for face in faces:
            draw_quad(face, surf_id)

    mx0, my0, mz0 = ROOM_MIN
    mx1, my1, mz1 = ROOM_MAX
    draw_quad([(mx0, my0, mz0), (mx1, my0, mz0),
               (mx1, my0, mz1), (mx0, my0, mz1)], FLOOR)
    draw_quad([(mx0, my1, mz0), (mx1, my1, mz0),
               (mx1, my1, mz1), (mx0, my1, mz1)], CEILING)
    # x = 0 and x = 2.5 walls (full).
    draw_quad([(mx0, my0, mz0), (mx0, my1, mz0),
               (mx0, my1, mz1), (mx0, my0, mz1)], WALL)
    draw_quad([(mx1, my0, mz0), (mx1, my1, mz0),
               (mx1, my1, mz1), (mx1, my0, mz1)], WALL)
    # z = 2.25 back wall (full) and z = 0 front wall around the doorway.
    draw_quad([(mx0, my0, mz1), (mx1, my0, mz1),
               (mx1, my1, mz1), (mx0, my1, mz1)], WALL)
    dx0, dx1, _ = DOORWAY
    draw_quad([(mx0, my0, 0.0), (dx0, my0, 0.0),
               (dx0, my1, 0.0), (mx0, my1, 0.0)], WALL)          # left of door
    draw_quad([(dx1, my0, 0.0), (mx1, my0, 0.0),
               (mx1, my1, 0.0), (dx1, my1, 0.0)], WALL)          # right of door
    draw_quad([(dx0, DOORWAY_TOP, 0.0), (dx1, DOORWAY_TOP, 0.0),
               (dx1, my1, 0.0), (dx0, my1, 0.0)], WALL)          # above door
    draw_quad([(dx0, my0, 0.0), (dx1, my0, 0.0),
               (dx1, DOORWAY_TOP, 0.0), (dx0, DOORWAY_TOP, 0.0)], DOOR)

    # Furniture (non-planar structure; breaks two-view planar degeneracy).
    draw_box((0.7, 0.68, 0.8), (1.7, 0.75, 1.6), TABLE)            # top
    for lx in (0.75, 1.55):
        for lz in (0.85, 1.45):
            draw_box((lx, 0.0, lz), (lx + 0.08, 0.68, lz + 0.08), TABLE)
    draw_box((1.95, 0.42, 0.95), (2.35, 0.47, 1.35), CHAIR)        # seat
    draw_box((1.95, 0.0, 0.95), (2.0, 0.42, 1.0), CHAIR)
    draw_box((2.3, 0.0, 0.95), (2.35, 0.42, 1.0), CHAIR)
    draw_box((1.95, 0.0, 1.3), (2.0, 0.42, 1.35), CHAIR)
    draw_box((2.3, 0.0, 1.3), (2.35, 0.42, 1.35), CHAIR)
    draw_box((2.3, 0.47, 0.95), (2.35, 0.95, 1.35), CHAIR)         # backrest
    draw_box((0.55, 1.55, 1.5), (0.75, 1.75, 1.7), LAMP)

    # Architectural clutter: the measured failure mode at this stage is
    # PLANAR DEGENERACY -- essential-matrix recovery diverges when a
    # pair's inliers lie (nearly) on one plane, and bare walls/ceiling
    # dominate every view. Real rooms initialize because walls carry
    # non-planar features (skirting, rails, window frames). These thin
    # boxes put off-plane structure on every wall.
    def bands(axis_fixed: float, along: str, lo: float, hi: float) -> None:
        """Skirting + picture-rail bands along one wall (protrude 2 cm)."""
        p = 0.02
        if along == "x":  # wall at z = axis_fixed, band runs along x
            draw_box((lo, 0.0, axis_fixed), (hi, 0.1, axis_fixed + p), WALL)
            draw_box((lo, 1.55, axis_fixed), (hi, 1.62, axis_fixed + p), WALL)
        else:             # wall at x = axis_fixed, band runs along z
            draw_box((axis_fixed, 0.0, lo), (axis_fixed + p, 0.1, hi), WALL)
            draw_box((axis_fixed, 1.55, lo), (axis_fixed + p, 1.62, hi), WALL)
    bands(0.0, "x", 0.0, 2.5)       # z=0 wall (door wall; bands overlap door -- fine, door is recessed 0)
    bands(2.25 - 0.02, "x", 0.0, 2.5)   # z=2.25 wall (protrude inward: -0.02 handled by fixed+p up to 2.25)
    bands(0.0, "z", 0.0, 2.25)      # x=0 wall
    bands(2.5 - 0.02, "z", 0.0, 2.25)   # x=2.5 wall
    # Window frame on the x=2.5 wall: a protruding rectangular frame.
    draw_box((2.46, 0.9, 0.7), (2.5, 1.0, 1.8), WALL)   # sill
    draw_box((2.46, 1.7, 0.7), (2.5, 1.8, 1.8), WALL)   # head
    draw_box((2.46, 0.9, 0.7), (2.5, 1.8, 0.8), WALL)   # jamb
    draw_box((2.46, 0.9, 1.7), (2.5, 1.8, 1.8), WALL)   # jamb
    draw_box((2.46, 0.9, 1.15), (2.5, 1.8, 1.35), WALL) # mullion
    # Small clutter: book stack on the table, floor box (non-planar mass).
    draw_box((0.9, 0.75, 1.0), (1.2, 0.80, 1.35), CHAIR)   # books
    draw_box((0.9, 0.80, 1.05), (1.15, 0.85, 1.30), TABLE)
    draw_box((0.2, 0.0, 1.9), (0.5, 0.35, 2.2), TABLE)     # crate

    # Shade from world positions.
    img = np.zeros((IMAGE_H, IMAGE_W, 3))
    covered = sbuf >= 0
    if covered.any():
        base_table = np.array([SURFACE_BASE[i] for i in range(len(SURFACE_BASE))],
                              dtype=float)
        wpos = pbuf[covered]
        sid = sbuf[covered]
        img[covered] = _shade(wpos, base_table[sid])
    return img.astype(np.uint8)


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "datasets/room_capture")
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    images_dir = out / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    cams = camera_trajectory(n)
    manifest = {
        "dataset": "synthetic-room-capture",
        "room_min_m": list(ROOM_MIN),
        "room_max_m": list(ROOM_MAX),
        "image_size": [IMAGE_W, IMAGE_H],
        "intrinsics_px": {"fx": FX, "fy": FY, "cx": CX, "cy": CY},
        "scale_reference": {
            "method": "known_camera_baseline",
            "description": (
                "operator-measured distance between adjacent camera "
                "stations; positions below are the measured tripod "
                "coordinates in meters"
            ),
            "metric": True,
        },
        "images": [],
    }
    # Operator-measured baselines: the measured distance between each
    # consecutive pair of camera stations (what a tape measure gives you
    # on site). Every pair is recorded so scale anchoring works even when
    # one station fails SfM registration.
    measured_baselines = []
    for i, cam in enumerate(cams):
        R = look_at_matrix(cam["position"], cam["look_at"])
        t = -R @ cam["position"]
        img = render_image(R, t)
        name = f"IMG_{i:04d}.jpg"
        Image.fromarray(img).save(images_dir / name, quality=95)
        manifest["images"].append({
            "file": name,
            "position_m": [round(v, 6) for v in cam["position"]],
            "look_at_m": [round(v, 6) for v in cam["look_at"]],
        })
        if i > 0:
            prev = cams[i - 1]["position"]
            measured_baselines.append({
                "evidence_id_a": f"IMG_{i - 1:04d}",
                "evidence_id_b": f"IMG_{i:04d}",
                "distance_m": round(float(np.linalg.norm(cam["position"] - prev)), 6),
            })

    manifest["measured_baselines"] = measured_baselines
    manifest["measured_baseline_m"] = (
        measured_baselines[-1]["distance_m"] if measured_baselines else None
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Rendered {len(cams)} images -> {images_dir}")
    print(f"Room {ROOM_MIN}..{ROOM_MAX} m; baseline {manifest['measured_baseline_m']} m")


if __name__ == "__main__":
    main()
