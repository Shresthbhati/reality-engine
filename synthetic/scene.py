"""Planar synthetic scenes with a vectorised ray-caster.

A scene is a list of RECTANGULAR quads (``origin + s*u + r*v``, ``u`` perpendicular to ``v``) carrying a semantic
label, a colour and the id of the element they belong to, plus a ``truth`` dictionary the builders fill in
(storeys, rooms, openings, stairs, ramps, clutter). Walls with doors and windows are decomposed into the rectangles
around each opening, so an opening is a real hole the camera sees through, not a texture. Windows are filled with a
``glass`` quad: it renders in RGB but a depth sensor returns no valid range through it (see ``rgbd.SensorModel``).

World frame: metres, Z up, right-handed. Quads are double-sided. Everything is deterministic (no randomness here).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

Vec = Tuple[float, float, float]


def _v(a) -> np.ndarray:
    return np.asarray(a, dtype=float)


@dataclass(frozen=True)
class Quad:
    origin: Vec
    u: Vec
    v: Vec
    label: str
    color: Tuple[int, int, int] = (180, 180, 180)
    ref: str = ""

    def __post_init__(self):
        u, v = _v(self.u), _v(self.v)
        if np.linalg.norm(u) < 1e-9 or np.linalg.norm(v) < 1e-9:
            raise ValueError(f"degenerate quad {self.ref or self.label}")
        if abs(float(u @ v)) > 1e-6 * np.linalg.norm(u) * np.linalg.norm(v):
            raise ValueError(f"quad {self.ref or self.label} is not rectangular (u.v != 0)")

    @property
    def normal(self) -> np.ndarray:
        n = np.cross(_v(self.u), _v(self.v))
        return n / np.linalg.norm(n)

    @property
    def area(self) -> float:
        return float(np.linalg.norm(np.cross(_v(self.u), _v(self.v))))


@dataclass(frozen=True)
class Opening:
    """A door or window in a wall: ``offset`` along the wall from its start, ``bottom``/``top`` above the wall base."""

    kind: str  # "door" | "window"
    offset: float
    width: float
    bottom: float
    top: float


@dataclass
class Scene:
    quads: List[Quad] = field(default_factory=list)
    truth: Dict[str, list] = field(
        default_factory=lambda: {"storeys": [], "rooms": [], "openings": [], "stairs": [], "ramps": [], "clutter": []}
    )
    _cache: Optional[tuple] = field(default=None, repr=False, compare=False)

    def add(self, quad: Quad) -> None:
        self.quads.append(quad)
        self._cache = None

    def arrays(self):
        if self._cache is None or len(self._cache[0]) != len(self.quads):
            o = np.array([q.origin for q in self.quads], dtype=float)
            u = np.array([q.u for q in self.quads], dtype=float)
            v = np.array([q.v for q in self.quads], dtype=float)
            n = np.cross(u, v)
            self._cache = (o, u, v, n, (u * u).sum(1), (v * v).sum(1))
        return self._cache

    def refs(self) -> List[str]:
        return sorted({q.ref for q in self.quads})


def raycast(scene: Scene, origin: Vec, dirs: np.ndarray, tmin: float = 1e-4):
    """Nearest hit along ``origin + t*dirs[i]``. Returns ``(t, quad_index)``; ``t = inf`` and index ``-1`` for a miss.

    ``dirs`` need not be unit length: with a camera ray ``(x, y, 1)`` rotated into the world, ``t`` IS the depth along
    the optical axis, which is what a depth map stores."""
    o = _v(origin)
    d = np.asarray(dirs, dtype=float)
    best_t = np.full(len(d), np.inf)
    best_i = np.full(len(d), -1, dtype=np.int32)
    qo, qu, qv, qn, uu, vv = scene.arrays()
    for i in range(len(qo)):
        denom = d @ qn[i]
        ok = np.abs(denom) > 1e-12
        t = np.full(len(d), np.inf)
        t[ok] = ((qo[i] - o) @ qn[i]) / denom[ok]
        cand = np.nonzero((t > tmin) & (t < best_t))[0]
        if cand.size == 0:
            continue
        rel = o + t[cand, None] * d[cand] - qo[i]
        s = (rel @ qu[i]) / uu[i]
        r = (rel @ qv[i]) / vv[i]
        hit = cand[(s >= 0) & (s <= 1) & (r >= 0) & (r <= 1)]
        best_t[hit] = t[hit]
        best_i[hit] = i
    return best_t, best_i


def nearest_quad(scene: Scene, points: np.ndarray):
    """``(distance, quad_index)`` of the nearest quad for each point: the ground-truth surface error of a reconstructed
    point, and which element (label/ref) it belongs to."""
    qo, qu, qv, qn, uu, vv = scene.arrays()
    pts = np.asarray(points, dtype=float)
    best = np.full(len(pts), np.inf)
    arg = np.full(len(pts), -1, dtype=np.int32)
    for i in range(len(qo)):
        rel = pts - qo[i]
        s = np.clip((rel @ qu[i]) / uu[i], 0.0, 1.0)
        r = np.clip((rel @ qv[i]) / vv[i], 0.0, 1.0)
        closest = qo[i] + s[:, None] * qu[i] + r[:, None] * qv[i]
        dist = np.linalg.norm(pts - closest, axis=1)
        better = dist < best
        best[better] = dist[better]
        arg[better] = i
    return best, arg


def distance_to_scene(scene: Scene, points: np.ndarray) -> np.ndarray:
    """Distance from each point to the nearest quad."""
    return nearest_quad(scene, points)[0]


# ----------------------------------------------------------------------------------------------- builders

WALL_COLOR = (205, 198, 185)
FLOOR_COLOR = (150, 120, 95)
CEILING_COLOR = (235, 235, 230)
CLUTTER_COLOR = (110, 80, 60)
GLASS_COLOR = (170, 205, 230)
STAIR_COLOR = (125, 125, 135)
RAMP_COLOR = (140, 160, 120)


def add_box(scene: Scene, lo: Vec, hi: Vec, *, label: str, color, ref: str) -> None:
    """A closed axis-aligned box (six quads): furniture, a pillar, a stair block."""
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    dx, dy, dz = (x1 - x0, 0, 0), (0, y1 - y0, 0), (0, 0, z1 - z0)
    faces = [
        ((x0, y0, z0), dx, dy),  # bottom
        ((x0, y0, z1), dx, dy),  # top
        ((x0, y0, z0), dx, dz),  # y0 side
        ((x0, y1, z0), dx, dz),  # y1 side
        ((x0, y0, z0), dy, dz),  # x0 side
        ((x1, y0, z0), dy, dz),  # x1 side
    ]
    for o, u, v in faces:
        scene.add(Quad(o, u, v, label, color, ref))


def add_slab(scene: Scene, x0: float, x1: float, y0: float, y1: float, z: float, *, label: str, color, ref: str,
             hole: Optional[Tuple[float, float, float, float]] = None) -> None:
    """A horizontal sheet at height ``z``, optionally with one rectangular hole (``hx0, hx1, hy0, hy1``): the stairwell
    through a floor/ceiling, as four quads around the hole."""
    if hole is None:
        scene.add(Quad((x0, y0, z), (x1 - x0, 0, 0), (0, y1 - y0, 0), label, color, ref))
        return
    hx0, hx1, hy0, hy1 = hole
    pieces = [(x0, hx0, y0, y1), (hx1, x1, y0, y1), (hx0, hx1, y0, hy0), (hx0, hx1, hy1, y1)]
    for a0, a1, b0, b1 in pieces:
        if a1 - a0 > 1e-9 and b1 - b0 > 1e-9:
            scene.add(Quad((a0, b0, z), (a1 - a0, 0, 0), (0, b1 - b0, 0), label, color, ref))


def add_wall(scene: Scene, p0: Tuple[float, float], p1: Tuple[float, float], z0: float, z1: float, *, ref: str,
             color=WALL_COLOR, openings: Sequence[Opening] = ()) -> None:
    """A vertical wall from ``p0`` to ``p1`` (plan coordinates) between heights ``z0`` and ``z1``, with openings cut
    out. A door reaches the floor; a window has a sill and a head, and is filled with glass."""
    d = np.array([p1[0] - p0[0], p1[1] - p0[1], 0.0])
    length = float(np.linalg.norm(d))
    d /= length
    base = np.array([p0[0], p0[1], 0.0])
    up = np.array([0.0, 0.0, 1.0])

    def piece(a: float, b: float, h0: float, h1: float, label="wall", col=color):
        if b - a > 1e-9 and h1 - h0 > 1e-9:
            scene.add(Quad(tuple(base + d * a + up * h0), tuple(d * (b - a)), tuple(up * (h1 - h0)), label, col, ref))

    cursor = 0.0
    for op in sorted(openings, key=lambda o: o.offset):
        if op.offset < cursor - 1e-9 or op.offset + op.width > length + 1e-9:
            raise ValueError(f"opening {op} overlaps another or leaves wall {ref} (length {length:.3f})")
        piece(cursor, op.offset, z0, z1)
        piece(op.offset, op.offset + op.width, z0, z0 + op.bottom)           # below a window (none for a door)
        piece(op.offset, op.offset + op.width, z0 + op.top, z1)              # above the opening
        if op.kind == "window":
            piece(op.offset, op.offset + op.width, z0 + op.bottom, z0 + op.top, label="glass", col=GLASS_COLOR)
        cursor = op.offset + op.width
    piece(cursor, length, z0, z1)


def add_stairs(scene: Scene, start: Vec, direction: Tuple[float, float], *, width: float, run: float, rise: float,
               steps: int, ref: str, color=STAIR_COLOR) -> None:
    """A straight flight: ``steps`` treads (horizontal) and risers (vertical). ``start`` is the foot of the first
    riser; the top tread is at ``start.z + rise``. Total run/rise are split evenly."""
    dx, dy = direction
    n = np.hypot(dx, dy)
    fwd = np.array([dx / n, dy / n, 0.0])
    side = np.array([-fwd[1], fwd[0], 0.0])
    up = np.array([0.0, 0.0, 1.0])
    s0 = np.array(start, dtype=float) - side * (width / 2.0)
    tread, riser = run / steps, rise / steps
    for i in range(steps):
        foot = s0 + fwd * (i * tread) + up * (i * riser)
        scene.add(Quad(tuple(foot), tuple(side * width), tuple(up * riser), "stair_riser", color, ref))
        scene.add(Quad(tuple(foot + up * riser), tuple(fwd * tread), tuple(side * width), "stair_tread", color, ref))


def add_ramp(scene: Scene, start: Vec, end: Vec, *, width: float, ref: str, color=RAMP_COLOR) -> None:
    """A tilted rectangular ramp from ``start`` (low) to ``end`` (high); width is horizontal and perpendicular."""
    s, e = _v(start), _v(end)
    run = e - s
    horiz = np.array([run[0], run[1], 0.0])
    horiz /= np.linalg.norm(horiz)
    side = np.array([-horiz[1], horiz[0], 0.0])
    scene.add(Quad(tuple(s - side * (width / 2.0)), tuple(run), tuple(side * width), "ramp", color, ref))
