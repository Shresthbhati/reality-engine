"""Synthetic evidence for the spatial compiler benchmark and tests (NEVER imported by production code).

A `GTScene` is ground truth in the plan view (walls as segments, openings along host walls, rooms as wall
cycles, storeys, stairs). `simulate_observations` turns it into per-image `StructuralObservation`s with the
failure modes the compiler must survive: Gaussian end-point noise scaled by camera range (the same
`sigma_for` the production lifting uses), partial / occluded visibility that truncates extents, whole walls
missed by every view, gross outliers, and an image with a systematic bias (conflicting measurements).

Everything is seeded; the same seed yields the same observations. These are simulated observations over a
known truth -- they establish what the compiler does under stated noise, not how perception performs on real
photographs (see the benchmark report's "limits" section).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from reconstruction.spatial.model import Kind, StructuralObservation
from world_ir.observations import DepthSource, Visibility, sigma_for

Pt = tuple[float, float]


@dataclass
class GTWall:
    id: str
    a: Pt
    b: Pt
    storey: str = "L0"


@dataclass
class GTOpening:
    id: str
    host: str
    t0: float            # metres along the host from its `a`
    t1: float
    kind: Kind = Kind.DOOR
    storey: str = "L0"


@dataclass
class GTScene:
    name: str
    walls: list[GTWall]
    openings: list[GTOpening] = field(default_factory=list)
    rooms: dict[str, list[str]] = field(default_factory=dict)      # room id -> wall ids (any order)
    storeys: dict[str, float] = field(default_factory=lambda: {"L0": 0.0})   # storey -> floor z
    stairs: list[tuple[str, Pt, Pt, str, str]] = field(default_factory=list)  # id, base, top, lower, upper
    height: float = 2.7

    def wall(self, wid: str) -> GTWall:
        return next(w for w in self.walls if w.id == wid)


def _ring(prefix: str, pts: list[Pt], storey: str = "L0") -> list[GTWall]:
    return [GTWall(f"{prefix}{i}", pts[i], pts[(i + 1) % len(pts)], storey) for i in range(len(pts))]


def rect_room(w=5.0, d=4.0) -> GTScene:
    ws = _ring("w", [(0, 0), (w, 0), (w, d), (0, d)])
    return GTScene("rectangular_room", ws,
                   [GTOpening("door0", "w0", 1.0, 1.9), GTOpening("win0", "w2", 1.5, 3.0, Kind.WINDOW)],
                   {"R0": [x.id for x in ws]})


def l_shape() -> GTScene:
    ws = _ring("w", [(0, 0), (6, 0), (6, 3), (3, 3), (3, 6), (0, 6)])
    return GTScene("l_shaped_room", ws, [GTOpening("door0", "w0", 2.0, 2.9)], {"R0": [x.id for x in ws]})


def irregular_non_manhattan() -> GTScene:
    """Seven-sided room with no right angles."""
    pts = [(0.0, 0.0), (5.2, -0.8), (7.0, 2.4), (5.6, 5.3), (2.9, 6.1), (0.4, 4.2), (-0.9, 1.9)]
    ws = _ring("w", pts)
    return GTScene("irregular_non_manhattan_room", ws,
                   [GTOpening("door0", "w1", 0.8, 1.7), GTOpening("win0", "w3", 0.5, 1.8, Kind.WINDOW)],
                   {"R0": [x.id for x in ws]})


def two_rooms_shared_wall() -> GTScene:
    ws = [GTWall("s0", (0, 0), (8, 0)), GTWall("e", (8, 0), (8, 4)), GTWall("n0", (8, 4), (0, 4)),
          GTWall("w", (0, 4), (0, 0)), GTWall("mid", (4, 0), (4, 4))]
    return GTScene("two_connected_rooms", ws,
                   [GTOpening("door_mid", "mid", 1.2, 2.1), GTOpening("door_out", "s0", 1.0, 1.9)],
                   {"RA": ["s0", "mid", "n0", "w"], "RB": ["s0", "e", "n0", "mid"]})


def corridor_with_repeated_rooms(n=5, room_w=3.0) -> GTScene:
    """Hotel-style repeated pattern: identical rooms every `room_w` m, so parallel partitions are look-alikes."""
    L = n * room_w
    ws = [GTWall("back", (0, 5), (L, 5)), GTWall("front", (0, 2), (L, 2)), GTWall("end_w", (0, 0), (0, 5)),
          GTWall("end_e", (L, 0), (L, 5)), GTWall("corr_s", (0, 0), (L, 0))]
    ws += [GTWall(f"p{i}", (i * room_w, 2), (i * room_w, 5)) for i in range(1, n)]
    ops = [GTOpening(f"door{i}", "front", i * room_w + 0.9, i * room_w + 1.8) for i in range(n)]
    rooms = {f"R{i}": ["back", "front", f"p{i}" if i else "end_w", f"p{i + 1}" if i < n - 1 else "end_e"]
             for i in range(n)}
    rooms["RC"] = ["front", "corr_s", "end_w", "end_e"]   # the corridor itself is also an enclosed space
    return GTScene("corridor_repeated_rooms", ws, ops, rooms)


def open_plan() -> GTScene:
    """Perimeter plus two free-standing partitions that do not close any further room."""
    ws = _ring("w", [(0, 0), (10, 0), (10, 6), (0, 6)])
    ws += [GTWall("part0", (3, 0), (3, 3)), GTWall("part1", (6.5, 6), (6.5, 3.2))]
    return GTScene("open_plan", ws, [GTOpening("win0", "w2", 2.0, 4.5, Kind.WINDOW)],
                   {"R0": [f"w{i}" for i in range(4)]})


def two_storeys_with_stairs() -> GTScene:
    lo = _ring("a", [(0, 0), (5, 0), (5, 4), (0, 4)], "L0")
    hi = _ring("b", [(0, 0), (5, 0), (5, 4), (0, 4)], "L1")
    return GTScene("two_storeys_stairs", lo + hi, [GTOpening("door0", "a0", 1, 1.9, Kind.DOOR, "L0")],
                   {"R0": [w.id for w in lo], "R1": [w.id for w in hi]}, {"L0": 0.0, "L1": 2.8},
                   [("stair0", (1.0, 3.0), (3.5, 3.0), "L0", "L1")])


def from_surveyed(path: Path) -> GTScene:
    """Scene from a surveyed `ground_truth.json` (benchmarks/measured_rooms); openings placed on their wall."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    ws = [GTWall(w["id"], tuple(w["start"]), tuple(w["end"])) for w in d["walls"]]
    ops = []
    for o in d["openings"]:
        host = next(w for w in ws if w.id == o["wall"])
        c = np.asarray(o["center"][:2], float)
        a, b = np.asarray(host.a), np.asarray(host.b)
        L = float(np.linalg.norm(b - a))
        tc = float(np.dot(c - a, (b - a) / L))
        ops.append(GTOpening(o["id"], host.id, max(0.0, tc - o["width_m"] / 2), min(L, tc + o["width_m"] / 2),
                             Kind.DOOR if o["kind"] == "door" else Kind.WINDOW))
    return GTScene(d["room"]["id"], ws, ops, {d["room"]["id"]: [w.id for w in ws]}, height=d["room"]["height_m"])


@dataclass
class SimConfig:
    n_images: int = 6
    p_see_wall: float = 0.85          # per image, per wall
    min_coverage: float = 0.55        # visible fraction of a wall in one image
    p_truncate: float = 0.5           # probability a visible stretch is cut by an occluder / image border
    outlier_rate: float = 0.0         # fraction of detections with a gross offset error
    biased_image: int | None = None  # index of an image with a systematic offset
    bias_m: float = 0.0
    drop_walls: tuple[str, ...] = ()  # walls NO view sees (missing observations)
    depth: DepthSource = DepthSource.METRIC
    label_confidence: float | None = 0.9
    noise_scale: float = 1.0          # multiplies the true noise but NOT the declared sigma (miscalibration probe)
    seed: int = 0


def simulate_observations(scene: GTScene, cfg: SimConfig) -> list[StructuralObservation]:
    rng = np.random.default_rng(cfg.seed)
    xs = [p[0] for w in scene.walls for p in (w.a, w.b)]
    ys = [p[1] for w in scene.walls for p in (w.a, w.b)]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    out: list[StructuralObservation] = []
    for im in range(cfg.n_images):
        cam = np.array([cx, cy]) + rng.normal(0, 0.8, 2)
        bias = np.array([cfg.bias_m, cfg.bias_m * 0.5]) if cfg.biased_image == im else np.zeros(2)
        for w in scene.walls:
            if w.id in cfg.drop_walls or rng.random() > cfg.p_see_wall:
                continue
            a, b = np.asarray(w.a, float), np.asarray(w.b, float)
            L = float(np.linalg.norm(b - a))
            cov = rng.uniform(cfg.min_coverage, 1.0)
            truncated = rng.random() < cfg.p_truncate
            t0 = rng.uniform(0, 1 - cov) if truncated else 0.0
            t1 = min(1.0, t0 + cov) if truncated else 1.0
            e0, e1 = t0 == 0.0, t1 == 1.0
            p0, p1 = a + (b - a) * t0, a + (b - a) * t1
            sig = sigma_for(cfg.depth, float(np.linalg.norm((p0 + p1) / 2 - cam)))
            n = np.array([-(b - a)[1], (b - a)[0]]) / L
            gross = n * rng.choice([-1, 1]) * rng.uniform(0.5, 1.5) if rng.random() < cfg.outlier_rate else 0.0
            q0 = p0 + rng.normal(0, sig * cfg.noise_scale, 2) + bias + gross
            q1 = p1 + rng.normal(0, sig * cfg.noise_scale, 2) + bias + gross
            vis = Visibility.FULL if (e0 and e1) else (Visibility.PARTIAL if cov > 0.7 else Visibility.OCCLUDED)
            out.append(StructuralObservation(
                f"img{im}#{w.id}", f"img{im}", Kind.WALL, tuple(q0), tuple(q1), sig,
                scene.storeys[w.storey] + rng.normal(0, 0.03), scene.storeys[w.storey] + scene.height + rng.normal(0, 0.03), vis, e0, e1, cfg.label_confidence, cfg.depth,
                storey=w.storey, producer="synthetic@bench"))
        for o in scene.openings:
            if rng.random() > cfg.p_see_wall:
                continue
            h = scene.wall(o.host)
            a, b = np.asarray(h.a, float), np.asarray(h.b, float)
            u = (b - a) / float(np.linalg.norm(b - a))
            p0, p1 = a + u * o.t0, a + u * o.t1
            sig = sigma_for(cfg.depth, float(np.linalg.norm((p0 + p1) / 2 - cam)))
            q0 = p0 + rng.normal(0, sig * cfg.noise_scale, 2) + bias
            q1 = p1 + rng.normal(0, sig * cfg.noise_scale, 2) + bias
            out.append(StructuralObservation(
                f"img{im}#{o.id}", f"img{im}", o.kind, tuple(q0), tuple(q1), sig, 0.0,
                2.1 if o.kind is Kind.DOOR else 1.2, Visibility.FULL, True, True, cfg.label_confidence, cfg.depth,
                storey=o.storey, producer="synthetic@bench"))
        for sid, base, top, lo, hi in scene.stairs:
            if rng.random() > cfg.p_see_wall:
                continue
            sig = sigma_for(cfg.depth, 3.0)
            q0, q1 = np.asarray(base) + rng.normal(0, sig, 2), np.asarray(top) + rng.normal(0, sig, 2)
            out.append(StructuralObservation(
                f"img{im}#{sid}", f"img{im}", Kind.STAIR, tuple(q0), tuple(q1), sig, scene.storeys[lo],
                scene.storeys[hi], Visibility.FULL, True, True, cfg.label_confidence, cfg.depth, storey=lo,
                producer="synthetic@bench"))
    return out


def all_scenes() -> list[GTScene]:
    return [rect_room(), l_shape(), irregular_non_manhattan(), two_rooms_shared_wall(),
            corridor_with_repeated_rooms(), open_plan(), two_storeys_with_stairs()]


@dataclass
class PoseSim:
    sigma_theta: float = 0.02         # rad, true prior pose error AND the sigma declared for it
    sigma_t: float = 0.10             # m
    declared_scale: float = 1.0       # declared sigma = true sigma * this (<1 = overconfident producer)
    bad_image: int | None = None   # one image whose prior pose is grossly wrong (but declared with the usual sigma)
    bad_t: float = 0.6
    seed: int = 100


def simulate_multiview(scene: GTScene, cfg: SimConfig, ps: PoseSim):
    """Observations in each IMAGE frame plus prior and true poses.

    Returns (local_observations, prior_poses, true_poses). True poses place each image frame in the world; the observations
    are the world-frame simulation mapped through the INVERSE true pose, so applying the true pose recovers the usual
    noisy plan-view observations, while applying the (erroneous) prior pose reproduces what a pose-agnostic pipeline sees.
    """
    from dataclasses import replace

    from reconstruction.spatial.multiview import ImagePose
    world_obs = simulate_observations(scene, cfg)
    rng = np.random.default_rng(ps.seed + cfg.seed)
    imgs = sorted({o.image_id for o in world_obs})
    true, prior = {}, {}
    for i, im in enumerate(imgs):
        th, tx, ty = float(rng.uniform(-math.pi, math.pi)), float(rng.normal(0, 3)), float(rng.normal(0, 3))
        true[im] = ImagePose(im, th, tx, ty)
        e_th, e_t = rng.normal(0, ps.sigma_theta), rng.normal(0, ps.sigma_t, 2)
        if ps.bad_image == i:
            e_t = e_t + np.array([ps.bad_t, ps.bad_t * 0.5])
        prior[im] = ImagePose(im, th + float(e_th), tx + float(e_t[0]), ty + float(e_t[1]),
                              ps.sigma_theta * ps.declared_scale, ps.sigma_t * ps.declared_scale)
    local = []
    for o in world_obs:
        T = true[o.image_id]
        c, s = math.cos(T.theta), math.sin(T.theta)

        def inv(p, c=c, s=s, T=T):
            dx, dy = p[0] - T.tx, p[1] - T.ty
            return (c * dx + s * dy, -s * dx + c * dy)

        local.append(replace(o, p0=inv(o.p0), p1=inv(o.p1), frame_id=o.image_id))
    return local, prior, true
