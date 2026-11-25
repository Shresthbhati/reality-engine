"""Coordinate-frame management: similarity transforms (scale + rotation + translation) between named frames.

Why this exists next to `world_ir/frame_graph.py`: that graph holds rigid transforms with provenance. The spatial
compiler additionally needs (a) SCALE (monocular / SfM frames are only defined up to scale), (b) an explicit
validity check on every transform, (c) a loop-consistency check, and (d) a rule that nothing is ever expressed in a
frame the registry cannot reach: `convert` raises instead of guessing.

A `Sim3` maps p_to = s * R p_from + t, R in SO(3). Invalid transforms (non-finite, s <= 0, non-orthonormal R)
are refused at construction -- there is no 'repair' path. `FrameRegistry` stores one transform per directed pair
with its provenance string; `convert` composes along the unique shortest path and reports the path used.
Every canonical geometry object must name its frame: `Frame3`-tagged points make that explicit.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


class FrameError(ValueError):
    pass


def _check_rotation(R: np.ndarray, tol: float) -> None:
    if R.shape != (3, 3) or not np.all(np.isfinite(R)):
        raise FrameError("rotation must be a finite 3x3 matrix")
    if np.abs(R.T @ R - np.eye(3)).max() > tol or abs(np.linalg.det(R) - 1.0) > tol:
        raise FrameError("rotation is not a proper orthonormal matrix (det +1)")


def rotation_z(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


@dataclass(frozen=True)
class Sim3:
    from_frame: str
    to_frame: str
    scale: float
    R: np.ndarray
    t: np.ndarray

    def __post_init__(self):
        if not self.from_frame or not self.to_frame:
            raise FrameError("both frames must be named")
        if not (math.isfinite(self.scale) and self.scale > 0):
            raise FrameError(f"scale must be finite and > 0, got {self.scale!r}")
        object.__setattr__(self, "R", np.asarray(self.R, float))
        object.__setattr__(self, "t", np.asarray(self.t, float).reshape(3))
        _check_rotation(self.R, 1e-8)
        if not np.all(np.isfinite(self.t)):
            raise FrameError("translation must be finite")

    @staticmethod
    def identity(frame: str) -> Sim3:
        return Sim3(frame, frame, 1.0, np.eye(3), np.zeros(3))

    @staticmethod
    def from_plan(from_frame: str, to_frame: str, theta: float, tx: float, ty: float, scale: float = 1.0, tz: float = 0.0) -> Sim3:
        """Plan-view pose (rotation about the vertical axis + horizontal translation)."""
        return Sim3(from_frame, to_frame, scale, rotation_z(theta), np.array([tx, ty, tz]))

    def apply(self, p) -> np.ndarray:
        p = np.asarray(p, float)
        return self.scale * (p @ self.R.T) + self.t

    def inverse(self) -> Sim3:
        Ri = self.R.T
        return Sim3(self.to_frame, self.from_frame, 1.0 / self.scale, Ri, -(Ri @ self.t) / self.scale)

    def then(self, other: Sim3) -> Sim3:
        """self: A->B, other: B->C  =>  A->C. Refuses a frame mismatch (no silent mixing)."""
        if self.to_frame != other.from_frame:
            raise FrameError(f"cannot compose {self.from_frame}->{self.to_frame} with {other.from_frame}->{other.to_frame}")
        R = other.R @ self.R
        u, _, vt = np.linalg.svd(R)          # re-orthonormalise: composition drift stays bounded over long chains
        R = u @ vt
        return Sim3(self.from_frame, other.to_frame, self.scale * other.scale, R, other.scale * (other.R @ self.t) + other.t)

    def plan_params(self, tilt_tol_rad: float = 0.02) -> tuple[float, float, float, float]:
        """(theta, tx, ty, scale) if this is a rotation about the vertical axis; FrameError if the frame is tilted."""
        zc = self.R[:, 2]
        tilt = math.acos(max(-1.0, min(1.0, float(zc[2]))))
        if tilt > tilt_tol_rad:
            raise FrameError(f"{self.from_frame}->{self.to_frame} tilts the vertical axis by {math.degrees(tilt):.2f} deg")
        return math.atan2(self.R[1, 0], self.R[0, 0]), float(self.t[0]), float(self.t[1]), self.scale

    def to_dict(self) -> dict:
        return {"from": self.from_frame, "to": self.to_frame, "scale": self.scale, "R": self.R.tolist(), "t": self.t.tolist()}

    @staticmethod
    def from_dict(d: dict) -> Sim3:
        return Sim3(d["from"], d["to"], d["scale"], np.array(d["R"]), np.array(d["t"]))


@dataclass(frozen=True)
class FramedPoints:
    """Points that carry their frame: canonical geometry is never a bare array."""
    frame: str
    xyz: np.ndarray

    def in_frame(self, reg: FrameRegistry, target: str) -> FramedPoints:
        return FramedPoints(target, reg.convert(self.xyz, self.frame, target))


class FrameRegistry:
    def __init__(self):
        self._edges: dict[tuple[str, str], tuple[Sim3, str]] = {}

    def add(self, tr: Sim3, provenance: str) -> None:
        if not provenance:
            raise FrameError("every transform needs a provenance string")
        self._edges[(tr.from_frame, tr.to_frame)] = (tr, provenance)
        self._edges[(tr.to_frame, tr.from_frame)] = (tr.inverse(), provenance + " (inverse)")

    def frames(self) -> list[str]:
        return sorted({f for e in self._edges for f in e})

    def path(self, a: str, b: str) -> list[str]:
        if a == b:
            return [a]
        prev, queue = {a: None}, [a]
        for cur in queue:
            for (x, y) in sorted(self._edges):
                if x == cur and y not in prev:
                    prev[y] = cur
                    queue.append(y)
        if b not in prev:
            raise FrameError(f"no transform path from {a!r} to {b!r}")
        out, n = [], b
        while n is not None:
            out.append(n)
            n = prev[n]
        return out[::-1]

    def transform(self, a: str, b: str) -> Sim3:
        p = self.path(a, b)
        tr = Sim3.identity(a)
        for x, y in zip(p, p[1:]):
            tr = tr.then(self._edges[(x, y)][0])
        return tr

    def convert(self, pts, a: str, b: str) -> np.ndarray:
        return self.transform(a, b).apply(pts)

    def provenance(self, a: str, b: str) -> list[str]:
        p = self.path(a, b)
        return [self._edges[(x, y)][1] for x, y in zip(p, p[1:])]

    def loop_residuals(self, tol_t: float = 1e-6, tol_s: float = 1e-9) -> list[tuple[list[str], float, float]]:
        """Transforms added along different routes between the same frames must agree. Returns (frames, |dt|, |ds|)
        for every pair whose DIRECT edge disagrees with the composed multi-hop path (a frame-graph inconsistency)."""
        bad = []
        for (a, b), (tr, _) in sorted(self._edges.items()):
            others = {k: v for k, v in self._edges.items() if k != (a, b) and k != (b, a)}
            sub = FrameRegistry()
            sub._edges = others
            try:
                alt = sub.transform(a, b)
            except FrameError:
                continue
            dt = float(np.linalg.norm(alt.t - tr.t))
            ds = abs(alt.scale - tr.scale)
            if dt > tol_t or ds > tol_s or np.abs(alt.R - tr.R).max() > 1e-6:
                bad.append(([a, b], dt, ds))
        return bad

    def scale_consistent(self, frames: Sequence[str], metric_frame: str, tol: float = 1e-6) -> dict[str, float | None]:
        """Scale of each frame relative to `metric_frame` (None when unreachable): a frame with scale != 1 is not metric."""
        out: dict[str, float | None] = {}
        for f in frames:
            try:
                out[f] = self.transform(f, metric_frame).scale
            except FrameError:
                out[f] = None
        return out
