"""Coordinate alignment: frames, similarity transforms, robust registration, and the guard that stops incomparable values
from being compared.

* ``FrameGraph`` records which coordinate frames exist and which similarity transforms between them are KNOWN. Two values are
  comparable only if they share a frame or a transform path exists; if metric distances are being compared, neither frame may be
  of relative scale unless the path went through a transform marked as scale-anchored.
* ``estimate_similarity`` is the closed-form (Umeyama) Sim(3) fit with explicit degeneracy checks.
* ``ransac_similarity`` rejects outlier correspondences. The inlier threshold has NO default: it is a physical tolerance that must
  come from the producing engine's calibration, so it is a required argument rather than an invented constant.
* Registration failure, degenerate geometry and scale mismatch are returned as explicit statuses, never as a best-effort transform.

Bundle adjustment is NOT implemented here (COLMAP performs it inside the SfM stage); the reprojection utilities in consensus.py
are what other modules use to judge a candidate against the original image evidence.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


class AlignmentError(ValueError):
    pass


class FrameMismatch(AlignmentError):
    """Two values were about to be compared across frames that have no known transform (or whose scales are incomparable)."""


class DegenerateGeometry(AlignmentError):
    """Too few or too degenerate correspondences to determine a transform."""


@dataclass(frozen=True)
class Frame:
    frame_id: str
    kind: str = "world"  # "world" | "camera" | "image" | "depth" | "model"
    scale_state: str = "unknown"  # "metric" | "relative" | "unknown"
    description: str = ""

    def __post_init__(self):
        if not self.frame_id:
            raise AlignmentError("a frame needs an id")
        if self.scale_state not in ("metric", "relative", "unknown"):
            raise AlignmentError(f"unknown scale_state {self.scale_state!r}")


@dataclass(frozen=True)
class Sim3:
    """x' = scale * R @ x + t."""

    scale: float
    rotation: Tuple[Tuple[float, float, float], ...]
    translation: Tuple[float, float, float]

    @staticmethod
    def from_arrays(scale: float, rotation: np.ndarray, translation: np.ndarray) -> Sim3:
        return Sim3(
            float(scale),
            tuple(tuple(float(v) for v in row) for row in rotation),
            tuple(float(v) for v in translation),
        )

    @property
    def R(self) -> np.ndarray:
        return np.array(self.rotation, dtype=float)

    @property
    def t(self) -> np.ndarray:
        return np.array(self.translation, dtype=float)

    def apply(self, points: np.ndarray) -> np.ndarray:
        p = np.atleast_2d(np.asarray(points, dtype=float))
        return self.scale * p @ self.R.T + self.t

    def inverse(self) -> Sim3:
        Rt = self.R.T
        s = 1.0 / self.scale
        return Sim3.from_arrays(s, Rt, -s * Rt @ self.t)

    def compose(self, then: Sim3) -> Sim3:
        """Apply ``self`` first, then ``then``."""
        return Sim3.from_arrays(
            then.scale * self.scale, then.R @ self.R, then.scale * then.R @ self.t + then.t
        )


@dataclass(frozen=True)
class _Edge:
    transform: Sim3
    source: str
    scale_anchored: bool


class FrameGraph:
    def __init__(self) -> None:
        self._frames: Dict[str, Frame] = {}
        self._edges: Dict[Tuple[str, str], _Edge] = {}

    def add_frame(self, frame: Frame) -> None:
        if frame.frame_id in self._frames and self._frames[frame.frame_id] != frame:
            raise AlignmentError(f"frame {frame.frame_id!r} is already defined differently")
        self._frames[frame.frame_id] = frame

    def frame(self, frame_id: str) -> Frame:
        try:
            return self._frames[frame_id]
        except KeyError:
            raise FrameMismatch(f"unknown coordinate frame {frame_id!r}") from None

    def add_transform(
        self, src: str, dst: str, transform: Sim3, *, source: str, scale_anchored: bool = False
    ) -> None:
        """Record a KNOWN transform src -> dst. ``source`` says where it came from (a registration result, a calibration, ...)."""
        self.frame(src)
        self.frame(dst)
        if not source:
            raise AlignmentError("a transform must state its source")
        self._edges[(src, dst)] = _Edge(transform, source, scale_anchored)
        self._edges[(dst, src)] = _Edge(transform.inverse(), source + " (inverse)", scale_anchored)

    def _path(self, src: str, dst: str) -> List[str] | None:
        if src == dst:
            return [src]
        seen = {src}
        queue = deque([[src]])
        while queue:
            path = queue.popleft()
            for a, b in self._edges:
                if a == path[-1] and b not in seen:
                    if b == dst:
                        return path + [b]
                    seen.add(b)
                    queue.append(path + [b])
        return None

    def comparable(self, a: str, b: str, *, metric: bool = False) -> Tuple[bool, str]:
        """(ok, reason). With ``metric=True`` the comparison is of metric lengths/areas, which needs a metric path."""
        fa, fb = self.frame(a), self.frame(b)
        path = self._path(a, b)
        if path is None:
            return False, f"no known transform between frames {a!r} and {b!r}"
        if metric:
            anchored = any(
                self._edges[(path[i], path[i + 1])].scale_anchored for i in range(len(path) - 1)
            )
            if "relative" in (fa.scale_state, fb.scale_state) and not anchored:
                return False, (
                    f"frames {a!r} ({fa.scale_state}) and {b!r} ({fb.scale_state}) are not on a common metric scale: "
                    "no scale-anchored transform connects them"
                )
            if "unknown" in (fa.scale_state, fb.scale_state) and not anchored:
                return (
                    False,
                    f"the scale of frame {a!r} or {b!r} is unknown, so metric values cannot be compared",
                )
        return True, ""

    def require_comparable(self, a: str, b: str, *, metric: bool = False) -> None:
        ok, why = self.comparable(a, b, metric=metric)
        if not ok:
            raise FrameMismatch(why)

    def transform_points(self, points: np.ndarray, src: str, dst: str) -> np.ndarray:
        path = self._path(src, dst)
        if path is None:
            raise FrameMismatch(f"no known transform between frames {src!r} and {dst!r}")
        out = np.atleast_2d(np.asarray(points, dtype=float))
        for i in range(len(path) - 1):
            out = self._edges[(path[i], path[i + 1])].transform.apply(out)
        return out


# ------------------------------------------------------------------------------------------------ estimation


def _check_points(src: np.ndarray, dst: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    a = np.asarray(src, dtype=float)
    b = np.asarray(dst, dtype=float)
    if a.ndim != 2 or a.shape[1] != 3 or a.shape != b.shape:
        raise AlignmentError(f"need matching (N,3) arrays; got {a.shape} and {b.shape}")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise AlignmentError("correspondences contain non-finite coordinates")
    return a, b


def estimate_similarity(
    src: np.ndarray, dst: np.ndarray, *, with_scale: bool = True, weights: np.ndarray | None = None
) -> Sim3:
    """Closed-form least-squares Sim(3) (Umeyama 1991). Raises ``DegenerateGeometry`` for < 3 points or collinear points
    (rank < 2): such a set cannot determine a rotation, and returning one would be inventing it."""
    a, b = _check_points(src, dst)
    n = len(a)
    if n < 3:
        raise DegenerateGeometry(
            f"{n} correspondences; at least 3 are needed to determine a similarity transform"
        )
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != (n,) or (w < 0).any() or w.sum() <= 0:
        raise AlignmentError("weights must be a non-negative (N,) vector with a positive sum")
    w = w / w.sum()
    mu_a, mu_b = w @ a, w @ b
    ac, bc = a - mu_a, b - mu_b
    cov = (bc * w[:, None]).T @ ac
    u, d, vt = np.linalg.svd(cov)
    extent = float(np.abs(ac).max())
    if (
        extent == 0.0
        or np.linalg.matrix_rank(ac, tol=1e-9 * extent) < 2
        or d[1] < 1e-12 * max(d[0], 1e-300)
    ):
        raise DegenerateGeometry(
            "correspondences are (near) collinear; the rotation is undetermined"
        )
    s_mat = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s_mat[2, 2] = -1.0
    rot = u @ s_mat @ vt
    var_a = float((w * (ac**2).sum(axis=1)).sum())
    scale = float(np.trace(np.diag(d) @ s_mat) / var_a) if with_scale else 1.0
    if scale <= 0 or not math.isfinite(scale):
        raise DegenerateGeometry(f"estimated scale {scale!r} is not positive and finite")
    return Sim3.from_arrays(scale, rot, mu_b - scale * rot @ mu_a)


@dataclass(frozen=True)
class AlignmentResult:
    status: str  # "ok" | "degenerate" | "registration_failed"
    transform: Sim3 | None
    inliers: np.ndarray | None  # bool mask over the input correspondences
    n_points: int
    rms_inlier_residual: float | None
    inlier_threshold: float
    notes: Tuple[str, ...] = ()

    @property
    def n_inliers(self) -> int:
        return 0 if self.inliers is None else int(self.inliers.sum())

    @property
    def inlier_ratio(self) -> float:
        return 0.0 if not self.n_points else self.n_inliers / self.n_points

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def ransac_similarity(
    src: np.ndarray,
    dst: np.ndarray,
    *,
    inlier_threshold: float,
    with_scale: bool = True,
    min_inliers: int | None = None,
    iterations: int = 500,
    seed: int = 0,
) -> AlignmentResult:
    """Robust Sim(3) between two sets of corresponding points. ``inlier_threshold`` is in ``dst`` units and has no default
    (it must come from the engine's calibration). Returns an explicit failure status instead of a poor transform."""
    if not inlier_threshold > 0:
        raise AlignmentError(
            "inlier_threshold must be a positive tolerance in the destination frame's units"
        )
    a, b = _check_points(src, dst)
    n = len(a)
    need = 3 if min_inliers is None else max(3, int(min_inliers))
    if n < need:
        return AlignmentResult(
            "degenerate",
            None,
            None,
            n,
            None,
            inlier_threshold,
            (f"{n} correspondences < required {need}",),
        )
    rng = np.random.default_rng(seed)
    best_mask: np.ndarray | None = None
    best_count = -1  # a hypothesis with zero inliers is still a hypothesis (-> registration_failed, not degenerate)
    for _ in range(iterations):
        idx = rng.choice(n, size=3, replace=False)
        try:
            tr = estimate_similarity(a[idx], b[idx], with_scale=with_scale)
        except DegenerateGeometry:
            continue
        mask = np.linalg.norm(tr.apply(a) - b, axis=1) <= inlier_threshold
        if int(mask.sum()) > best_count:
            best_count, best_mask = int(mask.sum()), mask
    if best_mask is None:
        return AlignmentResult(
            "degenerate",
            None,
            None,
            n,
            None,
            inlier_threshold,
            ("every sampled subset was degenerate",),
        )
    if best_count < need:
        return AlignmentResult(
            "registration_failed",
            None,
            best_mask,
            n,
            None,
            inlier_threshold,
            (
                f"best hypothesis had {best_count} inliers < required {need} at threshold {inlier_threshold:g}",
            ),
        )
    try:
        refit = estimate_similarity(a[best_mask], b[best_mask], with_scale=with_scale)
    except DegenerateGeometry as exc:
        return AlignmentResult(
            "degenerate", None, best_mask, n, None, inlier_threshold, (str(exc),)
        )
    resid = np.linalg.norm(refit.apply(a) - b, axis=1)
    final = resid <= inlier_threshold
    if int(final.sum()) < need:
        return AlignmentResult(
            "registration_failed",
            None,
            final,
            n,
            None,
            inlier_threshold,
            ("refit on the inlier set lost inliers below the required count",),
        )
    rms = float(np.sqrt(np.mean(resid[final] ** 2)))
    return AlignmentResult("ok", refit, final, n, rms, inlier_threshold)


def scale_mismatch(result: AlignmentResult, *, expected: float, tolerance: float) -> str | None:
    """If both frames are supposed to be metric the fitted scale should be ~1. Returns a description of the mismatch, or None."""
    if result.transform is None:
        return "no transform to check"
    s = result.transform.scale
    return (
        None
        if abs(s - expected) <= tolerance
        else f"fitted scale {s:.6g} differs from the expected {expected:g} by more than {tolerance:g}"
    )


# ------------------------------------------------------------------------------------------------ camera poses

Pose = Tuple[np.ndarray, np.ndarray]  # (R_cam_to_world 3x3, camera centre in world coordinates)


def pose_residuals(
    poses_a: Dict[str, Pose], poses_b: Dict[str, Pose], transform: Sim3
) -> Dict[str, Tuple[float, float]]:
    """For every camera present in both reconstructions: (centre error in b's units, rotation error in degrees) after mapping
    reconstruction a into b's frame with ``transform``."""
    out: Dict[str, Tuple[float, float]] = {}
    for cam in sorted(set(poses_a) & set(poses_b)):
        ra, ca = poses_a[cam]
        rb, cb = poses_b[cam]
        mapped_c = transform.apply(np.asarray(ca, dtype=float))[0]
        mapped_r = transform.R @ np.asarray(ra, dtype=float)
        cos = (np.trace(mapped_r.T @ np.asarray(rb, dtype=float)) - 1.0) / 2.0
        out[cam] = (
            float(np.linalg.norm(mapped_c - np.asarray(cb, dtype=float))),
            float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))),
        )
    return out


def flag_inconsistent_poses(
    residuals: Dict[str, Tuple[float, float]], *, mad_k: float = 3.5
) -> List[str]:
    """Cameras whose centre residual is a robust (median/MAD) outlier. ``mad_k`` is the usual modified-z-score cut (3.5, Iglewicz &
    Hoaglin); with fewer than 4 cameras no outlier can be identified and the list is empty rather than guessed."""
    if len(residuals) < 4:
        return []
    ids = sorted(residuals)
    vals = np.array([residuals[i][0] for i in ids])
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    if mad == 0.0:
        return [i for i, v in zip(ids, vals) if v > med]
    z = 0.6745 * (vals - med) / mad
    return [i for i, zz in zip(ids, z) if zz > mad_k]
