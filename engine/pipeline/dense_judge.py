"""Is a dense candidate a BETTER description of the scene than the sparse world it would extend?

Dense MVS succeeding is not evidence that its output should enter the world: a wrong frame, a wrong scale, or
depth-map noise produces a large, confident-looking cloud that contradicts the established geometry. So the dense
cloud is judged against the sparse structure before anything is written, with the same discipline as
``candidate_selection`` -- ordered rules, the deciding rule recorded, no combined score:

    1. judgeable      both clouds are large enough to compare; otherwise the dense cloud is REJECTED (unjustified)
    2. regression     the dense cloud must reproduce the established structure (``sparse_retained``) and stay inside
                      the scene the sparse model established (``outside``); otherwise REJECTED
    3. improvement    it must be materially denser (``density_gain``) AND add surface the sparse model does not
                      already sample (``new_fraction``); otherwise EQUIVALENT -- the coherent sparse world is kept
    4. otherwise      ACCEPTED

Thresholds are EXPERIMENTAL starting points (scale-free: distances are measured in units of the sparse cloud's own
point spacing, so the judge works in relative and metric worlds alike). Every verdict carries the measurements.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

ACCEPTED, EQUIVALENT, REJECTED = "accepted", "equivalent", "rejected"

MIN_JUDGE_POINTS = 50            # below this either cloud says nothing measurable
MATCH_SPACINGS = 5.0             # "the same surface": within this many sparse spacings
MIN_SPARSE_RETAINED = 0.6        # share of sparse points a dense cloud must still reproduce
MAX_OUTSIDE = 0.25               # share of dense points allowed outside the (padded) sparse extent
EXTENT_PAD = 0.25                # padding of the sparse extent, as a fraction of each axis span
MIN_DENSITY_GAIN = 2.0           # dense/sparse point-count ratio needed to call it an improvement
MIN_NEW_FRACTION = 0.25          # share of dense points on surface the sparse cloud does not sample
_SAMPLE = 5000                   # spacing is estimated on a deterministic subsample


def _spacing(tree, pts: np.ndarray) -> float:
    step = max(1, len(pts) // _SAMPLE)
    d, _ = tree.query(pts[::step], k=2)
    return float(np.median(d[:, 1]))


def judge_dense(sparse_xyz: Sequence[Sequence[float]], dense_xyz: Sequence[Sequence[float]]) -> Dict[str, object]:
    """Judge ``dense_xyz`` against ``sparse_xyz`` (both in the SAME world frame and scale)."""
    from scipy.spatial import cKDTree

    sp = np.asarray(sparse_xyz, dtype=float).reshape(-1, 3)
    de = np.asarray(dense_xyz, dtype=float).reshape(-1, 3)
    sp, de = sp[np.isfinite(sp).all(1)], de[np.isfinite(de).all(1)]
    base = {"thresholds_status": "EXPERIMENTAL", "n_sparse": int(len(sp)), "n_dense": int(len(de))}
    if len(sp) < MIN_JUDGE_POINTS or len(de) < MIN_JUDGE_POINTS:
        return dict(base, verdict=REJECTED, deciding="judgeable",
                    why=f"cannot be judged: {len(sp)} sparse / {len(de)} dense points (needs >= {MIN_JUDGE_POINTS} each)")

    sp_tree, de_tree = cKDTree(sp), cKDTree(de)
    spacing = _spacing(sp_tree, sp)
    if not spacing > 0:
        return dict(base, verdict=REJECTED, deciding="judgeable", why="sparse cloud has no measurable point spacing")
    tol = MATCH_SPACINGS * spacing

    retained = float((de_tree.query(sp)[0] <= tol).mean())
    lo, hi = np.percentile(sp, 1, axis=0), np.percentile(sp, 99, axis=0)
    pad = EXTENT_PAD * np.maximum(hi - lo, tol)
    outside = float(((de < lo - pad) | (de > hi + pad)).any(1).mean())
    new_fraction = float((sp_tree.query(de)[0] > tol).mean())
    gain = len(de) / len(sp)
    measures = dict(base, sparse_spacing=round(spacing, 6), sparse_retained=round(retained, 3),
                    outside=round(outside, 3), density_gain=round(gain, 2), new_fraction=round(new_fraction, 3))

    if retained < MIN_SPARSE_RETAINED:
        return dict(measures, verdict=REJECTED, deciding="regression",
                    why=f"dense cloud reproduces only {retained:.0%} of the established structure "
                        f"(needs >= {MIN_SPARSE_RETAINED:.0%}): it contradicts the sparse model")
    if outside > MAX_OUTSIDE:
        return dict(measures, verdict=REJECTED, deciding="regression",
                    why=f"{outside:.0%} of the dense points lie outside the established scene extent "
                        f"(allowed <= {MAX_OUTSIDE:.0%})")
    if gain < MIN_DENSITY_GAIN or new_fraction < MIN_NEW_FRACTION:
        return dict(measures, verdict=EQUIVALENT, deciding="improvement",
                    why=f"no material gain (density x{gain:.1f}, {new_fraction:.0%} new surface): "
                        "the existing coherent sparse world is kept")
    return dict(measures, verdict=ACCEPTED, deciding="improvement",
                why=f"reproduces {retained:.0%} of the established structure and adds density x{gain:.1f} "
                    f"with {new_fraction:.0%} new surface")
