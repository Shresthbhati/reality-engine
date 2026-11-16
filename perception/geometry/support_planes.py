"""Floors are floors (BUILD_SPEC B09): separate a storey's real floor from table tops, steps and shelves.

Pure array logic so it is testable without a reconstruction. Near-horizontal plane heights are clustered with a 1-D
gap rule; a cluster is the floor of a storey only if it is the lowest one that covers enough of the storey footprint
and touches wall bases. Every other cluster inside the storey is a ``support_surface``.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

GAP_M = 0.08
MIN_AREA_SHARE = 0.40
MIN_WALL_BASES = 2


def split_floor_candidates(
    heights: Sequence[float],
    areas: Sequence[float],
    footprint_area: float,
    wall_bases: Sequence[int],
    gap_m: float = GAP_M,
    min_area_share: float = MIN_AREA_SHARE,
) -> list[str]:
    """Role per plane: ``"floor"`` or ``"support_surface"``. ``wall_bases[i]`` = wall base lines reaching plane i."""
    h = np.asarray(heights, dtype=float); a = np.asarray(areas, dtype=float); wb = np.asarray(wall_bases)
    if not (len(h) == len(a) == len(wb)):
        raise ValueError("heights, areas and wall_bases must have the same length")
    if len(h) == 0:
        return []
    order = np.argsort(h, kind="stable")
    cluster = np.zeros(len(h), dtype=int)
    cluster[order] = np.concatenate([[0], np.cumsum(np.diff(h[order]) > gap_m)])
    k = int(cluster.max()) + 1
    area = np.bincount(cluster, weights=a, minlength=k)
    bases = np.bincount(cluster, weights=wb, minlength=k)
    mean_h = np.bincount(cluster, weights=h, minlength=k) / np.bincount(cluster, minlength=k)
    ok = (area >= min_area_share * footprint_area) & (bases >= MIN_WALL_BASES)
    floor_cluster = int(np.flatnonzero(ok)[np.argmin(mean_h[ok])]) if ok.any() else -1
    return ["floor" if c == floor_cluster else "support_surface" for c in cluster]
