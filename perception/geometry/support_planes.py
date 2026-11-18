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
#: A second qualifying cluster closer than this above an accepted floor is a slab or step on it, not another floor
#: (the same tolerance as ``room_graph.FLOOR_HEIGHT_TOLERANCE_M``).
MIN_FLOOR_SEPARATION_M = 0.15


def split_floor_candidates(
    heights: Sequence[float],
    areas: Sequence[float],
    footprint_area: float | Sequence[float],
    wall_bases: Sequence[int],
    gap_m: float = GAP_M,
    min_area_share: float = MIN_AREA_SHARE,
) -> list[str]:
    """Role per plane: ``"floor"`` or ``"support_surface"``. ``wall_bases[i]`` = wall base lines reaching plane i.

    ``footprint_area`` is the footprint the plane's walls enclose: one value for all planes, or one per plane (rooms of
    different sizes each judged against their own walls). Every height cluster that reaches enough walls and spans enough
    of its footprint is a floor -- a building has one per room and storey, not one in total; only a qualifying cluster
    within ``MIN_FLOOR_SEPARATION_M`` above an accepted floor is a slab on it."""
    h = np.asarray(heights, dtype=float); a = np.asarray(areas, dtype=float); wb = np.asarray(wall_bases)
    if not (len(h) == len(a) == len(wb)):
        raise ValueError("heights, areas and wall_bases must have the same length")
    if len(h) == 0:
        return []
    fp = np.broadcast_to(np.asarray(footprint_area, dtype=float), h.shape)
    order = np.argsort(h, kind="stable")
    cluster = np.zeros(len(h), dtype=int)
    cluster[order] = np.concatenate([[0], np.cumsum(np.diff(h[order]) > gap_m)])
    k = int(cluster.max()) + 1
    area = np.bincount(cluster, weights=a, minlength=k)
    bases = np.bincount(cluster, weights=wb, minlength=k)
    mean_h = np.bincount(cluster, weights=h, minlength=k) / np.bincount(cluster, minlength=k)
    footprint = np.zeros(k)
    np.maximum.at(footprint, cluster, fp)
    ok = (area >= min_area_share * footprint) & (bases >= MIN_WALL_BASES)
    floors: list[int] = []
    for c in np.flatnonzero(ok)[np.argsort(mean_h[ok])]:
        if not floors or mean_h[c] - mean_h[floors[-1]] > MIN_FLOOR_SEPARATION_M:
            floors.append(int(c))
    return ["floor" if c in floors else "support_surface" for c in cluster]
