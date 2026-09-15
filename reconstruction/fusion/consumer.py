"""Plural-source fusion consumer (P6-02 remainder).

The fusion ENGINE (`reconstruction/fusion/multi_source.py`) was
verified but nothing ingested plural sources and called it -- the
ledger's named gap: "no pipeline consumer yet ... and no WorldIR
geometry write-back". This module is that consumer:

    sparse SfM points + depth-unprojected RGB-D points
        -> spatial association (nearest neighbor within tolerance)
        -> fuse_depth_observations per co-observed point
        -> fused point set (passthrough for unassociated points)
        -> ready for WorldIR POINTCLOUD geometry write-back.

Honesty rules:
- association is a measured spatial predicate (KD-tree distance), with
  the tolerance declared by the caller and recorded in the facts;
- unassociated points pass through as single-source observations --
  dropping them would be data loss; fusing them with an imagined
  partner would be fabrication;
- conflicts are recorded per point (FusedEstimate.provenance ==
  CONFLICT); the fused value is still computed but the disagreement
  is never silently resolved by a winner-take-all pick;
- every count that explains the stage's behavior is in the returned
  facts so the report is auditable.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree

from provenance import Provenance
from reconstruction.backend.interface import ReconstructedPoint
from reconstruction.fusion.fusion import FusedEstimate
from reconstruction.fusion.multi_source import (
    DepthSourceObservation,
    fuse_depth_observations,
)


def _source_observation(point: ReconstructedPoint, source: str) -> DepthSourceObservation:
    """One pipeline point as a fusable depth claim. Precision is the
    point's recorded uncertainty sigma when present (real measured
    quality), else a declared conservative default -- never zero
    (zero precision would claim infinite confidence)."""
    sigma = getattr(point.uncertainty, "sigma", None) or getattr(
        point.uncertainty, "std_dev", None
    )
    precision = float(sigma) if sigma and sigma > 0 else 0.05
    return DepthSourceObservation(
        source=source,
        depth_m=float(np.linalg.norm(point.position)),
        precision_m=precision,
        provenance=Provenance.RECONSTRUCTED,
        confidence=0.6,
        evidence_ids=tuple(point.source_evidence_ids),
    )


def fuse_pipeline_points(
    *,
    sparse_points: Sequence[ReconstructedPoint],
    depth_points: Sequence[ReconstructedPoint],
    association_tolerance_m: float,
) -> Dict:
    """Associate co-observed sparse/depth points and fuse them.

    Fusion runs on the points' range (distance from the world origin
    along the viewing axis) -- the quantity both sources actually
    estimate about the same world point. The fused result keeps the
    association; callers that need 3D positions combine the fused
    range with the associated points' direction. Unassociated points
    pass through as single-source estimates.
    """
    facts: Dict[str, object] = {
        "n_sparse": len(sparse_points),
        "n_depth": len(depth_points),
        "association_tolerance_m": association_tolerance_m,
        "status": "ran",
    }
    fused: List[FusedEstimate] = []
    n_associated = 0
    n_conflicts = 0
    n_passthrough_sparse = 0
    n_passthrough_depth = 0

    if depth_points and sparse_points:
        sparse_arr = np.array([p.position for p in sparse_points], dtype=float)
        depth_arr = np.array([p.position for p in depth_points], dtype=float)
        tree = cKDTree(sparse_arr)
        distances, indices = tree.query(depth_arr, k=1)
        paired_depth: set = set()
        paired_sparse: set = set()
        for di, (dist, si) in enumerate(zip(distances, indices)):
            if dist <= association_tolerance_m:
                paired_depth.add(di)
                paired_sparse.add(int(si))
                si = int(si)
                obs = [
                    _source_observation(sparse_points[si], "sfm_sparse"),
                    _source_observation(depth_points[di], "rgbd_depth"),
                ]
                est = fuse_depth_observations(obs, point_id=f"assoc-{si}-{di}")
                fused.append(est)
                if est.provenance is Provenance.CONFLICT:
                    n_conflicts += 1
                n_associated += 1
        n_passthrough_sparse = len(sparse_points) - len(paired_sparse)
        n_passthrough_depth = len(depth_points) - len(paired_depth)
        for si, point in enumerate(sparse_points):
            if si not in paired_sparse:
                fused.append(fuse_depth_observations(
                    [_source_observation(point, "sfm_sparse")],
                    point_id=f"sparse-passthrough-{si}",
                ))
        for di, point in enumerate(depth_points):
            if di not in paired_depth:
                fused.append(fuse_depth_observations(
                    [_source_observation(point, "rgbd_depth")],
                    point_id=f"depth-passthrough-{di}",
                ))
    else:
        # Single-source input: everything passes through -- there is
        # nothing to associate, and saying "fused" would be a lie.
        source = "sfm_sparse" if sparse_points else "rgbd_depth"
        for si, point in enumerate(sparse_points):
            fused.append(fuse_depth_observations(
                [_source_observation(point, source)],
                point_id=f"sparse-passthrough-{si}",
            ))
        for di, point in enumerate(depth_points):
            fused.append(fuse_depth_observations(
                [_source_observation(point, source)],
                point_id=f"depth-passthrough-{di}",
            ))
        n_passthrough_sparse = len(sparse_points)
        n_passthrough_depth = len(depth_points)

    facts.update({
        "n_associated": n_associated,
        "n_conflicts": n_conflicts,
        "n_passthrough_sparse": n_passthrough_sparse,
        "n_passthrough_depth": n_passthrough_depth,
        "fused": fused,
    })
    return facts
