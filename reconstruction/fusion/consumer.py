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


def _classify_point_source(point: ReconstructedPoint) -> str:
    """Classify a point by its track_id prefix for fusion purposes."""
    track_id = getattr(point, 'track_id', '')
    if track_id.startswith('depth-'):
        return 'rgbd_depth'
    elif track_id.startswith('dense_mvs:'):
        return 'dense_mvs'
    else:
        return 'sfm_sparse'


def fuse_pipeline_points(
    *,
    sparse_points: Sequence[ReconstructedPoint],
    depth_points: Sequence[ReconstructedPoint],
    dense_mvs_points: Sequence[ReconstructedPoint] = (),
    association_tolerance_m: float,
) -> Dict:
    """Associate co-observed sparse/depth/dense_mvs points and fuse them.

    Fusion runs on the points' range (distance from the world origin
    along the viewing axis) -- the quantity all sources actually
    estimate about the same world point. The fused result keeps the
    association; callers that need 3D positions combine the fused
    range with the associated points' direction. Unassociated points
    pass through as single-source estimates.
    """
    facts: Dict[str, object] = {
        "n_sparse": len(sparse_points),
        "n_depth": len(depth_points),
        "n_dense_mvs": len(dense_mvs_points),
        "association_tolerance_m": association_tolerance_m,
        "status": "ran",
    }
    fused: List[FusedEstimate] = []
    n_associated = 0
    n_conflicts = 0
    n_passthrough_sparse = 0
    n_passthrough_depth = 0
    n_passthrough_dense_mvs = 0

    # Group points by source type
    all_points_by_source = {
        'sfm_sparse': list(sparse_points),
        'rgbd_depth': list(depth_points),
        'dense_mvs': list(dense_mvs_points),
    }

    # If we have at least two source types with points, do cross-source fusion
    sources_with_points = {k: v for k, v in all_points_by_source.items() if v}
    
    if len(sources_with_points) >= 2:
        # Use the first source as the base for KD-tree association
        source_names = list(sources_with_points.keys())
        base_source = source_names[0]
        base_points = sources_with_points[base_source]
        
        base_arr = np.array([p.position for p in base_points], dtype=float)
        tree = cKDTree(base_arr)
        
        # Track which points are paired
        paired = {src: set() for src in sources_with_points}
        
        # Compare each other source against the base
        for other_source in source_names[1:]:
            other_points = sources_with_points[other_source]
            other_arr = np.array([p.position for p in other_points], dtype=float)
            distances, indices = tree.query(other_arr, k=1)
            
            for di, (dist, si) in enumerate(zip(distances, indices)):
                if dist <= association_tolerance_m:
                    paired[base_source].add(int(si))
                    paired[other_source].add(di)
                    si = int(si)
                    obs = [
                        _source_observation(base_points[si], base_source),
                        _source_observation(other_points[di], other_source),
                    ]
                    est = fuse_depth_observations(obs, point_id=f"assoc-{base_source}-{other_source}-{si}-{di}")
                    fused.append(est)
                    if est.provenance is Provenance.CONFLICT:
                        n_conflicts += 1
                    n_associated += 1
        
        # Add passthrough points
        for src_name, points in all_points_by_source.items():
            src_paired = paired.get(src_name, set())
            count = 0
            for i, point in enumerate(points):
                if i not in src_paired:
                    fused.append(fuse_depth_observations(
                        [_source_observation(point, src_name)],
                        point_id=f"{src_name}-passthrough-{i}",
                    ))
                    count += 1
            if src_name == 'sfm_sparse':
                n_passthrough_sparse = count
            elif src_name == 'rgbd_depth':
                n_passthrough_depth = count
            elif src_name == 'dense_mvs':
                n_passthrough_dense_mvs = count
    else:
        # Single-source input: everything passes through
        for src_name, points in sources_with_points.items():
            for i, point in enumerate(points):
                fused.append(fuse_depth_observations(
                    [_source_observation(point, src_name)],
                    point_id=f"{src_name}-passthrough-{i}",
                ))
            if src_name == 'sfm_sparse':
                n_passthrough_sparse = len(points)
            elif src_name == 'rgbd_depth':
                n_passthrough_depth = len(points)
            elif src_name == 'dense_mvs':
                n_passthrough_dense_mvs = len(points)

    facts.update({
        "n_associated": n_associated,
        "n_conflicts": n_conflicts,
        "n_passthrough_sparse": n_passthrough_sparse,
        "n_passthrough_depth": n_passthrough_depth,
        "n_passthrough_dense_mvs": n_passthrough_dense_mvs,
        "fused": fused,
    })
    return facts
