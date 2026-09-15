"""Multi-source depth fusion (P6-02): MVS + RGB-D + LiDAR + stereo +
monocular + neural depth, weighted/robust-fused at a single world point.

``reconstruction/fusion/fusion.py`` already implements the generic
inverse-variance fusion engine for ANY scalar quantity with a declared
uncertainty (see ``fuse_quantity``). Depth at a world point is exactly
such a quantity: multiple sources claim a distance in meters, each
with its own honestly-declared precision. This module is a thin,
depth-specific adapter over that engine -- it adds no new math, no new
conflict rule, no new weighting scheme. It exists only so callers
don't have to hand-build ``FusableObservation`` tuples with
``Unit.METER`` themselves.

Per DEPTH_FUSION.md's contract, a source's precision must be honestly
known; there is no default. A source with unknown uncertainty cannot
be fused here -- see ``DepthSourceObservation.__post_init__`` in
fusion.py's ``FusableObservation``, which this module reuses verbatim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

from engine.core.units import Unit
from provenance import Provenance

from .fusion import FusableObservation, FusedEstimate, fuse_quantity


@dataclass(frozen=True)
class DepthSourceObservation:
    """One source's depth claim (meters) for one world point.

    ``source`` names the sensing/estimation pipeline (e.g. "lidar",
    "mvs", "rgbd", "stereo", "monocular_midas", "neural_depth"), not a
    frame or image id -- ``evidence_ids`` carries that provenance.
    """

    source: str
    depth_m: float
    precision_m: float
    provenance: Provenance
    confidence: float
    evidence_ids: Tuple[str, ...] = ()

    def to_observation(self) -> FusableObservation:
        return FusableObservation(
            value=self.depth_m,
            unit=Unit.METER,
            source=self.source,
            provenance=self.provenance,
            confidence=self.confidence,
            precision=self.precision_m,
            evidence_ids=self.evidence_ids,
        )


def fuse_depth_observations(
    observations: Sequence[DepthSourceObservation],
    point_id: str,
) -> FusedEstimate:
    """Fuse depth claims from 2+ sources for the same world point.

    Delegates entirely to ``fuse_quantity``: inverse-variance weighted
    mean when sources agree (ESTIMATED), CONFLICT -- fused value still
    computed, never a winner-take-all pick or silent discard -- when
    any pair disagrees beyond ``CONFLICT_SIGMA`` combined sigmas. A
    single source passes through unchanged. Raises ``FusionError`` on
    an empty sequence (from ``fuse_quantity``).
    """
    return fuse_quantity(
        [o.to_observation() for o in observations],
        quantity=f"depth:{point_id}",
    )
