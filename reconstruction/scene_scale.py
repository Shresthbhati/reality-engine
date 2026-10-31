"""Working unit for RELATIVE reconstructions.

COLMAP's model scale is arbitrary: one solve puts a living room 0.01 units wide, the next 40. Every structure threshold
downstream (plane tolerance, "a wall spans >= 1 m", room sizes) is written in metres, so whether a room is found at all
was a lottery on the solver's scale (measured: a fresh solve of 20 living-room photos found no plane, so no up axis and
no room; an incremental one found planes but demoted every one of them as sub-room-sized).

``working_unit`` rescales a RELATIVE result so the typical camera-to-scene distance is a plausible interior viewing
distance. It is NOT a metric claim: the world stays ``relative``, sizes stay ``model_units``, and any measured
reference replaces it (references are recorded in the solver's own units and applied BEFORE this runs).
"""

from __future__ import annotations

import dataclasses
import math
from typing import Optional, Tuple

import numpy as np

from reconstruction.backend.interface import ReconstructionResult

#: Typical camera-to-surface distance in a furnished room photographed handheld (nominal, not measured).
NOMINAL_VIEW_DISTANCE = 3.0
#: A candidate in the same solver frame has the same scale; beyond this drift it is a different solve, so the head's
#: unit does not apply to it.
REUSE_TOLERANCE = 0.25
_MAX_POINTS = 20000


def view_distance(result: ReconstructionResult) -> Optional[float]:
    """Median distance (model units) from each registered camera to the points it observes; None when unmeasurable."""
    poses = {p.evidence_id: np.asarray(p.position, dtype=float) for p in result.camera_poses}
    if not poses or not result.points:
        return None
    step = max(1, len(result.points) // _MAX_POINTS)
    dists = []
    for pt in result.points[::step]:
        x = np.asarray(pt.position, dtype=float)
        for eid in pt.source_evidence_ids or ():
            c = poses.get(eid)
            if c is not None:
                dists.append(float(np.linalg.norm(x - c)))
    if not dists:                                   # no observation lists: distance to the camera centroid
        centre = np.mean(list(poses.values()), axis=0)
        dists = [float(np.linalg.norm(np.asarray(p.position, dtype=float) - centre)) for p in result.points[::step]]
    d = float(np.median(dists)) if dists else 0.0
    return d if math.isfinite(d) and d > 1e-12 else None


def rescale(result: ReconstructionResult, k: float) -> ReconstructionResult:
    """Uniform scale of positions by ``k`` (rotations untouched), keeping order and every other field."""
    return ReconstructionResult(
        points=[dataclasses.replace(p, position=tuple(c * k for c in p.position)) for p in result.points],
        camera_poses=[dataclasses.replace(p, position=tuple(c * k for c in p.position)) for p in result.camera_poses],
        registration_status=result.registration_status,
    )


def working_unit(result: ReconstructionResult, head_k: Optional[float] = None
                 ) -> Tuple[ReconstructionResult, Optional[dict]]:
    """(rescaled result, facts) for a RELATIVE reconstruction; (result, None) when its scale cannot be judged.

    ``head_k`` is the unit the current version already uses: reused while this solve's own estimate agrees with it
    (same solver frame), so successive versions of one world never drift apart in unit."""
    d = view_distance(result)
    if d is None:
        return result, None
    own = NOMINAL_VIEW_DISTANCE / d
    reused = bool(head_k) and abs(own / head_k - 1.0) <= REUSE_TOLERANCE
    k = float(head_k) if reused else own
    return rescale(result, k), {
        "unit_k": k, "own_estimate_k": own, "reused_from_head": reused, "median_view_distance_model": d,
        "basis": (f"solver units scaled so the median camera-to-point distance is {NOMINAL_VIEW_DISTANCE:g} working "
                  "units (a typical interior viewing distance); a working unit, NOT metres"),
    }
