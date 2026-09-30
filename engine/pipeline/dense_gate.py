"""When is dense reconstruction (Level 3) justified by the evidence? Measured facts, never a photo count.

Dense MVS is expensive and only as good as the sparse model it rectifies against. It is worth running when the
sparse reconstruction already shows the properties stereo matching needs:

    registration     most of the supplied photographs are placed in ONE model
    geometric support  the structure points are seen by several cameras (median track length)
    coverage         the cameras look at the scene from enough different directions (angular spread)
    structure        there is enough sparse structure to anchor the rectification

There is deliberately no "N photos = Level 3" rule and no combined score: each criterion is measured, compared with
its own stated threshold, and reported with the reason it passed or failed. The thresholds are EXPERIMENTAL starting
points (documented as such in docs/PROGRESSIVE_RECONSTRUCTION.md), not calibrated constants; change them with
measurements, and the facts returned here show exactly which measurement decided.

Not measured here (and therefore never claimed): GSD/resolution adequacy, per-camera confidence, depth availability.
COLMAP's own dense capability is probed separately by the dense stage, which refuses honestly when it is absent.
"""

from __future__ import annotations

from statistics import median
from typing import Dict, Optional

#: EXPERIMENTAL thresholds -- see module docstring.
MIN_REGISTERED_FRACTION = 0.8
MIN_MEDIAN_TRACK_LENGTH = 3
MIN_COVERAGE_DEG = 15.0
MIN_STRUCTURE_POINTS = 300


def dense_readiness(result, n_input: int, coverage_deg: Optional[float] = None) -> Dict[str, object]:
    """Measure the sparse reconstruction against the dense-readiness criteria.

    ``result`` is a ReconstructionResult (camera_poses, points); ``coverage_deg`` the angular coverage of the
    registered cameras around the scene centre when already computed (None = not measurable -> that criterion
    fails, it is never assumed).
    """
    poses = list(getattr(result, "camera_poses", None) or [])
    points = list(getattr(result, "points", None) or [])
    registered = len(poses)
    fraction = registered / n_input if n_input else 0.0
    tracks = [len(getattr(p, "source_evidence_ids", None) or []) for p in points]
    median_track = float(median(tracks)) if tracks else 0.0

    checks = [
        ("registration", fraction >= MIN_REGISTERED_FRACTION,
         f"{registered} of {n_input} photographs registered ({fraction:.0%}; needs >= {MIN_REGISTERED_FRACTION:.0%})"),
        ("geometric support", median_track >= MIN_MEDIAN_TRACK_LENGTH,
         f"median structure point is seen by {median_track:g} cameras (needs >= {MIN_MEDIAN_TRACK_LENGTH})"),
        ("coverage", coverage_deg is not None and coverage_deg >= MIN_COVERAGE_DEG,
         ("angular coverage not measurable" if coverage_deg is None else
          f"cameras span {coverage_deg:.0f} degrees around the scene (needs >= {MIN_COVERAGE_DEG:g})")),
        ("structure", len(points) >= MIN_STRUCTURE_POINTS,
         f"{len(points)} sparse structure points (needs >= {MIN_STRUCTURE_POINTS})"),
    ]
    failed = [f"{name}: {why}" for name, ok, why in checks if not ok]
    return {
        "escalate": not failed,
        "criteria": {name: {"passed": ok, "measured": why} for name, ok, why in checks},
        "reasons": failed or ["every measured criterion passed"],
        "thresholds_status": "EXPERIMENTAL",
        "measured": {"registered": registered, "input": n_input, "registered_fraction": round(fraction, 3),
                     "median_track_length": median_track, "structure_points": len(points),
                     "coverage_deg": None if coverage_deg is None else round(float(coverage_deg), 1)},
    }
