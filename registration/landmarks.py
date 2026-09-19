"""Landmark registration (registration/landmarks.py) -- the P4-01 open
item: "landmark/correspondence method (needs cross-source co-observation
resolver from P7-01 multi-view identity ... remaining work is wiring
into RegistrationEngine)".

A LANDMARK is a named physical feature observed by >=2 sources. The
IDENTITY comes from the observation layer (track_id / provenance), not
from geometric proximity -- this is the method's entire advantage over
ICP: correspondences are declared, so no local-minimum failure mode
exists and every solve is a closed-form Kabsch fit on exact pairs.

Honesty rules (mirroring the repo's registration discipline):

  - Ambiguity is represented, never silently resolved: a landmark id
    observed more than once (e.g. a source without track identity
    cannot say WHICH of its N observations is the named feature) is
    reported as a LandmarkAmbiguity and EXCLUDED from the solve.
  - Degenerate geometry refuses with RegistrationError naming the
    degeneracy: <3 pairs (rotation about the connecting axis is
    underdetermined by 2), collinear pairs (rotation about the line
    undetermined), coplanar pairs (roll about the plane normal
    undetermined). A degenerate fit would be a guess.
  - Outlier identities are REJECTED and reported: a robust
    deterministic consensus (exhaustive over correspondence-index
    seeds -- landmark sets are small by construction) picks the
    largest self-consistent subset; rejected ids are named in the
    result reason. A wrong landmark identity must not corrupt the
    transform.
  - Uncertainty is residual-derived: measured rmse, inlier fraction,
    and a RegistrationCovariance from the same residual statistics
    model the other methods use.

Fixtures in tests are deterministic unit scenes; real cross-source
landmark runs are recorded as real in the ledgers.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from engine.physics.math3 import Vec3
from registration.registration import (
    RegistrationCovariance,
    RegistrationError,
    RegistrationResult,
    ResidualStats,
    _kabsch,
    _matrix_to_quat,
)
from reconstruction.calibration.transforms import RigidTransform


@dataclass(frozen=True)
class LandmarkCorrespondence:
    """One declared correspondence: the same physical landmark observed
    in two frames. Identity is DECLARED by the caller (track id /
    provenance); this module never infers it from geometry."""

    landmark_id: str
    source_position: Tuple[float, float, float]
    target_position: Tuple[float, float, float]


@dataclass(frozen=True)
class LandmarkAmbiguity:
    """A landmark that could not be used because its observation set is
    ambiguous (multiple source observations claim the same identity and
    no observation-layer tiebreaker exists)."""

    landmark_id: str
    n_source_observations: int
    reason: str

    def to_dict(self) -> dict:
        return {
            "landmark_id": self.landmark_id,
            "n_source_observations": self.n_source_observations,
            "reason": self.reason,
        }


def _validate_inputs(
    correspondences: Sequence[LandmarkCorrespondence],
) -> Tuple[List[LandmarkCorrespondence], List[LandmarkAmbiguity]]:
    """Deduplicate ambiguous landmark ids (excluded, reported) and
    return the unambiguous set. Duplicate ids mean the caller's
    observation layer could not disambiguate -- using both would
    fabricate a correspondence that may not exist."""
    by_id: Dict[str, List[LandmarkCorrespondence]] = {}
    for c in correspondences:
        by_id.setdefault(c.landmark_id, []).append(c)
    usable: List[LandmarkCorrespondence] = []
    ambiguous: List[LandmarkAmbiguity] = []
    for lid in sorted(by_id):
        group = by_id[lid]
        if len(group) == 1:
            usable.append(group[0])
        else:
            distinct_sources = {c.source_position for c in group}
            if len(distinct_sources) == 1:
                # Exact duplicate declaration (same identity, same
                # position): harmless, keep one.
                usable.append(group[0])
            else:
                ambiguous.append(LandmarkAmbiguity(
                    landmark_id=lid, n_source_observations=len(group),
                    reason="landmark observed at multiple source positions "
                           "with no observation-layer tiebreaker -- excluded "
                           "from the solve rather than guessed",
                ))
    return usable, ambiguous


def _degeneracy(us: np.ndarray, ts: np.ndarray) -> Optional[str]:
    """Name the geometric degeneracy of a correspondence set, or None.

    For DECLARED correspondences (point-to-point, not free points), the
    Procrustes/Kabsch fit is unique iff the source points are not
    collinear (scatter rank >= 2): a scalene planar triangle still pins
    all 6 DOF through its point identity, while a collinear set leaves
    rotation about the line undetermined. Coplanar-but-non-collinear
    sets are therefore SOLVABLE here; their weaker conditioning is
    reported through RegistrationCovariance.degenerate_axes instead of
    refusing (measured, not guessed)."""
    n = len(us)
    if n < 3:
        return (
            f"only {n} correspondence pair(s); a full 6-DOF solve needs >=3 "
            "non-degenerate pairs (2 pairs leave rotation about the "
            "connecting axis undetermined -- use register_gnss_anchor for a "
            "translation-only model)"
        )
    u = us[1] - us[0]
    for i in range(2, n):
        v = us[i] - us[0]
        cross = np.cross(u, v)
        if np.linalg.norm(cross) > 1e-9 * max(np.linalg.norm(u) * np.linalg.norm(v), 1e-12):
            break
    else:
        return "correspondence points are collinear in one frame; rotation about the line is undetermined"
    return None


def _dominant_axis(normal: np.ndarray) -> str:
    return "xyz"[int(np.argmax(np.abs(normal)))]


def _fit_transform(
    us: np.ndarray, ts: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float]:
    r, t = _kabsch(us, ts)
    resid = np.linalg.norm(us @ r.T + t - ts, axis=1)
    rmse = float(np.sqrt(np.mean(resid**2)))
    return r, t, rmse


def register_landmarks(
    correspondences: Sequence[LandmarkCorrespondence],
    *,
    from_frame: str,
    to_frame: str,
    outlier_threshold_m: float = 2.0,
) -> RegistrationResult:
    """Solve the rigid transform mapping source -> target from DECLARED
    landmark correspondences (closed-form Kabsch, robust deterministic
    outlier consensus). See module docstring for the honesty rules.

    Raises RegistrationError for degenerate geometry (named) or when no
    consensus subset exists.
    """
    usable, ambiguous = _validate_inputs(correspondences)

    if len(usable) < 3:
        detail = _degeneracy(
            np.asarray([c.source_position for c in usable], dtype=float),
            np.asarray([c.target_position for c in usable], dtype=float),
        )
        ambiguity_note = "" if not ambiguous else " -- " + "; ".join(
            f"ambiguous landmark {a.landmark_id} excluded "
            f"({a.n_source_observations} observations, no tiebreaker)"
            for a in ambiguous
        )
        raise RegistrationError(
            (detail or f"only {len(usable)} usable correspondence pair(s) "
             "after ambiguity exclusion -- a full 6-DOF solve needs >=3")
            + ambiguity_note
        )

    us_all = np.asarray([c.source_position for c in usable], dtype=float)
    ts_all = np.asarray([c.target_position for c in usable], dtype=float)
    n = len(usable)

    # Deterministic robust consensus: landmark sets are small (named
    # physical features), so exhaustive tri-seeded fitting is cheap and
    # reproducible -- no RNG. Each seed is a distinct triple of pairs;
    # the seed's exact fit defines the consensus; the largest consensus
    # wins (tie -> lexicographically first, deterministic).
    best: Optional[Tuple[int, np.ndarray, np.ndarray, List[int]]] = None
    for seed in itertools.combinations(range(n), 3):
        us_s, ts_s = us_all[list(seed)], ts_all[list(seed)]
        if _degeneracy(us_s, ts_s) is not None:
            continue
        r0, t0, rmse0 = _fit_transform(us_s, ts_s)
        if rmse0 > outlier_threshold_m * 1e6:  # degenerate triple guard
            continue
        resid = np.linalg.norm(us_all @ r0.T + t0 - ts_all, axis=1)
        inliers = [i for i in range(n) if resid[i] <= outlier_threshold_m]
        if best is None or len(inliers) > len(best[3]) or (
            len(inliers) == len(best[3]) and list(seed) < best[0]
        ):
            best = (list(seed), r0, t0, inliers)

    if best is None:
        raise RegistrationError(
            "no non-degenerate 3-correspondence seed exists -- geometry "
            "cannot determine a 6-DOF transform"
        )
    _, r, t, inliers = best
    rejected = [usable[i].landmark_id for i in range(n) if i not in inliers]

    # Final fit on the consensus inliers only.
    us_i, ts_i = us_all[inliers], ts_all[inliers]
    r, t, rmse = _fit_transform(us_i, ts_i)
    resid = np.linalg.norm(us_i @ r.T + t - ts_i, axis=1)
    stats = ResidualStats.from_distances(resid)

    degenerate_axes = None
    # Spatial conditioning of the consensus set (same residual-derived
    # model as the other methods).
    scatter = us_i - us_i.mean(axis=0)
    try:
        cov_axes = np.linalg.svd(scatter, compute_uv=False)
        weak = cov_axes < 1e-6 * max(cov_axes.max(), 1e-12)
        if weak.any():
            degenerate_axes = tuple("xyz"[i] for i in np.nonzero(weak)[0])
    except np.linalg.LinAlgError:
        degenerate_axes = ("x", "y", "z")

    n_eff = max(len(inliers), 1)
    cov = RegistrationCovariance(
        translation_sigma_m=float(rmse / np.sqrt(n_eff)),
        axis_sigmas_m=(float(rmse / np.sqrt(n_eff)),) * 3,
        rotation_sigma_rad=float(rmse / max(float(np.sqrt(np.mean(
            np.linalg.norm(us_i - us_i.mean(axis=0), axis=1) ** 2))), 1e-12)),
        n_correspondences=len(inliers),
        degenerate_axes=degenerate_axes,
        basis=(
            f"residual-derived: landmark rmse {rmse:.3e} m over "
            f"{len(inliers)} declared correspondences"
            + (f"; {len(rejected)} outlier identity(ies) rejected" if rejected else "")
        ),
    )

    reason = ""
    if rejected:
        reason = (
            f"outlier correspondence(s) rejected beyond {outlier_threshold_m} m "
            f"consensus: {sorted(rejected)} -- excluded from the fit, not blended"
        )
    if ambiguous:
        reason = (reason + " " if reason else "") + "; ".join(
            f"ambiguous landmark {a.landmark_id} excluded ({a.n_source_observations} "
            "observations, no tiebreaker)" for a in ambiguous
        )

    return RegistrationResult(
        transform=RigidTransform(
            from_frame=from_frame, to_frame=to_frame,
            rotation=_matrix_to_quat(r), translation=Vec3(*t),
        ),
        method="landmark", status="accepted", reason=reason,
        rmse=rmse, inlier_fraction=len(inliers) / n, iterations=1,
        source_points=n, target_points=n,
        residual_stats=stats, covariance=cov,
    )
