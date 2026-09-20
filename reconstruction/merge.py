"""Merging multiple SfM sub-models into one ReconstructionResult
(P1 fidelity coverage work).

COLMAP's mapper may emit MULTIPLE sub-models (sparse/0, sparse/1, ...)
when the imagery splits into components. Each sub-model is a rigidly
related reconstruction OF THE SAME SCENE, so the honest merge is a
REGISTRATION problem: recover the rigid transform mapping each
sub-model into the reference model's frame using the canonical
registration machinery (registration.cross_session.align_session),
with every refusal recorded and wrong-frame geometry EXCLUDED rather
than identity-placed or silently dropped.

Why the aligner and not a bespoke merge: the cross-session machinery
already implements the repo's registration discipline -- identity
verification, coarse rotation search, ICP polish, measured RMSE and
inlier fraction, refusal with reason when two clouds share no plausible
contact. A sub-model merge that bypasses it would create a second,
competing registration implementation.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from reconstruction.backend.interface import (
    ReconstructedCameraPose,
    ReconstructionResult,
    ReconstructedPoint,
)
from reconstruction.calibration.transforms import RigidTransform
from registration.cross_session import (
    CrossSessionReport,
    align_reconstructed_sessions,
)

#: A sub-model with fewer points than this CANNOT be verified as a
#: same-scene registration and is refused. Measured failure
#: (run_20260920T200832): a 3-point sub-model ICP-snapped onto the
#: 5,392-point reference with ~zero residual and passed every RMSE /
#: inlier gate (3 points always fit 3 nearest neighbors), and its one
#: camera then dragged the whole evaluation gauge. A degenerate cloud
#: fits ANY target, so no acceptance gate can certify it -- the honest
#: answer is refusal with reason. 8 points is far below any real
#: mapper component (real sub-models here: 5,392 points) and above the
#: degenerate regime.
MIN_MERGE_POINTS = 8


def _apply_transform_to_result(
    result: ReconstructionResult,
    transform: RigidTransform,
) -> ReconstructionResult:
    """Return a copy of `result` with poses and points mapped through
    `transform` (from_frame -> to_frame). The pose's camera CENTER is a
    point (transforms with translation); its rotation composes as
    R_to * R_from."""
    q = transform.rotation
    R_to = np.array([
        [1 - 2 * (q.y**2 + q.z**2), 2 * (q.x * q.y - q.z * q.w), 2 * (q.x * q.z + q.y * q.w)],
        [2 * (q.x * q.y + q.z * q.w), 1 - 2 * (q.x**2 + q.z**2), 2 * (q.y * q.z - q.x * q.w)],
        [2 * (q.x * q.z - q.y * q.w), 2 * (q.y * q.z + q.x * q.w), 1 - 2 * (q.x**2 + q.y**2)],
    ])
    t = np.array([transform.translation.x, transform.translation.y, transform.translation.z])

    poses = []
    for pose in result.camera_poses:
        c = np.array(pose.position)
        c2 = R_to @ c + t
        R_from = _quat_matrix(pose.rotation)
        R2 = R_to @ R_from
        q2 = _matrix_quat(R2)
        poses.append(ReconstructedCameraPose(
            evidence_id=pose.evidence_id,
            position=(float(c2[0]), float(c2[1]), float(c2[2])),
            rotation=q2,
            uncertainty=pose.uncertainty,
        ))

    points = []
    for point in result.points:
        p = np.array(point.position)
        p2 = R_to @ p + t
        points.append(ReconstructedPoint(
            position=(float(p2[0]), float(p2[1]), float(p2[2])),
            track_id=point.track_id,
            source_evidence_ids=point.source_evidence_ids,
            uncertainty=point.uncertainty,
        ))

    return ReconstructionResult(
        points=points, camera_poses=poses,
        registration_status=result.registration_status,
    )


def _quat_matrix(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _matrix_quat(R: np.ndarray) -> Tuple[float, float, float, float]:
    trace = R[0][0] + R[1][1] + R[2][2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2][1] - R[1][2]) * s
        y = (R[0][2] - R[2][0]) * s
        z = (R[1][0] - R[0][1]) * s
    elif R[0][0] > R[1][1] and R[0][0] > R[2][2]:
        s = 2.0 * np.sqrt(1.0 + R[0][0] - R[1][1] - R[2][2])
        w = (R[2][1] - R[1][2]) / s
        x = 0.25 * s
        y = (R[0][1] + R[1][0]) / s
        z = (R[0][2] + R[2][0]) / s
    elif R[1][1] > R[2][2]:
        s = 2.0 * np.sqrt(1.0 + R[1][1] - R[0][0] - R[2][2])
        w = (R[0][2] - R[2][0]) / s
        x = (R[0][1] + R[1][0]) / s
        y = 0.25 * s
        z = (R[1][2] + R[2][1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2][2] - R[0][0] - R[1][1])
        w = (R[1][0] - R[0][1]) / s
        x = (R[0][2] + R[2][0]) / s
        y = (R[1][2] + R[2][1]) / s
        z = 0.25 * s
    n = np.sqrt(w * w + x * x + y * y + z * z)
    return (float(w / n), float(x / n), float(y / n), float(z / n))


def merge_submodel_results(
    results: Dict[str, ReconstructionResult],
    reference: str,
) -> Tuple[ReconstructionResult, CrossSessionReport]:
    """Merge independently-reconstructed sub-models of the SAME scene
    into one ReconstructionResult in `reference`'s frame.

    Every sub-model that the canonical aligner REGISTERS is transformed
    into the reference frame and included. Every sub-model whose
    registration is REFUSED is EXCLUDED from the merged geometry and
    reported as unresolved with its reason -- wrong-frame data must
    never enter the scene (no identity placement), and nothing is
    silently dropped (the report carries the refusal).
    """
    if reference not in results:
        raise ValueError(f"reference sub-model {reference!r} not in results")
    # Degenerate sub-models are refused BEFORE registration: no ICP
    # gate can certify a cloud too small to constrain a transform
    # (measured: 3 points fit any target with ~zero residual). They
    # are excluded from geometry with a recorded reason -- never
    # identity-placed, never silently dropped.
    admitted = {
        name: r for name, r in results.items()
        if name == reference or len(r.points) >= MIN_MERGE_POINTS
    }
    # Sub-models of ONE mapper run reconstruct the same scene up to
    # per-model arbitrary gauges, so their RAW clouds can sit far apart
    # even though the scene is shared. The contact pre-check is relaxed
    # accordingly; every acceptance gate (ICP RMSE, inlier fraction,
    # scale sanity) remains active, so a wrong merge still refuses --
    # with the same-scene prior stated here rather than assumed away.
    report = align_reconstructed_sessions(
        admitted, reference, contact_margin_fraction=10.0,
    )

    merged_points: List[ReconstructedPoint] = []
    merged_poses: List[ReconstructedCameraPose] = []
    statuses = []
    for name in sorted(results):
        result = results[name]
        if name != reference and name not in admitted:
            statuses.append("partial")
            continue  # excluded; reason recorded below
        if result.registration_status == "failed" or not result.points:
            statuses.append(result.registration_status)
            continue  # aligner already recorded the skip reason
        if name == reference:
            merged_points.extend(result.points)
            merged_poses.extend(result.camera_poses)
            statuses.append(result.registration_status)
            continue
        if report.status_by_session.get(name) == "aligned" and report.transforms.get(name) is not None:
            moved = _apply_transform_to_result(result, report.transforms[name])
            merged_points.extend(moved.points)
            merged_poses.extend(moved.camera_poses)
            statuses.append(result.registration_status)
        else:
            # Refused/unresolved: excluded from geometry; the report
            # records why. Reflected in the merged status as partial.
            statuses.append("partial")
    for name in sorted(results):
        if name != reference and name not in admitted:
            report.status_by_session[name] = "unresolved"
            report.reasons[name] = (
                f"sub-model has {len(results[name].points)} points, below the "
                f"{MIN_MERGE_POINTS}-point verification minimum: a degenerate cloud "
                "ICP-fits any target with near-zero residual (measured: a "
                "3-point sub-model snapped onto a 5k-point reference), so it "
                "cannot be certified as a same-scene registration. Excluded "
                "from geometry; cameras and points are NOT silently dropped "
                "-- they are refused with this reason."
            )

    if not merged_points:
        status = "failed"
    elif all(s == "success" for s in statuses):
        status = "success"
    elif any(s == "unresolved" for s in statuses) or any(
        report.status_by_session.get(n) == "unresolved"
        for n in results if n != reference
    ):
        status = "partial"
    else:
        status = "partial" if any(s != "success" for s in statuses) else "success"

    return ReconstructionResult(
        points=merged_points,
        camera_poses=merged_poses,
        registration_status=status,
        # Diagnostics MUST survive the merge: the backend used to
        # discard this report, so a false merge left no trace.
        merge_report=report.to_dict() if len(results) > 1 else None,
    ), report
