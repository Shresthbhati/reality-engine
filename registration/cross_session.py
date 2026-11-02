"""Cross-session registration: aligning reconstructions from DIFFERENT
capture runs into one frame (reliability priority: cross-session
registration -- the fusion precondition for multi-session worlds).

Sessions reconstructed independently land in arbitrary frames. Nothing
here may assume identity alignment: a session either earns a REAL rigid
transform from shared structure, or it is refused with a recorded
reason and stays refused.

Pipeline per session pair (source = the session being aligned, target =
the reference session's geometry):

  1. CONTACT GATE -- expanded bounding volumes must intersect. If they
     do not, no physical surface can be shared and registration is
     refused before any ICP can hallucinate a plausible-looking local
     minimum (a uniformly-offset cloud is internally self-consistent
     and can pass relative inlier gates). The gate is conservative:
     failing it proves no contact; passing it does not prove contact
     -- the ICP acceptance gates remain the sufficient check.
  2. COARSE GLOBAL ALIGNMENT -- centroid alignment plus a scored search
     over the 60 rotations of the icosahedral group (deterministic,
     near-uniform coverage of SO(3), no RNG). Plain ICP from identity
     is NOT a global method: on real cross-session offsets it converges
     to local minima (measured on this module's own fixtures).
  3. ICP REFINEMENT -- the top coarse candidates are refined with the
     repository's existing register_icp (outlier rejection, overlap and
     absolute-scale acceptance gates), first on a bounded downsample,
     then one polish pass on the full clouds. The best accepted result
     wins; every rejection is explained.

`align_session_chain` composes accepted pair alignments from arbitrary
sessions into the reference frame (BFS over accepted anchors); sessions
with no accepted path are reported "unresolved", never identity-placed.

All fixture-level behavior is deterministic; real-data runs are recorded
as real in the ledgers, never fabricated from synthetic numbers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree

from engine.math import Vec3
from reconstruction.backend.interface import ReconstructionResult
from reconstruction.calibration.camera import quat_to_matrix
from reconstruction.calibration.transforms import RigidTransform
from registration.registration import register_icp

#: Bounding volumes are expanded by this fraction of the combined scene
#: diagonal before the contact test (margin for partially-overlapping
#: captures of one structure).
DEFAULT_CONTACT_MARGIN_FRACTION = 0.15
#: Identity is "verified" (not searched) when the source already sits on
#: the target within this fraction of the target's own point spacing.
DEFAULT_IDENTITY_TOLERANCE_FRACTION = 1e-3
#: Coarse-stage cap per cloud (deterministic stride downsample).
DEFAULT_COARSE_MAX_POINTS = 2048
#: How many of the scored coarse candidates get a full ICP refinement.
DEFAULT_TOP_K_STARTS = 3


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SessionAlignment:
    """One session-pair alignment attempt: a real transform or an
    explained refusal. `transform` is None iff `status == "refused"`."""

    from_session: str
    to_session: str
    status: str  # "accepted" | "refused"
    method: str  # "identity_verified" | "icp" | "none"
    transform: Optional[RigidTransform]
    rmse: float
    inlier_fraction: float
    reason: str
    source_points: int
    target_points: int

    def to_dict(self) -> dict:
        return {
            "from_session": self.from_session,
            "to_session": self.to_session,
            "status": self.status,
            "method": self.method,
            "transform": self.transform.to_dict() if self.transform else None,
            "rmse": self.rmse,
            "inlier_fraction": self.inlier_fraction,
            "reason": self.reason,
            "source_points": self.source_points,
            "target_points": self.target_points,
        }


@dataclass(frozen=True)
class CrossSessionReport:
    """Chain-level outcome: every session in the chain resolved into the
    reference frame, or labeled unresolved with its reason."""

    reference_session: str
    transforms: Dict[str, Optional[RigidTransform]] = field(default_factory=dict)
    status_by_session: Dict[str, str] = field(default_factory=dict)
    reasons: Dict[str, str] = field(default_factory=dict)
    #: Pose-graph refinement record (refine_session_chain): None = the transforms are the plain BFS composition.
    #: When set it always carries the INITIAL (pre-optimisation) transforms next to the optimised ones.
    refinement: Optional[dict] = None

    def to_dict(self) -> dict:
        d = {
            "reference_session": self.reference_session,
            "transforms": {
                name: (t.to_dict() if t else None)
                for name, t in self.transforms.items()
            },
            "status_by_session": dict(self.status_by_session),
            "reasons": dict(self.reasons),
        }
        if self.refinement is not None:
            d["refinement"] = self.refinement
        return d


# ---------------------------------------------------------------------------
# Coarse alignment: the 60 icosahedral rotations (deterministic)
# ---------------------------------------------------------------------------

_PHI = (1.0 + math.sqrt(5.0)) / 2.0


def _axis_rotation(axis: Sequence[float], angle: float) -> np.ndarray:
    """Rodrigues rotation matrix about `axis` (need not be normalized)."""
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    K = np.array(
        [[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]]
    )
    return np.eye(3) + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)


def _icosahedral_rotations() -> List[np.ndarray]:
    """The 60 rotations of the icosahedral rotation group, in a
    deterministic order (closure of one 5-fold and one 2-fold generator,
    fixed iteration order). Near-uniform SO(3) coverage: some element is
    within ~44 degrees of any true rotation, which is what makes the
    coarse stage a real global search rather than an identity assumption."""
    g5 = _axis_rotation((0.0, 1.0, _PHI), math.radians(72.0))
    # 2-fold axis through the midpoint of icosahedron edge
    # (0, 1, phi) -- (1, phi, 0) (edge length 2 in this construction).
    g2 = _axis_rotation((0.5, (1.0 + _PHI) / 2.0, _PHI / 2.0), math.pi)
    elements = [np.eye(3)]
    changed = True
    while changed and len(elements) < 60:
        changed = False
        for existing in list(elements):
            for generator in (g5, g2):
                candidate = generator @ existing
                if not any(
                    np.allclose(candidate, other, atol=1e-9) for other in elements
                ):
                    elements.append(candidate)
                    changed = True
    return elements


_COARSE_ROTATIONS: Optional[List[np.ndarray]] = None


def _coarse_rotations() -> List[np.ndarray]:
    global _COARSE_ROTATIONS
    if _COARSE_ROTATIONS is None:
        _COARSE_ROTATIONS = _icosahedral_rotations()
    return _COARSE_ROTATIONS


def _euler_rotation(a_x: float, a_y: float, a_z: float) -> np.ndarray:
    """Rz(a_z) @ Ry(a_y) @ Rx(a_x), deterministic composition."""
    cx, sx = math.cos(a_x), math.sin(a_x)
    cy, sy = math.cos(a_y), math.sin(a_y)
    cz, sz = math.cos(a_z), math.sin(a_z)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=float)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=float)
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=float)
    return rz @ ry @ rx


def _refine_rotation(
    r0: np.ndarray,
    src: np.ndarray,
    tree: cKDTree,
    c_src: np.ndarray,
    c_tgt: np.ndarray,
    levels: int = 2,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Deterministic coarse-to-fine local rotation descent (Go-ICP
    style, bounded): at each level, score every Euler offset on a fixed
    grid around the current best, then halve the step. Pure enumeration
    -- no RNG, no stochastic restarts -- so results are reproducible.
    Returns (rotation, translation, rmse) with translation re-centered."""
    best_r = r0
    best_t = c_tgt - r0 @ c_src
    best_rmse = float(np.sqrt(np.mean(tree.query(src @ best_r.T + best_t)[0] ** 2)))
    step = math.radians(15.0)
    for _ in range(levels):
        offsets = [
            (dx, dy, dz)
            for dx in (-2, -1, 0, 1, 2)
            for dy in (-2, -1, 0, 1, 2)
            for dz in (-2, -1, 0, 1, 2)
        ]
        for dx, dy, dz in offsets:
            r = _euler_rotation(dx * step, dy * step, dz * step) @ best_r
            t = c_tgt - r @ c_src
            rmse = float(np.sqrt(np.mean(tree.query(src @ r.T + t)[0] ** 2)))
            if rmse < best_rmse - 1e-12:
                best_rmse, best_r, best_t = rmse, r, t
        step /= 2.0
    return best_r, best_t, best_rmse


def _stride_sample(points: Sequence[Vec3], max_points: int) -> List[Vec3]:
    if len(points) <= max_points:
        return list(points)
    stride = int(math.ceil(len(points) / max_points))
    return list(points[::stride])


def _bbox(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return points.min(axis=0), points.max(axis=0)


def _volumes_overlap(
    src: np.ndarray, tgt: np.ndarray, margin_fraction: float
) -> bool:
    """Do the bounding volumes, expanded by `margin_fraction` of the
    combined diagonal, intersect on every axis?"""
    smin, smax = _bbox(src)
    tmin, tmax = _bbox(tgt)
    combined = max(smax.max(), tmax.max()) - min(smin.min(), tmin.min())
    margin = margin_fraction * max(combined, 1e-12)
    return bool(
        np.all(smin - margin <= tmax + margin) and np.all(tmin - margin <= smax + margin)
    )


def _point_spacing(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    tree = cKDTree(points)
    self_nn = tree.query(points, k=2)[0][:, 1]
    return float(np.median(self_nn))


# ---------------------------------------------------------------------------
# Pair alignment
# ---------------------------------------------------------------------------


def align_session(
    source: Sequence[Vec3],
    target: Sequence[Vec3],
    from_session: str,
    to_session: str,
    *,
    contact_margin_fraction: float = DEFAULT_CONTACT_MARGIN_FRACTION,
    identity_tolerance_fraction: float = DEFAULT_IDENTITY_TOLERANCE_FRACTION,
    coarse_max_points: int = DEFAULT_COARSE_MAX_POINTS,
    top_k_starts: int = DEFAULT_TOP_K_STARTS,
    min_overlap: float = 0.5,
    max_rmse_scale: float = 5.0,
) -> SessionAlignment:
    """Recover the rigid transform mapping `source` (session
    `from_session`) into `target`'s frame (session `to_session`), or
    refuse with a recorded reason. See module docstring for the
    pipeline; no code path assumes identity alignment without evidence.
    """
    n_src, n_tgt = len(source), len(target)
    if n_src < 3 or n_tgt < 3:
        return SessionAlignment(
            from_session=from_session, to_session=to_session,
            status="refused", method="none", transform=None, rmse=float("nan"),
            inlier_fraction=0.0,
            reason=f"degenerate geometry: need >=3 points per session, got "
                   f"{n_src} source / {n_tgt} target",
            source_points=n_src, target_points=n_tgt,
        )

    src_full = np.asarray([[p.x, p.y, p.z] for p in source], dtype=float)
    tgt_full = np.asarray([[p.x, p.y, p.z] for p in target], dtype=float)

    if not _volumes_overlap(src_full, tgt_full, contact_margin_fraction):
        return SessionAlignment(
            from_session=from_session, to_session=to_session,
            status="refused", method="none", transform=None, rmse=float("nan"),
            inlier_fraction=0.0,
            reason="no plausible contact: bounding volumes expanded by "
                   f"{contact_margin_fraction:.0%} of the combined scene diagonal do "
                   "not intersect -- the sessions share no surface to register "
                   "against, so no transform is attempted",
            source_points=n_src, target_points=n_tgt,
        )

    # Identity verification: if the source ALREADY sits on the target
    # within a tight fraction of the target's own spacing, the honest
    # answer is "already aligned", evidenced by the measured rmse.
    tgt_tree_full = cKDTree(tgt_full)
    identity_rmse = float(
        np.sqrt(np.mean(tgt_tree_full.query(src_full)[0] ** 2))
    )
    spacing = _point_spacing(tgt_full)
    identity_tol = identity_tolerance_fraction * max(spacing, 1e-12)
    if identity_rmse <= identity_tol:
        transform = RigidTransform(
            from_frame=from_session, to_frame=to_session,
        )
        return SessionAlignment(
            from_session=from_session, to_session=to_session,
            status="accepted", method="identity_verified", transform=transform,
            rmse=identity_rmse, inlier_fraction=1.0,
            reason=f"source already sits on target within {identity_tol:.3g} m "
                   "(measured); identity transform verified, not assumed",
            source_points=n_src, target_points=n_tgt,
        )

    # Coarse stage on a bounded downsample (deterministic stride).
    src_pts = np.asarray(
        [[p.x, p.y, p.z] for p in _stride_sample(source, coarse_max_points)],
        dtype=float,
    )
    tgt_pts = np.asarray(
        [[p.x, p.y, p.z] for p in _stride_sample(target, coarse_max_points)],
        dtype=float,
    )
    c_src = src_pts.mean(axis=0)
    c_tgt = tgt_pts.mean(axis=0)
    tgt_tree = cKDTree(tgt_pts)

    scored: List[Tuple[float, int, np.ndarray, np.ndarray]] = []
    for idx, rotation in enumerate(_coarse_rotations()):
        translation = c_tgt - rotation @ c_src
        transformed = src_pts @ rotation.T + translation
        distances = tgt_tree.query(transformed)[0]
        rmse = float(np.sqrt(np.mean(distances**2)))
        scored.append((rmse, idx, rotation, translation))
    scored.sort(key=lambda item: (item[0], item[1]))

    # Coarse-to-fine descent from the best group element: the discrete
    # group cannot contain the true rotation (its elements are ~44 deg
    # apart at worst), so a bounded deterministic refinement closes the
    # gap before ICP takes over.
    _, best_idx, r0, _ = scored[0]
    r_ref, t_ref, rmse_ref = _refine_rotation(
        r0, src_pts, tgt_tree, c_src, c_tgt
    )
    starts: List[Tuple[float, np.ndarray, np.ndarray]] = [
        (rmse_ref, r_ref, t_ref)
    ]
    # Keep the runner-up group elements as ICP starts too (multimodal
    # scenes can have several plausible basins).
    for rmse0, idx, rotation, translation in scored[1 : max(1, top_k_starts)]:
        starts.append((rmse0, rotation, translation))

    best: Optional[SessionAlignment] = None
    best_block_reason = ""
    for rmse0, rotation, translation in starts:
        coarse_result = register_icp(
            _stride_sample(source, coarse_max_points),
            _stride_sample(target, coarse_max_points),
            from_frame=from_session, to_frame=to_session,
            initial_rotation=rotation, initial_translation=translation,
            min_overlap=min_overlap, max_rmse_scale=max_rmse_scale,
        )
        if coarse_result.status != "accepted" or coarse_result.transform is None:
            if coarse_result.reason and not best_block_reason:
                best_block_reason = coarse_result.reason
            continue
        # Polish pass on the FULL clouds (bounded coarse transform first):
        # final statistics describe the real point sets, not the downsample.
        if n_src > coarse_max_points or n_tgt > coarse_max_points:
            assert coarse_result.transform is not None
            r_init = np.asarray(
                quat_to_matrix(coarse_result.transform.rotation), dtype=float
            )
            t_init = np.array([
                coarse_result.transform.translation.x,
                coarse_result.transform.translation.y,
                coarse_result.transform.translation.z,
            ], dtype=float)
            final_result = register_icp(
                source, target,
                from_frame=from_session, to_frame=to_session,
                initial_rotation=r_init, initial_translation=t_init,
                min_overlap=min_overlap, max_rmse_scale=max_rmse_scale,
            )
        else:
            final_result = coarse_result
        if final_result.status == "accepted" and final_result.transform is not None:
            candidate = SessionAlignment(
                from_session=from_session, to_session=to_session,
                status="accepted", method="icp", transform=final_result.transform,
                rmse=final_result.rmse, inlier_fraction=final_result.inlier_fraction,
                reason=f"global coarse alignment ({len(_coarse_rotations())} "
                       f"deterministic starts) + ICP refinement; rmse measured on "
                       f"{final_result.source_points}x{final_result.target_points} points",
                source_points=n_src, target_points=n_tgt,
            )
            if best is None or candidate.rmse < best.rmse:
                best = candidate

    if best is not None:
        return best
    return SessionAlignment(
        from_session=from_session, to_session=to_session,
        status="refused", method="icp", transform=None,
        rmse=float("nan"), inlier_fraction=0.0,
        reason=best_block_reason or
               "no coarse start produced an accepted registration under the "
               "overlap and scale gates",
        source_points=n_src, target_points=n_tgt,
    )


# ---------------------------------------------------------------------------
# Reconstruction-result integration (vertical slice over the backend
# contract: orchestrator outputs go in, aligned results come out)
# ---------------------------------------------------------------------------


def result_to_points(result: ReconstructionResult) -> List[Vec3]:
    """Extract a registration-ready point cloud from a ReconstructionResult."""
    return [Vec3(*p.position) for p in result.points]


def align_reconstructed_sessions(
    results: Dict[str, ReconstructionResult],
    reference_session: str,
    all_pairs: bool = False,
    refine: bool = False,
    **align_kwargs,
) -> CrossSessionReport:
    """Resolve independently-reconstructed sessions into one frame.

    ``all_pairs`` additionally registers every pair of non-reference sessions (O(n^2) alignments), which creates
    loop closures; ``refine`` then solves all accepted alignments jointly with the pose-graph optimiser
    (refine_session_chain) instead of trusting the first BFS path. Both default off: behaviour is unchanged.

    Takes ReconstructionResults straight from IReconstructionBackend runs
    (via the orchestrator), registers every session that produced real
    geometry into `reference_session`'s frame, and returns the chain
    report. Failed/empty results are skipped with a recorded reason (a
    session with no geometry cannot be registered -- that is data, not an
    error); sessions whose registration is refused stay "unresolved".
    Raises ValueError when the reference session is unknown.
    """
    if reference_session not in results:
        raise ValueError(
            f"reference session {reference_session!r} not among the supplied "
            f"results ({sorted(results)})"
        )
    clouds: Dict[str, List[Vec3]] = {}
    reasons: Dict[str, str] = {}
    usable: Dict[str, ReconstructionResult] = {}
    for name in sorted(results):
        result = results[name]
        if result.registration_status == "failed" or not result.points:
            reasons[name] = (
                "skipped: session produced no usable geometry "
                f"(registration_status={result.registration_status!r}, "
                f"{len(result.points)} points)"
            )
            continue
        clouds[name] = result_to_points(result)
        usable[name] = result

    if reference_session not in clouds:
        raise ValueError(
            f"reference session {reference_session!r} produced no usable "
            "geometry; there is no frame to align into"
        )

    anchors: Dict[Tuple[str, str], SessionAlignment] = {}
    for name in sorted(clouds):
        if name == reference_session:
            continue
        anchors[(name, reference_session)] = align_session(
            clouds[name], clouds[reference_session],
            from_session=name, to_session=reference_session,
            **align_kwargs,
        )

    if all_pairs:
        names = [n for n in sorted(clouds) if n != reference_session]
        for a_idx, a in enumerate(names):
            for b in names[a_idx + 1:]:
                anchors[(b, a)] = align_session(
                    clouds[b], clouds[a], from_session=b, to_session=a, **align_kwargs,
                )
    chain = [reference_session] + [n for n in results if n != reference_session]
    report = (refine_session_chain(anchors, chain) if refine else align_session_chain(anchors, chain))
    # Merge the skip reasons into the chain report: a skipped session is
    # "unresolved" with its SPECIFIC reason (why it produced no
    # geometry), not the generic no-path reason.
    for name, reason in reasons.items():
        report.status_by_session[name] = "unresolved"
        report.transforms[name] = None
        report.reasons[name] = reason
    return report


# ---------------------------------------------------------------------------
# Chain composition
# ---------------------------------------------------------------------------


def align_session_chain(
    anchors: Dict[Tuple[str, str], SessionAlignment],
    chain: Sequence[str],
) -> CrossSessionReport:
    """Resolve every session in `chain` into `chain[0]`'s frame by
    composing accepted pair alignments. Sessions without an accepted
    registration path stay "unresolved" -- never identity-placed.

    Raises ValueError on an empty chain (there is no reference to
    resolve into; silent identity would be a lie).
    """
    if not chain:
        raise ValueError(
            "align_session_chain requires a non-empty chain: the first session "
            "is the reference frame"
        )
    reference = chain[0]
    transforms: Dict[str, Optional[RigidTransform]] = {reference: None}
    status_by_session: Dict[str, str] = {reference: "reference"}
    reasons: Dict[str, str] = {reference: "reference frame; all transforms map into this session's frame"}

    # Accepted edges: from_session -> to_session with a real transform.
    edges: Dict[str, List[Tuple[str, RigidTransform, SessionAlignment]]] = {}
    for pair_report in anchors.values():
        if (
            pair_report.status == "accepted"
            and pair_report.transform is not None
            and pair_report.from_session != pair_report.to_session
        ):
            edges.setdefault(pair_report.from_session, []).append(
                (pair_report.to_session, pair_report.transform, pair_report)
            )

    # Resolve outward from the reference: a pair transform maps
    # from_session -> to_session, so a session becomes resolvable when
    # the session it aligns INTO is already resolved. Iterate to a fixed
    # point (multi-level chains resolve through intermediaries).
    resolved = {reference}
    progress = True
    while progress:
        progress = False
        for from_session in sorted(edges):
            if from_session in resolved:
                continue
            for to_session, transform, pair_report in sorted(
                edges[from_session], key=lambda item: item[0]
            ):
                if to_session not in resolved:
                    continue
                total = (
                    transform
                    if to_session == reference
                    else transform.compose(transforms[to_session])  # type: ignore[arg-type]
                )
                transforms[from_session] = total
                status_by_session[from_session] = "aligned"
                via = "directly" if to_session == reference else f"via {to_session}"
                reasons[from_session] = (
                    f"aligned {via} into {to_session} "
                    f"(rmse={pair_report.rmse:.4g}, method={pair_report.method}); "
                    "transform maps into the reference frame by composition"
                )
                resolved.add(from_session)
                progress = True
                break

    for session in chain:
        if session not in status_by_session:
            status_by_session[session] = "unresolved"
            transforms[session] = None
            reasons[session] = (
                f"session {session!r}: no accepted registration connects it to "
                "the reference -- refusing to assume identity alignment"
            )

    return CrossSessionReport(
        reference_session=reference,
        transforms=transforms,
        status_by_session=status_by_session,
        reasons=reasons,
    )


# ---------------------------------------------------------------------------
# Pose-graph refinement
# ---------------------------------------------------------------------------

#: Rotation sigma is not measured by ICP; it is derived as sigma_t / characteristic_length (rad per metre of
#: lever arm). A heuristic, recorded in the refinement record, and overridable.
DEFAULT_CHARACTERISTIC_LENGTH_M = 10.0
MIN_SIGMA_T_M = 1e-3
MIN_SIGMA_R_RAD = 1e-4


def _to_pose(t: Optional[RigidTransform]):
    from registration.pose_graph import Pose
    if t is None:
        return Pose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    r = t.rotation
    return Pose((t.translation.x, t.translation.y, t.translation.z), (r.x, r.y, r.z, r.w))


def _from_pose(pose, from_frame: str, to_frame: str) -> RigidTransform:
    from engine.math import Quat
    x, y, z, w = pose.q
    return RigidTransform(from_frame=from_frame, to_frame=to_frame, rotation=Quat(w=w, x=x, y=y, z=z).normalized(),
                          translation=Vec3(*pose.t))


def refine_session_chain(
    anchors: Dict[Tuple[str, str], SessionAlignment],
    chain: Sequence[str],
    characteristic_length_m: float = DEFAULT_CHARACTERISTIC_LENGTH_M,
) -> CrossSessionReport:
    """align_session_chain, then a joint pose-graph solve over ALL accepted pairwise alignments.

    The BFS composition uses one path per session and ignores every other accepted alignment; when sessions form
    loops (A->B, B->C and C->A), the accumulated ICP error is never reconciled. Here each accepted alignment is an
    edge (rmse -> sigma_t), the BFS result is the initial guess, and the reference session stays at identity.
    The report keeps the BFS transforms under ``refinement["initial_transforms"]`` (optimised estimates never
    replace the evidence of how they were obtained). Falls back to the plain BFS report -- with the reason recorded
    in ``refinement`` -- when there is no loop to exploit or the graph cannot be solved."""
    from dataclasses import replace
    from scipy.spatial.transform import Rotation
    from registration.pose_graph import Edge, Pose, PoseGraphError, optimize_pose_graph

    initial = align_session_chain(anchors, chain)
    ref = initial.reference_session
    resolved = [s for s in chain if initial.status_by_session.get(s) in ("reference", "aligned")]
    edges, meta = [], {}
    for (frm, to), a in sorted(anchors.items()):
        if a.status != "accepted" or a.transform is None or frm == to or frm not in resolved or to not in resolved:
            continue
        sigma_t = max(float(a.rmse), MIN_SIGMA_T_M)
        sigma_r = max(sigma_t / characteristic_length_m, MIN_SIGMA_R_RAD)
        eid = f"{frm}->{to}"
        # transform X maps frm-coordinates into to-coordinates, i.e. the pose of `frm` seen from `to`
        e = _to_pose(a.transform)
        edges.append(Edge(eid, to, frm, e.t, e.q, sigma_t=sigma_t, sigma_r=sigma_r))
        meta[eid] = {"id": eid, "method": a.method, "rmse": a.rmse, "sigma_t": sigma_t, "sigma_r": sigma_r}
    loops = len(edges) - (len(resolved) - 1)
    if len(resolved) < 2 or loops <= 0:
        return replace(initial, refinement={"status": "skipped", "backend": None, "loop_closures": max(loops, 0),
                                            "reason": "no loop closures: the BFS composition is already the only "
                                                      "consistent solution, nothing to reconcile"})
    poses = {s: _to_pose(initial.transforms.get(s)) for s in resolved}
    try:
        res = optimize_pose_graph(poses, edges)
    except PoseGraphError as exc:
        return replace(initial, refinement={"status": "failed", "backend": None, "reason": str(exc),
                                            "loop_closures": loops})

    if res.ambiguous:
        return replace(initial, refinement={
            "status": "ambiguous", "backend": res.backend, "loop_closures": loops, "ambiguous_edges": res.ambiguous,
            "reason": "alignments are mutually inconsistent but the offender cannot be identified from this graph "
                      f"(removing any of {res.ambiguous} would restore consistency); kept the BFS composition. "
                      "Add independent alignments (more overlapping sessions) or review these pairs.",
            "initial_transforms": {s: (t.to_dict() if t else None) for s, t in initial.transforms.items()}})

    # Pose graphs fix a gauge, not necessarily the reference: re-express everything relative to the reference.
    r0 = res.poses[ref]
    inv = Rotation.from_quat(r0.q).inv()
    transforms = dict(initial.transforms)
    reasons = dict(initial.reasons)
    shifts = {}
    for s in resolved:
        if s == ref:
            continue
        p = res.poses[s]
        rel = Pose(tuple(float(v) for v in inv.apply(np.array(p.t) - np.array(r0.t))),
                   tuple(float(v) for v in (inv * Rotation.from_quat(p.q)).as_quat()))
        old = initial.transforms[s]
        transforms[s] = _from_pose(rel, old.from_frame, old.to_frame)
        shifts[s] = math.dist(tuple(rel.t), (old.translation.x, old.translation.y, old.translation.z))
        reasons[s] = (f"{initial.reasons[s]}; refined by pose graph over {len(edges)} alignments "
                      f"({loops} loop closure(s), moved {shifts[s]:.4g} m)")
    rejected = {r["id"]: r for r in res.rejected_edges}
    return replace(initial, transforms=transforms, reasons=reasons, refinement={
        "status": "refined", "backend": res.backend, "provenance": "registration:pose_graph",
        "converged": res.converged, "iterations": res.iterations,
        "initial_cost": res.initial_cost, "final_cost": res.final_cost, "loop_closures": loops,
        "characteristic_length_m": characteristic_length_m,
        "sigma_basis": "sigma_t = max(rmse, 1e-3 m); sigma_r = sigma_t / characteristic_length_m (heuristic)",
        "edges": [dict(meta[e.id], chi2=res.edge_chi2.get(e.id), rejected=e.id in rejected) for e in edges],
        "rejected_edges": res.rejected_edges, "warnings": res.warnings,
        "max_shift_m": max(shifts.values(), default=0.0),
        "initial_transforms": {s: (t.to_dict() if t else None) for s, t in initial.transforms.items()},
    })


def record_cross_session_registration(world, report: CrossSessionReport) -> dict:
    """Store the cross-session registration (and how it was obtained) on a WorldIR, at
    ``world.metadata["registration"]["cross_session"]`` (the slot world_ir.frame_graph.build_frame_graph_from_world
    already reserves for registration data).

    WorldIR's Frame enum has no per-session frames, so the transforms cannot be frame-graph edges; the metadata
    dict is serialised verbatim.
    Provenance is explicit: "registration:pose_graph" only when a refinement actually ran, otherwise
    "registration:chain" (plain BFS composition). Unresolved sessions are recorded as such, never as identity."""
    refined = bool(report.refinement and report.refinement.get("status") == "refined")
    record = {
        "schema": 1,
        "provenance": "registration:pose_graph" if refined else "registration:chain",
        "report": report.to_dict(),
    }
    world.metadata.setdefault("registration", {})["cross_session"] = record
    return record
