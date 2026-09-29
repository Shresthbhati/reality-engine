"""Progressive reconstruction: evidence in -> the strongest valid world out.

The product contract is not "N images or nothing". Whatever evidence
exists is turned into the best spatial representation it honestly
supports, and the SAME world improves as evidence accumulates:

    LEVEL 0  single-image visual hypothesis        (engine.pipeline.single_image)
    LEVEL 1  multi-image rough reconstruction       (SfM registered only PART of the set)
    LEVEL 2  sparse photogrammetric reconstruction  (SfM registered every image)
    LEVEL 3  dense reconstruction                   (dense MVS contributed points; not yet wired)
    LEVEL 4  semantic / topological refinement      (rooms / storeys / corridors compiled)
    LEVEL 5  incremental evidence refinement        (built on top of earlier evidence)

Levels are cumulative capabilities ACHIEVED, never claimed. If a higher
level fails, the failure is recorded in ``attempts`` and the strongest
lower level that succeeded is returned -- a COLMAP failure never
degrades into an empty world. If nothing at all can be built the caller
gets an explicit exception with the reason, never a fabricated world.

Progressive refinement is by union: a new batch is reconstructed together
with every earlier image of the same world. compile(A+B) is therefore the
definition of V2, not an approximation of it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from evidence.contribution import ContributionReport, analyze_contribution, quality_levels
from evidence.image_check import ImageFacts
from evidence.session import EvidenceItem
from engine.pipeline.guidance import build_guidance, coverage_degrees
from engine.pipeline.single_image import BootstrapUnavailable, bootstrap_single_image, fuse_single_views

LEVEL_NAMES = {
    0: "single-image visual hypothesis",
    1: "multi-image rough reconstruction",
    2: "sparse photogrammetric reconstruction",
    3: "dense reconstruction",
    4: "semantic/topological refinement",
    5: "incremental evidence refinement",
}
_TOPOLOGY_TYPES = {"room", "storey", "corridor", "level"}


class InsufficientEvidence(RuntimeError):
    """No evidence that can honestly produce any spatial model."""

    failure_kind = "insufficient_evidence"
    retryable = False


class ReconstructionUnavailable(RuntimeError):
    """Evidence exists but every level failed / its dependencies are missing."""

    failure_kind = "reconstruction_unavailable"
    retryable = False


@dataclass(frozen=True)
class EvidenceInput:
    item: EvidenceItem
    facts: ImageFacts
    name: str
    path: Path


@dataclass(frozen=True)
class ProgressiveResult:
    world: object
    points: tuple
    camera_poses: tuple
    level: int
    level_name: str
    model_state: str                  # ROUGH | PARTIAL | REFINED
    registration_status: str
    scale_state: str
    meters_per_unit: Optional[float]
    stage_facts: Dict[str, object]
    attempts: List[dict]
    contribution: ContributionReport
    quality: List[dict]
    guidance: List[dict]
    degraded: List[str]
    input_ids: List[str]
    registered_ids: List[str]
    excluded: List[dict]              # invalid / non-geometry evidence, with reasons
    counts: Dict[str, int] = field(default_factory=dict)  # compile counts
    vs_result: Optional[object] = None


def _scene_center(points) -> Optional[tuple]:
    """Robust (median) horizontal centre of the reconstructed points in the
    canonical +Y-up frame; None when there are no points."""
    if not points:
        return None
    xs = sorted(p[0] for p in points)
    zs = sorted(p[2] for p in points)
    return (xs[len(xs) // 2], zs[len(zs) // 2])


def _roles(world) -> List[str]:
    return sorted({e.type.value for e in world.entities.values()
                   if e.type.value in ("wall", "floor", "ceiling")})


def run_progressive(
    inputs: Sequence[EvidenceInput],
    *,
    vs_options,
    prior_ids: Sequence[str] = (),
    artifact_store=None,
    depth_model: str = "DPT_Hybrid",
    vertical_slice_fn: Optional[Callable] = None,
    bootstrap_depth_map=None,
) -> ProgressiveResult:
    eligible = [i for i in inputs if i.facts.geometry_eligible]
    excluded = []
    for i in inputs:
        if not i.facts.ok:
            excluded.append({"evidence_id": i.item.id, "name": i.name, "kind": "invalid",
                             "reason": i.facts.reason})
        elif not i.facts.geometry_eligible:
            excluded.append({
                "evidence_id": i.item.id, "name": i.name, "kind": "context",
                "reason": (f"classified {i.facts.evidence_class} ({i.facts.class_basis}): kept as "
                           "context evidence, never mixed into observed geometry"),
                "evidence_class": i.facts.evidence_class,
            })
    if not eligible:
        raise InsufficientEvidence(
            "no photographic evidence that can build a spatial model: "
            + ("; ".join(f"{e['name']}: {e['reason']}" for e in excluded) if excluded else "no images")
        )

    names = {i.item.id: i.name for i in inputs}
    contribution = analyze_contribution([(i.item.id, i.path) for i in eligible], prior_ids)
    attempts: List[dict] = []
    vs = None

    if len(eligible) >= 2:
        if vertical_slice_fn is None:
            from engine.pipeline.vertical_slice import vertical_slice as vertical_slice_fn  # noqa: PLC0415
        try:
            vs = vertical_slice_fn([i.item for i in eligible], vs_options)
            attempts.append({"level": 2, "name": LEVEL_NAMES[2], "outcome": "succeeded",
                             "detail": f"{vs.cameras_registered}/{vs.cameras_input} cameras registered"})
        except Exception as exc:  # noqa: BLE001 -- recorded, then a lower level is tried
            attempts.append({"level": 2, "name": LEVEL_NAMES[2], "outcome": "failed",
                             "detail": f"{type(exc).__name__}: {exc}"[:400]})
            vs = None

    degraded: List[str] = []
    if excluded:
        degraded.append(f"{len(excluded)} evidence item(s) excluded from geometry")

    if vs is not None:
        world = vs.world
        registered_ids = [p[0] for p in vs.camera_poses]
        # Level follows how much was actually registered, not whether EVERY
        # image was: 2 cameras is a rough two-view result, >= 3 registered in
        # one sparse model is a photogrammetric reconstruction (measured:
        # 9 of 10 registered is a bigger model than 6 of 6, not a rougher one).
        # Unregistered images are recorded as degradation, never hidden.
        level = 2 if len(registered_ids) >= 3 else 1
        if vs.registration_status != "success":
            degraded.append(f"registration {vs.registration_status!r}: "
                            f"{vs.cameras_registered}/{vs.cameras_input} cameras")
        if level >= 2 and any(e.type.value in _TOPOLOGY_TYPES for e in world.entities.values()):
            level = 4
        if prior_ids and level >= 2:
            level = 5
        result_points, poses = vs.points, vs.camera_poses
        scale_state, mpu, stage_facts = vs.scale_state, vs.meters_per_unit, dict(vs.stage_facts)
        registration = vs.registration_status
        counts = {"entities": len(world.entities), "measurements": vs.compile.measurements_count,
                  "relationships": vs.compile.relationships_count}
        bootstrap_facts = None
    else:
        # Multi-view registration did not (or could not) run. Every usable
        # photograph still contributes: bootstrap each one and fuse the
        # hypotheses into ONE world. N images never silently become 1.
        prior = set(prior_ids)
        ordered = sorted(eligible, key=lambda c: c.item.id not in prior)  # earliest evidence first (stable)
        boots = []
        for k, cand in enumerate(ordered):
            try:
                boots.append((cand, bootstrap_single_image(
                    cand.item, cand.facts, artifact_store=artifact_store, depth_model=depth_model,
                    depth_map=bootstrap_depth_map, tag="" if len(ordered) == 1 else f"{k + 1}",
                )))
            except BootstrapUnavailable as exc:
                attempts.append({"level": 0, "name": LEVEL_NAMES[0], "outcome": "failed",
                                 "detail": f"{cand.name}: {exc}"})
        if not boots:
            raise ReconstructionUnavailable(
                "no reconstruction level could be produced: "
                + "; ".join(f"L{a['level']} {a['outcome']}: {a['detail']}" for a in attempts)
            )
        boot = boots[0][1] if len(boots) == 1 else fuse_single_views([(c.item, r) for c, r in boots])
        used = [c for c, _ in boots]
        attempts.append({"level": 0, "name": LEVEL_NAMES[0], "outcome": "succeeded",
                         "detail": (f"single-view hypothesis from {used[0].name}" if len(used) == 1 else
                                    f"{len(used)} independent single-view hypotheses fused (unregistered)")})
        world, level = boot.world, 0
        # only a lone image is anchored by its own assumed pose; fused views are unrelated frames
        registered_ids = [used[0].item.id] if len(used) == 1 else []
        result_points, poses = boot.points, boot.camera_poses
        scale_state, mpu, registration = "relative", None, "partial"
        fused = len(used) > 1
        stage_facts = {
            "backend": "single_image_bootstrap_fused" if fused else "single_image_bootstrap",
            "depth": {"status": "ran", "model": depth_model, "note": "monocular relative depth"},
            "perception": {"status": "skipped", "note": "not run in the single-image path"},
            "mesh": {"status": "skipped", "note": "not run in the single-image path"},
            "bootstrap": boot.facts,
        }
        counts = {"entities": len(world.entities), "measurements": 0, "relationships": 0}
        bootstrap_facts = boot.facts
        if len(eligible) == 1:
            degraded.append("single-view hypothesis: partial by construction "
                            "(relative scale, hidden surfaces unknown)")
        else:
            dropped = len(eligible) - len(used)
            degraded.append(
                f"multi-view reconstruction of {len(eligible)} images failed; {len(used)} photographs "
                "kept as independent single-view hypotheses (their relative placement is unknown)"
                + (f"; {dropped} could not be estimated" if dropped else ""))

    # REFINED needs topology/incremental refinement AND every image placed;
    # anything less honest stays PARTIAL. A rough (<= level 1) model is ROUGH.
    if level <= 1:
        model_state = "ROUGH"
    elif level >= 4 and registration == "success":
        model_state = "REFINED"
    else:
        model_state = "PARTIAL"
    center = _scene_center(result_points) if level >= 1 else None
    coverage_deg = coverage_degrees(poses, center) if level >= 1 else None
    quality = quality_levels(contribution, len(eligible), coverage_deg)
    if vs is None and len(eligible) >= 2:
        # Feature matches alone said the images relate, but the reconstruction
        # engine could not use them: never show that as good overlap.
        for q in quality:
            if q["name"] == "Overlap":
                q["level"] = "low"
                q["basis"] += ("; BUT the reconstruction engine found no usable image pair, so the "
                               "overlap is not sufficient for 3D")
    guidance = build_guidance(
        level=level, entity_roles=_roles(world), names=names,
        input_ids=[i.item.id for i in eligible], registered_ids=registered_ids,
        camera_poses=poses, contribution_summary=contribution.summary(),
        bootstrap_facts=bootstrap_facts, attempts=attempts, scene_center=center,
    )
    return ProgressiveResult(
        world=world, points=tuple(result_points), camera_poses=tuple(poses), level=level,
        level_name=LEVEL_NAMES[level], model_state=model_state, registration_status=registration,
        scale_state=scale_state, meters_per_unit=mpu, stage_facts=stage_facts, attempts=attempts,
        contribution=contribution, quality=quality, guidance=guidance, degraded=degraded,
        input_ids=[i.item.id for i in eligible], registered_ids=registered_ids, excluded=excluded,
        counts=counts, vs_result=vs,
    )
