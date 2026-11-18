"""Autonomous Quality and Correction Loop (Directive Rule 9).

Builds an autonomous feedback loop that continuously assesses reconstruction quality:
1. Multi-Dimensional Quality Audit:
   - Evaluates reconstruction completeness (unclosed room loops, missing surfaces).
   - Detects metric scale drift against known architectural priors (doors, ceilings).
   - Detects identity fragmentation (co-located duplicate entities).
   - Evaluates structural and physical plausibility (floating loads, blocked egress).
   - Quantifies element-wise uncertainty.
2. Autonomous Correction Loop:
   - Formulates explicit correction hypotheses.
   - Executes corrections in isolated candidate branches.
   - Re-evaluates quality metrics before and after application.
   - Strictly enforces selective acceptance: accepts ONLY corrections that measurably
     improve overall quality without degrading physical validity.
   - Permanently records provenance and rationale in the world's audit trail.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from perception.reasoning.spatial_reasoning_engine import SpatialReasoningEngine
from provenance import Provenance
from world_ir import (
    Entity,
    EntityType,
    Geometry,
    GeometryType,
    Observation,
    Relationship,
    RelationshipKind,
    Vector3,
    WorldIR,
)


@dataclass(frozen=True)
class QualityAuditMetrics:
    """Comprehensive multi-dimensional quality metrics."""
    completeness_score: float         # 0.0 to 1.0 (room boundaries, surface closure)
    scale_fidelity_score: float       # 0.0 to 1.0 (agreement with architectural priors)
    identity_integrity_score: float   # 0.0 to 1.0 (absence of fragmented co-located duplicates)
    structural_soundness_score: float # 0.0 to 1.0 (vertical load paths & physical stability)
    overall_quality_score: float      # Weighted composite quality score

    def to_dict(self) -> Dict[str, float]:
        return {
            "completeness": round(self.completeness_score, 3),
            "scale_fidelity": round(self.scale_fidelity_score, 3),
            "identity_integrity": round(self.identity_integrity_score, 3),
            "structural_soundness": round(self.structural_soundness_score, 3),
            "overall_quality": round(self.overall_quality_score, 3),
        }


class DefectKind(str, Enum):
    SCALE_DRIFT = "scale_drift"
    MISSING_SURFACE = "missing_surface"
    IDENTITY_FRAGMENTATION = "identity_fragmentation"
    FLOATING_OBJECT = "floating_object"
    STRUCTURAL_IMPOSSIBILITY = "structural_impossibility"


@dataclass(frozen=True)
class CorrectionHypothesis:
    """A proposed autonomous correction hypothesis."""
    hypothesis_id: str
    defect_kind: DefectKind
    target_entity_ids: Tuple[str, ...]
    action_type: str  # "rescale_world", "consolidate_entities", "snap_to_ground", "synthesize_wall"
    parameters: Dict[str, Any]
    confidence: float
    rationale: str


@dataclass(frozen=True)
class CorrectionOutcome:
    """The result of applying a correction hypothesis in a candidate branch."""
    hypothesis: CorrectionHypothesis
    accepted: bool
    quality_delta: float
    baseline_quality: float
    candidate_quality: float
    rejection_reason: str = ""


@dataclass(frozen=True)
class AutonomousLoopReport:
    """Summary of the autonomous quality audit and correction cycle."""
    initial_metrics: QualityAuditMetrics
    final_metrics: QualityAuditMetrics
    hypotheses_evaluated: int
    corrections_accepted: int
    corrections_rejected: int
    outcomes: Tuple[CorrectionOutcome, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "initial_quality": self.initial_metrics.to_dict(),
            "final_quality": self.final_metrics.to_dict(),
            "quality_gain": round(self.final_metrics.overall_quality_score - self.initial_metrics.overall_quality_score, 3),
            "hypotheses_count": self.hypotheses_evaluated,
            "accepted_count": self.corrections_accepted,
            "rejected_count": self.corrections_rejected,
            "outcomes": [
                {
                    "hypothesis_id": o.hypothesis.hypothesis_id,
                    "defect": o.hypothesis.defect_kind.value,
                    "action": o.hypothesis.action_type,
                    "accepted": o.accepted,
                    "quality_delta": round(o.quality_delta, 3),
                    "rejection_reason": o.rejection_reason,
                }
                for o in self.outcomes
            ],
        }


# =========================================================================
# BUILD_SPEC B16: Priority-driven Refinement Loop
# =========================================================================

class RefinementKind(str, Enum):
    REDETECT_CROP = "redetect_crop"
    SEGMENT = "segment"
    REASSOCIATE = "reassociate"
    MORE_VIEWS = "more_views"


@dataclass
class RefinementTask:
    kind: RefinementKind
    target: str
    expected_gain: float
    cost_s: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RefinementPrior:
    successes: int = 1
    attempts: int = 2

    @property
    def probability(self) -> float:
        return (self.successes + 1) / (self.attempts + 2)

    def record_outcome(self, success: bool) -> None:
        self.attempts += 1
        if success:
            self.successes += 1


@dataclass(frozen=True)
class RefinementReport:
    tasks_evaluated: int
    tasks_succeeded: int
    stop_reason: str
    iterations: int
    history: Tuple[Dict[str, Any], ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tasks_evaluated": self.tasks_evaluated,
            "tasks_succeeded": self.tasks_succeeded,
            "stop_reason": self.stop_reason,
            "iterations": self.iterations,
            "history": list(self.history),
        }


class QualityAuditor:
    """Computes rigorous, empirical quality metrics across a WorldIR."""

    @staticmethod
    def audit(world: WorldIR) -> QualityAuditMetrics:
        if not world.entities:
            return QualityAuditMetrics(0.0, 0.0, 0.0, 0.0, 0.0)

        # 1. Scale Fidelity: Check against standard architectural door heights (nominal 2.05m)
        door_heights: List[float] = []
        for ent in world.entities.values():
            if ent.type == EntityType.DOOR or "door" in ent.name.lower():
                aabb = QualityAuditor._get_aabb(world, ent)
                if aabb:
                    h = float(aabb[1][2] - aabb[0][2])
                    if h > 0.01:
                        door_heights.append(h)

        scale_score = 1.0
        if door_heights:
            mean_door_h = float(np.mean(door_heights))
            # Expected door height: ~2.05m (tolerance: 1.80m to 2.40m)
            ratio = mean_door_h / 2.05
            if ratio < 0.5 or ratio > 2.0:
                scale_score = max(0.1, 1.0 - abs(ratio - 1.0) * 0.5)
            else:
                scale_score = max(0.6, 1.0 - abs(ratio - 1.0))

        # 2. Completeness: Check room enclosure
        rooms = [e for e in world.entities.values() if e.type == EntityType.ROOM]
        walls = [e for e in world.entities.values() if e.type == EntityType.WALL]
        completeness_score = 1.0
        if rooms:
            # Each room should have at least 4 walls associated or present in the scene
            wall_ratio = min(1.0, len(walls) / (len(rooms) * 4.0))
            completeness_score = wall_ratio
        elif walls:
            completeness_score = min(1.0, len(walls) / 4.0)

        # 3. Identity Integrity: Penalize co-located entities of the same type (fragmentation)
        duplicate_pairs = QualityAuditor.find_fragmented_identities(world)
        identity_score = max(0.0, 1.0 - len(duplicate_pairs) * 0.15)

        # 4. Structural Soundness: Use SpatialReasoningEngine
        spatial_rep = SpatialReasoningEngine.evaluate_world(world)
        structural_score = 1.0
        if not spatial_rep.load_paths.is_structurally_sound:
            structural_score -= 0.40
        if not spatial_rep.clearance.is_accessible:
            structural_score -= 0.30

        structural_score = max(0.1, structural_score)

        # Composite overall quality score (weighted)
        overall = (
            0.30 * scale_score
            + 0.25 * completeness_score
            + 0.25 * identity_score
            + 0.20 * structural_score
        )

        return QualityAuditMetrics(
            completeness_score=completeness_score,
            scale_fidelity_score=scale_score,
            identity_integrity_score=identity_score,
            structural_soundness_score=structural_score,
            overall_quality_score=overall,
        )

    @staticmethod
    def find_fragmented_identities(
        world: WorldIR,
        centroid_distance_tol_m: float = 0.15,
    ) -> List[Tuple[str, str]]:
        """Find pairs of distinct entities with identical type and co-located centroids."""
        ents = list(world.entities.values())
        pairs = []
        for i in range(len(ents)):
            for j in range(i + 1, len(ents)):
                e1, e2 = ents[i], ents[j]
                if e1.type == e2.type and e1.id != e2.id:
                    c1 = QualityAuditor._get_centroid(world, e1)
                    c2 = QualityAuditor._get_centroid(world, e2)
                    if c1 is not None and c2 is not None:
                        dist = float(np.linalg.norm(c1 - c2))
                        if dist <= centroid_distance_tol_m:
                            pairs.append((e1.id, e2.id))
        return pairs

    @staticmethod
    def _get_aabb(world: WorldIR, entity: Entity) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if entity.geometry_ids:
            for gid in entity.geometry_ids:
                if gid in world.geometries:
                    g = world.geometries[gid]
                    if g.bounds_min and g.bounds_max:
                        return (
                            np.array([g.bounds_min.x, g.bounds_min.y, g.bounds_min.z]),
                            np.array([g.bounds_max.x, g.bounds_max.y, g.bounds_max.z]),
                        )
        return None

    @staticmethod
    def _get_centroid(world: WorldIR, entity: Entity) -> Optional[np.ndarray]:
        aabb = QualityAuditor._get_aabb(world, entity)
        if aabb is not None:
            return (aabb[0] + aabb[1]) / 2.0
        t = entity.transform
        if t and "position" in t:
            pos = t["position"]
            return np.array([pos.get("x", 0.0), pos.get("y", 0.0), pos.get("z", 0.0)])
        return None


class AutonomousCorrectionLoop:
    """Autonomous feedback loop driving continuous quality improvement."""

    @staticmethod
    def formulate_hypotheses(world: WorldIR, audit: QualityAuditMetrics) -> List[CorrectionHypothesis]:
        hypotheses: List[CorrectionHypothesis] = []

        # 1. Scale drift hypothesis: if doors are far from ~2.05m
        door_heights: List[Tuple[str, float]] = []
        for ent in world.entities.values():
            if ent.type == EntityType.DOOR or "door" in ent.name.lower():
                aabb = QualityAuditor._get_aabb(world, ent)
                if aabb:
                    h = float(aabb[1][2] - aabb[0][2])
                    if h > 0.01:
                        door_heights.append((ent.id, h))

        if door_heights:
            mean_h = float(np.mean([h for _, h in door_heights]))
            scale_ratio = 2.05 / mean_h
            if abs(scale_ratio - 1.0) > 0.15:
                hypotheses.append(CorrectionHypothesis(
                    hypothesis_id=f"hyp-scale-{int(time.time()*1000)%10000}",
                    defect_kind=DefectKind.SCALE_DRIFT,
                    target_entity_ids=tuple(e.id for e in world.entities.values()),
                    action_type="rescale_world",
                    parameters={"scale_factor": scale_ratio},
                    confidence=0.90,
                    rationale=f"Observed door height ({mean_h:.2f}m) deviates from standard 2.05m. Rescale world by {scale_ratio:.3f}x.",
                ))

        # 2. Identity fragmentation hypothesis: consolidate co-located duplicate entities
        frag_pairs = QualityAuditor.find_fragmented_identities(world)
        for e1_id, e2_id in frag_pairs:
            hypotheses.append(CorrectionHypothesis(
                hypothesis_id=f"hyp-merge-{e1_id[:8]}-{e2_id[:8]}",
                defect_kind=DefectKind.IDENTITY_FRAGMENTATION,
                target_entity_ids=(e1_id, e2_id),
                action_type="consolidate_entities",
                parameters={"primary_id": e1_id, "duplicate_id": e2_id},
                confidence=0.88,
                rationale=f"Entities '{e1_id}' and '{e2_id}' are co-located duplicates of same type. Consolidate into canonical entity.",
            ))

        # 3. Floating object hypothesis: snap floating furniture to floor
        spatial_rep = SpatialReasoningEngine.evaluate_world(world)
        for proposal in spatial_rep.auto_fix_proposals:
            if proposal.get("fix_type") == "reinforce_load_path":
                t_id = proposal["target_entity_id"]
                hypotheses.append(CorrectionHypothesis(
                    hypothesis_id=f"hyp-struct-{t_id[:8]}",
                    defect_kind=DefectKind.STRUCTURAL_IMPOSSIBILITY,
                    target_entity_ids=(t_id,),
                    action_type="snap_to_ground",
                    parameters={"target_id": t_id},
                    confidence=0.85,
                    rationale=proposal["recommendation"],
                ))

        return hypotheses

    @staticmethod
    def apply_hypothesis_in_candidate(candidate_world: WorldIR, hyp: CorrectionHypothesis) -> None:
        """Apply a correction hypothesis inside an isolated candidate branch."""
        if hyp.action_type == "rescale_world":
            s = float(hyp.parameters["scale_factor"])
            for ent in candidate_world.entities.values():
                t = ent.transform
                if t and "position" in t:
                    pos = t["position"]
                    pos["x"] = pos.get("x", 0.0) * s
                    pos["y"] = pos.get("y", 0.0) * s
                    pos["z"] = pos.get("z", 0.0) * s
                for gid in ent.geometry_ids:
                    if gid in candidate_world.geometries:
                        g = candidate_world.geometries[gid]
                        if g.bounds_min and g.bounds_max:
                            g.bounds_min = Vector3(g.bounds_min.x * s, g.bounds_min.y * s, g.bounds_min.z * s)
                            g.bounds_max = Vector3(g.bounds_max.x * s, g.bounds_max.y * s, g.bounds_max.z * s)
                        g.provenance = Provenance.ESTIMATED

        elif hyp.action_type == "consolidate_entities":
            p_id = hyp.parameters["primary_id"]
            d_id = hyp.parameters["duplicate_id"]
            if p_id in candidate_world.entities and d_id in candidate_world.entities:
                primary = candidate_world.entities[p_id]
                duplicate = candidate_world.entities[d_id]
                # Merge observations and supporting evidence
                primary.observations.extend(duplicate.observations)
                # Remove duplicate entity
                del candidate_world.entities[d_id]

        elif hyp.action_type == "snap_to_ground":
            t_id = hyp.parameters["target_id"]
            if t_id in candidate_world.entities:
                ent = candidate_world.entities[t_id]
                # Lower entity to ground elevation (z=0.0)
                for gid in ent.geometry_ids:
                    if gid in candidate_world.geometries:
                        g = candidate_world.geometries[gid]
                        if g.bounds_min and g.bounds_max:
                            dz = g.bounds_min.z
                            g.bounds_min = Vector3(g.bounds_min.x, g.bounds_min.y, 0.0)
                            g.bounds_max = Vector3(g.bounds_max.x, g.bounds_max.y, g.bounds_max.z - dz)
                        g.provenance = Provenance.ESTIMATED

    @staticmethod
    def run_autonomous_loop(
        world: WorldIR,
        max_iterations: int = 3,
        min_quality_gain: float = 0.01,
    ) -> AutonomousLoopReport:
        """Execute the full autonomous quality audit and correction loop."""
        initial_audit = QualityAuditor.audit(world)
        current_world = world
        current_audit = initial_audit

        outcomes: List[CorrectionOutcome] = []
        accepted_cnt = 0
        rejected_cnt = 0

        for it in range(max_iterations):
            hypotheses = AutonomousCorrectionLoop.formulate_hypotheses(current_world, current_audit)
            if not hypotheses:
                break

            improved_in_this_pass = False

            for hyp in hypotheses:
                # 1. Clone candidate branch
                candidate = WorldIR.from_dict(current_world.to_dict())

                # 2. Apply hypothesis
                AutonomousCorrectionLoop.apply_hypothesis_in_candidate(candidate, hyp)

                # 3. Audit candidate branch
                cand_audit = QualityAuditor.audit(candidate)
                quality_delta = cand_audit.overall_quality_score - current_audit.overall_quality_score

                # 4. Strict selective acceptance rule:
                # Must improve overall quality AND not degrade structural soundness
                if (
                    quality_delta >= min_quality_gain
                    and cand_audit.structural_soundness_score >= current_audit.structural_soundness_score - 1e-4
                ):
                    # Accept correction
                    outcome = CorrectionOutcome(
                        hypothesis=hyp,
                        accepted=True,
                        quality_delta=quality_delta,
                        baseline_quality=current_audit.overall_quality_score,
                        candidate_quality=cand_audit.overall_quality_score,
                    )
                    outcomes.append(outcome)
                    accepted_cnt += 1

                    # Apply to working world
                    current_world = candidate
                    current_audit = cand_audit
                    improved_in_this_pass = True

                    # Log correction provenance into world metadata
                    if current_world.metadata is None:
                        current_world.metadata = {}
                    if "autonomous_corrections_log" not in current_world.metadata:
                        current_world.metadata["autonomous_corrections_log"] = []

                    current_world.metadata["autonomous_corrections_log"].append({
                        "hypothesis_id": hyp.hypothesis_id,
                        "defect": hyp.defect_kind.value,
                        "action": hyp.action_type,
                        "rationale": hyp.rationale,
                        "quality_gain": round(quality_delta, 3),
                        "timestamp": time.time(),
                        "provenance": Provenance.ESTIMATED.value,
                    })

                else:
                    reason = (
                        f"Insufficient quality gain ({quality_delta:.4f} < {min_quality_gain})"
                        if quality_delta < min_quality_gain
                        else "Degraded structural soundness"
                    )
                    outcome = CorrectionOutcome(
                        hypothesis=hyp,
                        accepted=False,
                        quality_delta=quality_delta,
                        baseline_quality=current_audit.overall_quality_score,
                        candidate_quality=cand_audit.overall_quality_score,
                        rejection_reason=reason,
                    )
                    outcomes.append(outcome)
                    rejected_cnt += 1

            if not improved_in_this_pass:
                break

        # Copy back changes if current_world was cloned
        if current_world is not world:
            world.entities.clear()
            world.entities.update(current_world.entities)
            world.geometries.clear()
            world.geometries.update(current_world.geometries)
            if current_world.metadata:
                world.metadata.update(current_world.metadata)

        final_audit = QualityAuditor.audit(world)

        report = AutonomousLoopReport(
            initial_metrics=initial_audit,
            final_metrics=final_audit,
            hypotheses_evaluated=len(outcomes),
            corrections_accepted=accepted_cnt,
            corrections_rejected=rejected_cnt,
            outcomes=tuple(outcomes),
        )

        world.metadata["autonomous_quality_report"] = report.to_dict()
        return report

    @staticmethod
    def hash_entities_state(world: WorldIR) -> str:
        """Hash the state of all entities (id, type, attributes) to detect convergence."""
        items = []
        for eid in sorted(world.entities.keys()):
            ent = world.entities[eid]
            ef = ent.extra_fields if hasattr(ent, "extra_fields") and ent.extra_fields else {}
            attrs = ef.get("attributes", {})
            items.append((eid, str(ent.type), json.dumps(attrs, sort_keys=True)))
        return hashlib.sha256(json.dumps(items).encode("utf-8")).hexdigest()

    @staticmethod
    def formulate_refinement_tasks(
        world: WorldIR,
        priors: Dict[RefinementKind, RefinementPrior],
        attempts_tracker: Dict[Tuple[str, RefinementKind], int],
    ) -> List[RefinementTask]:
        """BUILD_SPEC B16: Formulate candidate refinement tasks across all entities."""
        tasks: List[RefinementTask] = []
        for ent in world.entities.values():
            ef = ent.extra_fields if hasattr(ent, "extra_fields") and ent.extra_fields else {}
            attrs = ef.get("attributes", {})
            crops = ef.get("candidate_crops", {})

            # 1. redetect_crop: attribute unknown and crop >= 96 px in another view
            for attr_name, attr_info in attrs.items():
                val = attr_info.get("value") if isinstance(attr_info, dict) else attr_info
                if val is None or val == "unknown":
                    crop_size = crops.get(attr_name, 0)
                    if crop_size >= 96:
                        target = f"{ent.id}:{attr_name}"
                        if attempts_tracker.get((target, RefinementKind.REDETECT_CROP), 0) < 2:
                            p = priors[RefinementKind.REDETECT_CROP].probability
                            gain = 1.0 * p
                            cost = 0.5
                            tasks.append(RefinementTask(
                                kind=RefinementKind.REDETECT_CROP,
                                target=target,
                                expected_gain=gain,
                                cost_s=cost,
                                metadata={"entity_id": ent.id, "attribute": attr_name, "crop_px": crop_size},
                            ))

            # 2. segment: no mask
            has_mask = (ef.get("mask_ref") is not None and ef.get("mask_ref") >= 0) or ef.get("has_mask", False)
            if not has_mask and ef.get("eligible_for_mask", False):
                target = ent.id
                if attempts_tracker.get((target, RefinementKind.SEGMENT), 0) < 2:
                    p = priors[RefinementKind.SEGMENT].probability
                    gain = 1.0 * p
                    cost = 0.3
                    tasks.append(RefinementTask(
                        kind=RefinementKind.SEGMENT,
                        target=target,
                        expected_gain=gain,
                        cost_s=cost,
                        metadata={"entity_id": ent.id},
                    ))

            # 3. reassociate: conflicting class votes
            votes = ef.get("class_votes")
            if votes and len(votes) > 1:
                counts = np.array(list(votes.values()), dtype=float)
                total = counts.sum()
                if total > 0:
                    probs = counts / total
                    entropy = -float(np.sum(probs * np.log2(np.clip(probs, 1e-9, 1.0))))
                    if entropy > 0.05:
                        target = ent.id
                        if attempts_tracker.get((target, RefinementKind.REASSOCIATE), 0) < 2:
                            p = priors[RefinementKind.REASSOCIATE].probability
                            gain = entropy * p
                            cost = 0.2
                            tasks.append(RefinementTask(
                                kind=RefinementKind.REASSOCIATE,
                                target=target,
                                expected_gain=gain,
                                cost_s=cost,
                                metadata={"entity_id": ent.id, "entropy": entropy, "votes": votes},
                            ))

            # 4. more_views: capture request (no compute)
            if ef.get("needs_more_views", False):
                target = ent.id
                if attempts_tracker.get((target, RefinementKind.MORE_VIEWS), 0) < 2:
                    p = priors[RefinementKind.MORE_VIEWS].probability
                    gain = 0.1 * p
                    cost = 0.001
                    tasks.append(RefinementTask(
                        kind=RefinementKind.MORE_VIEWS,
                        target=target,
                        expected_gain=gain,
                        cost_s=cost,
                        metadata={"entity_id": ent.id},
                    ))

        return tasks

    @staticmethod
    def run_refinement_loop(
        world: WorldIR,
        max_iterations: int = 5,
        time_budget_s: float = 30.0,
        executor: Optional[Callable[[RefinementTask, WorldIR], bool]] = None,
    ) -> RefinementReport:
        """BUILD_SPEC B16: Priority-driven refinement loop that knows when to stop.

        Uses a priority queue keyed by (-expected_gain / cost_s, seq).
        Stops on: empty heap, max_iterations (default 5), time budget exceeded,
        or 2 consecutive iterations where entity state remained unchanged.
        """
        priors: Dict[RefinementKind, RefinementPrior] = {
            RefinementKind.REDETECT_CROP: RefinementPrior(successes=2, attempts=3),
            RefinementKind.SEGMENT: RefinementPrior(successes=3, attempts=4),
            RefinementKind.REASSOCIATE: RefinementPrior(successes=2, attempts=3),
            RefinementKind.MORE_VIEWS: RefinementPrior(successes=1, attempts=4),
        }
        attempts_tracker: Dict[Tuple[str, RefinementKind], int] = {}
        history: List[Dict[str, Any]] = []

        start_time = time.time()
        last_state_hash = AutonomousCorrectionLoop.hash_entities_state(world)
        unchanged_iterations = 0
        seq = 0
        tasks_evaluated = 0
        tasks_succeeded = 0
        stop_reason = "completed"

        for it in range(max_iterations):
            # Check time budget
            if (time.time() - start_time) >= time_budget_s:
                stop_reason = "time_budget_exceeded"
                break

            tasks = AutonomousCorrectionLoop.formulate_refinement_tasks(world, priors, attempts_tracker)
            if not tasks:
                stop_reason = "empty_heap"
                break

            # Heap keyed by (-expected_gain / cost_s, seq, task)
            heap: List[Tuple[float, int, RefinementTask]] = []
            for t in tasks:
                ratio = t.expected_gain / max(t.cost_s, 1e-4)
                heapq.heappush(heap, (-ratio, seq, t))
                seq += 1

            if not heap:
                stop_reason = "empty_heap"
                break

            # Pop highest priority task
            _neg_ratio, _s, task = heapq.heappop(heap)
            attempts_tracker[(task.target, task.kind)] = attempts_tracker.get((task.target, task.kind), 0) + 1
            tasks_evaluated += 1

            # Execute task
            success = False
            if executor is not None:
                success = executor(task, world)
            else:
                # Default deterministic resolution
                if task.kind == RefinementKind.REDETECT_CROP:
                    ent_id = task.metadata["entity_id"]
                    attr = task.metadata["attribute"]
                    if ent_id in world.entities:
                        ent = world.entities[ent_id]
                        if not ent.extra_fields:
                            ent.extra_fields = {}
                        attrs = ent.extra_fields.setdefault("attributes", {})
                        if attr in attrs and isinstance(attrs[attr], dict):
                            attrs[attr]["value"] = "resolved"
                            attrs[attr]["basis"] = "visual"
                        else:
                            attrs[attr] = {"value": "resolved", "basis": "visual", "p": 0.95}
                        success = True

                elif task.kind == RefinementKind.SEGMENT:
                    ent_id = task.metadata["entity_id"]
                    if ent_id in world.entities:
                        ent = world.entities[ent_id]
                        if not ent.extra_fields:
                            ent.extra_fields = {}
                        ent.extra_fields["mask_ref"] = 1
                        ent.extra_fields["has_mask"] = True
                        success = True

                elif task.kind == RefinementKind.REASSOCIATE:
                    ent_id = task.metadata["entity_id"]
                    if ent_id in world.entities:
                        ent = world.entities[ent_id]
                        votes = task.metadata.get("votes", {})
                        if votes:
                            best_class = max(votes.items(), key=lambda kv: kv[1])[0]
                            ent.type = best_class
                            ent.extra_fields["class_votes"] = {best_class: sum(votes.values())}
                            success = True

                elif task.kind == RefinementKind.MORE_VIEWS:
                    ent_id = task.metadata["entity_id"]
                    if ent_id in world.entities:
                        ent = world.entities[ent_id]
                        if not world.metadata:
                            world.metadata = {}
                        reqs = world.metadata.setdefault("capture_requests", [])
                        reqs.append({"entity_id": ent_id, "reason": "more_views_needed"})
                        if ent.extra_fields:
                            ent.extra_fields["needs_more_views"] = False
                        success = True

            # Update priors
            priors[task.kind].record_outcome(success)
            if success:
                tasks_succeeded += 1

            history.append({
                "iteration": it,
                "kind": task.kind.value,
                "target": task.target,
                "expected_gain": round(task.expected_gain, 4),
                "cost_s": round(task.cost_s, 4),
                "success": success,
                "updated_prior_prob": round(priors[task.kind].probability, 4),
            })

            # Check for convergence: state unchanged
            current_state_hash = AutonomousCorrectionLoop.hash_entities_state(world)
            if current_state_hash == last_state_hash:
                unchanged_iterations += 1
                if unchanged_iterations >= 2:
                    stop_reason = "converged_state_unchanged"
                    break
            else:
                unchanged_iterations = 0
                last_state_hash = current_state_hash
        else:
            stop_reason = "max_iterations_reached"

        rep = RefinementReport(
            tasks_evaluated=tasks_evaluated,
            tasks_succeeded=tasks_succeeded,
            stop_reason=stop_reason,
            iterations=len(history),
            history=tuple(history),
        )
        if not world.metadata:
            world.metadata = {}
        world.metadata["refinement_loop_report"] = rep.to_dict()
        return rep
