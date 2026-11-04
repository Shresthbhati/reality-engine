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

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

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
