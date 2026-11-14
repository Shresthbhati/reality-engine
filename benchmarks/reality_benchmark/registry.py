"""Reality Engine benchmark registry: metric categories, metrics, critical gates and the 24 scenarios.

These are PROPOSED benchmark definitions and targets, not measurements of the current engine. A target of ``None``
means "scene-specific": the threshold comes from the scenario's manifest tolerances, never from a default here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

BENCHMARK_VERSION = "1.0.0"
STATUSES = ("PASS", "FAIL", "PROVISIONAL", "BLOCKED", "NOT_APPLICABLE", "NOT_RUN")
#: statuses that carry no measurement
UNMEASURED = ("BLOCKED", "NOT_APPLICABLE", "NOT_RUN")
#: categories every scenario is scored on, whatever else it tests (system-level)
ALWAYS_APPLICABLE = ("G", "H", "I", "J", "K", "L")


@dataclass(frozen=True)
class Category:
    id: str
    name: str
    weight: float                    # percent of the weighted scorecard; 0 = reported, not weighted


@dataclass(frozen=True)
class MetricSpec:
    id: str                          # "<category>.<name>"
    category: str
    name: str
    unit: str
    op: Optional[str] = None         # ">=" | "<=" | "==" ; None = scene-specific
    target: Optional[float] = None
    critical: bool = False
    small_object: bool = False       # evaluated per entity group, never folded into a room-level score
    needs_ground_truth: bool = True
    min_samples: int = 30            # fewer samples than this can never be better than PROVISIONAL


@dataclass(frozen=True)
class Gate:
    """A pass/fail condition enforced independently of the weighted score."""
    id: str
    name: str
    metric_ids: Tuple[str, ...]
    rule: str = "all_pass"           # "all_pass": every metric PASS | "measured": every metric has a measurement


@dataclass(frozen=True)
class Scenario:
    id: str
    slug: str
    title: str
    tests: str
    categories: Tuple[str, ...]
    entity_groups: Tuple[str, ...] = ()


# Weights as proposed; they sum to 100. Materials (F) had no proposed weight: it is reported but not weighted.
CATEGORIES: Tuple[Category, ...] = (
    Category("A", "Object detection and recognition", 15.0),
    Category("B", "3D geometry and reconstruction", 20.0),
    Category("C", "Multi-view intelligence", 15.0),
    Category("D", "Architectural topology", 12.0),
    Category("E", "Fine-grained entities", 10.0),
    Category("F", "Materials and appearance", 0.0),
    Category("G", "WorldIR integrity", 10.0),
    Category("H", "Uncertainty and hallucination", 5.0),
    Category("I", "Performance and scalability", 5.0),
    Category("J", "Editing and interaction", 2.5),
    Category("K", "Export and interoperability", 2.5),
    Category("L", "Reliability", 3.0),
)

ENTITY_GROUPS = {
    "electrical": "Socket, switch, plug, distribution board",
    "plumbing": "Tap, basin, shower, toilet, pipe",
    "hvac": "AC, vents, ducts, outdoor units",
    "lighting": "Bulbs, recessed lights, pendants, fixtures",
    "doors": "Frame, hinges, handle, lock",
    "windows": "Frame, glass, grill, latch",
    "furniture": "Legs, arms, cushions, shelves",
    "vegetation": "Plant, stem, leaf, pot",
    "street": "Lamppost, sign, pole, bench",
}


def _m(cat: str, name: str, title: str, unit: str, op: Optional[str] = None, target: Optional[float] = None, **kw) -> MetricSpec:
    return MetricSpec(id=f"{cat}.{name}", category=cat, name=title, unit=unit, op=op, target=target, **kw)


def _build_metrics() -> Tuple[MetricSpec, ...]:
    pct = "%"
    out: List[MetricSpec] = [
        # A. object detection and recognition
        _m("A", "precision", "Precision", pct, ">=", 95), _m("A", "recall", "Recall", pct, ">=", 95),
        _m("A", "f1", "F1 score", pct, ">=", 95), _m("A", "map50", "mAP@50", pct, ">=", 95),
        _m("A", "map50_95", "mAP@50:95", pct, ">=", 85), _m("A", "classification_accuracy", "Classification accuracy", pct, ">=", 95),
        _m("A", "instance_miou", "Instance segmentation mIoU", pct, ">=", 85), _m("A", "boundary_f1", "Boundary F1", pct, ">=", 90),
        # B. geometry
        _m("B", "chamfer_distance", "Chamfer distance", "m"), _m("B", "surface_fscore", "Surface F-score", pct, ">=", 90),
        _m("B", "geometry_completeness", "Geometry completeness", pct, ">=", 95), _m("B", "normal_consistency", "Normal consistency", pct, ">=", 95),
        _m("B", "dimension_mae_pct", "Dimension MAE", pct, "<=", 2), _m("B", "reprojection_error_px", "Reprojection error", "px", "<=", 2),
        _m("B", "critical_mesh_validity", "Valid critical meshes", pct, "==", 100, critical=True),
        # C. multi-view
        _m("C", "camera_pose_accuracy", "Camera pose accuracy", "m"), _m("C", "cross_view_entity_matching", "Cross-view entity matching", pct, ">=", 95),
        _m("C", "duplicate_object_rate", "Duplicate object rate", pct, "<=", 2), _m("C", "cross_view_geometry_consistency", "Cross-view geometry consistency", pct, ">=", 95),
        _m("C", "point_cloud_alignment", "Point-cloud alignment", "m"), _m("C", "loop_closure_accuracy", "Loop closure accuracy", pct, ">=", 95),
        # D. architecture
        _m("D", "wall_recall", "Wall detection recall", pct, ">=", 98), _m("D", "floor_ceiling_detection", "Floor/ceiling detection", pct, ">=", 98),
        _m("D", "door_window_detection", "Door/window detection", pct, ">=", 95), _m("D", "room_segmentation", "Room segmentation", pct, ">=", 95),
        _m("D", "topology_edge_f1", "Topology edge F1", pct, ">=", 95), _m("D", "floorplan_dimension_error_pct", "Floor-plan dimension error", pct, "<=", 2),
        _m("D", "staircase_accuracy", "Staircase accuracy", pct, ">=", 95),
    ]
    # E. fine-grained entities: four metrics per group, small_object so they are never folded into a room-level score
    for g in ENTITY_GROUPS:
        out += [_m("E", f"{g}.precision", f"{g} precision", pct, ">=", 95, small_object=True),
                _m("E", f"{g}.recall", f"{g} recall", pct, ">=", 95, small_object=True),
                _m("E", f"{g}.localization_error", f"{g} localization error", "m", small_object=True),
                _m("E", f"{g}.instance_accuracy", f"{g} instance accuracy", pct, ">=", 95, small_object=True)]
    out += [
        # F. materials (no proposed targets: scene-specific)
        _m("F", "material_classification_accuracy", "Material classification accuracy", pct),
        _m("F", "texture_consistency", "Texture consistency", pct), _m("F", "reflective_surface_identification", "Reflective surface identification", pct),
        _m("F", "glass_transparency_handling", "Glass transparency handling", pct), _m("F", "material_boundary_accuracy", "Material boundary accuracy", pct),
        _m("F", "lighting_independent_recognition", "Lighting-independent material recognition", pct),
        # G. WorldIR integrity (checks on the engine's own output: no ground truth needed unless noted)
        _m("G", "id_uniqueness", "Entity ID uniqueness", pct, "==", 100, needs_ground_truth=False),
        _m("G", "referential_integrity", "Referential integrity", pct, "==", 100, critical=True, needs_ground_truth=False),
        _m("G", "parent_child_validity", "Parent-child relationship validity", pct, "==", 100, needs_ground_truth=False),
        _m("G", "spatial_relationship_consistency", "Spatial relationship consistency", pct, ">=", 95),
        _m("G", "geometry_entity_association", "Geometry-to-entity association", pct, "==", 100, needs_ground_truth=False),
        _m("G", "provenance_completeness", "Provenance completeness", pct, "==", 100, needs_ground_truth=False),
        _m("G", "version_consistency", "Version consistency", pct, "==", 100, needs_ground_truth=False),
        _m("G", "transaction_rollback_correctness", "Transaction and rollback correctness", pct, "==", 100, needs_ground_truth=False),
        _m("G", "invalid_commits_reported_success", "Invalid WorldIR commits reported as successful", "count", "==", 0, critical=True, needs_ground_truth=False, min_samples=1),
        _m("G", "corrupted_commits", "Corrupted WorldIR commits", "count", "==", 0, critical=True, needs_ground_truth=False, min_samples=1),
        _m("G", "false_success_states", "False-success reconstruction states", "count", "==", 0, critical=True, needs_ground_truth=False, min_samples=1),
        # H. uncertainty and hallucination
        _m("H", "ece", "Expected calibration error", "ratio"), _m("H", "brier", "Brier score", "ratio"),
        _m("H", "false_entity_rate", "False entity generation rate", pct, "<=", 2), _m("H", "unsupported_geometry_rate", "Unsupported geometry rate", pct, "<=", 2),
        _m("H", "occlusion_uncertainty_accuracy", "Occlusion uncertainty accuracy", pct), _m("H", "unknown_object_handling", "Unknown-object handling", pct),
        _m("H", "contradictory_evidence_detection", "Contradictory evidence detection", pct),
        _m("H", "fabricated_architecture_count", "Fabricated architectural entities", "count", "==", 0, critical=True, min_samples=1),
    ]
    # I. performance (thresholds depend on hardware: host-specific)
    for k, title, unit in (("ingestion_latency", "Image ingestion latency", "s"), ("detection_latency", "Detection latency", "s"),
                           ("reconstruction_latency", "Reconstruction latency", "s"), ("gpu_memory", "GPU memory", "MB"),
                           ("cpu_ram", "CPU/RAM consumption", "MB"), ("throughput", "Throughput", "images/min"),
                           ("scaling", "Scaling with image count", "ratio"), ("recovery_time", "Recovery time after failure", "s")):
        out.append(_m("I", k, title, unit, needs_ground_truth=False, min_samples=1))
    # J. editing, K. export, L. reliability
    for k, title in (("selection_accuracy", "Entity selection accuracy"), ("geometry_edit_correctness", "Geometry editing correctness"),
                     ("undo_redo_integrity", "Undo/redo integrity"), ("deletion_correctness", "Object deletion correctness"),
                     ("relationship_preservation", "Relationship preservation after edits"), ("save_reload_consistency", "Save/reload consistency")):
        out.append(_m("J", k, title, pct, "==", 100, needs_ground_truth=False))
    out.append(_m("J", "update_latency", "World update latency", "s", needs_ground_truth=False, min_samples=1))
    out.append(_m("K", "glb_validity", "GLB/GLTF validity", pct, "==", 100, needs_ground_truth=False))
    for k, title in (("mesh_completeness", "Mesh export completeness"), ("texture_preservation", "Texture preservation"),
                     ("material_preservation", "Material preservation"), ("coordinate_consistency", "Coordinate-system consistency"),
                     ("scale_preservation", "Scale preservation"), ("reimport_consistency", "Re-import consistency")):
        out.append(_m("K", k, title, pct, ">=", 95, needs_ground_truth=False))
    for k, title, op, tgt, unit in (("crash_free_session_rate", "Crash-free session rate", ">=", 99, pct),
                                    ("memory_leak", "Memory growth over a long session", None, None, "MB/h"),
                                    ("bg_job_recovery", "Background-job recovery", ">=", 99, pct),
                                    ("db_consistency", "Database consistency", "==", 100, pct),
                                    ("session_persistence", "Session persistence", "==", 100, pct),
                                    ("interrupted_recovery", "Interrupted-processing recovery", ">=", 99, pct),
                                    ("reproducibility", "Reproducibility of reconstruction runs", ">=", 95, pct)):
        out.append(_m("L", k, title, unit, op, tgt, needs_ground_truth=False, min_samples=1))
    return tuple(out)


METRICS: Tuple[MetricSpec, ...] = _build_metrics()
METRIC_BY_ID: Dict[str, MetricSpec] = {m.id: m for m in METRICS}
CATEGORY_BY_ID: Dict[str, Category] = {c.id: c for c in CATEGORIES}

#: critical pass/fail gates, enforced independently of the weighted score
GATES: Tuple[Gate, ...] = (
    Gate("worldir_commit_integrity", "No invalid or corrupted WorldIR commit is ever reported as successful",
         ("G.invalid_commits_reported_success", "G.corrupted_commits")),
    Gate("worldir_referential_integrity", "Every WorldIR reference resolves", ("G.referential_integrity",)),
    Gate("critical_mesh_validity", "Every critical mesh is valid", ("B.critical_mesh_validity",)),
    Gate("no_false_success", "No false-success reconstruction state", ("G.false_success_states",)),
    Gate("no_invented_architecture", "No architectural entity is fabricated to look complete", ("H.fabricated_architecture_count",)),
)
#: scenario-dependent gate: the small-object metrics of every required entity group must be reported on their own
SMALL_OBJECT_GATE_ID = "small_object_reporting"


def _s(i: int, slug: str, title: str, tests: str, cats: str, groups: str = "") -> Scenario:
    return Scenario(f"{i:03d}", slug, title, tests, tuple(cats.split(",")), tuple(g for g in groups.split(",") if g))


SCENARIOS: Tuple[Scenario, ...] = (
    _s(1, "furnished_living_room", "Furnished living room", "Object detection and geometry", "A,B,C,D,E", "furniture,lighting,hvac,windows,electrical,vegetation,doors"),
    _s(2, "open_plan_apartment", "Open-plan apartment", "Room topology", "A,B,C,D", "doors,furniture"),
    _s(3, "empty_room", "Empty room", "Architectural reconstruction", "B,D,E", "windows,doors,electrical"),
    _s(4, "multi_door_room", "Room with multiple doors", "Openings and wall segmentation", "D,E", "doors,windows"),
    _s(5, "kitchen", "Kitchen", "Cabinets, counters, appliances", "A,B,E", "plumbing,electrical,furniture,hvac"),
    _s(6, "bathroom", "Bathroom", "Sanitary fixtures, plumbing objects", "A,E", "plumbing,lighting,electrical"),
    _s(7, "bedroom", "Bedroom", "Furniture and occlusion", "A,H,E", "furniture,lighting,windows"),
    _s(8, "corridor", "Corridor", "Long-range geometry", "B,C,D,E", "doors,lighting"),
    _s(9, "staircase", "Staircase", "Steps, risers, railings, elevation", "B,D,E", "lighting"),
    _s(10, "electrical_installation", "Electrical installation", "Switches, sockets, plug points", "E", "electrical"),
    _s(11, "complex_ceiling", "Complex ceiling", "Beams, fans, lights, false ceiling", "B,E", "lighting,hvac"),
    _s(12, "glass_architecture", "Glass architecture", "Transparent surfaces", "B,F,H,E", "windows,doors"),
    _s(13, "outdoor_building", "Outdoor building", "Facades and architectural features", "B,D,E", "windows,doors"),
    _s(14, "footpath_road", "Footpath and road", "Outdoor geometry", "B,E", "street"),
    _s(15, "trees_plants", "Trees and plants", "Vegetation recognition", "A,E", "vegetation"),
    _s(16, "street_infrastructure", "Street infrastructure", "Lampposts, signs, poles", "A,E", "street,lighting"),
    _s(17, "multiple_floors", "Multiple floors", "Vertical topology", "C,D,E", "doors"),
    _s(18, "repeated_viewpoints", "Repeated viewpoints", "Same-world identity", "C"),
    _s(19, "different_lighting", "Different lighting", "Photometric robustness", "A,B,F"),
    _s(20, "occluded_objects", "Occluded objects", "Uncertainty estimation", "A,H"),
    _s(21, "architectural_styles", "Different architectural styles", "Generalization", "A,B,D"),
    _s(22, "damaged_structures", "Damaged structures", "Irregular geometry", "B,H"),
    _s(23, "dense_furniture", "Dense furniture", "Instance segmentation", "A,C,E", "furniture"),
    _s(24, "building_exterior_interior", "Building exterior and interior", "Cross-space consistency", "C,D"),
)
SCENARIO_BY_ID: Dict[str, Scenario] = {s.id: s for s in SCENARIOS}


def applicable_categories(scenario: Scenario) -> Tuple[str, ...]:
    return tuple(c.id for c in CATEGORIES if c.id in scenario.categories or c.id in ALWAYS_APPLICABLE)


def applicable_metrics(scenario: Scenario) -> List[MetricSpec]:
    cats = set(applicable_categories(scenario))
    out = []
    for m in METRICS:
        if m.category not in cats:
            continue
        if m.category == "E" and m.id.split(".")[1] not in scenario.entity_groups:
            continue                                   # only the entity groups the scenario really contains
        out.append(m)
    return out
