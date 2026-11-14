"""Per-scenario manifest skeletons. Every field a quantitative run needs is present and EMPTY: nothing is pre-filled,
and ``ground_truth_status`` stays "missing" until surveyed truth and verified camera poses are supplied."""

from __future__ import annotations

from typing import List

from benchmarks.reality_benchmark.registry import (
    BENCHMARK_VERSION, ENTITY_GROUPS, GATES, SMALL_OBJECT_GATE_ID, Scenario, applicable_metrics,
)

MIN_IMAGES_PER_DATASET = 8


def manifest_template(scenario: Scenario) -> dict:
    metrics = applicable_metrics(scenario)
    return {
        "benchmark_version": BENCHMARK_VERSION,
        "scenario_id": scenario.id,
        "slug": scenario.slug,
        "title": scenario.title,
        "tests": scenario.tests,
        # A: controlled synthetic scenes; B: real captures. Synthetic images alone cannot establish real-world accuracy.
        "datasets": {
            "A_synthetic": {"images": [], "min_images": MIN_IMAGES_PER_DATASET, "source": None},
            "B_real": {"images": [], "min_images": MIN_IMAGES_PER_DATASET, "source": None},
        },
        "camera_poses": {"verified": False, "poses": []},
        "ground_truth": {"geometry": None, "dimensions": None, "object_identities": None, "topology": None},
        "ground_truth_status": "missing",
        "expected_entities": [],
        "expected_topology": {"rooms": [], "openings": [], "connections": []},
        "entity_groups": {g: ENTITY_GROUPS[g] for g in scenario.entity_groups},
        # per metric id: {"op": ">=", "value": n} overrides; required for every scene-specific metric you want scored
        "tolerances": {m.id: None for m in metrics if m.op is None},
        "pass_fail": {
            "required_metrics": [m.id for m in metrics],
            "critical_gates": [g.id for g in GATES] + ([SMALL_OBJECT_GATE_ID] if scenario.entity_groups else []),
            "failure_conditions": _failure_conditions(scenario),
        },
    }


def _failure_conditions(scenario: Scenario) -> List[str]:
    out = ["any critical gate fails", "any measured metric misses its threshold",
           "any architectural entity is created that no evidence supports"]
    if scenario.entity_groups:
        out.append("a small-object metric is folded into a room-level score instead of reported on its own")
    return out
