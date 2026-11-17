import time
import pytest

from engine.quality.autonomous_correction_loop import (
    AutonomousCorrectionLoop,
    RefinementKind,
    RefinementPrior,
    RefinementReport,
    RefinementTask,
)
from world_ir import Entity, EntityType, WorldIR


def test_refinement_loop_nothing_to_gain_zero_tasks():
    world = WorldIR(id="w1", name="Clean World")
    ent = Entity(
        id="ent-1",
        name="Chair",
        type=EntityType.FURNITURE,
        custom_properties={"attributes": {"material": {"value": "wood", "basis": "visual"}}},
    )
    world.entities[ent.id] = ent

    report = AutonomousCorrectionLoop.run_refinement_loop(world)
    assert report.tasks_evaluated == 0
    assert report.stop_reason == "empty_heap"


def test_refinement_loop_unknown_attribute_resolved():
    world = WorldIR(id="w1", name="Test World")
    ent = Entity(
        id="ent-almirah",
        name="Almirah",
        type=EntityType.FURNITURE,
        custom_properties={
            "attributes": {"lock_type": {"value": None, "basis": "unknown"}},
            "candidate_crops": {"lock_type": 128},  # >= 96 px
        },
    )
    world.entities[ent.id] = ent

    report = AutonomousCorrectionLoop.run_refinement_loop(world)
    assert report.tasks_evaluated == 1
    assert report.tasks_succeeded == 1
    assert report.history[0]["kind"] == "redetect_crop"
    assert report.history[0]["target"] == "ent-almirah:lock_type"
    assert ent.extra_fields["attributes"]["lock_type"]["value"] == "resolved"


def test_refinement_loop_task_that_fails_not_retried_more_than_twice():
    world = WorldIR(id="w1", name="Test World")
    ent = Entity(
        id="ent-cabinet",
        name="Cabinet",
        type=EntityType.FURNITURE,
        custom_properties={
            "attributes": {"handle_type": {"value": "unknown", "basis": "unknown"}},
            "candidate_crops": {"handle_type": 100},
        },
    )
    world.entities[ent.id] = ent

    # Executor that always fails to resolve
    def failing_executor(task: RefinementTask, w: WorldIR) -> bool:
        return False

    report = AutonomousCorrectionLoop.run_refinement_loop(world, executor=failing_executor)
    # Target ent-cabinet:handle_type attempted at most 2 times
    assert report.tasks_evaluated == 2
    assert report.tasks_succeeded == 0
    assert report.stop_reason in ("empty_heap", "converged_state_unchanged")


def test_refinement_loop_stops_on_state_convergence():
    world = WorldIR(id="w1", name="Test World")
    ent = Entity(
        id="ent-mask",
        name="Sofa",
        type=EntityType.FURNITURE,
        custom_properties={
            "eligible_for_mask": True,
            "has_mask": False,
        },
    )
    world.entities[ent.id] = ent

    # Executor returns True but doesn't change entity state
    def no_change_executor(task: RefinementTask, w: WorldIR) -> bool:
        return True

    report = AutonomousCorrectionLoop.run_refinement_loop(world, executor=no_change_executor)
    assert report.stop_reason in ("converged_state_unchanged", "empty_heap")


def test_refinement_prior_updates():
    prior = RefinementPrior(successes=1, attempts=2)
    assert prior.probability == 2 / 4  # (1+1)/(2+2) = 0.5
    prior.record_outcome(True)
    assert prior.successes == 2 and prior.attempts == 3
    assert prior.probability == 3 / 5  # (2+1)/(3+2) = 0.6
