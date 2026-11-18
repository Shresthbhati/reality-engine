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


def _almirah():
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
    return world, ent


def test_without_a_detector_the_unknown_attribute_stays_unknown_and_nothing_is_invented():
    world, ent = _almirah()

    report = AutonomousCorrectionLoop.run_refinement_loop(world)

    assert report.tasks_evaluated == 0 and report.tasks_succeeded == 0 and report.tasks_unavailable == 1
    assert report.stop_reason == "no_executor_available"
    assert report.history[0]["status"] == "unavailable" and report.history[0]["kind"] == "redetect_crop"
    assert ent.extra_fields["attributes"]["lock_type"]["value"] is None            # not "resolved", not anything


def test_an_injected_engine_resolves_the_attribute_and_the_loop_records_it():
    world, ent = _almirah()

    def detector(task: RefinementTask, w: WorldIR):
        w.entities[task.metadata["entity_id"]].extra_fields["attributes"][task.metadata["attribute"]] = {
            "value": "numpad", "basis": "visual", "p": 0.8}
        return True

    report = AutonomousCorrectionLoop.run_refinement_loop(world, executor=detector)
    assert report.tasks_evaluated == 1 and report.tasks_succeeded == 1 and report.tasks_unavailable == 0
    assert report.history[0]["target"] == "ent-almirah:lock_type"
    assert ent.extra_fields["attributes"]["lock_type"]["value"] == "numpad"


def test_more_views_is_a_real_capture_request_and_a_vote_tie_changes_nothing():
    world = WorldIR(id="w1", name="T")
    sofa = Entity(id="s", name="Sofa", type=EntityType.FURNITURE,
                  custom_properties={"needs_more_views": True, "class_votes": {"sofa": 3, "bed": 3}})
    world.entities["s"] = sofa

    report = AutonomousCorrectionLoop.run_refinement_loop(world)

    assert world.metadata["capture_requests"] == [{"entity_id": "s", "reason": "more_views_needed"}]
    assert sofa.extra_fields["class_votes"] == {"sofa": 3, "bed": 3}             # 3-3 tie: the vote decided nothing
    assert report.tasks_unavailable == 0 and "reassociated_by" not in sofa.extra_fields


def test_a_clear_majority_vote_reassociates_and_says_how():
    world = WorldIR(id="w1", name="T")
    ent = Entity(id="e", name="X", type=EntityType.FURNITURE, custom_properties={"class_votes": {"chair": 5, "stool": 1}})
    world.entities["e"] = ent
    AutonomousCorrectionLoop.run_refinement_loop(world)
    assert ent.extra_fields["reassociated_by"] == "majority_vote" and ent.extra_fields["resolved_class"] == "chair"
    assert ent.type == EntityType.FURNITURE                    # "chair" is an ontology class, not an EntityType member: type untouched


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
