"""Tests for the consolidated stage lifecycle state machine (Cline 216df recovery)."""

from __future__ import annotations

import pytest

from engine.pipeline.stages.status import (
    can_transition, grading_impact, is_non_productive, is_terminal,
    validate_transition,
)


def test_legal_happy_path_transitions():
    validate_transition("queued", "running")
    validate_transition("running", "succeeded")
    validate_transition("running", "degraded")
    validate_transition("running", "cancellation_requested")
    validate_transition("cancellation_requested", "cancelled")


def test_illegal_resurrection_is_rejected():
    with pytest.raises(ValueError, match="illegal stage status transition"):
        validate_transition("failed", "succeeded")
    with pytest.raises(ValueError):
        validate_transition("succeeded", "running")
    with pytest.raises(ValueError):
        validate_transition("cancelled", "succeeded")
    with pytest.raises(ValueError):
        validate_transition("timed_out", "succeeded")


def test_cancellation_cannot_jump_to_success():
    assert can_transition("cancellation_requested", "cancelled")
    assert not can_transition("cancellation_requested", "succeeded")


def test_terminal_and_non_productive_sets():
    assert is_terminal("failed") and is_terminal("succeeded") and is_terminal("timed_out")
    assert not is_terminal("running") and not is_terminal("queued")
    assert is_non_productive("failed") and is_non_productive("cancelled") and is_non_productive("timed_out")
    assert not is_non_productive("succeeded") and not is_non_productive("degraded")


def test_unknown_status_is_not_terminal_and_cannot_transition():
    assert not is_terminal("garbage")
    assert not can_transition("garbage", "running")
    with pytest.raises(ValueError):
        validate_transition("garbage", "running")


def test_grading_impact_never_upgrades_failure():
    assert grading_impact("failed", required=True) == "failed"
    assert grading_impact("failed", required=False) == "partial"
    assert grading_impact("timed_out", required=True) == "failed"
    assert grading_impact("cancelled", required=True) == "failed"
    assert grading_impact("skipped", required=True) == "failed"
    assert grading_impact("degraded", required=True) == "partial"
    assert grading_impact("partial", required=True) == "partial"
    assert grading_impact("succeeded", required=True) == "ok"
    assert grading_impact("running", required=True) == "ok"  # in-flight is not a verdict
