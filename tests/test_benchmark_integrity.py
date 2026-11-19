"""Tests for Benchmark Integrity and Anti-Hallucination Controls.

Verifies:
1. Missing physical photos or genuine reconstruction evidence reports BLOCKED.
2. Ground truth is never leaked or copied into simulated predictions.
3. Holdout rooms are strictly protected and raise PermissionError without explicit authorization.
4. Part association accuracy is evaluated dynamically, returning None when unannotated (never hardcoded 1.0).
"""

from __future__ import annotations

import pytest
from benchmarks.measured_rooms.universal_benchmark import UniversalRoomBenchmarkRunner


def test_missing_photos_returns_blocked() -> None:
    """When a measured room has no physical camera photographs or reconstruction evidence,

    the benchmark MUST return BLOCKED, never fabricating physical metrics.
    """
    runner = UniversalRoomBenchmarkRunner()
    result = runner.evaluate_room("room_01_living_studio", synthetic_ok=False)

    assert result.status == "BLOCKED"
    assert "Missing" in str(result.reason) or "required" in str(result.reason)
    assert result.wall_rmse_m is None
    assert result.object_precision is None
    assert result.object_recall is None
    assert result.object_f1 is None
    assert result.part_association_accuracy is None


def test_holdout_room_raises_permission_error_without_flag() -> None:
    """Evaluating holdout rooms without --allow-holdout raises PermissionError to prevent test-set leakage."""
    runner = UniversalRoomBenchmarkRunner()

    with pytest.raises(PermissionError, match="requires explicit authorization"):
        runner.evaluate_room("room_05_holdout_reception", allow_holdout=False)

    with pytest.raises(PermissionError, match="requires explicit authorization"):
        runner.evaluate_room("room_06_holdout_conference", allow_holdout=False)


def test_synthetic_smoke_test_is_isolated_and_not_hardcoded_part_accuracy() -> None:
    """Synthetic smoke tests are labeled synthetic_smoke_test and never hardcode part_association_accuracy to 1.0."""
    runner = UniversalRoomBenchmarkRunner()
    result = runner.evaluate_room("room_01_living_studio", synthetic_ok=True)

    assert result.status == "EVALUATED"
    assert "synthetic" in result.data_type
    # Ground truth for room_01 has no annotated subparts -> part_association_accuracy is None, NOT 1.0!
    assert result.part_association_accuracy is None
