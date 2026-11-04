"""Tests verifying honest job progress, stage counts, and failure reasons (Task 1.6)."""

import pytest
from unittest.mock import MagicMock
from apps.api.models import Job
from apps.api.routes_jobs import _progress_view


@pytest.mark.asyncio
async def test_progress_view_honest_stage_counts_and_failure_reason():
    """Verify that stage_counts are exposed from payload and failure reasons are never hidden."""
    db = MagicMock()

    # 1. Running job with stage and compile counts
    job_running = Job(
        id="job-1",
        status="running",
        stage="reconstructing",
        progress=45.0,
        entity_type="world",
        entity_id="w-1",
        payload={
            "compile": {"entities": 14, "measurements": 3, "relationships": 2},
        },
    )

    view = await _progress_view(db, job_running)
    assert view["status"] == "running"
    assert view["stage"] == "reconstructing"
    assert view["stage_counts"] == {"entities": 14, "measurements": 3, "relationships": 2}
    assert view["failure_reason"] is None
    assert view["terminal"] is False

    # 2. Failed job with explicit error reason
    job_failed = Job(
        id="job-2",
        status="failed",
        stage="reconstructing",
        progress=30.0,
        entity_type="world",
        entity_id="w-2",
        error="Only 2 of 12 photos overlap enough to calculate camera positions",
        payload={"reason": "Sparse reconstruction failure in COLMAP mapper"},
    )

    view_failed = await _progress_view(db, job_failed)
    assert view_failed["status"] == "failed"
    assert view_failed["failure_reason"] == "Only 2 of 12 photos overlap enough to calculate camera positions"
    assert view_failed["terminal"] is True
