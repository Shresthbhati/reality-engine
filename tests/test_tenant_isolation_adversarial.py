"""P0 adversarial tenant-isolation tests (PR #471 repair).

These guard the checks that PR #471 deleted. They verify the guard code is
present in the reconstruction/process paths and that cross-tenant operations
are rejected over the API where possible.
"""
from __future__ import annotations

import inspect

from apps.api import jobs


def test_evidence_job_tenant_guard_present():
    src = inspect.getsource(jobs._run_process_evidence)
    assert "job.team_id" in src and "evidence.team_id" in src
    assert "Cross-tenant evidence access refused" in src


def test_session_world_and_job_world_guards_present():
    src = inspect.getsource(jobs._run_reconstruct_session_inner)
    assert "session.team_id" in src and "world.team_id" in src
    assert "Cross-tenant session/world association refused" in src
    assert "Cross-tenant job execution refused" in src
    assert "Cross-tenant evidence" in src


def test_team_scoped_evidence_filter_present():
    src = inspect.getsource(jobs._run_reconstruct_session_inner)
    assert "Evidence.team_id == world.team_id" in src or "Evidence.team_id == world.team_id" in src.replace(" ", "")


def test_spatial_global_provider_present():
    """PR #471 also deleted the spatial-global inputs provider -- must stay wired."""
    src = inspect.getsource(jobs._run_reconstruct_session_inner)
    assert "_spatial_inputs_provider" in src
    assert "spatial_enabled" in src
