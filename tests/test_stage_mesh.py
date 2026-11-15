"""P108.3: the mesh stage, run directly with the deterministic backend of test_cli_compile."""
from __future__ import annotations

from pathlib import Path

import pytest

from engine.pipeline.stages.runner import run_stages
from engine.pipeline.stages.s08_mesh import Mesh
from engine.pipeline.vertical_slice import vertical_slice
from tests.test_stage_dense import _state_through_compile

pytest.importorskip("PIL")


def _mesh(ctx):
    return run_stages([Mesh()], ctx).stages[0]


def test_spec() -> None:
    s = Mesh.spec
    assert (s.name, s.version, s.resource_class, s.required) == ("mesh", "1", "highmem", False)


def test_disabled_by_options_is_skipped_with_reason(tmp_path: Path) -> None:
    _, _, ctx = _state_through_compile(tmp_path, mesh_enabled=False)
    result = _mesh(ctx)
    assert result.status == "skipped" and any("disabled (mesh_enabled=False)" in w for w in result.warnings)


def test_no_artifact_store_is_skipped_not_succeeded(tmp_path: Path) -> None:
    items, opts, ctx = _state_through_compile(tmp_path)
    result = _mesh(ctx)
    assert result.status == "skipped" and any("no artifact_store" in w for w in result.warnings)
    assert ctx.state["mesh"].facts == vertical_slice(items, opts).stage_facts["mesh"]     # same facts as the CLI path


def test_too_few_points_is_skipped_with_the_evidence_reason(tmp_path: Path) -> None:
    from world_ir.artifact_store import FileArtifactStore

    _, _, ctx = _state_through_compile(tmp_path, artifact_store=FileArtifactStore(tmp_path / "art"))
    result = _mesh(ctx)
    assert result.status == "skipped" and any("too few for surface reconstruction" in w for w in result.warnings)


def test_ran_is_succeeded_and_failed_is_degraded(tmp_path: Path, monkeypatch) -> None:
    import engine.pipeline.vertical_slice as vs

    monkeypatch.setattr(vs, "_mesh_stage", lambda *a, **k: {"status": "ran", "vertices": 10, "triangles": 8})
    _, _, ctx = _state_through_compile(tmp_path)
    result = _mesh(ctx)
    assert result.status == "succeeded" and result.metrics == {"vertices": 10.0, "triangles": 8.0}
    monkeypatch.setattr(vs, "_mesh_stage", lambda *a, **k: {"status": "failed", "note": "MeshingError: boom"})
    _, _, ctx = _state_through_compile(tmp_path / "again")
    assert _mesh(ctx).status == "degraded"
