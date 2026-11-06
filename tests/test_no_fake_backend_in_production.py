"""Test doubles must never be part of a production default backend chain."""

from pathlib import Path

import engine.orchestrator.universal_pipeline as up
from apps.cli.main import _default_backends
from reconstruction.backend.fake import FakeReconstructionBackend


def test_cli_default_backends_exclude_the_fake():
    assert not any(isinstance(b, FakeReconstructionBackend) for b in _default_backends("colmap", False))


def test_universal_pipeline_default_backends_exclude_the_fake(monkeypatch):
    captured = {}

    class _Stop(Exception):
        pass

    class _Capture:
        def __init__(self, backends):
            captured["backends"] = list(backends)
            raise _Stop

    monkeypatch.setattr(up, "ReconstructionOrchestrator", _Capture)
    opts = up.UniversalPipelineOptions()
    hw = type("HW", (), {"is_gpu": False})()
    try:
        up.UniversalPipeline._run_reconstruction([], opts, Path("."), hw)
    except _Stop:
        pass
    assert captured["backends"], "no backends were constructed"
    assert not any(isinstance(b, FakeReconstructionBackend) for b in captured["backends"])
