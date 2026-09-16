"""Dense MVS pipeline contract tests (P6-01 integration).

Every subprocess interaction is captured through a recording fake (no
real COLMAP invocation in the test suite -- the real CUDA run is
already verified in the capability registry; these tests pin the
CONTRACT: stage order, arguments, honest failures, and the canonical
ingestion tail). The fake writes a real CRLF-header fused.ply so the
header-count path is exercised against COLMAP's actual output shape.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import reconstruction.dense_pipeline as dp
from reconstruction.dense_pipeline import (
    DenseMVSRunError,
    DenseMVSUnavailableError,
    dense_mvs_available,
    run_dense_mvs,
)


def _write_fused_ply(path: Path, n: int = 12) -> None:
    """A minimal COLMAP-shaped fused.ply: CRLF header (the real Windows
    output shape), binary little endian, vertex-only float32 xyz."""
    import struct
    header = (
        "ply\r\n"
        "format binary_little_endian 1.0\r\n"
        f"element vertex {n}\r\n"
        "property float x\r\n"
        "property float y\r\n"
        "property float z\r\n"
        "end_header\r\n"
    )
    body = b"".join(struct.pack("<fff", float(i), 0.0, 1.0) for i in range(n))
    path.write_bytes(header.encode("ascii") + body)


class _FakeProc:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


@pytest.fixture()
def capture_env(tmp_path, monkeypatch):
    """Recording fake for subprocess.run inside dense_pipeline: creates
    the fused.ply on the final stage and records every argv."""
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        assert kwargs.get("shell") is False, "COLMAP must never run via shell"
        assert "QT_QPA_PLATFORM" in kwargs["env"]
        if argv[1] == "stereo_fusion":
            out = argv[argv.index("--StereoFusion.output") + 1]
            _write_fused_ply(Path(out), n=12)
        return _FakeProc(0, "ok", "")

    monkeypatch.setattr(dp.subprocess, "run", fake_run)
    monkeypatch.setattr(
        dp.shutil, "which", lambda name: f"C:/fake/{name}.exe"
    )
    return calls


def _inputs(tmp_path: Path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    sparse = tmp_path / "sparse"
    sparse.mkdir()
    return image_dir, sparse, tmp_path / "ws"


class TestContract:
    def test_three_stages_in_order_with_expected_args(self, capture_env, tmp_path):
        image_dir, sparse, ws = _inputs(tmp_path)
        run = run_dense_mvs(
            image_dir=image_dir, sparse_model_dir=sparse, workspace=ws,
            binary="colmap", use_gpu=True,
        )
        steps = [c[1] for c in capture_env]
        assert steps == ["image_undistorter", "patch_match_stereo", "stereo_fusion"]
        pm = capture_env[1]
        assert "--PatchMatchStereo.geom_consistency" in pm
        assert pm[pm.index("--PatchMatchStereo.geom_consistency") + 1] == "true"
        assert pm[pm.index("--PatchMatchStereo.use_gpu") + 1] == "1"
        fusion = capture_env[2]
        assert fusion[fusion.index("--StereoFusion.input_type") + 1] == "geometric"
        # the observed facts are real, not invented
        assert run.n_fused_points == 12
        assert Path(run.fused_ply_path).is_file()
        assert set(run.stage_durations_s) == {
            "image_undistorter", "patch_match_stereo", "stereo_fusion",
        }

    def test_use_gpu_omitted_leaves_colmap_choice(self, capture_env, tmp_path):
        image_dir, sparse, ws = _inputs(tmp_path)
        run_dense_mvs(image_dir=image_dir, sparse_model_dir=sparse, workspace=ws)
        assert "--PatchMatchStereo.use_gpu" not in capture_env[1]

    def test_stage_failure_raises_with_step_and_stderr(self, tmp_path, monkeypatch):
        image_dir, sparse, ws = _inputs(tmp_path)
        monkeypatch.setattr(dp.shutil, "which", lambda n: "C:/fake/colmap.exe")

        def fake_run(argv, **kwargs):
            if argv[1] == "patch_match_stereo":
                return _FakeProc(1, "", "CUDA error: out of memory")
            return _FakeProc(0, "", "")

        monkeypatch.setattr(dp.subprocess, "run", fake_run)
        with pytest.raises(DenseMVSRunError, match="patch_match_stereo"):
            run_dense_mvs(image_dir=image_dir, sparse_model_dir=sparse,
                          workspace=ws)

    def test_success_without_fused_ply_is_an_error(self, tmp_path, monkeypatch):
        image_dir, sparse, ws = _inputs(tmp_path)
        monkeypatch.setattr(dp.shutil, "which", lambda n: "C:/fake/colmap.exe")
        monkeypatch.setattr(
            dp.subprocess, "run", lambda argv, **kw: _FakeProc(0, "", "")
        )
        with pytest.raises(DenseMVSRunError, match="no fused.ply"):
            run_dense_mvs(image_dir=image_dir, sparse_model_dir=sparse,
                          workspace=ws)

    def test_missing_input_dirs_refused_before_any_subprocess(self, tmp_path, monkeypatch):
        called = []
        monkeypatch.setattr(dp.shutil, "which", lambda n: "C:/fake/colmap.exe")
        monkeypatch.setattr(
            dp.subprocess, "run", lambda argv, **kw: called.append(argv) or _FakeProc(0)
        )
        image_dir = tmp_path / "images"  # exists
        image_dir.mkdir()
        with pytest.raises(DenseMVSRunError, match="sparse_model_dir"):
            run_dense_mvs(
                image_dir=image_dir,
                sparse_model_dir=tmp_path / "sparse",  # missing
                workspace=tmp_path / "ws",
            )
        assert called == []

    def test_timeout_wrapped_as_named_error(self, tmp_path, monkeypatch):
        image_dir, sparse, ws = _inputs(tmp_path)
        monkeypatch.setattr(dp.shutil, "which", lambda n: "C:/fake/colmap.exe")

        def fake_run(argv, **kwargs):
            raise subprocess.TimeoutExpired(cmd=argv[0], timeout=1.0)

        monkeypatch.setattr(dp.subprocess, "run", fake_run)
        with pytest.raises(DenseMVSRunError, match="timed out"):
            run_dense_mvs(image_dir=image_dir, sparse_model_dir=sparse,
                          workspace=ws, timeout_s=1.0)


class TestProbe:
    def test_probe_true_requires_patch_match_stereo_in_help(self, monkeypatch):
        monkeypatch.setattr(dp.shutil, "which", lambda n: "C:/fake/colmap.exe")
        monkeypatch.setattr(
            dp.subprocess, "run",
            lambda argv, **kw: _FakeProc(0, "patch_match_stereo stereo_fusion", ""),
        )
        assert dense_mvs_available("colmap") is True

    def test_probe_false_when_command_missing(self, monkeypatch):
        monkeypatch.setattr(dp.shutil, "which", lambda n: "C:/fake/colmap.exe")
        monkeypatch.setattr(
            dp.subprocess, "run", lambda argv, **kw: _FakeProc(0, "mapper", "")
        )
        assert dense_mvs_available("colmap") is False

    def test_probe_false_when_binary_absent(self, monkeypatch):
        monkeypatch.setattr(dp.shutil, "which", lambda n: None)
        assert dense_mvs_available("colmap") is False

    def test_missing_binary_raises_named_error(self, monkeypatch):
        monkeypatch.setattr(dp.shutil, "which", lambda n: None)
        with pytest.raises(DenseMVSUnavailableError):
            run_dense_mvs(
                image_dir=Path("x"), sparse_model_dir=Path("y"),
                workspace=Path("z"),
            )

    def test_probe_never_runs_colmap_through_a_shell(self, monkeypatch):
        """The probe executes a real binary: list-argv + shell=False is the
        module's documented subprocess contract, and it must hold on the
        probe path too (not only on the stage path)."""
        seen = []
        monkeypatch.setattr(dp.shutil, "which", lambda n: "C:/fake/colmap.exe")

        def fake_run(argv, **kwargs):
            seen.append((list(argv), kwargs))
            return _FakeProc(0, "patch_match_stereo", "")

        monkeypatch.setattr(dp.subprocess, "run", fake_run)
        assert dense_mvs_available("colmap") is True
        argv, kwargs = seen[0]
        assert isinstance(argv, list)  # never a shell string
        assert kwargs.get("shell") is False


