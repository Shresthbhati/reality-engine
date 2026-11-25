"""GPU auto-detect for COLMAP (reconstruction.backend.colmap_config.detect_colmap_gpu) and the stage report's fallback walker."""
import importlib.util
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from reconstruction.backend import colmap_config as cc


def _fake_run(banner, smi_free):
    def run(cmd, **kw):
        if cmd[0] == "nvidia-smi":
            return SimpleNamespace(stdout="" if smi_free is None else f"{smi_free}\n", stderr="")
        return SimpleNamespace(stdout=banner + "\nUsage:\n", stderr="")
    return run


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("RE_COLMAP_GPU", raising=False)
    cc._GPU_CACHE.clear()
    yield
    cc._GPU_CACHE.clear()


def test_cuda_build_with_vram_uses_gpu(monkeypatch):
    monkeypatch.setattr(subprocess, "run", _fake_run("COLMAP 4.2.0 (Commit x on 2026-08-31 with CUDA)", 5000))
    d = cc.detect_colmap_gpu("colmap")
    assert d["use_gpu"] is True and d["free_vram_mb"] == 5000


def test_cpu_build_falls_back_with_reason(monkeypatch):
    monkeypatch.setattr(subprocess, "run", _fake_run("COLMAP 3.9 (Commit x without CUDA)", 5000))
    d = cc.detect_colmap_gpu("colmap")
    assert d["use_gpu"] is False and "not CUDA" in d["reason"]


def test_low_vram_falls_back_with_reason(monkeypatch):
    monkeypatch.setattr(subprocess, "run", _fake_run("COLMAP 4.2.0 with CUDA", 300))
    d = cc.detect_colmap_gpu("colmap")
    assert d["use_gpu"] is False and "300 MB" in d["reason"]


def test_no_device_falls_back(monkeypatch):
    monkeypatch.setattr(subprocess, "run", _fake_run("COLMAP 4.2.0 with CUDA", None))
    monkeypatch.setattr(cc, "_free_vram_mb", lambda: None)
    assert cc.detect_colmap_gpu("colmap")["use_gpu"] is False


@pytest.mark.parametrize("val,expected", [("0", False), ("1", True)])
def test_env_override_wins(monkeypatch, val, expected):
    monkeypatch.setenv("RE_COLMAP_GPU", val)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("env override must not probe"))
    d = cc.detect_colmap_gpu("colmap")
    assert d["use_gpu"] is expected and d["source"] == "env"


def test_stage_report_walk_fallbacks_lists_skips_with_effect_or_not_stated():
    spec = importlib.util.spec_from_file_location("stage_report", Path(__file__).resolve().parents[1] / "scripts" / "stage_report.py")
    sr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sr)
    facts = {"status": "ran", "masks": {"fallback": "depth_band", "reason": "sam2 checkpoint missing", "quality_effect": "masks leak floor"},
             "mesh": {"status": "skipped", "note": "world is not METRIC-scale"}}
    fb = sr.walk_fallbacks(facts)
    by = {f["where"]: f for f in fb}
    assert by["masks"]["effect_stated_by_pipeline"] == "masks leak floor"
    assert by["mesh"]["effect_stated_by_pipeline"] == "NOT STATED by the pipeline"
    assert by["mesh"]["reason"] == "world is not METRIC-scale"
    assert "<stage>" not in by  # a ran stage with no fallback is not reported


def test_mixed_sizes_share_one_camera_per_size_folder(tmp_path):
    from PIL import Image

    from reconstruction.backend.colmap_backend import _size_folders

    a, b, c = tmp_path / "a.png", tmp_path / "b.png", tmp_path / "c.png"
    for p, size in ((a, (64, 48)), (b, (64, 48)), (c, (48, 64))):
        Image.new("RGB", size).save(p)
    f = _size_folders([a, b, c])
    assert f[a] == f[b] == "size_64x48" and f[c] == "size_48x64"
    assert _size_folders([a, b]) is None            # uniform: the single_camera path handles it
    assert _size_folders([a, tmp_path / "missing.png"]) is None   # unreadable: keep the per-image default
