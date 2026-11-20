"""A sticky/out-of-memory CUDA context must make the resource probe report 'GPU unusable' (so the executor falls back to CPU)
instead of raising out of the probe and failing the whole evidence stage."""

from __future__ import annotations

from types import SimpleNamespace

from engine.multiengine.executor import TorchResourceProbe


def _probe(cuda):
    p = TorchResourceProbe()
    p._torch = lambda: SimpleNamespace(cuda=cuda)  # type: ignore[method-assign]
    return p


def _broken():
    def boom():
        raise RuntimeError("CUDA error: out of memory\nSticky error detected")

    return SimpleNamespace(is_available=lambda: True, mem_get_info=boom)


def test_sticky_cuda_error_means_gpu_unavailable_and_is_recorded():
    p = _probe(_broken())
    assert p.gpu_available() is False
    assert p.free_vram_mb() is None
    assert p.cuda_error and "out of memory" in p.cuda_error


def test_healthy_cuda_still_reports_vram():
    p = _probe(SimpleNamespace(is_available=lambda: True, mem_get_info=lambda: (2 * 1024 * 1024 * 1024, 4 * 1024 * 1024 * 1024)))
    assert p.gpu_available() is True and p.free_vram_mb() == 2048.0 and p.cuda_error is None


def test_no_cuda_is_false_not_error():
    p = _probe(SimpleNamespace(is_available=lambda: False, mem_get_info=lambda: (_ for _ in ()).throw(AssertionError)))
    assert p.gpu_available() is False and p.free_vram_mb() is None and p.cuda_error is None
