"""Tests for reconstruction/backend/registry.py (P5-01 backend
registry, P5-02 automatic selection): discovery honesty (COLMAP is
genuinely installed on this machine, verified via shutil.which before
writing these assertions; the other four are genuinely absent) and
deterministic ranking policy.
"""

import shutil

import pytest

from reconstruction.backend.colmap_backend import ColmapReconstructionBackend
from reconstruction.backend.registry import (
    BackendDescriptor,
    DEFAULT_REGISTRY,
    discover_backends,
    select_backends,
)

_COLMAP_INSTALLED = shutil.which("colmap") is not None


class TestDiscovery:
    def test_all_five_candidates_reported(self):
        results = discover_backends()
        names = [r.name for r in results]
        assert names == [
            "colmap", "openmvs", "alicevision_meshroom", "opensfm", "opendronemap",
        ]

    @pytest.mark.skipif(not _COLMAP_INSTALLED, reason="colmap not on PATH on this machine")
    def test_colmap_genuinely_available(self):
        results = discover_backends()
        colmap = next(r for r in results if r.name == "colmap")
        assert colmap.available is True
        assert colmap.binary_path
        assert colmap.has_adapter is True
        assert colmap.license == "BSD-3-Clause"

    def test_unimplemented_adapters_have_no_adapter_flag(self):
        results = discover_backends()
        for name in ("openmvs", "alicevision_meshroom", "opensfm", "opendronemap"):
            entry = next(r for r in results if r.name == name)
            assert entry.has_adapter is False

    def test_genuinely_uninstalled_tools_report_unavailable(self):
        # These four are real environment facts on this machine, not mocked.
        results = discover_backends()
        for name in ("openmvs", "alicevision_meshroom", "opensfm", "opendronemap"):
            entry = next(r for r in results if r.name == name)
            assert entry.available is False
            assert entry.binary_path == ""

    def test_discover_never_raises_on_empty_registry(self):
        assert discover_backends(registry=()) == ()


class TestSelection:
    @pytest.mark.skipif(not _COLMAP_INSTALLED, reason="colmap not on PATH on this machine")
    def test_colmap_selected_when_installed(self):
        assert select_backends() == ("colmap",)

    def test_no_adapter_excludes_even_if_installed(self):
        fake_installed_but_no_adapter = BackendDescriptor(
            name="fake", binary_names=("python",), license="MIT", adapter_factory=None
        )
        assert select_backends(registry=(fake_installed_but_no_adapter,)) == ()

    def test_adapter_present_but_binary_missing_excluded(self):
        fake_adapter_no_binary = BackendDescriptor(
            name="fake",
            binary_names=("definitely_not_a_real_binary_xyz123",),
            license="MIT",
            adapter_factory=ColmapReconstructionBackend,
        )
        assert select_backends(registry=(fake_adapter_no_binary,)) == ()

    def test_preference_order_preserved(self):
        # Two runnable fakes pointing at 'python' (always on PATH in test env)
        a = BackendDescriptor("a", ("python",), "MIT", ColmapReconstructionBackend)
        b = BackendDescriptor("b", ("python",), "MIT", ColmapReconstructionBackend)
        assert select_backends(registry=(a, b)) == ("a", "b")
        assert select_backends(registry=(b, a)) == ("b", "a")

    def test_default_registry_length(self):
        assert len(DEFAULT_REGISTRY) == 5
