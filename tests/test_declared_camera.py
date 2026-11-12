"""Operator-declared camera: supplies intrinsics when EXIF is stripped, never invents them."""
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from apps.api import worldstore_service as wss
from apps.api.jobs import _intrinsics_options


def _facts(w=1024, h=768, f35=None):
    return SimpleNamespace(geometry_eligible=True, width=w, height=h,
                           exif={"focal_length_35mm": f35} if f35 else {})


def test_no_exif_no_declaration_stays_unset():
    kw, rec = _intrinsics_options([_facts()])
    assert "intrinsics" not in kw and rec["intrinsics"] is None


def test_declared_35mm_supplies_intrinsics():
    kw, rec = _intrinsics_options([_facts(), _facts()], declared={"focal_length_35mm": 28.0})
    fx = 28.0 * math.hypot(1024, 768) / 43.267
    assert kw["intrinsics"] == pytest.approx((fx, fx, 512.0, 384.0))
    assert rec["intrinsics"]["source"] == "operator_declared_camera"
    assert kw["intrinsics_trust"] == "estimated"


def test_declared_hfov_and_px():
    kw, _ = _intrinsics_options([_facts()], declared={"hfov_deg": 90.0})
    assert kw["intrinsics"][0] == pytest.approx(512.0)          # tan(45 deg) = 1 -> f = w/2
    kw, _ = _intrinsics_options([_facts()], declared={"focal_px": 800.0})
    assert kw["intrinsics"][0] == 800.0


def test_declaration_overrides_exif():
    kw, rec = _intrinsics_options([_facts(f35=50.0)], declared={"hfov_deg": 90.0})
    assert rec["intrinsics"]["source"] == "operator_declared_camera"


def test_store_validates_and_roundtrips(tmp_path, monkeypatch):
    monkeypatch.setenv("WORLDSTORE_ROOT", str(tmp_path))
    assert wss.get_camera_declaration("w1") is None
    assert wss.set_camera_declaration("w1", hfov_deg=70) == {"hfov_deg": 70.0}
    assert wss.get_camera_declaration("w1") == {"hfov_deg": 70.0}
    for bad in ({}, {"hfov_deg": 70, "focal_px": 800}, {"hfov_deg": 400}, {"focal_length_35mm": 1}):
        with pytest.raises(ValueError):
            wss.set_camera_declaration("w1", **bad)
    assert wss.get_camera_declaration("w1") == {"hfov_deg": 70.0}      # failed attempts changed nothing
