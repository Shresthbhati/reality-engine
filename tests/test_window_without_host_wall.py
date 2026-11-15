"""A detected window whose host plane is not a wall entity is skipped and reported, not promoted.

Found on a real multi-view run: one orphan window made the world validator reject the whole compile
("opening window-001 (window) references no surface"), so the ladder discarded the multi-view world and fell back to a
single-image one. The orphan must cost one window, not the world.
"""

from __future__ import annotations

import dataclasses

import perception.architecture.windows as windows_mod
from engine.compiler import CompileOptions, compile_reconstruction_to_world
from perception.architecture.classify import PlaneInput
from perception.architecture.windows import detect_window
from tests.test_canonical_interior import UP, _canonical_interior_scene
from world_ir import EntityType
from world_ir.validation import validate_world_ir


def _real_window_fit():
    """A genuine WindowFit from a wall plane with a 1.2 x 1.0 m hole (same construction as test_interior_compiler)."""
    pts = []
    for xi in range(81):
        x = round(xi * 0.05, 3)
        for zi in range(61):
            z = round(zi * 0.05, 3)
            if 1.4 <= x <= 2.6 and 0.9 <= z <= 1.9:
                continue
            pts.append((x, 0.0, z))
    pi = PlaneInput(plane_id="wall-ext-1", normal=(0.0, 1.0, 0.0), centroid=(2.0, 0.0, 1.5),
                    bounds_min=(0.0, 0.0, 0.0), bounds_max=(4.0, 0.0, 3.0), inlier_positions=tuple(pts))
    fit = detect_window(pi, up=(0.0, 0.0, 1.0), floor_height=0.0)
    assert fit is not None
    return fit


def _compile_with_window_on(monkeypatch, host_plane):
    """Compile the canonical scene; the window detector reports one real fit hosted on ``host_plane`` (a callable of the
    plane being inspected, or a fixed id) on the first wall plane it is asked about."""
    fit = _real_window_fit()
    calls = []

    def fake_detect(pi, **kwargs):
        calls.append(pi.plane_id)
        if len(calls) > 1:
            return None
        pid = host_plane(pi) if callable(host_plane) else host_plane
        return dataclasses.replace(fit, wall_plane_id=pid)

    monkeypatch.setattr(windows_mod, "detect_window", fake_detect)
    options = CompileOptions(seed=42, up=UP, promote_corridors=True, promote_windows=True, promote_stairs=True,
                             promote_building=True)
    return compile_reconstruction_to_world(_canonical_interior_scene(), options)


def _windows(world):
    return [e for e in world.entities.values() if e.type == EntityType.WINDOW]


def _warnings(diag):
    return [str(w) for name in ("warnings", "interior_warnings", "notes") for w in (getattr(diag, name, None) or [])]


def test_control_window_on_a_real_wall_is_promoted(monkeypatch):
    world, diag = _compile_with_window_on(monkeypatch, lambda pi: pi.plane_id)
    assert len(_windows(world)) == 1, _warnings(diag)
    assert diag.validation_issues == []


def test_window_on_a_non_wall_plane_is_skipped_with_a_warning_and_the_world_stays_valid(monkeypatch):
    world, diag = _compile_with_window_on(monkeypatch, "plane-not-a-wall")
    assert _windows(world) == []
    assert diag.validation_issues == []
    assert validate_world_ir(world).is_valid()
    assert any("not promoted" in w and "plane-not-a-wall" in w for w in _warnings(diag)), _warnings(diag)
