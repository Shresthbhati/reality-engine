"""Tests for T-3.1 Closed-Window Recognition & Multi-Modal Window Fusion.

Verifies:
1. Closed window with glass/frame inlier points is recognized as a window (shut window != wall).
2. Visual detector extracts glass, frame, mullion cues on wall planes.
3. Fusion combines geometric gap detection and visual cues without replacing gap detector.
4. Spatially overlapping gap + visual cues reinforce each other with boosted confidence.
5. Multi-view partial observations aggregate into a single PARTIALLY_OBSERVED window entity (Scenario A3).
"""

from __future__ import annotations

import pytest

from perception.architecture.parametric import FitRefused
from perception.architecture.windows import (
    VisualWindowEvidence,
    aggregate_multiview_window_observations,
    detect_visual_windows,
    detect_window,
    fuse_window_evidence,
)
from tests.test_window_roof_detectors import _wall_inliers_with_window, _wall_plane


def test_closed_window_without_empty_gap_detected_via_visual_cues():
    """A closed/shut window has points on the glass and does not form an empty gap in the wall points.

    Gap detector refuses with WindowRefused; fused detector recognises the window
    with is_closed=True (shut window != wall in WorldIR).
    """
    # 1. Solid wall plane with points everywhere (no empty gap)
    step = 0.05
    pts = [(i * step, 0.0, j * step) for i in range(81) for j in range(61)]
    wall = _wall_plane(pts)

    # Gap detector alone refuses because there is no geometric empty hole in the points
    with pytest.raises(FitRefused):
        detect_window(wall)

    # 2. Visual detector detects closed window from glass reflection and frame detection
    visual_cues = [
        {
            "wall_plane_id": wall.plane_id,
            "position": (2.0, 0.0, 1.5),
            "width_m": 1.4,
            "height_m": 1.2,
            "sill_height_m": 0.9,
            "confidence": 0.88,
            "is_closed": True,
            "has_frame": True,
            "has_mullions": True,
            "glass_sheen": 0.85,
        }
    ]
    vis_windows = detect_visual_windows(wall, visual_cues)
    assert len(vis_windows) == 1
    assert vis_windows[0].is_closed is True
    assert vis_windows[0].has_mullions is True

    # 3. Fuse evidence: produces WindowFit even without geometric point gap
    fused = fuse_window_evidence(gap_fit=None, visual_fits=vis_windows, wall=wall)
    assert len(fused) == 1
    win = fused[0]
    assert win.kind == "window"
    assert win.is_closed is True
    assert win.has_frame is True
    assert win.has_mullions is True
    assert win.source == "visual"
    assert win.width_m == pytest.approx(1.4)
    assert win.sill_height_m == pytest.approx(0.9)


def test_open_window_gap_and_visual_cues_reinforce_confidence():
    """When both geometric gap and visual window cues are present, fusion boosts confidence."""
    # Opening in wall points: 1.2m wide, sill at 0.9m
    inliers = _wall_inliers_with_window(1.4, 2.6, 0.9, 1.9)
    wall = _wall_plane(inliers)

    gap_fit = detect_window(wall)
    assert gap_fit.kind == "window"
    base_conf = gap_fit.confidence

    visual_cues = [
        {
            "wall_plane_id": wall.plane_id,
            "position": (2.0, 0.0, 1.4),
            "width_m": 1.2,
            "height_m": 1.0,
            "sill_height_m": 0.9,
            "confidence": 0.80,
            "is_closed": False,
        }
    ]
    vis_windows = detect_visual_windows(wall, visual_cues)
    fused = fuse_window_evidence(gap_fit=gap_fit, visual_fits=vis_windows, wall=wall)

    assert len(fused) == 1
    win = fused[0]
    assert win.source == "fused"
    # Multi-cue confirmation reinforces confidence
    assert win.confidence >= base_conf
    assert win.confidence >= 0.80


def test_scenario_a3_multiview_partial_observations():
    """Scenario A3: Window partly visible in 3 images resolves into one Entity, PARTIALLY_OBSERVED."""
    obs1 = VisualWindowEvidence(
        evidence_id="ev_cam1",
        wall_plane_id="wall_east",
        position=(1.5, 0.0, 1.4),
        width_m=1.2,
        height_m=1.2,
        sill_height_m=0.8,
        confidence=0.75,
        observed_fraction=0.45,
        image_ids=("cam1",),
    )
    obs2 = VisualWindowEvidence(
        evidence_id="ev_cam2",
        wall_plane_id="wall_east",
        position=(1.52, 0.0, 1.39),
        width_m=1.21,
        height_m=1.19,
        sill_height_m=0.81,
        confidence=0.79,
        observed_fraction=0.55,
        image_ids=("cam2",),
    )
    obs3 = VisualWindowEvidence(
        evidence_id="ev_cam3",
        wall_plane_id="wall_east",
        position=(1.49, 0.0, 1.41),
        width_m=1.19,
        height_m=1.21,
        sill_height_m=0.79,
        confidence=0.82,
        observed_fraction=0.30,
        image_ids=("cam3",),
    )

    fit, coverage_state = aggregate_multiview_window_observations([obs1, obs2, obs3], window_id="win-east")
    assert fit.kind == "window"
    assert coverage_state == "PARTIALLY_OBSERVED"
    assert fit.coverage_state == "PARTIALLY_OBSERVED"
    assert len(fit.evidence_ids) == 3
    assert fit.wall_plane_id == "wall_east"
