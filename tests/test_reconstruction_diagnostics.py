"""Reconstruction diagnostics: per-camera support and structured, correction-ready findings.

Synthetic cases pin each rule; one test runs the analysis on a REAL reconstruction (the COLMAP reference model shipped
with the committed South Building dataset) so the numbers are not only ever seen on toy data.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from reconstruction.backend.interface import ReconstructedCameraPose, ReconstructedPoint
from reconstruction.diagnostics import (
    VERY_WEAK_OBSERVATIONS,
    WEAK_OBSERVATIONS,
    analyze,
    camera_support,
)
from world_ir.schema_v1 import Uncertainty

ROOT = Path(__file__).resolve().parents[1]


def _pose(eid, pos=(0.0, 0.0, 0.0)):
    return ReconstructedCameraPose(evidence_id=eid, position=pos, rotation=(1.0, 0.0, 0.0, 0.0))


def _pts(n, seen_by, conf=0.9, start=0):
    return [
        ReconstructedPoint(position=(float(i), 0.0, 0.0), track_id=f"t{start + i}", source_evidence_ids=list(seen_by),
                           uncertainty=Uncertainty(confidence=conf))
        for i in range(n)
    ]


def _types(report, kind):
    return [d for d in report["diagnostics"] if d["type"] == kind]


def test_support_counts_observations_per_registered_camera():
    points = _pts(100, ["a", "b"]) + _pts(20, ["b", "c"], start=100)
    support = camera_support(points, [_pose("a"), _pose("b"), _pose("c")])
    assert {k: v["observations"] for k, v in support.items()} == {"a": 100, "b": 120, "c": 20}
    assert support["a"]["mean_point_confidence"] == 0.9


def test_weak_and_unregistered_cameras_are_reported_with_measurements_and_safety():
    points = _pts(100, ["a", "b"]) + _pts(20, ["b", "c"], start=100) + _pts(5, ["c", "d"], start=120)
    report = analyze(points, [_pose("a"), _pose("b"), _pose("c"), _pose("d")], ["a", "b", "c", "d", "e"])

    weak = {d["subject"]: d for d in _types(report, "WEAKLY_SUPPORTED_CAMERA")}
    assert set(weak) == {"c", "d"}                       # 25 and 5 observations, both under 30; a and b are fine
    assert weak["c"]["severity"] == "warning" and weak["c"]["measured"]["observations"] == 25
    assert weak["d"]["severity"] == "error" and weak["d"]["measured"]["observations"] == 5
    assert weak["d"]["threshold"] == {"warning_below": WEAK_OBSERVATIONS, "error_below": VERY_WEAK_OBSERVATIONS}
    assert weak["d"]["safe_to_auto_correct"] is True, "excluding a weak camera from dense/measurement loses nothing"

    unregistered = _types(report, "UNREGISTERED_IMAGE")
    assert [d["subject"] for d in unregistered] == ["e"]
    assert unregistered[0]["safe_to_auto_correct"] is False, "a missing photo needs new evidence, not a quiet fix"
    assert report["summary"]["registered"] == 4 and report["summary"]["input"] == 5
    assert report["summary"]["min_observations_per_camera"] == 5
    assert json.dumps(report), "the report must be JSON-serialisable"


def test_a_healthy_model_has_no_diagnostics():
    points = _pts(200, ["a", "b", "c"])
    report = analyze(points, [_pose("a"), _pose("b"), _pose("c")], ["a", "b", "c"])
    assert report["diagnostics"] == []
    assert report["summary"]["by_severity"] == {"info": 0, "warning": 0, "error": 0}


def test_invalid_numbers_and_inconsistent_references_are_errors_never_auto_corrected():
    points = _pts(60, ["a", "b"]) + [
        ReconstructedPoint(position=(math.nan, 0.0, 0.0), track_id="bad", source_evidence_ids=["a", "ghost"]),
        ReconstructedPoint(position=(0.0, math.inf, 0.0), track_id="bad2", source_evidence_ids=["a", "b"]),
    ]
    poses = [_pose("a"), _pose("b"), _pose("stranger", pos=(0.0, math.nan, 0.0))]
    report = analyze(points, poses, ["a", "b"])

    nf = _types(report, "NON_FINITE_GEOMETRY")[0]
    assert nf["severity"] == "error" and nf["measured"] == {"points": 2, "poses": 1}
    assert _types(report, "DANGLING_OBSERVATION")[0]["measured"] == {"observations": 1}
    assert _types(report, "UNKNOWN_CAMERA")[0]["measured"] == {"cameras": 1}
    for kind in ("NON_FINITE_GEOMETRY", "DANGLING_OBSERVATION", "UNKNOWN_CAMERA"):
        assert _types(report, kind)[0]["safe_to_auto_correct"] is False, kind


def test_low_registration_is_a_model_level_warning():
    ids = ["a", "b"] + list("cdefgh")
    report = analyze(_pts(100, ["a", "b"]), [_pose("a"), _pose("b")], ids)
    low = _types(report, "LOW_REGISTRATION")
    assert low and low[0]["subject"] == "reconstruction"
    assert low[0]["measured"] == {"registered": 2, "input": 8, "fraction": 0.25}


def test_empty_result_does_not_crash():
    report = analyze([], [], ["a", "b"])
    assert report["summary"]["registered"] == 0
    assert [d["type"] for d in report["diagnostics"]].count("UNREGISTERED_IMAGE") == 2


SPARSE = ROOT / "datasets" / "south_building" / "sparse"


@pytest.mark.skipif(not (SPARSE / "images.txt").exists(), reason="committed South Building reference model not present")
def test_real_reference_reconstruction_is_well_supported_everywhere():
    from reconstruction.backend.colmap_backend import _parse_images_txt, _parse_points3d_txt

    images_text = (SPARSE / "images.txt").read_text()
    lines = [ln for ln in images_text.splitlines() if ln.strip() and not ln.startswith("#")]
    id_to_name = {lines[i].split()[0]: lines[i].split()[9] for i in range(0, len(lines), 2)}
    poses = _parse_images_txt(images_text, {})
    points = _parse_points3d_txt((SPARSE / "points3D.txt").read_text(), id_to_name)

    report = analyze(points, poses, sorted(id_to_name.values()))
    s = report["summary"]
    # The committed reference holds the 32 committed photographs, all registered (it was solved with the full 128-photo
    # set, whose in-between photographs bridge the gaps that make a stride-4 subset register only 22 of 32).
    assert s["registered"] == s["input"] == len(id_to_name) == 32
    assert s["points"] > 40000
    # Measured on this model: median 2634 observations per camera, minimum 396, mean track length 5.76.
    assert s["median_observations_per_camera"] > 1000, s
    assert s["min_observations_per_camera"] > WEAK_OBSERVATIONS, s
    assert not _types(report, "WEAKLY_SUPPORTED_CAMERA") and not _types(report, "UNREGISTERED_IMAGE")
    for forbidden in ("NON_FINITE_GEOMETRY", "UNKNOWN_CAMERA"):
        assert not _types(report, forbidden), forbidden
    # The point tracks still cite the 96 photographs that were cropped out of the committed subset. That is a real
    # inconsistency for THIS 32-photo subset, and the diagnostics must say so rather than ignore it.
    assert _types(report, "DANGLING_OBSERVATION")[0]["measured"]["observations"] > 100000
