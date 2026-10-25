"""Level 3 (dense) is reached by MEASURED evidence, never by a photo count.

Model-free: reconstruction results are synthetic stand-ins with known properties, so every decision is provable.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

from engine.pipeline import dense_gate as dg


def _result(n_cams: int, n_pts: int = 600, track: int = 4, arc_deg: float = 120.0, n_input: int | None = None):
    """n_cams cameras on an arc of ``arc_deg`` around the origin; n_pts structure points each seen by ``track`` cameras."""
    cams = []
    for i in range(n_cams):
        a = math.radians(arc_deg) * (i / max(1, n_cams - 1)) - math.radians(arc_deg) / 2
        cams.append(SimpleNamespace(evidence_id=f"ev-{i}", position=(6 * math.sin(a), 0.0, -6 * math.cos(a)),
                                    rotation=(1.0, 0.0, 0.0, 0.0)))
    pts = [SimpleNamespace(position=(0.01 * i, 0.0, 0.0), source_evidence_ids=[f"ev-{k}" for k in range(track)])
           for i in range(n_pts)]
    return (SimpleNamespace(camera_poses=cams, points=pts, arc_deg=arc_deg),
            (n_input if n_input is not None else n_cams))


def _coverage(res):
    """The angular spread the cameras were placed with (the gate takes the measured number as an input)."""
    return res.arc_deg


def test_well_supported_sparse_model_escalates_with_every_criterion_reported():
    res, n = _result(6)
    facts = dg.dense_readiness(res, n, _coverage(res))
    assert facts["escalate"] is True and facts["thresholds_status"] == "EXPERIMENTAL"
    assert set(facts["criteria"]) == {"registration", "geometric support", "coverage", "structure"}
    assert all(c["passed"] for c in facts["criteria"].values())


def test_photo_count_alone_decides_nothing_three_photos_can_qualify_twenty_can_fail():
    few, n = _result(3)
    assert dg.dense_readiness(few, n, _coverage(few))["escalate"] is True      # 3 photos, strong measured support
    many, n = _result(20, arc_deg=8.0)                                          # 20 photos all looking from one spot
    facts = dg.dense_readiness(many, n, _coverage(many))
    assert facts["escalate"] is False
    assert "coverage" in " ".join(facts["reasons"])


def test_each_failed_criterion_is_named_with_its_measurement():
    res, n = _result(6, n_pts=100)
    f = dg.dense_readiness(res, n, _coverage(res))
    assert not f["escalate"] and "structure: 100 sparse structure points" in " ".join(f["reasons"])
    res, _ = _result(6)
    f = dg.dense_readiness(res, 10, _coverage(res))                              # only 6 of 10 photographs registered
    assert not f["escalate"] and "registration: 6 of 10" in " ".join(f["reasons"])
    res, n = _result(6, track=2)
    f = dg.dense_readiness(res, n, _coverage(res))
    assert not f["escalate"] and "geometric support" in " ".join(f["reasons"])
    res, n = _result(6)
    f = dg.dense_readiness(res, n, None)                                         # coverage not measurable -> never assumed
    assert not f["escalate"] and f["criteria"]["coverage"]["measured"] == "angular coverage not measurable"


def test_empty_reconstruction_never_escalates():
    f = dg.dense_readiness(SimpleNamespace(camera_poses=[], points=[]), 0, None)
    assert f["escalate"] is False and f["measured"]["registered"] == 0


def _measured_coverage(monkeypatch, res):
    monkeypatch.setattr("engine.pipeline.guidance.coverage_degrees", lambda poses, center: {"degrees": res.arc_deg})


def test_auto_options_point_the_dense_stage_at_the_session_workspace_and_use_scratch_output(tmp_path, monkeypatch):
    from engine.pipeline.vertical_slice import VerticalSliceOptions, _auto_dense_options

    ws = tmp_path / "staging"
    (ws / "images").mkdir(parents=True)
    (ws / "sparse" / "0").mkdir(parents=True)
    res, n = _result(6)
    _measured_coverage(monkeypatch, res)
    monkeypatch.setattr("reconstruction.dense_pipeline.dense_mvs_available", lambda binary: True)
    opts = VerticalSliceOptions(colmap_session=SimpleNamespace(staging=ws), dense_auto=True)
    out, facts, scratch = _auto_dense_options(res, n, opts)
    try:
        assert out.dense_mvs_enabled and out.dense_mvs_image_dir == ws / "images"
        assert out.dense_mvs_sparse_model_dir == ws / "sparse" / "0"
        assert out.dense_mvs_workspace == scratch and scratch.is_dir() and scratch != ws    # dense output never enters the session
        assert facts["escalate"] and "escalated" in facts["decision"]
    finally:
        import shutil

        shutil.rmtree(scratch, ignore_errors=True)


def test_auto_options_stay_sparse_without_a_workspace_or_when_the_evidence_does_not_justify_dense(tmp_path, monkeypatch):
    from engine.pipeline.vertical_slice import VerticalSliceOptions, _auto_dense_options

    res, n = _result(6)
    _measured_coverage(monkeypatch, res)
    out, facts, scratch = _auto_dense_options(res, n, VerticalSliceOptions(dense_auto=True))   # no COLMAP session
    assert not out.dense_mvs_enabled and scratch is None and not facts["escalate"]
    assert "dense inputs unavailable" in facts["decision"]
    weak, n = _result(6, n_pts=50)
    _measured_coverage(monkeypatch, weak)
    out, facts, scratch = _auto_dense_options(weak, n, VerticalSliceOptions(dense_auto=True))
    assert not out.dense_mvs_enabled and scratch is None and "not justified" in facts["decision"]


def _progressive(tmp_path, stage_facts):
    """run_progressive over three real (noise) photos with a stand-in multi-view slice reporting ``stage_facts``."""
    from engine.pipeline.progressive import EvidenceInput, run_progressive
    from evidence.image_check import inspect_image
    from evidence.session import EvidenceItem, EvidenceKind
    from tests.test_progressive_units import _noise_photo
    from world_ir import WorldIR

    tmp_path.mkdir(parents=True, exist_ok=True)
    inputs = []
    for k in range(3):
        p = _noise_photo(tmp_path / f"p{k}.jpg", seed=k)
        item = EvidenceItem(id=f"ev-{k}", kind=EvidenceKind.PHOTO, source_uri=p.as_uri())
        inputs.append(EvidenceInput(item=item, facts=inspect_image(p), name=p.name, path=p))

    def slice_fn(items, options):
        return SimpleNamespace(
            world=WorldIR(), camera_poses=[(f"ev-{k}", (float(k), 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)) for k in range(3)],
            registration_status="success", cameras_registered=3, cameras_input=3, scale_state="relative",
            meters_per_unit=None, stage_facts=stage_facts, points=[(0.0, 0.0, float(i)) for i in range(50)],
            compile=SimpleNamespace(measurements_count=0, relationships_count=0))

    return run_progressive(inputs, vs_options=None, vertical_slice_fn=slice_fn)


def test_level_3_is_granted_only_when_dense_actually_ran_and_added_points(tmp_path):
    gate = {"escalate": True, "reasons": ["every measured criterion passed"]}
    res = _progressive(tmp_path, {"dense_gate": gate, "dense_mvs": {"status": "ran", "dense_points_added": 1234}})
    assert res.level == 3 and res.level_name == "dense reconstruction"
    att = [a for a in res.attempts if a["level"] == 3][0]
    assert att["outcome"] == "succeeded" and "1234 dense points" in att["detail"]


def test_a_failed_or_refused_dense_run_keeps_the_sparse_level_and_says_why(tmp_path):
    gate = {"escalate": True, "reasons": ["every measured criterion passed"]}
    failed = _progressive(tmp_path / "a", {"dense_gate": gate,
                                           "dense_mvs": {"status": "failed", "note": "DenseMVSRunError: out of memory"}})
    assert failed.level == 2
    att = [a for a in failed.attempts if a["level"] == 3][0]
    assert att["outcome"] == "failed" and "out of memory" in att["detail"]

    skipped = _progressive(tmp_path / "b", {"dense_gate": {"escalate": False, "reasons": ["coverage: cameras span 4 degrees"]},
                                            "dense_mvs": {"status": "skipped"}})
    assert skipped.level == 2
    att = [a for a in skipped.attempts if a["level"] == 3][0]
    assert att["outcome"] == "skipped" and "coverage" in att["detail"]

    off = _progressive(tmp_path / "c", {"dense_mvs": {"status": "skipped"}})             # dense_auto off: no gate at all
    assert off.level == 2 and not [a for a in off.attempts if a["level"] == 3]
