"""The ladder swaps a structureless multi-view result for the learned single-image layout, and only then.

Real world that motivated it: 9 of 10 AI-generated photos registered on only 206 sparse points; the point-plane stage
promoted five floors (one above the ceiling) and one wall. A credible multi-view model must be left alone.
"""
from types import SimpleNamespace

import pytest

import tests.test_surface_layout as room
from engine.pipeline.progressive import EvidenceInput, run_progressive
from evidence.image_check import inspect_image
from evidence.session import EvidenceItem, EvidenceKind
from tests.test_progressive_units import _noise_photo
from world_ir import WorldIR

N_PHOTOS = 3


def _inputs(tmp_path):
    out = []
    for k in range(N_PHOTOS):
        p = _noise_photo(tmp_path / f"p{k}.jpg", seed=k)
        out.append(EvidenceInput(item=EvidenceItem(id=f"ev-{k}", kind=EvidenceKind.PHOTO, source_uri=p.as_uri()),
                                 facts=inspect_image(p), name=p.name, path=p))
    return out


def _slice(n_points, registered=N_PHOTOS):
    def fn(items, options):
        return SimpleNamespace(
            world=WorldIR(), camera_poses=[(f"ev-{k}", (float(k), 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)) for k in range(registered)],
            registration_status="success", cameras_registered=registered, cameras_input=N_PHOTOS, scale_state="relative",
            meters_per_unit=None, stage_facts={}, points=[(0.0, 0.0, float(i)) for i in range(n_points)],
            compile=SimpleNamespace(measurements_count=0, relationships_count=0))
    return fn


@pytest.fixture()
def analyzer(monkeypatch):
    monkeypatch.setattr(room, "HFOV", 60.0)
    depth, masks = room._render(pitch_deg=5.0)
    calls = []

    def analyze(item):
        calls.append(item.id)
        return SimpleNamespace(depth_m=depth, masks=masks)

    analyze.calls = calls
    return analyze


def test_a_structureless_multiview_model_is_replaced_by_the_layout(tmp_path, analyzer):
    res = run_progressive(_inputs(tmp_path), vs_options=None, vertical_slice_fn=_slice(n_points=60), layout_analyzer=analyzer)
    assert res.level == 0 and res.registration_status == "partial"
    assert [a["outcome"] for a in res.attempts if a["level"] == 2] == ["rejected"]
    assert "not credible" in res.attempts[0]["detail"]
    kinds = [e.type.value for e in res.world.entities.values()]
    assert kinds.count("floor") == 1 and kinds.count("ceiling") == 1 and kinds.count("window") == 1
    assert any("room shell built from one photograph" in d for d in res.degraded)  # the other photos are accounted for


def test_a_credible_multiview_model_is_left_alone(tmp_path, analyzer):
    res = run_progressive(_inputs(tmp_path), vs_options=None, vertical_slice_fn=_slice(n_points=400), layout_analyzer=analyzer)
    assert res.level == 2 and analyzer.calls == []  # not even analysed: no cost when multi-view works


def test_layout_models_unavailable_falls_back_to_the_classic_path_and_says_so(tmp_path):
    import numpy as np

    from perception.depth.interface import DepthMap

    def unavailable(item):
        raise RuntimeError("weights not cached")

    h, w = 240, 320
    dm = DepthMap(evidence_id="x", width=w, height=h,
                  values=(np.linspace(0.05, 0.9, h)[:, None] + np.linspace(0.0, 0.1, w)[None, :]).tolist())
    res = run_progressive(_inputs(tmp_path)[:1], vs_options=None, bootstrap_depth_map=dm, layout_analyzer=unavailable)
    assert res.level == 0  # the classic single-view path still produced a world
    assert any("learned layout unavailable" in a["detail"] for a in res.attempts if a["outcome"] == "failed")


def test_a_weak_multiview_model_is_kept_when_the_layout_cannot_replace_it(tmp_path):
    """Never trade a weak model for nothing: unavailable models leave the multi-view result in place."""
    def unavailable(item):
        raise RuntimeError("weights not cached")

    res = run_progressive(_inputs(tmp_path), vs_options=None, vertical_slice_fn=_slice(n_points=60), layout_analyzer=unavailable)
    assert res.level == 2 and res.vs_result is not None
    assert [a["outcome"] for a in res.attempts if a["level"] == 2] == ["succeeded"]
    assert any("learned layout unavailable" in a["detail"] for a in res.attempts if a["level"] == 0)
