"""The DINOv2 appearance embedder and its pipeline wiring: real-model behaviour is checked live, the logic here is checked
with a stand-in model so it runs anywhere."""
from __future__ import annotations

import numpy as np
import pytest

from perception.instances.dinov2_hf import Dinov2Embedder, Dinov2Unavailable


class _FakeOut:
    def __init__(self, cls):
        import torch

        self.last_hidden_state = torch.as_tensor(cls)[:, None, :]


class _FakeModel:
    """Embeds a crop as its per-channel mean, so equal-coloured crops match and differently coloured ones do not."""

    def __call__(self, pixel_values):
        return _FakeOut(pixel_values.mean(dim=(2, 3)).numpy())


def _embedder():
    e = Dinov2Embedder()
    e._model, e._device = _FakeModel(), "cpu"
    return e


def _image():
    img = np.zeros((120, 160, 3), dtype=np.uint8)
    img[:, :80] = (200, 30, 30)    # red half
    img[:, 80:] = (30, 30, 200)    # blue half
    return img


def _mask(x0, x1):
    m = np.zeros((120, 160), dtype=bool)
    m[10:110, x0:x1] = True
    return m


def test_one_unit_vector_per_mask_in_order():
    vecs = _embedder().embed_masked_crops(_image(), [_mask(0, 80), _mask(80, 160), _mask(0, 80)])
    assert len(vecs) == 3 and all(v is not None for v in vecs)
    assert all(abs(np.linalg.norm(v) - 1.0) < 1e-5 for v in vecs)
    assert float(vecs[0] @ vecs[2]) > 0.999          # same region -> same vector
    assert float(vecs[0] @ vecs[1]) < float(vecs[0] @ vecs[2])


def test_a_mask_too_small_to_describe_an_object_gets_none_not_a_made_up_vector():
    tiny = np.zeros((120, 160), dtype=bool)
    tiny[5:8, 5:8] = True
    vecs = _embedder().embed_masked_crops(_image(), [tiny, _mask(0, 80)])
    assert vecs[0] is None and vecs[1] is not None


def test_a_mask_of_the_wrong_size_is_refused_rather_than_resized_silently():
    assert _embedder().embed_masked_crops(_image(), [np.ones((10, 10), dtype=bool)]) == [None]


def test_missing_checkpoint_raises_unavailable_with_the_fix():
    with pytest.raises(Dinov2Unavailable, match="snapshot_download"):
        Dinov2Embedder(model_id="facebook/no-such-dinov2-checkpoint-xyz").embed_masked_crops(_image(), [_mask(0, 80)])


def test_pipeline_helper_geometry_only_when_disabled():
    from engine.pipeline.vertical_slice import _region_appearance

    vectors, facts = _region_appearance(None, [], [], [])
    assert vectors == {} and facts["requested"] is None and facts["embedded"] == 0


def test_pipeline_helper_rejects_an_unknown_kind():
    from engine.pipeline.vertical_slice import VerticalSliceError, _region_appearance

    with pytest.raises(VerticalSliceError, match="perception_appearance"):
        _region_appearance("clip", [], [], [])


def test_pipeline_helper_failure_is_a_recorded_skip_not_a_crash(monkeypatch):
    from engine.pipeline.vertical_slice import _region_appearance

    class _Seg:
        evidence_id = "e1"
        regions = [type("R", (), {"region_id": "r1", "mask": [[True]]})()]

    class _Hyp:
        evidence_id, region_id = "e1", "r1"

    class _Item:
        evidence_id = "e1"

    monkeypatch.setattr("engine.pipeline.open_vocab_furniture.item_path", lambda item: "no/such/photo.jpg")
    vectors, facts = _region_appearance("dinov2", [_Item()], [_Seg()], [_Hyp()])
    assert vectors == {}                       # all-or-nothing: no half-embedded run
    assert facts["fallback_reason"] and facts["embedded"] == 0
