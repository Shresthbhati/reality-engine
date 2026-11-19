"""The DINOv2 appearance embedder and its pipeline wiring: real-model behaviour is checked live, the logic here is checked
with a stand-in model so it runs anywhere."""
from __future__ import annotations

import numpy as np
import pytest

from perception.instances.dinov2_hf import Dinov2Embedder, Dinov2Unavailable

# The stand-in model returns torch tensors, so these tests need torch (an optional extra, not in `.[dev]`).
pytest.importorskip("torch", reason="torch is not installed (optional extra)")


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


def test_pool_tokens_is_a_weighted_unit_mean_and_refuses_specks():
    import numpy as np

    from perception.instances.dinov2_hf import pool_tokens

    tokens = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]], np.float32)
    a, b, speck, wrong = pool_tokens(tokens, [
        np.array([1, 0, 0, 1], np.float32),         # the two x-tokens
        np.array([0, 1, 1, 0], np.float32),         # the two y-tokens
        np.array([0.2, 0, 0, 0.3], np.float32),     # under one patch of coverage in total
        np.ones(3, np.float32)])                    # wrong length
    assert np.allclose(a, [1, 0]) and np.allclose(b, [0, 1]) and speck is None and wrong is None


def test_embed_pooled_one_forward_separates_textures_in_the_real_model():
    import numpy as np
    import pytest

    from perception.instances.dinov2_hf import Dinov2Embedder, Dinov2Unavailable

    rng = np.random.default_rng(0)
    img = np.zeros((224, 448, 3), np.uint8)
    stripes = np.tile(np.array([[200, 30, 30], [30, 30, 200]], np.uint8).repeat(8, axis=0), (14, 1))[:224]   # horizontal bands
    img[:, :224] = np.repeat(stripes[:, None, :], 224, axis=1)[:, :224]
    img[:, 224:] = rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)                                         # noise
    left = np.zeros(img.shape[:2], bool); left[20:200, 20:200] = True
    left2 = np.zeros_like(left); left2[40:200, 30:190] = True
    right = np.zeros_like(left); right[20:200, 244:424] = True
    emb = Dinov2Embedder()
    try:
        v = emb.embed_pooled(img, [left, left2, right])
    except Dinov2Unavailable as exc:
        pytest.skip(str(exc))
    assert all(x is not None and abs(float(np.linalg.norm(x)) - 1.0) < 1e-4 for x in v)
    assert float(v[0] @ v[1]) > float(v[0] @ v[2])          # two views of the stripes beat stripes vs noise
