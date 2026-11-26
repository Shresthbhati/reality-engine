"""A5 mask-contract tests: ObservationRecord carries segmentation masks,
persist_masks stamps mask_ref into MaskStore, and masks survive save/load.

These tests exercise the production mask path end-to-end without needing the
SAM 2 weights (a fake bool array stands in for a model-produced mask).
"""
from __future__ import annotations

import numpy as np
import pytest

from perception.engine_evidence import (
    CoordinateFrame,
    ModelIdentity,
    ObservationRecord,
    _mask_descriptor,
)
from perception.mask_persistence import load_mask_store, persist_masks, save_mask_store
from world_ir.mask_store import MaskStore, decode, encode


def _make_record(mask, mask_ref=None, label="segmentation_mask") -> ObservationRecord:
    return ObservationRecord(
        evidence_id=f"sam2:img_a.png:0:{label}",
        source_image_id="img_a.png",
        coordinate_frame=CoordinateFrame(frame_id="image", scale_state="image", units="pixels"),
        model_id=ModelIdentity(
            engine_id="segmentation", architecture_id="sam2", checkpoint_id="facebook/sam2.1-hiera-small",
            license="Apache-2.0", hardware_class="gpu"),
        label=label,
        category="segmentation",
        confidence=None,
        mask=mask,
        mask_ref=mask_ref,
        metadata={"mask_area_px": int(np.asarray(mask).sum()) if mask is not None else 0},
    )


def test_mask_descriptor_none_and_present():
    assert _mask_descriptor(None) is None
    m = np.zeros((4, 6), bool); m[1:3, 1:5] = True
    d = _mask_descriptor(m)
    assert d == {"shape": [4, 6], "present": True, "coverage": 0.3333}


def test_observation_record_to_dict_keeps_mask_descriptor_and_ref():
    m = np.ones((8, 8), bool)
    rec = _make_record(m, mask_ref=3)
    d = rec.to_dict()
    assert d["mask"] == {"shape": [8, 8], "present": True, "coverage": 1.0}
    assert d["mask_ref"] == 3


def test_observation_record_to_dict_mask_absent():
    rec = _make_record(None)
    d = rec.to_dict()
    assert d["mask"] is None
    assert d["mask_ref"] is None


def test_persist_masks_stamps_mask_ref_and_round_trips(tmp_path):
    m1 = np.random.default_rng(0).random((10, 12)) > 0.5
    m2 = np.zeros((5, 5), bool)
    recs = [_make_record(m1), _make_record(m2), _make_record(None)]
    store = MaskStore()
    out = persist_masks(recs, store)

    assert len(out) == 3
    assert out[0].mask_ref == 0
    assert out[1].mask_ref == 1
    assert out[2].mask_ref is None  # no mask -> untouched
    assert np.array_equal(out[0].mask, m1)
    assert np.array_equal(out[1].mask, m2)
    assert out[2].mask is None

    # Store encodes/decodes exactly.
    assert np.array_equal(decode(encode(m1), m1.shape), m1)
    assert np.array_equal(decode(encode(m2), m2.shape), m2)

    # Persisted store is recoverable by a fresh reader.
    p = tmp_path / "masks.npz"
    save_mask_store(store, p)
    loaded = load_mask_store(p)
    assert len(loaded) == 2
    assert np.array_equal(loaded.get(0), m1)
    assert np.array_equal(loaded.get(1), m2)


def test_persist_masks_returns_new_list_input_unchanged():
    m = np.ones((3, 3), bool)
    rec = _make_record(m)
    store = MaskStore()
    out = persist_masks([rec], store)
    assert out is not [rec]
    assert rec.mask_ref is None  # input frozen record untouched
    assert out[0].mask_ref == 0


def test_mask_ref_flows_through_to_dict_for_matrix():
    m = np.random.default_rng(1).random((20, 30)) > 0.7
    recs = [_make_record(m)]
    store = MaskStore()
    stamped = persist_masks(recs, store)[0]
    d = stamped.to_dict()
    assert d["mask_ref"] == 0
    assert d["mask"]["shape"] == [20, 30]
    assert 0.0 <= d["mask"]["coverage"] <= 1.0
