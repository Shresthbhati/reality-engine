"""B06: RLE masks round-trip exactly, including all-false / all-true, and survive save/load."""
import numpy as np

from world_ir.mask_store import MaskStore, decode, encode


def test_roundtrip_random_and_degenerate():
    r = np.random.default_rng(0)
    for m in [r.random((17, 23)) > 0.5, np.zeros((5, 7), bool), np.ones((5, 7), bool)]:
        assert np.array_equal(decode(encode(m), m.shape), m)


def test_store_save_load_rows_resolve(tmp_path):
    r = np.random.default_rng(1)
    masks = [r.random((8, 9)) > 0.4, r.random((12, 5)) > 0.6]
    s = MaskStore()
    rows = [s.add(m) for m in masks]
    s.save(tmp_path / "m.npz")
    t = MaskStore.load(tmp_path / "m.npz")
    assert len(t) == 2
    for row, m in zip(rows, masks):
        assert np.array_equal(t.get(row), m)
