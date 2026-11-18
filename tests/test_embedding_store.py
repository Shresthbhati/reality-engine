"""B07 / RE-PERF-000009: embeddings live in one float16 unit-vector matrix, memory-mapped on read."""
import numpy as np
import pytest

from world_ir.embedding_store import EmbeddingStore, embedding_ref, parse_embedding_ref


def _store(n=20, d=384, seed=0):
    rng = np.random.default_rng(seed)
    vecs = rng.normal(size=(n, d)).astype(np.float32)
    s = EmbeddingStore()
    rows = [s.add(v) for v in vecs]
    return s, vecs, rows


def test_rows_are_float16_unit_vectors_and_cosine_matches_float32_truth():
    s, vecs, rows = _store()
    assert rows == list(range(20)) and s.matrix().dtype == np.float16 and s.dim == 384
    assert np.allclose(np.linalg.norm(s.matrix().astype(np.float32), axis=1), 1.0, atol=2e-3)
    unit = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    assert np.abs(s.cosine([0, 1, 2], [3, 4]) - unit[:3] @ unit[3:5].T).max() < 2e-3


def test_save_and_load_memory_mapped_round_trip(tmp_path):
    s, _, _ = _store()
    p = tmp_path / "emb.npy"
    s.save(p)
    m = EmbeddingStore.load(p)
    assert isinstance(m.matrix(), np.memmap) and len(m) == 20 and m.dim == 384
    assert np.array_equal(m.matrix(), s.matrix())
    assert np.allclose(m.cosine([5], [5]), 1.0, atol=2e-3)
    with pytest.raises(RuntimeError):
        m.add(np.ones(384))                                     # a loaded store is read-only


def test_vectors_that_cannot_be_compared_are_refused_not_given_a_row():
    s = EmbeddingStore()
    s.add(np.ones(8))
    for bad in (np.zeros(8), np.full(8, np.nan), np.ones(5)):
        with pytest.raises(ValueError):
            s.add(bad)
    assert len(s) == 1


def test_wrong_dtype_files_are_rejected(tmp_path):
    np.save(tmp_path / "x.npy", np.zeros((3, 4), np.float32))
    with pytest.raises(ValueError):
        EmbeddingStore.load(tmp_path / "x.npy")


def test_embedding_refs_round_trip():
    assert parse_embedding_ref(embedding_ref("abc123", 7)) == ("abc123", 7)
    for bad in ("abc", "#3", "abc#x"):
        with pytest.raises(ValueError):
            parse_embedding_ref(bad)
