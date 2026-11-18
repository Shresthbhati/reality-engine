"""Appearance-embedding store (BUILD_SPEC B07 / 2.3): one float16 matrix, L2-normalised at write time, memory-mapped on read.

Observations hold a row number (``ObservationTable.emb_row``) or a reference ``"<artifact_sha>#<row>"``, never a Python list of
floats. A float16 unit vector keeps cosine similarity accurate to ~1e-3, which is far below the gates that use it; the matrix
product is done in float32.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np


class EmbeddingStore:
    def __init__(self, dim: Optional[int] = None) -> None:
        self.dim = dim
        self._rows: list[np.ndarray] = []          # float16 unit vectors while building
        self._mm: Optional[np.ndarray] = None      # read-only (N, D) float16, possibly memory-mapped, once loaded

    def __len__(self) -> int:
        return len(self._mm) if self._mm is not None else len(self._rows)

    def add(self, vec: Sequence[float]) -> int:
        """Normalise ``vec`` to unit length, store it as float16, return its row. Zero, non-finite or wrong-sized vectors
        are refused (ValueError): a vector that cannot be compared must not get a row that looks comparable."""
        if self._mm is not None:
            raise RuntimeError("this store was loaded read-only")
        v = np.asarray(vec, dtype=np.float32).ravel()
        if not np.all(np.isfinite(v)):
            raise ValueError("embedding contains NaN or inf")
        n = float(np.linalg.norm(v))
        if n < 1e-8:
            raise ValueError("embedding has zero length")
        if self.dim is None:
            self.dim = int(v.size)
        elif v.size != self.dim:
            raise ValueError(f"embedding has {v.size} dimensions, the store holds {self.dim}")
        self._rows.append((v / n).astype(np.float16))
        return len(self._rows) - 1

    def matrix(self) -> np.ndarray:
        if self._mm is not None:
            return self._mm
        return np.stack(self._rows) if self._rows else np.zeros((0, self.dim or 0), np.float16)

    def get(self, row: int) -> np.ndarray:
        return np.asarray(self.matrix()[row], dtype=np.float32)

    def cosine(self, rows_a: Sequence[int], rows_b: Sequence[int]) -> np.ndarray:
        """(len(rows_a), len(rows_b)) cosine similarities. Reads only the requested rows (cheap on a memory map)."""
        m = self.matrix()
        a = np.asarray(m[np.asarray(rows_a, dtype=np.int64)], dtype=np.float32)
        b = np.asarray(m[np.asarray(rows_b, dtype=np.int64)], dtype=np.float32)
        return a @ b.T

    def save(self, path: str | Path) -> None:
        np.save(path, self.matrix(), allow_pickle=False)

    @classmethod
    def load(cls, path: str | Path, mmap: bool = True) -> "EmbeddingStore":
        m = np.load(path, mmap_mode="r" if mmap else None, allow_pickle=False)
        if m.ndim != 2 or m.dtype != np.float16:
            raise ValueError(f"expected a 2-D float16 matrix, got {m.dtype} {m.shape}")
        s = cls(dim=int(m.shape[1]))
        s._mm = m
        return s


def embedding_ref(artifact_sha: str, row: int) -> str:
    return f"{artifact_sha}#{int(row)}"


def parse_embedding_ref(ref: str) -> Tuple[str, int]:
    sha, sep, row = ref.rpartition("#")
    if not sep or not sha or not row.isdigit():
        raise ValueError(f"not an embedding reference: {ref!r}")
    return sha, int(row)
