"""Run-length-encoded mask store (BUILD_SPEC B06 / 2.3): one .npz per job, never a PNG per mask."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def encode(mask: np.ndarray) -> np.ndarray:
    """Column-major (COCO-style) run lengths, starting with a run of False (possibly 0)."""
    flat = np.asarray(mask, dtype=bool).ravel(order="F")
    if flat.size == 0:
        return np.zeros(0, np.uint32)
    change = np.flatnonzero(np.diff(flat.astype(np.int8))) + 1
    bounds = np.concatenate([[0], change, [flat.size]])
    runs = np.diff(bounds).astype(np.uint32)
    return runs if not flat[0] else np.concatenate([np.zeros(1, np.uint32), runs])


def decode(counts: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    vals = (np.arange(len(counts)) % 2 == 1)
    flat = np.repeat(vals, np.asarray(counts, dtype=np.int64))
    if flat.size != shape[0] * shape[1]:
        raise ValueError("run lengths do not match the mask shape")
    return flat.reshape(shape, order="F")


class MaskStore:
    def __init__(self) -> None:
        self._counts: list[np.ndarray] = []
        self._shapes: list[tuple[int, int]] = []

    def __len__(self) -> int:
        return len(self._counts)

    def add(self, mask: np.ndarray) -> int:
        self._counts.append(encode(mask)); self._shapes.append(tuple(mask.shape))
        return len(self._counts) - 1

    def get(self, row: int) -> np.ndarray:
        return decode(self._counts[row], self._shapes[row])

    def save(self, path: str | Path) -> None:
        lens = np.array([len(c) for c in self._counts], dtype=np.int64)
        np.savez_compressed(
            path,
            counts=np.concatenate(self._counts) if self._counts else np.zeros(0, np.uint32),
            offsets=np.concatenate([[0], np.cumsum(lens)]).astype(np.int64),
            shape=np.asarray(self._shapes, dtype=np.uint16).reshape(-1, 2),
        )

    @classmethod
    def load(cls, path: str | Path) -> "MaskStore":
        z = np.load(path)
        s = cls()
        for i in range(len(z["shape"])):
            s._counts.append(z["counts"][z["offsets"][i]:z["offsets"][i + 1]])
            s._shapes.append((int(z["shape"][i][0]), int(z["shape"][i][1])))
        return s
