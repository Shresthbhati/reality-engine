"""Sparse fused geometry as a voxel hash (BUILD_SPEC B11 / 2.4).

Points from several sources (SfM, MVS, mono-depth, device depth) integrate with inverse-variance weights.
A voxel is *observed surface* only with count >= MIN_OBSERVED; absence of a voxel means NOT observed, never "empty".
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SFM, MVS, MONO_DEPTH, DEVICE_DEPTH = 1, 2, 4, 8
MIN_OBSERVED = 3
_OFF = 2 ** 20


def _keys(points: np.ndarray, voxel_m: float) -> np.ndarray:
    ijk = np.floor(points / voxel_m).astype(np.int64) + _OFF
    if ijk.min() < 0 or ijk.max() >= 2 ** 21:
        raise ValueError("points outside the packable voxel range for this voxel size")
    return (ijk[:, 0] << 42) | (ijk[:, 1] << 21) | ijk[:, 2]


@dataclass
class VoxelCloud:
    voxel_m: float = 0.02
    key: np.ndarray = field(default_factory=lambda: np.empty(0, np.int64))
    mean: np.ndarray = field(default_factory=lambda: np.empty((0, 3), np.float32))
    rgb: np.ndarray = field(default_factory=lambda: np.empty((0, 3), np.uint8))
    count: np.ndarray = field(default_factory=lambda: np.empty(0, np.uint32))
    sources: np.ndarray = field(default_factory=lambda: np.empty(0, np.uint8))
    var_m2: np.ndarray = field(default_factory=lambda: np.empty(0, np.float32))
    _wsum: np.ndarray = field(default_factory=lambda: np.empty(0, np.float64), repr=False)
    _wpos: np.ndarray = field(default_factory=lambda: np.empty((0, 3), np.float64), repr=False)
    _crgb: np.ndarray = field(default_factory=lambda: np.empty((0, 3), np.float64), repr=False)

    def integrate(self, points: np.ndarray, rgb: np.ndarray, sigma_m: float, source_bit: int) -> None:
        p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if len(p) == 0:
            return
        c = np.asarray(rgb, dtype=np.float64).reshape(-1, 3)
        w = np.full(len(p), 1.0 / float(sigma_m) ** 2)
        allk = np.concatenate([self.key, _keys(p, self.voxel_m)])
        uk, inv = np.unique(allk, return_inverse=True)
        n_old = len(self.key)
        wsum = np.zeros(len(uk)); wpos = np.zeros((len(uk), 3)); crgb = np.zeros((len(uk), 3))
        cnt = np.zeros(len(uk), np.uint32); src = np.zeros(len(uk), np.uint8)
        if n_old:
            np.add.at(wsum, inv[:n_old], self._wsum); np.add.at(wpos, inv[:n_old], self._wpos)
            np.add.at(crgb, inv[:n_old], self._crgb); np.add.at(cnt, inv[:n_old], self.count)
            np.bitwise_or.at(src, inv[:n_old], self.sources)
        np.add.at(wsum, inv[n_old:], w); np.add.at(wpos, inv[n_old:], p * w[:, None])
        np.add.at(crgb, inv[n_old:], c); np.add.at(cnt, inv[n_old:], 1)
        np.bitwise_or.at(src, inv[n_old:], np.uint8(source_bit))
        self.key, self._wsum, self._wpos, self._crgb, self.count, self.sources = uk, wsum, wpos, crgb, cnt, src
        self.mean = (wpos / wsum[:, None]).astype(np.float32)
        self.rgb = np.clip(crgb / cnt[:, None], 0, 255).astype(np.uint8)
        self.var_m2 = (1.0 / wsum).astype(np.float32)

    def lookup(self, points: np.ndarray) -> np.ndarray:
        """Row of each point's voxel, or -1 when that voxel is not observed."""
        k = _keys(np.asarray(points, dtype=np.float64).reshape(-1, 3), self.voxel_m)
        if len(self.key) == 0:
            return np.full(len(k), -1, np.int64)
        pos = np.minimum(np.searchsorted(self.key, k), len(self.key) - 1)
        return np.where(self.key[pos] == k, pos, -1)

    def observed_mask(self) -> np.ndarray:
        return self.count >= MIN_OBSERVED
