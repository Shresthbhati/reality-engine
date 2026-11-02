"""Classical SIFT matcher: SIFT + Lowe ratio + mutual nearest neighbour + fundamental-matrix RANSAC (OpenCV).

This is the logic `scene_identity.SiftPairVerifier` always used, moved behind `IMatcher` so a learned matcher can be
benchmarked against it on identical inputs. Defaults are the measured ones recorded in
docs/implementation/SCENE_IDENTITY.md (ratio 0.75 / 0.5 px / mutual NN). Features are cached per image.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict

import numpy as np

from reconstruction.matching.interface import MatchResult


class SiftMatcher:
    name = "sift@opencv"
    license = "Apache-2.0"
    non_commercial = False

    def __init__(self, paths: Dict[str, Path], max_side: int = 960, ratio: float = 0.75, ransac_px: float = 0.5):
        self.paths, self.max_side, self.ratio, self.ransac_px = dict(paths), max_side, ratio, ransac_px
        self._feat: Dict[str, tuple] = {}

    def _features(self, eid: str):
        if eid not in self._feat:
            import cv2

            img = cv2.imread(str(self.paths[eid]), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"unreadable image for evidence {eid}: {self.paths[eid]}")
            scale = self.max_side / max(img.shape)
            if scale < 1:
                img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            kp, des = cv2.SIFT_create(nfeatures=4000).detectAndCompute(img, None)
            self._feat[eid] = (np.float32([k.pt for k in kp]), des)
        return self._feat[eid]

    def match(self, a: str, b: str) -> MatchResult:
        import cv2

        t0 = time.perf_counter()
        (pa, da), (pb, db) = self._features(a), self._features(b)
        empty = np.zeros((0, 2), dtype=np.float32)

        def result(n_match: int, n_in: int, p1=empty, p2=empty) -> MatchResult:
            return MatchResult(a, b, p1, p2, n_match, n_in, n_in / n_match if n_match else 0.0,
                               time.perf_counter() - t0, self.name)

        if da is None or db is None or len(da) < 8 or len(db) < 8:
            return result(0, 0)
        bf = cv2.BFMatcher(cv2.NORM_L2)
        back = {m.queryIdx: m.trainIdx for m in bf.match(db, da)}
        good = [m[0] for m in bf.knnMatch(da, db, k=2)
                if len(m) == 2 and m[0].distance < self.ratio * m[1].distance and back.get(m[0].trainIdx) == m[0].queryIdx]
        if len(good) < 8:
            return result(len(good), 0)
        p1 = pa[[m.queryIdx for m in good]]
        p2 = pb[[m.trainIdx for m in good]]
        _, mask = cv2.findFundamentalMat(p1, p2, cv2.FM_RANSAC, self.ransac_px, 0.999, 5000)
        if mask is None:
            return result(len(good), 0)
        keep = mask.ravel().astype(bool)
        return result(len(good), int(keep.sum()), p1[keep], p2[keep])
