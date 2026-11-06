"""DeDoDe feature matcher (Block 11 of 50-tech evaluation) via kornia.feature.DeDoDe (L detector, B descriptor).

Weights must already be in the torch.hub checkpoint cache; nothing is downloaded. If kornia/torch or the weights are
missing, `DeDoDeMatcher` raises `MatcherUnavailableError` -- it never substitutes another detector under the DeDoDe
name (use "akaze"/"brisk").
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from reconstruction.matching.classical import DescriptorMatcher, torch_hub_checkpoints
from reconstruction.matching.interface import MatcherUnavailableError

_WEIGHTS = ("dedode_detector_L.pth", "dedode_descriptor_B.pth")


def probe() -> Tuple[bool, str]:
    try:
        import torch  # noqa: F401
        from kornia.feature import DeDoDe  # noqa: F401
    except ImportError as e:
        return False, f"DeDoDe needs torch and a kornia with kornia.feature.DeDoDe: {e}"
    missing = [w for w in _WEIGHTS if not (torch_hub_checkpoints() / w).is_file()]
    if missing:
        return False, f"DeDoDe weights not cached in {torch_hub_checkpoints()}: {', '.join(missing)}"
    return True, "available"


class DeDoDeMatcher(DescriptorMatcher):
    name = "dedode@kornia-L-upright+B-upright"
    license = "MIT"
    non_commercial = False

    def __init__(self, paths: Dict[str, Path], max_side: int = 960, num_keypoints: int = 2048,
                 ransac_px: float = 1.0) -> None:
        ok, msg = probe()
        if not ok:
            raise MatcherUnavailableError(msg)
        super().__init__(paths, max_side=max_side, ransac_px=ransac_px)
        self.num_keypoints = num_keypoints
        self._model = None

    def _extract(self, img_bgr) -> Tuple[np.ndarray, np.ndarray]:
        import torch
        from kornia.feature import DeDoDe

        try:
            if self._model is None:
                self._model = DeDoDe.from_pretrained("L-upright", "B-upright", amp_dtype=torch.float32).eval()
            rgb = np.ascontiguousarray(img_bgr[:, :, ::-1])
            h, w = rgb.shape[:2]
            with torch.no_grad():
                kps, _scores, descs = self._model(torch.from_numpy(rgb).permute(2, 0, 1).float()[None] / 255.0,
                                                  n=self.num_keypoints)
        except Exception as e:  # noqa: BLE001
            raise MatcherUnavailableError(f"DeDoDe failed to load/run: {e}") from e
        k = kps[0].cpu().numpy()  # normalised [-1, 1] -> pixels
        px = np.stack([(k[:, 0] + 1) * 0.5 * w, (k[:, 1] + 1) * 0.5 * h], axis=1).astype(np.float32)
        return px, descs[0].cpu().numpy().astype(np.float32)
