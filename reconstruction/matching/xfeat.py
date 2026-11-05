"""XFeat accelerated feature matcher (Block 10 of 50-tech evaluation).

XFeat (verlab/accelerated_features, Apache-2.0) is loaded through torch.hub ONLY from an already-cached repo checkout
(weights ship inside it); nothing is downloaded. If torch or the cached repo is missing, `XFeatMatcher` raises
`MatcherUnavailableError` -- it never substitutes another detector under the XFeat name (use "akaze"/"brisk").
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from reconstruction.matching.classical import DescriptorMatcher, torch_hub_dir
from reconstruction.matching.interface import MatcherUnavailableError

_REPO_DIR = "verlab_accelerated_features_main"


def probe() -> Tuple[bool, str]:
    try:
        import torch  # noqa: F401
    except ImportError as e:
        return False, f"XFeat needs torch: {e}"
    repo = torch_hub_dir() / _REPO_DIR
    if not repo.is_dir():
        return False, (f"XFeat repo not in torch.hub cache ({repo}); run once with network: "
                       "torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained=True)")
    return True, "available"


class XFeatMatcher(DescriptorMatcher):
    name = "xfeat@accelerated_features"
    license = "Apache-2.0"
    non_commercial = False

    def __init__(self, paths: Dict[str, Path], max_side: int = 960, top_k: int = 4096, ransac_px: float = 1.0) -> None:
        ok, msg = probe()
        if not ok:
            raise MatcherUnavailableError(msg)
        super().__init__(paths, max_side=max_side, ransac_px=ransac_px)
        self.top_k = top_k
        self._model = None

    def _extract(self, img_bgr) -> Tuple[np.ndarray, np.ndarray]:
        import torch

        try:
            if self._model is None:
                self._model = torch.hub.load("verlab/accelerated_features", "XFeat", pretrained=True,
                                             top_k=self.top_k, trust_repo=True)
            rgb = np.ascontiguousarray(img_bgr[:, :, ::-1])
            with torch.no_grad():
                out = self._model.detectAndCompute(torch.from_numpy(rgb).permute(2, 0, 1).float()[None] / 255.0,
                                                   top_k=self.top_k)[0]
        except Exception as e:  # noqa: BLE001
            raise MatcherUnavailableError(f"XFeat failed to load/run: {e}") from e
        return out["keypoints"].cpu().numpy().astype(np.float32), out["descriptors"].cpu().numpy().astype(np.float32)
