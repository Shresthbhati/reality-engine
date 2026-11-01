"""Local-feature matcher interface (plan §4 Phase 2).

A matcher takes two images and returns the correspondences that survive its own geometric verification, plus enough
measured facts (counts, runtime) to benchmark one matcher against another. Backends are swappable: classical SIFT is
always available; learned matchers (LightGlue and friends) must declare their licence and whether it is
non-commercial, and are refused unless `REALITY_ALLOW_NC_MODELS=1` (decision D2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


class MatcherUnavailableError(RuntimeError):
    """A requested matcher cannot run here (library not installed, weights missing, or non-commercial and not enabled)."""


@dataclass(frozen=True)
class MatchResult:
    a: str
    b: str
    points_a: np.ndarray  # (inliers, 2) pixel coordinates in image a, geometrically verified
    points_b: np.ndarray  # (inliers, 2), row-aligned with points_a
    matches: int  # correspondences that survived the matcher's own filtering
    inliers: int  # of those, consistent with one epipolar geometry
    inlier_ratio: float
    runtime_s: float
    matcher: str  # "name@version" -- recorded so results stay traceable to the model that produced them

    def to_dict(self) -> dict:
        return {"a": self.a, "b": self.b, "matches": self.matches, "inliers": self.inliers,
                "inlier_ratio": round(self.inlier_ratio, 4), "runtime_s": round(self.runtime_s, 4),
                "matcher": self.matcher}


@runtime_checkable
class IMatcher(Protocol):
    name: str
    license: str
    non_commercial: bool

    def match(self, a: str, b: str) -> MatchResult: ...
