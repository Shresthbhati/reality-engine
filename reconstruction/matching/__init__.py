"""Matcher registry. `get_matcher("sift", paths)` always works; anything else must be installed AND, if its licence is
non-commercial, enabled with REALITY_ALLOW_NC_MODELS=1 (decision D2). An unavailable matcher raises
`MatcherUnavailableError` with the reason -- it never silently falls back to a different matcher, because a benchmark or
a version report must name the model that actually produced the matches."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

from reconstruction.matching.interface import IMatcher, MatcherUnavailableError, MatchResult

__all__ = ["IMatcher", "MatchResult", "MatcherUnavailableError", "get_matcher", "available_matchers", "allow_nc_models"]

#: matchers the plan names but this repo has no backend for yet: name -> (reason, non_commercial)
_NOT_IMPLEMENTED: Dict[str, tuple] = {
    "lightglue": ("LightGlue (ALIKED/DISK features) backend not implemented: needs kornia or the cvg/LightGlue package, "
                  "neither installed", False),
    "loftr": ("LoFTR backend not implemented (needs kornia)", False),
    "mast3r": ("MASt3R backend not implemented; weights are CC BY-NC", True),
}


def allow_nc_models() -> bool:
    return os.environ.get("REALITY_ALLOW_NC_MODELS", "").strip() == "1"


def available_matchers() -> Dict[str, str]:
    """name -> 'available' or the reason it is not."""
    out = {"sift": "available"}
    for name, (reason, nc) in _NOT_IMPLEMENTED.items():
        out[name] = reason + ("; also needs REALITY_ALLOW_NC_MODELS=1" if nc and not allow_nc_models() else "")
    return out


def get_matcher(name: str, paths: Dict[str, Path], **kwargs) -> IMatcher:
    if name == "sift":
        from reconstruction.matching.sift import SiftMatcher

        return SiftMatcher(paths, **kwargs)
    if name in _NOT_IMPLEMENTED:
        reason, nc = _NOT_IMPLEMENTED[name]
        if nc and not allow_nc_models():
            raise MatcherUnavailableError(f"{name} is non-commercial; set REALITY_ALLOW_NC_MODELS=1 to enable it ({reason})")
        raise MatcherUnavailableError(reason)
    raise MatcherUnavailableError(f"unknown matcher {name!r}; known: {sorted(available_matchers())}")
