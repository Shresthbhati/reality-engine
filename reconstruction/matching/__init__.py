"""Matcher registry. `get_matcher("sift", paths)`, `get_matcher("aliked", paths)`,
and `get_matcher("lightglue", paths)` work; anything else must be installed AND,
if its licence is non-commercial, enabled with REALITY_ALLOW_NC_MODELS=1 (decision D2).
An unavailable matcher raises `MatcherUnavailableError` with the reason -- it never
silently falls back to a different matcher, because a benchmark or a version report
must name the model that actually produced the matches."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

from reconstruction.matching.interface import IMatcher, MatcherUnavailableError, MatchResult

__all__ = [
    "ALIKEDLightGlueMatcher",
    "IMatcher",
    "LightGlueMatcher",
    "MatchResult",
    "MatcherUnavailableError",
    "allow_nc_models",
    "available_matchers",
    "get_matcher",
]

#: matchers the plan names but this repo has no backend for yet: name -> (reason, non_commercial)
_NOT_IMPLEMENTED: Dict[str, tuple] = {
    "loftr": ("LoFTR backend not implemented (needs kornia LoFTR adapter)", False),
    "mast3r": ("MASt3R backend not implemented; weights are CC BY-NC", True),
}


def allow_nc_models() -> bool:
    return os.environ.get("REALITY_ALLOW_NC_MODELS", "").strip() == "1"


def is_lightglue_available() -> tuple[bool, str]:
    try:
        import torch
        import kornia.feature as kf
        return True, "available"
    except ImportError as e:
        return False, f"LightGlue needs torch and kornia: {e}"


def available_matchers() -> Dict[str, str]:
    """name -> 'available' or the reason it is not."""
    out = {"sift": "available", "xfeat": "available", "dedode": "available"}
    lg_ok, lg_msg = is_lightglue_available()
    out["lightglue"] = lg_msg
    out["lightglue@disk"] = lg_msg
    out["lightglue@aliked"] = lg_msg
    out["aliked"] = lg_msg
    for name, (reason, nc) in _NOT_IMPLEMENTED.items():
        out[name] = reason + ("; also needs REALITY_ALLOW_NC_MODELS=1" if nc and not allow_nc_models() else "")
    return out


def get_matcher(name: str, paths: Dict[str, Path], **kwargs) -> IMatcher:
    norm = name.lower().strip()
    if norm == "sift":
        from reconstruction.matching.sift import SiftMatcher

        return SiftMatcher(paths, **kwargs)

    if norm in ("lightglue", "lightglue@disk"):
        lg_ok, lg_msg = is_lightglue_available()
        if not lg_ok:
            raise MatcherUnavailableError(lg_msg)
        from reconstruction.matching.lightglue import LightGlueMatcher

        return LightGlueMatcher(paths, features="disk", **kwargs)

    if norm in ("lightglue@aliked", "aliked", "aliked+lightglue"):
        lg_ok, lg_msg = is_lightglue_available()
        if not lg_ok:
            raise MatcherUnavailableError(lg_msg)
        from reconstruction.matching.lightglue import ALIKEDLightGlueMatcher

        return ALIKEDLightGlueMatcher(paths, **kwargs)

    if norm in ("xfeat", "xfeat@accelerated_features"):
        from reconstruction.matching.xfeat import XFeatMatcher

        return XFeatMatcher(paths, **kwargs)

    if norm in ("dedode", "dedode@v2"):
        from reconstruction.matching.dedode import DeDoDeMatcher

        return DeDoDeMatcher(paths, **kwargs)

    if norm in _NOT_IMPLEMENTED:
        reason, nc = _NOT_IMPLEMENTED[norm]
        if nc and not allow_nc_models():
            raise MatcherUnavailableError(f"{norm} is non-commercial; set REALITY_ALLOW_NC_MODELS=1 to enable it ({reason})")
        raise MatcherUnavailableError(reason)

    raise ValueError(f"unknown matcher {name!r}; choose from {sorted(available_matchers().keys())}")
