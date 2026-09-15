"""Input-profile-driven reconstruction backend selection (P5-02).

`reconstruction/backend/registry.py` answers "what can this machine
run?" (discovery + static preference order). This module answers the
P5-02 question: "given THIS input's characteristics, which backend
should run, and what do we do when it fails?" -- the spec's
input characteristics -> ranking -> run -> assess -> retry/alternate
loop.

Selection is EXPLICIT POLICY (constitution: "selection is explicit
policy, not a language model"): deterministic rules over the profile's
declared characteristics. Every decision carries its reason. A profile
the repo cannot serve is DECLINED with the reason -- never routed to a
best-effort backend that would fabricate a reconstruction. The
assess/advance loop never repeats a failed candidate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple


@dataclass(frozen=True)
class InputProfile:
    """Declared input characteristics (what the evidence actually
    contains). Fields absent from the evidence are declared False by
    the CALLER's honest manifest -- this module never inspects raw
    data to guess them."""

    n_images: int
    has_gnss: bool
    has_depth: bool
    has_lidar: bool
    scene_type: str  # "indoor" | "outdoor" | ...
    gpu_available: bool


@dataclass(frozen=True)
class SelectionDecision:
    """One selection step's outcome. `backend` is None iff status is
    not "selected". `attempted` carries the candidates already tried
    and failed on this input -- the loop's memory."""

    backend: Optional[str]
    status: str  # "selected" | "declined" | "not_applicable" | "exhausted"
    reason: str
    attempted: Tuple[str, ...] = ()
    remaining: Tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "backend": self.backend,
            "status": self.status,
            "reason": self.reason,
            "attempted": list(self.attempted),
            "remaining": list(self.remaining),
        }


#: Minimum images for multi-view stereo to be meaningful. Below this
#: a photogrammetric reconstruction is a guess, not a measurement.
MIN_MVS_IMAGES = 3


def select_backend_for(
    profile: InputProfile,
    available: Sequence[str],
) -> SelectionDecision:
    """Rank runnability-filtered candidates for this profile.

    Policy (declared, deterministic):
    1. Depth-bearing input: RGB-D unprojection + fusion is the right
       path (P6 fusion), not MVS photogrammetry -- "not_applicable"
       for the SfM/MVS registry. Depth + images together still fuse
       through the same path; MVS adds nothing the depth doesn't
       already measure better.
    2. LiDAR-only input: this repo has no LiDAR registration pipeline
       (P4-01 ICP consumes prepared clouds; no LiDAR ingestion exists)
       -- declined honestly.
    3. Too few images for MVS: declined -- running a doomed
       reconstruction would fabricate geometry.
    4. Otherwise: first available candidate in registry preference
       order (the caller passes `available` already ordered, from
       `registry.select_backends`).
    """
    remaining = tuple(available)

    if profile.has_depth:
        return SelectionDecision(
            backend=None, status="not_applicable",
            reason=(
                "input carries measured depth: RGB-D unprojection + "
                "fusion (P6) is the correct path; MVS photogrammetry "
                "would re-estimate what was already measured"
            ),
            attempted=(), remaining=remaining,
        )

    if profile.has_lidar and profile.n_images == 0:
        return SelectionDecision(
            backend=None, status="declined",
            reason=(
                "LiDAR-only input: no LiDAR ingestion/registration "
                "pipeline exists in this repo (honest gap, P4-01 ICP "
                "consumes prepared point clouds)"
            ),
            attempted=(), remaining=remaining,
        )

    if profile.n_images < MIN_MVS_IMAGES:
        return SelectionDecision(
            backend=None, status="declined",
            reason=(
                f"{profile.n_images} image(s) cannot sustain multi-view "
                f"stereo (minimum {MIN_MVS_IMAGES}); a reconstruction "
                "from this input would be fabricated"
            ),
            attempted=(), remaining=remaining,
        )

    if not remaining:
        return SelectionDecision(
            backend=None, status="declined",
            reason="no runnable backend on this machine for this input",
            attempted=(), remaining=(),
        )

    return SelectionDecision(
        backend=remaining[0], status="selected",
        reason=(
            f"photogrammetric profile ({profile.n_images} images, "
            f"{profile.scene_type}); first runnable candidate in "
            "preference order"
        ),
        attempted=(), remaining=remaining[1:],
    )


def assess_and_advance(
    decision: SelectionDecision,
    *,
    run_succeeded: bool,
) -> SelectionDecision:
    """The assess step of the spec loop: a successful run stops the
    loop; a failed run advances to the next candidate, recording the
    failure -- never re-running a backend that already failed."""
    if run_succeeded:
        return SelectionDecision(
            backend=decision.backend, status="selected",
            reason=f"{decision.backend} run assessed as successful",
            attempted=decision.attempted, remaining=decision.remaining,
        )

    attempted = decision.attempted + (decision.backend,)
    if not decision.remaining:
        return SelectionDecision(
            backend=None, status="exhausted",
            reason=(
                "all candidates for this input failed: "
                + ", ".join(attempted)
                + " -- no alternates remain; declining honestly"
            ),
            attempted=attempted, remaining=(),
        )
    return SelectionDecision(
        backend=decision.remaining[0], status="selected",
        reason=(
            f"{decision.backend} failed; advancing to next candidate "
            "in preference order"
        ),
        attempted=attempted, remaining=decision.remaining[1:],
    )
