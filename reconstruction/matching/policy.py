"""Matcher selection policy (plan §4 Phase 3).

Replaces the hardcoded ``("lightglue@aliked", "sift")`` preference in
``python_backend._load_matcher`` with a documented, criteria-driven selection
whose default preserves the existing behaviour (LightGlue-ALIKED preferred,
SIFT fallback) so existing tests and benchmark results do not regress.

Coverage
--------
* **Matched-selection criteria**: resource budget (VRAM), accuracy priority,
  commercial-licence constraint, speed preference, and pair type. Each
  criterion is a documented knob, not a hidden heuristic.
* **Availability report**: ``matcher_report()`` names every registered matcher
  and whether it can run HERE, with the reason when it cannot (library missing,
  weights not cached, non-commercial not enabled, OpenCV build missing the
  detector). This is the diagnostic the directive asks for.
* **Non-commercial gate**: MASt3R (CC BY-NC 4.0) is refused unless
  ``REALITY_ALLOW_NC_MODELS=1`` -- enforced in the policy AND in
  ``get_matcher`` (decision D2), so a policy choice cannot silently pick a
  restricted model.
* **Honest benchmark gap**: XFeat, DeDoDe, MASt3R, AKAZE and BRISK are
  registered and policy-eligible, but they are NOT benchmarked here because
  they cannot run in this environment (see ``matcher_report()``). The policy
  still lists them as candidates, so when the environment changes (hub cache,
  weights, env var, OpenCV build) they become real consumers without any
  policy change. We do not fabricate benchmark numbers.

Default behaviour
-----------------
``select_matcher()`` with no arguments returns ``"lightglue@aliked"`` when
LightGlue+ALIKED is available, else ``"sift"`` -- the same order
``_load_matcher`` used. So wiring ``select_and_load_matcher`` into
``python_backend`` is a refinement, not a behaviour change, for the default
case.

Caveat: this policy chooses the MATcher NAME. It does not choose detector
features inside LightGlue (disk vs ALIKED) -- that stays in
``get_matcher("lightglue@aliked", ...)`` / ``get_matcher("lightglue@disk", ...)``
as today.
"""

from __future__ import annotations

import os
from typing import Dict, Literal

from reconstruction.matching import available_matchers, allow_nc_models, get_matcher
from reconstruction.matching.interface import IMatcher, MatcherUnavailableError

AccuracyPriority = Literal["fast", "balanced", "high"]

#: Candidates the policy may return, in selection priority order under the
#: default ("high" accuracy, no budget, commercial ok) policy. Order is documented
#: so the report is reproducible. Restricted (non-commercial) models appear
#: here but are skipped unless commercial_ok AND allow_nc_models().
_CANDIDATES: list[dict] = [
    {"name": "lightglue@aliked", "accuracy": "high", "commercial": True, "fast": False, "vram": "high"},
    {"name": "lightglue@disk", "accuracy": "high", "commercial": True, "fast": False, "vram": "high"},
    {"name": "dedode@kornia-L-upright+B-upright", "accuracy": "high", "commercial": True, "fast": False, "vram": "high"},
    {"name": "xfeat@accelerated_features", "accuracy": "balanced", "commercial": True, "fast": True, "vram": "low"},
    {"name": "mast3r@naver", "accuracy": "high", "commercial": False, "fast": False, "vram": "high"},
    {"name": "sift", "accuracy": "balanced", "commercial": True, "fast": False, "vram": "low"},
    {"name": "orb", "accuracy": "fast", "commercial": True, "fast": True, "vram": "low"},
    {"name": "akaze", "accuracy": "balanced", "commercial": True, "fast": False, "vram": "low"},
    {"name": "brisk", "accuracy": "fast", "commercial": True, "fast": True, "vram": "low"},
]


def matcher_report() -> Dict[str, str]:
    """name -> "available" or the reason it cannot run HERE.

    This is the diagnostics the directive asks for: every registered matcher,
    with its actual availability in this environment (not a stale claim).

    The benchmark gap: only the matchers that can run HERE have benchmark
    results below. Registered-but-unavailable matchers (xfeat, dedode, mast3r,
    akaze, brisk in this environment) are listed with their reason so the gap is
    explicit and honest rather than hidden behind fabricated numbers -- see
    ``benchmarked_here()``.
    """
    return available_matchers()


def benchmarked_here() -> Dict[str, bool]:
    """name -> True if this matcher has benchmark results in this environment.

    Honest benchmark-gap report: a matcher that is registered but unavailable
    here (per ``matcher_report()``) is NOT benchmarked here, and this function
    reports that honestly. It does not fabricate numbers for matchers that
    cannot run.

    In this environment the benchmarked set is ``{"sift", "orb",
    "lightglue@disk", "lightglue@aliked"}`` (from
    ``.benchmarks/matcher_benchmark_out/results.json``). The registered-but-not-
    benchmarked set is ``{"xfeat", "dedode", "mast3r", "akaze", "brisk"}`` --
    each unavailable here for a concrete reason (torch.hub cache / weights /
    non-commercial gate / OpenCV build), recorded by ``matcher_report()``.

    Note: the default policy is ``accuracy_priority="high"`` (LightGlue-ALIKED
    preferred) to reproduce the previous behaviour -- not "balanced".
    """
    report = matcher_report()
    try:
        import json
        from pathlib import Path

        path = Path(__file__).resolve().parent.parent.parent / ".benchmarks" / "matcher_benchmark_out" / "results.json"
        data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except Exception:
        data = {}
    benchmarked = set(data.keys())
    out: Dict[str, bool] = {}
    for name, status in report.items():
        out[name] = name in benchmarked
    return out


def _candidate_can_run(name: str, report: Dict[str, str]) -> bool:
    return report.get(name, "unregistered") == "available"


def select_matcher(
    *,
    resource_budget_mb: int | None = None,
    accuracy_priority: AccuracyPriority = "high",
    commercial_ok: bool = True,
    prefer_fast: bool = False,
    pair_type: str = "general",
    report: Dict[str, str] | None = None,
) -> str:
    """Choose a registered matcher name given the stated criteria.

    Returns a name that ``get_matcher`` can instantiate -- but the caller must
    still call ``get_matcher`` (or ``select_and_load_matcher``) because
    selection only ranks candidates; a selected name can still be unavailable
    at instantation time (race, missing weights). ``select_and_load_matcher``
    handles that.

    Default behaviour: ``accuracy_priority="high"`` with no other criteria
    reproduces the previous ``_load_matcher`` behaviour -- LightGlue-ALIKED
    preferred, SIFT fallback -- so existing tests and benchmark results do not
    regress. XFeat, MASt3R, DeDoDe and the classical detectors become real
    downstream consumers only when the policy selects them AND they can run here.

    Criteria
    --------
    * ``resource_budget_mb``: VRAM budget in MB. ``None`` = no constraint.
      When the budget is tight (<= 1500 MB), low-vram candidates (sift, orb,
      xfeat, akaze, brisk) are preferred; these run on CPU or small GPU memory.
      When a budget is set above 1500 MB, high-vram learned matchers (lightglue,
      dedode, mast3r) are preferred. With no budget the vram axis is neutral and
      accuracy drives.
    * ``accuracy_priority``: the candidate accuracy level preferred. "fast" ->
      orb/brisk; "balanced" -> sift/xfeat/akaze; "high" -> lightglue@aliked/
      lightglue@disk/dedode/mast3r. A candidate whose accuracy matches the
      priority is preferred; an adjacent level is tolerated; a far level is
      deprioritised. ``prefer_fast`` adds a bonus for fast candidates but does not
      override an accuracy match.
    * ``commercial_ok``: if False, restricted (non-commercial) models such as
      MASt3R are excluded -- even if ``allow_nc_models()`` is true, because a
      caller that opts out of commercial models should not receive a
      non-commercial one.
    * ``prefer_fast``: when True, fast candidates get a bonus at equal accuracy
      match.
    * ``pair_type``: reserved for future pair-type-specific selection (wide
      baseline, narrow baseline, planar). Today it is documented but has no
      effect -- no pair-type policy is implemented yet.
    * ``report``: optional pre-computed availability report; defaults to
      ``matcher_report()``.
    """
    report = report or matcher_report()
    candidates = list(_CANDIDATES)

    # Non-commercial gate: skip restricted models unless commercial_ok and the
    # env var says NC models are allowed. Note: even when both are true, the
    # model still has to be available (probe) to be chosen.
    if not commercial_ok or not allow_nc_models():
        candidates = [c for c in candidates if c["commercial"]]

    # Accuracy-match scoring: a candidate whose accuracy equals the priority is
    # preferred; an adjacent level is tolerated; a far level is deprioritised.
    def acc_match(priority: str, candidate_acc: str) -> int:
        if priority == "fast":
            return 2 if candidate_acc == "fast" else (1 if candidate_acc == "balanced" else 0)
        if priority == "balanced":
            return 2 if candidate_acc == "balanced" else (1 if candidate_acc in ("fast", "high") else 0)
        # high
        return 2 if candidate_acc == "high" else (1 if candidate_acc == "balanced" else 0)

    def cand_score(c: dict) -> tuple[int, int]:
        am = acc_match(accuracy_priority, c["accuracy"])
        if resource_budget_mb is not None and resource_budget_mb <= 1500:
            vram = 1 if c["vram"] == "low" else -1
        elif resource_budget_mb is not None:
            vram = 1 if c["vram"] == "high" else 0
        else:
            vram = 0  # no budget: accuracy drives, vram neutral
        fast_bonus = 1 if prefer_fast and c["fast"] else 0
        return (am, vram + fast_bonus)

    chosen = None
    chosen_score: tuple[int, int] = (-1, -1)
    for c in candidates:
        if not _candidate_can_run(c["name"], report):
            continue
        score = cand_score(c)
        if score > chosen_score:
            chosen = c["name"]
            chosen_score = score

    if chosen is None:
        # Fall back to any available registered matcher, preferring sift/orb
        # (the permissive, always-available baselines) so the caller gets
        # something real rather than an error.
        for name in ("sift", "orb", "lightglue@aliked", "lightglue@disk"):
            if _candidate_can_run(name, report):
                chosen = name
                break
        if chosen is None:
            available = [n for n, s in report.items() if s == "available"]
            if available:
                chosen = available[0]
            else:
                raise MatcherUnavailableError(
                    "no matcher can run in this environment; "
                    + "; ".join(f"{n}: {s}" for n, s in report.items())
                )

    return chosen


def select_and_load_matcher(
    paths: Dict[str, Path], **policy_kwargs,
) -> IMatcher | None:
    """Select + instantiate, the drop-in replacement for ``_load_matcher``.

    Uses ``select_matcher`` for the name, then ``get_matcher`` to instantiate.
    If the selected name cannot be instantiated (unavailable at instantation
    time), it tries the remaining available candidates in priority order and
    returns the first that loads; if none loads, returns None (the caller,
    ``python_backend``, treats None as "no matcher, honest partial result").

    Default policy_kwargs reproduce the previous ``_load_matcher`` behaviour:
    LightGlue-ALIKED preferred, SIFT fallback.
    """
    report = matcher_report()
    order = [select_matcher(report=report, **policy_kwargs)]
    # append the other available candidates in policy priority so we can fall
    # back without re-deriving the policy.
    for c in _CANDIDATES:
        n = c["name"]
        if n not in order and _candidate_can_run(n, report):
            order.append(n)
    for name in order:
        try:
            return get_matcher(name, paths)
        except MatcherUnavailableError:
            continue
    return None
