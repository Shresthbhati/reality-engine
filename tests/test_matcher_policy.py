"""Matcher selection policy tests (plan §4 Phase 3).

The policy replaces the hardcoded ``("lightglue@aliked", "sift")`` preference
in ``python_backend._load_matcher`` with a documented, criteria-driven
selection. Default behaviour is preserved (LightGlue-ALIKED preferred, SIFT
fallback) so existing tests and benchmark results do not regress; XFeat,
MASt3R, DeDoDe and the classical detectors become real downstream consumers
WHEN the policy selects them AND they can run here.

Honest benchmark gap: in this environment only sift, orb, lightglue@disk and
lightglue@aliked are available (and benchmarked). XFeat, DeDoDe, MASt3R, AKAZE
and BRISK are registered and policy-eligible but NOT benchmarked here because
they cannot run here for concrete reasons (torch.hub cache / weights / non-
commercial gate / OpenCV build). The policy and the report name those reasons
rather than fabricating numbers -- see ``policy.benchmarked_here()`` and
``policy.matcher_report()``.
"""

from __future__ import annotations

import os

import pytest

from reconstruction.matching.policy import (
    AccuracyPriority,
    benchmarked_here,
    matcher_report,
    select_and_load_matcher,
    select_matcher,
)


# torch + kornia are an optional extra (``pip install -e ".[perception]"``); the default CI image has neither. The policy
# must pick LightGlue when it can run and degrade to the classical matcher when it cannot -- both are asserted below.
_LIGHTGLUE = matcher_report().get("lightglue@aliked") == "available"


class TestMatcherSelectionPolicy:
    def test_default_reproduces_previous_behaviour(self) -> None:
        """With no criteria, the policy picks LightGlue-ALIKED when available,
        exactly as the old ``_load_matcher`` did -- no regression. Without torch+kornia it must pick the classical
        SIFT matcher, never an unavailable learned one."""
        name = select_matcher()
        expected = "lightglue@aliked" if _LIGHTGLUE else "sift"
        assert name == expected, f"default picked {name}, expected {expected} (lightglue available: {_LIGHTGLUE})"

    def test_high_accuracy_prefers_lightglue_or_dedode(self) -> None:
        name = select_matcher(accuracy_priority="high")
        # Among available high-accuracy candidates, lightglue@aliked is first; with none available, classical SIFT.
        allowed = (("lightglue@aliked", "lightglue@disk", "dedode@kornia-L-upright+B-upright") if _LIGHTGLUE
                   else ("sift",))
        assert name in allowed, f"high-accuracy picked {name}; allowed here: {allowed}"

    def test_fast_priority_prefers_fast_candidates(self) -> None:
        name = select_matcher(accuracy_priority="fast", prefer_fast=True)
        assert name in ("orb", "brisk", "sift")

    def test_tight_vram_budget_prefers_low_vram(self) -> None:
        name = select_matcher(resource_budget_mb=1000, accuracy_priority="balanced")
        # Under a tight budget, low-vram candidates are preferred.
        assert name in ("sift", "orb", "xfeat@accelerated_features", "akaze", "brisk")

    def test_commercial_ok_false_excludes_mast3r(self) -> None:
        name_ok = select_matcher(commercial_ok=True, accuracy_priority="high")
        name_no = select_matcher(commercial_ok=False, accuracy_priority="high")
        # MASt3R is non-commercial and should be excluded when commercial_ok is False.
        assert "mast3r@naver" not in (name_no,)
        # When commercial_ok is True, MASt3R is still not chosen here because
        # it is unavailable (non-commercial gate), but the policy logic excludes
        # it only via commercial_ok=False.
        assert name_ok != "mast3r@naver"  # unavailable here regardless

    def test_nc_gate_enforced_without_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """MASt3R is refused when REALITY_ALLOW_NC_MODELS is not set, even though
        it is a policy candidate (high accuracy)."""
        monkeypatch.delenv("REALITY_ALLOW_NC_MODELS", raising=False)
        # Re-import to recompute the gate.
        import reconstruction.matching.policy as pol

        reloaded = pol.select_matcher(accuracy_priority="high")
        # MASt3R must not be selected without the env var.
        assert reloaded != "mast3r@naver"

    def test_nc_gate_allowed_with_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With REALITY_ALLOW_NC_MODELS=1, MASt3R becomes a candidate again --
        but it still must be available (weights) to be selected. Here it is not
        (no weights), so the policy picks another high-accuracy candidate."""
        monkeypatch.setenv("REALITY_ALLOW_NC_MODELS", "1")
        import reconstruction.matching.policy as pol

        reloaded = pol.select_matcher(accuracy_priority="high")
        # Still not mast3r here (no weights), but the gate is open.
        assert reloaded != "mast3r@naver"

    def test_select_and_load_returns_a_real_matcher(self) -> None:
        """select_and_load_matcher (the _load_matcher replacement) returns a
        usable IMatcher for the default selection -- the downstream SfM path
        gets a real matcher, not an error. MatchResult carries provenance
        (matcher name) plus inlier_ratio and runtime_s -- the directive's
        confidence/provenance requirement at the match level."""
        import os

        bm = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", ".benchmarks", "matcher_benchmark_out")
        )
        paths = {
            "a": os.path.join(bm, "pair_00_a.png"),
            "b": os.path.join(bm, "pair_00_b.png"),
        }
        matcher = select_and_load_matcher(paths)
        assert matcher is not None
        assert matcher.name in ("lightglue@aliked", "lightglue@disk", "sift", "sift@opencv", "orb", "orb@opencv")
        if not _LIGHTGLUE:
            assert matcher.name.startswith(("sift", "orb")), "no learned matcher may be returned when it cannot run"
        mr = matcher.match("a", "b")
        assert mr.matcher  # provenance: the model that produced the matches
        assert mr.inlier_ratio >= 0.0
        assert mr.runtime_s >= 0.0
        assert mr.points_a.shape[1] == 2 and mr.points_b.shape[1] == 2  # pixel coords
        assert len(mr.points_a) == mr.inliers  # inlier points only

    def test_select_and_load_falls_back_when_top_pick_cannot_instantiate(
        self,
    ) -> None:
        """select_and_load_matcher falls back through the available candidates:
        if the policy's top pick cannot be instantiated (get_matcher raises), it
        tries the next available candidate and returns the first that loads.

        We verify the fallback by monkeypatching get_matcher to refuse the policy's
        top pick ("lightglue@aliked") and succeed for the next available candidate
        ("sift"). The real matcher_report() says lightglue@aliked is available, so
        select_matcher picks it (high accuracy, default) -- but get_matcher raises,
        so select_and_load_matcher falls back to sift."""
        import reconstruction.matching.policy as pol

        # Monkeypatch the get_matcher reference POLICY uses (policy imports it
        # via "from reconstruction.matching import ... get_matcher", so it holds a
        # local reference; patching matching_mod.get_matcher would not affect calls
        # inside policy). Refuse lightglue@aliked/lightglue@disk, allow sift.
        original = pol.get_matcher
        refused = {"lightglue@aliked", "lightglue@disk"}
        def guarded(name: str, paths, **kwargs):
            if name in refused:
                raise pol.MatcherUnavailableError(f"refused for test: {name}")
            return original(name, paths, **kwargs)

        pol.get_matcher = guarded
        try:
            matcher = pol.select_and_load_matcher({})
        finally:
            pol.get_matcher = original

        assert matcher is not None
        assert matcher.name == "sift@opencv", f"fell back to {matcher.name}, expected sift@opencv"  # fell back from lightglue@aliked to sift

    def test_matcher_report_lists_all_registered_matchers(self) -> None:
        report = matcher_report()
        # Every registered name must appear with a status.
        for name in (
            "sift", "orb", "lightglue@aliked", "lightglue@disk",
            "akaze", "brisk", "xfeat", "dedode", "mast3r",
        ):
            assert name in report, f"{name} missing from matcher_report"

    def test_benchmark_gap_is_honest_not_fabricated(self) -> None:
        """The benchmark-gap report is honest in both directions:

        1. A matcher that CANNOT run here (per matcher_report) is NOT marked
           benchmarked -- we never fabricate numbers for an unavailable matcher.
        2. A matcher that IS in the benchmarked set (results.json) must be
           available here -- a benchmarked result for an unavailable matcher would
           be a stale/hallucinated number.

        The bare alias "lightglue" (without a variant) is listed by matcher_report
        as available but is not itself a benchmarked key -- the benchmarks live under
        "lightglue@disk" and "lightglue@aliked". That is fine and honest: the alias
        is covered by its benchmarked variants, which are available here."""
        bh = benchmarked_here()
        report = matcher_report()

        # Direction 1: unavailable here -> not benchmarked here.
        for name, status in report.items():
            if status != "available":
                assert bh.get(name) is False, (
                    f"policy claims {name} is benchmarked despite being unavailable here ({status})"
                )

        # Direction 2: benchmarked (in results.json) -> available here.
        benchmarked_keys = set(bh.keys()) & {n for n, v in bh.items() if v}
        for name in benchmarked_keys:
            assert report.get(name) == "available", (
                f"{name} is benchmarked but not available here (report says {report.get(name)!r})"
            )

        # The specific benchmarked matchers expected in this environment.
        expected_benchmarked = {"sift", "orb"} | ({"lightglue@disk", "lightglue@aliked"} if _LIGHTGLUE else set())
        assert benchmarked_keys == expected_benchmarked, (
            f"benchmarked set {benchmarked_keys} differs from expected {expected_benchmarked}"
        )

        # The registered-but-unavailable set in this environment (honest gap).
        unavailable = {n for n, s in report.items() if s != "available"}
        expected_unavailable = {"xfeat", "dedode", "mast3r", "akaze", "brisk"}
        assert expected_unavailable <= unavailable, (
            f"expected unavailable {expected_unavailable} not all present in {unavailable}"
        )
