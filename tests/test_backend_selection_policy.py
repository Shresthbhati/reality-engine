"""Tests for reconstruction/backend/selection.py (P5-02 automatic
backend selection): input characteristics -> explicit ranking decision
-> run -> assess -> retry/alternate, on synthetic input profiles.

Policy-under-test rules (constitution: selection is explicit policy,
never a language model; honest unavailability, never a fabricated
capability):
- an input profile the repo cannot serve (e.g. LiDAR-only, no adapter)
  yields a declined selection, not a best-effort backend;
- ranking never returns a backend that is not runnable here;
- the assess step maps a failed run to a recorded failure and the
  loop tries the next candidate -- it never repeats a failed backend
  hoping for a different answer.
"""

import pytest

from reconstruction.backend.selection import (
    InputProfile,
    SelectionDecision,
    select_backend_for,
    assess_and_advance,
)


class TestProfileRanking:
    def test_photogrammetry_profile_selects_colmap(self):
        profile = InputProfile(
            n_images=120, has_gnss=True, has_depth=False, has_lidar=False,
            scene_type="outdoor", gpu_available=True,
        )
        decision = select_backend_for(profile, available=("colmap", "openmvs"))
        assert decision.backend == "colmap"
        assert decision.reason, "a selection must state its why"

    def test_unavailable_backend_never_selected(self):
        profile = InputProfile(
            n_images=120, has_gnss=False, has_depth=False, has_lidar=False,
            scene_type="outdoor", gpu_available=False,
        )
        decision = select_backend_for(profile, available=())
        assert decision.backend is None
        assert decision.status == "declined"
        assert "no runnable backend" in decision.reason

    def test_depth_profile_prefers_fusion_path(self):
        # RGB-D input does not need MVS photogrammetry; the profile
        # says so and the decision must reflect the input, not a
        # fixed default.
        profile = InputProfile(
            n_images=0, has_gnss=False, has_depth=True, has_lidar=False,
            scene_type="indoor", gpu_available=False,
        )
        decision = select_backend_for(profile, available=("colmap",))
        assert decision.backend is None
        assert decision.status == "not_applicable"
        assert "depth" in decision.reason

    def test_lidar_only_profile_declined(self):
        # No LiDAR registration backend exists in this repo (P4-01
        # ICP consumes point clouds; there is no LiDAR pipeline).
        profile = InputProfile(
            n_images=0, has_gnss=False, has_depth=False, has_lidar=True,
            scene_type="outdoor", gpu_available=True,
        )
        decision = select_backend_for(profile, available=("colmap",))
        assert decision.status == "declined"

    def test_two_image_profile_declined(self):
        # Two images cannot sustain MVS; the policy must refuse, not
        # run a doomed reconstruction.
        profile = InputProfile(
            n_images=2, has_gnss=False, has_depth=False, has_lidar=False,
            scene_type="indoor", gpu_available=False,
        )
        decision = select_backend_for(profile, available=("colmap",))
        assert decision.status == "declined"
        assert "image" in decision.reason


class TestAssessAndAdvance:
    def _profile(self):
        return InputProfile(
            n_images=50, has_gnss=False, has_depth=False, has_lidar=False,
            scene_type="indoor", gpu_available=False,
        )

    def test_failure_advances_to_next_candidate(self):
        decision = select_backend_for(self._profile(), available=("colmap", "openmvs"))
        assert decision.backend == "colmap"
        nxt = assess_and_advance(decision, run_succeeded=False)
        assert nxt.backend == "openmvs"
        assert nxt.attempted == ("colmap",)

    def test_success_stops_the_loop(self):
        decision = select_backend_for(self._profile(), available=("colmap", "openmvs"))
        nxt = assess_and_advance(decision, run_succeeded=True)
        assert nxt.backend == "colmap"
        assert nxt.status == "selected"

    def test_exhausted_candidates_decline_honestly(self):
        decision = select_backend_for(self._profile(), available=("colmap",))
        assert decision.backend == "colmap"
        exhausted = assess_and_advance(decision, run_succeeded=False)
        assert exhausted.backend is None
        assert exhausted.status == "exhausted"
        assert exhausted.attempted == ("colmap",)

    def test_never_repeats_a_failed_backend(self):
        decision = select_backend_for(
            self._profile(), available=("colmap", "openmvs", "opensfm")
        )
        seen = []
        current = decision
        while current.backend is not None and current.status == "selected":
            seen.append(current.backend)
            current = assess_and_advance(current, run_succeeded=False)
        assert len(seen) == len(set(seen)), "a failed backend must not be retried"
        assert current.status == "exhausted"
