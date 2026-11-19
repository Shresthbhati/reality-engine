"""RELEASE.md may not tick a box the repository's own evidence contradicts.

Guards the claim that was wrong once: "Six measured rooms captured and scored" was ticked while
``dataset_verification_report.json`` recorded zero verified physical rooms.

Synthetic verification: reads two committed files; no network.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE = (ROOT / "RELEASE.md").read_text(encoding="utf-8")
REPORT = json.loads((ROOT / "benchmarks" / "measured_rooms" / "dataset_verification_report.json").read_text(encoding="utf-8"))


def _box(label_regex: str) -> str | None:
    """'x' / ' ' for the checklist line matching ``label_regex``; None if no such line."""
    m = re.search(rf"^- \[([ xX])\] .*{label_regex}", RELEASE, re.MULTILINE)
    return m.group(1).lower() if m else None


def test_measured_rooms_box_is_ticked_only_if_every_configured_room_is_physically_verified():
    box = _box(r"measured rooms captured and scored")
    assert box is not None, "the measured-rooms checklist line is missing from RELEASE.md"
    all_verified = REPORT["total_verified"] == REPORT["total_rooms_configured"] and REPORT["total_blocked"] == 0
    if not all_verified:
        assert box == " ", (
            f"RELEASE.md ticks 'six measured rooms captured and scored' but the dataset report shows "
            f"{REPORT['total_verified']}/{REPORT['total_rooms_configured']} verified and {REPORT['total_blocked']} blocked")


def test_release_states_that_no_physical_ground_truth_exists_while_none_is_verified():
    if REPORT["total_verified"] == 0:
        assert re.search(r"none exists yet|No dimensional accuracy is demonstrated", RELEASE), (
            "RELEASE.md must say no physical ground truth exists while the report shows zero verified rooms")


def test_branch_protection_box_is_not_ticked_without_a_recorded_inspection():
    box = _box(r"Branch protection on `main`")
    assert box is not None
    if box == "x":
        assert re.search(r"inspected \d{4}-\d{2}-\d{2}", RELEASE, re.IGNORECASE), (
            "branch protection may be ticked only with a dated inspection recorded next to it")
