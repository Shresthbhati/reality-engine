"""Reconstruction robustness (ROBUSTNESS directive P0).

Real captures are large, messy, incomplete, and imperfect. These tests
pin the item-level outcome classifier that sits between evidence import
and reconstruction: every item ends ACCEPTED / REJECTED / DEGRADED /
UNRESOLVED / FAILED with a REASON -- never silently discarded, never
converted from failure into success.

Metrics schema mirrors evidence/importers._measure_photo_quality:
  laplacian_variance (blur, higher = sharper), luma_mean,
  clipped_fraction, measured (1.0 when pixel metrics exist).
"""

from __future__ import annotations

import time

import pytest

from evidence.session import EvidenceItem
from evidence.importers import EvidenceKind
from reconstruction.robustness import (
    ACCEPTED,
    DEGRADED,
    FAILED,
    REJECTED,
    UNRESOLVED,
    AdmissionReport,
    classify_evidence_items,
    classify_reconstruction_run,
)


def _item(eid: str, metrics: dict | None, kind: str = "photo") -> EvidenceItem:
    metadata = {"quality": metrics} if metrics is not None else {}
    return EvidenceItem(
        id=eid,
        kind=EvidenceKind(kind),
        source_uri=f"file:///{eid}.jpg",
        metadata=metadata,
    )


SHARP = {
    "laplacian_variance": 1200.0,
    "luma_mean": 128.0,
    "clipped_fraction": 0.01,
    "measured": 1.0,
}
BLURRY = dict(SHARP, laplacian_variance=5.0)
SOFT_BLUR = dict(SHARP, laplacian_variance=120.0)
BLOWN = dict(SHARP, clipped_fraction=0.7)
SOFT_BLOWN = dict(SHARP, clipped_fraction=0.35)
DECODE_FAILED = {"measured": 0.0, "quality_note": "pixel decode failed (OSError); metrics unmeasured"}
NO_METRICS = None


class TestItemClassification:
    def test_every_item_gets_exactly_one_outcome_and_reason(self):
        items = [
            _item("sharp", SHARP),
            _item("blurry", BLURRY),
            _item("soft-blur", SOFT_BLUR),
            _item("blown", BLOWN),
            _item("decode-failed", DECODE_FAILED),
            _item("no-metrics", NO_METRICS),
        ]
        report = classify_evidence_items(items)
        outcomes = {a.evidence_id: a.outcome for a in report.admissions}
        reasons = {a.evidence_id: a.reason for a in report.admissions}
        # All six present, each classified, every reason non-empty.
        assert len(report.admissions) == 6
        assert set(outcomes) == {i.id for i in items}
        for eid, reason in reasons.items():
            assert reason, f"{eid} has an empty reason"

    def test_acceptance(self):
        report = classify_evidence_items([_item("sharp", SHARP)])
        assert report.admissions[0].outcome == ACCEPTED
        assert report.counts[ACCEPTED] == 1

    def test_hard_blur_is_rejected_with_measured_value(self):
        report = classify_evidence_items([_item("blurry", BLURRY)])
        a = report.admissions[0]
        assert a.outcome == REJECTED
        assert "laplacian_variance" in a.reason
        assert "5" in a.reason  # the measured value is named, not hidden

    def test_soft_blur_is_degraded_not_dropped(self):
        report = classify_evidence_items([_item("soft-blur", SOFT_BLUR)])
        assert report.admissions[0].outcome == DEGRADED

    def test_clipping_rejects_and_degrades(self):
        assert classify_evidence_items([_item("blown", BLOWN)]).admissions[0].outcome == REJECTED
        assert classify_evidence_items([_item("soft", SOFT_BLOWN)]).admissions[0].outcome == DEGRADED

    def test_decode_failure_is_failed_not_unresolved(self):
        report = classify_evidence_items([_item("decode-failed", DECODE_FAILED)])
        assert report.admissions[0].outcome == FAILED

    def test_missing_metrics_is_unresolved_never_accepted(self):
        report = classify_evidence_items([_item("no-metrics", NO_METRICS)])
        assert report.admissions[0].outcome == UNRESOLVED

    def test_report_counts_are_consistent_and_deterministic(self):
        items = [_item(f"i{i}", m) for i, m in enumerate(
            [SHARP, BLURRY, SOFT_BLUR, BLOWN, DECODE_FAILED, NO_METRICS]
        )]
        report = classify_evidence_items(items)
        assert sum(report.counts.values()) == len(items)
        # Input order preserved in admissions.
        assert [a.evidence_id for a in report.admissions] == [i.id for i in items]
        # Determinism: same input, same report.
        again = classify_evidence_items(items)
        assert report.to_dict() == again.to_dict()

    def test_admitted_ids_split_by_outcome(self):
        items = [_item("ok", SHARP), _item("bad", BLURRY)]
        report = classify_evidence_items(items)
        assert report.accepted_ids == ["ok"]
        assert report.rejected_ids == ["bad"]


class TestRunClassification:
    def _run(self, status: str):
        from reconstruction.backend.interface import (
            ReconstructedCameraPose,
            ReconstructedPoint,
            ReconstructionResult,
        )

        return ReconstructionResult(
            points=[ReconstructedPoint((0, 0, 0), "t1", ["a"])],
            camera_poses=[ReconstructedCameraPose("a", (0, 0, 0), (1, 0, 0, 0))],
            registration_status=status,
        )

    def test_success_run_is_accepted(self):
        assert classify_reconstruction_run(self._run("success")).outcome == ACCEPTED

    def test_partial_run_is_degraded_with_reason(self):
        outcome = classify_reconstruction_run(self._run("partial"))
        assert outcome.outcome == DEGRADED
        assert "partial" in outcome.reason

    def test_failed_run_is_failed_never_accepted(self):
        outcome = classify_reconstruction_run(self._run("failed"))
        assert outcome.outcome == FAILED


class TestInputGateScale:
    """The orchestrator's input gate is on the path of EVERY run; its
    duplicate check must not be quadratic in the batch size."""

    def test_large_duplicate_scan_is_linear_enough(self):
        n = 20000
        items = [
            EvidenceItem(id=f"img-{i}", kind=EvidenceKind.PHOTO,
                         source_uri=f"file:///{i}.jpg")
            for i in range(n)
        ]
        # One duplicate pair to exercise the duplicate path.
        items.append(EvidenceItem(id="img-0", kind=EvidenceKind.PHOTO,
                                  source_uri="file:///dup.jpg"))
        from reconstruction.orchestrator import (
            EvidenceValidationError,
            validate_evidence,
        )
        start = time.perf_counter()
        with pytest.raises(EvidenceValidationError) as excinfo:
            validate_evidence(items)
        elapsed = time.perf_counter() - start
        # The duplicate is still named in the refusal (no silent skip).
        assert "img-0" in str(excinfo.value)
        # O(N^2) on 20k items is ~40k^2 comparisons (>10 s measured);
        # a linear pass finishes well under a second.
        assert elapsed < 2.0, f"input gate took {elapsed:.2f}s for {n} items"
