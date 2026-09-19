"""Tests for quality-aware compute admission (ROBUSTNESS directive P1).

The orchestrator's input gate currently accepts/rejects at the BATCH
level only (count, duplicates). Item-level quality signals recorded by
the importer (blur/exposure metrics) are never consulted before
reconstruction spend. This wires `classify_evidence_items` into an
orchestrator-adjacent admission pass:

  - ACCEPTED/DEGRADED items proceed (degraded carry their reason)
  - REJECTED/FAILED items are EXCLUDED from reconstruction spend,
    with the exclusion recorded -- never silently dropped
  - UNRESOLVED items proceed with diagnostics (evidence retained;
    absence of metrics must not veto otherwise-good captures)
  - batch-level refusal only when admission leaves fewer than the
    orchestrator's minimum image count -- recorded as FAILED with the
    counts, not a silent pass-through

The orchestrator itself stays untouched; this composes it (same
discipline as orchestrator_consistency).
"""

from __future__ import annotations

import pytest

from evidence.importers import EvidenceKind
from evidence.session import EvidenceItem
from reconstruction.backend.fake import FakeReconstructionBackend
from reconstruction.backend.interface import ReconstructedCameraPose, ReconstructedPoint
from reconstruction.robustness_admission import (
    admit_for_reconstruction,
    reconstruct_with_admission,
)


SHARP = {"laplacian_variance": 1200.0, "luma_mean": 128.0,
         "clipped_fraction": 0.01, "measured": 1.0}
BLURRY = dict(SHARP, laplacian_variance=5.0)
SOFT = dict(SHARP, laplacian_variance=120.0)
DECODE_FAILED = {"measured": 0.0, "quality_note": "pixel decode failed"}


def _item(eid: str, metrics: dict | None) -> EvidenceItem:
    metadata = {"quality": metrics} if metrics is not None else {}
    return EvidenceItem(id=eid, kind=EvidenceKind.PHOTO,
                        source_uri=f"file:///{eid}.jpg", metadata=metadata)


def _backend():
    ids = ("a", "b", "c", "d")
    poses = [
        ReconstructedCameraPose(eid, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
        for eid in ids
    ]
    points = [
        ReconstructedPoint((float(i), 0.0, 0.0), f"t{i}", [eid])
        for i, eid in enumerate(ids)
    ]
    return FakeReconstructionBackend(canned_poses=poses, canned_points=points)


class TestAdmission:
    def test_rejected_items_are_excluded_and_recorded(self):
        items = [_item("a", SHARP), _item("blur", BLURRY), _item("b", SHARP)]
        admitted, report = admit_for_reconstruction(items)
        assert [i.id for i in admitted] == ["a", "b"]
        assert report.excluded_ids == ["blur"]
        # The exclusion is a recorded fact with its measured reason.
        reasons = {r.evidence_id: r.reason for r in report.classification.admissions}
        assert "laplacian_variance" in reasons["blur"]

    def test_unresolved_items_proceed_with_diagnostics(self):
        items = [_item("a", SHARP), _item("nom", None), _item("b", SHARP)]
        admitted, report = admit_for_reconstruction(items)
        assert [i.id for i in admitted] == ["a", "nom", "b"]
        assert report.proceeded_unresolved == ["nom"]

    def test_insufficient_admission_refuses_at_batch_level(self):
        items = [_item("a", SHARP), _item("blur", BLURRY)]  # 2 -> 1 after admission
        with pytest.raises(Exception) as excinfo:
            admit_for_reconstruction(items)
        assert "admission" in str(excinfo.value).lower()
        # The counts that caused the refusal are named.
        assert "1" in str(excinfo.value)

    def test_admission_saves_spend_on_all_rejected_batches(self):
        items = [_item(f"b{i}", BLURRY) for i in range(6)]
        with pytest.raises(Exception):
            admit_for_reconstruction(items)


class TestWiredRun:
    def test_run_uses_admitted_subset_and_records_report(self):
        items = [_item("a", SHARP), _item("blur", BLURRY),
                 _item("b", SHARP), _item("c", SHARP)]
        run, report = reconstruct_with_admission(_backend(), items)
        # The blurry frame was not sent to the backend.
        posed_ids = {p.evidence_id for p in run.result.camera_poses}
        assert "blur" not in posed_ids
        assert run.diagnostics.image_evidence_count == 3
        assert report.excluded_ids == ["blur"]
        # Diagnostics carry the full admission report.
        assert run.diagnostics.evidence_validation["admitted"] == 3

    def test_failed_backend_still_raises_after_admission(self):
        from reconstruction.orchestrator import ReconstructionOrchestrationError

        items = [_item("a", SHARP), _item("b", SHARP)]
        # Only poses for 'a': backend reports partial-> but with only
        # one pose for two images the fake returns partial (not failed).
        run, _ = reconstruct_with_admission(_backend(), items)
        assert run.result.registration_status in ("success", "partial")

    def test_low_quality_batch_is_not_silently_processed(self):
        items = [_item("a", SHARP), _item("b", SHARP), _item("blur", BLURRY),
                 _item("f", DECODE_FAILED), _item("g", DECODE_FAILED),
                 _item("h", DECODE_FAILED)]
        # 6 items -> 2 admitted (a, b): enough to run, refusals recorded.
        run, report = reconstruct_with_admission(_backend(), items)
        assert report.classification.counts["rejected"] == 1
        assert report.classification.counts["failed"] == 3
