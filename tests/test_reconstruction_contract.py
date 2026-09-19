"""Tests for the reconstruction contract surface (perception/contract.py,
RECONSTRUCTION_CONTRACT_V1 handoff to the frontend/agent consumers).

Contract rules under test (multi-agent execution contract: "A HANDOFF IS A
TESTED INTERFACE"; honesty vocabulary UNAVAILABLE/BLOCKED/DEGRADED):

  - The contract COMPOSES the canonical stage types (DepthMap,
    ReconstructedCameraPose, Detection, SegmentedRegion, FusedEstimate,
    EvidenceQualityReport, ReconstructionResult) -- it never redefines a
    competing shape. Identity checks prove the reuse.
  - Every stage result carries an explicit status:
    "complete" | "partial" | "unavailable" | "failed". Absence of data is
    never silently upgraded to success, and data is never carried under a
    failed/unavailable status (zombie data forbidden).
  - status "complete" with a missing payload raises; payload under
    "failed"/"unavailable" raises. ObservationResult's empty lists are
    legitimate only when the stage actually ran (no detections found is a
    real outcome; a failed stage carrying detections is not).
  - to_dict exposes every field the UI needs, including the reason detail
    for unavailable/failed stages and the nested canonical payload.
  - contract_reconstruction_from_run composes the top-level surface from
    the orchestrator's ReconstructionRun verbatim: status = diagnostics'
    final_status, error text preserved in detail -- never rewritten.
"""

from __future__ import annotations

import unittest

from evidence.session import EvidenceItem, EvidenceKind
from perception.contract import (
    CONTRACT_VERSION,
    ContractError,
    DepthResult,
    FusionResult,
    ObservationResult,
    PoseResult,
    QualityResult,
    StageStatus,
    contract_reconstruction_from_run,
)
from perception.depth.interface import DepthMap
from perception.quality.assessment import EvidenceQualityReport
from reconstruction.backend.interface import (
    ReconstructionResult as CanonicalReconstructionResult,
    ReconstructedCameraPose,
    ReconstructedPoint,
)
from reconstruction.orchestrator import (
    ReconstructionRun,
    ReconstructionRunDiagnostics,
)


def _depth_map() -> DepthMap:
    return DepthMap(
        evidence_id="photo-1",
        width=2,
        height=2,
        values=[[1.0, 1.2], [0.9, 1.1]],
        unit="meters",
    )


class TestStageStatus(unittest.TestCase):
    def test_status_vocabulary_is_exact(self):
        self.assertEqual(
            {s.value for s in StageStatus},
            {"complete", "partial", "unavailable", "failed"},
        )

    def test_contract_version(self):
        self.assertEqual(CONTRACT_VERSION, "1.0")


class TestDepthResult(unittest.TestCase):
    def test_complete_carries_canonical_depth_map(self):
        r = DepthResult(evidence_id="photo-1", status=StageStatus.COMPLETE,
                        depth_map=_depth_map())
        d = r.to_dict()
        self.assertEqual(d["status"], "complete")
        self.assertEqual(d["depth_map"]["evidence_id"], "photo-1")
        self.assertEqual(d["depth_map"]["unit"], "meters")
        self.assertNotIn("detail", d)

    def test_complete_without_payload_raises(self):
        with self.assertRaises(ContractError):
            DepthResult(evidence_id="photo-1", status=StageStatus.COMPLETE)

    def test_unavailable_carries_reason_not_data(self):
        r = DepthResult(evidence_id="photo-1", status=StageStatus.UNAVAILABLE,
                        detail="MiDaS checkpoint not installed")
        d = r.to_dict()
        self.assertEqual(d["status"], "unavailable")
        self.assertEqual(d["detail"], "MiDaS checkpoint not installed")
        self.assertNotIn("depth_map", d)

    def test_payload_under_failed_status_raises(self):
        with self.assertRaises(ContractError):
            DepthResult(evidence_id="photo-1", status=StageStatus.FAILED,
                        depth_map=_depth_map(),
                        detail="backend crashed mid-run")

    def test_identity_with_canonical_depth_map(self):
        # The contract composes the canonical DepthMap -- no competing type.
        r = DepthResult(evidence_id="photo-1", status=StageStatus.COMPLETE,
                        depth_map=_depth_map())
        self.assertIs(type(r.depth_map), DepthMap)


class TestPoseResult(unittest.TestCase):
    def test_complete_carries_canonical_pose(self):
        pose = ReconstructedCameraPose(
            evidence_id="photo-1", position=(0.0, 0.0, 5.0),
            rotation=(1.0, 0.0, 0.0, 0.0))
        r = PoseResult(evidence_id="photo-1", status=StageStatus.COMPLETE,
                       pose=pose)
        d = r.to_dict()
        self.assertEqual(d["status"], "complete")
        self.assertEqual(d["pose"]["evidence_id"], "photo-1")

    def test_failed_with_pose_raises(self):
        pose = ReconstructedCameraPose(
            evidence_id="photo-1", position=(0.0, 0.0, 5.0),
            rotation=(1.0, 0.0, 0.0, 0.0))
        with self.assertRaises(ContractError):
            PoseResult(evidence_id="photo-1", status=StageStatus.FAILED,
                       pose=pose, detail="pose solver diverged")

    def test_unavailable_names_reason(self):
        r = PoseResult(evidence_id="photo-1", status=StageStatus.UNAVAILABLE,
                       detail="no COLMAP binary on PATH")
        self.assertEqual(r.to_dict()["detail"], "no COLMAP binary on PATH")


class TestObservationResult(unittest.TestCase):
    def test_complete_with_empty_observations_is_legal(self):
        # Genuinely ran, found nothing: a real outcome, not a failure.
        r = ObservationResult(evidence_id="photo-1", status=StageStatus.COMPLETE)
        self.assertEqual(r.to_dict()["detections"], [])
        self.assertEqual(r.to_dict()["segmentations"], [])

    def test_failed_cannot_carry_observations(self):
        with self.assertRaises(ContractError):
            ObservationResult(
                evidence_id="photo-1", status=StageStatus.FAILED,
                detections=[{"label": "ghost"}],
                detail="detector crashed")

    def test_unavailable_cannot_carry_observations(self):
        with self.assertRaises(ContractError):
            ObservationResult(
                evidence_id="photo-1", status=StageStatus.UNAVAILABLE,
                segmentations=[{"label": "ghost-region"}],
                detail="SAM weights missing")


class TestFusionResult(unittest.TestCase):
    def test_unavailable_without_fused_estimate(self):
        r = FusionResult(quantity="room_width",
                         status=StageStatus.UNAVAILABLE,
                         detail="only one source observed this quantity")
        d = r.to_dict()
        self.assertEqual(d["quantity"], "room_width")
        self.assertNotIn("fused", d)

    def test_complete_requires_fused_estimate(self):
        with self.assertRaises(ContractError):
            FusionResult(quantity="room_width", status=StageStatus.COMPLETE)


class TestQualityResult(unittest.TestCase):
    def test_complete_carries_canonical_report(self):
        report = EvidenceQualityReport(
            gsd_mm_per_px=12.5, detail_tier="medium",
            observed_fraction=0.83, view_counts={"p1": 4, "p2": 3},
            unprojectable_point_ids=(), overclaim_count=0)
        r = QualityResult(status=StageStatus.COMPLETE, report=report)
        d = r.to_dict()
        self.assertEqual(d["report"]["detail_tier"], "medium")
        self.assertEqual(d["report"]["gsd_mm_per_px"], 12.5)

    def test_complete_requires_report(self):
        with self.assertRaises(ContractError):
            QualityResult(status=StageStatus.COMPLETE)


class TestReconstructionContractFromRun(unittest.TestCase):
    def _run(self, final_status: str, error: str = "") -> ReconstructionRun:
        result = CanonicalReconstructionResult(
            points=[ReconstructedPoint(position=(0.0, 0.0, 0.0), track_id="t1",
                                       source_evidence_ids=["photo-1"])],
            camera_poses=[ReconstructedCameraPose(
                evidence_id="photo-1", position=(0.0, 0.0, 5.0),
                rotation=(1.0, 0.0, 0.0, 0.0))],
            registration_status=final_status,
        )
        diagnostics = ReconstructionRunDiagnostics(
            evidence_count=1, image_evidence_count=1,
            evidence_validation={"total": 1, "image": 1, "kinds": ["photo"],
                                 "issues": []},
            attempts=(), final_status=final_status, error=error)
        return ReconstructionRun(result=result, diagnostics=diagnostics)

    def test_success_run_maps_verbatim(self):
        run = self._run("success")
        r = contract_reconstruction_from_run(run)
        self.assertEqual(r.status, "success")
        self.assertIs(r.result, run.result)  # canonical result carried VERBATIM
        self.assertEqual(len(r.result.points), 1)
        d = r.to_dict()
        self.assertEqual(d["result"]["registration_status"], "success")
        self.assertEqual(d["result"]["point_count"], 1)

    def test_partial_run_preserves_error_text(self):
        r = contract_reconstruction_from_run(
            self._run("partial", error="2 of 8 poses unresolved"))
        d = r.to_dict()
        self.assertEqual(d["status"], "partial")
        self.assertEqual(d["detail"], "2 of 8 poses unresolved")
        self.assertIn("attempts", d["diagnostics"])

    def test_failed_run_is_failed_not_success(self):
        run = self._run("failed", error="bundle adjustment did not converge")
        r = contract_reconstruction_from_run(run)
        self.assertEqual(r.status, "failed")
        self.assertIsNone(r.result)  # failed runs carry no result payload
        self.assertNotIn("result", r.to_dict())
        self.assertEqual(r.detail, "bundle adjustment did not converge")


if __name__ == "__main__":
    unittest.main()
