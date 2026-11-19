# Benchmark Integrity and Honest Ground Truth Report

**Date:** 2026-11-19  
**Module:** `benchmarks/measured_rooms/universal_benchmark.py`  
**Status:** Audit & Remediation Complete

---

## 1. Executive Summary

A zero-tolerance audit of the Reality Engine benchmark and evaluation framework was conducted to eliminate all forms of manufactured accuracy, synthetic ground-truth reading, and unearned performance metrics.

Benchmark integrity is essential: claiming millimeter accuracy or perfect part association without physical millimeter ground truth falsifies system capabilities.

---

## 2. Key Audit Findings and Remediations

### 2.1 Elimination of Synthetic Ground-Truth Reading
- **Defect:** Previously, `universal_benchmark.py` read ground-truth bounding box dimensions and added artificial random noise ($\pm 0.003$ m to $\pm 0.005$ m) to simulate "reconstructed" entities, falsely claiming sub-centimeter accuracy without running actual perception or geometry models.
- **Remediation:** Removed synthetic ground-truth reading entirely. When running benchmark evaluations, only real reconstructions produced by calibrated perception, depth unprojection, and fusion are measured against physical ground truth.
- **Isolated Smoke Tests:** When testing benchmark reporting mechanics in the absence of real reconstructed models, tests must explicitly pass `data_type="synthetic_smoke_test"`, clearly labeling the run as synthetic test-double verification rather than physical room accuracy.

### 2.2 Removal of Hardcoded Part Association Accuracy
- **Defect:** `part_association_accuracy` was hardcoded to `1.0` (100%), even when no part annotations or associations were present.
- **Remediation:** `part_association_accuracy` is now calculated dynamically from actual ground-truth part relationships. When ground truth contains no part annotations, the metric returns `None` with an explicit reason, never a fabricated `1.0`.

### 2.3 Strict Handling of Missing Physical Ground Truth
- When evaluating datasets that lack millimeter-grade laser surveys, high-precision CAD ground truth, or source photos, `evaluate_dataset()` reports `status="BLOCKED"` with an explicit rationale (e.g., `"Physical survey ground truth or source photos missing"`).
- It never fabricates synthetic geometry or creates plausible-looking metric errors.

### 2.4 Pinhole Camera Rigidity Validation
- Camera poses used in benchmarks must pass `VisibilityReprojectionValidator.validate_camera_metadata`.
- Extrinsics are strictly validated for $SO(3)$ rotation matrix properties:
  $$|\det(R) - 1.0| \le 10^{-3}, \quad \|R^T R - I\|_\infty \le 10^{-3}$$
- Any malformed or uncalibrated camera input immediately fails validation rather than falling back to an unverified identity camera.

---

## 3. Benchmark Verification Suite

The remediation is verified by `tests/test_benchmark_integrity.py`:
1. `test_benchmark_blocked_on_missing_photos_or_ground_truth`: Asserts `BLOCKED` status when physical data is absent.
2. `test_benchmark_isolated_synthetic_smoke_test`: Asserts synthetic test runs are explicitly flagged with `data_type="synthetic_smoke_test"` and do not claim physical accuracy.
3. `test_benchmark_no_hardcoded_part_association`: Asserts `part_association_accuracy` is `None` or dynamically evaluated, never defaulting to `1.0`.
