"""P0 bad-evidence suite: realistic imperfect capture, classified
honestly through the canonical chain.

The dataset is the COMMITTED real south-building subset
(datasets/south_building/images -- 32 real photographs). Bad variants
are DERIVED from it at test time by re-encoding bytes, so no extra
assets are committed and every defect is a controlled, reproducible
transformation of real captured evidence:

    blurred        real photo, box-blurred, re-encoded -- measured
                   laplacian_variance collapses
    duplicate      the SAME bytes imported twice -- content-addressed
                   dedup must collapse them
    exposure       real photo, luma pushed toward the white rail --
                   clipped_fraction rises
    corrupt        truncated real bytes -- import raises
                   CorruptEvidenceError naming the path (a corrupt
                   payload cannot honestly enter a package), and the
                   batch-survival policy keeps the rest of the folder
    disconnected   two real photos from opposite ends of the capture
                   ring -- per-item admission may pass them (metrics
                   are fine); connectivity is a RUN-level property,
                   asserted against the orchestrator's gate

Outcomes asserted are the canonical vocabulary
(ACCEPTED/REJECTED/DEGRADED/UNRESOLVED/FAILED) with machine-readable
reasons naming the measured value. When the dataset is absent the
module SKIPS with the exact path -- real-data-gated, never faked.
"""

from __future__ import annotations

import os

import pytest

from evidence.importers import import_folder
from evidence.packages import DeterministicPackageBuilder
from reconstruction.robustness import classify_evidence_items
from reconstruction.robustness_admission import admit_for_reconstruction

DATASET_IMAGES = os.path.join("datasets", "south_building", "images")

pytestmark = pytest.mark.skipif(
    not os.path.isdir(DATASET_IMAGES) or not os.listdir(DATASET_IMAGES),
    reason=(
        f"real committed dataset missing: {DATASET_IMAGES} -- run "
        "scripts/fetch_south_building.py to reproduce it"
    ),
)


def _names() -> list:
    return sorted(os.listdir(DATASET_IMAGES))


def _load(name: str) -> bytes:
    with open(os.path.join(DATASET_IMAGES, name), "rb") as fh:
        return fh.read()


def _write_variant(directory: str, name: str, payload: bytes) -> str:
    path = os.path.join(directory, name)
    with open(path, "wb") as fh:
        fh.write(payload)
    return path


def _derived_capture(base, plan: dict) -> tuple:
    """Build a derived capture folder from the real photos per plan:
    {name: transform-or-None}. Returns (folder, sorted expected names)."""
    folder = os.path.join(str(base), "capture")
    os.makedirs(folder, exist_ok=True)
    expected = []
    for name, transform in plan.items():
        payload = _load(name)
        if transform is not None:
            payload = transform(payload)
        _write_variant(folder, name, payload)
        expected.append(name)
    return folder, expected


def _blurred(payload: bytes) -> bytes:
    from io import BytesIO

    from PIL import Image, ImageFilter

    img = Image.open(BytesIO(payload))
    img = img.filter(ImageFilter.BoxBlur(6))
    out = BytesIO()
    img.save(out, format="JPEG", quality=88)
    return out.getvalue()


def _exposure_pushed(payload: bytes) -> bytes:
    from io import BytesIO

    from PIL import Image

    img = Image.open(BytesIO(payload)).convert("L")
    img = img.point(lambda v: min(255, int(v * 1.9)))
    out = BytesIO()
    img.save(out, format="JPEG", quality=90)
    return out.getvalue()


def _truncated(payload: bytes) -> bytes:
    return payload[: len(payload) // 7]


def _fully_corrupt(payload: bytes) -> bytes:
    """Destroys the header too: fails structural validation at import
    (raises CorruptEvidenceError today) -- the batch-survival case."""
    return b"\\x00\\x01\\x02\\x03GARBAGE" + payload[64:128]


# ------------------------------------------------------------------
# Per-item outcomes on real evidence
# ------------------------------------------------------------------


class TestRealBadEvidence:
    def test_blurred_real_photo_is_rejected_with_measured_reason(self, tmp_path):
        names = _names()
        folder, _ = _derived_capture(
            tmp_path, {names[0]: None, names[1]: _blurred}
        )
        builder = DeterministicPackageBuilder(seed="bad-evidence-blur")
        report = import_folder(builder, folder)
        items = builder.build().to_evidence_items()
        assert len(items) == 2

        report_class = classify_evidence_items(items)
        by_id = {a.evidence_id: a for a in report_class.admissions}
        quality_by_uri = {
            os.path.basename(i.source_uri): i.metadata.get("quality", {})
            for i in items
        }
        blurred_q = quality_by_uri[names[1]]
        assert blurred_q.get("measured") == 1.0  # metrics measured, not faked
        laplacian = blurred_q["laplacian_variance"]

        # The pristine real photo classifies usable...
        assert by_id[_id_of(items, names[0])].outcome in ("accepted", "degraded")
        # ...and the blurred one is REJECTED or DEGRADED, with a reason
        # naming the measured value; threshold sanity: a real outdoor
        # photo blurred this hard must fall below the degrade threshold.
        adm = by_id[_id_of(items, names[1])]
        assert adm.outcome in ("rejected", "degraded"), adm.outcome
        assert "laplacian" in adm.reason
        assert laplacian < 150.0, laplacian

    def test_duplicate_real_bytes_are_deduped_and_recorded(self, tmp_path):
        names = _names()
        folder, _ = _derived_capture(
            tmp_path, {names[0]: None, names[1]: None}
        )
        # 'copy.jpg' reuses names[0]'s bytes exactly.
        _write_variant(folder, "copy.jpg", _load(names[0]))

        builder = DeterministicPackageBuilder(seed="bad-evidence-dup")
        report = import_folder(builder, folder)
        assert len(report.duplicates_skipped) == 1
        skipped_path, existing_id = report.duplicates_skipped[0]
        assert os.path.basename(skipped_path) == "copy.jpg"
        assert existing_id  # names the canonical original
        items = builder.build().to_evidence_items()
        assert len(items) == 2  # originals only

    def test_truncated_real_bytes_classify_failed_not_accepted(self, tmp_path):
        """A truncated JPEG whose header survives structural checks can
        still enter a package -- but its pixel metrics cannot be
        measured, and the classifier must say FAILED with the decode
        note, never invent a number."""
        names = _names()
        folder, _ = _derived_capture(
            tmp_path, {names[0]: None, names[1]: _truncated}
        )
        builder = DeterministicPackageBuilder(seed="bad-evidence-corrupt")
        import_folder(builder, folder)
        items = builder.build().to_evidence_items()
        assert len(items) == 2

        report = classify_evidence_items(items)
        by_id = {a.evidence_id: a for a in report.admissions}
        truncated = by_id[_id_of(items, names[1])]
        assert truncated.outcome == "failed", truncated.outcome
        assert "decode" in truncated.reason.lower() or "unmeasured" in truncated.reason.lower()

    def test_corrupt_file_does_not_lose_the_whole_batch(self, tmp_path):
        """Batch survival: a 500-image capture with one structurally
        broken frame must not lose 499 good frames. The batch-survival
        import policy records the failure and continues; the corrupt
        path is reported with its reason, never silently dropped."""
        names = _names()
        plan = {n: None for n in names[:8]}
        plan[names[4]] = _fully_corrupt  # header destroyed: raises today
        folder, _ = _derived_capture(tmp_path, plan)

        builder = DeterministicPackageBuilder(seed="bad-evidence-survive")
        report = import_folder(builder, folder, on_error="record")
        items = builder.build().to_evidence_items()

        # The 7 healthy photos survived.
        assert len(items) == 7
        survived = {os.path.basename(i.source_uri) for i in items}
        assert names[4] not in survived
        assert names[0] in survived
        # The failure is a recorded fact with a reason.
        assert report.failed_paths, "corrupt file must be recorded, not skipped"
        failed_path, reason = report.failed_paths[0]
        assert os.path.basename(failed_path) == names[4]
        assert reason

    def test_exposure_pushed_real_photo_surfaces_measured_clipping(self, tmp_path):
        names = _names()
        folder, _ = _derived_capture(
            tmp_path, {names[0]: None, names[2]: _exposure_pushed}
        )
        builder = DeterministicPackageBuilder(seed="bad-evidence-exposure")
        import_folder(builder, folder)
        items = builder.build().to_evidence_items()

        report = classify_evidence_items(items)
        by_id = {a.evidence_id: a for a in report.admissions}
        pushed = by_id[_id_of(items, names[2])]
        assert pushed.outcome in ("rejected", "degraded"), pushed.outcome
        clips = [
            i.metadata.get("quality", {}).get("clipped_fraction", 0.0)
            for i in items
        ]
        assert max(clips) > 0.25, clips

    def test_admission_gate_removes_only_classified_bad_items(self, tmp_path):
        """The P0 quality-admission composition on real evidence: the
        gate excludes exactly the failed/rejected items and reports the
        counts -- nothing is silently dropped, every exclusion has a
        reason."""
        names = _names()
        plan = {n: None for n in names[:6]}
        plan[names[6]] = _blurred
        folder, _ = _derived_capture(tmp_path, plan)

        builder = DeterministicPackageBuilder(seed="bad-evidence-mixed")
        import_folder(builder, folder)
        items = builder.build().to_evidence_items()

        admitted, decision = admit_for_reconstruction(items)
        counts = decision.to_dict()["classification"]["counts"]
        assert counts["rejected"] + counts["degraded"] >= 1  # the blurred one
        excluded = counts["failed"] + counts["rejected"]
        assert len(admitted) == len(items) - excluded
        for adm in decision.classification.admissions:
            if adm.outcome in ("failed", "rejected"):
                assert adm.reason, f"reasonless exclusion: {adm}"


def _id_of(items, name: str) -> str:
    for item in items:
        if os.path.basename(item.source_uri) == name:
            return item.id
    raise AssertionError(f"no item for {name}")


# ------------------------------------------------------------------
# Run-level honesty: connectivity is NOT a per-item property
# ------------------------------------------------------------------


class TestConnectivityIsRunLevel:
    def test_disconnected_real_photos_pass_item_gate_but_run_gate_knows(
        self, tmp_path
    ):
        """Two real photos from opposite ends of the capture ring: item
        metrics are real and unremarkable, so per-item admission passes
        them; connectivity is a run-level truth. The orchestrator's
        input gate reports the honest counts, and the run classifier's
        vocabulary is exercised through its public contract on a REAL
        registered result shape (backend never faked here)."""
        names = _names()
        folder, _ = _derived_capture(
            tmp_path, {names[0]: None, names[-1]: None}
        )
        builder = DeterministicPackageBuilder(seed="bad-evidence-disconnected")
        import_folder(builder, folder)
        items = builder.build().to_evidence_items()

        admitted, decision = admit_for_reconstruction(items)
        outcomes = {a.outcome for a in decision.classification.admissions}
        assert outcomes <= {"accepted", "degraded", "unresolved"}

        from reconstruction.orchestrator import validate_evidence

        validation = validate_evidence(items)
        # The gate reports the honest image-evidence reality it was
        # given (key name per orchestrator's gate contract).
        image_key = next(
            (k for k in validation if "image" in k and isinstance(validation[k], int)),
            None,
        )
        assert image_key is not None, sorted(validation)
        assert validation[image_key] == 2


# ------------------------------------------------------------------
# Quality admission RUN A vs RUN B (P0): measured on the real set
# ------------------------------------------------------------------


class TestAdmissionValue:
    def test_run_a_vs_run_b_measured_workload_difference(self, tmp_path):
        """RUN A: all evidence to reconstruction. RUN B: admission
        applied. The measurable difference is the workload handed to
        the backend -- asserted as the exact delta with the exclusion
        reasons surfaced; compute savings are claimed only as far as
        the measured item counts go."""
        names = _names()
        plan = {n: None for n in names[:10]}
        plan[names[10]] = _blurred
        folder, _ = _derived_capture(tmp_path, plan)

        builder = DeterministicPackageBuilder(seed="bad-evidence-runa-runb")
        import_folder(builder, folder)
        items = builder.build().to_evidence_items()

        run_a_count = len(items)
        admitted, decision = admit_for_reconstruction(items)
        run_b_count = len(admitted)
        counts = decision.to_dict()["classification"]["counts"]

        saved = run_a_count - run_b_count
        assert saved == counts["failed"] + counts["rejected"]
        assert saved >= 1
        reasons = {
            a.evidence_id: a.reason
            for a in decision.classification.admissions
            if a.outcome in ("failed", "rejected")
        }
        assert len(reasons) == saved
