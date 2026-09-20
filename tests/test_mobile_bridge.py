"""Tests for apps/cli/mobile_bridge.py -- the engine-side mobile bundle bridge.

Covers the full mobile <-> engine loop against real files on disk (no
mocking): a bundle built with real JPEG payloads (same fixture approach
as tests/test_cli.py) is verified, ingested into a real MultiSourceSession
through the real add_source path, and re-fed to the task derivation.
Integrity failures (tampered payloads), source-level dedupe, telemetry
absence (UNAVAILABLE, never guessed), and a 200-frame large bundle are
all exercised. The bundle JSON here mirrors what the phone writes
(frontend/src/apps/mobile/bundle.ts, re.mobile-session-bundle/v1) --
same schema, same frame identity rule (frame:<sha256[0:16]>).
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import time

import pytest

from apps.cli.main import main
from apps.cli.mobile_bridge import (
    MobileBundleError,
    MobileBundleIntegrityError,
    compare_coverage_snapshots,
    derive_capture_tasks,
    ingest_mobile_bundle_to_dir,
    load_mobile_bundle,
    merge_capture_tasks,
    parse_exported_at,
    write_capture_tasks,
)
from evidence.multi_source import MultiSourceSession


def _jpeg_bytes(color) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color=color).save(buf, format="JPEG")
    return buf.getvalue()


def _frame_entry(color, verdict="USEFUL", reasons=("sharpness_ok", "exposure_ok"), telemetry=None):
    payload = _jpeg_bytes(color)
    sha = hashlib.sha256(payload).hexdigest()
    return {
        "frameId": f"frame:{sha[:16]}",
        "contentSha256": sha,
        "capturedAt": "2026-09-20T10:00:00Z",
        "width": 64,
        "height": 64,
        "bytes": len(payload),
        "mime": "image/jpeg",
        "verdict": verdict,
        "reasons": list(reasons),
        "quality": {
            "sharpness": 500, "exposure": 0.5, "clippedRatio": 0.0, "diffVsPrevious": 0.3,
        },
        "telemetry": telemetry or {},
        "provenance": {
            "origin": "device_camera", "device": "test-phone",
            "capturedBy": "operator", "ingestion": "mobile_local",
        },
        "taskId": None,
        "payloadBase64": base64.b64encode(payload).decode(),
    }


def _bundle_doc(
    frames,
    session_id="sess-mobile-1",
    bundle_id="bundle:test00000000001",
    exported_at="2026-09-20T10:05:00Z",
):
    return {
        "schema": "re.mobile-session-bundle/v1",
        "bundleId": bundle_id,
        "exportedAt": exported_at,
        "device": "test-phone",
        "session": {
            "sessionId": session_id,
            "name": "Test capture",
            "intent": "field",
            "createdAt": "2026-09-20T10:00:00Z",
            "updatedAt": "2026-09-20T10:05:00Z",
            "status": "CAPTURING",
        },
        "frames": frames,
        "skippedFrames": [{"frameId": "frame:gone000000000000", "reason": "local_copy_missing"}],
        "taskOutcomes": [],
    }


def _telemetry(heading=None, lat=None, lon=None):
    out = {}
    if heading is not None:
        out["headingDeg"] = heading
    if lat is not None:
        out["geolocation"] = {"latitude": lat, "longitude": lon or 2.0, "accuracyM": 5}
    return out


def _write_bundle(tmp_path, doc, name="bundle.json"):
    path = tmp_path / name
    path.write_text(json.dumps(doc), encoding="utf-8")
    return str(path)


def _new_session(session_id="sess-mobile-1"):
    return MultiSourceSession(session_id=session_id, name="Test capture")


class TestLoadMobileBundle:
    def test_roundtrip_verifies_and_counts(self, tmp_path):
        doc = _bundle_doc([
            _frame_entry((255, 0, 0), telemetry=_telemetry(heading=10.0)),
            _frame_entry((0, 255, 0), verdict="REJECTED", reasons=("sharpness_low",)),
        ])
        bundle = load_mobile_bundle(_write_bundle(tmp_path, doc))
        assert len(bundle.frames) == 2
        assert bundle.integrity_failures == []
        assert bundle.skipped_frames == [
            {"frameId": "frame:gone000000000000", "reason": "local_copy_missing"}
        ]
        assert bundle.session_id == "sess-mobile-1"
        first = bundle.frames[0]
        assert first.frame_id == f"frame:{hashlib.sha256(first.payload).hexdigest()[:16]}"

    def test_tampered_payload_is_integrity_failure_not_evidence(self, tmp_path):
        entry = _frame_entry((255, 0, 0))
        entry["payloadBase64"] = base64.b64encode(_jpeg_bytes((1, 2, 3))).decode()
        bundle = load_mobile_bundle(_write_bundle(tmp_path, _bundle_doc([entry])))
        assert bundle.frames == []
        assert len(bundle.integrity_failures) == 1
        assert "does not match" in bundle.integrity_failures[0]["reason"]

    def test_wrong_frame_id_is_rejected(self, tmp_path):
        entry = _frame_entry((9, 9, 9))
        entry["frameId"] = "frame:deadbeefdeadbeef"
        bundle = load_mobile_bundle(_write_bundle(tmp_path, _bundle_doc([entry])))
        assert bundle.frames == []
        assert len(bundle.integrity_failures) == 1

    def test_structural_errors_raise(self, tmp_path):
        bad_schema = tmp_path / "bad.json"
        bad_schema.write_text(json.dumps({"schema": "nope"}), encoding="utf-8")
        with pytest.raises(MobileBundleError):
            load_mobile_bundle(str(bad_schema))
        with pytest.raises(MobileBundleError):
            load_mobile_bundle(str(tmp_path / "missing.json"))

    def test_missing_session_id_raises(self, tmp_path):
        doc = _bundle_doc([])
        del doc["session"]["sessionId"]
        with pytest.raises(MobileBundleIntegrityError):
            load_mobile_bundle(_write_bundle(tmp_path, doc))


class TestIngest:
    def test_ingest_lands_frames_as_real_assets(self, tmp_path):
        doc = _bundle_doc([
            _frame_entry((255, 0, 0), telemetry=_telemetry(heading=10.0)),
            _frame_entry((0, 255, 0)),
            _frame_entry((0, 0, 255), verdict="REJECTED", reasons=("sharpness_low",)),
        ])
        bundle = load_mobile_bundle(_write_bundle(tmp_path, doc))
        session = _new_session()
        report = ingest_mobile_bundle_to_dir(bundle, session, tmp_path / "session")

        assert report["frames_verified"] == 3
        assert report["frames_integrity_failed"] == 0
        assert report["source_status"] == "ingested"
        assert len(session.package.all_assets()) == 3
        assert report["advisory_triage_recorded"] == 3

        restored = MultiSourceSession.from_dict(json.loads(json.dumps(session.to_dict())))
        assert len(restored.package.all_assets()) == 3

    def test_reingest_dedupes_no_silent_loss(self, tmp_path):
        doc = _bundle_doc([_frame_entry((10, 20, 30)), _frame_entry((40, 50, 60))])
        bundle = load_mobile_bundle(_write_bundle(tmp_path, doc))
        session = _new_session()
        first = ingest_mobile_bundle_to_dir(bundle, session, tmp_path / "session")
        second = ingest_mobile_bundle_to_dir(bundle, session, tmp_path / "session")
        assert first["files_written"] == 2
        assert second["files_written"] == 0
        assert second["files_already_present"] == 2
        assert len(session.package.all_assets()) == 2  # no duplication

    def test_integrity_failures_listed_and_never_ingested(self, tmp_path):
        good = _frame_entry((255, 0, 0))
        bad = _frame_entry((0, 255, 0))
        bad["payloadBase64"] = base64.b64encode(b"not a jpeg but real bytes").decode()
        bundle = load_mobile_bundle(_write_bundle(tmp_path, _bundle_doc([good, bad])))
        session = _new_session()
        report = ingest_mobile_bundle_to_dir(bundle, session, tmp_path / "session")
        assert report["frames_verified"] == 1
        assert report["frames_integrity_failed"] == 1
        assert len(session.package.all_assets()) == 1
        assert report["integrity_failures"][0]["frameId"] == bad["frameId"]

    def test_task_outcome_recorded(self, tmp_path):
        entry = _frame_entry((255, 0, 0))
        doc = _bundle_doc([entry])
        task_id = "task-abc123"
        entry["taskId"] = task_id
        doc["taskOutcomes"] = [{
            "taskId": task_id, "status": "DONE",
            "frameIds": [entry["frameId"]], "completedAt": "2026-09-20T10:04:00Z",
        }]
        bundle = load_mobile_bundle(_write_bundle(tmp_path, doc))
        session = _new_session()
        report = ingest_mobile_bundle_to_dir(bundle, session, tmp_path / "session")
        assert report["task_outcomes_recorded"] == 1



class TestDeriveCaptureTasks:
    def test_coverage_gaps_from_real_headings(self, tmp_path):
        doc = _bundle_doc([
            _frame_entry((1, 0, 0), telemetry=_telemetry(heading=10.0, lat=1.0)),
            _frame_entry((0, 2, 0), telemetry=_telemetry(heading=90.0, lat=1.0001)),
        ])
        bundle = load_mobile_bundle(_write_bundle(tmp_path, doc))
        payload = derive_capture_tasks(bundle)
        analysis = payload["coverageAnalysis"]
        assert analysis["state"] == "AVAILABLE"
        assert analysis["coveredOctants"] == ["E", "N"]
        assert analysis["headingFrames"] == 2
        assert analysis["totalOctants"] == 8
        kinds = [(t["kind"], t["region"]["octant"]) for t in payload["tasks"]]
        assert ("coverage_gap", "NE") in kinds
        assert ("coverage_gap", "S") in kinds
        # every task carries the full desktop->mobile contract, in the exact
        # camelCase shape the phone's importTaskFile parses
        for task in payload["tasks"]:
            assert {
                "taskId", "kind", "sessionId", "createdAt", "createdBy",
                "region", "desiredViewpoint", "evidenceType", "priority",
                "reason", "reasonCodes", "reasons", "guidance",
                "expectedCoverageContribution", "targetFrameIds", "status",
            } <= set(task)

    def test_rejected_frame_becomes_high_priority_reframe(self, tmp_path):
        doc = _bundle_doc([
            _frame_entry((3, 0, 0), verdict="REJECTED", reasons=("sharpness_low",)),
        ])
        bundle = load_mobile_bundle(_write_bundle(tmp_path, doc))
        payload = derive_capture_tasks(bundle)
        reframes = [t for t in payload["tasks"] if t["kind"] == "reframe"]
        assert len(reframes) == 1
        task = reframes[0]
        assert task["priority"] == "high"
        assert task["targetFrameIds"] == [bundle.frames[0].frame_id]
        assert task["reasons"] == ["sharpness_low"]
        assert "mobile_triage:sharpness_low" in task["reasonCodes"]
        assert "sharpness_low" in task["reason"]
        assert task["sessionId"] == bundle.session_id
        assert task["createdBy"] == "reality-engine"
        assert task["desiredViewpoint"]["sourceFrameId"] == bundle.frames[0].frame_id
        # no pose was ever measured on this device, so none is claimed
        assert task["desiredViewpoint"]["poseAvailable"] is False

    def test_no_telemetry_is_unavailable_and_never_guessed(self, tmp_path):
        doc = _bundle_doc([_frame_entry((5, 0, 0)), _frame_entry((0, 6, 0))])
        bundle = load_mobile_bundle(_write_bundle(tmp_path, doc))
        payload = derive_capture_tasks(bundle)
        assert payload["coverageAnalysis"]["state"] == "UNAVAILABLE"
        kinds = [t["kind"] for t in payload["tasks"]]
        assert "coverage_gap" not in kinds
        assert "telemetry" in kinds
        telemetry_task = next(t for t in payload["tasks"] if t["kind"] == "telemetry")
        assert telemetry_task["desiredViewpoint"] is None
        assert telemetry_task["reasonCodes"] == [
            "telemetry_unavailable:heading", "telemetry_unavailable:geolocation",
        ]
        # nothing was measured, so no coordinate/heading is asserted anywhere
        assert payload["gpsBounds"] is None
        assert payload["coverageAnalysis"]["coveredOctants"] == []

    def test_task_ids_are_deterministic_and_done_tasks_superseded(self, tmp_path):
        doc = _bundle_doc([_frame_entry((7, 0, 0), telemetry=_telemetry(heading=10.0))])
        bundle = load_mobile_bundle(_write_bundle(tmp_path, doc))
        ids_1 = [t["taskId"] for t in derive_capture_tasks(bundle)["tasks"]]
        ids_2 = [t["taskId"] for t in derive_capture_tasks(bundle)["tasks"]]
        assert ids_1 == ids_2

        bundle.task_outcomes = [{
            "taskId": ids_1[0], "status": "DONE", "frameIds": [], "completedAt": None,
        }]
        after = derive_capture_tasks(bundle)
        assert ids_1[0] not in [t["taskId"] for t in after["tasks"]]

    def test_write_capture_tasks_is_deterministic_json(self, tmp_path):
        doc = _bundle_doc([_frame_entry((8, 0, 0), telemetry=_telemetry(heading=45.0))])
        bundle = load_mobile_bundle(_write_bundle(tmp_path, doc))
        payload = derive_capture_tasks(bundle)
        p1 = tmp_path / "tasks1.json"
        p2 = tmp_path / "tasks2.json"
        write_capture_tasks(payload, str(p1))
        write_capture_tasks(payload, str(p2))
        assert p1.read_text(encoding="utf-8") == p2.read_text(encoding="utf-8")
        assert json.loads(p1.read_text(encoding="utf-8"))["schema"] == "re.mobile-capture-task/v1"



class TestLargeBundle:
    def test_200_frame_bundle_roundtrip_without_loss(self, tmp_path):
        frames = []
        for i in range(200):
            color = (i % 251 + 1, (i * 7) % 251 + 1, (i * 13) % 251 + 1)
            frames.append(_frame_entry(color, telemetry=_telemetry(heading=(i * 1.8) % 360)))
        doc = _bundle_doc(frames, session_id="sess-large", bundle_id="bundle:large0000000002")
        path = _write_bundle(tmp_path, doc, name="large_bundle.json")

        start = time.perf_counter()
        bundle = load_mobile_bundle(path)
        parse_s = time.perf_counter() - start
        assert len(bundle.frames) == 200
        assert bundle.integrity_failures == []

        session = _new_session("sess-large")
        start = time.perf_counter()
        report = ingest_mobile_bundle_to_dir(bundle, session, tmp_path / "session")
        ingest_s = time.perf_counter() - start
        assert report["files_written"] == 200
        assert len(session.package.all_assets()) == 200
        # every frame's exact content is on disk, keyed by its sha256
        rgb = tmp_path / "session" / "mobile" / "rgb"
        on_disk = {p.stem for p in rgb.iterdir()}
        assert on_disk == {f.content_sha256 for f in bundle.frames}
        # keep the suite honest about scale: not silently slow
        assert parse_s < 60
        assert ingest_s < 120

        payload = derive_capture_tasks(bundle)
        assert payload["coverageAnalysis"]["headingFrames"] == 200
        assert payload["coverageAnalysis"]["usefulFrames"] == 200
        assert len(payload["coverageAnalysis"]["coveredOctants"]) == 8
        assert [t["kind"] for t in payload["tasks"]] == []  # fully covered, no gaps


def _derive(tmp_path, frames, *, bundle_id, exported_at, session_id="sess-mobile-1"):
    """A real derived task payload for a bundle built from ``frames``."""
    doc = _bundle_doc(
        frames, session_id=session_id, bundle_id=bundle_id, exported_at=exported_at
    )
    bundle = load_mobile_bundle(_write_bundle(tmp_path, doc, name=f"{bundle_id[-6:]}.json"))
    return derive_capture_tasks(bundle)


def _octant_task(payload, octant):
    return next(
        t
        for t in payload["tasks"]
        if t["kind"] == "coverage_gap" and t["region"].get("octant") == octant
    )


def _north_east_bundle(tmp_path, *, bundle_id, exported_at):
    """Coverage in N and E, telemetry present -> real gaps for the other six."""
    frames = [
        _frame_entry((10, 0, 0), telemetry=_telemetry(heading=10.0, lat=1.0)),
        _frame_entry((0, 12, 0), telemetry=_telemetry(heading=90.0, lat=1.0)),
    ]
    return _derive(tmp_path, frames, bundle_id=bundle_id, exported_at=exported_at)


def _north_only_bundle(tmp_path, *, bundle_id, exported_at):
    """Coverage in N only -- the older, narrower measurement."""
    frames = [_frame_entry((20, 0, 0), telemetry=_telemetry(heading=10.0, lat=1.0))]
    return _derive(tmp_path, frames, bundle_id=bundle_id, exported_at=exported_at)


class TestSnapshotPrecedence:
    """A newer bundle analysis must never lose to stale persisted state."""

    OLD_AT = "2026-09-20T10:05:00Z"
    NEW_AT = "2026-09-20T11:05:00Z"

    def test_parse_exported_at_reads_the_contract_format(self):
        assert parse_exported_at("2026-09-20T10:05:00Z") == pytest.approx(
            parse_exported_at("2026-09-20T10:05:00+00:00")
        )
        # naive strings are read as UTC, per the contract
        assert parse_exported_at("2026-09-20T10:05:00") == parse_exported_at(
            "2026-09-20T10:05:00Z"
        )
        # unknown age, never "old"
        assert parse_exported_at("") is None
        assert parse_exported_at("not-a-time") is None
        assert parse_exported_at(None) is None

    def test_comparison_selects_the_newer_bundle(self):
        newer = {
            "bundleId": "b2",
            "generatedAt": self.NEW_AT,
            "coverageAnalysis": {"state": "AVAILABLE"},
        }
        older = {
            "bundleId": "b1",
            "generatedAt": self.OLD_AT,
            "coverageAnalysis": {"state": "UNAVAILABLE"},
        }

        decision = compare_coverage_snapshots(newer, older)
        assert decision["selected"] == "incoming"
        assert decision["comparison"] == "newer"
        assert decision["incumbentBundleId"] == "b1"

        # the same pair, handed over the other way round: no regression
        decision = compare_coverage_snapshots(older, newer)
        assert decision["selected"] == "incumbent"
        assert decision["comparison"] == "older"

        decision = compare_coverage_snapshots(newer, dict(newer))
        assert decision["comparison"] == "same"
        assert decision["selected"] == "incoming"

    def test_unparseable_timestamps_are_reported_as_unknown(self):
        decision = compare_coverage_snapshots(
            {"bundleId": "b2", "generatedAt": "", "coverageAnalysis": {}},
            {"bundleId": "b1", "generatedAt": "garbage", "coverageAnalysis": {}},
        )
        assert decision["comparison"] == "unknown"
        # the basis is visible: it never silently claims "newer"
        assert decision["incomingExportedAt"] == ""

    def test_newer_bundle_replaces_a_stale_persisted_snapshot(self, tmp_path):
        old = _north_only_bundle(
            tmp_path, bundle_id="bundle:old00000000001", exported_at=self.OLD_AT
        )
        new = _north_east_bundle(
            tmp_path, bundle_id="bundle:new00000000002", exported_at=self.NEW_AT
        )
        assert old["coverageAnalysis"]["coveredOctants"] == ["N"]

        merged, report = merge_capture_tasks(new, old)
        assert report["snapshot"]["selected"] == "incoming"
        assert report["snapshot"]["comparison"] == "newer"
        assert report["staleBundle"] is False
        assert merged["coverageAnalysis"]["coveredOctants"] == ["E", "N"]
        assert merged["coverageAnalysis"]["usefulFrames"] == 2
        assert merged["bundleId"] == "bundle:new00000000002"
        assert merged["coverageSnapshot"]["selected"] == "incoming"

    def test_stale_bundle_never_rolls_back_the_persisted_snapshot(self, tmp_path):
        old = _north_only_bundle(
            tmp_path, bundle_id="bundle:old00000000001", exported_at=self.OLD_AT
        )
        new = _north_east_bundle(
            tmp_path, bundle_id="bundle:new00000000002", exported_at=self.NEW_AT
        )

        merged, report = merge_capture_tasks(old, new)
        assert report["staleBundle"] is True
        assert report["snapshot"]["comparison"] == "older"
        assert report["snapshot"]["selected"] == "incumbent"
        assert report["selectedBundleId"] == "bundle:new00000000002"
        # the newer measurement survived; the stale one did not overwrite it
        assert merged["coverageAnalysis"]["coveredOctants"] == ["E", "N"]
        assert merged["coverageAnalysis"]["usefulFrames"] == 2
        assert merged["coverageSnapshot"]["comparison"] == "older"


class TestTaskMerge:
    OLD_AT = "2026-09-20T10:05:00Z"
    NEW_AT = "2026-09-20T11:05:00Z"

    def test_task_set_is_a_union_with_ids_stable(self, tmp_path):
        old = _north_only_bundle(
            tmp_path, bundle_id="bundle:old00000000001", exported_at=self.OLD_AT
        )
        new = _north_east_bundle(
            tmp_path, bundle_id="bundle:new00000000002", exported_at=self.NEW_AT
        )
        old_ids = {t["taskId"] for t in old["tasks"]}
        new_ids = {t["taskId"] for t in new["tasks"]}

        merged, report = merge_capture_tasks(new, old)
        merged_ids = [t["taskId"] for t in merged["tasks"]]
        assert len(merged_ids) == len(set(merged_ids))  # no duplicates
        assert old_ids | new_ids == set(merged_ids)  # nothing retracted silently
        assert report["tasks_added"] == len(new_ids - old_ids)

        # idempotent: re-merging the same payload changes nothing
        again, report_again = merge_capture_tasks(merged, merged)
        assert [t["taskId"] for t in again["tasks"]] == merged_ids
        assert report_again["tasks_added"] == 0
        assert report_again["tasks_superseded"] == []

    def test_gap_serviced_is_superseded_not_deleted(self, tmp_path):
        old = _north_only_bundle(
            tmp_path, bundle_id="bundle:old00000000001", exported_at=self.OLD_AT
        )
        new = _north_east_bundle(
            tmp_path, bundle_id="bundle:new00000000002", exported_at=self.NEW_AT
        )
        east_task = _octant_task(old, "E")
        west_task = _octant_task(old, "W")

        merged, report = merge_capture_tasks(new, old)
        by_id = {t["taskId"]: t for t in merged["tasks"]}
        serviced = by_id[east_task["taskId"]]
        assert serviced["status"] == "SUPERSEDED"
        assert serviced["supersededBy"]["bundleId"] == "bundle:new00000000002"
        assert "E octant" in serviced["supersededBy"]["reason"]
        assert report["tasks_superseded"] == [east_task["taskId"]]
        # an octant that is still uncovered keeps asking for pixels
        assert by_id[west_task["taskId"]]["status"] == "OPEN"

    def test_telemetry_request_retracted_once_sensors_report(self, tmp_path):
        blind = _derive(
            tmp_path,
            [_frame_entry((30, 0, 0)), _frame_entry((0, 31, 0))],
            bundle_id="bundle:blind0000000001",
            exported_at=self.OLD_AT,
        )
        assert blind["coverageAnalysis"]["state"] == "UNAVAILABLE"
        telemetry_task = next(t for t in blind["tasks"] if t["kind"] == "telemetry")

        sighted = _north_east_bundle(
            tmp_path, bundle_id="bundle:new00000000002", exported_at=self.NEW_AT
        )
        merged, report = merge_capture_tasks(sighted, blind)
        by_id = {t["taskId"]: t for t in merged["tasks"]}
        assert by_id[telemetry_task["taskId"]]["status"] == "SUPERSEDED"
        assert report["tasks_superseded"] == [telemetry_task["taskId"]]
        assert merged["coverageAnalysis"]["state"] == "AVAILABLE"

    def test_first_merge_into_an_empty_file_records_the_snapshot(self, tmp_path):
        payload = _north_east_bundle(
            tmp_path, bundle_id="bundle:new00000000002", exported_at=self.NEW_AT
        )
        merged, report = merge_capture_tasks(payload, {})
        assert report["snapshot"]["selected"] == "incoming"
        assert report["tasks_retained"] == 0
        assert report["staleBundle"] is False
        assert merged["coverageSnapshot"]["incomingBundleId"] == "bundle:new00000000002"

    def test_stale_bundle_preserves_incumbent_metadata_and_blocks_intermediate_regression(self, tmp_path):
        old = _north_only_bundle(
            tmp_path, bundle_id="bundle:old00000000001", exported_at="2026-09-20T10:05:00Z"
        )
        mid = _north_only_bundle(
            tmp_path, bundle_id="bundle:mid00000000002", exported_at="2026-09-20T10:30:00Z"
        )
        new = _north_east_bundle(
            tmp_path, bundle_id="bundle:new00000000003", exported_at="2026-09-20T11:05:00Z"
        )
        # 1. Merge old into new: new should remain authoritative
        stale_merged, rep = merge_capture_tasks(old, new)
        assert rep["staleBundle"] is True
        assert stale_merged["bundleId"] == "bundle:new00000000003"
        assert stale_merged["generatedAt"] == "2026-09-20T11:05:00Z"

        # 2. Merging an intermediate bundle afterwards must still be recognized as older than the 11:05 snapshot
        mid_merged, rep_mid = merge_capture_tasks(mid, stale_merged)
        assert rep_mid["staleBundle"] is True
        assert rep_mid["snapshot"]["comparison"] == "older"
        assert mid_merged["bundleId"] == "bundle:new00000000003"
        assert mid_merged["coverageAnalysis"]["coveredOctants"] == ["E", "N"]

    def test_gps_bounds_retention_and_expansion(self, tmp_path):
        # Bundle with GPS
        outdoor = _derive(
            tmp_path,
            [_frame_entry((10, 0, 0), telemetry=_telemetry(heading=10.0, lat=37.77, lon=-122.41))],
            bundle_id="bundle:outdoor000001",
            exported_at="2026-09-20T10:00:00Z",
        )
        assert outdoor["gpsBounds"] is not None
        orig_bounds = outdoor["gpsBounds"]

        # Subsequent indoor bundle (no GPS frames)
        indoor = _derive(
            tmp_path,
            [_frame_entry((20, 0, 0), telemetry={"heading": 15.0})],
            bundle_id="bundle:indoor0000002",
            exported_at="2026-09-20T11:00:00Z",
        )
        assert indoor["gpsBounds"] is None

        # Merging indoor into outdoor keeps outdoor GPS bounds rather than clobbering to None
        merged, _ = merge_capture_tasks(indoor, outdoor)
        assert merged["gpsBounds"] == orig_bounds

        # Another outdoor bundle with further coordinates expands the bounding box
        outdoor2 = _derive(
            tmp_path,
            [_frame_entry((30, 0, 0), telemetry=_telemetry(heading=20.0, lat=37.79, lon=-122.39))],
            bundle_id="bundle:outdoor000003",
            exported_at="2026-09-20T12:00:00Z",
        )
        merged2, _ = merge_capture_tasks(outdoor2, merged)
        assert merged2["gpsBounds"]["minLat"] == pytest.approx(37.77)
        assert merged2["gpsBounds"]["maxLat"] == pytest.approx(37.79)
        assert merged2["gpsBounds"]["minLon"] == pytest.approx(-122.41)
        assert merged2["gpsBounds"]["maxLon"] == pytest.approx(-122.39)

    def test_empty_tasks_with_coverage_preserves_incumbent_snapshot(self, tmp_path):
        incumbent = {
            "schema": "re.mobile-capture-task/v1",
            "bundleId": "bundle:incumbent001",
            "sessionId": "sess-1",
            "generatedAt": "2026-09-20T12:00:00Z",
            "coverageAnalysis": {"state": "AVAILABLE", "coveredOctants": ["N", "E"]},
            "tasks": [],
        }
        stale_incoming = _north_only_bundle(
            tmp_path, bundle_id="bundle:old00000000001", exported_at="2026-09-20T10:00:00Z"
        )
        merged, rep = merge_capture_tasks(stale_incoming, incumbent)
        assert rep["staleBundle"] is True
        assert merged["bundleId"] == "bundle:incumbent001"
        assert merged["coverageAnalysis"]["coveredOctants"] == ["N", "E"]


class TestCliVerbs:
    def _setup(self, tmp_path, capsys):
        session_dir = tmp_path / "session"
        frames = [
            _frame_entry((255, 0, 0), telemetry=_telemetry(heading=10.0, lat=1.0)),
            _frame_entry((0, 255, 0), verdict="REJECTED", reasons=("sharpness_low",)),
        ]
        bundle_path = _write_bundle(tmp_path, _bundle_doc(frames), name="cli_bundle.json")
        assert main(["session", "create", "sess-mobile-1", "-o", str(session_dir)]) == 0
        capsys.readouterr()
        return session_dir, bundle_path

    def test_ingest_mobile_verb(self, tmp_path, capsys):
        session_dir, bundle_path = self._setup(tmp_path, capsys)
        rc = main(["session", "ingest-mobile", bundle_path, str(session_dir)])
        out = capsys.readouterr().out
        assert rc == 0
        assert '"frames_verified": 2' in out
        rc_list = main(["session", "list", str(session_dir)])
        assert rc_list == 0
        assert "rgb" in capsys.readouterr().out

    def test_mobile_tasks_verb(self, tmp_path, capsys):
        session_dir, bundle_path = self._setup(tmp_path, capsys)
        tasks_path = tmp_path / "tasks.json"
        assert main(["session", "mobile-tasks", bundle_path, "-o", str(tasks_path)]) == 0
        capsys.readouterr()
        data = json.loads(tasks_path.read_text(encoding="utf-8"))
        assert data["schema"] == "re.mobile-capture-task/v1"
        assert data["coverageAnalysis"]["state"] == "AVAILABLE"
        kinds = {t["kind"] for t in data["tasks"]}
        assert "reframe" in kinds and "coverage_gap" in kinds

    def test_ingest_mobile_rejects_tampered_bundle(self, tmp_path, capsys):
        session_dir, _ = self._setup(tmp_path, capsys)
        entry = _frame_entry((11, 22, 33))
        entry["payloadBase64"] = base64.b64encode(_jpeg_bytes((44, 55, 66))).decode()
        bundle_path = _write_bundle(tmp_path, _bundle_doc([entry]), name="tampered.json")
        rc = main(["session", "ingest-mobile", bundle_path, str(session_dir)])
        err = capsys.readouterr().err
        assert rc == 1
        assert "integrity failure" in err.lower()

    def test_newer_bundle_analysis_survives_a_stale_rerun(self, tmp_path, capsys):
        """Regression: the persisted snapshot used to clobber the fresh one."""
        old_bundle = _write_bundle(
            tmp_path,
            _bundle_doc(
                [_frame_entry((60, 0, 0), telemetry=_telemetry(heading=10.0, lat=1.0))],
                bundle_id="bundle:old00000000001",
                exported_at="2026-09-20T10:05:00Z",
            ),
            name="old_bundle.json",
        )
        new_bundle = _write_bundle(
            tmp_path,
            _bundle_doc(
                [
                    _frame_entry((61, 0, 0), telemetry=_telemetry(heading=10.0, lat=1.0)),
                    _frame_entry((0, 62, 0), telemetry=_telemetry(heading=90.0, lat=1.0)),
                ],
                bundle_id="bundle:new00000000002",
                exported_at="2026-09-20T11:05:00Z",
            ),
            name="new_bundle.json",
        )
        tasks_path = tmp_path / "tasks.json"

        assert main(["session", "mobile-tasks", old_bundle, "-o", str(tasks_path)]) == 0
        capsys.readouterr()
        first = json.loads(tasks_path.read_text(encoding="utf-8"))
        assert first["coverageAnalysis"]["coveredOctants"] == ["N"]
        east_task_id = next(
            t["taskId"]
            for t in first["tasks"]
            if t["kind"] == "coverage_gap" and t["region"]["octant"] == "E"
        )

        # the phone goes back out and covers the east side
        assert main(["session", "mobile-tasks", new_bundle, "-o", str(tasks_path)]) == 0
        out = capsys.readouterr().out
        assert "coverage snapshot: incoming (newer;" in out
        second = json.loads(tasks_path.read_text(encoding="utf-8"))
        assert second["coverageAnalysis"]["coveredOctants"] == ["E", "N"]
        assert second["coverageAnalysis"]["usefulFrames"] == 2
        by_id = {t["taskId"]: t for t in second["tasks"]}
        assert by_id[east_task_id]["status"] == "SUPERSEDED"

        # an older export replayed later must not undo that measurement
        assert main(["session", "mobile-tasks", old_bundle, "-o", str(tasks_path)]) == 0
        captured = capsys.readouterr()
        assert "coverage snapshot: incumbent (older;" in captured.out
        assert "stale bundle" in captured.err.lower()
        third = json.loads(tasks_path.read_text(encoding="utf-8"))
        assert third["coverageAnalysis"]["coveredOctants"] == ["E", "N"]
        assert third["coverageAnalysis"]["usefulFrames"] == 2
        assert third["coverageSnapshot"]["comparison"] == "older"
        # no evidence-bearing task was dropped along the way
        assert {t["taskId"] for t in third["tasks"]} >= {
            t["taskId"] for t in second["tasks"]
        }

    def test_duplicate_bundle_is_idempotent(self, tmp_path, capsys):
        bundle_path = _write_bundle(
            tmp_path,
            _bundle_doc([_frame_entry((70, 0, 0), telemetry=_telemetry(heading=10.0, lat=1.0))]),
            name="dup_bundle.json",
        )
        tasks_path = tmp_path / "tasks.json"
        assert main(["session", "mobile-tasks", bundle_path, "-o", str(tasks_path)]) == 0
        capsys.readouterr()
        first = json.loads(tasks_path.read_text(encoding="utf-8"))
        assert main(["session", "mobile-tasks", bundle_path, "-o", str(tasks_path)]) == 0
        out = capsys.readouterr().out
        second = json.loads(tasks_path.read_text(encoding="utf-8"))
        assert "coverage snapshot: incoming (same;" in out
        assert [t["taskId"] for t in second["tasks"]] == [t["taskId"] for t in first["tasks"]]
        assert second["coverageAnalysis"] == first["coverageAnalysis"]

    def test_mobile_tasks_rejects_malformed_existing_file(self, tmp_path, capsys):
        bundle_path = _write_bundle(tmp_path, _bundle_doc([_frame_entry((80, 0, 0))]))
        tasks_path = tmp_path / "tasks.json"
        tasks_path.write_text("{ not json", encoding="utf-8")
        rc = main(["session", "mobile-tasks", bundle_path, "-o", str(tasks_path)])
        assert rc == 1
        assert "not valid JSON" in capsys.readouterr().err

    def test_overwrite_flag_replaces_the_task_file(self, tmp_path, capsys):
        old_bundle = _write_bundle(
            tmp_path,
            _bundle_doc(
                [_frame_entry((90, 0, 0), telemetry=_telemetry(heading=10.0, lat=1.0))],
                bundle_id="bundle:old00000000001",
                exported_at="2026-09-20T10:05:00Z",
            ),
            name="old_bundle.json",
        )
        new_bundle = _write_bundle(
            tmp_path,
            _bundle_doc(
                [
                    _frame_entry((91, 0, 0), telemetry=_telemetry(heading=10.0, lat=1.0)),
                    _frame_entry((0, 92, 0), telemetry=_telemetry(heading=90.0, lat=1.0)),
                ],
                bundle_id="bundle:new00000000002",
                exported_at="2026-09-20T11:05:00Z",
            ),
            name="new_bundle.json",
        )
        tasks_path = tmp_path / "tasks.json"
        assert main(["session", "mobile-tasks", old_bundle, "-o", str(tasks_path)]) == 0
        capsys.readouterr()
        assert (
            main(["session", "mobile-tasks", old_bundle, "-o", str(tasks_path), "--overwrite"]) == 0
        )
        capsys.readouterr()
        replaced = json.loads(tasks_path.read_text(encoding="utf-8"))
        assert replaced["coverageAnalysis"]["coveredOctants"] == ["N"]
        assert replaced["bundleId"] == "bundle:old00000000001"
        # --overwrite is an explicit operator choice: the newer bundle replaces it
        assert (
            main(["session", "mobile-tasks", new_bundle, "-o", str(tasks_path), "--overwrite"]) == 0
        )
        capsys.readouterr()
        final = json.loads(tasks_path.read_text(encoding="utf-8"))
        assert final["coverageAnalysis"]["coveredOctants"] == ["E", "N"]


