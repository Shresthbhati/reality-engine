"""Tests for the evidence/Session/Dataset foundation (spec §3-6, Exit Goals A & B)."""

import pytest

from evidence import (
    Session, EvidenceItem, EvidenceKind, SessionStatus,
    Dataset, DuplicateEvidenceError, UnknownEvidenceError, SessionClosedError,
    UnknownSessionError, DuplicateSessionError,
)
from world_ir.coordinates import Frame
from provenance import Provenance


class TestEvidenceItem:
    def test_default_provenance_is_observed(self):
        item = EvidenceItem(id="e1", kind=EvidenceKind.PHOTO, source_uri="file://a.jpg")
        assert item.provenance == Provenance.OBSERVED

    def test_no_wall_clock_default(self):
        item = EvidenceItem(id="e1", kind=EvidenceKind.PHOTO, source_uri="file://a.jpg")
        assert item.captured_at is None  # not datetime.now() -- same lesson as the WorldIR timestamp fix

    def test_roundtrip(self):
        item = EvidenceItem(id="e1", kind=EvidenceKind.LIDAR, source_uri="s3://bucket/scan.las",
                             captured_at=12.5, sha256="abc123", metadata={"points": 40000})
        restored = EvidenceItem.from_dict(item.to_dict())
        assert restored == item


class TestSession:
    def test_add_and_get_evidence(self):
        s = Session("sess1", name="Front yard walk", created_at=100.0)
        item = EvidenceItem(id="e1", kind=EvidenceKind.PHOTO, source_uri="file://a.jpg")
        s.add_evidence(item)
        assert s.get_evidence("e1") is item
        assert s.status == SessionStatus.OPEN

    def test_duplicate_evidence_rejected(self):
        s = Session("sess1")
        item = EvidenceItem(id="e1", kind=EvidenceKind.PHOTO, source_uri="file://a.jpg")
        s.add_evidence(item)
        with pytest.raises(DuplicateEvidenceError):
            s.add_evidence(item)

    def test_unknown_evidence_raises(self):
        s = Session("sess1")
        with pytest.raises(UnknownEvidenceError):
            s.get_evidence("nope")

    def test_evidence_order_is_deterministic_capture_order(self):
        s = Session("sess1")
        for i in [3, 1, 2]:
            s.add_evidence(EvidenceItem(id=f"e{i}", kind=EvidenceKind.PHOTO, source_uri=f"file://{i}.jpg"))
        assert [e.id for e in s.all_evidence()] == ["e3", "e1", "e2"]

    def test_evidence_by_kind_filters(self):
        s = Session("sess1")
        s.add_evidence(EvidenceItem(id="e1", kind=EvidenceKind.PHOTO, source_uri="file://a.jpg"))
        s.add_evidence(EvidenceItem(id="e2", kind=EvidenceKind.LIDAR, source_uri="file://a.las"))
        assert [e.id for e in s.evidence_by_kind(EvidenceKind.LIDAR)] == ["e2"]

    def test_archive_prevents_further_evidence(self):
        s = Session("sess1")
        s.archive()
        assert s.status == SessionStatus.ARCHIVED
        with pytest.raises(SessionClosedError):
            s.add_evidence(EvidenceItem(id="e1", kind=EvidenceKind.PHOTO, source_uri="file://a.jpg"))

    def test_processing_moves_status_to_processed(self):
        s = Session("sess1")
        s.record_processing("feature_extraction")
        assert s.status == SessionStatus.PROCESSED

    def test_serialize_deserialize_roundtrip_preserves_evidence_and_history(self):
        s = Session("sess1", name="Backyard", coordinate_frame=Frame.SESSION_LOCAL, created_at=42.0)
        s.add_evidence(EvidenceItem(id="e1", kind=EvidenceKind.PHOTO, source_uri="file://a.jpg", captured_at=1.0))
        s.add_evidence(EvidenceItem(id="e2", kind=EvidenceKind.GPS_TRACK, source_uri="file://track.gpx"))
        s.record_processing("feature_extraction", at_tick=5)

        restored = Session.from_dict(s.to_dict())
        assert restored.id == s.id
        assert restored.name == s.name
        assert restored.created_at == 42.0
        assert restored.status == SessionStatus.PROCESSED
        assert [e.id for e in restored.all_evidence()] == ["e1", "e2"]
        assert restored.get_evidence("e1") == s.get_evidence("e1")
        assert len(restored.processing_history) == 1
        assert restored.processing_history[0].operation == "feature_extraction"

    def test_deserialize_rejects_unsupported_version(self):
        s = Session("sess1")
        with pytest.raises(ValueError):
            s.deserialize({"format_version": 2})


class TestDataset:
    def test_add_and_list_sessions(self):
        ds = Dataset("ds1", name="Riverside survey")
        ds.add_session(Session("sess1"))
        ds.add_session(Session("sess2"))
        assert [s.id for s in ds.list_sessions()] == ["sess1", "sess2"]

    def test_duplicate_session_rejected(self):
        ds = Dataset("ds1")
        ds.add_session(Session("sess1"))
        with pytest.raises(DuplicateSessionError):
            ds.add_session(Session("sess1"))

    def test_unknown_session_raises(self):
        ds = Dataset("ds1")
        with pytest.raises(UnknownSessionError):
            ds.get_session("nope")

    def test_dataset_roundtrip_preserves_sessions_and_evidence(self):
        ds = Dataset("ds1", name="Survey")
        s1 = Session("sess1", created_at=1.0)
        s1.add_evidence(EvidenceItem(id="e1", kind=EvidenceKind.PHOTO, source_uri="file://a.jpg"))
        ds.add_session(s1)

        restored = Dataset.from_dict(ds.to_dict())
        assert restored.get_session("sess1").get_evidence("e1").source_uri == "file://a.jpg"


class TestMergeSessions:
    def test_merge_requires_at_least_two_sessions(self):
        ds = Dataset("ds1")
        ds.add_session(Session("sess1"))
        with pytest.raises(ValueError):
            ds.merge("m1", ["sess1"])

    def test_merge_unknown_session_raises(self):
        ds = Dataset("ds1")
        ds.add_session(Session("sess1"))
        with pytest.raises(UnknownSessionError):
            ds.merge("m1", ["sess1", "nope"])

    def test_merge_preserves_source_session_provenance(self):
        """The core invariant of §5: merging never destroys or mutates the
        contributing sessions -- they remain independently queryable."""
        ds = Dataset("ds1")
        s1 = Session("sess1")
        s1.add_evidence(EvidenceItem(id="e1", kind=EvidenceKind.PHOTO, source_uri="file://a.jpg"))
        s2 = Session("sess2")
        s2.add_evidence(EvidenceItem(id="e2", kind=EvidenceKind.PHOTO, source_uri="file://b.jpg"))
        ds.add_session(s1)
        ds.add_session(s2)

        merged = ds.merge("m1", ["sess1", "sess2"])

        # sources untouched and still independently valid
        assert ds.get_session("sess1").get_evidence("e1").source_uri == "file://a.jpg"
        assert ds.get_session("sess2").get_evidence("e2").source_uri == "file://b.jpg"
        assert merged.source_session_ids == ["sess1", "sess2"]
        assert merged.evidence_count == 2

    def test_merge_flags_coordinate_frame_conflict(self):
        ds = Dataset("ds1")
        s1 = Session("sess1", coordinate_frame=Frame.SESSION_LOCAL)
        s2 = Session("sess2", coordinate_frame=Frame.WORLD)
        ds.add_session(s1)
        ds.add_session(s2)

        merged = ds.merge("m1", ["sess1", "sess2"])
        assert merged.has_conflicts()
        assert "coordinate_frame mismatch" in merged.conflicts[0]

    def test_merge_no_conflict_when_frames_agree(self):
        ds = Dataset("ds1")
        ds.add_session(Session("sess1", coordinate_frame=Frame.SESSION_LOCAL))
        ds.add_session(Session("sess2", coordinate_frame=Frame.SESSION_LOCAL))
        merged = ds.merge("m1", ["sess1", "sess2"])
        assert not merged.has_conflicts()

    def test_merge_computes_temporal_overlap_when_timestamps_present(self):
        ds = Dataset("ds1")
        ds.add_session(Session("sess1", created_at=10.0))
        ds.add_session(Session("sess2", created_at=25.0))
        merged = ds.merge("m1", ["sess1", "sess2"])
        assert merged.temporal_overlap == {"earliest": 10.0, "latest": 25.0}

    def test_merge_records_processing_on_source_sessions(self):
        ds = Dataset("ds1")
        ds.add_session(Session("sess1"))
        ds.add_session(Session("sess2"))
        ds.merge("m1", ["sess1", "sess2"])
        assert ds.get_session("sess1").processing_history[-1].operation == "merged"

    def test_registration_honestly_reported_as_not_performed(self):
        """No CV/registration backend exists yet -- this must say so, not
        fake a registration result (spec §31: do not fabricate results)."""
        ds = Dataset("ds1")
        ds.add_session(Session("sess1"))
        ds.add_session(Session("sess2"))
        merged = ds.merge("m1", ["sess1", "sess2"])
        assert merged.registration_status == "not_performed"

    def test_merge_roundtrips_through_dataset_serialization(self):
        ds = Dataset("ds1")
        ds.add_session(Session("sess1", coordinate_frame=Frame.WORLD))
        ds.add_session(Session("sess2", coordinate_frame=Frame.WORLD))
        ds.merge("m1", ["sess1", "sess2"])

        restored = Dataset.from_dict(ds.to_dict())
        assert "m1" in restored.merges
        assert restored.merges["m1"].source_session_ids == ["sess1", "sess2"]
