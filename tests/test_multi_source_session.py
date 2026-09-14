"""Tests for evidence/multi_source.py (unified source ingestion /
multi-source session milestone).

Fixtures are minimal real-magic-byte payloads (same convention as
tests/test_evidence_packages.py) so ids and hashes are hand-derivable.
"""

from __future__ import annotations

import pytest

from evidence.multi_source import MultiSourceSession, SourceStatus, SourceType


def _jpeg(seed_byte: int = 0x01) -> bytes:
    return b"\xff\xd8\xff\xe0" + bytes([seed_byte]) * 64


def _png(seed_byte: int = 0x02) -> bytes:
    return b"\x89PNG\r\n\x1a\n" + bytes([seed_byte]) * 64


def _las() -> bytes:
    return b"LASF" + b"\x00" * 64


class TestSingleSource:
    def test_add_image_file_ingests_one_asset(self, tmp_path):
        photo = tmp_path / "a.jpg"
        photo.write_bytes(_jpeg())
        session = MultiSourceSession(session_id="sess1")

        record = session.add_source(str(photo))

        assert record.status is SourceStatus.INGESTED
        assert record.source_type is SourceType.IMAGE
        assert len(record.asset_ids) == 1
        assert len(session.package.all_assets()) == 1

    def test_default_evidence_source_id_matches_the_source_record_id(self, tmp_path):
        """Identity: an EvidenceAsset produced by add_source() must embed
        an EvidenceSource whose source_id equals the owning
        SourceRecord.source_id -- one canonical id, joinable in both
        directions, not a separate basename-derived id ("disk:<name>")
        that silently diverges from the record's own src-NNNN-<hash>
        identity for the same physical source."""
        photo = tmp_path / "a.jpg"
        photo.write_bytes(_jpeg())
        session = MultiSourceSession(session_id="sess1")

        record = session.add_source(str(photo))

        assert record.asset_ids
        assets_by_id = {a.id: a for a in session.package.all_assets()}
        for asset_id in record.asset_ids:
            asset = assets_by_id[asset_id]
            assert asset.source.source_id == record.source_id

    def test_add_unknown_extension_is_unsupported_not_raised(self, tmp_path):
        junk = tmp_path / "notes.txt"
        junk.write_text("hello")
        session = MultiSourceSession(session_id="sess1")

        record = session.add_source(str(junk))

        assert record.status is SourceStatus.UNSUPPORTED
        assert record.error is not None
        assert len(session.package.all_assets()) == 0

    def test_missing_path_raises(self, tmp_path):
        session = MultiSourceSession(session_id="sess1")
        with pytest.raises(FileNotFoundError):
            session.add_source(str(tmp_path / "does-not-exist.jpg"))


class TestSourceLevelDedup:
    def test_adding_the_same_file_twice_is_a_noop(self, tmp_path):
        photo = tmp_path / "a.jpg"
        photo.write_bytes(_jpeg())
        session = MultiSourceSession(session_id="sess1")

        first = session.add_source(str(photo))
        second = session.add_source(str(photo))

        assert second is first
        assert second.status is SourceStatus.INGESTED
        assert len(session.sources()) == 1
        assert len(session.package.all_assets()) == 1

    def test_copy_of_identical_folder_dedupes_by_content(self, tmp_path):
        folder_a = tmp_path / "a"
        folder_a.mkdir()
        (folder_a / "p.jpg").write_bytes(_jpeg())
        folder_b = tmp_path / "b"
        folder_b.mkdir()
        (folder_b / "p.jpg").write_bytes(_jpeg())

        session = MultiSourceSession(session_id="sess1")
        first = session.add_source(str(folder_a))
        second = session.add_source(str(folder_b))

        assert second is first
        assert len(session.sources()) == 1


class TestIncrementalMultiSource:
    def test_second_source_does_not_change_first_sources_asset_ids(self, tmp_path):
        photo_a = tmp_path / "a.jpg"
        photo_a.write_bytes(_jpeg(0x01))
        photo_b = tmp_path / "b.jpg"
        photo_b.write_bytes(_jpeg(0x03))

        session = MultiSourceSession(session_id="sess1")
        first = session.add_source(str(photo_a))
        first_asset_ids = list(first.asset_ids)

        second = session.add_source(str(photo_b))

        assert first.asset_ids == first_asset_ids  # unchanged (never mutated)
        assert session.sources()[0].asset_ids == first_asset_ids  # still in the package
        assert second.status is SourceStatus.INGESTED
        assert len(session.package.all_assets()) == 2
        assert set(first.asset_ids).issubset({a.id for a in session.package.all_assets()})

    def test_mixed_source_types_accumulate_in_one_session(self, tmp_path):
        photo = tmp_path / "a.jpg"
        photo.write_bytes(_jpeg())
        las = tmp_path / "scan.las"
        las.write_bytes(_las())

        session = MultiSourceSession(session_id="sess1")
        session.add_source(str(photo))
        session.add_source(str(las))

        assert len(session.sources()) == 2
        assert {s.source_type for s in session.sources()} == {SourceType.IMAGE, SourceType.POINT_CLOUD}
        assert len(session.package.all_assets()) == 2

    def test_folder_source_records_source_type_dataset(self, tmp_path):
        folder = tmp_path / "batch"
        folder.mkdir()
        (folder / "a.jpg").write_bytes(_jpeg(0x01))
        (folder / "b.png").write_bytes(_png(0x02))

        session = MultiSourceSession(session_id="sess1")
        record = session.add_source(str(folder))

        assert record.source_type is SourceType.DATASET
        assert record.status is SourceStatus.INGESTED
        assert len(record.asset_ids) == 2


class TestCompositeDetection:
    def test_plain_photo_folder_has_no_composite_components(self, tmp_path):
        from evidence.multi_source import _detect_composite_components, _is_composite_capture

        folder = tmp_path / "photos"
        folder.mkdir()
        (folder / "a.jpg").write_bytes(_jpeg())

        components = _detect_composite_components(str(folder))

        assert components == {}
        assert _is_composite_capture(components) is False

    def test_rgb_plus_gps_subfolders_detected_as_composite(self, tmp_path):
        from evidence.multi_source import CaptureComponent, _detect_composite_components, _is_composite_capture

        folder = tmp_path / "phone_capture_001"
        (folder / "rgb").mkdir(parents=True)
        (folder / "rgb" / "0001.jpg").write_bytes(_jpeg(0x01))
        (folder / "gps").mkdir()
        (folder / "gps" / "track.csv").write_text("lat,lon\n1.0,2.0\n")

        components = _detect_composite_components(str(folder))

        assert components[CaptureComponent.RGB] == ["rgb/0001.jpg"]
        assert components[CaptureComponent.GPS] == ["gps/track.csv"]
        assert _is_composite_capture(components) is True

    def test_rgb_only_subfolder_is_not_composite(self, tmp_path):
        # A folder with only a visual component (no sidecar) is not a
        # synchronized composite acquisition -- just an organized photo
        # folder. Must stay DATASET, not falsely promoted.
        from evidence.multi_source import _detect_composite_components, _is_composite_capture

        folder = tmp_path / "rgb_only"
        (folder / "rgb").mkdir(parents=True)
        (folder / "rgb" / "a.jpg").write_bytes(_jpeg())

        components = _detect_composite_components(str(folder))

        assert _is_composite_capture(components) is False

    def test_empty_component_subfolder_is_ignored(self, tmp_path):
        from evidence.multi_source import CaptureComponent, _detect_composite_components

        folder = tmp_path / "capture"
        (folder / "rgb").mkdir(parents=True)
        (folder / "rgb" / "a.jpg").write_bytes(_jpeg())
        (folder / "imu").mkdir()  # empty -- no files

        components = _detect_composite_components(str(folder))

        assert CaptureComponent.IMU not in components
        assert CaptureComponent.RGB in components

    def test_source_type_of_plain_folder_is_dataset(self, tmp_path):
        from evidence.multi_source import _source_type_of, SourceType

        folder = tmp_path / "photos"
        folder.mkdir()
        (folder / "a.jpg").write_bytes(_jpeg())

        assert _source_type_of(str(folder)) is SourceType.DATASET

    def test_source_type_of_composite_without_capture_type_is_generic(self, tmp_path):
        from evidence.multi_source import _source_type_of, SourceType

        folder = tmp_path / "capture"
        (folder / "rgb").mkdir(parents=True)
        (folder / "rgb" / "a.jpg").write_bytes(_jpeg())
        (folder / "imu").mkdir()
        (folder / "imu" / "log.csv").write_text("t,ax,ay,az\n")

        assert _source_type_of(str(folder)) is SourceType.COMPOSITE_CAPTURE

    def test_source_type_of_composite_with_explicit_phone_capture_type(self, tmp_path):
        from evidence.multi_source import _source_type_of, SourceType

        folder = tmp_path / "capture"
        (folder / "rgb").mkdir(parents=True)
        (folder / "rgb" / "a.jpg").write_bytes(_jpeg())
        (folder / "gps").mkdir()
        (folder / "gps" / "track.csv").write_text("lat,lon\n")

        assert _source_type_of(str(folder), capture_type="phone") is SourceType.PHONE_CAPTURE

    def test_source_type_of_composite_with_explicit_drone_capture_type(self, tmp_path):
        from evidence.multi_source import _source_type_of, SourceType

        folder = tmp_path / "capture"
        (folder / "video").mkdir(parents=True)
        (folder / "video" / "flight.mp4").write_bytes(b"not-a-real-video")
        (folder / "telemetry").mkdir()
        (folder / "telemetry" / "log.json").write_text("{}")

        assert _source_type_of(str(folder), capture_type="drone") is SourceType.DRONE_CAPTURE


class TestSourceRecordSerializationDefaults:
    def test_round_trips_components_and_registration(self):
        from evidence.multi_source import CaptureComponent, SourceRecord, SourceStatus, SourceType

        record = SourceRecord(
            source_id="src-0000-abc",
            original_path="/tmp/capture",
            source_type=SourceType.COMPOSITE_CAPTURE,
            content_hash="abc123",
            status=SourceStatus.INGESTED,
            components={CaptureComponent.GPS.value: ["gps/track.csv"]},
        )

        restored = SourceRecord.from_dict(record.to_dict())

        assert restored.components == {"gps": ["gps/track.csv"]}
        assert restored.registration == {"status": "unknown", "transform": None}

    def test_from_dict_defaults_missing_new_fields_for_old_format(self):
        # Simulates a session.json written before this plan existed --
        # no "components"/"registration" keys at all.
        from evidence.multi_source import SourceRecord

        old_format = {
            "source_id": "src-0000-abc",
            "original_path": "/tmp/a.jpg",
            "source_type": "image",
            "content_hash": "abc123",
            "status": "ingested",
            "asset_ids": ["ev-x-0000-abc"],
            "unhandled_paths": [],
            "error": None,
        }

        restored = SourceRecord.from_dict(old_format)

        assert restored.components == {}
        assert restored.registration == {"status": "unknown", "transform": None}


class TestEvidenceSummary:
    def test_not_ready_below_minimum_image_evidence(self, tmp_path):
        photo = tmp_path / "a.jpg"
        photo.write_bytes(_jpeg())
        session = MultiSourceSession(session_id="sess1")
        session.add_source(str(photo))

        summary = session.evidence_summary()

        assert summary["source_count"] == 1
        assert summary["asset_counts"] == {"photo": 1}
        assert summary["ready_for_reconstruction"] is False
        assert summary["readiness_issues"]

    def test_ready_once_minimum_image_evidence_reached(self, tmp_path):
        session = MultiSourceSession(session_id="sess1")
        for i in range(2):
            photo = tmp_path / f"p{i}.jpg"
            photo.write_bytes(_jpeg(seed_byte=i + 1))
            session.add_source(str(photo))

        summary = session.evidence_summary()

        assert summary["asset_counts"] == {"photo": 2}
        assert summary["ready_for_reconstruction"] is True
        assert summary["readiness_issues"] == []

    def test_gps_asset_count_reflects_real_exif_gps(self, tmp_path):
        # No GPS in these synthetic JPEGs (no real EXIF), so the count
        # must be honestly zero rather than fabricated.
        photo = tmp_path / "a.jpg"
        photo.write_bytes(_jpeg())
        session = MultiSourceSession(session_id="sess1")
        session.add_source(str(photo))

        summary = session.evidence_summary()

        assert summary["gps_asset_count"] == 0


class TestSerialization:
    def test_round_trips_sources_and_package(self, tmp_path):
        photo = tmp_path / "a.jpg"
        photo.write_bytes(_jpeg())
        session = MultiSourceSession(session_id="sess1", name="Building A")
        session.add_source(str(photo))

        restored = MultiSourceSession.from_dict(session.to_dict())

        assert restored.session_id == session.session_id
        assert restored.name == session.name
        assert [s.source_id for s in restored.sources()] == [s.source_id for s in session.sources()]
        assert restored.package.to_dict() == session.package.to_dict()

    def test_restored_session_still_dedupes_by_content(self, tmp_path):
        photo = tmp_path / "a.jpg"
        photo.write_bytes(_jpeg())
        session = MultiSourceSession(session_id="sess1")
        original = session.add_source(str(photo))

        restored = MultiSourceSession.from_dict(session.to_dict())
        again = restored.add_source(str(photo))

        assert again.source_id == original.source_id
        assert len(restored.sources()) == 1

    def test_restored_session_accepts_a_new_source_incrementally(self, tmp_path):
        photo_a = tmp_path / "a.jpg"
        photo_a.write_bytes(_jpeg(0x01))
        photo_b = tmp_path / "b.jpg"
        photo_b.write_bytes(_jpeg(0x05))

        session = MultiSourceSession(session_id="sess1")
        session.add_source(str(photo_a))
        restored = MultiSourceSession.from_dict(session.to_dict())

        restored.add_source(str(photo_b))

        assert len(restored.sources()) == 2
        assert len(restored.package.all_assets()) == 2

    def test_unsupported_format_version_raises(self):
        with pytest.raises(ValueError):
            MultiSourceSession.from_dict({"format_version": 99})
