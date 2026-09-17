"""evidence/session_store.py: multi-source sessions -- persistence,
incremental ingestion, duplicate handling, sensor logs, point clouds,
composite captures, payload security.

Uses real decoders (PIL/cv2) exactly like test_media_preprocessing.py
does; where a test needs no decoder it stays runnable without one.
No clocks, no network, no model weights."""

import json

import pytest

from evidence.session_store import (
    AddSourceResult,
    DuplicateSessionError,
    IngestResult,
    SessionHandle,
    SessionWorkspace,
    SourcePayloadStore,
    UnknownSessionError,
    parse_pcd_header,
    parse_ply_header,
    parse_sensor_csv,
)
from evidence.sources import SourceStatus, SourceType
from evidence.session import EvidenceKind

Image = pytest.importorskip("PIL.Image")


def _exif(make="Pixel", model="8 Pro", when="2026:01:02 03:04:06"):
    exif = Image.Exif()
    exif[0x010F] = make
    exif[0x0110] = model
    sub = exif.get_ifd(0x8769)
    sub[0x9003] = when
    return exif


def _save_photo(path, exif=None, color=(120, 140, 160), size=(64, 48)):
    image = Image.new("RGB", size, color)
    kwargs = {"exif": exif} if exif is not None else {}
    image.save(path, format="JPEG", **kwargs)
    return str(path)


def _workspace(tmp_path):
    return SessionWorkspace(str(tmp_path / "workspace"))


class TestWorkspaceLifecycle:
    def test_create_then_open_round_trips(self, tmp_path):
        ws = _workspace(tmp_path)
        handle = ws.create_session("Building A")
        assert handle.session.id == "Building-A"
        assert handle.session.name == "Building A"
        reopened = ws.open_session("Building-A")
        assert reopened.session.name == "Building A"
        assert reopened.session.id == handle.session.id

    def test_duplicate_create_refused(self, tmp_path):
        ws = _workspace(tmp_path)
        ws.create_session("Building A")
        with pytest.raises(DuplicateSessionError):
            ws.create_session("Building A")

    def test_open_unknown_refused(self, tmp_path):
        with pytest.raises(UnknownSessionError):
            _workspace(tmp_path).open_session("nope")

    def test_invalid_session_id_refused(self, tmp_path):
        ws = _workspace(tmp_path)
        with pytest.raises(ValueError):
            ws.create_session("ok", session_id="../evil")
        with pytest.raises(ValueError):
            ws.open_session("..\\evil")

    def test_explicit_id_respected(self, tmp_path):
        ws = _workspace(tmp_path)
        handle = ws.create_session("Site 42", session_id="site-42")
        assert handle.session.id == "site-42"
        assert ws.open_session("site-42").session.name == "Site 42"

    def test_list_sessions_reports_counts(self, tmp_path):
        ws = _workspace(tmp_path)
        ws.create_session("Alpha")
        ws.create_session("Beta")
        entries = {e["id"]: e for e in ws.list_sessions()}
        assert set(entries) == {"Alpha", "Beta"}
        assert entries["Alpha"]["source_count"] == 0


class TestImageSourceIngestion:
    def test_add_ingest_single_photo(self, tmp_path):
        ws = _workspace(tmp_path)
        handle = ws.create_session("S")
        photo = _save_photo(tmp_path / "p.jpg", _exif(make="DJI", model="Mini 3"))
        added = handle.add_source(photo, platform="drone")
        assert added.created is True
        assert added.source_type is SourceType.IMAGE
        result = handle.ingest_source(added.source_id)
        assert result.ok
        assert result.status is SourceStatus.INGESTED
        assert len(result.asset_ids) == 1
        asset = handle.package.assets[result.asset_ids[0]]
        # Provenance: the asset's evidence source IS the canonical source id.
        assert asset.source.source_id == added.source_id
        assert asset.kind is EvidenceKind.PHOTO
        assert asset.acquired_at == 1767323046.0  # EXIF DateTimeOriginal, never a wall clock
        source = handle.sources.get(added.source_id)
        assert source.status is SourceStatus.INGESTED
        assert source.acquired_at == 1767323046.0
        assert source.device == "DJI Mini 3"  # synced back from EXIF
        # The asset is mirrored into the append-only session as evidence.
        assert len(handle.session.all_evidence()) == 1

    def test_readding_same_file_resolves_not_duplicates(self, tmp_path):
        ws = _workspace(tmp_path)
        handle = ws.create_session("S")
        photo = _save_photo(tmp_path / "p.jpg")
        first = handle.add_source(photo)
        again = handle.add_source(photo)
        assert again.created is False
        assert again.source_id == first.source_id
        assert "already" in again.note.lower()
        assert len(handle.sources) == 1

    def test_reingest_ingested_source_is_idempotent(self, tmp_path):
        ws = _workspace(tmp_path)
        handle = ws.create_session("S")
        photo = _save_photo(tmp_path / "p.jpg")
        source_id = handle.add_source(photo).source_id
        first = handle.ingest_source(source_id)
        second = handle.ingest_source(source_id)
        assert second.already_ingested is True
        assert second.asset_ids == first.asset_ids
        assert len(handle.package.all_assets()) == 1
        assert len(handle.session.all_evidence()) == 1

    def test_corrupt_photo_fails_explicitly_never_quietly(self, tmp_path):
        ws = _workspace(tmp_path)
        handle = ws.create_session("S")
        bad = tmp_path / "bad.jpg"
        bad.write_bytes(b"\x00\x01\x02" + b"junk" * 20)  # not a JPEG
        source_id = handle.add_source(str(bad)).source_id
        result = handle.ingest_source(source_id)
        assert result.ok is False
        assert result.status is SourceStatus.FAILED
        assert result.errors
        assert handle.sources.get(source_id).status is SourceStatus.FAILED

    def test_missing_path_refused_at_add(self, tmp_path):
        ws = _workspace(tmp_path)
        handle = ws.create_session("S")
        with pytest.raises(ValueError):
            handle.add_source(str(tmp_path / "does_not_exist.jpg"))


class TestIncrementalMultiSource:
    def test_second_source_appends_without_reingesting_first(self, tmp_path):
        ws = _workspace(tmp_path)
        handle = ws.create_session("S")
        p1 = _save_photo(tmp_path / "a.jpg", _exif(when="2026:01:02 03:04:06"))
        p2 = _save_photo(tmp_path / "b.jpg", _exif(when="2026:01:02 03:04:07"), color=(10, 20, 30))
        id1 = handle.add_source(p1).source_id
        id2 = handle.add_source(p2).source_id
        r1 = handle.ingest_source(id1)
        assets_after_first = list(handle.package.all_assets())
        r2 = handle.ingest_source(id2)
        assert id1 != id2
        # The first source's assets are untouched by the second ingest.
        assert [a.id for a in handle.package.all_assets()[: len(assets_after_first)]] == [
            a.id for a in assets_after_first
        ]
        assert len(handle.package.all_assets()) == 2
        assert len(handle.session.all_evidence()) == 2
        assert handle.sources.get(id1).asset_ids == tuple(r1.asset_ids)
        assert handle.sources.get(id2).asset_ids == tuple(r2.asset_ids)

    def test_same_bytes_in_two_files_is_one_source_two_paths(self, tmp_path):
        ws = _workspace(tmp_path)
        handle = ws.create_session("S")
        p1 = _save_photo(tmp_path / "a.jpg")
        p2 = _save_photo(tmp_path / "copy.jpg")
        # Make the bytes identical: same PIL save parameters, same image.
        import shutil

        shutil.copyfile(p1, p2)
        r1 = handle.add_source(p1)
        r2 = handle.add_source(p2)
        assert r1.source_id == r2.source_id
        assert r2.created is False

    def test_persistence_round_trip(self, tmp_path):
        ws = _workspace(tmp_path)
        handle = ws.create_session("S")
        p1 = _save_photo(tmp_path / "a.jpg", _exif(make="Pixel", model="8"))
        p2 = _save_photo(tmp_path / "b.jpg", color=(9, 9, 9))
        id1 = handle.add_source(p1).source_id
        id2 = handle.add_source(p2).source_id
        handle.ingest_source(id1)
        handle.ingest_source(id2)
        handle.sources.update(id1, status=SourceStatus.INGESTED)  # explicit state
        # Simulate a new process: fresh handle over the same workspace.
        reopened = ws.open_session("S")
        assert len(reopened.sources) == 2
        assert reopened.sources.get(id1).status is SourceStatus.INGESTED
        assert reopened.sources.get(id2).status is SourceStatus.INGESTED
        assert len(reopened.package.all_assets()) == 2
        assert len(reopened.session.all_evidence()) == 2
        assert reopened.package.package_id == handle.package.package_id
        # Deterministic serialization: save() of an untouched session is a no-op byte-wise.
        before = json.dumps(reopened.package.to_dict(), sort_keys=True)
        reopened.save()
        after = json.dumps(reopened.package.to_dict(), sort_keys=True)
        assert before == after

    def test_directory_source_ingests_known_files(self, tmp_path):
        ws = _workspace(tmp_path)
        handle = ws.create_session("S")
        folder = tmp_path / "photos"
        folder.mkdir()
        _save_photo(folder / "1.jpg", _exif())
        _save_photo(folder / "2.jpg", color=(1, 2, 3))
        (folder / "readme.txt").write_text("not evidence")
        source_id = handle.add_source(str(folder)).source_id
        result = handle.ingest_source(source_id)
        assert result.status is SourceStatus.INGESTED
        assert len(result.asset_ids) == 2
        assert any("readme.txt" in w for w in result.warnings)  # unhandled, recorded not silent


_IMU_CSV = "timestamp,ax,ay,az,gx,gy,gz\n0.0,0.01,0.0,9.81,0.0,0.0,0.0\n0.1,0.02,0.0,9.80,0.01,0.0,0.0\n"
_GPS_CSV = "timestamp,lat,lon,alt\n10.0,37.5,-122.3,12.0\n10.1,37.5,-122.3,12.5\n"


class TestSensorLogs:
    def test_imu_csv_detected_and_measured(self, tmp_path):
        path = tmp_path / "imu.csv"
        path.write_text(_IMU_CSV)
        kind, summary = parse_sensor_csv(str(path))
        assert kind is EvidenceKind.IMU
        assert summary["row_count"] == 2
        assert summary["completeness"] == 1.0
        assert summary["has_accel"] is True
        assert summary["has_gyro"] is True
        assert summary["first_timestamp"] == 0.0
        assert summary["last_timestamp"] == 0.1

    def test_gps_csv_detected(self, tmp_path):
        path = tmp_path / "gps.csv"
        path.write_text(_GPS_CSV)
        kind, summary = parse_sensor_csv(str(path))
        assert kind is EvidenceKind.GPS_TRACK
        assert summary["has_latlon"] is True
        assert summary["row_count"] == 2

    def test_broken_rows_count_against_completeness(self, tmp_path):
        path = tmp_path / "imu.csv"
        path.write_text(_IMU_CSV + "0.2,NOT_A_NUMBER,0,9.8,0,0,0\n")
        _kind, summary = parse_sensor_csv(str(path))
        assert summary["completeness"] == pytest.approx(2 / 3)

    def test_unrecognized_csv_refused_honestly(self, tmp_path):
        path = tmp_path / "junk.csv"
        path.write_text("a,b,c\n1,2,3\n")
        with pytest.raises(ValueError):
            parse_sensor_csv(str(path))

    def test_csv_without_timestamps_warns_about_alignment(self, tmp_path):
        path = tmp_path / "imu.csv"
        path.write_text("ax,ay,az\n0.0,0.0,9.8\n")
        _kind, summary = parse_sensor_csv(str(path))
        assert summary["has_timestamps"] is False
        assert "UNKNOWN" in summary["warning"]

    def test_imu_source_ingests_as_imu_evidence(self, tmp_path):
        ws = _workspace(tmp_path)
        handle = ws.create_session("S")
        path = tmp_path / "imu.csv"
        path.write_text(_IMU_CSV)
        source_id = handle.add_source(str(path), platform="drone").source_id
        result = handle.ingest_source(source_id)
        assert result.status is SourceStatus.INGESTED
        asset = handle.package.assets[result.asset_ids[0]]
        assert asset.kind is EvidenceKind.IMU
        assert asset.sensor_metadata["row_count"] == 2
        assert asset.acquired_at is None  # log-relative time is not acquisition time


_PLY_ASCII = (
    b"ply\nformat ascii 1.0\nelement vertex 3\n"
    b"property float x\nproperty float y\nproperty float z\nend_header\n"
    b"0 0 0\n1 0 0\n0 1 0\n"
)


class TestPointClouds:
    def test_ply_header_parse(self, tmp_path):
        path = tmp_path / "cloud.ply"
        path.write_bytes(_PLY_ASCII)
        meta = parse_ply_header(str(path))
        assert meta["point_count"] == 3
        assert meta["ply_format"] == "ascii"

    def test_corrupt_ply_refused(self, tmp_path):
        path = tmp_path / "bad.ply"
        path.write_bytes(b"not a ply at all")
        with pytest.raises(Exception):
            parse_ply_header(str(path))

    def test_pcd_header_parse(self, tmp_path):
        path = tmp_path / "cloud.pcd"
        path.write_bytes(b"# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z\nPOINTS 42\nDATA ascii\n")
        meta = parse_pcd_header(str(path))
        assert meta["point_count"] == 42
        assert meta["fields"] == ["x", "y", "z"]
        assert meta["data_mode"] == "ascii"

    def test_ply_source_ingests_with_point_count_metadata(self, tmp_path):
        ws = _workspace(tmp_path)
        handle = ws.create_session("S")
        path = tmp_path / "cloud.ply"
        path.write_bytes(_PLY_ASCII)
        source_id = handle.add_source(str(path), platform="lidar").source_id
        result = handle.ingest_source(source_id)
        assert result.status is SourceStatus.INGESTED
        asset = handle.package.assets[result.asset_ids[0]]
        assert asset.kind is EvidenceKind.POINT_CLOUD
        assert asset.sensor_metadata["point_count"] == 3


class TestCompositeCapture:
    def _make_composite(self, tmp_path, include_missing=False):
        root = tmp_path / "drone_run"
        (root / "images").mkdir(parents=True)
        (root / "imu").mkdir()
        (root / "gps").mkdir()
        (root / "calibration").mkdir()
        _save_photo(root / "images" / "f1.jpg", _exif(make="DJI", model="Mini 3"))
        _save_photo(root / "images" / "f2.jpg", color=(3, 2, 1))
        (root / "imu" / "imu.csv").write_text(_IMU_CSV)
        (root / "gps" / "gps.csv").write_text(_GPS_CSV)
        (root / "calibration" / "camera.json").write_text(json.dumps({"fx": 800.0, "fy": 800.0, "cx": 320.0, "cy": 240.0}))
        manifest = {
            "format_version": 1,
            "platform": "drone",
            "device": "DJI Mini 3",
            "components": {
                "images": "images",
                "imu": "imu/imu.csv",
                "gps": "gps/gps.csv",
                "calibration": "calibration/camera.json",
            },
        }
        if include_missing:
            manifest["components"]["telemetry"] = "telemetry/tlog.csv"
        (root / "manifest.json").write_text(json.dumps(manifest))
        return str(root)

    def test_composite_manifest_components_recorded(self, tmp_path):
        ws = _workspace(tmp_path)
        handle = ws.create_session("S")
        added = handle.add_source(self._make_composite(tmp_path))
        source = handle.sources.get(added.source_id)
        assert source.source_type is SourceType.DRONE_CAPTURE
        roles = {c.role for c in source.components}
        assert roles == {"image", "imu", "gps", "calibration"}
        assert all(c.sha256 for c in source.components)

    def test_composite_ingest_produces_all_kinds(self, tmp_path):
        ws = _workspace(tmp_path)
        handle = ws.create_session("S")
        added = handle.add_source(self._make_composite(tmp_path))
        result = handle.ingest_source(added.source_id)
        assert result.status is SourceStatus.INGESTED
        kinds = {handle.package.assets[a].kind for a in result.asset_ids}
        assert EvidenceKind.PHOTO in kinds
        assert EvidenceKind.IMU in kinds
        assert EvidenceKind.GPS_TRACK in kinds
        assert EvidenceKind.OTHER in kinds  # calibration
        # Calibration metadata is preserved verbatim.
        calib_assets = [
            handle.package.assets[a] for a in result.asset_ids
            if "calibration" in handle.package.assets[a].sensor_metadata
        ]
        assert calib_assets[0].sensor_metadata["calibration"]["fx"] == 800.0
        # Every asset cites the composite source id.
        assert all(
            handle.package.assets[a].source.source_id == added.source_id
            for a in result.asset_ids
        )

    def test_composite_missing_component_is_partial_not_silent(self, tmp_path):
        ws = _workspace(tmp_path)
        handle = ws.create_session("S")
        added = handle.add_source(self._make_composite(tmp_path, include_missing=True))
        result = handle.ingest_source(added.source_id)
        assert result.status is SourceStatus.PARTIAL
        assert any("telemetry" in e for e in result.errors)
        assert result.asset_ids  # the healthy components still ingested

    def test_unknown_manifest_role_warns(self, tmp_path):
        ws = _workspace(tmp_path)
        handle = ws.create_session("S")
        root = self._make_composite(tmp_path)
        manifest_path = tmp_path / "drone_run" / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["components"]["sonar"] = "sonar/sonar.dat"
        manifest_path.write_text(json.dumps(manifest))
        added = handle.add_source(root)
        source = handle.sources.get(added.source_id)
        assert any("sonar" in w for w in source.warnings)


class TestPayloadStoreSecurity:
    def test_put_get_round_trip_idempotent(self, tmp_path):
        store = SourcePayloadStore(str(tmp_path / "payloads"))
        data = b"\xff\xd8\xff\xe0" + b"payload" * 10
        digest = __import__("hashlib").sha256(data).hexdigest()
        path1 = store.put(data, digest)
        path2 = store.put(data, digest)
        assert path1 == path2
        assert store.get(digest) == data

    def test_invalid_digests_never_touch_filesystem(self, tmp_path):
        store = SourcePayloadStore(str(tmp_path / "payloads"))
        for evil in ("../victim", "a" * 63, "A" * 64, "", "zz" * 32, "digest/../../x"):
            with pytest.raises(ValueError):
                store.put(b"data", evil)
            with pytest.raises(ValueError):
                store.path_for(evil)

    def test_raw_source_bytes_stored_content_addressed(self, tmp_path):
        ws = _workspace(tmp_path)
        handle = ws.create_session("S")
        photo = _save_photo(tmp_path / "p.jpg")
        added = handle.add_source(photo)
        source = handle.sources.get(added.source_id)
        stored_digest = source.metadata["stored_digest"]
        assert stored_digest == source.sha256
        assert handle._payloads.get(stored_digest) == open(photo, "rb").read()


class TestVideoIngestion:
    def _write_video(self, path, frame_count=30, fps=10.0, size=(64, 48)):
        cv2 = pytest.importorskip("cv2")
        np = pytest.importorskip("numpy")
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
        assert writer.isOpened()
        for i in range(frame_count):
            frame = np.full((size[1], size[0], 3), (i * 8 % 255, 100, 150), dtype=np.uint8)
            writer.write(frame)
        writer.release()
        return str(path)

    def test_video_frames_carry_provenance_and_timestamps(self, tmp_path):
        ws = _workspace(tmp_path)
        handle = ws.create_session("S")
        video = self._write_video(tmp_path / "clip.avi")
        source_id = handle.add_source(video, platform="drone").source_id
        result = handle.ingest_source(source_id)
        assert result.status is SourceStatus.INGESTED
        assets = [handle.package.assets[a] for a in result.asset_ids]
        video_assets = [a for a in assets if a.kind is EvidenceKind.VIDEO]
        frame_assets = [a for a in assets if a.kind is EvidenceKind.PHOTO]
        assert len(video_assets) == 1
        assert len(frame_assets) == 8  # default uniform sampling target
        video_asset_id = video_assets[0].id
        for frame in frame_assets:
            # Every derived frame is traceable to the original video asset.
            assert frame.sensor_metadata["source_video_asset_id"] == video_asset_id
            assert frame.sensor_metadata["frame_timestamp_s"] >= 0.0
            assert frame.source.source_id == source_id
            assert frame.acquired_at is None  # container-relative time is not acquisition time
        stamps = sorted(f.sensor_metadata["frame_timestamp_s"] for f in frame_assets)
        assert stamps[0] == 0.0  # first frame always included
        assert stamps[-1] <= 3.0  # last sampled within the 3s duration
        assert handle.sources.get(source_id).status is SourceStatus.INGESTED

    def test_video_sampling_is_deterministic(self, tmp_path):
        ws = _workspace(tmp_path)
        video = self._write_video(tmp_path / "clip.avi")
        stamps_per_run = []
        for _ in range(2):
            handle = ws.create_session(f"S{len(stamps_per_run)}")
            source_id = handle.add_source(video).source_id
            result = handle.ingest_source(source_id)
            stamps = sorted(
                handle.package.assets[a].sensor_metadata["frame_timestamp_s"]
                for a in result.asset_ids
                if handle.package.assets[a].kind is EvidenceKind.PHOTO
            )
            stamps_per_run.append(stamps)
        assert stamps_per_run[0] == stamps_per_run[1]

    def test_corrupt_video_container_fails_honestly(self, tmp_path):
        ws = _workspace(tmp_path)
        handle = ws.create_session("S")
        bad = tmp_path / "broken.mp4"
        bad.write_bytes(b"\x00\x00\x00\x18ftypisom" + b"\x00" * 400)  # valid magic, no frames
        source_id = handle.add_source(str(bad)).source_id
        result = handle.ingest_source(source_id)
        assert result.ok is False
        assert result.status in (SourceStatus.FAILED, SourceStatus.PARTIAL)
        assert result.errors
