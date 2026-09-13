"""Tests for media preprocessing (evidence/importers.py + frames.py,
spec sec 3 MEDIA PREPROCESSING + sec 6 EVIDENCE INGESTION).

Fixtures are synthetic capture folders written to tmp_path with the
real encoders available in this environment (Pillow JPEGs with real
EXIF/GPS IFDs, OpenCV MJPG AVIs, hand-built LAS public headers), so
every expectation -- decoded GPS degrees, EXIF epochs, selected frame
indices, duplicate verdicts, package ids -- is hand-computable, and the
import path under test is the same code path a real capture takes.

The determinism claim is tested at full strength: importing the same
folder twice into fresh builders reproduces byte-identical packages
(no wall clock, no mtime reads -- timestamps come from EXIF only).
"""

from __future__ import annotations

import calendar
import io
import os
import shutil
import struct

import pytest

from evidence.frames import (
    FrameCandidate,
    FrameSelectionError,
    IFrameSelectionStrategy,
    UniformTimeSamplingStrategy,
)
from evidence.importers import (
    CorruptEvidenceError,
    FolderImportReport,
    ImportedAsset,
    NEAR_DUPLICATE_HAMMING_THRESHOLD,
    import_file,
    import_folder,
)
from evidence.packages import DeterministicPackageBuilder, EvidencePackage, EvidenceSource
from evidence.session import EvidenceKind

Image = pytest.importorskip("PIL.Image")
cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")


# ------------------------------------------------------------ fixtures

def _exif(make="Pixel", model="8 Pro", when="2026:01:02 03:04:06",
          lat_ref="N", lat=(37.0, 46.0, 12.5), lon_ref="W", lon=(122.0, 25.0, 0.0),
          altitude=12.0):
    exif = Image.Exif()
    exif[0x010F] = make
    exif[0x0110] = model
    sub = exif.get_ifd(0x8769)
    sub[0x9003] = when          # DateTimeOriginal
    sub[0x829A] = (1.0, 250.0)  # ExposureTime
    sub[0x829D] = (28, 10)      # FNumber 2.8
    sub[0x8827] = 100           # ISO
    gps = exif.get_ifd(0x8825)
    gps[0] = bytes((2, 3, 0, 0))
    gps[1] = lat_ref
    gps[2] = lat
    gps[3] = lon_ref
    gps[4] = lon
    gps[5] = 0
    gps[6] = altitude
    return exif


def _save_photo(path, exif=None, color=(120, 140, 160), size=(64, 48)):
    image = Image.new("RGB", size, color)
    kwargs = {"exif": exif} if exif is not None else {}
    image.save(path, format="JPEG", **kwargs)
    return path


def _las_payload(point_count=123456, version=b"1.4"):
    return (
        b"LASF" + b"\x00" * 20
        + version
        + b"\x00" * 80
        + struct.pack("<I", point_count)
        + b"\x00" * 32
    )


def _write_video(path, frame_count=30, fps=10.0, size=(64, 48)):
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size
    )
    assert writer.isOpened()
    for i in range(frame_count):
        frame = np.full((size[1], size[0], 3), (i * 8 % 255, 100, 150), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


@pytest.fixture
def capture_folder(tmp_path):
    """A mixed capture folder: EXIF photo + byte-identical copy + a
    near-duplicate (different bytes, same 9x8 gradient signature), a
    LAS scan, an unknown-extension file, and a 30-frame video."""
    folder = tmp_path / "capture"
    folder.mkdir()
    _save_photo(str(folder / "IMG_0001.jpg"), _exif())
    shutil.copyfile(
        str(folder / "IMG_0001.jpg"), str(folder / "IMG_0001.copy.jpg")
    )
    # Near duplicate: 1-unit color shift changes bytes and (barely) the
    # luma grid; with a 64x48 flat-ish image the dhash is stable, so it
    # is detected as NEAR, not exact.
    _save_photo(str(folder / "IMG_0002.jpg"), _exif(when="2026:01:02 03:04:07"), color=(121, 140, 160))
    (folder / "scan.las").write_bytes(_las_payload())
    (folder / "notes.txt").write_text("not evidence")
    _write_video(folder / "clip.avi", frame_count=30, fps=10.0)
    return folder


def _import(capture_folder, **kwargs):
    builder = DeterministicPackageBuilder(seed="cap")
    report = import_folder(
        builder, str(capture_folder),
        frame_strategy=UniformTimeSamplingStrategy(target_count=6),
        **kwargs,
    )
    return builder, report


# ------------------------------------------------- frame selection


class TestFrameSelection:
    def _candidates(self, n, fps=10.0):
        return [
            FrameCandidate(frame_index=i, timestamp_s=i / fps, width=64, height=48)
            for i in range(n)
        ]

    def test_fewer_than_target_returns_all_in_order(self):
        candidates = self._candidates(4)
        selected = UniformTimeSamplingStrategy(target_count=8).select(candidates)
        assert selected == candidates

    def test_spacing_is_even_and_includes_first_and_last(self):
        selected = UniformTimeSamplingStrategy(target_count=6).select(self._candidates(30))
        indices = [c.frame_index for c in selected]
        assert indices == [0, 6, 12, 17, 23, 29]  # round(i*29/5), hand-computed
        assert selected[0].timestamp_s == 0.0
        assert selected[-1].timestamp_s == 2.9

    def test_deterministic_across_repeated_calls(self):
        strategy = UniformTimeSamplingStrategy(target_count=6)
        first = strategy.select(self._candidates(30))
        second = strategy.select(self._candidates(30))
        assert first == second

    def test_preserves_candidate_metadata(self):
        selected = UniformTimeSamplingStrategy(target_count=3).select(self._candidates(10))
        assert all(isinstance(c, FrameCandidate) for c in selected)
        assert [c.width for c in selected] == [64] * 3

    def test_target_below_two_refused(self):
        with pytest.raises(FrameSelectionError, match=">= 2"):
            UniformTimeSamplingStrategy(target_count=1)

    def test_policy_name_is_stable_and_recorded(self):
        strategy = UniformTimeSamplingStrategy(target_count=6)
        assert strategy.name == "uniform_time_sampling(target=6)"

    def test_empty_video_selects_nothing(self):
        assert UniformTimeSamplingStrategy(target_count=6).select([]) == []

    def test_strategy_is_an_extensible_interface(self):
        class EveryTenth(IFrameSelectionStrategy):
            @property
            def name(self):
                return "every_tenth"

            def select(self, candidates):
                return list(candidates[::10])

        selected = EveryTenth().select(self._candidates(30))
        assert [c.frame_index for c in selected] == [0, 10, 20]


# ------------------------------------------------- EXIF / GPS decoding


class TestExifDecoding:
    def _photo_asset(self, tmp_path, exif, name="p.jpg"):
        path = _save_photo(str(tmp_path / name), exif)
        builder = DeterministicPackageBuilder(seed="x")
        report = import_folder(builder, str(tmp_path))
        package = builder.build()
        asset = next(a for a in package.all_assets() if a.source_uri == path)
        return asset, report

    def test_camera_identity_and_exposure_decode(self, tmp_path):
        asset, _ = self._photo_asset(tmp_path, _exif())
        assert asset.sensor_metadata["make"] == "Pixel"
        assert asset.sensor_metadata["model"] == "8 Pro"
        assert asset.sensor_metadata["exposure_time_s"] == pytest.approx(1 / 250)
        assert asset.sensor_metadata["f_number"] == pytest.approx(2.8)
        assert asset.sensor_metadata["iso"] == 100

    def test_gps_decodes_to_signed_decimal_degrees(self, tmp_path):
        asset, _ = self._photo_asset(tmp_path, _exif())
        # 37d 46m 12.5s = 37.770138..., 122d 25m = -122.41666... (W)
        assert asset.sensor_metadata["latitude_deg"] == pytest.approx(37 + 46 / 60 + 12.5 / 3600)
        assert asset.sensor_metadata["longitude_deg"] == pytest.approx(-(122 + 25 / 60))
        assert asset.sensor_metadata["altitude_m"] == 12.0
        assert "WGS84 assumed" in asset.sensor_metadata["gps_datum_note"]

    def test_southern_and_eastern_hemispheres_flip_sign(self, tmp_path):
        asset, _ = self._photo_asset(
            tmp_path, _exif(lat_ref="S", lon_ref="E", lat=(10.0, 0.0, 0.0), lon=(20.0, 30.0, 0.0))
        )
        assert asset.sensor_metadata["latitude_deg"] == pytest.approx(-10.0)
        assert asset.sensor_metadata["longitude_deg"] == pytest.approx(20.5)

    def test_datetime_original_becomes_acquired_at(self, tmp_path):
        asset, _ = self._photo_asset(tmp_path, _exif())
        expected = calendar.timegm((2026, 1, 2, 3, 4, 6))
        assert asset.acquired_at == expected

    def test_no_exif_means_no_timestamp_never_wall_clock(self, tmp_path):
        asset, _ = self._photo_asset(tmp_path, None)
        assert asset.acquired_at is None
        assert "make" not in asset.sensor_metadata

    def test_corrupt_exif_degrades_metadata_not_rejection(self, tmp_path):
        # Valid JPEG magic (passes the builder's container gate), garbage
        # after (PIL cannot parse): the asset must still import, with
        # honestly empty metadata and unmeasured quality.
        payload = b"\xff\xd8\xff\xe0" + b"not-really-jpeg" * 8
        path = tmp_path / "broken.jpg"
        path.write_bytes(payload)
        builder = DeterministicPackageBuilder(seed="x")
        report = import_folder(builder, str(tmp_path))
        package = builder.build()
        asset = next(a for a in package.all_assets() if a.source_uri == str(path))
        assert asset.acquired_at is None
        assert "make" not in asset.sensor_metadata
        assert asset.quality["measured"] == 0.0
        assert "unmeasured" in asset.quality.get("quality_note", "") or asset.quality
        assert any(a.asset_id == asset.id for a in report.imported)


# ------------------------------------------------- quality metrics


class TestQualityMetrics:
    def _import_one(self, tmp_path, name, **save_kwargs):
        path = _save_photo(str(tmp_path / name), **save_kwargs)
        builder = DeterministicPackageBuilder(seed="q")
        import_folder(builder, str(tmp_path))
        package = builder.build()
        return next(a for a in package.all_assets() if a.source_uri == path)

    def test_flat_image_has_zero_laplacian_variance(self, tmp_path):
        asset = self._import_one(tmp_path, "flat.jpg", color=(120, 140, 160))
        assert asset.quality["measured"] == 1.0
        assert asset.quality["laplacian_variance"] == pytest.approx(0.0, abs=1e-9)
        # ITU-R 601 luma of (120,140,160) = 136.3 (rounded by uint8)
        assert asset.quality["luma_mean"] == pytest.approx(136.3, abs=0.7)
        assert asset.quality["clipped_fraction"] == pytest.approx(0.0)

    def test_black_image_reports_clipping(self, tmp_path):
        asset = self._import_one(tmp_path, "black.jpg", color=(0, 0, 0))
        assert asset.quality["clipped_fraction"] == pytest.approx(1.0)

    def test_structure_raises_laplacian_variance(self, tmp_path):
        # A hard checkerboard is far "sharper" (higher Laplacian energy)
        # than the same scene blurred by JPEG smoothing of a gradient.
        checker = tmp_path / "checker.jpg"
        image = Image.new("L", (64, 48), 0)
        for y in range(0, 48, 4):
            for x in range(0, 64, 8):
                image.paste(255, (x + (y // 4 % 2) * 4, y, x + 4 + (y // 4 % 2) * 4, y + 4))
        image.save(checker, format="JPEG", quality=100)
        gradient = tmp_path / "gradient.jpg"
        ramp = np.tile(np.linspace(0, 255, 64, dtype=np.uint8), (48, 1))
        Image.fromarray(ramp, mode="L").save(gradient, format="JPEG", quality=100)

        builder = DeterministicPackageBuilder(seed="q")
        import_folder(builder, str(tmp_path))
        package = builder.build()
        by_uri = {a.source_uri: a for a in package.all_assets()}
        sharp = by_uri[str(checker)].quality["laplacian_variance"]
        smooth = by_uri[str(gradient)].quality["laplacian_variance"]
        assert sharp > smooth * 10

    def test_resolution_comes_from_container_headers(self, tmp_path):
        asset = self._import_one(tmp_path, "sized.jpg", size=(64, 48))
        assert asset.quality["width_px"] == 64.0
        assert asset.quality["height_px"] == 48.0


# ------------------------------------------------- folder import


class TestFolderImport:
    def test_full_mixed_folder_counts(self, capture_folder):
        builder, report = _import(capture_folder)
        # 2 photos (near-dup still imports, marked) + 1 copy-skipped
        # + 1 LAS + 1 video + 6 frames = 10 imports, 9 assets.
        kinds = {}
        for imported in report.imported:
            kinds[imported.kind] = kinds.get(imported.kind, 0) + 1
        assert kinds[EvidenceKind.PHOTO] == 8  # 2 originals + 6 frames
        assert kinds[EvidenceKind.LIDAR] == 1
        assert kinds[EvidenceKind.VIDEO] == 1
        assert len(report.duplicates_skipped) == 1
        assert len(report.near_duplicates_marked) == 1
        assert len(report.unhandled_paths) == 1
        assert report.unhandled_paths[0].endswith("notes.txt")
        # Package holds 10 distinct assets: 2 photos + 1 LAS + 1 video
        # + 6 frames (the exact copy was skipped).
        assert len(builder.build().all_assets()) == 10

    def test_exact_duplicate_is_skip_and_record_not_raise(self, capture_folder):
        builder, report = _import(capture_folder)
        skipped_path, existing_id = report.duplicates_skipped[0]
        # Sorted order: IMG_0001.copy.jpg sorts BEFORE IMG_0001.jpg, so
        # the copy is the FIRST-seen asset and the original is skipped.
        assert skipped_path.endswith("IMG_0001.jpg")
        package = builder.build()
        assert existing_id in package.assets
        assert not any(a.path == skipped_path for a in report.imported)

    def test_near_duplicate_marked_with_reference(self, capture_folder):
        _builder, report = _import(capture_folder)
        assert len(report.near_duplicates_marked) == 1
        path, reference = report.near_duplicates_marked[0]
        assert path.endswith("IMG_0002.jpg")
        assert reference  # the first-seen asset's import key

    def test_rebuild_reproduces_the_package_byte_for_byte(self, capture_folder):
        builder_a, _ = _import(capture_folder)
        builder_b, _ = _import(capture_folder)
        package_a, package_b = builder_a.build(), builder_b.build()
        assert package_a.to_dict() == package_b.to_dict()
        assert package_a == package_b

    def test_corrupt_photo_raises_naming_the_path(self, tmp_path):
        (tmp_path / "faked.jpg").write_bytes(b"GIF89a" + b"\x00" * 64)
        builder = DeterministicPackageBuilder(seed="c")
        with pytest.raises(CorruptEvidenceError, match="faked.jpg"):
            import_folder(builder, str(tmp_path))

    def test_empty_folder_yields_valid_empty_package(self, tmp_path):
        builder = DeterministicPackageBuilder(seed="e")
        report = import_folder(builder, str(tmp_path))
        assert report.imported == []
        package = builder.build()
        assert package.all_assets() == []

    def test_custom_source_is_recorded_on_assets(self, capture_folder):
        source = EvidenceSource(
            source_id="drone-01", platform="drone", device="Mavic 3", operator="pilot"
        )
        builder, _ = _import(capture_folder, source=source)
        package = builder.build()
        for asset in package.all_assets():
            assert asset.source.source_id == "drone-01"

    def test_import_file_returns_the_video_asset_for_video(self, capture_folder):
        builder = DeterministicPackageBuilder(seed="v")
        video_path = str(capture_folder / "clip.avi")
        imported = import_file(
            builder, video_path, frame_strategy=UniformTimeSamplingStrategy(target_count=4)
        )
        assert imported.kind is EvidenceKind.VIDEO
        assert imported.asset_id in builder.build().assets

    def test_report_serializes(self, capture_folder):
        _builder, report = _import(capture_folder)
        data = report.to_dict()
        assert set(data.keys()) == {
            "imported", "duplicates_skipped", "near_duplicates_marked", "unhandled_paths"
        }
        assert len(data["imported"]) == len(report.imported)


# ------------------------------------------------- video pipeline


class TestVideoPipeline:
    def test_selects_exactly_target_frames_first_and_last(self, tmp_path):
        video = _write_video(tmp_path / "v.avi", frame_count=30, fps=10.0)
        builder = DeterministicPackageBuilder(seed="v")
        report = import_folder(
            builder, str(tmp_path), frame_strategy=UniformTimeSamplingStrategy(target_count=6)
        )
        frames = [
            a for a in builder.build().all_assets() if "#frame-" in a.source_uri
        ]
        indices = [a.quality["frame_index"] for a in frames]
        assert indices == [0.0, 6.0, 12.0, 17.0, 23.0, 29.0]
        assert all(a.kind is EvidenceKind.PHOTO for a in frames)

    def test_frames_reference_their_source_video_asset(self, tmp_path):
        video = _write_video(tmp_path / "v.avi", frame_count=12, fps=10.0)
        builder = DeterministicPackageBuilder(seed="v")
        import_folder(builder, str(tmp_path), frame_strategy=UniformTimeSamplingStrategy(target_count=4))
        package = builder.build()
        video_assets = [a for a in package.all_assets() if a.kind is EvidenceKind.VIDEO]
        frames = [a for a in package.all_assets() if "#frame-" in a.source_uri]
        assert len(video_assets) == 1
        assert {a.sensor_metadata["source_video_asset_id"] for a in frames} == {
            video_assets[0].id
        }
        assert all("uniform_time_sampling" in a.sensor_metadata["selection_strategy"] for a in frames)

    def test_frame_timestamps_are_container_relative_metadata_only(self, tmp_path):
        _write_video(tmp_path / "v.avi", frame_count=12, fps=10.0)
        builder = DeterministicPackageBuilder(seed="v")
        import_folder(builder, str(tmp_path), frame_strategy=UniformTimeSamplingStrategy(target_count=4))
        frames = [a for a in builder.build().all_assets() if "#frame-" in a.source_uri]
        # Frame time lives in sensor metadata; acquired_at stays None --
        # a container-relative offset never claims to be an acquisition time.
        assert all(a.acquired_at is None for a in frames)
        # round(i*11/3): 0, 4, 7, 11 -> 0.0s, 0.4s, 0.7s, 1.1s at 10 fps
        assert [a.sensor_metadata["frame_timestamp_s"] for a in frames] == [0.0, 0.4, 0.7, 1.1]

    def test_never_processes_every_frame(self, tmp_path):
        video = _write_video(tmp_path / "v.avi", frame_count=100, fps=25.0)
        builder = DeterministicPackageBuilder(seed="v")
        report = import_folder(
            builder, str(tmp_path), frame_strategy=UniformTimeSamplingStrategy(target_count=10)
        )
        frame_count = sum(1 for a in report.imported if "#frame-" in a.path)
        assert frame_count == 10  # not 100

    def test_garbage_video_container_refused(self, tmp_path):
        path = tmp_path / "junk.avi"
        path.write_bytes(b"not a container at all" * 4)
        builder = DeterministicPackageBuilder(seed="v")
        with pytest.raises(CorruptEvidenceError, match="junk.avi"):
            import_folder(builder, str(tmp_path))


# ------------------------------------------------- end to end


class TestEndToEndCaptureToObservation:
    def test_folder_to_package_to_reconstruction_to_world(self, capture_folder):
        """The full promised chain: folder on disk -> validated package
        -> reconstruction items -> compiled WorldIR, with every 3D point
        traceable to a real ingested asset."""
        from reconstruction.backend.fake import FakeReconstructionBackend
        from reconstruction.backend.interface import (
            ReconstructedCameraPose,
            ReconstructedPoint,
        )
        from provenance import Uncertainty
        from engine.compiler.world_compiler import CompileOptions, compile_reconstruction_to_world

        builder, report = _import(capture_folder)
        package = builder.build()
        items = package.to_evidence_items()
        photo_items = [i for i in items if i.kind is EvidenceKind.PHOTO and "#frame" not in i.source_uri]
        # Two DISTINCT photos imported: the byte-identical copy won the
        # sorted-order race (IMG_0001.copy.jpg imports first; the
        # original is the skipped duplicate) and the near-dup imports
        # marked. Frames are excluded by the #frame filter.
        assert len(photo_items) == 2

        # Canned reconstruction consuming the imported evidence ids:
        id_a = photo_items[0].id
        id_b = next(i.id for i in items if i.kind is EvidenceKind.VIDEO)
        points = [
            ReconstructedPoint(
                position=(x * 0.5, 0.0, z * 0.5),
                track_id=f"t-{x}-{z}",
                source_evidence_ids=[id_a if (x + z) % 2 == 0 else id_b],
            )
            for x in range(5)
            for z in range(5)
        ]
        poses = [
            ReconstructedCameraPose(
                evidence_id=id_a, position=(1.0, 1.5, 1.0), rotation=(1.0, 0.0, 0.0, 0.0),
                uncertainty=Uncertainty(),
            ),
            ReconstructedCameraPose(
                evidence_id=id_b, position=(1.5, 1.5, 1.0), rotation=(1.0, 0.0, 0.0, 0.0),
                uncertainty=Uncertainty(),
            ),
        ]
        backend = FakeReconstructionBackend(canned_points=points, canned_poses=poses)
        result = backend.reconstruct(items)
        assert result.registration_status in ("success", "partial")

        world, diagnostics = compile_reconstruction_to_world(result, CompileOptions(seed=42))
        assert diagnostics.points_total == len(points)
        # Evidence traceability: every reconstructed point's evidence ids
        # resolve to real assets in the package built from the folder.
        for point in result.points:
            for evidence_id in point.source_evidence_ids:
                assert evidence_id in package.assets

    def test_folder_to_observation_set_to_fused_measurement(self, capture_folder):
        """folder -> package -> ObservationSet -> fuse_quantity(): the
        evidence-referenced observation loop closes on imported files."""
        from engine.core.units import Unit
        from provenance import Provenance
        from reconstruction.fusion import FusableObservation, fuse_quantity

        builder, report = _import(capture_folder)
        package = builder.build()
        photo_ids = [
            a.id for a in package.all_assets()
            if a.kind is EvidenceKind.PHOTO and "#frame" not in a.source_uri
        ]
        assert len(photo_ids) == 2  # copy + near-dup; original was the skipped dup

        from evidence.packages import EvidenceReference, ObservationSet

        las_id = next(a.id for a in package.all_assets() if a.kind is EvidenceKind.LIDAR)
        references = (
            EvidenceReference(
                reference_id="r-lidar", asset_id=las_id,
                observation_id="obs-lidar", role="measurement_source",
            ),
            EvidenceReference(
                reference_id="r-photo", asset_id=photo_ids[0],
                observation_id="obs-photo", role="support",
            ),
        )
        package.add_observation_set(
            ObservationSet(set_id="s1", quantity="wall_width_m", references=references)
        )
        fused = fuse_quantity(
            [
                FusableObservation(
                    value=3.17, unit=Unit.METER, source="lidar",
                    provenance=Provenance.OBSERVED, confidence=0.98, precision=0.005,
                    evidence_ids=(las_id,),
                ),
                FusableObservation(
                    value=3.17, unit=Unit.METER, source="photogrammetry",
                    provenance=Provenance.RECONSTRUCTED, confidence=0.9, precision=0.005,
                    evidence_ids=(photo_ids[0],),
                ),
            ],
            "wall_width_m",
        )
        # Agreeing sources: ESTIMATED, and both claims cite real imported assets.
        assert fused.provenance is Provenance.ESTIMATED
        for obs in fused.contributing:
            assert all(evidence_id in package.assets for evidence_id in obs.evidence_ids)

    def test_package_round_trip_survives_imported_metadata(self, capture_folder):
        builder, _ = _import(capture_folder)
        package = builder.build()
        restored = EvidencePackage.from_dict(package.to_dict())
        assert restored == package
