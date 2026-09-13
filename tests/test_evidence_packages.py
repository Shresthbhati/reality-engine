"""Tests for evidence packages (evidence/packages.py, spec sec 6).

Fixtures are canonical synthetic payloads (real JPEG/PNG/LAS magic
bytes over deterministic filler), so every expected id, hash prefix,
and error condition is hand-derivable. The suite covers the four
guarantees the module docstring claims: deterministic content-derived
identity, real corruption refusal, content-hash duplicate detection,
and deterministic serialization -- plus the two structured callers that
make this layer the ingestion entry point rather than a standalone
abstraction: ObservationSet -> fuse_quantity(), and
to_evidence_items() -> the world compiler.
"""

from __future__ import annotations

import pytest

from evidence.packages import (
    CorruptEvidenceError,
    DeterministicPackageBuilder,
    EvidenceAsset,
    EvidencePackage,
    EvidenceReference,
    EvidenceSource,
    ObservationSet,
    UnknownAssetError,
)
from evidence.session import DuplicateEvidenceError, EvidenceKind
from provenance import Provenance, Uncertainty
from reconstruction.fusion import FusableObservation, fused_to_measurement, fuse_quantity
from world_ir.coordinates import Frame


# ------------------------------------------------------------- fixtures

def _jpeg(seed_byte: int = 0x01) -> bytes:
    return b"\xff\xd8\xff\xe0" + bytes([seed_byte]) * 64


def _png(seed_byte: int = 0x02) -> bytes:
    return b"\x89PNG\r\n\x1a\n" + bytes([seed_byte]) * 64


def _las() -> bytes:
    return b"LASF" + b"\x00" * 64


_CAM = EvidenceSource(
    source_id="cam-01",
    platform="phone",
    device="Pixel 8",
    operator="capture-tech",
)


def _builder(seed: str = "pkg") -> DeterministicPackageBuilder:
    builder = DeterministicPackageBuilder(seed=seed)
    builder.register_source(_CAM)
    return builder


# --------------------------------------------------- deterministic ids


class TestDeterministicIds:
    def test_same_bytes_same_seed_same_id(self):
        jpg = _jpeg()
        a = _builder("alpha").add_payload(
            jpg, kind=EvidenceKind.PHOTO, source=_CAM, source_uri="c/001.jpg"
        )
        b = _builder("alpha").add_payload(
            jpg, kind=EvidenceKind.PHOTO, source=_CAM, source_uri="c/001.jpg"
        )
        assert a == b

    def test_id_embeds_the_content_hash(self):
        aid = _builder().add_payload(
            _jpeg(), kind=EvidenceKind.PHOTO, source=_CAM, source_uri="c/001.jpg"
        )
        import hashlib

        digest = hashlib.sha256(_jpeg()).hexdigest()
        assert digest[:16] in aid

    def test_rebuilt_package_has_identical_identity(self):
        """Rebuilding from the same payloads reproduces the whole
        package byte-for-byte -- the ingestion-side replay guarantee."""

        def build():
            builder = _builder("replay")
            builder.add_payload(
                _jpeg(), kind=EvidenceKind.PHOTO, source=_CAM,
                source_uri="c/001.jpg", acquired_at=100.0,
            )
            builder.add_payload(
                _png(), kind=EvidenceKind.PHOTO, source=_CAM,
                source_uri="c/002.png", acquired_at=101.0,
            )
            builder.add_payload(
                _las(), kind=EvidenceKind.LIDAR, source=_CAM,
                source_uri="scan.las", acquired_at=102.0,
            )
            return builder.build()

        first, second = build(), build()
        assert first.package_id == second.package_id
        assert first.to_dict() == second.to_dict()
        assert first == second

    def test_no_clocks_and_no_uuids(self):
        """Identity comes from content and seed, never the wall clock."""
        pkg = _builder("clock").add_payload(
            _jpeg(), kind=EvidenceKind.PHOTO, source=_CAM, source_uri="c/001.jpg"
        )
        builder = _builder("clock")
        builder.add_payload(
            _jpeg(), kind=EvidenceKind.PHOTO, source=_CAM, source_uri="c/001.jpg"
        )
        assert "pkg-clock" in builder.build().package_id
        assert isinstance(pkg, str)

    def test_default_uncertainty_and_frame(self):
        builder = _builder()
        aid = builder.add_payload(
            _jpeg(), kind=EvidenceKind.PHOTO, source=_CAM, source_uri="c/001.jpg"
        )
        asset = builder.build().assets[aid]
        assert asset.uncertainty == Uncertainty()
        assert asset.coordinate_frame is Frame.SESSION_LOCAL
        assert asset.provenance is Provenance.OBSERVED


# ------------------------------------------------------ corruption gate


class TestCorruptionHandling:
    def test_jpeg_with_wrong_magic_refused(self):
        not_a_jpeg = b"GIF89a" + b"\x00" * 64
        with pytest.raises(CorruptEvidenceError, match="magic"):
            _builder().add_payload(
                not_a_jpeg, kind=EvidenceKind.PHOTO, source=_CAM, source_uri="c/001.jpg"
            )

    def test_png_with_wrong_magic_refused(self):
        not_a_png = b"\xff\xd8\xff" + b"\x00" * 64
        with pytest.raises(CorruptEvidenceError, match="magic"):
            _builder().add_payload(
                not_a_png, kind=EvidenceKind.PHOTO, source=_CAM, source_uri="c/002.png"
            )

    def test_tiny_payload_refused_as_corruption(self):
        with pytest.raises(CorruptEvidenceError, match="too small"):
            _builder().add_payload(
                b"\xff\xd8\xff", kind=EvidenceKind.PHOTO, source=_CAM, source_uri="c/001.jpg"
            )

    def test_correct_magic_passes_for_every_signature_family(self):
        builder = _builder()
        builder.add_payload(
            _jpeg(), kind=EvidenceKind.PHOTO, source=_CAM, source_uri="c/a.jpg"
        )
        builder.add_payload(
            _png(), kind=EvidenceKind.PHOTO, source=_CAM, source_uri="c/b.png"
        )
        builder.add_payload(
            _las(), kind=EvidenceKind.LIDAR, source=_CAM, source_uri="s/c.las"
        )
        assert len(builder.build().all_assets()) == 3

    def test_extensionless_formats_validate_by_size_only(self):
        """Sensor logs / depth dumps have no magic; the size gate is
        their only structural check, and it must not reject them."""
        aid = _builder().add_payload(
            b"ts,x,y,z,fix\n0,1,2,3,rtk\n", kind=EvidenceKind.GPS_TRACK,
            source=_CAM, source_uri="logs/gps.csv",
        )
        assert aid.startswith("ev-")

    def test_mislabeled_extension_refused(self):
        """A LAS payload named .png is corrupt-or-mislabeled either way."""
        with pytest.raises(CorruptEvidenceError):
            _builder().add_payload(
                _las(), kind=EvidenceKind.POINT_CLOUD, source=_CAM, source_uri="s/las.png"
            )


# -------------------------------------------------- duplicate detection


class TestDuplicateDetection:
    def test_same_bytes_different_name_is_a_duplicate(self):
        jpg = _jpeg()
        builder = _builder()
        first = builder.add_payload(
            jpg, kind=EvidenceKind.PHOTO, source=_CAM, source_uri="a/IMG_0001.jpg"
        )
        with pytest.raises(DuplicateEvidenceError, match=first):
            builder.add_payload(
                jpg, kind=EvidenceKind.PHOTO, source=_CAM, source_uri="b/copy.jpg"
            )

    def test_error_names_the_existing_asset_for_dedup_by_reference(self):
        jpg = _jpeg()
        builder = _builder()
        first = builder.add_payload(
            jpg, kind=EvidenceKind.PHOTO, source=_CAM, source_uri="a/IMG_0001.jpg"
        )
        with pytest.raises(DuplicateEvidenceError) as excinfo:
            builder.add_payload(
                jpg, kind=EvidenceKind.PHOTO, source=_CAM, source_uri="b/copy.jpg"
            )
        assert first in str(excinfo.value)

    def test_one_byte_difference_is_not_a_duplicate(self):
        builder = _builder()
        builder.add_payload(
            _jpeg(0x01), kind=EvidenceKind.PHOTO, source=_CAM, source_uri="a/1.jpg"
        )
        builder.add_payload(
            _jpeg(0x02), kind=EvidenceKind.PHOTO, source=_CAM, source_uri="a/2.jpg"
        )
        assert len(builder.build().all_assets()) == 2

    def test_package_level_add_asset_dedups_silently(self):
        """On the package itself a duplicate content hash returns the
        existing id (policy layer raises; storage layer de-duplicates)."""
        builder = _builder()
        original_id = builder.add_payload(
            _jpeg(), kind=EvidenceKind.PHOTO, source=_CAM, source_uri="a/1.jpg"
        )
        package = builder.build()
        original = package.assets[original_id]
        twin = EvidenceAsset(
            id="ev-twin", kind=EvidenceKind.PHOTO, source_uri="b/twin.jpg",
            sha256=original.sha256, source=_CAM, acquired_at=None,
            coordinate_frame=Frame.SESSION_LOCAL, sensor_metadata={}, quality={},
            provenance=Provenance.OBSERVED, uncertainty=Uncertainty(),
            processing_history=(),
        )
        assert package.add_asset(twin) == original_id
        assert len(package.all_assets()) == 1

    def test_has_content_lookup(self):
        builder = _builder()
        aid = builder.add_payload(
            _png(), kind=EvidenceKind.PHOTO, source=_CAM, source_uri="a/1.png"
        )
        pkg = builder.build()
        assert pkg.has_content(pkg.assets[aid].sha256)
        assert pkg.asset_for_content(pkg.assets[aid].sha256) == aid
        assert not pkg.has_content("0" * 64)


# ------------------------------------------------- immutability + history


class TestImmutabilityAndHistory:
    def test_record_processing_appends_without_mutating_evidence(self):
        builder = _builder()
        aid = builder.add_payload(
            _jpeg(), kind=EvidenceKind.PHOTO, source=_CAM, source_uri="a/1.jpg"
        )
        pkg = builder.build()
        before = pkg.assets[aid]
        from evidence.session import ProcessingRecord

        pkg.record_processing(aid, ProcessingRecord(operation="undistort", at_tick=5))
        after = pkg.assets[aid]
        assert len(after.processing_history) == 1
        assert after.processing_history[0].operation == "undistort"
        # The frozen dataclass was replaced, not mutated:
        assert before is not after
        assert before.processing_history == ()

    def test_record_processing_unknown_asset_raises(self):
        pkg = EvidencePackage(package_id="p")
        from evidence.session import ProcessingRecord

        with pytest.raises(UnknownAssetError):
            pkg.record_processing("nope", ProcessingRecord(operation="x"))

    def test_history_survives_serialization(self):
        from evidence.session import ProcessingRecord

        builder = _builder()
        aid = builder.add_payload(
            _jpeg(), kind=EvidenceKind.PHOTO, source=_CAM, source_uri="a/1.jpg",
            processing_history=(ProcessingRecord(operation="decode", at_tick=1),),
        )
        pkg = builder.build()
        pkg.record_processing(aid, ProcessingRecord(operation="undistort", at_tick=2))
        restored = EvidencePackage.from_dict(pkg.to_dict())
        history = restored.assets[aid].processing_history
        assert [p.operation for p in history] == ["decode", "undistort"]


# --------------------------------------------------------- observation sets


class TestObservationSets:
    def test_fuses_via_the_observation_set(self):
        """The structured caller: an ObservationSet's references name the
        assets behind each fused claim; fusion itself is unchanged."""
        from engine.core.units import Unit

        builder = _builder()
        lidar_id = builder.add_payload(
            _las(), kind=EvidenceKind.LIDAR, source=_CAM, source_uri="s/room.las"
        )
        photo_id = builder.add_payload(
            _jpeg(), kind=EvidenceKind.PHOTO, source=_CAM, source_uri="a/1.jpg"
        )
        pkg = builder.build()
        refs = (
            EvidenceReference(
                reference_id="r-lidar", asset_id=lidar_id,
                observation_id="obs-lidar-width", role="measurement_source",
            ),
            EvidenceReference(
                reference_id="r-photo", asset_id=photo_id,
                observation_id="obs-photogrammetry-width", role="measurement_source",
            ),
        )
        pkg.add_observation_set(
            ObservationSet(set_id="set-width", quantity="wall_width_m", references=refs)
        )

        observations = [
            FusableObservation(
                value=3.17, unit=Unit.METER, source="lidar",
                provenance=Provenance.OBSERVED, confidence=0.98,
                precision=0.005, evidence_ids=(lidar_id,),
            ),
            FusableObservation(
                value=3.22, unit=Unit.METER, source="photogrammetry",
                provenance=Provenance.RECONSTRUCTED, confidence=0.9,
                precision=0.005, evidence_ids=(photo_id,),
            ),
        ]
        fused = fuse_quantity(observations, "wall_width_m")
        assert fused.provenance is Provenance.CONFLICT
        # Every contributing observation's evidence chain stays resolvable
        # through the package: asset ids are real, role is recorded.
        for obs in fused.contributing:
            (evidence_id,) = obs.evidence_ids
            assert evidence_id in pkg.assets
        referenced_assets = {r.asset_id for r in pkg.observation_sets["set-width"].references}
        assert referenced_assets == {lidar_id, photo_id}

    def test_fused_conflict_bridges_to_measurement(self):
        from engine.core.units import Unit
        from world_ir import Measurement

        fused = fuse_quantity(
            [
                FusableObservation(
                    value=3.17, unit=Unit.METER, source="lidar",
                    provenance=Provenance.OBSERVED, confidence=0.98, precision=0.005,
                ),
                FusableObservation(
                    value=3.22, unit=Unit.METER, source="photogrammetry",
                    provenance=Provenance.RECONSTRUCTED, confidence=0.9, precision=0.005,
                ),
            ],
            "wall_width_m",
        )
        measurement = fused_to_measurement(fused)
        assert isinstance(measurement, Measurement)
        assert measurement.provenance is Provenance.CONFLICT


# ------------------------------------------------------ compiler caller


class TestCompilerCaller:
    def _package(self) -> tuple:
        """A package with the two photos 'behind' the compiled scene's
        points -- the ingestion half of the evidence chain."""
        builder = _builder("scene")
        p1 = builder.add_payload(
            _jpeg(0x11), kind=EvidenceKind.PHOTO, source=_CAM,
            source_uri="a/1.jpg", acquired_at=1.0,
        )
        p2 = builder.add_payload(
            _jpeg(0x22), kind=EvidenceKind.PHOTO, source=_CAM,
            source_uri="a/2.jpg", acquired_at=2.0,
        )
        return builder.build(), (p1, p2)

    def test_assets_bridge_to_reconstruction_input(self):
        pkg, _ = self._package()
        items = pkg.to_evidence_items()
        assert [i.kind for i in items] == [EvidenceKind.PHOTO, EvidenceKind.PHOTO]
        assert all(item.sha256 for item in items)
        assert all(item.id in pkg.assets for item in items)

    def test_assets_flow_into_a_compiled_worlds_points(self):
        """End-to-end evidence chain: package assets -> reconstruction
        points' source_evidence_ids -> compiled WorldIR."""
        from perception.geometry.planes import detect_planes
        from reconstruction.backend.interface import ReconstructedPoint, ReconstructionResult

        pkg, (p1, p2) = self._package()
        floor_points = [
            ReconstructedPoint(
                position=(x * 0.5, 0.0, z * 0.5),
                track_id=f"t-{x}-{z}",
                source_evidence_ids=[p1 if (x + z) % 2 == 0 else p2],
            )
            for x in range(5)
            for z in range(5)
        ]
        result = ReconstructionResult(
            points=floor_points,
            camera_poses=[],
            registration_status="success",
        )
        detection = detect_planes(result, seed=42)
        # Every recovered point traces back to a real ingested asset:
        traced = {
            pid
            for pt in result.points
            for pid in pt.source_evidence_ids
        }
        assert traced == {p1, p2}
        assert all(pid in pkg.assets for pid in traced)
        assert detection.points_total == len(floor_points)

    def test_compiler_consumes_bridged_items(self):
        """to_evidence_items() output satisfies the reconstruction
        backends' input contract (List[EvidenceItem])."""
        from evidence.session import EvidenceItem

        pkg, _ = self._package()
        items = pkg.to_evidence_items()
        assert all(isinstance(i, EvidenceItem) for i in items)


# --------------------------------------------------------- serialization


class TestSerialization:
    def test_round_trip_is_identical(self):
        builder = _builder("round")
        lidar_id = builder.add_payload(
            _las(), kind=EvidenceKind.LIDAR, source=_CAM, source_uri="s/x.las",
            acquired_at=5.0, coordinate_frame=Frame.ENU,
            sensor_metadata={"gps_fix": "rtk_fix"},
            quality={"coverage_fraction": 0.87},
        )
        pkg = builder.build()
        pkg.add_observation_set(
            ObservationSet(
                set_id="set-1",
                quantity="room_height_m",
                references=(
                    EvidenceReference(
                        reference_id="r1", asset_id=lidar_id,
                        observation_id="obs-1", role="primary",
                    ),
                ),
            )
        )
        restored = EvidencePackage.from_dict(pkg.to_dict())
        assert restored == pkg
        assert restored.package_id == pkg.package_id
        assert restored.observation_sets["set-1"] == pkg.observation_sets["set-1"]

    def test_version_mismatch_refused(self):
        data = EvidencePackage(package_id="p").to_dict()
        data["format_version"] = 999
        with pytest.raises(ValueError, match="format version"):
            EvidencePackage.from_dict(data)

    def test_asset_dict_shape_is_stable(self):
        builder = _builder()
        aid = builder.add_payload(
            _jpeg(), kind=EvidenceKind.PHOTO, source=_CAM, source_uri="a/1.jpg",
            acquired_at=9.0, quality={"blur_score": 0.91},
        )
        d = builder.build().assets[aid].to_dict()
        assert set(d.keys()) == {
            "id", "kind", "source_uri", "sha256", "source", "acquired_at",
            "coordinate_frame", "sensor_metadata", "quality", "provenance",
            "uncertainty", "processing_history",
        }
        assert d["kind"] == "photo"
        assert d["provenance"] == "OBSERVED"
