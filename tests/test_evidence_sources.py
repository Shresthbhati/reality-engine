"""evidence/sources.py: canonical Source model + registry (source
identity, deterministic hashes, duplicate detection, serialization,
lifecycle). Deterministic offline tests -- no decoders required."""

import pytest

from evidence.sources import (
    Source,
    SourceComponent,
    SourceRegistry,
    SourceStatus,
    SourceType,
    UnknownSourceError,
    classify_source,
    hash_file,
    hash_tree,
    source_id_for_digest,
)


def _make_file(tmp_path, name="a.jpg", payload=b"\xff\xd8\xff\xe0" + b"x" * 64):
    path = tmp_path / name
    path.write_bytes(payload)
    return str(path)


class TestContentIdentity:
    def test_source_id_is_content_derived_and_stable(self, tmp_path):
        path = _make_file(tmp_path)
        d1 = hash_file(path)
        copy = tmp_path / "copy_of_a.jpg"
        copy.write_bytes(open(path, "rb").read())
        assert hash_file(str(copy)) == d1
        assert source_id_for_digest(d1) == source_id_for_digest(d1)
        assert source_id_for_digest(d1).startswith("src-")

    def test_hash_tree_is_root_location_independent(self, tmp_path):
        tree1 = tmp_path / "t1" / "capture"
        tree2 = tmp_path / "t2" / "elsewhere"
        for base in (tree1, tree2):
            (base / "sub").mkdir(parents=True)
            (base / "sub" / "f.bin").write_bytes(b"same")
            (base / "g.txt").write_bytes(b"same2")
        assert hash_tree(str(tree1)) == hash_tree(str(tree2))

    def test_hash_tree_changes_when_contents_change(self, tmp_path):
        tree = tmp_path / "cap"
        tree.mkdir()
        (tree / "a.bin").write_bytes(b"one")
        d1 = hash_tree(str(tree))
        (tree / "a.bin").write_bytes(b"two")
        assert hash_tree(str(tree)) != d1

    def test_no_wall_clock_in_source(self, tmp_path):
        path = _make_file(tmp_path)
        registry = SourceRegistry()
        source, _ = registry.register(path, source_type=SourceType.IMAGE, sha256=hash_file(path))
        assert source.acquired_at is None  # unknown stays explicitly unknown
        assert source.imported_at is None


class TestRegistryDedupe:
    def test_re_registering_same_bytes_resolves_to_existing_source(self, tmp_path):
        path = _make_file(tmp_path)
        registry = SourceRegistry(session_id="s1")
        source, created = registry.register(path, source_type=SourceType.IMAGE, sha256=hash_file(path))
        assert created is True
        source2, created2 = registry.register(path, source_type=SourceType.IMAGE, sha256=hash_file(path))
        assert created2 is False
        assert source2.source_id == source.source_id
        assert len(registry) == 1

    def test_different_bytes_get_different_sources(self, tmp_path):
        p1 = _make_file(tmp_path, "a.jpg")
        p2 = _make_file(tmp_path, "b.jpg", payload=b"\xff\xd8\xff\xe1" + b"y" * 64)
        registry = SourceRegistry()
        r1, _ = registry.register(p1, source_type=SourceType.IMAGE, sha256=hash_file(p1))
        r2, _ = registry.register(p2, source_type=SourceType.IMAGE, sha256=hash_file(p2))
        assert r1.source_id != r2.source_id
        assert len(registry) == 2

    def test_get_unknown_raises(self):
        with pytest.raises(UnknownSourceError):
            SourceRegistry().get("src-nonexistent")

    def test_update_preserves_identity_and_order(self, tmp_path):
        path = _make_file(tmp_path)
        registry = SourceRegistry()
        source, _ = registry.register(path, source_type=SourceType.IMAGE, sha256=hash_file(path))
        updated = registry.update(
            source.source_id,
            status=SourceStatus.INGESTED,
            asset_ids=("ev-1", "ev-2"),
        )
        assert updated.source_id == source.source_id
        assert updated.status is SourceStatus.INGESTED
        assert registry.all_sources()[0].source_id == source.source_id


class TestSerialization:
    def test_source_round_trip(self, tmp_path):
        path = _make_file(tmp_path)
        source = Source(
            source_id="src-abc",
            source_type=SourceType.COMPOSITE_CAPTURE,
            original_name="scan_00027",
            path=path,
            format="",
            sha256="a" * 64,
            session_id="s1",
            status=SourceStatus.INGESTED,
            components=(
                SourceComponent(role="image", path=path, sha256="b" * 64, size_bytes=64),
                SourceComponent(role="imu", path=path + ".csv", sha256="c" * 64),
            ),
            asset_ids=("ev-1",),
            warnings=("w",),
            errors=(),
            metadata={"platform_note": "drone run 7"},
        )
        assert Source.from_dict(source.to_dict()) == source

    def test_source_rejects_unknown_format_version(self):
        with pytest.raises(ValueError):
            Source.from_dict({"format_version": 99})

    def test_registry_round_trip_preserves_dedupe_index(self, tmp_path):
        path = _make_file(tmp_path)
        registry = SourceRegistry(session_id="s1")
        source, _ = registry.register(path, source_type=SourceType.IMAGE, sha256=hash_file(path))
        revived = SourceRegistry.from_dict(registry.to_dict())
        assert revived.get(source.source_id) == source
        source2, created = revived.register(path, source_type=SourceType.IMAGE, sha256=hash_file(path))
        assert created is False
        assert source2.source_id == source.source_id


class TestClassification:
    def test_file_types(self, tmp_path):
        assert classify_source(str(tmp_path / "x.mp4")) is SourceType.VIDEO
        assert classify_source(str(tmp_path / "x.jpg")) is SourceType.IMAGE
        assert classify_source(str(tmp_path / "x.las")) is SourceType.POINT_CLOUD
        assert classify_source(str(tmp_path / "imu.csv")) is SourceType.SENSOR_LOG
        assert classify_source(str(tmp_path / "weird.xyz")) is SourceType.OTHER

    def test_directory_is_dataset(self, tmp_path):
        d = tmp_path / "cap"
        d.mkdir()
        assert classify_source(str(d)) is SourceType.DATASET
