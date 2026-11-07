"""T-6.3: the S3-compatible artifact store behaves exactly like the local one, and local stays the default.

A fake in-memory S3 client stands in for boto3 (no network, no moto). It raises botocore's real ClientError for missing keys,
so the "absent vs. real fault" handling is exercised against the genuine exception shape.
"""
from __future__ import annotations

import hashlib
import io

import pytest

botocore_exc = pytest.importorskip("botocore.exceptions")

from world_ir.artifact_store import (  # noqa: E402
    ArtifactNotFoundError, FileArtifactStore, S3ArtifactStore, make_artifact_store)


class FakeS3:
    def __init__(self):
        self.objects: dict[str, bytes] = {}
        self.puts = 0
        self.fail_head_with: str | None = None

    def _err(self, code, op):
        return botocore_exc.ClientError({"Error": {"Code": code, "Message": code}}, op)

    def head_object(self, Bucket, Key):
        if self.fail_head_with:
            raise self._err(self.fail_head_with, "HeadObject")
        if Key not in self.objects:
            raise self._err("404", "HeadObject")
        return {}

    def put_object(self, Bucket, Key, Body):
        self.puts += 1
        self.objects[Key] = Body

    def get_object(self, Bucket, Key):
        if Key not in self.objects:
            raise self._err("NoSuchKey", "GetObject")
        return {"Body": io.BytesIO(self.objects[Key])}


def test_put_get_round_trip_uses_the_same_uri_as_the_local_store(tmp_path):
    s3, local = S3ArtifactStore("b", FakeS3(), "tenant/"), FileArtifactStore(tmp_path)
    uri, digest = s3.put(b"wall geometry")
    assert (uri, digest) == local.put(b"wall geometry")            # swapping stores never changes a data_uri
    assert s3.get(uri) == b"wall geometry" and s3.has(digest)
    assert list(s3.client.objects) == [f"tenant/{digest[:2]}/{digest}.bin"]


def test_identical_bytes_are_uploaded_once():
    s3 = S3ArtifactStore("b", FakeS3())
    s3.put(b"x"), s3.put(b"x"), s3.put(b"x")
    assert s3.client.puts == 1


def test_unknown_artifact_is_not_found_and_a_bad_uri_is_refused():
    s3 = S3ArtifactStore("b", FakeS3())
    with pytest.raises(ArtifactNotFoundError):
        s3.get("artifact://" + "0" * 64)
    assert not s3.has("0" * 64)
    for bad in ("artifact://../../etc/passwd", "artifact://" + "A" * 64, "http://x"):
        with pytest.raises(ValueError):
            s3.get(bad)


def test_a_swapped_or_corrupt_object_reads_as_missing_never_as_wrong_bytes():
    s3 = S3ArtifactStore("b", FakeS3())
    uri, digest = s3.put(b"good")
    s3.client.objects[s3._key(digest)] = b"tampered"
    with pytest.raises(ArtifactNotFoundError):
        s3.get(uri)


def test_a_real_s3_fault_is_not_mistaken_for_absence():
    s3 = S3ArtifactStore("b", FakeS3())
    s3.client.fail_head_with = "AccessDenied"
    with pytest.raises(botocore_exc.ClientError):
        s3.has("0" * 64)
    with pytest.raises(botocore_exc.ClientError):
        s3.put(b"never silently skipped")


def test_the_factory_defaults_to_local_and_builds_s3_from_env(tmp_path):
    assert isinstance(make_artifact_store(tmp_path, env={}), FileArtifactStore)
    with pytest.raises(ValueError, match="REALITY_S3_BUCKET"):
        make_artifact_store(tmp_path, env={"REALITY_ARTIFACT_STORE": "s3"})
    store = make_artifact_store(tmp_path, env={"REALITY_ARTIFACT_STORE": "S3", "REALITY_S3_BUCKET": "reality-prod",
                                               "REALITY_S3_ENDPOINT": "http://localhost:9000", "REALITY_S3_PREFIX": "p",
                                               "REALITY_S3_REGION": "us-east-1"})
    assert isinstance(store, S3ArtifactStore) and store.bucket == "reality-prod" and store.prefix == "p/"
