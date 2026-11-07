"""S3 Storage Adapter Test Suite (T-6.3).

Verifies:
1. S3ArtifactStore implements the ArtifactStore interface seamlessly.
2. Content-addressed storage: put, get, has, delete against an S3-compatible backend (MinIO / R2 / S3).
3. Hash integrity verification and tamper detection.
4. Path-traversal prevention on incoming untrusted data_uri.
5. create_artifact_store factory function (local file store default preserved; S3 selected via env/param).
6. End-to-end integration over real HTTP using an in-process S3 REST protocol server.
"""

from __future__ import annotations

import http.server
import threading

import pytest

pytest.importorskip("boto3")

from world_ir.artifact_store import (
    ArtifactNotFoundError,
    FileArtifactStore,
    MemoryArtifactStore,
    S3ArtifactStore,
    create_artifact_store,
)


class MockS3Handler(http.server.BaseHTTPRequestHandler):
    """In-process HTTP server implementing minimal S3 REST protocol (MinIO/R2 compatible)."""

    blobs: dict[str, bytes] = {}

    def log_message(self, format, *args):
        pass  # Suppress noisy HTTP request logging during tests

    def do_PUT(self):
        content_length = int(self.headers.get("Content-Length", 0))
        data = self.rfile.read(content_length)
        key = self.path.lstrip("/")
        MockS3Handler.blobs[key] = data
        self.send_response(200)
        self.send_header("ETag", f'"{hash(data)}"')
        self.end_headers()

    def do_GET(self):
        key = self.path.lstrip("/")
        if key in MockS3Handler.blobs:
            data = MockS3Handler.blobs[key]
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        else:
            self.send_response(404)
            self.send_header("Content-Type", "application/xml")
            self.end_headers()
            self.wfile.write(b"<Error><Code>NoSuchKey</Code><Message>The specified key does not exist.</Message></Error>")

    def do_HEAD(self):
        key = self.path.lstrip("/")
        if key in MockS3Handler.blobs:
            self.send_response(200)
            self.send_header("Content-Length", str(len(MockS3Handler.blobs[key])))
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def do_DELETE(self):
        key = self.path.lstrip("/")
        MockS3Handler.blobs.pop(key, None)
        self.send_response(204)
        self.end_headers()


@pytest.fixture(scope="module")
def mock_s3_server():
    """Start an in-process S3-compatible HTTP server on an ephemeral local port."""
    MockS3Handler.blobs.clear()
    server = http.server.HTTPServer(("127.0.0.1", 0), MockS3Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()
    server.server_close()


@pytest.fixture
def s3_store(mock_s3_server) -> S3ArtifactStore:
    MockS3Handler.blobs.clear()
    return S3ArtifactStore(
        bucket="test-reality-bucket",
        endpoint_url=mock_s3_server,
        region_name="us-east-1",
        aws_access_key_id="mock-minio-key",
        aws_secret_access_key="mock-minio-secret",
        prefix="artifacts",
    )


def test_s3_artifact_store_put_and_get(s3_store: S3ArtifactStore):
    payload = b"test point cloud geometry binary data xyz 123456"
    uri, digest = s3_store.put(payload)

    assert uri.startswith("artifact://")
    assert uri == f"artifact://{digest}"
    assert s3_store.has(digest) is True

    # Read back exact bytes
    read_back = s3_store.get(uri)
    assert read_back == payload


def test_s3_artifact_store_content_addressed_idempotency(s3_store: S3ArtifactStore):
    payload = b"repeatable room mesh payload"
    uri1, digest1 = s3_store.put(payload)
    uri2, digest2 = s3_store.put(payload)

    assert uri1 == uri2
    assert digest1 == digest2


def test_s3_artifact_store_missing_raises_artifact_not_found(s3_store: S3ArtifactStore):
    unknown_digest = "a" * 64
    assert s3_store.has(unknown_digest) is False

    with pytest.raises(ArtifactNotFoundError):
        s3_store.get(f"artifact://{unknown_digest}")


def test_s3_artifact_store_path_traversal_rejection(s3_store: S3ArtifactStore):
    """Malicious data_uri with path traversal or invalid characters is rejected before network call."""
    with pytest.raises(ValueError):
        s3_store.get("artifact://../../etc/passwd")

    with pytest.raises(ValueError):
        s3_store.get("artifact://not-a-valid-sha256-hex")

    assert s3_store.has("../../bad/path") is False


def test_s3_artifact_store_corrupted_blob_raises_error(s3_store: S3ArtifactStore):
    """If S3 payload does not match the content digest, it is rejected as corrupt."""
    payload = b"original uncorrupted data"
    uri, digest = s3_store.put(payload)

    # Directly tamper with the raw blob on S3
    key = f"test-reality-bucket/artifacts/{digest[:2]}/{digest}.bin"
    MockS3Handler.blobs[key] = b"tampered garbage bytes"

    with pytest.raises(ArtifactNotFoundError) as exc_info:
        s3_store.get(uri)
    assert "corrupt artifact: hash mismatch" in str(exc_info.value)


def test_s3_artifact_store_delete(s3_store: S3ArtifactStore):
    payload = b"temporary mesh payload"
    uri, digest = s3_store.put(payload)
    assert s3_store.has(digest) is True

    # Delete via uri
    assert s3_store.delete(uri) is True
    assert s3_store.has(digest) is False

    # Second delete is idempotent
    assert s3_store.delete(digest) is True


def test_create_artifact_store_factory_defaults_to_local_file(tmp_path, monkeypatch):
    monkeypatch.delenv("REALITY_STORAGE_BACKEND", raising=False)
    monkeypatch.delenv("S3_BUCKET_NAME", raising=False)
    monkeypatch.delenv("REALITY_S3_BUCKET", raising=False)

    store = create_artifact_store(root=tmp_path / "artifacts")
    assert isinstance(store, FileArtifactStore)


def test_create_artifact_store_factory_selects_memory():
    store = create_artifact_store(store_type="memory")
    assert isinstance(store, MemoryArtifactStore)


def test_create_artifact_store_factory_selects_s3_via_env(mock_s3_server, monkeypatch):
    monkeypatch.setenv("REALITY_STORAGE_BACKEND", "s3")
    monkeypatch.setenv("S3_BUCKET_NAME", "my-env-bucket")
    monkeypatch.setenv("S3_ENDPOINT_URL", mock_s3_server)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")

    store = create_artifact_store()
    assert isinstance(store, S3ArtifactStore)
    assert store.bucket == "my-env-bucket"

    # Verify functionality
    uri, digest = store.put(b"hello from env configured s3 store")
    assert store.get(uri) == b"hello from env configured s3 store"
