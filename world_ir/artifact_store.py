"""Minimal content-addressed geometry artifact store (P0.10).

The thing `Geometry.data_uri`/`data_hash` (world_ir/schema_v1.py) were
designed to reference but nothing wrote until now. Content-addressed so
storing the same bytes twice (e.g. recompiling the same reconstruction
with the same seed) reuses the same artifact rather than duplicating it
-- the same determinism discipline the rest of the compiler already
follows.

Two backends behind one interface: `MemoryArtifactStore` (default,
process-local, for tests and short-lived sessions) and
`FileArtifactStore` (a real on-disk store, for anything that needs to
survive the process). Neither talks to a network -- artifacts are
supplied by the caller, never fetched.
"""

from __future__ import annotations

import hashlib
import os
import re
import uuid
from abc import abstractmethod
from pathlib import Path
from typing import Any

from engine.core.artifact_store_interface import (
    ArtifactNotFoundError as CoreArtifactNotFoundError,
)
from engine.core.artifact_store_interface import (
    ArtifactStoreInterface,
)


class ArtifactNotFoundError(KeyError, CoreArtifactNotFoundError):
    """Raised when an artifact is not found in the store."""


#: sha256 hex digest: exactly 64 lowercase hex characters. `digest_of()`
#: enforces this on every incoming data_uri -- a `data_uri` is untrusted
#: input (it round-trips through WorldIR JSON, which can come from
#: anywhere), and `FileArtifactStore._path_for()` joins the digest
#: directly onto its root path. Without this check, an artifact://../../
#: (or an absolute-path-like) digest would let get() read arbitrary
#: files outside the store root -- a real path-traversal vulnerability,
#: not a hypothetical one.
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class ArtifactStore(ArtifactStoreInterface):
    """Content-addressed store: `put(data)` returns a `data_uri` string
    (`artifact://<sha256>`) that `get()` resolves back to the same
    bytes. The digest is also handed back separately so callers can set
    `Geometry.data_hash` without re-parsing the uri."""

    @abstractmethod
    def put(self, data: bytes) -> tuple[str, str]:
        """Store `data`, returning (data_uri, sha256_hex)."""

    @abstractmethod
    def get(self, data_uri: str) -> bytes:
        """Resolve a data_uri back to its bytes. Raises
        ArtifactNotFoundError if unknown to this store."""

    @abstractmethod
    def has(self, digest: str) -> bool:
        """True when the store holds an artifact with this digest."""

    @staticmethod
    def uri_for(digest: str) -> str:
        return f"artifact://{digest}"

    @staticmethod
    def digest_of(data_uri: str) -> str:
        if not data_uri.startswith("artifact://"):
            raise ValueError(f"not an artifact:// uri: {data_uri!r}")
        digest = data_uri[len("artifact://"):]
        if not _DIGEST_RE.match(digest):
            # Never a valid sha256 hex digest -- most importantly, never
            # something a path traversal (../, an absolute path, a
            # symlink-style segment) could resolve outside the store
            # root. Refuse before either backend touches the filesystem
            # or its internal dict with this value.
            raise ValueError(
                f"not a valid sha256 hex digest: {digest!r} (from data_uri {data_uri!r})"
            )
        return digest


class MemoryArtifactStore(ArtifactStore):
    def __init__(self):
        self._blobs: dict[str, bytes] = {}

    def put(self, data: bytes) -> tuple[str, str]:
        digest = _digest(data)
        self._blobs[digest] = data
        return self.uri_for(digest), digest

    def get(self, data_uri: str) -> bytes:
        digest = self.digest_of(data_uri)
        try:
            return self._blobs[digest]
        except KeyError as exc:
            raise ArtifactNotFoundError(data_uri) from exc

    def __len__(self) -> int:
        return len(self._blobs)

    def has(self, digest: str) -> bool:
        return digest in self._blobs


class FileArtifactStore(ArtifactStore):
    """Sharded on-disk store: `<root>/<digest[:2]>/<digest>.bin`, the
    same sharding convention as Git's object store (avoids one huge flat
    directory for large worlds)."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path_for(self, digest: str) -> Path:
        return self.root / digest[:2] / f"{digest}.bin"

    def put(self, data: bytes) -> tuple[str, str]:
        digest = _digest(data)
        path = self._path_for(digest)
        if path.is_symlink():
            # A planted symlink here would make write_bytes follow the
            # link and write outside the store: drop the link, then
            # write the real blob. Local tamper never escapes the root.
            path.unlink()
        # Content-addressed: identical intact bytes are not rewritten. An existing blob that no longer hashes to its
        # name (a torn write from a disk-full/crash, or later corruption) is NOT trusted -- it is replaced, or every
        # retry of the same content would point at a rotten blob forever.
        # ponytail: re-hashes an existing blob on duplicate puts; fine for world-sized payloads.
        if not path.exists() or _digest(path.read_bytes()) != digest:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(f"{path.name}.tmp-{uuid.uuid4().hex[:8]}")
            try:
                tmp.write_bytes(data)          # a torn write only ever damages the temp file
                os.replace(tmp, path)          # atomic: readers see the old blob or the whole new one
            finally:
                tmp.unlink(missing_ok=True)
        return self.uri_for(digest), digest

    def get(self, data_uri: str) -> bytes:
        digest = self.digest_of(data_uri)
        path = self._path_for(digest)
        if path.is_symlink() or not path.exists():
            # Never follow a symlink out of the store: a planted link is
            # surfaced as missing (honest failure downstream), the same
            # way a corrupt blob surfaces as a hash mismatch.
            raise ArtifactNotFoundError(data_uri)
        return path.read_bytes()

    def has(self, digest: str) -> bool:
        path = self._path_for(digest)
        return path.is_file() and not path.is_symlink()


class S3ArtifactStore(ArtifactStore):
    """S3-compatible content-addressed artifact store (AWS S3, MinIO, Cloudflare R2).

    T-6.3: Integrates seamlessly behind the ArtifactStore interface.
    Local storage stays default; S3 is used when S3_BUCKET_NAME / REALITY_STORAGE_BACKEND=s3 is set.
    """

    def __init__(
        self,
        bucket: str,
        *,
        endpoint_url: str | None = None,
        region_name: str | None = None,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        prefix: str = "artifacts",
        s3_client: Any | None = None,
    ):
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        if s3_client is not None:
            self.s3_client = s3_client
        else:
            import boto3

            session = boto3.session.Session()
            kwargs: dict[str, Any] = {}
            if endpoint_url:
                kwargs["endpoint_url"] = endpoint_url
            if region_name:
                kwargs["region_name"] = region_name
            if aws_access_key_id and aws_secret_access_key:
                kwargs["aws_access_key_id"] = aws_access_key_id
                kwargs["aws_secret_access_key"] = aws_secret_access_key
            self.s3_client = session.client("s3", **kwargs)

    def _key_for(self, digest: str) -> str:
        if self.prefix:
            return f"{self.prefix}/{digest[:2]}/{digest}.bin"
        return f"{digest[:2]}/{digest}.bin"

    def put(self, data: bytes) -> tuple[str, str]:
        digest = _digest(data)
        key = self._key_for(digest)
        # Content-addressed: check if intact object already exists to avoid redundant upload
        if not self.has(digest):
            self.s3_client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=data,
                ContentType="application/octet-stream",
            )
        return self.uri_for(digest), digest

    def get(self, data_uri: str) -> bytes:
        digest = self.digest_of(data_uri)
        key = self._key_for(digest)
        try:
            resp = self.s3_client.get_object(Bucket=self.bucket, Key=key)
            data = resp["Body"].read()
        except Exception as exc:
            code = ""
            if hasattr(exc, "response") and isinstance(exc.response, dict):
                code = str(exc.response.get("Error", {}).get("Code", ""))
            if code in ("NoSuchKey", "404", "NotFound") or "not found" in str(exc).lower():
                raise ArtifactNotFoundError(data_uri) from exc
            raise ArtifactNotFoundError(data_uri) from exc

        # Integrity check: verify content matches hash
        if _digest(data) != digest:
            raise ArtifactNotFoundError(f"corrupt artifact: hash mismatch for {data_uri}")
        return data

    def has(self, digest: str) -> bool:
        if not _DIGEST_RE.match(digest):
            return False
        key = self._key_for(digest)
        try:
            self.s3_client.head_object(Bucket=self.bucket, Key=key)
            return True
        except Exception as exc:
            code = ""
            if hasattr(exc, "response") and isinstance(exc.response, dict):
                code = str(exc.response.get("Error", {}).get("Code", ""))
            if code in ("NoSuchKey", "404", "NotFound") or "not found" in str(exc).lower():
                return False
            return False

    def delete(self, digest_or_uri: str) -> bool:
        if digest_or_uri.startswith("artifact://"):
            digest = self.digest_of(digest_or_uri)
        else:
            digest = digest_or_uri
        key = self._key_for(digest)
        try:
            self.s3_client.delete_object(Bucket=self.bucket, Key=key)
            return True
        except Exception:
            return False


def create_artifact_store(
    store_type: str | None = None,
    *,
    root: str | Path | None = None,
    bucket: str | None = None,
    endpoint_url: str | None = None,
    region_name: str | None = None,
    aws_access_key_id: str | None = None,
    aws_secret_access_key: str | None = None,
    prefix: str = "artifacts",
) -> ArtifactStore:
    """Factory creating an ArtifactStore. Local filesystem stays default."""
    selected_type = (
        store_type
        or os.environ.get("REALITY_STORAGE_BACKEND")
        or ("s3" if (bucket or os.environ.get("S3_BUCKET_NAME") or os.environ.get("REALITY_S3_BUCKET")) else "file")
    ).lower()

    if selected_type == "s3":
        bkt = bucket or os.environ.get("S3_BUCKET_NAME") or os.environ.get("REALITY_S3_BUCKET") or "reality-artifacts"
        endpoint = endpoint_url or os.environ.get("S3_ENDPOINT_URL") or os.environ.get("AWS_ENDPOINT_URL")
        region = region_name or os.environ.get("AWS_REGION") or os.environ.get("S3_REGION") or "us-east-1"
        key_id = aws_access_key_id or os.environ.get("AWS_ACCESS_KEY_ID") or os.environ.get("S3_ACCESS_KEY_ID")
        secret_key = aws_secret_access_key or os.environ.get("AWS_SECRET_ACCESS_KEY") or os.environ.get("S3_SECRET_ACCESS_KEY")
        return S3ArtifactStore(
            bucket=bkt,
            endpoint_url=endpoint,
            region_name=region,
            aws_access_key_id=key_id,
            aws_secret_access_key=secret_key,
            prefix=prefix,
        )
    elif selected_type == "memory":
        return MemoryArtifactStore()
    else:
        root_path = root or os.environ.get("STORAGE_ROOT") or os.environ.get("WORLDSTORE_ROOT") or "./data/artifacts"
        return FileArtifactStore(root_path)
