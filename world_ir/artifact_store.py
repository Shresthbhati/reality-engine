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
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict


class ArtifactNotFoundError(KeyError):
    pass


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class ArtifactStore(ABC):
    """Content-addressed store: `put(data)` returns a `data_uri` string
    (`artifact://<sha256>`) that `get()` resolves back to the same
    bytes. The digest is also handed back separately so callers can set
    `Geometry.data_hash` without re-parsing the uri."""

    @abstractmethod
    def put(self, data: bytes) -> "tuple[str, str]":
        """Store `data`, returning (data_uri, sha256_hex)."""

    @abstractmethod
    def get(self, data_uri: str) -> bytes:
        """Resolve a data_uri back to its bytes. Raises
        ArtifactNotFoundError if unknown to this store."""

    @staticmethod
    def uri_for(digest: str) -> str:
        return f"artifact://{digest}"

    @staticmethod
    def digest_of(data_uri: str) -> str:
        if not data_uri.startswith("artifact://"):
            raise ValueError(f"not an artifact:// uri: {data_uri!r}")
        return data_uri[len("artifact://"):]


class MemoryArtifactStore(ArtifactStore):
    def __init__(self):
        self._blobs: Dict[str, bytes] = {}

    def put(self, data: bytes) -> "tuple[str, str]":
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


class FileArtifactStore(ArtifactStore):
    """Sharded on-disk store: `<root>/<digest[:2]>/<digest>.bin`, the
    same sharding convention as Git's object store (avoids one huge flat
    directory for large worlds)."""

    def __init__(self, root: "str | Path"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path_for(self, digest: str) -> Path:
        return self.root / digest[:2] / f"{digest}.bin"

    def put(self, data: bytes) -> "tuple[str, str]":
        digest = _digest(data)
        path = self._path_for(digest)
        if not path.exists():  # content-addressed: identical bytes, no rewrite
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        return self.uri_for(digest), digest

    def get(self, data_uri: str) -> bytes:
        digest = self.digest_of(data_uri)
        path = self._path_for(digest)
        if not path.exists():
            raise ArtifactNotFoundError(data_uri)
        return path.read_bytes()
