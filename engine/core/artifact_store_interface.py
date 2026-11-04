"""Abstract artifact store interface for the core layer.

Both `provenance` (core) and `world_ir.artifact_store` (peer layer)
depend on this interface. This lets `provenance.graph` verify and
query artifacts without importing the concrete `world_ir` layer,
preserving the import boundary: core must not import above it.

The concrete `world_ir.artifact_store.FileArtifactStore` implements
this interface (see `world_ir/artifact_store.py`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class ArtifactNotFoundError(FileNotFoundError):
    """Raised when an artifact is not found in the store."""


class ArtifactStoreInterface(ABC):
    """Content-addressed artifact store: get bytes by content URI."""

    @abstractmethod
    def get(self, uri: str) -> bytes:
        """Return the raw bytes stored at ``uri``.

        Raises `ArtifactNotFoundError` when nothing is stored at ``uri``.
        """

    @abstractmethod
    def uri_for(self, digest: str) -> str:
        """Return the content URI for a digest (sha256 hex)."""

    @abstractmethod
    def has(self, digest: str) -> bool:
        """True when the store holds an artifact with this digest."""
