"""Content-addressed artifact storage for application uploads.

Files are stored by sha256 under STORAGE_ROOT. Raw filesystem paths are
never exposed to clients; artifacts are referenced by URI of the form
``sha256://<hash>`` and served through controlled API routes.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path


def storage_root() -> Path:
    root = os.environ.get("STORAGE_ROOT")
    if root:
        return Path(root)
    return Path("./data/artifacts")


def store_bytes(data: bytes) -> tuple[str, Path]:
    """Store bytes content-addressed. Returns (hash, absolute path)."""
    digest = hashlib.sha256(data).hexdigest()
    root = storage_root()
    obj_dir = root / digest[:2] / digest[2:4]
    obj_dir.mkdir(parents=True, exist_ok=True)
    path = obj_dir / digest
    if not path.exists():
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)
    return digest, path


def resolve_artifact(uri: str) -> Path | None:
    """Resolve a sha256:// URI to a file path, validating the layout.
    Returns None for unknown/unresolvable artifacts."""
    if not uri.startswith("sha256://"):
        return None
    digest = uri.removeprefix("sha256://")
    if not digest or any(c not in "0123456789abcdef" for c in digest):
        return None
    path = storage_root() / digest[:2] / digest[2:4] / digest
    if not path.exists():
        return None
    return path