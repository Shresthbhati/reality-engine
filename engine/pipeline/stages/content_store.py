"""Content-addressed blob store under STORAGE_ROOT (engine layer; ``apps.api.storage`` re-exports it).

Lives in ``engine.pipeline`` so pipeline stages can persist artifacts without importing ``apps.api`` (the layer rule forbids
that direction). Blobs are stored by sha256 at ``<root>/<h[:2]>/<h[2:4]>/<h>`` and referenced as ``sha256://<h>``; raw paths are
never given to clients.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path


def storage_root() -> Path:
    # Always absolute: store_bytes() and resolve_artifact() must agree on the layout, and a relative root made them
    # disagree (stored path relative vs resolved absolute), breaking round-trip equality.
    root = os.environ.get("STORAGE_ROOT")
    base = Path(root) if root else Path("./data/artifacts")
    return base.resolve()


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
    """Resolve a sha256:// URI to a file path, validating the layout. None for unknown/unresolvable artifacts.

    Path traversal protection: the resolved path must stay inside STORAGE_ROOT (Path.resolve() + relative_to)."""
    if not uri.startswith("sha256://"):
        return None
    digest = uri.removeprefix("sha256://")
    if not digest or any(c not in "0123456789abcdef" for c in digest):
        return None
    root = storage_root().resolve()
    path = (root / digest[:2] / digest[2:4] / digest).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        return None
    if not path.exists():
        return None
    return path
