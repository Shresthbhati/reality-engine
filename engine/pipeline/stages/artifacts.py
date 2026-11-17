"""Content-addressed artifact storage helpers for pipeline stages (B01).

Implements put_bytes, get_bytes, put_array, get_array, put_json, get_json, put_npz, get_npz.
Reuses the store behind apps.api.storage.store_bytes.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any
import numpy as np

from apps.api.storage import resolve_artifact, store_bytes


def _uri(sha: str) -> str:
    return sha if sha.startswith("sha256://") else f"sha256://{sha}"


def put_bytes(data: bytes) -> str:
    digest, _ = store_bytes(data)
    return digest


def get_bytes(sha: str) -> bytes:
    path = resolve_artifact(_uri(sha))
    if path is None:
        raise FileNotFoundError(f"Artifact {sha} not found in storage")
    return path.read_bytes()


def put_json(obj: Any) -> str:
    data = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return put_bytes(data)


def get_json(sha: str) -> Any:
    return json.loads(get_bytes(sha).decode("utf-8"))


def put_array(arr: np.ndarray) -> str:
    bio = io.BytesIO()
    np.save(bio, arr, allow_pickle=False)
    return put_bytes(bio.getvalue())


def get_array(sha: str, mmap: bool = True) -> np.ndarray:
    path = resolve_artifact(_uri(sha))
    if path is None:
        raise FileNotFoundError(f"Artifact {sha} not found in storage")
    return np.load(path, mmap_mode="r" if mmap else None, allow_pickle=False)


def put_npz(**arrays: np.ndarray) -> str:
    bio = io.BytesIO()
    np.savez_compressed(bio, **arrays)
    return put_bytes(bio.getvalue())


def get_npz(sha: str) -> np.lib.npyio.NpzFile:
    path = resolve_artifact(_uri(sha))
    if path is None:
        raise FileNotFoundError(f"Artifact {sha} not found in storage")
    return np.load(path, allow_pickle=False)
