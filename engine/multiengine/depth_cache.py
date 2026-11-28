"""Content-addressed cache of raw depth maps.

A depth map is a pure function of (engine, checkpoint weights, image bytes), so it can be reused -- but ONLY when all three are
known exactly. The key therefore needs the checkpoint's SHA-256: an engine whose weights hash is unknown is never cached (a changed
or different checkpoint would silently reuse a stale map). The cache lives under the world's own directory (tenant-scoped) and every
read re-verifies the stored bytes against their content address, so a damaged entry is a miss, never a wrong depth map.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
from pathlib import Path

import numpy as np

from world_ir.artifact_store import FileArtifactStore

_SHA = re.compile(r"^[0-9a-f]{64}$")

#: Version of the MEANING of a cached depth array. Bump it whenever a depth backend changes what its numbers are
#: (units, depth vs inverse depth, normalisation, hole encoding), because the key (engine, weights, image bytes) does not
#: change when only the output convention does and a stale map would then be served under the new contract.
#: v2 = 156f3e36: Depth Anything passes raw inverse depth through (no per-image min-max), holes are NaN + valid_mask.
DEPTH_CONVENTION = "v2"


def file_sha256(path: str | os.PathLike[str]) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class DepthCache:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self._index_path = self.root / "index.json"
        self._store = FileArtifactStore(self.root / "artifacts")

    @staticmethod
    def key(engine_id: str, checkpoint_sha256: str | None, image_sha256: str) -> str | None:
        """``None`` (= not cacheable) unless the checkpoint hash and the image hash are both real SHA-256 digests."""
        if not (checkpoint_sha256 and _SHA.match(checkpoint_sha256) and _SHA.match(image_sha256)):
            return None
        return f"{engine_id}|{DEPTH_CONVENTION}|{checkpoint_sha256}|{image_sha256}"

    def _index(self) -> dict[str, str]:
        try:
            return json.loads(self._index_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def get(self, key: str) -> np.ndarray | None:
        uri = self._index().get(key)
        if uri is None:
            return None
        try:
            data = self._store.get(uri)
            if hashlib.sha256(data).hexdigest() != uri.removeprefix("artifact://"):  # content address must still match
                return None
            arr = np.load(io.BytesIO(data), allow_pickle=False)
            return arr if arr.ndim == 2 and np.isfinite(arr).all() else None
        except Exception:  # noqa: BLE001 -- unreadable / damaged entry: a miss
            return None

    def put(self, key: str, array: np.ndarray) -> None:
        buf = io.BytesIO()
        np.save(buf, np.asarray(array), allow_pickle=False)
        uri, _ = self._store.put(buf.getvalue())
        idx = self._index()
        idx[key] = uri
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self._index_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(idx, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self._index_path)
