"""A persistent COLMAP workspace for one world: the state incremental registration builds on.

Without this, every rebuild starts from nothing: features are re-extracted, every pair is
re-matched and the mapper solves the whole scene again, so the camera frame and every pose
can change between versions. With it, a new version can REGISTER new photographs into the
previous sparse model (``colmap image_registrator``) and keep what was already established.

Layout under ``root``::

    current/    the COMMITTED state: database.db (features + matches), images/, sparse/<n>/, manifest.json
    staging/    a working copy for the run in progress (never read as authoritative)

Lifecycle (this is what keeps a rejected candidate from poisoning later runs):

    begin()    discard any stale staging; copy ``current`` into staging when it is usable
    ...        the backend mutates staging only
    commit()   called by the job AFTER the new version was adopted and read back -> staging becomes current
    discard()  called on rejection / failure -> staging is deleted, ``current`` is untouched

Nothing here runs COLMAP; it only owns directories and the manifest.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

MANIFEST = "manifest.json"
MANIFEST_VERSION = 1


class ColmapSession:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.current = self.root / "current"
        self.staging = self.root / "staging"
        #: facts about the last run, filled by the backend (mode, counts, timings)
        self.last_run_info: Dict[str, object] = {}

    # ---------------------------------------------------------------- reading
    @staticmethod
    def _read(directory: Path) -> Optional[dict]:
        path = directory / MANIFEST
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) and data.get("version") == MANIFEST_VERSION else None

    def committed_manifest(self) -> Optional[dict]:
        return self._read(self.current)

    # --------------------------------------------------------------- lifecycle
    def begin(self, signature: dict, wanted_image_names: Iterable[str]) -> Tuple[Path, Optional[dict]]:
        """Prepare staging. Returns (staging_dir, prior_manifest_or_None).

        The prior state is used only when ALL of these hold; otherwise staging starts empty and the
        backend does a full reconstruction (the honest fallback, never a guess):
          * the committed manifest exists and was written by this format version
          * the pipeline signature (SIFT flavour, matching mode, pinned intrinsics) is unchanged
          * every image in the committed model is still part of the evidence
          * the committed model exists on disk and may be extended (``incremental_ok``)
        """
        self.discard()
        self.root.mkdir(parents=True, exist_ok=True)
        manifest = self._read(self.current)
        wanted = set(wanted_image_names)
        usable = (
            manifest is not None
            and manifest.get("signature") == signature
            and manifest.get("incremental_ok", False)
            and set(manifest.get("images", {})) <= wanted
            and (self.current / "database.db").is_file()
            and (self.current / str(manifest.get("model_dir", ""))).is_dir()
        )
        if usable:
            shutil.copytree(self.current, self.staging)
            # A manifest in staging must mean "the run that wrote it FINISHED". The copy carries the
            # committed one, which would make a crashed run look complete and committable.
            (self.staging / MANIFEST).unlink(missing_ok=True)
            return self.staging, manifest
        self.staging.mkdir(parents=True)
        return self.staging, None

    def write_manifest(self, manifest: dict) -> None:
        """Atomically write the manifest into staging (called by the backend on success)."""
        body = dict(manifest, version=MANIFEST_VERSION)
        tmp = self.staging / (MANIFEST + ".tmp")
        tmp.write_text(json.dumps(body, indent=1, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self.staging / MANIFEST)

    def commit(self) -> bool:
        """Make staging the committed state. False when there is nothing valid to commit."""
        if self._read(self.staging) is None:
            self.discard()
            return False
        previous = self.root / "previous"
        shutil.rmtree(previous, ignore_errors=True)
        if self.current.exists():
            os.replace(self.current, previous)
        os.replace(self.staging, self.current)
        shutil.rmtree(previous, ignore_errors=True)
        return True

    def discard(self) -> None:
        shutil.rmtree(self.staging, ignore_errors=True)
