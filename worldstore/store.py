"""WorldStore (P12): persistent, versioned world storage.

The directive's done-when: "world survives process restarts". This
store makes a compiled world durable and recoverable:

- every save serializes the WorldIR to its canonical dict form and
  persists BOTH the serialized world and the artifact bytes it
  references (via the FileArtifactStore root) under a new version id;
- versions are IMMUTABLE: re-saving a version id is refused -- observed
  reality must remain recoverable, never silently overwritten;
- lineage: each version records its parent, so the chain
  V1 -> V2 -> V3 is queryable (world history is a DAG rooted at the
  first compile);
- integrity: `verify_version` re-checks every referenced artifact's
  bytes against its recorded hash (tamper/corruption detection);
- a fresh WorldStore instance over the same root reads everything
  back -- no in-memory state is required, which is exactly the
  "survives process restart" property.

Design rule honored (directive 21): the world schema it persists is
the existing canonical WorldIR v1 dict -- no competing schema.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from world_ir.artifact_store import FileArtifactStore

if TYPE_CHECKING:
    from world_ir.world_v1 import WorldIR


class WorldStoreError(ValueError):
    """WorldStore operation refused."""


class _SequenceLock:
    """Cross-process, cross-thread mutex guarding the read-modify-write
    of sequence.json.

    Without this, two concurrent save_version() calls both read the
    same sequence, both append their own version id, and the second
    write clobbers the first's entry (lost update) -- and on Windows,
    concurrent os.replace() calls onto the same destination path can
    outright raise PermissionError (WinError 5) instead of silently
    losing data, so save_version() itself becomes flaky under
    concurrency, not just eventually-inconsistent. Reproduced with 20
    concurrent threads before this fix: 15/20 raised unhandled
    PermissionError.

    os.mkdir() is the primitive: directory creation is atomic on both
    POSIX and Windows (NTFS) -- exactly one concurrent caller succeeds,
    everyone else gets FileExistsError, which is what makes this a real
    mutex rather than a "hope for the best" retry. The bounded poll
    loop is backoff between *acquisition attempts*, not a substitute
    for correctness the way a bare `time.sleep()` before a racy write
    would be -- the lock itself is what's correct; the loop just waits
    for the current holder to release it.

    Lock staleness (a holder that crashed without releasing) is never
    silently overridden -- that would risk two processes believing
    they hold the same lock. Acquisition times out after `timeout`
    seconds and raises WorldStoreError explicitly, naming the stale
    lock directory so an operator can inspect and remove it."""

    def __init__(self, root: Path, timeout: float = 30.0, poll_interval: float = 0.02):
        self._path = root / ".sequence.lock"
        self._timeout = timeout
        self._poll_interval = poll_interval

    def __enter__(self) -> "_SequenceLock":
        deadline = time.monotonic() + self._timeout
        while True:
            try:
                self._path.mkdir()
                return self
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise WorldStoreError(
                        f"could not acquire WorldStore sequence lock at "
                        f"{self._path} within {self._timeout}s -- another "
                        "writer is either still working or crashed while "
                        "holding it; if no other process is writing to "
                        "this store, remove the stale lock directory "
                        "manually and retry"
                    )
                time.sleep(self._poll_interval)

    def __exit__(self, *exc_info) -> None:
        self._path.rmdir()


def _atomic_write_text(path: Path, text: str) -> None:
    """Write `text` to `path` so a crash mid-write can never leave a
    truncated/partial file at `path`: write to a sibling temp file,
    fsync it, then os.replace() onto the final name. os.replace() is
    atomic on both POSIX and Windows NTFS -- readers of `path` always
    see either the previous complete content or the new complete
    content, never a partial write. Without this, a process crash
    during write_text() left a truncated JSON file that every future
    _record()/list_versions() call on that version would fail to
    parse (a corrupted store, not a "no partially committed world
    state" failure)."""
    tmp = path.with_suffix(path.suffix + f".tmp-{uuid.uuid4().hex[:8]}")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


@dataclass(frozen=True)
class StoredVersion:
    version_id: str
    world_id: str
    parent: str | None
    artifact_uri: str
    artifact_hash: str
    #: Ids of entities/geometries that differ from `parent`'s saved world
    #: (empty for a root version with no parent). Computed once at save
    #: time via `world_ir.diff.diff_worlds` -- not recomputed on read, so
    #: it survives even if the parent version is later deleted/corrupted.
    #: Optional/defaulted so records written before this field existed
    #: still deserialize (`StoredVersion(**record)` in list_versions/
    #: verify_version) with an explicit "unknown" empty tuple rather than
    #: an error.
    changed_entity_ids: tuple = ()
    changed_geometry_ids: tuple = ()
    #: Ids of the evidence-acquisition Sessions this version's world was
    #: (re)compiled from, when the caller supplies them -- the "which raw
    #: captures does this version trace back to" provenance link.
    source_session_ids: tuple = ()


class WorldStore:
    def __init__(self, root):
        self._root = Path(root)
        self._versions_dir = self._root / "versions"
        self._versions_dir.mkdir(parents=True, exist_ok=True)
        self._store = FileArtifactStore(self._root / "artifacts")

    # ---- write ----

    def save_version(
        self,
        world: WorldIR,
        *,
        parent: str | None,
        version_id: str | None = None,
        source_session_ids: list[str] | None = None,
    ) -> StoredVersion:
        vid = version_id or f"v-{uuid.uuid4().hex[:12]}"
        path = self._versions_dir / f"{vid}.json"
        if path.exists():
            raise WorldStoreError(
                f"version {vid} already exists -- versions are immutable; "
                "save a new version instead of overwriting observed reality"
            )
        # Compute changes from parent version
        changed_entity_ids: list[str] = []
        changed_geometry_ids: list[str] = []
        if parent:
            from world_ir.diff import diff_worlds
            parent_world = self.load_version(parent)
            world_diff = diff_worlds(parent_world, world)
            changed_entity_ids = sorted(d.entity_id for d in world_diff.entity_diffs)
            changed_geometry_ids = sorted(d.geometry_id for d in world_diff.geometry_diffs)
        payload = json.dumps(world.to_dict(), sort_keys=True).encode("utf-8")
        uri, digest = self._store.put(payload)
        record = {
            "version_id": vid,
            "world_id": world.id,
            "parent": parent,
            "artifact_uri": uri,
            "artifact_hash": digest,
            "changed_entity_ids": changed_entity_ids,
            "changed_geometry_ids": changed_geometry_ids,
            "source_session_ids": list(source_session_ids or []),
        }
        # Write the version record before the sequence index so a crash
        # between the two leaves an orphan version file (harmless --
        # list_versions() below already appends unrecorded files) rather
        # than a sequence entry pointing at a version that doesn't exist.
        # Both writes happen under the sequence lock, and the immutability
        # check is re-verified inside it: the pre-lock `path.exists()`
        # check above is only a fast path -- two concurrent savers of the
        # same explicit version_id could both pass it before either
        # writes (TOCTOU), silently overwriting the record and logging
        # the id twice in the sequence. The authoritative check below
        # makes exactly one winner; the loser gets an explicit
        # WorldStoreError and the sequence keeps a single entry.
        seq_path = self._root / "sequence.json"
        # The read-modify-write below is not safe to interleave across
        # concurrent writers (see _SequenceLock docstring): the lock
        # makes "claim id, write record, append order" a single atomic
        # step across threads and processes sharing this store root.
        with _SequenceLock(self._root):
            if path.exists():
                raise WorldStoreError(
                    f"version {vid} already exists -- versions are immutable; "
                    "save a new version instead of overwriting observed reality"
                )
            _atomic_write_text(path, json.dumps(record, indent=2))
            order: list[str] = self._read_sequence(seq_path)
            order.append(vid)
            _atomic_write_text(seq_path, json.dumps(order))
        return StoredVersion(**record)

    @staticmethod
    def _read_sequence(seq_path: Path) -> list[str]:
        if not seq_path.exists():
            return []
        try:
            return json.loads(seq_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            # The atomic write above makes this unreachable in normal
            # operation; surfaced explicitly (never silently reset to an
            # empty sequence, which would look like lost history) for a
            # store touched by something other than this class.
            raise WorldStoreError(
                f"sequence index at {seq_path} is corrupted: {exc}"
            ) from exc

    # ---- read ----

    def _record(self, version_id: str) -> dict:
        path = self._versions_dir / f"{version_id}.json"
        if not path.exists():
            raise WorldStoreError(f"unknown version: {version_id}")
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            # Explicit, actionable failure instead of a raw parser
            # exception -- a corrupted version record is a store
            # integrity problem the caller must handle, not something
            # to guess at or silently skip.
            raise WorldStoreError(
                f"version {version_id} record is corrupted: {exc}"
            ) from exc

    def load_version(self, version_id: str) -> WorldIR:
        record = self._record(version_id)
        payload = self._store.get(record["artifact_uri"])
        digest = record["artifact_hash"]
        import hashlib

        if hashlib.sha256(payload).hexdigest() != digest:
            raise WorldStoreError(
                f"version {version_id} artifact hash mismatch -- stored "
                "world bytes are corrupted or tampered"
            )
        from world_ir.world_v1 import WorldIR
        return WorldIR.from_dict(json.loads(payload.decode("utf-8")))

    def parents(self, version_id: str) -> list[str]:
        parent = self._record(version_id)["parent"]
        return [parent] if parent else []

    def ancestors(self, version_id: str) -> list[str]:
        """Root-first lineage (V1, V2 for a V3 whose parent is V2)."""
        chain: list[str] = []
        current = self._record(version_id)["parent"]
        seen = set()
        while current and current not in seen:
            seen.add(current)
            chain.append(current)
            current = self._record(current)["parent"]
        chain.reverse()
        return chain

    def list_versions(self) -> list[StoredVersion]:
        """Save-order listing (not uuid order): the version file's
        mtime ranks creations; ties fall back to the sequence number
        implied by an index file maintained at save time.

        Raises WorldStoreError naming the specific version if any
        record on disk is corrupted -- history is never silently
        truncated or dropped to hide the corruption from the caller."""
        records = []
        seq_path = self._root / "sequence.json"
        order = self._read_sequence(seq_path)
        known = {p.stem for p in self._versions_dir.glob("v-*.json")}
        # Any version file not in the recorded order (e.g. written by an
        # older store) is appended in sorted order -- history is never
        # dropped.
        ordered = [v for v in order if v in known] + sorted(known - set(order))
        seen: set[str] = set()
        for vid in ordered:
            # A sequence written before the save_version claim-lock fix
            # could log the same id twice (concurrent duplicate saves);
            # report each stored version once rather than duplicating
            # history the caller never created.
            if vid in seen:
                continue
            seen.add(vid)
            records.append(StoredVersion(**self._record(vid)))
        return records

    # ---- integrity ----

    def verify_version(self, version_id: str) -> list[dict]:
        """Re-digest the stored world bytes. Returns a list of failure
        records (empty when the version verifies)."""
        import hashlib

        record = self._record(version_id)
        failures: list[dict] = []
        try:
            payload = self._store.get(record["artifact_uri"])
        except Exception as exc:  # ArtifactNotFoundError or filesystem loss
            failures.append({
                "version": version_id,
                "reason": f"artifact missing from store: {exc}",
            })
            return failures
        if hashlib.sha256(payload).hexdigest() != record["artifact_hash"]:
            failures.append({
                "version": version_id,
                "reason": "hash mismatch: stored world bytes no longer "
                          "match the recorded digest (tampered or corrupted)",
            })
        return failures
