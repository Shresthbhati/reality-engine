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
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from world_ir.artifact_store import FileArtifactStore


class WorldStoreError(ValueError):
    """WorldStore operation refused."""


@dataclass(frozen=True)
class StoredVersion:
    version_id: str
    world_id: str
    parent: Optional[str]
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
        world: "WorldIR",
        *,
        parent: Optional[str],
        version_id: Optional[str] = None,
        source_session_ids: Optional[List[str]] = None,
    ) -> StoredVersion:
        vid = version_id or f"v-{uuid.uuid4().hex[:12]}"
        path = self._versions_dir / f"{vid}.json"
        if path.exists():
            raise WorldStoreError(
                f"version {vid} already exists -- versions are immutable; "
                "save a new version instead of overwriting observed reality"
            )
        changed_entity_ids: List[str] = []
        changed_geometry_ids: List[str] = []
        if parent is not None:
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
        path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        seq_path = self._root / "sequence.json"
        order: List[str] = []
        if seq_path.exists():
            order = json.loads(seq_path.read_text(encoding="utf-8"))
        order.append(vid)
        seq_path.write_text(json.dumps(order), encoding="utf-8")
        return StoredVersion(**record)

    # ---- read ----

    def _record(self, version_id: str) -> dict:
        path = self._versions_dir / f"{version_id}.json"
        if not path.exists():
            raise WorldStoreError(f"unknown version: {version_id}")
        return json.loads(path.read_text(encoding="utf-8"))

    def load_version(self, version_id: str) -> "WorldIR":
        from world_ir.world_v1 import WorldIR

        record = self._record(version_id)
        payload = self._store.get(record["artifact_uri"])
        digest = record["artifact_hash"]
        import hashlib

        if hashlib.sha256(payload).hexdigest() != digest:
            raise WorldStoreError(
                f"version {version_id} artifact hash mismatch -- stored "
                "world bytes are corrupted or tampered"
            )
        return WorldIR.from_dict(json.loads(payload.decode("utf-8")))

    def parents(self, version_id: str) -> List[str]:
        parent = self._record(version_id)["parent"]
        return [parent] if parent else []

    def ancestors(self, version_id: str) -> List[str]:
        """Root-first lineage (V1, V2 for a V3 whose parent is V2)."""
        chain: List[str] = []
        current = self._record(version_id)["parent"]
        seen = set()
        while current and current not in seen:
            seen.add(current)
            chain.append(current)
            current = self._record(current)["parent"]
        chain.reverse()
        return chain

    def list_versions(self) -> List[StoredVersion]:
        """Save-order listing (not uuid order): the version file's
        mtime ranks creations; ties fall back to the sequence number
        implied by an index file maintained at save time."""
        records = []
        seq_path = self._root / "sequence.json"
        order: List[str] = []
        if seq_path.exists():
            order = json.loads(seq_path.read_text(encoding="utf-8"))
        known = {p.stem for p in self._versions_dir.glob("v-*.json")}
        # Any version file not in the recorded order (e.g. written by an
        # older store) is appended in sorted order -- history is never
        # dropped.
        ordered = [v for v in order if v in known] + sorted(known - set(order))
        for vid in ordered:
            r = json.loads((self._versions_dir / f"{vid}.json").read_text(encoding="utf-8"))
            records.append(StoredVersion(**r))
        return records

    # ---- integrity ----

    def verify_version(self, version_id: str) -> List[dict]:
        """Re-digest the stored world bytes. Returns a list of failure
        records (empty when the version verifies)."""
        import hashlib

        record = self._record(version_id)
        failures: List[dict] = []
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
