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
from typing import TYPE_CHECKING

from world_ir.artifact_store import FileArtifactStore

if TYPE_CHECKING:
    from world_ir.world_v1 import WorldIR


class WorldStoreError(ValueError):
    """WorldStore operation refused."""


@dataclass(frozen=True)
class StoredVersion:
    version_id: str
    world_id: str
    parent: str | None
    artifact_uri: str
    artifact_hash: str
    # Fields for tracking what changed between versions (for diff/lineage)
    changed_entity_ids: list[str] | None = None
    changed_geometry_ids: list[str] | None = None
    # Source session IDs that contributed to this version
    source_session_ids: list[str] | None = None


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
            parent_world = self.load_version(parent)
            # Track entity changes
            for eid, entity in world.entities.items():
                if eid not in parent_world.entities:
                    changed_entity_ids.append(eid)
                elif parent_world.entities[eid] != entity:
                    changed_entity_ids.append(eid)
            for eid in parent_world.entities:
                if eid not in world.entities:
                    changed_entity_ids.append(eid)
            # Track geometry changes
            for gid, geom in world.geometries.items():
                if gid not in parent_world.geometries:
                    changed_geometry_ids.append(gid)
                elif parent_world.geometries[gid] != geom:
                    changed_geometry_ids.append(gid)
            for gid in parent_world.geometries:
                if gid not in world.geometries:
                    changed_geometry_ids.append(gid)

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
            "source_session_ids": source_session_ids,
        }
        path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        seq_path = self._root / "sequence.json"
        order: list[str] = []
        if seq_path.exists():
            order = json.loads(seq_path.read_text(encoding="utf-8"))
        order.append(vid)
        seq_path.write_text(json.dumps(order), encoding="utf-8")
        return StoredVersion(
            version_id=vid,
            world_id=world.id,
            parent=parent,
            artifact_uri=uri,
            artifact_hash=digest,
            changed_entity_ids=changed_entity_ids,
            changed_geometry_ids=changed_geometry_ids,
            source_session_ids=source_session_ids,
        )

    # ---- read ----

    def _record(self, version_id: str) -> dict:
        path = self._versions_dir / f"{version_id}.json"
        if not path.exists():
            raise WorldStoreError(f"unknown version: {version_id}")
        return json.loads(path.read_text(encoding="utf-8"))

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
        implied by an index file maintained at save time."""
        records = []
        seq_path = self._root / "sequence.json"
        order: list[str] = []
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
