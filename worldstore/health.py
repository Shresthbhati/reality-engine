"""Read-only WorldOS health/diagnostics (WORLDOS_LIFECYCLE_HEALTH
mission, Mission 2). NOT a second metadata system: every fact reported
here is read directly from existing `WorldStore`/tile metadata
(`worldstore/store.py`, `worldstore/tiles.py`) -- this module only
inspects and reports, never writes, never invents a parallel schema.

Intended consumer per the mission: Desktop Studio and Antigravity, who
need a single call to answer "is this version okay to open/query" and
"how expensive is opening it" without independently re-deriving the
same checks.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Optional, Tuple

from worldstore.store import WorldStore, WorldStoreError
from worldstore.tiles import (
    _delta_path,
    _manifest_path,
    delta_chain_depth,
    open_version,
)


@dataclass(frozen=True)
class WorldVersionHealth:
    version_id: str
    #: "full" (a real TileManifest file), "delta" (a TileManifestDelta
    #: file), "whole_world_only" (saved via WorldStore.save_version()
    #: with no tile manifest at all), or "unknown" (no record exists).
    manifest_type: str
    parent: Optional[str]
    #: `worldstore.tiles.delta_chain_depth` -- 0 for "full"/"whole_world_only".
    chain_depth: int
    coordinate_frame: Optional[str]
    world_version: Optional[int]
    tile_count: int
    #: Total distinct artifacts this version's manifest references:
    #: every tile's artifact_uri plus the residual artifact, if present.
    referenced_artifact_count: int
    #: Tile keys (or "residual") whose referenced artifact is absent
    #: from the artifact store entirely.
    missing_artifacts: Tuple[str, ...]
    #: Tile keys (or "residual") whose artifact bytes exist but fail
    #: their recorded sha256 hash -- corruption or tampering.
    corrupt_artifacts: Tuple[str, ...]
    #: Entity ids listed in `unlocalized_entity_ids` that are NOT
    #: actually present in the residual artifact's entity set --
    #: a manifest/residual inconsistency, never expected in a
    #: correctly-written version.
    stale_references: Tuple[str, ...]
    #: True if `parent` is None (a root version) or resolvable (has its
    #: own manifest/delta/whole-world record); False if `parent` is set
    #: but nothing in the store answers for it.
    lineage_ok: bool
    #: Populated only when the version could not be inspected at all
    #: (e.g. no record exists for `version_id`) -- every other field is
    #: then a best-effort default, not a real observation.
    error: Optional[str]

    @property
    def is_healthy(self) -> bool:
        return (
            self.error is None
            and not self.missing_artifacts
            and not self.corrupt_artifacts
            and not self.stale_references
            and self.lineage_ok
        )


def _version_exists_anywhere(store: WorldStore, version_id: str) -> bool:
    if _manifest_path(store, version_id).exists() or _delta_path(store, version_id).exists():
        return True
    try:
        store._record(version_id)
        return True
    except WorldStoreError:
        return False


def check_world_health(store: WorldStore, version_id: str) -> WorldVersionHealth:
    """One read-only pass over everything already known about
    `version_id`: manifest type, chain depth, artifact integrity
    (existence + hash check on every referenced artifact, not just a
    sample), residual/unlocalized-entity consistency, coordinate frame,
    and immediate lineage resolvability. Never raises for a version
    with problems -- problems ARE the report; a version that doesn't
    exist anywhere in the store at all is also reported via `error`,
    never raised, so a health sweep over many version ids never has to
    wrap each call in try/except."""
    if _manifest_path(store, version_id).exists():
        manifest_type = "full"
    elif _delta_path(store, version_id).exists():
        manifest_type = "delta"
    else:
        try:
            record = store._record(version_id)
            manifest_type = "whole_world_only" if record.get("artifact_uri") else "unknown"
        except WorldStoreError:
            return WorldVersionHealth(
                version_id=version_id, manifest_type="unknown", parent=None, chain_depth=0,
                coordinate_frame=None, world_version=None, tile_count=0, referenced_artifact_count=0,
                missing_artifacts=(), corrupt_artifacts=(), stale_references=(), lineage_ok=False,
                error=f"no record, manifest, or delta found for version {version_id}",
            )

    if manifest_type == "whole_world_only":
        record = store._record(version_id)
        failures = store.verify_version(version_id)
        return WorldVersionHealth(
            version_id=version_id, manifest_type=manifest_type, parent=record.get("parent"),
            chain_depth=0, coordinate_frame=None, world_version=None, tile_count=0,
            referenced_artifact_count=1,
            missing_artifacts=tuple("whole_world_blob" for f in failures if "missing" in f["reason"]),
            corrupt_artifacts=tuple("whole_world_blob" for f in failures if "hash mismatch" in f["reason"]),
            stale_references=(),
            lineage_ok=record.get("parent") is None or _version_exists_anywhere(store, record["parent"]),
            error=None,
        )

    try:
        handle = open_version(store, version_id)
    except WorldStoreError as exc:
        return WorldVersionHealth(
            version_id=version_id, manifest_type=manifest_type, parent=None, chain_depth=0,
            coordinate_frame=None, world_version=None, tile_count=0, referenced_artifact_count=0,
            missing_artifacts=(), corrupt_artifacts=(), stale_references=(), lineage_ok=False,
            error=f"manifest/delta chain unresolvable: {exc}",
        )
    manifest = handle.manifest

    missing: list = []
    corrupt: list = []
    for ref in manifest.tiles:
        try:
            payload = store._store.get(ref.artifact_uri)
        except Exception:
            missing.append(f"tile:{ref.tile_key}")
            continue
        if hashlib.sha256(payload).hexdigest() != ref.content_hash:
            corrupt.append(f"tile:{ref.tile_key}")

    referenced = len(manifest.tiles)
    stale: Tuple[str, ...] = ()
    if manifest.residual_uri:
        referenced += 1
        try:
            residual_payload = store._store.get(manifest.residual_uri)
        except Exception:
            missing.append("residual")
        else:
            if hashlib.sha256(residual_payload).hexdigest() != manifest.residual_hash:
                corrupt.append("residual")
            else:
                try:
                    residual_data = json.loads(residual_payload.decode("utf-8"))
                    present = set(residual_data.get("entities", {}))
                    stale = tuple(
                        eid for eid in manifest.unlocalized_entity_ids if eid not in present
                    )
                except json.JSONDecodeError:
                    corrupt.append("residual")

    depth = delta_chain_depth(store, version_id)
    lineage_ok = manifest.parent is None or _version_exists_anywhere(store, manifest.parent)

    return WorldVersionHealth(
        version_id=version_id, manifest_type=manifest_type, parent=manifest.parent,
        chain_depth=depth, coordinate_frame=manifest.coordinate_frame, world_version=manifest.world_version,
        tile_count=len(manifest.tiles), referenced_artifact_count=referenced,
        missing_artifacts=tuple(missing), corrupt_artifacts=tuple(corrupt), stale_references=stale,
        lineage_ok=lineage_ok, error=None,
    )
