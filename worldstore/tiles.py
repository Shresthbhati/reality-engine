"""Lazy, tile-granular WorldStore access (WORLDOS lazy-spatial-world
checkpoint): an ADDITIVE layer on top of the existing whole-world
`WorldStore`/`FileArtifactStore`, not a replacement or a second
storage system.

Audit finding this module acts on: `WorldStore.save_version()`/
`load_version()` persist and read exactly one JSON blob per version --
the ENTIRE serialized `WorldIR`. There is no way to answer "which
spatial tiles exist in this version" or "give me just this one tile"
without deserializing everything. This module adds that capability
using the SAME `FileArtifactStore` the whole-world path already uses
(content-addressed, sha256-keyed) -- a tile's serialized entity/
geometry subset is just another artifact.

Design constraint honored: this is an INDEX over the world, not a
second world representation. The manifest records which tile holds
which entity ids and where its content-addressed blob lives; it does
not re-derive or duplicate WorldIR's own entity/geometry schema.

Existing whole-world `save_version()`/`load_version()` are UNCHANGED
and remain the source of truth for diff_worlds()-based lineage
(`StoredVersion.changed_entity_ids` etc.) -- `save_version_tiled()`
calls the existing `save_version()` internally and adds the tile
manifest as an additional, independently-loadable artifact for the
same version id.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, FrozenSet, Optional, Tuple

from world_ir.schema_v1 import Entity, Geometry  # noqa: F401 (Geometry used in type hints below)
from world_ir.spatial_tiles import SpatialTiles
from worldstore.store import StoredVersion, WorldStore, WorldStoreError, _atomic_write_text

if TYPE_CHECKING:
    from world_ir.incremental import IncrementalUpdateResult
    from world_ir.world_v1 import WorldIR

TileKey = Tuple[int, int, int]


def _tile_key_to_str(key: TileKey) -> str:
    return f"{key[0]},{key[1]},{key[2]}"


def _tile_key_from_str(s: str) -> TileKey:
    x, y, z = s.split(",")
    return (int(x), int(y), int(z))


@dataclass(frozen=True)
class TileArtifactRef:
    """One tile's entry in a version's manifest: where its content
    lives, what it contains, and its content identity."""
    tile_key: TileKey
    bounds_min: Tuple[float, float, float]
    bounds_max: Tuple[float, float, float]
    entity_ids: Tuple[str, ...]  # sorted, deterministic
    artifact_uri: str
    content_hash: str

    def to_dict(self) -> dict:
        return {
            "tile_key": list(self.tile_key),
            "bounds_min": list(self.bounds_min),
            "bounds_max": list(self.bounds_max),
            "entity_ids": list(self.entity_ids),
            "artifact_uri": self.artifact_uri,
            "content_hash": self.content_hash,
        }

    @staticmethod
    def from_dict(data: dict) -> "TileArtifactRef":
        return TileArtifactRef(
            tile_key=tuple(data["tile_key"]),
            bounds_min=tuple(data["bounds_min"]),
            bounds_max=tuple(data["bounds_max"]),
            entity_ids=tuple(data["entity_ids"]),
            artifact_uri=data["artifact_uri"],
            content_hash=data["content_hash"],
        )


@dataclass(frozen=True)
class TileManifest:
    """The minimum index needed to answer "which spatial tiles exist
    in this WorldStore version" without loading the whole world.
    Preserves version/parent lineage, coordinate frame, and
    deterministic tile ordering (sorted by tile_key)."""
    version_id: str
    world_id: str
    parent: Optional[str]
    tile_size: float
    coordinate_frame: str
    #: WorldIR's OWN schema version counter (`WorldIR.version`, e.g. 1, 2,
    #: 3, ...) -- distinct from `version_id` (the WorldStore-assigned
    #: `v-<uuid>` string). Partition-metadata audit (Task 4): a caller
    #: doing large-world operational triage needs both "which WorldStore
    #: version is this" and "what generation of this world's own
    #: versioning scheme" without opening the world itself.
    world_version: int
    tiles: Tuple[TileArtifactRef, ...]
    unlocalized_entity_ids: Tuple[str, ...]
    #: Set only by `save_version_partitioned` -- where the residual
    #: artifact (global world state + unlocalized entities) lives for
    #: this version. None for a manifest written by `save_version_tiled`
    #: (that path's whole-world blob already carries everything the
    #: residual would). Kept on the MANIFEST rather than the version
    #: record so `WorldStore.list_versions()`'s `StoredVersion(**record)`
    #: never has to change shape for a partitioned version.
    residual_uri: Optional[str] = None
    residual_hash: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "version_id": self.version_id,
            "world_id": self.world_id,
            "parent": self.parent,
            "tile_size": self.tile_size,
            "coordinate_frame": self.coordinate_frame,
            "world_version": self.world_version,
            "tiles": [t.to_dict() for t in self.tiles],
            "unlocalized_entity_ids": list(self.unlocalized_entity_ids),
            "residual_uri": self.residual_uri,
            "residual_hash": self.residual_hash,
        }

    @staticmethod
    def from_dict(data: dict) -> "TileManifest":
        return TileManifest(
            version_id=data["version_id"],
            world_id=data["world_id"],
            parent=data.get("parent"),
            tile_size=data["tile_size"],
            coordinate_frame=data["coordinate_frame"],
            # Manifests written before this field existed (WORLDOS
            # lazy-spatial-world checkpoint) default to 1 -- an unknown
            # world_version is reported as "assume unversioned/v1" rather
            # than erroring on a still-valid, older manifest file.
            world_version=data.get("world_version", 1),
            tiles=tuple(TileArtifactRef.from_dict(t) for t in data["tiles"]),
            unlocalized_entity_ids=tuple(data.get("unlocalized_entity_ids", ())),
            residual_uri=data.get("residual_uri"),
            residual_hash=data.get("residual_hash"),
        )

    def occupancy_summary(self) -> Dict[str, float]:
        """Per-tile entity-count stats -- large-world operational
        triage metadata (Task 4 "occupancy"): which tiles are hot
        (overloaded) without opening any tile artifact. Computed on
        demand from `tiles`/`unlocalized_entity_ids` rather than stored
        redundantly, so it can never drift from the authoritative data."""
        counts = [len(t.entity_ids) for t in self.tiles]
        total_localized = sum(counts)
        return {
            "tile_count": len(self.tiles),
            "total_entity_count": total_localized + len(self.unlocalized_entity_ids),
            "unlocalized_entity_count": len(self.unlocalized_entity_ids),
            "max_tile_occupancy": max(counts) if counts else 0,
            "min_tile_occupancy": min(counts) if counts else 0,
            "mean_tile_occupancy": (total_localized / len(counts)) if counts else 0.0,
        }


def _manifests_dir(store: WorldStore) -> Path:
    d = store._root / "tile_manifests"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _tile_bounds(tiles: SpatialTiles, key: TileKey) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    s = tiles.tile_size
    lo = (key[0] * s, key[1] * s, key[2] * s)
    hi = ((key[0] + 1) * s, (key[1] + 1) * s, (key[2] + 1) * s)
    return lo, hi


def _serialize_tile_blob(world: "WorldIR", entity_ids: Tuple[str, ...]) -> bytes:
    """A tile's content-addressed payload: the tile's own entities plus
    every geometry any of them references (so a loaded tile is
    immediately usable, not a dangling set of geometry_ids)."""
    entities = {eid: world.entities[eid].to_dict() for eid in entity_ids}
    geometry_ids = sorted({
        gid for eid in entity_ids for gid in world.entities[eid].geometry_ids
        if gid in world.geometries
    })
    geometries = {gid: world.geometries[gid].to_dict() for gid in geometry_ids}
    payload = {"entities": entities, "geometries": geometries}
    return json.dumps(payload, sort_keys=True).encode("utf-8")


def build_tile_manifest(
    world: "WorldIR",
    *,
    version_id: str,
    parent: Optional[str],
    store: WorldStore,
    tile_size: float = 10.0,
    reuse_from: Optional[TileManifest] = None,
    rebuilt_tile_keys: Optional[FrozenSet[TileKey]] = None,
) -> TileManifest:
    """Builds the tile manifest for `world` and writes each tile's
    content-addressed blob via the SAME `FileArtifactStore` the
    whole-world path already uses.

    Incremental reuse: when `reuse_from` (the parent version's
    manifest) and `rebuilt_tile_keys` (from an IncrementalUpdateResult
    -- rebuilt_tile_ids) are both given, any tile key NOT in
    `rebuilt_tile_keys` whose entity-id set is identical to the
    parent's is reused VERBATIM (same artifact_uri/content_hash, zero
    writes) instead of being re-serialized. A tile absent from the
    parent (e.g. the world grew) or present in `rebuilt_tile_keys` is
    always rebuilt, even if its content happens to hash the same --
    correctness over a micro-optimization.
    """
    tiles = SpatialTiles(world, tile_size=tile_size)
    reuse_map: Dict[TileKey, TileArtifactRef] = (
        {ref.tile_key: ref for ref in reuse_from.tiles} if reuse_from is not None else {}
    )
    rebuilt = rebuilt_tile_keys or frozenset()

    refs = []
    for key in tiles.tile_ids():
        entity_ids = tuple(sorted(tiles.entities_in_tile(key)))
        prior = reuse_map.get(key)
        if key not in rebuilt and prior is not None and prior.entity_ids == entity_ids:
            refs.append(prior)
            continue
        bounds_min, bounds_max = _tile_bounds(tiles, key)
        payload = _serialize_tile_blob(world, entity_ids)
        artifact_uri, content_hash = store._store.put(payload)
        refs.append(TileArtifactRef(
            tile_key=key, bounds_min=bounds_min, bounds_max=bounds_max,
            entity_ids=entity_ids, artifact_uri=artifact_uri, content_hash=content_hash,
        ))

    return TileManifest(
        version_id=version_id,
        world_id=world.id,
        parent=parent,
        tile_size=tile_size,
        coordinate_frame=world.coordinate_frame.value,
        world_version=world.version,
        tiles=tuple(sorted(refs, key=lambda r: r.tile_key)),
        unlocalized_entity_ids=tuple(sorted(tiles.unlocalized_entity_ids)),
    )


def _manifest_path(store: WorldStore, version_id: str) -> Path:
    return _manifests_dir(store) / f"{version_id}.json"


# ---------------------------------------------------------------------
# Delta manifests (WORLDOS real-data city-scale mission, item 8): even
# `save_version_partitioned` still rewrites the FULL tile manifest on
# every save -- O(tile count), not O(affected) -- because
# `build_tile_manifest` always emits one entry per tile in the world,
# reused or not. For a real city-scale world (millions of tiles), that
# alone defeats the O(affected) persistence goal: a one-entity change
# would still write a multi-megabyte manifest.
#
# `save_version_partitioned_delta` fixes this: it writes ONLY the tile
# refs that are new or rebuilt, plus which tile keys were vacated (no
# longer occupied) -- never the reused majority. `_resolve_manifest`
# reconstructs the effective full `TileManifest` on read by walking
# back to the nearest FULL manifest (or delta chain start) and replaying
# deltas forward -- a chain walk bounded by "how many partitioned-delta
# saves since the last full manifest", not by total tile count.


def _deltas_dir(store: WorldStore) -> Path:
    d = store._root / "tile_manifest_deltas"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _delta_path(store: WorldStore, version_id: str) -> Path:
    return _deltas_dir(store) / f"{version_id}.json"


@dataclass(frozen=True)
class TileManifestDelta:
    """What changed in one version's tile layout relative to its
    parent -- NOT a full tile listing. `changed_tiles` covers both new
    tile keys (the world grew into previously-empty space) and rebuilt
    ones; `removed_tile_keys` covers tile keys the parent had that this
    version no longer occupies at all (every entity that lived there
    moved away or was deleted)."""
    version_id: str
    parent: str  # a delta always has a parent -- the first version in any chain must be a full TileManifest
    world_id: str
    tile_size: float
    coordinate_frame: str
    world_version: int
    changed_tiles: Tuple[TileArtifactRef, ...]
    removed_tile_keys: Tuple[TileKey, ...]
    unlocalized_entity_ids: Tuple[str, ...]
    residual_uri: Optional[str] = None
    residual_hash: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "version_id": self.version_id,
            "parent": self.parent,
            "world_id": self.world_id,
            "tile_size": self.tile_size,
            "coordinate_frame": self.coordinate_frame,
            "world_version": self.world_version,
            "changed_tiles": [t.to_dict() for t in self.changed_tiles],
            "removed_tile_keys": [list(k) for k in self.removed_tile_keys],
            "unlocalized_entity_ids": list(self.unlocalized_entity_ids),
            "residual_uri": self.residual_uri,
            "residual_hash": self.residual_hash,
        }

    @staticmethod
    def from_dict(data: dict) -> "TileManifestDelta":
        return TileManifestDelta(
            version_id=data["version_id"],
            parent=data["parent"],
            world_id=data["world_id"],
            tile_size=data["tile_size"],
            coordinate_frame=data["coordinate_frame"],
            world_version=data.get("world_version", 1),
            changed_tiles=tuple(TileArtifactRef.from_dict(t) for t in data["changed_tiles"]),
            removed_tile_keys=tuple(tuple(k) for k in data.get("removed_tile_keys", ())),
            unlocalized_entity_ids=tuple(data.get("unlocalized_entity_ids", ())),
            residual_uri=data.get("residual_uri"),
            residual_hash=data.get("residual_hash"),
        )


def build_tile_manifest_delta(
    world: "WorldIR",
    *,
    version_id: str,
    parent: str,
    store: WorldStore,
    tile_size: float,
    parent_manifest: TileManifest,
    rebuilt_tile_keys: Optional[FrozenSet[TileKey]] = None,
) -> TileManifestDelta:
    """Same tile-reuse decision as `build_tile_manifest`, but returns
    ONLY the changed/new tile refs and the vacated tile keys -- never
    the reused majority."""
    tiles = SpatialTiles(world, tile_size=tile_size)
    parent_map: Dict[TileKey, TileArtifactRef] = {ref.tile_key: ref for ref in parent_manifest.tiles}
    rebuilt = rebuilt_tile_keys or frozenset()
    new_keys = set(tiles.tile_ids())

    changed: list = []
    for key in sorted(new_keys):
        entity_ids = tuple(sorted(tiles.entities_in_tile(key)))
        prior = parent_map.get(key)
        if key not in rebuilt and prior is not None and prior.entity_ids == entity_ids:
            continue  # unchanged -- omitted from the delta entirely
        bounds_min, bounds_max = _tile_bounds(tiles, key)
        payload = _serialize_tile_blob(world, entity_ids)
        artifact_uri, content_hash = store._store.put(payload)
        changed.append(TileArtifactRef(
            tile_key=key, bounds_min=bounds_min, bounds_max=bounds_max,
            entity_ids=entity_ids, artifact_uri=artifact_uri, content_hash=content_hash,
        ))

    removed_keys = tuple(sorted(set(parent_map.keys()) - new_keys))

    return TileManifestDelta(
        version_id=version_id,
        parent=parent,
        world_id=world.id,
        tile_size=tile_size,
        coordinate_frame=world.coordinate_frame.value,
        world_version=world.version,
        changed_tiles=tuple(changed),
        removed_tile_keys=removed_keys,
        unlocalized_entity_ids=tuple(sorted(tiles.unlocalized_entity_ids)),
    )


def _resolve_manifest(store: WorldStore, version_id: str) -> TileManifest:
    """Reconstructs the effective full `TileManifest` for `version_id`,
    whether it was written as a full manifest or a delta -- walks back
    to the nearest full manifest (or a delta with no further parent,
    which is a data-integrity error, not a normal case) and replays
    deltas forward in order. Cost is proportional to chain length
    (number of delta saves since the last full manifest), never to
    total tile count."""
    full_path = _manifest_path(store, version_id)
    if full_path.exists():
        return _read_manifest(store, version_id)

    delta_path = _delta_path(store, version_id)
    if not delta_path.exists():
        raise WorldStoreError(
            f"no tile manifest for version {version_id} -- it was saved "
            "without save_version_tiled()/save_version_partitioned()/"
            "save_version_partitioned_delta(), so lazy tile access is "
            "unavailable for it; use load_version() for the whole-world "
            "path instead"
        )

    # Walk parent pointers back to the nearest full manifest, collecting
    # the delta chain in reverse (newest first), then replay oldest-first.
    chain: list = []
    current_id = version_id
    while True:
        d_path = _delta_path(store, current_id)
        if not d_path.exists():
            raise WorldStoreError(
                f"tile manifest delta chain for version {version_id} is broken at "
                f"{current_id} -- no full manifest or delta found; the store may be corrupted"
            )
        try:
            delta = TileManifestDelta.from_dict(json.loads(d_path.read_text(encoding="utf-8")))
        except json.JSONDecodeError as exc:
            raise WorldStoreError(f"tile manifest delta for version {current_id} is corrupted: {exc}") from exc
        chain.append(delta)
        if _manifest_path(store, delta.parent).exists():
            base = _read_manifest(store, delta.parent)
            break
        current_id = delta.parent

    tile_map: Dict[TileKey, TileArtifactRef] = {t.tile_key: t for t in base.tiles}
    result_meta = base
    for delta in reversed(chain):
        for key in delta.removed_tile_keys:
            tile_map.pop(key, None)
        for ref in delta.changed_tiles:
            tile_map[ref.tile_key] = ref
        result_meta = delta

    return TileManifest(
        version_id=version_id,
        world_id=result_meta.world_id,
        parent=result_meta.parent,
        tile_size=result_meta.tile_size,
        coordinate_frame=result_meta.coordinate_frame,
        world_version=result_meta.world_version,
        tiles=tuple(sorted(tile_map.values(), key=lambda r: r.tile_key)),
        unlocalized_entity_ids=result_meta.unlocalized_entity_ids,
        residual_uri=result_meta.residual_uri,
        residual_hash=result_meta.residual_hash,
    )


def delta_chain_depth(store: WorldStore, version_id: str) -> int:
    """How many delta hops `version_id` is from the nearest full
    manifest -- 0 if `version_id` itself already has a full manifest.
    Lets a caller decide its own compaction cadence (e.g. "compact
    every 50 saves") without this module imposing one; an unbounded
    chain is a real cost (see `_resolve_manifest`'s docstring), but the
    right interval depends on save frequency and world size, which
    only the caller knows."""
    depth = 0
    current_id = version_id
    while True:
        if _manifest_path(store, current_id).exists():
            return depth
        d_path = _delta_path(store, current_id)
        if not d_path.exists():
            raise WorldStoreError(f"no tile manifest for version {current_id}")
        try:
            delta = TileManifestDelta.from_dict(json.loads(d_path.read_text(encoding="utf-8")))
        except json.JSONDecodeError as exc:
            raise WorldStoreError(f"tile manifest delta for version {current_id} is corrupted: {exc}") from exc
        current_id = delta.parent
        depth += 1


def compact_manifest_chain(store: WorldStore, version_id: str) -> int:
    """Materializes `version_id`'s EFFECTIVE manifest (resolved via
    `_resolve_manifest`, whatever the current chain depth) as a full
    manifest at `version_id`'s own path -- bounding every future
    `_resolve_manifest`/`delta_chain_depth` call for this version, and
    for any LATER version whose delta chain passes through it, to at
    most the hops added since this compaction.

    This is a pure read-path optimization, not a history rewrite: the
    version's actual tile content is unchanged (verified by
    `_resolve_manifest` producing byte-identical `content_hash` values
    before and after), and the original delta file is left on disk
    untouched -- `_resolve_manifest` already prefers a full manifest
    over a delta when both exist, so nothing else needs to change.
    Idempotent: compacting an already-full manifest just rewrites it
    with the same content.

    Returns the chain depth that was collapsed (0 if `version_id` was
    already a full manifest -- a harmless no-op)."""
    depth = delta_chain_depth(store, version_id)
    if depth == 0:
        return 0
    manifest = _resolve_manifest(store, version_id)
    _atomic_write_text(_manifest_path(store, version_id), json.dumps(manifest.to_dict(), indent=2))
    return depth


def _read_manifest(store: WorldStore, version_id: str) -> TileManifest:
    path = _manifest_path(store, version_id)
    if not path.exists():
        raise WorldStoreError(
            f"no tile manifest for version {version_id} -- it was saved "
            "without save_version_tiled()/save_incremental_tiled(), so "
            "lazy tile access is unavailable for it; use load_version() "
            "for the whole-world path instead"
        )
    try:
        return TileManifest.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except json.JSONDecodeError as exc:
        raise WorldStoreError(f"tile manifest for version {version_id} is corrupted: {exc}") from exc


def save_version_tiled(
    store: WorldStore,
    world: "WorldIR",
    *,
    parent: Optional[str],
    version_id: Optional[str] = None,
    tile_size: float = 10.0,
    source_session_ids=None,
    incremental_result: "Optional[IncrementalUpdateResult]" = None,
) -> StoredVersion:
    """Saves `world` via the existing whole-world `save_version()`
    (unchanged -- still the source of truth for diff-based lineage),
    THEN additionally writes a tile manifest for lazy access. The
    manifest is written LAST and atomically, after every tile artifact
    it references is already durably content-addressed -- a crash
    between tile writes and the manifest write leaves orphan artifacts
    (harmless, content-addressed garbage) but never a manifest
    referencing a tile that doesn't exist.

    `incremental_result` (an IncrementalUpdateResult from
    apply_incremental_update, when this save follows one) enables
    incremental tile reuse: unaffected tiles keep their parent's exact
    artifact reference instead of being re-serialized.
    """
    stored = store.save_version(
        world, parent=parent, version_id=version_id, source_session_ids=source_session_ids,
    )
    reuse_from = None
    if parent is not None:
        try:
            # _resolve_manifest (not _read_manifest): the parent may
            # itself be a delta-chained version -- reuse must still
            # work against its EFFECTIVE tile set, not fail silently
            # into "rebuild everything" just because the parent's own
            # manifest isn't a full one.
            reuse_from = _resolve_manifest(store, parent)
        except WorldStoreError:
            reuse_from = None  # parent has no tile manifest -- nothing to reuse from
    rebuilt = incremental_result.rebuilt_tile_ids if incremental_result is not None else None

    manifest = build_tile_manifest(
        world, version_id=stored.version_id, parent=parent, store=store,
        tile_size=tile_size, reuse_from=reuse_from, rebuilt_tile_keys=rebuilt,
    )
    _atomic_write_text(_manifest_path(store, stored.version_id), json.dumps(manifest.to_dict(), indent=2))
    return stored


class WorldVersionHandle:
    """Lazy handle to one WorldStore version: opening it reads ONLY the
    tile manifest (a small JSON index), never the whole world. Tiles
    are deserialized one at a time, on demand."""

    def __init__(self, store: WorldStore, manifest: TileManifest):
        self._store = store
        self.manifest = manifest
        # Per-handle tile cache (WORLDOS_LIFECYCLE_HEALTH mission,
        # Mission 5): measured that repeated/adjacent queries against
        # the SAME open handle re-fetched and re-verified identical
        # tile bytes every time -- pure waste, since a tile's content
        # is content-addressed and immutable for the lifetime of a
        # resolved manifest. Keyed by tile_key, not artifact_uri, so a
        # cache hit also skips the manifest-tile lookup, not just the
        # artifact read. Never invalidated because it never needs to
        # be: this handle's `manifest` itself never changes after
        # construction. Unbounded for now -- documented limitation for
        # a handle kept open across a very large number of distinct
        # tiles; an LRU cap would be the natural follow-up.
        self._tile_cache: Dict[TileKey, Tuple[Dict[str, Entity], Dict[str, Geometry]]] = {}

    @property
    def version_id(self) -> str:
        return self.manifest.version_id

    def list_tiles(self) -> Tuple[TileKey, ...]:
        return tuple(t.tile_key for t in self.manifest.tiles)

    def tile_bounds(self, tile_key: TileKey) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        for ref in self.manifest.tiles:
            if ref.tile_key == tile_key:
                return ref.bounds_min, ref.bounds_max
        raise WorldStoreError(f"unknown tile {tile_key} in version {self.version_id}")

    def load_tile(self, tile_key: TileKey) -> Dict[str, Entity]:
        """Deserializes and returns ONLY this tile's entities -- the
        whole point: opening a large world never requires this call to
        touch any other tile's data."""
        for ref in self.manifest.tiles:
            if ref.tile_key == tile_key:
                return self._load_ref(ref)[0]
        raise WorldStoreError(f"unknown tile {tile_key} in version {self.version_id}")

    def _load_ref(self, ref: TileArtifactRef) -> Tuple[Dict[str, Entity], Dict[str, Geometry]]:
        cached = self._tile_cache.get(ref.tile_key)
        if cached is not None:
            return cached
        payload = self._store._store.get(ref.artifact_uri)
        digest = hashlib.sha256(payload).hexdigest()
        if digest != ref.content_hash:
            raise WorldStoreError(
                f"tile {ref.tile_key} in version {self.version_id} artifact hash "
                "mismatch -- stored tile bytes are corrupted or tampered"
            )
        data = json.loads(payload.decode("utf-8"))
        entities = {eid: Entity.from_dict(d) for eid, d in data["entities"].items()}
        # `_serialize_tile_blob` already bundles each tile's referenced
        # geometries into the same payload -- previously discarded here,
        # which silently broke position resolution for any entity placed
        # via geometry centroid (no transform.position) once loaded
        # lazily instead of through the whole-world path.
        geometries = {gid: Geometry.from_dict(d) for gid, d in data.get("geometries", {}).items()}
        self._tile_cache[ref.tile_key] = (entities, geometries)
        return entities, geometries

    def query_region(
        self, bounds_min: Tuple[float, float, float], bounds_max: Tuple[float, float, float],
    ) -> Dict[str, Entity]:
        """Loads ONLY the tiles whose manifest bounds overlap the query
        region, then filters to entities within it. A query touching
        one tile never loads the rest of the world."""
        return self.query_region_with_geometries(bounds_min, bounds_max)[0]

    def query_region_with_geometries(
        self, bounds_min: Tuple[float, float, float], bounds_max: Tuple[float, float, float],
    ) -> Tuple[Dict[str, Entity], Dict[str, Geometry]]:
        """Same tile-overlap loading as `query_region`, but also returns
        the geometries those entities reference -- what `lazy_query.py`'s
        SpatialIndex adapter needs to resolve position for
        geometry-only-placed entities without loading the whole world."""
        entities: Dict[str, Entity] = {}
        geometries: Dict[str, Geometry] = {}
        for ref in self.manifest.tiles:
            if any(ref.bounds_max[i] < bounds_min[i] or ref.bounds_min[i] > bounds_max[i] for i in range(3)):
                continue  # tile's AABB doesn't overlap the query region at all
            e, g = self._load_ref(ref)
            entities.update(e)
            geometries.update(g)
        return entities, geometries

    def _overall_bounds(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """The AABB covering every tile in the manifest -- used by
        `lazy_query.py`'s ring-expanding nearest() to know when a search
        box already covers the whole indexed world."""
        if not self.manifest.tiles:
            return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
        mins = [min(t.bounds_min[i] for t in self.manifest.tiles) for i in range(3)]
        maxs = [max(t.bounds_max[i] for t in self.manifest.tiles) for i in range(3)]
        return tuple(mins), tuple(maxs)


def open_version(store: WorldStore, version_id: str) -> WorldVersionHandle:
    """Opens a version for lazy tile access: reads ONLY the tile
    manifest (or resolves a delta chain down to one), not the whole
    world. Raises WorldStoreError if this version was never saved with
    a tile manifest or delta (see save_version_tiled/
    save_version_partitioned_delta)."""
    manifest = _resolve_manifest(store, version_id)
    return WorldVersionHandle(store, manifest)


# ---------------------------------------------------------------------
# Partitioned persistence (WORLDOS real-data city-scale checkpoint,
# Task 5): `WorldStore.save_version()` always serializes and writes the
# ENTIRE world as one JSON blob, even when an incremental update only
# touched a handful of entities out of a city-scale world -- O(n) cost
# per save regardless of how localized the change was (measured:
# `benchmarks/city_scale_spatial_bench.py` shows save/load scaling to
# 45s/53s at 100K entities).
#
# `save_version_partitioned()` is an ADDITIVE alternative, not a
# replacement: `WorldStore.save_version()`/`load_version()` are
# UNCHANGED and remain fully correct for every version already saved
# through them (Task 5's "existing versions must remain readable"). A
# version saved via `save_version_partitioned()` instead writes:
#   - tile artifacts (reused verbatim where an IncrementalUpdateResult
#     says a tile is unaffected -- same reuse mechanism as
#     `save_version_tiled()`)
#   - one small "residual" artifact: everything NOT partitionable by
#     tile -- global world metadata (branches, scenarios, temporal
#     state, ...) and any UNLOCALIZED entity (no tile membership) plus
#     its geometries
#   - a version record in the SAME `_versions_dir` `WorldStore` already
#     uses, so `list_versions()`/`ancestors()`/`parents()` keep working
#     unmodified for partitioned versions too. `artifact_uri`/
#     `artifact_hash` are set to "" -- a version saved this way has no
#     whole-world blob, so `WorldStore.load_version()` correctly fails
#     loudly on it (a bad artifact_uri raises, never a silent wrong
#     read) instead of pretending to support it; use
#     `load_version_partitioned()` for these versions.
# For a change touching K entities in an N-entity, city-scale world
# with M >> K tiles, this writes O(K) + O(residual) bytes instead of
# O(N) -- the residual blob is bounded by unlocalized-entity count and
# global metadata, not total entity count.


@dataclass(frozen=True)
class PartitionedVersion:
    """Return value of `save_version_partitioned` -- mirrors the shape
    of `worldstore.store.StoredVersion` closely enough to reuse the
    same lineage fields, but is a distinct type since a partitioned
    version has no whole-world `artifact_uri`/`artifact_hash`."""
    version_id: str
    world_id: str
    parent: Optional[str]
    residual_uri: str
    residual_hash: str
    changed_entity_ids: Tuple[str, ...]
    changed_geometry_ids: Tuple[str, ...]
    source_session_ids: Tuple[str, ...]


def _serialize_residual_blob(world: "WorldIR", manifest: TileManifest) -> bytes:
    """Everything a tile artifact can't carry: reuses `WorldIR.to_dict()`
    (the canonical, bit-identical-round-trip serialization) so global
    world state is never hand-re-derived, then narrows `entities`/
    `geometries` down to just the unlocalized set -- every tile-resident
    entity/geometry is already durably stored in its own tile artifact
    and would be pure duplication here."""
    full = world.to_dict()
    unlocalized_ids = set(manifest.unlocalized_entity_ids)
    full["entities"] = {
        eid: world.entities[eid].to_dict() for eid in unlocalized_ids if eid in world.entities
    }
    geometry_ids = sorted({
        gid for eid in unlocalized_ids for gid in world.entities[eid].geometry_ids
        if gid in world.geometries
    })
    full["geometries"] = {gid: world.geometries[gid].to_dict() for gid in geometry_ids}
    return json.dumps(full, sort_keys=True).encode("utf-8")


def _partitioned_record_path(store: WorldStore, version_id: str) -> Path:
    # SAME directory WorldStore._record() reads from -- list_versions()/
    # ancestors()/parents() work on partitioned versions with zero
    # changes, since they only need version_id/parent, which this
    # record provides in the identical shape.
    return store._versions_dir / f"{version_id}.json"


def save_version_partitioned(
    store: WorldStore,
    world: "WorldIR",
    *,
    parent: Optional[str],
    version_id: Optional[str] = None,
    tile_size: float = 10.0,
    source_session_ids=None,
    incremental_result: "Optional[IncrementalUpdateResult]" = None,
) -> PartitionedVersion:
    """Persists `world` WITHOUT writing a whole-world JSON blob: only
    tile artifacts (reusing the parent's unaffected tiles verbatim) and
    one small residual artifact. Write order is crash-safe by
    construction, same principle as `save_version_tiled`: tile artifacts
    and the residual artifact are content-addressed and written before
    anything references them; the tile manifest is written next; the
    version record -- the only thing that makes this version
    "discoverable" at all -- is written LAST. A crash at any point
    before the version record lands leaves orphan (harmless) artifacts
    and no trace of this version id, so the parent version is
    untouched and fully valid.

    `incremental_result`, when given, both drives tile reuse (same as
    `save_version_tiled`) and supplies `changed_entity_ids`/
    `changed_geometry_ids` directly -- no need to load the parent's
    full world just to diff it, which would defeat the point of an
    O(affected) save path.
    """
    vid = version_id or f"v-{uuid.uuid4().hex[:12]}"
    record_path = _partitioned_record_path(store, vid)
    if record_path.exists():
        raise WorldStoreError(
            f"version {vid} already exists -- versions are immutable; "
            "save a new version instead of overwriting observed reality"
        )

    reuse_from = None
    if parent is not None:
        try:
            reuse_from = _resolve_manifest(store, parent)  # see save_version_tiled's identical comment
        except WorldStoreError:
            reuse_from = None  # parent has no tile manifest -- nothing to reuse from

    rebuilt = incremental_result.rebuilt_tile_ids if incremental_result is not None else None
    manifest = build_tile_manifest(
        world, version_id=vid, parent=parent, store=store,
        tile_size=tile_size, reuse_from=reuse_from, rebuilt_tile_keys=rebuilt,
    )

    residual_payload = _serialize_residual_blob(world, manifest)
    residual_uri, residual_hash = store._store.put(residual_payload)
    manifest = TileManifest(
        version_id=manifest.version_id, world_id=manifest.world_id, parent=manifest.parent,
        tile_size=manifest.tile_size, coordinate_frame=manifest.coordinate_frame,
        world_version=manifest.world_version,
        tiles=manifest.tiles, unlocalized_entity_ids=manifest.unlocalized_entity_ids,
        residual_uri=residual_uri, residual_hash=residual_hash,
    )

    # Tile manifest written only after every tile artifact AND the
    # residual artifact it references are already durable.
    _atomic_write_text(_manifest_path(store, vid), json.dumps(manifest.to_dict(), indent=2))

    changed_entity_ids = tuple(sorted(incremental_result.changed_entity_ids)) if incremental_result else ()
    changed_geometry_ids = tuple(sorted(incremental_result.changed_geometry_ids)) if incremental_result else ()
    # SAME shape `StoredVersion(**record)` expects (worldstore/store.py)
    # -- `WorldStore.list_versions()`/`ancestors()`/`parents()` read this
    # file with no awareness a partitioned version even exists.
    # artifact_uri/artifact_hash are "" -- the sentinel that tells
    # `WorldStore.load_version()` there is no whole-world blob to load;
    # use `load_version_partitioned()` for this version instead.
    record = {
        "version_id": vid,
        "world_id": world.id,
        "parent": parent,
        "artifact_uri": "",
        "artifact_hash": "",
        "changed_entity_ids": list(changed_entity_ids),
        "changed_geometry_ids": list(changed_geometry_ids),
        "source_session_ids": list(source_session_ids or []),
    }
    _atomic_write_text(record_path, json.dumps(record, indent=2))

    seq_path = store._root / "sequence.json"
    from worldstore.store import _SequenceLock  # reuse the SAME cross-process mutex WorldStore.save_version uses
    with _SequenceLock(store._root):
        order = store._read_sequence(seq_path)
        order.append(vid)
        _atomic_write_text(seq_path, json.dumps(order))

    return PartitionedVersion(
        version_id=vid, world_id=world.id, parent=parent,
        residual_uri=residual_uri, residual_hash=residual_hash,
        changed_entity_ids=changed_entity_ids, changed_geometry_ids=changed_geometry_ids,
        source_session_ids=tuple(source_session_ids or ()),
    )


def save_version_partitioned_delta(
    store: WorldStore,
    world: "WorldIR",
    *,
    parent: str,
    version_id: Optional[str] = None,
    tile_size: float = 10.0,
    source_session_ids=None,
    incremental_result: "Optional[IncrementalUpdateResult]" = None,
) -> PartitionedVersion:
    """The true O(affected) persistence path: writes a `TileManifestDelta`
    (only changed/new tile refs + vacated tile keys) instead of a full
    `TileManifest` -- the tile-manifest-is-O(tile-count) limitation
    `save_version_partitioned` still has does not apply here. `parent`
    is required (unlike `save_version_partitioned`): a delta chain must
    start from SOME resolvable manifest (full or delta); use
    `save_version_tiled`/`save_version_partitioned` for the first
    version of a world.

    Same crash-safety ordering as `save_version_partitioned`: tile
    artifacts and the residual artifact are written and content-
    addressed before the delta file references them; the delta is
    written before the version record; the version record is written
    last and is the only thing that makes this version discoverable.
    """
    vid = version_id or f"v-{uuid.uuid4().hex[:12]}"
    record_path = _partitioned_record_path(store, vid)
    if record_path.exists():
        raise WorldStoreError(
            f"version {vid} already exists -- versions are immutable; "
            "save a new version instead of overwriting observed reality"
        )

    parent_manifest = _resolve_manifest(store, parent)  # raises WorldStoreError with a clear message if unresolvable
    rebuilt = incremental_result.rebuilt_tile_ids if incremental_result is not None else None
    delta = build_tile_manifest_delta(
        world, version_id=vid, parent=parent, store=store,
        tile_size=tile_size, parent_manifest=parent_manifest, rebuilt_tile_keys=rebuilt,
    )

    # The residual blob still needs the FULL unlocalized-entity set
    # (which the delta already carries in full, not as a diff -- it's
    # typically small) -- build a throwaway full-shaped manifest just
    # for `_serialize_residual_blob`'s signature, no extra I/O.
    residual_payload = _serialize_residual_blob(
        world,
        TileManifest(
            version_id=vid, world_id=delta.world_id, parent=parent,
            tile_size=delta.tile_size, coordinate_frame=delta.coordinate_frame,
            world_version=delta.world_version, tiles=(),
            unlocalized_entity_ids=delta.unlocalized_entity_ids,
        ),
    )
    residual_uri, residual_hash = store._store.put(residual_payload)
    delta = TileManifestDelta(
        version_id=delta.version_id, parent=delta.parent, world_id=delta.world_id,
        tile_size=delta.tile_size, coordinate_frame=delta.coordinate_frame,
        world_version=delta.world_version, changed_tiles=delta.changed_tiles,
        removed_tile_keys=delta.removed_tile_keys, unlocalized_entity_ids=delta.unlocalized_entity_ids,
        residual_uri=residual_uri, residual_hash=residual_hash,
    )

    # Delta written only after every tile artifact AND the residual
    # artifact it references are already durable.
    _atomic_write_text(_delta_path(store, vid), json.dumps(delta.to_dict(), indent=2))

    changed_entity_ids = tuple(sorted(incremental_result.changed_entity_ids)) if incremental_result else ()
    changed_geometry_ids = tuple(sorted(incremental_result.changed_geometry_ids)) if incremental_result else ()
    record = {
        "version_id": vid,
        "world_id": world.id,
        "parent": parent,
        "artifact_uri": "",
        "artifact_hash": "",
        "changed_entity_ids": list(changed_entity_ids),
        "changed_geometry_ids": list(changed_geometry_ids),
        "source_session_ids": list(source_session_ids or []),
    }
    _atomic_write_text(record_path, json.dumps(record, indent=2))

    seq_path = store._root / "sequence.json"
    from worldstore.store import _SequenceLock
    with _SequenceLock(store._root):
        order = store._read_sequence(seq_path)
        order.append(vid)
        _atomic_write_text(seq_path, json.dumps(order))

    return PartitionedVersion(
        version_id=vid, world_id=world.id, parent=parent,
        residual_uri=residual_uri, residual_hash=residual_hash,
        changed_entity_ids=changed_entity_ids, changed_geometry_ids=changed_geometry_ids,
        source_session_ids=tuple(source_session_ids or ()),
    )


def load_version_partitioned(store: WorldStore, version_id: str) -> "WorldIR":
    """Reconstructs a full WorldIR from a partitioned version: the
    residual artifact (global state + unlocalized entities) merged with
    every tile artifact's entities/geometries. Raises WorldStoreError
    if `version_id` was not saved via `save_version_partitioned` (its
    manifest carries no residual reference) or if any artifact fails
    its hash check."""
    handle = open_version(store, version_id)  # raises WorldStoreError if no tile manifest at all
    if not handle.manifest.residual_uri:
        raise WorldStoreError(
            f"version {version_id} was not saved via save_version_partitioned() "
            "-- use WorldStore.load_version() for a whole-world version instead"
        )

    residual_payload = store._store.get(handle.manifest.residual_uri)
    if hashlib.sha256(residual_payload).hexdigest() != handle.manifest.residual_hash:
        raise WorldStoreError(
            f"version {version_id} residual artifact hash mismatch -- stored "
            "bytes are corrupted or tampered"
        )
    data = json.loads(residual_payload.decode("utf-8"))

    for ref in handle.manifest.tiles:
        entities, geometries = handle._load_ref(ref)
        for eid, entity in entities.items():
            data["entities"][eid] = entity.to_dict()
        for gid, geometry in geometries.items():
            data["geometries"][gid] = geometry.to_dict()

    from world_ir.world_v1 import WorldIR
    return WorldIR.from_dict(data)
