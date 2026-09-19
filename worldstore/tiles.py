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
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, FrozenSet, Optional, Tuple

from world_ir.schema_v1 import Entity, Geometry
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
    tiles: Tuple[TileArtifactRef, ...]
    unlocalized_entity_ids: Tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "version_id": self.version_id,
            "world_id": self.world_id,
            "parent": self.parent,
            "tile_size": self.tile_size,
            "coordinate_frame": self.coordinate_frame,
            "tiles": [t.to_dict() for t in self.tiles],
            "unlocalized_entity_ids": list(self.unlocalized_entity_ids),
        }

    @staticmethod
    def from_dict(data: dict) -> "TileManifest":
        return TileManifest(
            version_id=data["version_id"],
            world_id=data["world_id"],
            parent=data.get("parent"),
            tile_size=data["tile_size"],
            coordinate_frame=data["coordinate_frame"],
            tiles=tuple(TileArtifactRef.from_dict(t) for t in data["tiles"]),
            unlocalized_entity_ids=tuple(data.get("unlocalized_entity_ids", ())),
        )


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
        tiles=tuple(sorted(refs, key=lambda r: r.tile_key)),
        unlocalized_entity_ids=tuple(sorted(tiles.unlocalized_entity_ids)),
    )


def _manifest_path(store: WorldStore, version_id: str) -> Path:
    return _manifests_dir(store) / f"{version_id}.json"


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
            reuse_from = _read_manifest(store, parent)
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
                return self._load_ref(ref)
        raise WorldStoreError(f"unknown tile {tile_key} in version {self.version_id}")

    def _load_ref(self, ref: TileArtifactRef) -> Dict[str, Entity]:
        payload = self._store._store.get(ref.artifact_uri)
        digest = hashlib.sha256(payload).hexdigest()
        if digest != ref.content_hash:
            raise WorldStoreError(
                f"tile {ref.tile_key} in version {self.version_id} artifact hash "
                "mismatch -- stored tile bytes are corrupted or tampered"
            )
        data = json.loads(payload.decode("utf-8"))
        return {eid: Entity.from_dict(d) for eid, d in data["entities"].items()}

    def query_region(
        self, bounds_min: Tuple[float, float, float], bounds_max: Tuple[float, float, float],
    ) -> Dict[str, Entity]:
        """Loads ONLY the tiles whose manifest bounds overlap the query
        region, then filters to entities within it. A query touching
        one tile never loads the rest of the world."""
        out: Dict[str, Entity] = {}
        for ref in self.manifest.tiles:
            if any(ref.bounds_max[i] < bounds_min[i] or ref.bounds_min[i] > bounds_max[i] for i in range(3)):
                continue  # tile's AABB doesn't overlap the query region at all
            out.update(self._load_ref(ref))
        return out


def open_version(store: WorldStore, version_id: str) -> WorldVersionHandle:
    """Opens a version for lazy tile access: reads ONLY the tile
    manifest, not the whole world. Raises WorldStoreError if this
    version was never saved with a tile manifest (see save_version_tiled)."""
    manifest = _read_manifest(store, version_id)
    return WorldVersionHandle(store, manifest)
