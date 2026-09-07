"""World Save Format (spec sec 103).

The full format is a package directory with world.manifest, world.ir,
and per-subsystem directories (entities/, geometry/, materials/,
physics/, ...) that don't exist as real subsystems yet. This
implementation writes the two pieces this foundation layer actually
owns -- `world.manifest` and `world.ir` -- as a real, versioned,
round-trippable package, and leaves the rest as an explicit TODO rather
than faking empty directories that would look implemented.
"""

from __future__ import annotations

import json
from pathlib import Path

from .world import WorldIR

WORLD_SAVE_FORMAT_VERSION = 1

MANIFEST_FILENAME = "world.manifest"
IR_FILENAME = "world.ir"


class WorldFormatVersionError(ValueError):
    pass


class WorldPackageError(ValueError):
    pass


def save_world(world: WorldIR, path: str | Path) -> Path:
    """Write a world package directory containing world.manifest and
    world.ir. Overwrites an existing package at `path`."""
    package_dir = Path(path)
    package_dir.mkdir(parents=True, exist_ok=True)

    ir_dict = world.to_dict()

    manifest = {
        "format_version": WORLD_SAVE_FORMAT_VERSION,
        "world_id": world.id,
        "world_version": world.version,
        "entity_count": len(world.entities),
    }

    (package_dir / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    (package_dir / IR_FILENAME).write_text(
        json.dumps(ir_dict, indent=2, sort_keys=True), encoding="utf-8"
    )
    return package_dir


def load_world(path: str | Path) -> WorldIR:
    package_dir = Path(path)
    manifest_path = package_dir / MANIFEST_FILENAME
    ir_path = package_dir / IR_FILENAME

    if not manifest_path.exists() or not ir_path.exists():
        raise WorldPackageError(f"'{package_dir}' is not a valid world package")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != WORLD_SAVE_FORMAT_VERSION:
        raise WorldFormatVersionError(
            f"unsupported world.manifest format_version "
            f"{manifest.get('format_version')!r}, expected {WORLD_SAVE_FORMAT_VERSION}"
        )

    ir_dict = json.loads(ir_path.read_text(encoding="utf-8"))
    world = WorldIR.from_dict(ir_dict)

    if manifest.get("world_id") != world.id:
        raise WorldPackageError(
            f"manifest world_id '{manifest.get('world_id')}' does not match "
            f"world.ir id '{world.id}'"
        )

    return world
