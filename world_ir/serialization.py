"""World Save Format (spec sec 103).

The full format is a package directory with world.manifest, world.ir,
and per-subsystem directories (entities/, geometry/, materials/,
physics/, ...) that don't exist as real subsystems yet. This
implementation writes the two pieces this foundation layer actually
owns -- `world.manifest` and `world.ir` -- as a real, versioned,
round-trippable package, and leaves the rest as an explicit TODO rather
than faking empty directories that would look implemented.

Two WorldIR shapes exist in this codebase (see world_ir/__init__.py):
the canonical V1 schema (`world_ir.world_v1.WorldIR`, re-exported as
`world_ir.WorldIR`) that every production caller (WorldRuntime, the
compiler, WorldStore, exporters, the SDK) actually holds, and a legacy
foundation-layer container (`world_ir.world.WorldIR`, re-exported as
`world_ir.WorldIRLegacy`) kept for the handful of callers that still
construct it directly. `save_world`/`load_world` used to be typed and
implemented only against the legacy class -- which meant saving a V1
world "worked" (both classes expose `to_dict()`) but reloading it
crashed, because V1 serializes `entities` as an id-keyed dict and the
frame under `coordinate_frame`, while the legacy loader expects a list
under `entities` and reads `coordinate_system`.

`WorldIR.to_dict()` for the V1 schema always writes a `schema_version`
key (world_v1.py); the legacy schema never does. That key is used here
as an unambiguous, retroactive discriminator -- it lets `load_world`
correctly dispatch on packages written before this fix, not just new
ones -- and is cross-checked against an explicit `ir_kind` manifest
field so a hand-edited or truncated package fails loudly instead of
guessing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

from .world import WorldIR as WorldIRLegacy
from .world_v1 import WorldIR as WorldIRV1

WorldIRAny = Union[WorldIRV1, WorldIRLegacy]

WORLD_SAVE_FORMAT_VERSION = 1

MANIFEST_FILENAME = "world.manifest"
IR_FILENAME = "world.ir"

_KIND_V1 = "v1"
_KIND_LEGACY = "legacy"


class WorldFormatVersionError(ValueError):
    pass


class WorldPackageError(ValueError):
    pass


def _kind_of(world: WorldIRAny) -> str:
    if isinstance(world, WorldIRV1):
        return _KIND_V1
    if isinstance(world, WorldIRLegacy):
        return _KIND_LEGACY
    raise WorldPackageError(
        f"unsupported WorldIR type {type(world).__name__!r}: expected "
        f"world_ir.world_v1.WorldIR (canonical) or world_ir.world.WorldIR (legacy)"
    )


def _kind_of_ir_dict(ir_dict: dict) -> str:
    # V1's to_dict() always includes "schema_version"; legacy's never does.
    return _KIND_V1 if "schema_version" in ir_dict else _KIND_LEGACY


def save_world(world: WorldIRAny, path: str | Path) -> Path:
    """Write a world package directory containing world.manifest and
    world.ir. Overwrites an existing package at `path`.

    Accepts either the canonical V1 WorldIR or the legacy foundation-layer
    WorldIR; the package records which one was written so `load_world`
    reconstructs the same type.
    """
    package_dir = Path(path)
    package_dir.mkdir(parents=True, exist_ok=True)

    kind = _kind_of(world)
    ir_dict = world.to_dict()

    manifest = {
        "format_version": WORLD_SAVE_FORMAT_VERSION,
        "ir_kind": kind,
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


def load_world(path: str | Path) -> WorldIRAny:
    """Load a world package, returning the same WorldIR type it was saved
    as (canonical V1 by default -- see module docstring)."""
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
    ir_kind = _kind_of_ir_dict(ir_dict)

    # Packages written before this fix have no "ir_kind" manifest field at
    # all -- they were legacy-only in practice, so absence means legacy,
    # never a silent guess in favor of the canonical schema.
    manifest_kind = manifest.get("ir_kind", _KIND_LEGACY)
    if manifest_kind != ir_kind:
        raise WorldPackageError(
            f"'{package_dir}' is corrupt: world.manifest declares ir_kind "
            f"'{manifest_kind}' but world.ir content is '{ir_kind}'"
        )

    world_cls = WorldIRV1 if ir_kind == _KIND_V1 else WorldIRLegacy
    world = world_cls.from_dict(ir_dict)

    if manifest.get("world_id") != world.id:
        raise WorldPackageError(
            f"manifest world_id '{manifest.get('world_id')}' does not match "
            f"world.ir id '{world.id}'"
        )

    return world
