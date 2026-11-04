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

import hashlib
import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Optional, Union

from .world_v1 import WorldIR

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



def export_world_package_zip(
    world: WorldIR,
    output: Optional[Union[str, Path, io.BytesIO]] = None,
    artifact_store: Optional[Any] = None,
) -> Union[bytes, Path]:
    """Export a versioned WorldIR package as a .zip archive (Blueprint Task 11.2).

    Contains:
    - world.manifest: format version, checksum, entity/geom counts, geo_anchor.
    - world.ir: canonical deterministic JSON of the world state.
    - artifacts/<sha256>: optional enclosed binary geometry/texture artifacts.

    Returns the written Path or bytes if output is None or a BytesIO buffer.
    """
    ir_dict = world.to_dict()
    ir_json = json.dumps(ir_dict, indent=2, sort_keys=True)
    ir_bytes = ir_json.encode("utf-8")
    ir_hash = hashlib.sha256(ir_bytes).hexdigest()

    manifest = {
        "format_version": WORLD_SAVE_FORMAT_VERSION,
        "world_id": world.id,
        "world_name": world.name,
        "world_version": world.version,
        "created_at": world.created_at,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "ir_sha256": ir_hash,
        "entity_count": len(world.entities),
        "geometry_count": len(world.geometries),
        "material_count": len(world.materials),
        "geo_anchor": world.geo_anchor.to_dict() if getattr(world, "geo_anchor", None) else None,
    }
    manifest_bytes = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")

    buf = io.BytesIO() if output is None or isinstance(output, io.BytesIO) else None
    target_zip = buf if buf is not None else output

    with zipfile.ZipFile(target_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(MANIFEST_FILENAME, manifest_bytes)
        zf.writestr(IR_FILENAME, ir_bytes)

        if artifact_store is not None:
            uris = set()
            for g in world.geometries.values():
                if g.data_uri and ("://" in g.data_uri):
                    uris.add(g.data_uri)
            for m in world.materials.values():
                props = (
                    getattr(m, "custom_properties", None)
                    or getattr(getattr(m, "properties", None), "custom_properties", None)
                    or {}
                )
                tex = props.get("texture_uri")
                if tex and ("://" in str(tex)):
                    uris.add(str(tex))

            for uri in sorted(uris):
                digest = uri.split("://")[-1].split("/")[-1]
                try:
                    data = artifact_store.get(uri)
                    if data:
                        zf.writestr(f"artifacts/{digest}", data)
                except Exception:
                    pass

    if buf is not None:
        return buf.getvalue()
    return Path(output)


def import_world_package_zip(
    source: Union[str, Path, bytes, io.BytesIO],
    artifact_store: Optional[Any] = None,
) -> WorldIR:
    """Import and validate a WorldIR from a versioned .zip package archive (Blueprint Task 11.2).

    Guarantees that export -> import yields an identical world.
    Raises WorldPackageError or WorldFormatVersionError if corrupted or mismatched.
    """
    if isinstance(source, bytes):
        source_file: Union[io.BytesIO, Path, str] = io.BytesIO(source)
    else:
        source_file = source

    try:
        zf = zipfile.ZipFile(source_file, mode="r")
    except zipfile.BadZipFile as exc:
        raise WorldPackageError(f"Invalid or corrupted zip package: {exc}") from exc

    with zf:
        names = set(zf.namelist())
        if MANIFEST_FILENAME not in names:
            raise WorldPackageError(f"Package zip missing required {MANIFEST_FILENAME}")
        if IR_FILENAME not in names:
            raise WorldPackageError(f"Package zip missing required {IR_FILENAME}")

        try:
            manifest_bytes = zf.read(MANIFEST_FILENAME)
            manifest = json.loads(manifest_bytes.decode("utf-8"))
        except Exception as exc:
            raise WorldPackageError(f"Failed to parse {MANIFEST_FILENAME}: {exc}") from exc

        fmt_ver = manifest.get("format_version")
        if fmt_ver != WORLD_SAVE_FORMAT_VERSION:
            raise WorldFormatVersionError(
                f"unsupported world.manifest format_version {fmt_ver!r}, expected {WORLD_SAVE_FORMAT_VERSION}"
            )

        try:
            ir_bytes = zf.read(IR_FILENAME)
            ir_dict = json.loads(ir_bytes.decode("utf-8"))
        except Exception as exc:
            raise WorldPackageError(f"Failed to parse {IR_FILENAME}: {exc}") from exc

        expected_hash = manifest.get("ir_sha256")
        if expected_hash:
            actual_hash = hashlib.sha256(ir_bytes).hexdigest()
            if actual_hash != expected_hash:
                raise WorldPackageError(
                    f"Checksum mismatch for {IR_FILENAME}: expected {expected_hash}, got {actual_hash}"
                )

        world = WorldIR.from_dict(ir_dict)
        if manifest.get("world_id") and manifest["world_id"] != world.id:
            raise WorldPackageError(
                f"manifest world_id '{manifest['world_id']}' does not match world.ir id '{world.id}'"
            )

        if artifact_store is not None:
            for name in names:
                if name.startswith("artifacts/") and len(name) > len("artifacts/"):
                    digest = name.split("artifacts/")[-1]
                    blob_data = zf.read(name)
                    blob_hash = hashlib.sha256(blob_data).hexdigest()
                    if blob_hash == digest:
                        artifact_store.put(blob_data)

        return world
