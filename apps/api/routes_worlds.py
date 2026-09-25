"""Routes: worlds (WorldStore remains authoritative for computational data)."""

from __future__ import annotations

import json
import logging
import math
import os
import struct
from pathlib import Path

log = logging.getLogger("reality.api.worlds")

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.db import get_db
from apps.api.models import (
    ActivityEvent,
    Evidence,
    Location,
    Session,
    World,
    WorldVersion,
    new_id,
)
from world_ir.diff import diff_worlds
from world_ir.schema_v1 import EntityType
from world_ir.world_v1 import WorldIR
from world_ir.artifact_store import FileArtifactStore
from worldstore.store import WorldStore, WorldStoreError


def _worldstore_root() -> Path:
    """Same root the reconstruct job persists WorldStore versions under."""
    return Path(os.environ.get("WORLDSTORE_ROOT", "./data/worldstore"))


worlds = APIRouter(prefix="/api/worlds", tags=["worlds"])


class WorldIn(BaseModel):
    name: str
    description: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    coverage: dict | None = None


@worlds.post("", status_code=201)
async def create_world(body: WorldIn, db: AsyncSession = Depends(get_db)) -> dict:
    w = World(
        id=new_id("wld"),
        name=body.name,
        description=body.description,
        latitude=body.latitude,
        longitude=body.longitude,
        coverage=body.coverage,
    )
    db.add(w)
    db.add(
        ActivityEvent(
            id=new_id("act"),
            type="world.created",
            entity_type="world",
            entity_id=w.id,
            summary=f"World '{w.name}' created",
        )
    )
    await db.commit()
    await db.refresh(w)
    return {"id": w.id, "name": w.name}


@worlds.get("")
async def list_worlds(db: AsyncSession = Depends(get_db)) -> dict:
    res = await db.execute(select(World).order_by(World.updated_at.desc()).limit(200))
    ses_counts = dict(
        (await db.execute(
            select(Session.world_id, func.count()).group_by(Session.world_id)
        )).all()
    )
    return {
        "items": [
            {
                "id": w.id,
                "name": w.name,
                "description": w.description,
                "status": w.status,
                "latitude": w.latitude,
                "longitude": w.longitude,
                "current_version_id": w.current_version_id,
                "session_count": ses_counts.get(w.id, 0),
                "created_at": w.created_at.isoformat() if w.created_at else None,
            }
            for w in res.scalars().all()
        ]
    }


@worlds.get("/{world_id}")
async def get_world(world_id: str, db: AsyncSession = Depends(get_db)) -> dict:
    w = await db.get(World, world_id)
    if w is None:
        raise HTTPException(404, "World not found")
    session_count = (await db.execute(
        select(func.count()).select_from(Session).where(Session.world_id == world_id)
    )).scalar_one()
    return {
        "id": w.id,
        "name": w.name,
        "description": w.description,
        "status": w.status,
        "latitude": w.latitude,
        "longitude": w.longitude,
        "current_version_id": w.current_version_id,
        "session_count": session_count,
        "created_at": w.created_at.isoformat() if w.created_at else None,
    }


@worlds.get("/{world_id}/versions")
async def list_world_versions(world_id: str, db: AsyncSession = Depends(get_db)) -> dict:
    """Application mirror of WorldStore lineage. Reconciled against the
    on-disk store on every read (writers outside this process, or a
    version orphaned by a failed commit, must not drift silently from
    what is listed) -- never synthesized here. A corrupted store degrades
    to the DB mirror instead of failing the whole listing; the corrupt
    version itself still fails explicitly (409) when read."""
    w = await db.get(World, world_id)
    if w is None:
        raise HTTPException(404, "World not found")
    try:
        from apps.api import worldstore_service

        await worldstore_service.resync_versions(db, world_id)
    except Exception as exc:
        # A corrupted/unreadable store must not take the whole listing
        # down: fall back to the DB mirror. Per-version reads still fail
        # explicitly (409) via _load_world_or_409.
        log.warning("versions resync degraded for world %s: %s", world_id, exc)
    res = await db.execute(
        select(WorldVersion).where(WorldVersion.world_id == world_id).order_by(WorldVersion.created_at.desc())
    )
    return {
        "items": [
            {
                "id": v.id,
                "world_id": v.world_id,
                "parent_version_id": v.parent_version_id,
                "artifact_uri": v.artifact_uri,
                "artifact_hash": v.artifact_hash,
                "source_session_ids": v.source_session_ids or [],
                "changed_entity_ids": v.changed_entity_ids or [],
                "created_at": v.created_at.isoformat() if v.created_at else None,
                "is_current": v.id == w.current_version_id,
            }
            for v in res.scalars().all()
        ]
    }


@worlds.post("/{world_id}/attach/{session_id}")
async def attach_session(
    world_id: str, session_id: str, db: AsyncSession = Depends(get_db)
) -> dict:
    w = await db.get(World, world_id)
    s = await db.get(Session, session_id)
    if w is None or s is None:
        raise HTTPException(404, "World or Session not found")
    s.world_id = w.id
    await db.commit()
    return {"session_id": s.id, "world_id": w.id}


@worlds.get("/{world_id}/coverage")
async def world_coverage(world_id: str, db: AsyncSession = Depends(get_db)) -> dict:
    w = await db.get(World, world_id)
    if w is None:
        raise HTTPException(404, "World not found")
    res = await db.execute(
        select(Location).where(
            Location.entity_type == "session",
            Location.entity_id.in_(
                select(Session.id).where(Session.world_id == world_id)
            ),
        )
    )
    points = [
        {"lat": l.latitude, "lon": l.longitude, "accuracy": l.accuracy}
        for l in res.scalars().all()
    ]
    if w.coverage:
        return {"available": True, "coverage": w.coverage, "session_points": points}
    if points:
        return {"available": True, "coverage": None, "session_points": points}
    return {"available": False, "reason": "No spatial data captured for this world"}


# --------------------------------------------------------------------------
# Computational surfaces (WorldStore is authoritative)
#
# The Studio's spatial workstation fetches a world's actual reconstruction
# through these endpoints. Everything reads from the version lineage the
# reconstruct job created; a world with no versions has nothing to show and
# says so explicitly (no fabrication, no placeholder geometry). A corrupted
# artifact is a 409, never an empty success.
# --------------------------------------------------------------------------


async def _version_row(
    db: AsyncSession, world_id: str, version_id: str | None
) -> WorldVersion:
    """The mirror row for the world's effective version (current unless
    ?version= overrides). 404 world / 404 version / explicit None."""
    w = await db.get(World, world_id)
    if w is None:
        raise HTTPException(404, "World not found")
    if version_id is not None:
        row = await db.get(WorldVersion, version_id)
        if row is None or row.world_id != world_id:
            raise HTTPException(
                404, f"Version '{version_id}' not found for world {world_id}"
            )
        return row
    if not w.current_version_id:
        raise HTTPException(
            404, "World has no reconstruction versions yet - run reconstruction first"
        )
    row = await db.get(WorldVersion, w.current_version_id)
    if row is None:
        raise HTTPException(
            409,
            f"Current version '{w.current_version_id}' has no version record",
        )
    if row.world_id != world_id:
        raise HTTPException(
            409,
            f"Current version '{w.current_version_id}' does not belong to world {world_id}",
        )
    return row


def _load_world_or_409(row: WorldVersion):
    """Parse the version's WorldStore artifact. Corruption/absence is a 409
    with the store's own diagnosis -- never a silently empty world."""
    from worldstore.store import WorldStore, WorldStoreError

    store = WorldStore(_worldstore_root())
    try:
        return store.load_version(row.id)
    except (WorldStoreError, FileNotFoundError) as exc:
        raise HTTPException(409, f"Version '{row.id}' artifact unreadable: {exc}")


@worlds.get("/{world_id}/worldir")
async def world_worldir(
    world_id: str, version: str | None = None, db: AsyncSession = Depends(get_db)
) -> dict:
    """The world's compiled WorldIR (current version unless ?version=)."""
    row = await _version_row(db, world_id, version)
    world = _load_world_or_409(row)
    return world.to_dict()


@worlds.get("/{world_id}/entities/{entity_id}/provenance")
async def entity_provenance(
    world_id: str, entity_id: str, version: str | None = None,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Trace an entity back to the Evidence it was built from.

    The compile pipeline (engine.pipeline.vertical_slice) does not yet
    stamp per-observation evidence links on every entity it produces --
    only the version's source_session_ids record which capture sessions
    fed it. So tracing has two honest precision levels, tried in order:

      1. observation: an Observation on the entity carries a data_hash
         that matches a real Evidence.checksum (or a data_uri matching
         Evidence.artifact_uri) -- exact, file-level provenance, for
         whichever pipeline stage populates it.
      2. session: no per-observation match exists, so this falls back to
         every Evidence row belonging to the version's source sessions --
         still real, just coarser (this entity came from *this session's*
         capture, not a specific frame within it).

    A PROCEDURAL entity (e.g. a generated room) has no source sessions at
    all -- trace_level is "none", never a fabricated session guess.
    """
    row = await _version_row(db, world_id, version)
    world = _load_world_or_409(row)
    entity = world.entities.get(entity_id)
    if entity is None:
        raise HTTPException(404, f"Entity '{entity_id}' not found in version '{row.id}'")

    observation_matches: list[dict] = []
    for obs in getattr(entity, "observations", None) or []:
        ev = None
        if obs.data_hash:
            res = await db.execute(select(Evidence).where(Evidence.checksum == obs.data_hash))
            ev = res.scalar_one_or_none()
        if ev is None and obs.data_uri:
            res = await db.execute(select(Evidence).where(Evidence.artifact_uri == obs.data_uri))
            ev = res.scalar_one_or_none()
        if ev is not None:
            observation_matches.append({
                "observation_id": obs.id,
                "evidence_id": ev.id,
                "evidence_name": ev.name,
                "evidence_type": ev.type,
            })

    if observation_matches:
        return {
            "entity_id": entity_id,
            "version_id": row.id,
            "provenance": entity.provenance.value if entity.provenance else None,
            "trace_level": "observation",
            "evidence": observation_matches,
        }

    session_ids = row.source_session_ids or []
    if session_ids:
        res = await db.execute(select(Evidence).where(Evidence.session_id.in_(session_ids)))
        session_evidence = [
            {"evidence_id": e.id, "evidence_name": e.name, "evidence_type": e.type,
             "session_id": e.session_id}
            for e in res.scalars().all()
        ]
        return {
            "entity_id": entity_id,
            "version_id": row.id,
            "provenance": entity.provenance.value if entity.provenance else None,
            "trace_level": "session",
            "source_session_ids": session_ids,
            "evidence": session_evidence,
        }

    return {
        "entity_id": entity_id,
        "version_id": row.id,
        "provenance": entity.provenance.value if entity.provenance else None,
        "trace_level": "none",
        "evidence": [],
        "reason": "This version has no source capture sessions (procedurally generated, or predates session tracking).",
    }


@worlds.get("/{world_id}/report")
async def world_report(
    world_id: str, version: str | None = None, db: AsyncSession = Depends(get_db)
) -> dict:
    """The version's compile-pipeline report (current version unless
    ?version=) -- the real report.json stages, never fabricated."""
    row = await _version_row(db, world_id, version)
    if row.report is None:
        raise HTTPException(404, f"No pipeline report recorded for version '{row.id}'")
    return row.report


@worlds.get("/{world_id}/points")
async def world_points(
    world_id: str, version: str | None = None, db: AsyncSession = Depends(get_db)
) -> Response:
    """The version's reconstruction points as one PLY stream, rendered from
    the geometries' own artifact bytes (real data only)."""
    row = await _version_row(db, world_id, version)
    world = _load_world_or_409(row)
    ply = _render_points_ply(world)
    if ply is None:
        raise HTTPException(
            404,
            "Version has no point geometry - reconstruction produced no renderable points",
        )
    return Response(
        content=ply,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{row.id}_points.ply"'},
    )


def _render_points_ply(world) -> bytes | None:
    """Collect point positions from the world's geometries into one ASCII
    PLY. Each geometry's data_uri artifact is decoded with the canonical
    PointCloudData reader (RESPC001) when it carries one; geometries
    without a resolvable payload are skipped and counted in a comment
    header. Returns None when nothing usable exists (honest empty, not
    zeros)."""
    geoms = getattr(world, "geometries", None) or {}
    positions: list[tuple[float, float, float]] = []
    skipped = 0
    stores = None
    for g in geoms.values():
        data_uri = getattr(g, "data_uri", None)
        pts: list[tuple[float, float, float]] = []
        if data_uri:
            if stores is None:
                from world_ir.artifact_store import FileArtifactStore

                # Geometry payloads live in the pipeline-artifacts store
                # the compile pipeline was given; the WorldStore artifacts
                # store is the legacy location. Consult both so a version
                # renders regardless of which store its geometries reference.
                stores = [
                    FileArtifactStore(_worldstore_root() / "pipeline-artifacts"),
                    FileArtifactStore(_worldstore_root() / "artifacts"),
                ]
            raw = b""
            for store in stores:
                try:
                    raw = store.get(data_uri)
                except Exception:
                    continue
                break
            pts = _decode_point_payload(raw)
        if pts:
            positions.extend(pts)
        else:
            skipped += 1
    if not positions:
        return None

    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"comment reality-engine world points; geometries={len(geoms)} "
        f"skipped={skipped}\n"
        f"element vertex {len(positions)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "end_header\n"
    ).encode("ascii")
    body = "".join(
        f"{x:.6f} {y:.6f} {z:.6f}\n" for x, y, z in positions
    ).encode("ascii")
    return header + body


def _decode_point_payload(raw: bytes) -> list[tuple[float, float, float]]:
    """Decode a geometry artifact's point positions. Canonical Reality
    Engine payloads (RESPC001 PointCloudData) take the exact reader; a
    PLY fallback covers backend-parsed artifacts. Empty for anything
    else -- bytes are never guessed into coordinates."""
    if raw[:8] == b"RESPC001":
        from world_ir.geometry_data import PointCloudData

        try:
            cloud = PointCloudData.from_bytes(raw)
        except ValueError:
            return []
        return [(p[0], p[1], p[2]) for p in cloud.points]
    # PLY fallback: parse the header-declared vertex layout (ascii and
    # binary_little_endian); anything else decodes to empty, never crash.
    head = raw[:4096]
    end = head.find(b"end_header")
    if end == -1:
        return []
    header = head[:end].decode("ascii", "replace")
    body = raw[end + len(b"end_header"):].lstrip(b"\r\n")
    binary = "format binary_little_endian" in header
    n = 0
    props: list[str] = []
    for line in header.splitlines():
        if line.startswith("element vertex"):
            n = int(line.split()[-1])
        elif line.startswith("property"):
            props.append(line.split()[-1])
    if n <= 0 or not {"x", "y", "z"} <= set(props):
        return []
    out: list[tuple[float, float, float]] = []
    if binary:
        stride = 4 * len(props)
        xi, yi, zi = props.index("x"), props.index("y"), props.index("z")
        for i in range(n):
            off = i * stride
            if off + stride > len(body):
                break
            vals = struct.unpack_from("<" + "f" * len(props), body, off)
            out.append((vals[xi], vals[yi], vals[zi]))
        return out
    count = 0
    for raw_line in body.splitlines():
        if count >= n:
            break
        parts = raw_line.split()
        if len(parts) < len(props):
            continue
        vals = dict(zip(props, parts))
        try:
            out.append((float(vals["x"]), float(vals["y"]), float(vals["z"])))
            count += 1
        except ValueError:
            continue
    return out


@worlds.get("/{world_id}/cameras")
async def world_cameras(
    world_id: str, version: str | None = None, db: AsyncSession = Depends(get_db)
) -> dict:
    """Registered camera poses for the version, in the field shape the
    Studio's camera layer consumes (position_m / rotation_wxyz, world
    frame, +Y up, camera-to-world quaternions)."""
    row = await _version_row(db, world_id, version)
    world = _load_world_or_409(row)

    cameras = []
    image_size = None
    for ent in (world.entities or {}).values():
        obs = getattr(ent, "observations", None) or []
        for ob in obs:
            meta = getattr(ob, "metadata", None)
            if not isinstance(meta, dict):
                continue
            pose = meta.get("camera_pose")
            if isinstance(pose, dict) and "position" in pose:
                cameras.append({
                    "evidence_id": meta.get("evidence_id") or ent.id,
                    "position_m": [float(c) for c in pose["position"]],
                    "rotation_wxyz": [
                        float(c) for c in pose.get("rotation", [1.0, 0.0, 0.0, 0.0])
                    ],
                })
            if image_size is None and isinstance(meta.get("image_size"), (list, tuple)):
                image_size = [int(v) for v in meta["image_size"]]
    return {
        "frame": "world (meters, +Y up after frame canonicalization)",
        "rotation_convention": "camera-to-world quaternion (w, x, y, z)",
        "image_size": image_size,
        "version_id": row.id,
        "cameras": cameras,
    }


class DiffRequest(BaseModel):
    base_version: str | None = None
    head_version: str | None = None


@worlds.get("/{world_id}/diff")
async def get_world_diff(world_id: str, base: str | None = None, head: str | None = None, db: AsyncSession = Depends(get_db)):
    """Compute structural diff between two WorldIR versions."""
    w = await db.get(World, world_id)
    if w is None:
        raise HTTPException(404, "World not found")
    
    store_path = _worldstore_root()
    if not store_path.exists():
        raise HTTPException(404, "WorldStore not found")
    store = WorldStore(str(store_path))
    
    # Default to current version as head
    head_version = head or w.current_version_id
    if not head_version:
        raise HTTPException(404, "No current version to diff")
    
    # Default to parent of head as base
    if base is None:
        head_record = store._record(head_version)
        base = head_record.get("parent")
    
    if not base:
        raise HTTPException(404, "No base version available for diff")
    
    try:
        before = store.load_version(base)
        after = store.load_version(head_version)
    except Exception as e:
        raise HTTPException(404, f"Version not found: {e}")
    
    world_diff = diff_worlds(before, after)
    return world_diff.to_dict()


# Fields a correction commit may change on an entity. Everything else
# (id, geometry/material/surface/component linkage, provenance,
# observations, relationships, ...) is rejected: silently ignoring an
# unsupported field while still minting a version made failed corrections
# look successful, and setattr on an arbitrary attribute could corrupt
# entity invariants (e.g. overwriting `id` or `type` with a plain string).
_COMMIT_MUTABLE_FIELDS = (
    "name",
    "type",
    "confidence",
    "semantic_labels",
    "custom_properties",
    "transform",
)
_MAX_COMMIT_CHANGES = 32
_MAX_COMMIT_VALUE_BYTES = 16 * 1024


class CommitRequest(BaseModel):
    entity_id: str
    changes: dict
    parent_version_id: str | None = None
    commit_message: str | None = None


def _validate_commit_changes(changes: dict) -> dict:
    """Validate a correction payload. Returns sanitized {field: value};
    raises 422/413 naming the offending field -- never silently drops."""
    if not isinstance(changes, dict) or not changes:
        raise HTTPException(422, "changes must be a non-empty object")
    if len(changes) > _MAX_COMMIT_CHANGES:
        raise HTTPException(
            413, f"too many changed fields ({len(changes)} > {_MAX_COMMIT_CHANGES})"
        )
    validated: dict = {}
    for key, value in changes.items():
        if key not in _COMMIT_MUTABLE_FIELDS:
            raise HTTPException(
                422,
                f"unsupported mutation field '{key}'; mutable fields are "
                f"{list(_COMMIT_MUTABLE_FIELDS)}",
            )
        try:
            size = len(json.dumps(value, sort_keys=True, default=str).encode("utf-8"))
        except (TypeError, ValueError):
            raise HTTPException(422, f"field '{key}' is not JSON-serializable")
        if size > _MAX_COMMIT_VALUE_BYTES:
            raise HTTPException(
                413, f"field '{key}' exceeds {_MAX_COMMIT_VALUE_BYTES} bytes"
            )
        if key == "name":
            if not isinstance(value, str) or not value or len(value) > 300:
                raise HTTPException(
                    422, "field 'name' must be a non-empty string of at most 300 chars"
                )
            validated[key] = value
        elif key == "type":
            try:
                validated[key] = EntityType(value)
            except ValueError:
                raise HTTPException(
                    422,
                    f"field 'type' must be one of {[t.value for t in EntityType]}",
                )
        elif key == "confidence":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise HTTPException(422, "field 'confidence' must be a number")
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise HTTPException(
                    422, "field 'confidence' must be finite and within [0, 1]"
                )
            validated[key] = float(value)
        elif key == "semantic_labels":
            if (
                not isinstance(value, list)
                or len(value) > 100
                or not all(isinstance(v, str) for v in value)
            ):
                raise HTTPException(
                    422, "field 'semantic_labels' must be a list of at most 100 strings"
                )
            validated[key] = list(value)
        elif key in ("custom_properties", "transform"):
            if not isinstance(value, dict):
                raise HTTPException(422, f"field '{key}' must be an object")
            validated[key] = dict(value)
    return validated


@worlds.post("/{world_id}/commit")
async def commit_world_correction(world_id: str, body: CommitRequest, db: AsyncSession = Depends(get_db)):
    """Apply a correction to a world entity and persist as new WorldStore version.

    Failure contract: validation failures (422/413), stale parents (409),
    and store failures (404/409) all return BEFORE anything is persisted --
    a failed commit never moves HEAD, never writes a mirror row, and never
    looks successful. A byte-identical retry returns the current version
    (duplicate: true) instead of minting a duplicate version.
    """
    w = await db.get(World, world_id)
    if w is None:
        raise HTTPException(404, "World not found")

    # Load current WorldIR
    worldir = await _load_worldir_from_store(w)
    if worldir is None:
        raise HTTPException(404, "WorldIR not available for this world's current version")

    # Apply the entity change
    entity = worldir.entities.get(body.entity_id)
    if entity is None:
        raise HTTPException(404, f"Entity {body.entity_id} not found in world")

    validated = _validate_commit_changes(body.changes)
    # Unify identifiers (same invariant worldstore_service.commit_version
    # enforces): the WorldIR's own id is the application world id, so
    # on-disk versions filter by world and the DB mirror stays truthful.
    worldir.id = w.id
    base_snapshot = worldir.to_dict()
    for key, value in validated.items():
        if key == "name":
            entity.name = value
        elif key == "type":
            entity.type = value
        elif key == "confidence":
            entity.confidence = value
        elif key == "semantic_labels":
            entity.semantic_labels = value
        elif key == "custom_properties":
            entity.custom_properties = value
        elif key == "transform":
            entity.transform = value

    # Malformed WorldIR must never be persisted by a correction: validate
    # the corrected world and refuse with the validator's own diagnosis.
    # Only ERRORs block (warnings stay readable, matching the recon path).
    from world_ir.validation import validate_world_ir as _validate_world

    _validation = _validate_world(worldir)
    if not _validation.is_valid():
        raise HTTPException(
            422,
            "corrected world failed validation: "
            + "; ".join(_validation.messages()[:5]),
        )

    # Stale-client guard: an explicit parent that is no longer HEAD is a
    # 409, not a silent fork -- the caller must rebase onto HEAD.
    effective_parent = body.parent_version_id or w.current_version_id
    if body.parent_version_id is not None and body.parent_version_id != w.current_version_id:
        raise HTTPException(
            409,
            f"parent version '{body.parent_version_id}' is stale; "
            f"current HEAD is '{w.current_version_id}' -- reload and reapply",
        )

    # Duplicate-request guard: a byte-identical retry (same content, same
    # parent) returns the adopted version instead of minting a duplicate.
    # modified_at is wall-clock bookkeeping, excluded from the comparison.
    import time
    new_snapshot = worldir.to_dict()
    base_cmp = dict(base_snapshot)
    new_cmp = dict(new_snapshot)
    base_cmp.pop("modified_at", None)
    new_cmp.pop("modified_at", None)
    if (
        effective_parent is not None
        and effective_parent == w.current_version_id
        and json.dumps(base_cmp, sort_keys=True) == json.dumps(new_cmp, sort_keys=True)
    ):
        return {
            "version_id": w.current_version_id,
            "world_id": w.id,
            "entity_id": body.entity_id,
            "changed_fields": [],
            "duplicate": True,
        }
    worldir.modified_at = time.time()

    # Save as new version
    store_path = _worldstore_root()
    if not store_path.exists():
        store_path.mkdir(parents=True, exist_ok=True)
    store = WorldStore(str(store_path))

    try:
        stored = store.save_version(worldir, parent=effective_parent)
    except WorldStoreError as exc:
        msg = str(exc)
        if "unknown version" in msg:
            raise HTTPException(404, msg)
        raise HTTPException(409, msg)

    # Atomic adoption: advance HEAD only if it still holds the parent
    # this commit was based on. The conditional UPDATE is a single
    # atomic statement, so two concurrent commits cannot both win -- the
    # loser's rowcount is 0 and it gets an explicit 409 while its saved
    # version file stays on disk (harmless, never adopted). An in-memory
    # re-check could not close this race: concurrent requests use
    # separate sessions whose commits serialize after the check.
    if effective_parent is None:  # defensive: unreachable (404 above)
        raise HTTPException(404, "World has no HEAD version to commit onto")
    from sqlalchemy import update as _sa_update

    head_update = await db.execute(
        _sa_update(World)
        .where(World.id == w.id, World.current_version_id == effective_parent)
        .values(current_version_id=stored.version_id)
    )
    if head_update.rowcount == 0:
        current = (await db.get(World, w.id)).current_version_id
        raise HTTPException(
            409,
            f"HEAD moved during commit (now '{current}'); "
            f"version '{stored.version_id}' was saved but not adopted -- "
            "reload and reapply",
        )

    # Adopt atomically: HEAD pointer + DB mirror row + activity share one
    # commit, so a version is never HEAD without its mirror record (which
    # previously made committed worlds unreadable with 409).
    db.add(
        WorldVersion(
            id=stored.version_id,
            world_id=w.id,
            parent_version_id=stored.parent,
            artifact_uri=stored.artifact_uri,
            artifact_hash=stored.artifact_hash,
            source_session_ids=list(stored.source_session_ids),
            changed_entity_ids=list(stored.changed_entity_ids),
        )
    )
    db.add(
        ActivityEvent(
            id=new_id("act"),
            type="world.committed",
            entity_type="world",
            entity_id=w.id,
            summary=f"Committed correction to {body.entity_id}: {body.commit_message or 'no message'}",
        )
    )

    try:
        await db.commit()
    except Exception as exc:
        # The version file may already exist on disk (harmless orphan,
        # never adopted, visible to resync) -- but HEAD never moved and
        # no mirror row exists, so report explicitly instead of a bare 500.
        raise HTTPException(
            503, f"database unavailable; commit not adopted: {exc}"
        )

    return {
        "version_id": stored.version_id,
        "world_id": w.id,
        "entity_id": body.entity_id,
        "changed_fields": list(validated.keys()),
        "duplicate": False,
    }


async def _load_worldir_from_store(world: World, version_id: str | None = None) -> WorldIR | None:
    """Load WorldIR from WorldStore for a World (current or specified version)."""
    vid = version_id or world.current_version_id
    if not vid:
        return None
    store_path = _worldstore_root()
    if not store_path.exists():
        return None
    store = WorldStore(str(store_path))
    try:
        return store.load_version(vid)
    except Exception:
        return None


def _get_worldir_space_graph(worldir: WorldIR) -> dict:
    """Extract or derive interior space graph from WorldIR metadata and entities."""
    if "interior_space_graph" in worldir.metadata:
        return worldir.metadata["interior_space_graph"]

    levels = []
    rooms = []
    corridors = []
    openings = []
    stairs = []

    for eid, entity in worldir.entities.items():
        etype = entity.type.value if hasattr(entity.type, "value") else str(entity.type)
        if etype == "level":
            levels.append({
                "level_id": eid,
                "elevation_m": entity.custom_properties.get("elevation_m", 0.0),
                "room_ids": list(entity.custom_properties.get("room_ids", [])),
                "corridor_ids": list(entity.custom_properties.get("corridor_ids", [])),
                "stair_ids": list(entity.custom_properties.get("stair_ids", [])),
            })
        elif etype == "room":
            rooms.append({
                "room_id": eid,
                "floor_area_m2": entity.custom_properties.get("floor_area_m2", 0.0),
                "adjacent_room_ids": list(entity.custom_properties.get("adjacent_room_ids", [])),
                "corridor_ids": list(entity.custom_properties.get("corridor_ids", [])),
                "boundary_completeness": entity.custom_properties.get("boundary_completeness", 1.0),
                "notes": list(entity.custom_properties.get("notes", [])),
                "status": entity.custom_properties.get("status", "detected"),
            })
        elif etype == "corridor":
            corridors.append({
                "corridor_id": eid,
                "aspect_ratio": entity.custom_properties.get("aspect_ratio", 2.0),
                "width_m": entity.custom_properties.get("width_m", 1.2),
                "length_m": entity.custom_properties.get("length_m", 3.0),
                "height_m": entity.custom_properties.get("height_m", 2.4),
                "floor_area_m2": entity.custom_properties.get("floor_area_m2", 3.6),
                "connected_room_ids": list(entity.custom_properties.get("connected_room_ids", [])),
                "longitudinal_axis": list(entity.custom_properties.get("longitudinal_axis", [1.0, 0.0, 0.0])),
                "bounds_min": list(entity.custom_properties.get("bounds_min", [0.0, 0.0, 0.0])),
                "bounds_max": list(entity.custom_properties.get("bounds_max", [0.0, 0.0, 0.0])),
            })
        elif etype in ("door", "window"):
            openings.append({
                "opening_id": eid,
                "kind": "window" if etype == "window" else "doorway",
                "width_m": entity.custom_properties.get("width_m", 0.9),
                "height_m": entity.custom_properties.get("height_m", 2.1),
                "sill_height_m": entity.custom_properties.get("sill_height_m", 0.0),
                "connected_space_ids": list(entity.custom_properties.get("connected_space_ids", [])),
                "is_exterior": entity.custom_properties.get("is_exterior", etype == "window"),
            })
        elif etype == "stairs":
            stairs.append({
                "stair_id": eid,
                "step_count": entity.custom_properties.get("step_count", 0),
                "total_rise_m": entity.custom_properties.get("total_rise_m", 0.0),
                "total_run_m": entity.custom_properties.get("total_run_m", 0.0),
                "connected_level_ids": list(entity.custom_properties.get("connected_level_ids", [])),
            })

    for rel in worldir.relationships.values():
        rkind = rel.kind.value if hasattr(rel.kind, "value") else str(rel.kind)
        if rkind == "connects":
            for op in openings:
                if op["opening_id"] == rel.source_id and rel.target_id not in op["connected_space_ids"]:
                    op["connected_space_ids"].append(rel.target_id)
        elif rkind == "part_of":
            for lvl in levels:
                if lvl["level_id"] == rel.target_id:
                    if "room" in rel.source_id and rel.source_id not in lvl["room_ids"]:
                        lvl["room_ids"].append(rel.source_id)
                    elif "corridor" in rel.source_id and rel.source_id not in lvl["corridor_ids"]:
                        lvl["corridor_ids"].append(rel.source_id)

    return {
        "building_id": worldir.id,
        "levels": levels,
        "rooms": rooms,
        "corridors": corridors,
        "openings": openings,
        "stairs": stairs,
        "summary": {
            "level_count": len(levels),
            "room_count": len(rooms),
            "corridor_count": len(corridors),
            "opening_count": len(openings),
            "stair_count": len(stairs),
        },
    }


@worlds.get("/{world_id}/space-graph")
async def get_world_space_graph(
    world_id: str,
    version_id: str | None = None,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Retrieve the authoritative InteriorSpaceGraph for a world."""
    w = await db.get(World, world_id)
    if w is None:
        raise HTTPException(404, "World not found")

    worldir = await _load_worldir_from_store(w, version_id)
    if worldir is None:
        raise HTTPException(404, "WorldIR not available for this world/version")

    return _get_worldir_space_graph(worldir)


@worlds.get("/{world_id}/query/spaces")
async def query_world_spaces(
    world_id: str,
    room_id: str | None = None,
    corridor_id: str | None = None,
    level_id: str | None = None,
    space_id: str | None = None,
    space_a: str | None = None,
    space_b: str | None = None,
    level_a: str | None = None,
    level_b: str | None = None,
    version_id: str | None = None,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Query topological connectivity and spatial metrics within a world."""
    w = await db.get(World, world_id)
    if w is None:
        raise HTTPException(404, "World not found")

    worldir = await _load_worldir_from_store(w, version_id)
    if worldir is None:
        raise HTTPException(404, "WorldIR not available for this world/version")

    graph_dict = _get_worldir_space_graph(worldir)
    from perception.architecture.space_graph import InteriorSpaceGraph
    graph = InteriorSpaceGraph.from_dict(graph_dict)

    if corridor_id:
        return {
            "query": "corridor_reachability",
            "corridor_id": corridor_id,
            "reachable_rooms": graph.rooms_reachable_from_corridor(corridor_id),
        }
    if room_id:
        return {
            "query": "connected_rooms",
            "room_id": room_id,
            "connected_spaces": graph.rooms_connected_to(room_id),
        }
    if space_a and space_b:
        return {
            "query": "openings_connecting",
            "space_a": space_a,
            "space_b": space_b,
            "openings": graph.openings_connecting(space_a, space_b),
        }
    if level_a and level_b:
        return {
            "query": "stairs_connecting_levels",
            "level_a": level_a,
            "level_b": level_b,
            "stairs": graph.stairs_connecting_levels(level_a, level_b),
        }
    if space_id:
        return {
            "query": "space_info",
            "space_id": space_id,
            "level": graph.level_of_space(space_id),
            "evidence": graph.trace_space_evidence(space_id),
        }
    if level_id:
        return {
            "query": "level_spaces",
            "level_id": level_id,
            "level": graph.levels.get(level_id),
        }

    return {"query": "all", "space_graph": graph_dict}

