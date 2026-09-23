"""Routes: worlds (WorldStore remains authoritative for computational data)."""

from __future__ import annotations

import os
import struct
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.db import get_db
from apps.api.models import (
    ActivityEvent,
    Location,
    Session,
    World,
    WorldVersion,
    new_id,
)


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
    """Application mirror of WorldStore lineage. Empty until computation
    actually creates versions -- never synthesized here."""
    w = await db.get(World, world_id)
    if w is None:
        raise HTTPException(404, "World not found")
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
    store = None
    for g in geoms.values():
        data_uri = getattr(g, "data_uri", None)
        pts: list[tuple[float, float, float]] = []
        if data_uri:
            if store is None:
                from world_ir.artifact_store import FileArtifactStore

                store = FileArtifactStore(_worldstore_root() / "artifacts")
            try:
                raw = store.get(data_uri)
            except Exception:
                raw = b""
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
    return _parse_ply_xyz(raw)
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
