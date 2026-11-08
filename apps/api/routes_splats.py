"""Routes for Photorealistic Gaussian Splatting & Hybrid Viewport (Track C).

Provides endpoints to:
- Inspect Gaussian Splat metadata (count, bounds, formats, memory size).
- Stream compact 32-byte .splat or Inria 3DGS .ply binary streams with on-the-fly LOD decimation.
- Generate Gaussian Splats directly from a world's reconstruction point cloud.
- Upload custom .splat or .ply Gaussian Splat models.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, Response, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.auth import AuthContext, enforce_project_isolation, get_auth_context
from apps.api.db import get_db
from apps.api.models import World, WorldVersion
from apps.api.storage import resolve_artifact, store_bytes
from engine.pipeline.gaussian_splats import (
    GaussianSplatCloud,
    decimate_splats,
    export_ply_splats,
    export_splat_binary,
    generate_splats_from_points,
    parse_ply_splats,
    parse_splat_binary,
)

log = logging.getLogger("reality.api.splats")

splats_router = APIRouter(prefix="/api/worlds", tags=["splats"])


class SplatGenerateRequest(BaseModel):
    default_radius: float = Field(0.03, ge=0.001, le=1.0, description="Default splat radius in meters")
    max_points: Optional[int] = Field(None, ge=1, le=5000000, description="Optional maximum points limit for LOD")


async def _get_world_and_version(
    world_id: str,
    version_id: str | None,
    db: AsyncSession,
    auth: AuthContext,
) -> tuple[World, WorldVersion]:
    w = await db.get(World, world_id)
    if w is None:
        raise HTTPException(404, f"World '{world_id}' not found")
    enforce_project_isolation(w, auth)

    if version_id:
        row = await db.get(WorldVersion, version_id)
        if row is None or row.world_id != world_id:
            res = await db.execute(
                select(WorldVersion).where(
                    WorldVersion.world_id == world_id,
                    WorldVersion.label == version_id,
                ).order_by(WorldVersion.created_at.desc())
            )
            row = res.scalars().first()
        if row is None:
            raise HTTPException(404, f"Version '{version_id}' not found for world '{world_id}'")
        return w, row

    if not w.current_version_id:
        raise HTTPException(404, f"World '{world_id}' has no reconstruction versions yet")

    row = await db.get(WorldVersion, w.current_version_id)
    if row is None:
        raise HTTPException(404, f"Current version '{w.current_version_id}' not found")

    return w, row


def _load_splat_cloud_from_row(row: WorldVersion) -> GaussianSplatCloud | None:
    """Load GaussianSplatCloud from version report metadata and blob storage."""
    report = row.report or {}
    meta = report.get("gaussian_splats")
    if not meta or not meta.get("artifact_uri"):
        return None

    path = resolve_artifact(meta["artifact_uri"])
    if not path or not path.exists():
        log.warning("Splat artifact referenced at %s does not exist on disk", meta.get("artifact_uri"))
        return None

    data = path.read_bytes()
    fmt = meta.get("format", "splat")
    try:
        if fmt == "splat" or data[:3] != b"ply":
            return parse_splat_binary(data)
        else:
            return parse_ply_splats(data)
    except Exception as exc:
        log.error("Failed to parse splat artifact %s: %s", meta.get("artifact_uri"), exc)
        return None


@splats_router.get("/{world_id}/splats")
async def get_splat_metadata(
    world_id: str,
    version: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context),
) -> dict[str, Any]:
    """Retrieve metadata about Gaussian Splats for the current or specified world version."""
    w, row = await _get_world_and_version(world_id, version, db, auth)

    report = row.report or {}
    meta = report.get("gaussian_splats")

    can_generate = False
    # Check if point geometry is available in this version
    if row.points_artifact_uri:
        can_generate = True
    else:
        # Check if geometries exist in WorldIR
        try:
            from apps.api.routes_worlds import _load_world_or_409, _render_points_ply
            world_ir = _load_world_or_409(row)
            if _render_points_ply(world_ir) is not None:
                can_generate = True
        except Exception:
            pass

    if not meta or not meta.get("artifact_uri"):
        return {
            "available": False,
            "world_id": world_id,
            "version_id": row.id,
            "splat_count": 0,
            "bounds": None,
            "byte_size": 0,
            "formats": ["splat", "ply"],
            "artifact_uri": None,
            "can_generate": can_generate,
        }

    return {
        "available": True,
        "world_id": world_id,
        "version_id": row.id,
        "splat_count": meta.get("splat_count", 0),
        "bounds": meta.get("bounds"),
        "byte_size": meta.get("byte_size", 0),
        "formats": ["splat", "ply"],
        "artifact_uri": meta.get("artifact_uri"),
        "generated_from_points": meta.get("generated_from_points", False),
        "can_generate": can_generate,
        "updated_at": meta.get("updated_at"),
    }


@splats_router.get("/{world_id}/splats/stream")
async def stream_splats(
    world_id: str,
    version: Optional[str] = None,
    format: str = Query("splat", pattern="^(splat|ply)$"),
    max_points: Optional[int] = Query(None, ge=1, le=5000000),
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context),
) -> Response:
    """Stream Gaussian Splat primitives in binary .splat or 3DGS .ply format."""
    w, row = await _get_world_and_version(world_id, version, db, auth)
    cloud = _load_splat_cloud_from_row(row)

    if cloud is None or len(cloud) == 0:
        raise HTTPException(
            404,
            f"No Gaussian Splats available for world '{world_id}' version '{row.id}'. Generate or upload splats first.",
        )

    if max_points is not None and max_points < len(cloud):
        cloud = decimate_splats(cloud, target_count=max_points)

    if format == "splat":
        content = cloud.to_splat_binary()
        media_type = "application/octet-stream"
        filename = f"{world_id}_{row.id}_splats.splat"
    else:
        content = cloud.to_ply(binary=True)
        media_type = "application/octet-stream"
        filename = f"{world_id}_{row.id}_splats.ply"

    return Response(
        content=content,
        media_type=media_type,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "X-Splat-Count": str(len(cloud)),
            "X-Splat-Format": format,
            "Cache-Control": "public, max-age=3600",
        },
    )


@splats_router.post("/{world_id}/splats/generate")
async def generate_splats(
    world_id: str,
    req: SplatGenerateRequest = SplatGenerateRequest(),
    version: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context),
) -> dict[str, Any]:
    """Generate 3D Gaussian Splats from the world's existing reconstruction point cloud."""
    w, row = await _get_world_and_version(world_id, version, db, auth)

    from apps.api.routes_worlds import _load_world_or_409, _render_points_ply

    world_ir = _load_world_or_409(row)
    ply_bytes = _render_points_ply(world_ir)

    if not ply_bytes:
        raise HTTPException(
            404,
            f"World '{world_id}' has no reconstruction point geometry. Reconstruction must be completed first.",
        )

    # Parse reconstruction points
    positions: list[tuple[float, float, float]] = []
    lines = ply_bytes.decode("ascii", errors="replace").splitlines()
    in_body = False
    for line in lines:
        if line.strip() == "end_header":
            in_body = True
            continue
        if in_body and line.strip():
            parts = line.strip().split()
            if len(parts) >= 3:
                try:
                    positions.append((float(parts[0]), float(parts[1]), float(parts[2])))
                except ValueError:
                    continue

    if not positions:
        raise HTTPException(
            422,
            f"Failed to extract point coordinates from reconstruction geometry for world '{world_id}'.",
        )

    # Generate splat cloud
    cloud = generate_splats_from_points(positions, default_radius=req.default_radius)
    if req.max_points is not None and req.max_points < len(cloud):
        cloud = decimate_splats(cloud, target_count=req.max_points)

    # Store binary .splat artifact
    splat_bytes = cloud.to_splat_binary()
    digest, _ = store_bytes(splat_bytes)
    artifact_uri = f"sha256://{digest}"

    # Update version report
    report = dict(row.report or {})
    from apps.api.models import utcnow

    report["gaussian_splats"] = {
        "artifact_uri": artifact_uri,
        "format": "splat",
        "splat_count": len(cloud),
        "bounds": cloud.bounds,
        "byte_size": len(splat_bytes),
        "generated_from_points": True,
        "default_radius": req.default_radius,
        "updated_at": utcnow().isoformat(),
    }
    row.report = report
    await db.commit()

    log.info(
        "Generated %d Gaussian Splats for world %s version %s (digest %s)",
        len(cloud),
        world_id,
        row.id,
        digest,
    )

    return {
        "success": True,
        "world_id": world_id,
        "version_id": row.id,
        "splat_count": len(cloud),
        "bounds": cloud.bounds,
        "byte_size": len(splat_bytes),
        "artifact_uri": artifact_uri,
        "formats": ["splat", "ply"],
    }


@splats_router.post("/{world_id}/splats/upload")
async def upload_splats(
    world_id: str,
    file: UploadFile = File(...),
    version: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context),
) -> dict[str, Any]:
    """Upload custom .splat or .ply 3D Gaussian Splat file for a world."""
    w, row = await _get_world_and_version(world_id, version, db, auth)

    raw = await file.read()
    if len(raw) == 0:
        raise HTTPException(400, "Uploaded splat file is empty")

    filename = (file.filename or "").lower()
    is_ply = filename.endswith(".ply") or raw[:3] == b"ply"

    try:
        if is_ply:
            cloud = parse_ply_splats(raw)
        else:
            cloud = parse_splat_binary(raw)
    except Exception as exc:
        raise HTTPException(422, f"Invalid Gaussian Splat file: {exc}")

    if len(cloud) == 0:
        raise HTTPException(422, "Uploaded file contains 0 valid Gaussian splat primitives")

    # Store normalized .splat binary representation
    splat_bytes = cloud.to_splat_binary()
    digest, _ = store_bytes(splat_bytes)
    artifact_uri = f"sha256://{digest}"

    # Update version report
    report = dict(row.report or {})
    from apps.api.models import utcnow

    report["gaussian_splats"] = {
        "artifact_uri": artifact_uri,
        "format": "splat",
        "splat_count": len(cloud),
        "bounds": cloud.bounds,
        "byte_size": len(splat_bytes),
        "generated_from_points": False,
        "original_filename": file.filename,
        "updated_at": utcnow().isoformat(),
    }
    row.report = report
    await db.commit()

    log.info(
        "Uploaded and parsed %d Gaussian Splats for world %s version %s (digest %s)",
        len(cloud),
        world_id,
        row.id,
        digest,
    )

    return {
        "success": True,
        "world_id": world_id,
        "version_id": row.id,
        "splat_count": len(cloud),
        "bounds": cloud.bounds,
        "byte_size": len(splat_bytes),
        "artifact_uri": artifact_uri,
        "formats": ["splat", "ply"],
    }
