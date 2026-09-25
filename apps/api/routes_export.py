"""Routes: world export (Phase 8 of the golden loop).

Thin HTTP wrapper around sdk.reality.export() -- the real gltf/usda/
blender/cityjson/citygml exporters (exporters/). No export logic lives
here; this module resolves the version's WorldIR, calls the real
exporter, and stores the result content-addressed the same way
points.ply/cameras.json already are (apps/api/storage.py).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api import worldstore_service
from apps.api.db import get_db
from apps.api.routes_worlds import _load_world_or_409, _version_row
from apps.api.storage import resolve_artifact, store_bytes
from sdk import reality
from world_ir.artifact_store import FileArtifactStore

export = APIRouter(prefix="/api/worlds/{world_id}/export", tags=["export"])

_MEDIA_TYPES = {
    "gltf": "model/gltf+json",
    "usda": "text/plain",
    "blender": "application/x-python",
    "cityjson": "application/json",
    "citygml": "application/xml",
}


class ExportRequest(BaseModel):
    format: str
    version: str | None = None


@export.post("")
async def export_world(
    world_id: str, body: ExportRequest, db: AsyncSession = Depends(get_db),
) -> dict:
    """Export the version's WorldIR to a real format via sdk.reality.export().
    422 for an unsupported format (never a silent no-op); the exporter's
    own skip reasons are returned verbatim, never hidden."""
    row = await _version_row(db, world_id, body.version)
    world = _load_world_or_409(row)

    artifact_store = FileArtifactStore(worldstore_service.worldstore_root() / "pipeline-artifacts")
    try:
        content, report = reality.export(world, body.format, artifact_store=artifact_store)
    except reality.UnsupportedExportFormatError as exc:
        raise HTTPException(422, str(exc)) from exc

    if isinstance(content, dict):
        import json

        data = json.dumps(content, indent=2, sort_keys=True).encode("utf-8")
    elif isinstance(content, str):
        data = content.encode("utf-8")
    else:
        data = content

    digest, _ = store_bytes(data)
    return {
        "version_id": row.id,
        "format": body.format,
        "artifact_uri": f"sha256://{digest}",
        "download_url": f"/api/worlds/{world_id}/export/{digest}/download?format={body.format}",
        "content_hash": report.content_hash,
        "entities_exported": list(report.entities_exported),
        "entities_skipped": list(report.entities_skipped),
        "skip_reasons": list(report.skip_reasons),
    }


@export.get("/{content_hash}/download")
async def download_export(world_id: str, content_hash: str, format: str) -> FileResponse:
    """Stream back a previously exported artifact by its content hash."""
    path = resolve_artifact(f"sha256://{content_hash}")
    if path is None:
        raise HTTPException(404, "Export artifact missing from store")
    media_type = _MEDIA_TYPES.get(format, "application/octet-stream")
    filename = f"{world_id}.{format}"
    return FileResponse(path, media_type=media_type, filename=filename)
