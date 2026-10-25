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
    "ifc": "application/x-step",
    "geojson": "application/geo+json",
    "sdf": "application/xml",
    "tscn": "text/plain",
    "unreal": "application/x-python",
}
# profile bundles (multi-file, may hold binary) are stored as one JSON document; see _bundle_json
_MEDIA_TYPES.update({name: "application/json" for name in reality.EXPORT_PROFILES})


class ExportRequest(BaseModel):
    format: str
    version: str | None = None


def _bundle_json(files: dict) -> dict:
    """{filename: bytes} -> JSON-safe {filename: {"encoding", "content"}}. Same one-JSON-document bundle shape as the
    sumo/habitat formats, except a file is wrapped so binary (the GLB) survives: text stays utf-8, anything else is
    base64 (decode with base64.b64decode)."""
    import base64

    out = {}
    for name, data in files.items():
        try:
            out[name] = {"encoding": "utf-8", "content": data.decode("utf-8")}
        except UnicodeDecodeError:
            out[name] = {"encoding": "base64", "content": base64.b64encode(data).decode("ascii")}
    return out


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
        if body.format in reality.EXPORT_PROFILES:
            files, report = reality.export_profile(world, body.format, artifact_store=artifact_store)
            content = _bundle_json(files)
        else:
            content, report = reality.export(world, body.format, artifact_store=artifact_store)
    except reality.UnsupportedExportFormatError as exc:
        raise HTTPException(422, f"{exc}; profiles: {sorted(reality.EXPORT_PROFILES)}") from exc
    except ImportError as exc:      # an optional exporter dependency (IFC: IfcOpenShell) is not installed here
        raise HTTPException(501, f"{body.format} export is unavailable on this server: {exc}") from exc
    except ValueError as exc:       # nothing exportable (an empty IFC is not an export)
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
        "bundle": body.format in reality.EXPORT_PROFILES,
        "files": sorted(content) if body.format in reality.EXPORT_PROFILES else None,
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
    from apps.api.routes_misc import _safe_filename

    path = resolve_artifact(f"sha256://{content_hash}")
    if path is None:
        raise HTTPException(404, "Export artifact missing from store")
    media_type = _MEDIA_TYPES.get(format, "application/octet-stream")
    # world_id/format are request-controlled: sanitize before reflecting
    # them into Content-Disposition (header-injection hardening).
    filename = _safe_filename(f"{world_id}.{format}")
    return FileResponse(path, media_type=media_type, filename=filename)
