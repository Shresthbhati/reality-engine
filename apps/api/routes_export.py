"""Routes: world export (Phase 8 of the golden loop).

Thin HTTP wrapper around sdk.reality.export() -- the real gltf/usda/
blender/cityjson/citygml exporters (exporters/). No export logic lives
here; this module resolves the version's WorldIR, calls the real
exporter, and stores the result content-addressed the same way
points.ply/cameras.json already are (apps/api/storage.py).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api import worldstore_service
from apps.api.db import get_db
from apps.api.tenancy import world_scope
from apps.api.auth import AuthContext, get_auth_context
from apps.api.routes_worlds import _load_world_or_409, _version_row
from apps.api.storage import resolve_artifact, store_bytes
from sdk import reality
from world_ir.artifact_store import FileArtifactStore

export = APIRouter(prefix="/api/worlds/{world_id}/export", tags=["export"], dependencies=[Depends(world_scope)])

_MEDIA_TYPES = {
    "gltf": "model/gltf+json",
    "glb": "model/gltf-binary",
    "usda": "text/plain",
    "blender": "application/x-python",
    "cityjson": "application/json",
    "citygml": "application/xml",
    "ifc": "application/x-step",
    "geojson": "application/geo+json",
    "sdf": "application/xml",
    "tscn": "text/plain",
    "unreal": "application/x-python",
    "floorplan_svg": "image/svg+xml",
    "floorplan_dxf": "image/vnd.dxf",
    "room_report_pdf": "application/pdf",
    "room_schedule_csv": "text/csv",
    "dae": "model/vnd.collada+xml",
    "obj": "model/obj",
    "ply": "application/x-ply",
    "stl": "model/stl",
}
# profile bundles (multi-file, may hold binary) are stored as one JSON document; see _bundle_json
_MEDIA_TYPES.update({name: "application/json" for name in reality.EXPORT_PROFILES})


_STATE_OF_PROVENANCE = {
    "OBSERVED": "measured",
    "MEASURED": "measured",
    "RECONSTRUCTED": "measured",
    "INFERRED": "inferred",
    "ESTIMATED": "estimated",
    "GENERATED": "estimated",
}


def _element_report(world, ids) -> list:
    """Per exported element: what it is, whether it was measured or inferred, and how sure the engine is (T-2.7)."""
    rows = []
    for eid in ids:
        ent = world.entities.get(eid)
        if ent is None:
            continue
        prov = getattr(ent.provenance, "name", str(ent.provenance)).upper()
        unc = getattr(ent, "uncertainty", None)
        rows.append({
            "id": eid, "name": ent.name, "type": ent.type.value,
            "state": _STATE_OF_PROVENANCE.get(prov, "unknown"),
            "confidence": None if ent.confidence is None else round(float(ent.confidence), 3),
            "note": (getattr(unc, "note", None) or None),
        })
    return rows


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


@export.get("/formats")
async def export_formats(
    world_id: str, version: str | None = None, db: AsyncSession = Depends(get_db),
) -> dict:
    """Which formats this world can export right now. An unavailable format
    carries the one-line reason (no version yet, optional library missing) so
    the UI can disable it with an explanation instead of failing on click."""
    import importlib.util

    world_reason: str | None = None
    try:
        await _version_row(db, world_id, version)
    except HTTPException as exc:
        if exc.detail == "World not found":
            raise
        world_reason = str(exc.detail)

    def reason_for(name: str) -> str | None:
        if world_reason:
            return world_reason
        if name == "ifc" and importlib.util.find_spec("ifcopenshell") is None:
            return "IfcOpenShell is not installed on this server"
        return None

    names = sorted(reality.EXPORT_FORMATS | reality.EXPORT_PROFILES)
    return {
        "formats": [
            {"format": n, "available": reason_for(n) is None, "reason": reason_for(n)} for n in names
        ]
    }


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
    elements = _element_report(world, report.entities_exported)
    by_state: dict = {}
    for r in elements:
        by_state[r["state"]] = by_state.get(r["state"], 0) + 1
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
        "elements": elements,
        "evidence_summary": {"total": len(elements), "by_state": by_state},
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


@export.get("/package")
async def export_package(
    world_id: str,
    version: str | None = None,
    db: AsyncSession = Depends(get_db),
) -> Response:
    """Export the version's WorldIR as a versioned .zip package archive (Task 11.2)."""
    from apps.api.routes_misc import _safe_filename
    from world_ir.serialization import export_world_package_zip

    row = await _version_row(db, world_id, version)
    world = _load_world_or_409(row)

    artifact_store = FileArtifactStore(worldstore_service.worldstore_root() / "pipeline-artifacts")
    zip_bytes = export_world_package_zip(world, artifact_store=artifact_store)
    filename = _safe_filename(f"{world_id}_v{world.version}.zip")

    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )



class CompareExportRequest(BaseModel):
    format: str
    version_a: str = "as-built"
    version_b: str


@export.post("/compare")
async def export_options_compare(
    world_id: str,
    body: CompareExportRequest,
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context),
) -> dict:
    """Export two design options (e.g. as-built survey vs option) side by side (T-4b.3)."""
    import json
    row_a = await _version_row(db, world_id, body.version_a, auth=auth)
    row_b = await _version_row(db, world_id, body.version_b, auth=auth)

    world_a = _load_world_or_409(row_a)
    world_b = _load_world_or_409(row_b)

    artifact_store = FileArtifactStore(worldstore_service.worldstore_root() / "pipeline-artifacts")

    def _do_export(world):
        if body.format in reality.EXPORT_PROFILES:
            files, rep = reality.export_profile(world, body.format, artifact_store=artifact_store)
            content = _bundle_json(files)
        else:
            content, rep = reality.export(world, body.format, artifact_store=artifact_store)
        if isinstance(content, dict):
            data = json.dumps(content, indent=2, sort_keys=True).encode("utf-8")
        elif isinstance(content, str):
            data = content.encode("utf-8")
        else:
            data = content
        digest, _ = store_bytes(data)
        elements = _element_report(world, rep.entities_exported)
        return digest, rep, elements

    digest_a, rep_a, elem_a = _do_export(world_a)
    digest_b, rep_b, elem_b = _do_export(world_b)

    return {
        "world_id": world_id,
        "format": body.format,
        "option_a": {
            "version_id": row_a.id,
            "label": row_a.label or body.version_a,
            "artifact_uri": f"sha256://{digest_a}",
            "download_url": f"/api/worlds/{world_id}/export/{digest_a}/download?format={body.format}",
            "content_hash": rep_a.content_hash,
            "entities_exported": list(rep_a.entities_exported),
            "evidence_summary": {"total": len(elem_a)},
        },
        "option_b": {
            "version_id": row_b.id,
            "label": row_b.label or body.version_b,
            "artifact_uri": f"sha256://{digest_b}",
            "download_url": f"/api/worlds/{world_id}/export/{digest_b}/download?format={body.format}",
            "content_hash": rep_b.content_hash,
            "entities_exported": list(rep_b.entities_exported),
            "evidence_summary": {"total": len(elem_b)},
        },
        "side_by_side": True,
    }
