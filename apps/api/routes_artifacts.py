"""Route: world-scoped read of content-addressed artifacts (texture atlases referenced as ``artifact://<sha256>``)."""

from __future__ import annotations

import re

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.db import get_db
from apps.api.tenancy import world_scope
from apps.api.routes_worlds import _load_world_or_409, _version_row, _worldstore_root
from world_ir.artifact_store import FileArtifactStore

artifacts = APIRouter(prefix="/api/worlds", tags=["worlds"], dependencies=[Depends(world_scope)])

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _referenced_hashes(world) -> set[str]:
    """sha256 digests the world's entities reference as appearance textures."""
    out: set[str] = set()
    for ent in (getattr(world, "entities", None) or {}).values():
        props = getattr(ent, "custom_properties", None) or {}
        tex = (props.get("appearance") or {}).get("texture") if isinstance(props.get("appearance"), dict) else None
        if not isinstance(tex, dict):
            continue
        uri = tex.get("uri")
        if isinstance(uri, str) and uri.startswith("artifact://"):
            out.add(uri[len("artifact://"):])
        h = tex.get("hash")
        if isinstance(h, str):
            out.add(h)
    return out


@artifacts.get("/{world_id}/artifacts/{sha256}")
async def world_artifact(
    world_id: str, sha256: str, version: str | None = None, db: AsyncSession = Depends(get_db)
) -> Response:
    if not _SHA256.match(sha256):
        raise HTTPException(422, "sha256 must be 64 lowercase hex characters")
    row = await _version_row(db, world_id, version)
    if sha256 not in _referenced_hashes(_load_world_or_409(row)):
        raise HTTPException(404, "Artifact not referenced by this world")
    root = _worldstore_root()
    raw = None
    # pipeline-artifacts is where the compile pipeline writes; artifacts is the legacy store.
    for store in (FileArtifactStore(root / "pipeline-artifacts"), FileArtifactStore(root / "artifacts")):
        try:
            raw = store.get(FileArtifactStore.uri_for(sha256))
            break
        except KeyError:
            continue
    if raw is None:
        raise HTTPException(404, "Artifact not found")
    return Response(
        content=raw,
        media_type="image/png" if raw.startswith(_PNG_MAGIC) else "application/octet-stream",
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )
