"""Content-addressed artifact storage for application uploads.

Files are stored by sha256 under STORAGE_ROOT. Raw filesystem paths are
never exposed to clients; artifacts are referenced by URI of the form
``sha256://<hash>`` and served through controlled API routes.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path


def storage_root() -> Path:
    # Always absolute: store_bytes() and resolve_artifact() must agree on
    # the layout, and a relative root made them disagree (stored path
    # relative vs resolved absolute), breaking round-trip equality.
    root = os.environ.get("STORAGE_ROOT")
    base = Path(root) if root else Path("./data/artifacts")
    return base.resolve()


def store_bytes(data: bytes) -> tuple[str, Path]:
    """Store bytes content-addressed. Returns (hash, absolute path)."""
    digest = hashlib.sha256(data).hexdigest()
    root = storage_root()
    obj_dir = root / digest[:2] / digest[2:4]
    obj_dir.mkdir(parents=True, exist_ok=True)
    path = obj_dir / digest
    if not path.exists():
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)
    return digest, path


def staging_root() -> Path:
    """Directory for in-flight resumable upload chunks (not yet content-addressed)."""
    root = storage_root()
    staging = root / "staging"
    staging.mkdir(parents=True, exist_ok=True)
    return staging


def stage_chunk(upload_id: str, offset: int, data: bytes) -> int:
    """Append ``data`` to the staged partial upload for ``upload_id`` at ``offset``.

    Returns the new offset (bytes staged so far). Raises ``ValueError`` if the
    staged file does not exist or the offset does not match (client and server
    out of sync — resume from the returned offset instead).
    """
    stage = staging_root() / f"{upload_id}.partial"
    if not stage.exists():
        raise ValueError(f"no staged upload for {upload_id}")
    current = stage.stat().st_size
    if offset != current:
        raise ValueError(f"offset {offset} does not match staged size {current}")
    if offset == 0 and current == 0:
        # Fresh staging file.
        with stage.open("wb") as fh:
            fh.write(data)
            return len(data)
    with stage.open("ab") as fh:
        fh.write(data)
        return fh.tell()


def assemble_staged(upload_id: str) -> tuple[str, Path]:
    """Finalize a staged partial upload: hash and return (digest, path) without
    yet moving it into the content-addressed store. The caller persists the
    digest/URI and clears staging."""
    stage = staging_root() / f"{upload_id}.partial"
    if not stage.exists():
        raise ValueError(f"no staged upload for {upload_id}")
    data = stage.read_bytes()
    return hashlib.sha256(data).hexdigest(), stage


def clear_staging(upload_id: str) -> None:
    """Remove staged bytes for ``upload_id`` (no-op if none)."""
    stage = staging_root() / f"{upload_id}.partial"
    stage.unlink(missing_ok=True)


def resolve_artifact(uri: str) -> Path | None:
    """Resolve a sha256:// URI to a file path, validating the layout.
    Returns None for unknown/unresolvable artifacts.

    Path traversal protection: ensures the resolved path is within STORAGE_ROOT
    by using Path.resolve() and checking is_relative_to()."""
    if not uri.startswith("sha256://"):
        return None
    digest = uri.removeprefix("sha256://")
    if not digest or any(c not in "0123456789abcdef" for c in digest):
        return None
    root = storage_root().resolve()
    path = (root / digest[:2] / digest[2:4] / digest).resolve()
    # Ensure the resolved path is within the storage root (prevents path traversal)
    try:
        path.relative_to(root)
    except ValueError:
        return None
    if not path.exists():
        return None
    return path


# --------------------------------------------------------------------------- reference-checked blob GC


def blob_digest(uri: str | None) -> str | None:
    """The sha256 digest of a ``sha256://`` URI, or None for anything else (including malformed digests)."""
    if not uri or not uri.startswith("sha256://"):
        return None
    digest = uri.removeprefix("sha256://")
    return digest if digest and all(c in "0123456789abcdef" for c in digest) else None


async def referenced_blob_digests(db) -> set[str]:
    """Every ``sha256://`` digest any row still references (uploads, evidence, version points/cameras, thumbnails).
    A blob is garbage only if it is absent here -- add a column to this list whenever a new table stores a blob URI."""
    from sqlalchemy import select

    from apps.api.models import Asset, Evidence, Upload, World, WorldVersion

    found: set[str] = set()
    for col in (Upload.artifact_uri, Evidence.artifact_uri, WorldVersion.points_artifact_uri,
                WorldVersion.cameras_artifact_uri, World.thumbnail_uri, Asset.artifact_uri):
        for (uri,) in (await db.execute(select(col).where(col.like("sha256://%")).distinct())).all():
            d = blob_digest(uri)
            if d:
                found.add(d)
    return found


async def gc_blobs(db, candidates: set[str]) -> dict:
    """Delete the candidate blobs that no row references any more; blobs that are still referenced survive.

    Call it AFTER the rows that held the candidates were deleted and the transaction committed (so a rollback never
    costs a file). Only candidates are ever considered -- an unreferenced blob nobody asked about (an export, an
    in-flight upload) is left alone. Returns ``{"removed": [...], "kept_shared": [...]}``.
    ponytail: an upload of identical bytes racing between the reference check and the unlink could lose its blob;
    close with a per-digest lock if concurrent purge + re-upload of the same photo ever matters."""
    still_used = await referenced_blob_digests(db)
    removed, kept = [], []
    for digest in sorted(candidates):
        if digest in still_used:
            kept.append(digest)
            continue
        path = resolve_artifact(f"sha256://{digest}")
        if path is not None:
            path.unlink(missing_ok=True)
        removed.append(digest)
    return {"removed": removed, "kept_shared": kept}


def measure_storage_usage() -> dict:
    """Measure real filesystem storage usage in bytes for artifacts, worldstore, and database (Task 11.1)."""
    art_root = storage_root()
    art_bytes = 0
    art_count = 0
    if art_root.exists():
        for p in art_root.rglob("*"):
            if p.is_file():
                try:
                    art_bytes += p.stat().st_size
                    art_count += 1
                except (OSError, FileNotFoundError):
                    pass

    ws_root = Path(os.environ.get("WORLDSTORE_ROOT", "./data/worldstore")).resolve()
    ws_bytes = 0
    if ws_root.exists():
        for p in ws_root.rglob("*"):
            if p.is_file():
                try:
                    ws_bytes += p.stat().st_size
                except (OSError, FileNotFoundError):
                    pass

    db_bytes = 0
    db_url = os.environ.get("DATABASE_URL", "sqlite+aiosqlite:///app.db")
    if "sqlite" in db_url:
        path_str = db_url.split("sqlite:///")[-1].split("sqlite+aiosqlite:///")[-1]
        db_path = Path(path_str).resolve()
        if db_path.exists() and db_path.is_file():
            try:
                db_bytes = db_path.stat().st_size
            except (OSError, FileNotFoundError):
                pass

    return {
        "artifacts_bytes": art_bytes,
        "artifacts_count": art_count,
        "worldstore_bytes": ws_bytes,
        "database_bytes": db_bytes,
        "total_bytes": art_bytes + ws_bytes + db_bytes,
    }
