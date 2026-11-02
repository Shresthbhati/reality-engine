"""What a place card needs, from one batched read (B-11), plus its thumbnail (B-03).

A "place" is the consumer name for a World. The numbers come from stored records only: the current version's report for
what was placed and at what scale, the evidence rows for how many photographs exist, the job table for a running build.
Nothing here is estimated.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.models import Evidence, Job, Session, World, WorldVersion
from apps.api.storage import resolve_artifact, storage_root

QUALITY_BANDS = ("good", "fair", "weak", "unknown")
THUMB_SIDE = 480


def band_and_reason(report: dict | None, photo_count: int, placed: int) -> tuple[str, str | None]:
    """A coarse, rule-based band for the whole place. It is NOT a calibrated probability (master doc 14.2).

    unknown  no model yet
    weak     fewer than half the photographs are placed, or the engine fell back below multi-view (level < 2)
    good     the build was graded SUCCESS and at least 80% of the photographs are placed
    fair     everything else
    Per-entity bands for the viewer are a separate rule (B-07).
    """
    if not report:
        return "unknown", None
    level = report.get("level")
    if photo_count and placed / photo_count < 0.5:
        return "weak", "few_placed"
    if isinstance(level, int) and level < 2:
        return "weak", "rough_model"
    if report.get("status") == "SUCCESS" and (not photo_count or placed / photo_count >= 0.8):
        return "good", None
    return "fair", "partial_build" if report.get("status") != "SUCCESS" else "some_unplaced"


def quality_band(report: dict | None, photo_count: int, placed: int) -> str:
    """The band alone; see band_and_reason for why."""
    return band_and_reason(report, photo_count, placed)[0]


def _utc_iso(dt: datetime | None) -> str | None:
    """SQLite hands back naive datetimes that are UTC; say so, or a browser reads them as local time."""
    if dt is None:
        return None
    return (dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).isoformat()


def _placed(report: dict | None) -> int:
    return len(((report or {}).get("evidence") or {}).get("registered_ids") or [])


def _scale_state(report: dict | None) -> str | None:
    return (((report or {}).get("stages") or {}).get("scale") or {}).get("state")


async def summaries(db: AsyncSession, worlds: list[World]) -> dict[str, dict]:
    """Per-world card fields for all `worlds` in four queries (no N+1)."""
    ids = [w.id for w in worlds]
    if not ids:
        return {}
    sessions = (await db.execute(select(Session.id, Session.world_id).where(Session.world_id.in_(ids)))).all()
    world_of_session = {sid: wid for sid, wid in sessions}

    photo_counts: dict[str, int] = {i: 0 for i in ids}
    if world_of_session:
        rows = (await db.execute(
            select(Evidence.session_id).where(Evidence.session_id.in_(list(world_of_session)), Evidence.type == "photo")
        )).scalars().all()
        for sid in rows:
            photo_counts[world_of_session[sid]] += 1

    version_ids = [w.current_version_id for w in worlds if w.current_version_id]
    reports: dict[str, dict] = {}
    if version_ids:
        reports = {
            v.id: (v.report or {})
            for v in (await db.execute(select(WorldVersion).where(WorldVersion.id.in_(version_ids)))).scalars().all()
        }

    active: dict[str, dict] = {}
    if world_of_session:
        jobs = (await db.execute(
            select(Job).where(
                Job.entity_type == "session", Job.entity_id.in_(list(world_of_session)),
                Job.status.in_(("queued", "running")),
            ).order_by(Job.created_at.asc())
        )).scalars().all()
        for j in jobs:
            active[world_of_session[j.entity_id]] = {"id": j.id, "status": j.status, "stage": j.stage,
                                                      "progress": j.progress}

    out: dict[str, dict] = {}
    for w in worlds:
        report = reports.get(w.current_version_id) if w.current_version_id else None
        photos = photo_counts[w.id]
        placed = _placed(report)
        out[w.id] = {
            "photo_count": photos,
            "placed_count": placed,
            "quality_band": quality_band(report, photos, placed),
            "scale_state": _scale_state(report),
            "active_job": active.get(w.id),
            "updated_at": _utc_iso(w.updated_at),
            # the version in the query string makes a new version refresh the cached image
            "thumbnail_url": f"/api/worlds/{w.id}/thumbnail?v={w.current_version_id or 'none'}" if photos else None,
        }
    return out


async def pick_thumbnail_source(db: AsyncSession, world: World) -> Evidence | None:
    """A placed photograph of the current version when there is one, otherwise the first photograph uploaded."""
    session_ids = (await db.execute(select(Session.id).where(Session.world_id == world.id))).scalars().all()
    if not session_ids:
        return None
    photos = (await db.execute(
        select(Evidence).where(Evidence.session_id.in_(session_ids), Evidence.type == "photo",
                               Evidence.artifact_uri.is_not(None)).order_by(Evidence.created_at.asc(), Evidence.id.asc())
    )).scalars().all()
    if not photos:
        return None
    if world.current_version_id:
        version = await db.get(WorldVersion, world.current_version_id)
        placed = set(((version.report or {}).get("evidence") or {}).get("registered_ids") or []) if version else set()
        for ev in photos:
            if ev.id in placed:
                return ev
    return photos[0]


def thumbnail_path(world: World, ev: Evidence) -> Path:
    return storage_root() / "thumbnails" / f"{world.id}-{world.current_version_id or 'none'}-{ev.id}.jpg"


def render_thumbnail(ev: Evidence, dest: Path) -> Path | None:
    """Downscale the photograph to THUMB_SIDE px on its long side and cache it. None when the file is unreadable."""
    from PIL import Image, ImageOps

    src = resolve_artifact(ev.artifact_uri or "")
    if src is None:
        return None
    if dest.exists():
        return dest
    try:
        with Image.open(src) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")
            img.thumbnail((THUMB_SIDE, THUMB_SIDE))
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(".tmp")
            img.save(tmp, format="JPEG", quality=82)
            tmp.replace(dest)
    except Exception:  # undecodable / truncated: no thumbnail, never an error page
        return None
    return dest


# ------------------------------------------------------------------------------------------ per-entity quality (B-07)

GOOD_AT = 0.75
FAIR_AT = 0.5


def entity_band(confidence: float | None, provenance: str | None) -> str:
    """Quality band for ONE entity. A model score is not a probability (master doc 14.2); this is a coarse reading.

    unknown  provenance is UNKNOWN, or no usable confidence: "not measured", never shown as good or bad
    weak     provenance is CONFLICT (sources disagree), or confidence below 0.5
    fair     confidence 0.5 up to 0.75
    good     confidence 0.75 or above
    """
    prov = (provenance or "").upper()
    if prov == "UNKNOWN" or confidence is None or not isinstance(confidence, (int, float)) or confidence != confidence:
        return "unknown"
    if prov == "CONFLICT" or confidence < FAIR_AT:
        return "weak"
    return "good" if confidence >= GOOD_AT else "fair"


def quality_view(world_ir, card: dict, report: dict | None) -> dict:
    """The viewer's quality payload: the place-level card fields plus a band for every entity. Derived only from stored
    fields (entity confidence and provenance, the version report); there is no per-point band because the stored point
    cloud carries no per-point confidence."""
    entities: dict[str, dict] = {}
    counts = {b: 0 for b in QUALITY_BANDS}
    for eid, e in world_ir.entities.items():
        prov = getattr(getattr(e, "provenance", None), "value", None)
        conf = getattr(e, "confidence", None)
        band = entity_band(conf, prov)
        counts[band] += 1
        entities[eid] = {"band": band, "provenance": (prov or "unknown").upper(), "confidence": conf,
                         "type": getattr(getattr(e, "type", None), "value", None), "name": getattr(e, "name", "") or ""}
    _, reason = band_and_reason(report, card["photo_count"], card["placed_count"])
    conflicts = (report or {}).get("conflicts")
    open_conflicts = int(conflicts.get("unresolved", 0)) if isinstance(conflicts, dict) else len(conflicts or [])
    return {
        "band": card["quality_band"],
        "band_reason": reason,
        "photos_total": card["photo_count"],
        "photos_placed": card["placed_count"],
        "scale_state": card["scale_state"],
        "entity_counts": counts,
        "open_conflicts": open_conflicts,
        "entities": entities,
    }
