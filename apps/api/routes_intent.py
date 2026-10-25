"""Routes: World Intent (what the user is trying to construct) and an honest resource estimate.

    PUT/GET /api/worlds/{id}/intent     persisted scale + capabilities (advisory)
    POST    /api/intent/estimate        resource profile for an intent + a set of files

Honesty rule: every profile field is {value | null, basis, kind: detected|derived|unknown}. A number is only
returned when it was detected on this machine or derived from this database's own completed jobs. Nothing is
extrapolated from a table of guesses.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import statistics

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.db import get_db
from apps.api.models import Evidence, Job, World, WorldIntent

SCALES = ("interior_space", "building", "street", "neighborhood", "locality", "district", "city")
CAPABILITIES = ("geometry", "semantic_understanding", "measurement", "evidence_provenance",
                "temporal_state", "high_precision")
#: a video contributes at most this many sampled frames (routes_reconstructions.VIDEO_FRAME_BUDGET)
_FRAMES_PER_VIDEO = 24

intent_router = APIRouter(prefix="/api/worlds", tags=["intent"])
estimate_router = APIRouter(prefix="/api/intent", tags=["intent"])


def _check_scale(v: str) -> str:
    if v not in SCALES:
        raise ValueError(f"scale must be one of {list(SCALES)}")
    return v


def _check_caps(v: list[str]) -> list[str]:
    bad = [c for c in v if c not in CAPABILITIES]
    if bad:
        raise ValueError(f"unknown capabilities {bad}; allowed: {list(CAPABILITIES)}")
    return list(dict.fromkeys(v))  # de-duplicated, order kept


class IntentIn(BaseModel):
    scale: str
    capabilities: list[str]

    _s = field_validator("scale")(_check_scale)
    _c = field_validator("capabilities")(_check_caps)


class EstimateIn(IntentIn):
    file_count: int = Field(ge=0)
    total_bytes: int = Field(ge=0)
    video_count: int = Field(default=0, ge=0)


def _intent_out(row: WorldIntent) -> dict:
    return {"world_id": row.world_id, "scale": row.scale, "capabilities": list(row.capabilities or []),
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "updated_at": row.updated_at.isoformat() if row.updated_at else None}


@intent_router.put("/{world_id}/intent")
async def put_intent(world_id: str, body: IntentIn, db: AsyncSession = Depends(get_db)) -> dict:
    if await db.get(World, world_id) is None:
        raise HTTPException(404, "World not found")
    row = await db.get(WorldIntent, world_id)
    if row is None:
        row = WorldIntent(world_id=world_id, scale=body.scale, capabilities=body.capabilities)
        db.add(row)
    else:
        row.scale, row.capabilities = body.scale, body.capabilities
    await db.commit()
    await db.refresh(row)
    return _intent_out(row)


@intent_router.get("/{world_id}/intent")
async def get_intent(world_id: str, db: AsyncSession = Depends(get_db)) -> dict:
    if await db.get(World, world_id) is None:
        raise HTTPException(404, "World not found")
    row = await db.get(WorldIntent, world_id)
    if row is None:
        raise HTTPException(404, "No intent has been saved for this world")
    return _intent_out(row)


# ---- estimate -------------------------------------------------------------------------------------------------


def _field(value, basis: str, kind: str, **extra) -> dict:
    return {"value": value, "basis": basis, "kind": kind, **extra}


def detect_hardware() -> dict:
    """What this machine reports right now. CUDA: torch's own answer when torch is importable, else the presence of
    ``nvidia-smi`` (a driver is installed; it does not prove a usable GPU, so the basis says which one answered)."""
    cuda, cuda_basis = False, "no torch and no nvidia-smi on PATH"
    if importlib.util.find_spec("torch") is not None:
        try:
            import torch  # noqa: PLC0415 -- heavy, only when probing

            cuda = bool(torch.cuda.is_available())
            cuda_basis = "torch.cuda.is_available()"
        except Exception as exc:  # noqa: BLE001 -- a probe reports, it does not raise
            cuda_basis = f"torch could not be probed ({type(exc).__name__})"
    elif shutil.which("nvidia-smi"):
        cuda, cuda_basis = True, "nvidia-smi found on PATH (driver present; torch not installed to confirm)"
    ram = None
    try:
        import psutil  # noqa: PLC0415

        ram = int(psutil.virtual_memory().total)
    except Exception:  # noqa: BLE001
        if hasattr(os, "sysconf") and "SC_PHYS_PAGES" in getattr(os, "sysconf_names", {}):
            ram = int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    return {"cuda_gpu": cuda, "cuda_basis": cuda_basis, "colmap": shutil.which("colmap") is not None, "ram_bytes": ram}


def _warnings(scale: str, capabilities: list[str], files: int, hw: dict) -> list[str]:
    out = []
    if scale in ("district", "city") and files < 100:
        out.append(
            f"A {scale.replace('_', ' ')} is not a bigger photograph: it needs multi-source data (aerial or survey "
            f"imagery, maps, many captures). {files} file(s) is evidence for at most a single building or "
            "street-sized area (heuristic ceiling, not a measured limit)."
        )
    if not hw["colmap"]:
        out.append("COLMAP was not found on PATH, so photo reconstruction cannot run on this machine.")
    if "high_precision" in capabilities and not hw["cuda_gpu"]:
        out.append("High precision was requested but no CUDA GPU was detected; expect slower or reduced-quality runs.")
    return out


def build_estimate(scale: str, capabilities: list[str], file_count: int, total_bytes: int, video_count: int,
                   hw: dict, seconds_per_image: list[float]) -> dict:
    """Pure: the profile for given inputs, hardware facts and historical samples (seconds per image)."""
    images = max(file_count - video_count, 0) + video_count * _FRAMES_PER_VIDEO
    compute = _field({"cuda_gpu": hw["cuda_gpu"], "colmap": hw["colmap"]},
                     f"detected on this machine; CUDA: {hw['cuda_basis']}; COLMAP: PATH lookup", "detected")
    memory = (_field(hw["ram_bytes"], "installed RAM on this machine (the requirement itself is not estimated)",
                     "detected") if hw["ram_bytes"] else
              _field(None, "RAM could not be read on this machine", "unknown", reason="host RAM unavailable"))
    storage = (_field(total_bytes, "size of the selected input files only; working and temporary storage "
                      "is not estimated", "derived") if total_bytes > 0 else
               _field(None, "no files selected yet", "unknown", reason="no input size"))
    if seconds_per_image and images > 0:
        med = statistics.median(seconds_per_image)
        time = _field(round(med * images, 1),
                      f"median {med:.2f} s/image over {len(seconds_per_image)} completed run(s) in this "
                      f"database x {images} image(s) (video counted at up to {_FRAMES_PER_VIDEO} frames)",
                      "derived", sample_size=len(seconds_per_image))
    else:
        time = _field(None, "no previous runs on this machine to base an estimate on", "unknown",
                      reason="no previous runs", sample_size=0)
    return {"scale": scale, "capabilities": capabilities, "compute": compute, "memory": memory,
            "storage": storage, "time": time, "warnings": _warnings(scale, capabilities, file_count, hw)}


async def _seconds_per_image(db: AsyncSession) -> list[float]:
    jobs = (await db.execute(
        select(Job).where(Job.type == "RECONSTRUCT_SESSION", Job.status.in_(("succeeded", "partial")),
                          Job.started_at.is_not(None), Job.completed_at.is_not(None))
    )).scalars().all()
    out = []
    for j in jobs:
        n = len((await db.execute(
            select(Evidence.id).where(Evidence.session_id == j.entity_id, Evidence.type == "photo")
        )).all())
        secs = (j.completed_at - j.started_at).total_seconds()
        if n > 0 and secs > 0:
            out.append(secs / n)
    return out


@estimate_router.post("/estimate")
async def estimate(body: EstimateIn, db: AsyncSession = Depends(get_db)) -> dict:
    return build_estimate(body.scale, body.capabilities, body.file_count, body.total_bytes, body.video_count,
                          detect_hardware(), await _seconds_per_image(db))
