"""Routes: what the system has verified, and what this machine can run right now (engine.core.verification)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from apps.api.validation import is_local_client
from engine.core import verification

system = APIRouter(prefix="/api/system", tags=["system"])


@system.get("/verification")
async def get_verification() -> dict:
    """The verification registry (IMPLEMENTED / SYNTHETICALLY VERIFIED / REAL-WORLD VERIFIED, with EXTERNAL
    VERIFICATION PENDING items listed beside each capability) plus a probe-only runtime availability report."""
    return {**verification.report(), "runtime": [a.to_dict() for a in verification.runtime_availability()]}


class ColmapPathIn(BaseModel):
    path: str | None = None  # None / "" clears the saved path


def _tools_report() -> dict:
    from reconstruction.backend import locate

    loc = locate.locate_colmap("colmap")
    try:
        import cv2  # noqa: PLC0415

        video = {"available": True, "backend": f"OpenCV {cv2.__version__}",
                 "detail": "video frames are sampled and decoded with OpenCV (cv2); ffmpeg is not used"}
    except Exception as exc:  # noqa: BLE001 -- a probe reports, it does not raise
        video = {"available": False, "backend": None,
                 "detail": f"OpenCV (cv2) is not importable ({type(exc).__name__}): videos cannot be read; photos still work"}
    return {
        "colmap": {**loc.to_dict(), "saved_path": locate.read_settings().get("colmap_path"),
                   "env_vars": list(locate.ENV_VARS), "settings_file": str(locate.settings_file()),
                   "message": ("COLMAP found" if loc.found else locate.not_found_message("colmap"))},
        "video": video,
    }


@system.get("/tools")
async def get_tools() -> dict:
    """What this machine can run for photo/video reconstruction: where COLMAP is (and how it was found, or exactly
    where it was looked for) and whether videos can be decoded."""
    return _tools_report()


@system.put("/tools/colmap")
async def set_colmap_path(body: ColmapPathIn, request: Request) -> dict:
    """Persist (or clear) the COLMAP location used when no environment variable overrides it. A path that does not
    exist is refused with 422 -- never saved and silently ignored later."""
    from reconstruction.backend import locate

    # the saved path is later EXECUTED by the engine: only a client on this machine may change it
    if not is_local_client(request):
        raise HTTPException(403, "the COLMAP location can only be changed from the machine the API runs on")
    raw = (body.path or "").strip()
    if raw and locate.as_binary(raw) is None:
        raise HTTPException(422, f"no colmap executable at {raw!r} (give colmap.exe, or the folder that contains it)")
    locate.write_settings({"colmap_path": raw or None})
    return _tools_report()
