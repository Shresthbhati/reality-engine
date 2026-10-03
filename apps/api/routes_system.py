"""Routes: what the system has verified, and what this machine can run right now (engine.core.verification)."""

from __future__ import annotations

from fastapi import APIRouter

from engine.core import verification

system = APIRouter(prefix="/api/system", tags=["system"])


@system.get("/verification")
async def get_verification() -> dict:
    """The verification registry (IMPLEMENTED / SYNTHETICALLY VERIFIED / REAL-WORLD VERIFIED, with EXTERNAL
    VERIFICATION PENDING items listed beside each capability) plus a probe-only runtime availability report."""
    return {**verification.report(), "runtime": [a.to_dict() for a in verification.runtime_availability()]}
