"""Observability and diagnostic service for background jobs (T-6.6).

Provides:
- Structured JSON job logs with stage, level, and contextual metadata
- Root-cause error tracking and user-actionable remediation suggestions
- Per-stage timing dashboard identifying bottleneck stages
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.models import Job, JobLog, new_id, utcnow

FAILURE_REMEDIATIONS = {
    "InsufficientEvidence": (
        "Reconstruction requires at least 3-10 overlapping photographs taken around the perimeter "
        "of the room. Ensure high-contrast textured surfaces (e.g. rugs, framed art, furniture) are in view."
    ),
    "DegenerateGeometry": (
        "The photographed surfaces appear coplanar or collinear, preventing 3D triangulation. "
        "Take photos from diverse camera angles (low, eye-height, high, and corners)."
    ),
    "Timeout": (
        "The reconstruction exceeded the processing timeout threshold. "
        "Try processing fewer images at a time or enabling CPU plane-sweep fallbacks."
    ),
    "ResourceExhausted": (
        "The job ran out of memory. Try reducing photograph resolution or using batch reconstruction."
    ),
    "Cancelled": (
        "The job was cancelled by an operator or client request. A new reconstruction can be started anytime."
    ),
}


def classify_failure(error: str | None, payload: dict | None) -> tuple[str, str]:
    """Classify the root cause and remediation for a failed job."""
    err_str = (error or "").strip()
    p = payload or {}
    kind = p.get("failure_kind") or ""

    if "InsufficientEvidence" in err_str or "no usable evidence" in err_str.lower() or kind == "needs_more_evidence":
        return (
            "INSUFFICIENT_EVIDENCE",
            FAILURE_REMEDIATIONS["InsufficientEvidence"],
        )
    if "degenerate" in err_str.lower() or "coplanar" in err_str.lower():
        return (
            "DEGENERATE_GEOMETRY",
            FAILURE_REMEDIATIONS["DegenerateGeometry"],
        )
    if "timed out" in err_str.lower() or "timeout" in err_str.lower():
        return (
            "TIMEOUT",
            FAILURE_REMEDIATIONS["Timeout"],
        )
    if "memory" in err_str.lower() or "oom" in err_str.lower():
        return (
            "RESOURCE_EXHAUSTED",
            FAILURE_REMEDIATIONS["ResourceExhausted"],
        )
    if "cancelled" in err_str.lower():
        return (
            "CANCELLED_BY_OPERATOR",
            FAILURE_REMEDIATIONS["Cancelled"],
        )

    return (
        "PIPELINE_ERROR",
        "Inspect the structured job logs for details or verify input photo formatting and lighting.",
    )


async def record_job_log(
    db: AsyncSession,
    job_id: str,
    message: str,
    stage: str | None = None,
    level: str = "INFO",
    context: dict | None = None,
) -> JobLog:
    """Record a structured log entry for a job."""
    log_entry = JobLog(
        id=new_id("log"),
        job_id=job_id,
        timestamp=utcnow(),
        stage=stage,
        level=level.upper(),
        message=message,
        context_json=context,
    )
    db.add(log_entry)
    return log_entry


async def get_job_logs(
    db: AsyncSession,
    job_id: str,
    level: str | None = None,
    limit: int = 200,
) -> list[dict[str, Any]]:
    """Retrieve structured logs for a job in chronological order."""
    query = select(JobLog).where(JobLog.job_id == job_id).order_by(JobLog.timestamp.asc()).limit(limit)
    if level:
        query = query.where(JobLog.level == level.upper())
    result = await db.execute(query)
    entries = result.scalars().all()
    return [
        {
            "id": entry.id,
            "timestamp": entry.timestamp.isoformat() if entry.timestamp else None,
            "stage": entry.stage,
            "level": entry.level,
            "message": entry.message,
            "context": entry.context_json,
        }
        for entry in entries
    ]


async def get_job_diagnostics(db: AsyncSession, job: Job) -> dict[str, Any]:
    """Provide structured diagnostic report for a job."""
    logs = await get_job_logs(db, job.id, limit=500)
    error_logs = [log for log in logs if log["level"] in ("ERROR", "WARNING")]

    classification, remediation = classify_failure(job.error, job.payload)

    # Check if a failed job is diagnosable from logs alone
    diagnosable = True if (job.status != "failed" or len(logs) > 0 or job.error) else False

    root_cause = None
    if job.status == "failed":
        root_cause = {
            "stage_at_failure": job.stage,
            "classification": classification,
            "error_summary": job.error,
            "primary_error_log": error_logs[-1] if error_logs else None,
            "traceback_logged_safely": (job.payload or {}).get("traceback_logged", False),
        }

    return {
        "job_id": job.id,
        "status": job.status,
        "stage": job.stage,
        "diagnosable": diagnosable,
        "classification": classification if job.status == "failed" else "NORMAL",
        "root_cause": root_cause,
        "remediation_suggestion": remediation if job.status == "failed" else None,
        "total_log_entries": len(logs),
        "error_entries_count": len(error_logs),
        "recent_error_logs": error_logs[-10:],
    }


async def get_job_stage_timings(job: Job) -> dict[str, Any]:
    """Calculate per-stage timings dashboard for a job."""
    payload = job.payload or {}
    stages_info = payload.get("stages") or {}
    timings_s = stages_info.get("timings_s") or payload.get("timings_s") or {}

    total_duration = 0.0
    if job.started_at and job.completed_at:
        total_duration = max(0.0, (job.completed_at - job.started_at).total_seconds())
    elif job.started_at and job.status == "running":
        total_duration = max(0.0, (utcnow() - job.started_at).total_seconds())
    elif job.completed_at and job.created_at:
        total_duration = max(0.0, (job.completed_at - job.created_at).total_seconds())

    # If stage timings are not explicitly present in payload, provide stage bounds from timestamps
    formatted_timings: dict[str, float] = {}
    if isinstance(timings_s, dict):
        for k, v in timings_s.items():
            if isinstance(v, (int, float)):
                formatted_timings[k] = round(float(v), 3)

    sum_stage_time = sum(formatted_timings.values())
    if sum_stage_time > 0 and total_duration < sum_stage_time:
        total_duration = sum_stage_time

    percentages: dict[str, float] = {}
    slowest_stage = None
    max_time = -1.0
    for stage, s_time in formatted_timings.items():
        pct = round((s_time / total_duration * 100.0), 1) if total_duration > 0 else 0.0
        percentages[stage] = pct
        if s_time > max_time:
            max_time = s_time
            slowest_stage = {"stage": stage, "duration_s": s_time, "percentage": pct}

    return {
        "job_id": job.id,
        "status": job.status,
        "total_duration_s": round(total_duration, 3),
        "stage_timings_s": formatted_timings,
        "stage_percentages": percentages,
        "slowest_stage": slowest_stage,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
    }
