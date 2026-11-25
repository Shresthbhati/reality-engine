"""Per-stage RSS / available-RAM / VRAM report for a finished reconstruction job (reads ``job_stage_runs``; runs nothing).

Workflow on the RTX 4050 machine (admission records these numbers on every stage; ``RE_MEMORY_ADMISSION=record`` measures
without ever refusing a stage):

    set RE_MEMORY_ADMISSION=record
    <run one real reconstruction through the API / worker>
    python scripts/profile_stage_rss.py --job job_xxx            # one job
    python scripts/profile_stage_rss.py --latest 5               # the 5 most recent jobs

Prints one row per stage attempt: peak RSS during the stage (sampled every 100 ms), RSS before/after, system RAM available
before, free VRAM before, any admission decision and wait. Use the maxima to replace the estimates in
``engine/hardware/admission.py::STAGE_NEEDS``. Honours ``DATABASE_URL`` like the API.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


async def main(job: str | None, latest: int) -> int:
    from sqlalchemy import select

    from apps.api.db import get_sessionmaker
    from apps.api.models import Job, JobStageRun

    async with get_sessionmaker()() as db:
        ids = [job] if job else [r for (r,) in (await db.execute(select(Job.id).order_by(Job.created_at.desc()).limit(latest))).all()]
        for jid in ids:
            rows = (await db.execute(select(JobStageRun).where(JobStageRun.job_id == jid).order_by(JobStageRun.started_at))).scalars().all()
            print(f"\n{jid}: {len(rows)} stage rows")
            print(f"{'stage':<18}{'status':<10}{'peak_rss':>10}{'rss_before':>12}{'rss_after':>11}{'ram_avail':>11}{'vram_free':>11}  note")
            for r in rows:
                m = r.metrics or {}
                f = lambda v: "-" if v is None else f"{v:.0f}"  # noqa: E731
                note = r.error or "; ".join(r.warnings or [])
                print(f"{r.name:<18}{r.status:<10}{f(r.peak_rss_mb):>10}{f(m.get('rss_before_mb')):>12}{f(m.get('rss_after_mb')):>11}"
                      f"{f(m.get('avail_ram_before_mb')):>11}{f(m.get('free_vram_before_mb')):>11}  {note[:90]}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--job")
    ap.add_argument("--latest", type=int, default=1)
    a = ap.parse_args()
    raise SystemExit(asyncio.run(main(a.job, a.latest)))
