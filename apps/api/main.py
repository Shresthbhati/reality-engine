"""Reality Engine application API.

FastAPI service in front of the existing computational engine. The CLI and
this API share the same domain services; no computation is duplicated.

Run:  uvicorn apps.api.main:app --port 8100
"""

from __future__ import annotations

import asyncio
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from apps.api import jobs as jobrunner
from apps.api.db import dispose_db, init_db
from apps.api.routes_jobs import activity, jobs, notifications
from apps.api.routes_misc import evidence, uploads
from apps.api.routes_export import export
from apps.api.routes_procedural import procedural
from apps.api.routes_query import query
from apps.api.routes_reconstructions import reconstructions, world_status
from apps.api.routes_sessions import health, sessions
from apps.api.routes_system import system
from apps.api.routes_worlds import worlds
from reconstruction.proc import recover_orphaned_jobs

app = FastAPI(title="Reality Engine API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "http://localhost:3000").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

for router in (health, sessions, uploads, evidence, reconstructions, world_status, worlds, procedural, query, export, jobs, notifications, activity, system):
    app.include_router(router)


@app.on_event("startup")
async def _startup() -> None:
    await init_db()
    # Recover any jobs that were orphaned by a previous worker crash
    try:
        from apps.api.db import get_sessionmaker
        maker = get_sessionmaker()
        async with maker() as db:
            from reconstruction.proc import recover_orphaned_jobs
            await recover_orphaned_jobs(db)
    except Exception:
        import logging
        log = logging.getLogger("reality.api.startup")
        log.exception("startup orphan recovery failed")
    # A previous process may have died between writing a WorldStore version and adopting it: quarantine such
    # orphans BEFORE the worker starts, so a retry can never duplicate them. Never blocks startup.
    try:
        from apps.api import worldstore_service
        from apps.api.db import get_sessionmaker

        async with get_sessionmaker()() as db:
            report = await worldstore_service.reconcile_adoptions(db)
        if report["errors"]:
            import logging
            logging.getLogger("reality.api.startup").error("adoption recovery incomplete: %s", report["errors"])
    except Exception:
        import logging
        logging.getLogger("reality.api.startup").exception("startup adoption recovery failed")
    app.state.worker = asyncio.create_task(jobrunner.worker_loop())


@app.on_event("shutdown")
async def _shutdown() -> None:
    import logging

    log = logging.getLogger("reality.api.shutdown")
    task = getattr(app.state, "worker", None)
    if task:
        task.cancel()
        try:
            # Bounded shutdown: a worker wedged in an uninterruptible
            # await must not wedge process exit (or test-client
            # teardown) forever -- running jobs are reaped as stale on
            # the next startup instead.
            await asyncio.wait_for(asyncio.shield(task), timeout=10.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            log.warning("worker did not stop within 10s; leaving it behind")
    try:
        await asyncio.wait_for(dispose_db(), timeout=10.0)
    except asyncio.TimeoutError:
        log.warning("db dispose timed out; leaving pool behind")
