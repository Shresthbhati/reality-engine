"""Reality Engine application API.

FastAPI service in front of the existing computational engine. The CLI and
this API share the same domain services; no computation is duplicated.

Run:  uvicorn apps.api.main:app --port 8100
"""

from __future__ import annotations

import asyncio
import logging
import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from apps.api import jobs as jobrunner
from apps.api.auth import auth_middleware
from apps.api.db import dispose_db, init_db
from apps.api.routes_account import account_router, auth_router
from apps.api.routes_adoptions import adoptions_router
from apps.api.routes_artifacts import artifacts
from apps.api.routes_assets import assets_router
from apps.api.routes_assistant import assistant, assistant_tools
from apps.api.routes_completion import completion
from apps.api.routes_construction import construction
from apps.api.routes_evidence_quality import quality
from apps.api.routes_export import export
from apps.api.routes_furniture import furniture_router
from apps.api.routes_geo import geo
from apps.api.routes_intent import estimate_router as intent_estimate
from apps.api.routes_intent import intent_router
from apps.api.routes_jobs import activity, jobs, notifications
from apps.api.routes_measurements import measurements
from apps.api.routes_navigation import navigation
from apps.api.routes_misc import evidence, uploads
from apps.api.routes_procedural import procedural, templates_router
from apps.api.routes_proposals import pipeline_router, proposals
from apps.api.routes_query import query
from apps.api.routes_reconstructions import reconstructions, world_status
from apps.api.routes_registration import registration
from apps.api.routes_sessions import health, sessions
from apps.api.routes_sites import sites
from apps.api.routes_splats import splats_router
from apps.api.routes_temporal import temporal_router
from apps.api.routes_system import system
from apps.api.routes_technologies import technologies_router
from apps.api.routes_usage import router as usage_router
from apps.api.routes_billing import router as billing_router
from apps.api.routes_worlds import worlds

log = logging.getLogger("reality.api")

app = FastAPI(title="Reality Engine API", version="0.1.0")


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    log.exception("Unhandled server exception on %s %s: %s", request.method, request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred. Request ID logged."},
    )


@app.middleware("http")
async def security_headers_middleware(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    return response
# Added before CORS so CORS stays outermost (401 responses still carry CORS headers).
app.middleware("http")(auth_middleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "http://localhost:3000").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

# `quality` before `evidence`: /api/evidence/quality must not be captured by /api/evidence/{evidence_id}
for router in (health, sessions, uploads, quality, evidence, reconstructions, world_status, registration, worlds, adoptions_router, proposals, pipeline_router, artifacts, geo, sites, procedural, templates_router, measurements, query, export, jobs, notifications, activity, system, intent_router, intent_estimate, completion, assistant_tools, assistant, technologies_router, assets_router, usage_router, furniture_router, construction, navigation, splats_router, temporal_router):
    app.include_router(router)

app.include_router(auth_router, prefix="/api/auth", tags=["auth"])
app.include_router(account_router, prefix="/api/account", tags=["account"])
app.include_router(billing_router)


@app.get('/health', tags=['health'])
@app.get('/healthz', tags=['health'])
async def healthz_check() -> dict:
    from apps.api.models import utcnow
    return {'status': 'ok', 'time': utcnow().isoformat()}


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
    enable_worker = (
        not os.environ.get("REALITY_DISABLE_IN_PROCESS_WORKER")
        and os.environ.get("ENABLE_WORKER", "true").lower() in ("true", "1", "yes")
    )
    if enable_worker:
        app.state.worker = asyncio.create_task(jobrunner.worker_loop())
    else:
        app.state.worker = None


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
