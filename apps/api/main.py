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
from apps.api.routes_procedural import procedural
from apps.api.routes_query import query
from apps.api.routes_sessions import health, sessions
from apps.api.routes_worlds import worlds

app = FastAPI(title="Reality Engine API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "http://localhost:3000").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

for router in (health, sessions, uploads, evidence, worlds, procedural, query, jobs, notifications, activity):
    app.include_router(router)


@app.on_event("startup")
async def _startup() -> None:
    await init_db()
    app.state.worker = asyncio.create_task(jobrunner.worker_loop())


@app.on_event("shutdown")
async def _shutdown() -> None:
    task = getattr(app.state, "worker", None)
    if task:
        task.cancel()
    await dispose_db()
