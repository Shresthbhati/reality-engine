"""Run the job-queue consumer without the HTTP API: ``python -m apps.api.worker`` (the GPU box in a hosted deployment).

The API process then runs with ``REALITY_EMBEDDED_WORKER=0`` and only enqueues; both share the same database, so a worker restart
or crash loses no job (stale running jobs are requeued by ``reap_stale_jobs``, claims are atomic, live jobs heartbeat).
"""
from __future__ import annotations

import asyncio
import logging

from apps.api import jobs
from apps.api.db import dispose_db, init_db
from apps.api.obs import configure_logging


async def run() -> None:
    await init_db()
    try:
        await jobs.worker_loop()
    finally:
        await dispose_db()


def main() -> None:
    configure_logging()
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("reality.api.jobs").info("worker %s starting", jobs.WORKER_ID)
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
