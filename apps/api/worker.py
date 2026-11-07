"""Standalone GPU / Compute Queue Worker Daemon.

Decoupled execution daemon that claims and processes jobs from the database-backed
durable queue. Can run on a dedicated GPU host (RunPod, Lambda, Modal, or local worker)
separate from the API server.

Usage:
    python -m apps.api.worker
    python apps/api/worker.py --poll-seconds 1.0
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal

from apps.api.db import dispose_db, init_db
from apps.api.jobs import WORKER_ID, worker_loop

log = logging.getLogger("reality.worker")


async def run_worker(poll_seconds: float = 1.0) -> None:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    )
    log.info("Starting standalone queue worker: worker_id=%s, poll=%.1fs", WORKER_ID, poll_seconds)
    await init_db()

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _on_signal() -> None:
        log.info("Received shutdown signal; stopping worker loop...")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _on_signal)
        except (NotImplementedError, RuntimeError):
            # Non-POSIX systems (e.g. Windows) may not support loop.add_signal_handler
            pass

    worker_task = asyncio.create_task(worker_loop(poll_seconds=poll_seconds))

    try:
        wait_task = asyncio.create_task(stop_event.wait())
        done, _ = await asyncio.wait(
            [worker_task, wait_task],
            return_when=asyncio.FIRST_COMPLETED,
        )
        if wait_task in done:
            log.info("Shutdown signal received; cancelling worker loop...")
            worker_task.cancel()
            try:
                await asyncio.wait_for(worker_task, timeout=10.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
    finally:
        await dispose_db()
        log.info("Worker daemon stopped cleanly.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reality Engine GPU/Compute Queue Worker")
    parser.add_argument("--poll-seconds", type=float, default=1.0, help="Queue polling interval in seconds")
    args = parser.parse_args()
    try:
        asyncio.run(run_worker(poll_seconds=args.poll_seconds))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
