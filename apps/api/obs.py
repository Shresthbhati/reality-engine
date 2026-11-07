"""Observability for hosted use (plan T-6.6): a failed job is diagnosable from the logs alone, and a user never sees a traceback.

* ``clean_error(exc)``: the one-line, path-free message that goes into ``job.error`` (what the API returns).
* ``enter_stage(job, stage)``: sets ``job.stage``, accumulates per-stage seconds in ``job.payload["stage_timings"]`` and logs the change.
* ``job_log(...)`` and ``JsonFormatter``: structured log lines keyed by ``job_id`` / ``stage``, with the full traceback in ``exception``.
  Set ``REALITY_LOG_JSON=1`` to emit JSON lines (``configure_logging``).
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Dict, Optional, Tuple

log = logging.getLogger("reality.api.jobs")

MAX_ERROR_CHARS = 300
_WIN_PATH = re.compile(r"[A-Za-z]:[\\/][^\s'\"<>|]+")
_POSIX_PATH = re.compile(r"(?<![\w:/.])/(?:[\w.\-]+/)+[\w.\-]*")


def _basename(match: "re.Match[str]") -> str:
    return re.split(r"[\\/]", match.group(0).rstrip("\\/"))[-1]


def clean_error(exc: BaseException) -> str:
    """``ExceptionName: first line of the message`` with absolute server paths reduced to file names, at most 300 characters."""
    name = type(exc).__name__
    first = (str(exc).strip().splitlines() or [""])[0]
    first = _POSIX_PATH.sub(_basename, _WIN_PATH.sub(_basename, first)).strip()
    text = f"{name}: {first}" if first else name
    return text[:MAX_ERROR_CHARS]


_clock: Dict[str, Tuple[Optional[str], float]] = {}


def enter_stage(job: Any, stage: Optional[str]) -> None:
    """Move ``job`` to ``stage`` (None = finished): the stage it leaves is timed into ``payload["stage_timings"]`` and the change is logged."""
    now = time.monotonic()
    prev = _clock.get(job.id)
    seconds = None
    if prev and prev[0]:
        seconds = round(now - prev[1], 3)
        timings = dict((job.payload or {}).get("stage_timings") or {})
        timings[prev[0]] = round(timings.get(prev[0], 0.0) + seconds, 3)
        job.payload = {**(job.payload or {}), "stage_timings": timings}   # reassigned: a mutated JSON column is not tracked
    if stage is None:
        _clock.pop(job.id, None)
    else:
        _clock[job.id] = (stage, now)
    job.stage = stage
    job_log(log, "job.stage", job_id=job.id, stage=stage, previous=prev[0] if prev else None, seconds=seconds)


def job_log(logger: logging.Logger, event: str, *, job_id: Optional[str] = None, stage: Optional[str] = None,
            exc_info: bool = False, **fields: Any) -> None:
    logger.log(logging.ERROR if exc_info else logging.INFO, event,
               extra={"event": event, "job_id": job_id, "stage": stage, "fields": fields}, exc_info=exc_info)


class JsonFormatter(logging.Formatter):
    """One JSON object per line: ts, level, logger, message, event, job_id, stage, fields, exception."""

    def format(self, record: logging.LogRecord) -> str:
        out: Dict[str, Any] = {"ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"), "level": record.levelname,
                               "logger": record.name, "message": record.getMessage()}
        for key in ("event", "job_id", "stage"):
            v = getattr(record, key, None)
            if v is not None:
                out[key] = v
        extra = getattr(record, "fields", None)
        if extra:
            out.update({k: v for k, v in extra.items() if v is not None and k not in out})
        if record.exc_info:
            out["exception"] = self.formatException(record.exc_info)
        return json.dumps(out, default=str)


def configure_logging() -> None:
    """JSON lines on the root logger when ``REALITY_LOG_JSON=1`` (hosted); human-readable otherwise (unchanged)."""
    if os.environ.get("REALITY_LOG_JSON") != "1":
        return
    root = logging.getLogger()
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root.handlers = [handler]
    if root.level == logging.NOTSET or root.level > logging.INFO:
        root.setLevel(logging.INFO)
