"""Job-scoped external-process ownership for reconstruction backends.

Problem: backends (COLMAP, VIO tools, dense/meshing helpers) are
synchronous -- they block inside ``subprocess.run`` with no abort hook.
A job timeout or operator cancel could therefore strand the child
process: the job row said FAILED/CANCELLED while COLMAP kept crunching
for hours, holding GPU/CPU and temp space.

Model (minimal, no orchestration framework):

* :func:`run_owned` replaces ``subprocess.run`` at every backend launch
  site. It spawns via :class:`~subprocess.Popen`, registers the handle
  against the current job (see :func:`set_current_job`), waits with a
  timeout, unregisters, and returns a :class:`CompletedProcess`.
* On timeout it kills the whole process TREE (not just the direct
  child) and raises :exc:`subprocess.TimeoutExpired`, exactly like
  ``subprocess.run`` does -- existing handlers keep working unchanged.
* :func:`terminate_job_procs` kills a job's registered trees on demand
  (cancel endpoint, timeout path). It also records a kill order so a
  spawn that has not happened yet is refused instead of started
  (:exc:`ProcessKilledError`) -- this closes the cancel-before-start
  race deterministically. A kill order racing an in-flight spawn still
  converges safely: the endpoint sets the job flag first, so the
  handler grades CANCELLED at its next boundary, and the per-process
  timeout bounds the child regardless.
* :func:`discard_job` drops all state for a job; the worker calls it
  for every terminal outcome, so the registry is bounded by the number
  of in-flight jobs, never growing.

Platforms: POSIX kills the process group (children are spawned with
``start_new_session``); Windows uses ``taskkill /T`` on a fresh process
group. No shell anywhere: argv stays a list end to end.

Without a job context (CLI, tests calling backends directly) every
function degrades to plain bounded ``subprocess.run`` semantics.
"""

from __future__ import annotations

import contextvars
import logging
import os
import signal
import subprocess
import threading

log = logging.getLogger("reality.proc")

_current_job: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "reconstruction_job_id", default=None
)

_lock = threading.Lock()
_owned: dict[str, set[subprocess.Popen]] = {}
_kill_orders: set[str] = set()


class ProcessKilledError(RuntimeError):
    """A spawn was refused (or a wait aborted) because the owning job
    was cancelled/timed out. Handlers translate this into CANCELLED via
    the job's cancel flag -- never SUCCESS."""


def set_current_job(job_id: str | None) -> None:
    """Bind (or unbind, with None) the calling context to a job. The job
    runner sets this around handler execution; it propagates into worker
    threads, so synchronous backends inherit ownership transparently."""
    _current_job.set(job_id)


def owned_pids(job_id: str) -> list[int]:
    """PIDs currently owned by a job (observability for logs/tests)."""
    with _lock:
        return sorted(p.pid for p in _owned.get(job_id, set()) if p.poll() is None)


def _kill_tree(proc: subprocess.Popen) -> None:
    """Terminate a process and its descendants. Best-effort: a dead PID
    is success, not an error."""
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                capture_output=True,
                timeout=15,
            )
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError) as exc:
        log.debug("tree-kill for pid %s already gone: %s", proc.pid, exc)
    except Exception:
        log.exception("tree-kill for pid %s failed", proc.pid)


def terminate_job_procs(job_id: str) -> int:
    """Kill every process tree owned by `job_id` and refuse future spawns
    for it until :func:`discard_job`. Returns the kill count. Safe to
    call with nothing registered (e.g. cancel before start)."""
    with _lock:
        _kill_orders.add(job_id)
        procs = list(_owned.get(job_id, set()))
    killed = 0
    for proc in procs:
        if proc.poll() is None:
            _kill_tree(proc)
            killed += 1
    log.info("terminated %d owned process(es) for job %s", killed, job_id)
    return killed


def discard_job(job_id: str) -> None:
    """Forget all registry state for a job (terminal outcomes call this
    unconditionally). Pending kill orders are dropped with it."""
    with _lock:
        _owned.pop(job_id, None)
        _kill_orders.discard(job_id)


def run_owned(
    cmd: list[str],
    *,
    timeout: float | None = None,
    job_id: str | None = None,
    **kwargs,
) -> subprocess.CompletedProcess:
    """Drop-in bounded replacement for ``subprocess.run`` at backend
    launch sites. Registers the child against the current job (or the
    explicit `job_id`), waits with `timeout`, unregisters, and returns
    the completed process. Raises :exc:`subprocess.TimeoutExpired`
    (after tree-killing) like ``subprocess.run`` does, or
    :exc:`ProcessKilledError` if the job was cancelled before/during
    the spawn."""
    owner = job_id if job_id is not None else _current_job.get()
    if owner is not None:
        with _lock:
            if owner in _kill_orders:
                raise ProcessKilledError(
                    f"job {owner} was cancelled; refusing to spawn {cmd[0] if cmd else '?'}"
                )
    popen_kwargs = dict(kwargs)
    # capture_output is a subprocess.run convenience, not a Popen
    # argument: translate it exactly the way run() does, so migrated
    # call sites keep working byte-for-byte.
    if popen_kwargs.pop("capture_output", False):
        popen_kwargs.setdefault("stdout", subprocess.PIPE)
        popen_kwargs.setdefault("stderr", subprocess.PIPE)
    if owner is not None:
        if os.name == "nt":
            popen_kwargs.setdefault("creationflags", subprocess.CREATE_NEW_PROCESS_GROUP)
        else:
            popen_kwargs.setdefault("start_new_session", True)
    proc = subprocess.Popen(cmd, **popen_kwargs)
    if owner is not None:
        with _lock:
            _owned.setdefault(owner, set()).add(proc)
    # NOTE on the residual spawn race: a kill order landing between the
    # pre-check above and this registration is still safe. The cancel
    # endpoint sets the job flag BEFORE killing, so the handler grades
    # CANCELLED at its next boundary, and the per-process timeout bounds
    # the child regardless. No orphan, no false success -- just a
    # late-cancel semantic owned by the state machine, not the registry.
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)
    except subprocess.TimeoutExpired:
        _kill_tree(proc)
        try:
            proc.wait(timeout=10)
        except Exception:
            pass
        raise
    finally:
        if owner is not None:
            with _lock:
                live = _owned.get(owner)
                if live is not None:
                    live.discard(proc)
                    if not live:
                        _owned.pop(owner, None)
