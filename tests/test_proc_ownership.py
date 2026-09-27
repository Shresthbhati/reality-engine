"""Process ownership proofs (P0 mission §5): cooperative cancellation of
owned subprocesses with tree termination, kill orders, and timeouts.

Uses deterministic stub children (the interpreter itself sleeping) --
never COLMAP/ORB-SLAM. No shell anywhere: every spawn is list-argv.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time

import pytest

from reconstruction.proc import (
    ProcessKilledError,
    discard_job,
    owned_pids,
    run_owned,
    set_current_job,
    terminate_job_procs,
)

SLEEPER = [sys.executable, "-c", "import time; time.sleep(120)"]


@pytest.fixture(autouse=True)
def _clean_registry():
    yield
    for job in ("j-basic", "j-kill", "j-tree", "j-order", "j-timeout", "j-plain"):
        discard_job(job)
    set_current_job(None)


def test_plain_run_without_job_context():
    """No job bound: plain bounded subprocess.run semantics, nothing
    registered, nothing to clean."""
    proc = run_owned(
        [sys.executable, "-c", "print('hi')"],
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0
    assert proc.stdout.strip() == "hi"
    assert owned_pids("anyone") == []


def test_kill_running_child_marks_completed_process():
    """Terminate a live owned child: the waiter returns a nonzero
    completion (no hang, no exception type change), the child is dead,
    and the registry is empty afterwards."""
    results = []

    def _run():
        # threading.Thread (unlike asyncio.to_thread, which the worker
        # uses in production) does not inherit context: bind explicitly
        # to simulate the inherited job context faithfully.
        set_current_job("j-kill")
        results.append(run_owned(SLEEPER, capture_output=True, text=True, timeout=120))

    t = threading.Thread(target=_run)
    t.start()
    deadline = time.time() + 15.0
    while not owned_pids("j-kill"):
        assert time.time() < deadline, "child never registered"
        time.sleep(0.05)
    (pid,) = owned_pids("j-kill")
    assert terminate_job_procs("j-kill") == 1
    t.join(timeout=30)
    assert not t.is_alive()
    assert results and results[0].returncode != 0
    assert owned_pids("j-kill") == []
    # The OS-level process is really gone, not just forgotten.
    assert not _alive(pid)


def _alive(pid: int) -> bool:
    if os.name == "nt":
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV"],
            capture_output=True, text=True, timeout=15,
        )
        return f'"{pid}"' in (out.stdout or "")
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


_GRANDCHILD_SCRIPT = (
    "import subprocess, sys, time; "
    "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)']); "
    "print(p.pid, flush=True); "
    "p.wait()"
)


def test_descendant_tree_terminated(tmp_path):
    """A grandchild outlives naive child-only kills: the tree kill must
    take descendants too, on this platform's strategy."""
    set_current_job("j-tree")
    grandchild_pids: list[int] = []

    def _run():
        proc = subprocess.Popen(
            [sys.executable, "-c", _GRANDCHILD_SCRIPT],
            stdout=subprocess.PIPE, text=True,
            **({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
               if os.name == "nt" else {"start_new_session": True}),
        )
        # NOTE: raw Popen here (not run_owned) so the test observes the
        # exact tree semantics run_owned relies on: new session/group +
        # group kill. run_owned itself is covered by test_kill_running.
        import reconstruction.proc as _proc

        with _proc._lock:
            _proc._owned.setdefault("j-tree", set()).add(proc)
        try:
            line = proc.stdout.readline()
            grandchild_pids.append(int(line.strip()))
            proc.wait(timeout=120)
        finally:
            with _proc._lock:
                live = _proc._owned.get("j-tree")
                if live is not None:
                    live.discard(proc)

    t = threading.Thread(target=_run)
    t.start()
    deadline = time.time() + 20.0
    while not grandchild_pids:
        assert time.time() < deadline, "grandchild pid never reported"
        time.sleep(0.05)
    assert terminate_job_procs("j-tree") == 1
    t.join(timeout=30)
    assert not t.is_alive()
    assert not _alive(grandchild_pids[0]), "grandchild survived the tree kill"


def test_kill_order_refuses_late_spawn():
    """Cancel-before-start: a kill order with nothing registered makes
    the next spawn raise instead of starting doomed work."""
    assert terminate_job_procs("j-order") == 0
    with pytest.raises(ProcessKilledError):
        run_owned(SLEEPER, capture_output=True, text=True, timeout=30, job_id="j-order")
    # Orders clear with the job: a later incarnation may spawn again.
    discard_job("j-order")
    proc = run_owned(
        [sys.executable, "-c", "print('again')"],
        capture_output=True, text=True, timeout=30, job_id="j-order",
    )
    assert proc.returncode == 0


def test_timeout_kills_and_reaps():
    """Timeout propagates as TimeoutExpired (subprocess.run contract),
    the timed-out child is really dead (not leaked), and the registry
    is empty afterwards."""
    outcome: list = []

    def _run():
        set_current_job("j-timeout")
        try:
            run_owned(SLEEPER, capture_output=True, text=True, timeout=2)
            outcome.append("returned")
        except subprocess.TimeoutExpired:
            outcome.append("timeout")

    t = threading.Thread(target=_run)
    t.start()
    deadline = time.time() + 15.0
    pids = []
    while not pids:
        assert time.time() < deadline, "child never registered"
        time.sleep(0.05)
        pids = owned_pids("j-timeout")
    t.join(timeout=30)
    assert not t.is_alive()
    assert outcome == ["timeout"]
    assert owned_pids("j-timeout") == []
    assert not _alive(pids[0]), "timed-out child still alive"
