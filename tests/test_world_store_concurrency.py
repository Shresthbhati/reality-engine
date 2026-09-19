"""Concurrency tests for WorldStore (worldstore/store.py): concurrent
save_version() callers must never lose a write, corrupt sequence.json,
or crash with a raw OS error.

Before the _SequenceLock fix, 20 concurrent threads calling
save_version() against the same store root reproduced 15/20 unhandled
PermissionError (WinError 5) from os.replace() racing on
sequence.json, plus (on runs that didn't crash) lost updates in the
sequence index. This file locks that behavior down with both an
in-process thread race (fast, always runs) and a real multi-process
race (separate OS processes, no shared GIL/memory -- the realistic
"writer A + writer B" scenario from separate CLI invocations).
"""

from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path

import pytest

from provenance import Provenance
from world_ir import Entity, EntityType
from world_ir.world_v1 import WorldIR
from worldstore.store import WorldStore, WorldStoreError, _SequenceLock


def _world(i: int) -> WorldIR:
    w = WorldIR(id=f"w-{i}")
    w.entities[f"e-{i}"] = Entity(
        id=f"e-{i}", type=EntityType.STRUCTURE, provenance=Provenance.RECONSTRUCTED,
    )
    return w


class TestSequenceLock:
    def test_second_acquirer_blocks_until_first_releases(self, tmp_path):
        lock1 = _SequenceLock(tmp_path, timeout=5.0)
        acquired_second = threading.Event()

        with lock1:
            def try_acquire():
                with _SequenceLock(tmp_path, timeout=5.0):
                    acquired_second.set()

            t = threading.Thread(target=try_acquire)
            t.start()
            # The second acquirer must NOT succeed while lock1 is held.
            assert not acquired_second.wait(timeout=0.3)

        # Released now -- the waiting thread should acquire promptly.
        t.join(timeout=5.0)
        assert acquired_second.is_set()

    def test_acquire_timeout_raises_explicit_error(self, tmp_path):
        with _SequenceLock(tmp_path, timeout=10.0):
            with pytest.raises(WorldStoreError, match="could not acquire"):
                with _SequenceLock(tmp_path, timeout=0.2, poll_interval=0.05):
                    pass  # pragma: no cover -- must not be reached

    def test_lock_directory_removed_after_use(self, tmp_path):
        lock_path = tmp_path / ".sequence.lock"
        with _SequenceLock(tmp_path):
            assert lock_path.exists()
        assert not lock_path.exists()


class TestConcurrentThreadWriters:
    def test_no_lost_writes_under_thread_contention(self, tmp_path):
        store = WorldStore(tmp_path)
        n = 25
        errors: list[tuple[int, Exception]] = []

        def writer(i: int) -> None:
            try:
                store.save_version(_world(i), parent=None, version_id=f"v-{i}")
            except Exception as exc:  # noqa: BLE001 -- capturing for assertion
                errors.append((i, exc))

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"concurrent save_version() raised: {errors}"
        versions = store.list_versions()
        assert len(versions) == n
        assert sorted(v.version_id for v in versions) == sorted(f"v-{i}" for i in range(n))
        assert not (tmp_path / ".sequence.lock").exists()


class TestConcurrentProcessWriters:
    def test_no_lost_writes_across_real_processes(self, tmp_path):
        """Real OS processes, not threads -- no shared GIL or memory,
        the actual "writer A + writer B" deployment scenario (two
        separate CLI/pipeline invocations writing to the same store)."""
        n = 8
        script = tmp_path / "_mp_writer.py"
        repo_root = Path(__file__).resolve().parents[1]
        script.write_text(
            "import sys\n"
            f"sys.path.insert(0, {str(repo_root)!r})\n"
            "from worldstore.store import WorldStore\n"
            "from world_ir.world_v1 import WorldIR\n"
            "from world_ir.schema_v1 import Entity, EntityType\n"
            "from provenance import Provenance\n"
            "root, i = sys.argv[1], int(sys.argv[2])\n"
            "store = WorldStore(root)\n"
            "w = WorldIR(id=f'w-{i}')\n"
            "w.entities[f'e-{i}'] = Entity(id=f'e-{i}', type=EntityType.STRUCTURE, "
            "provenance=Provenance.RECONSTRUCTED)\n"
            "store.save_version(w, parent=None, version_id=f'v-{i}')\n",
            encoding="utf-8",
        )
        store_root = tmp_path / "store"
        procs = [
            subprocess.Popen(
                [sys.executable, str(script), str(store_root), str(i)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            for i in range(n)
        ]
        results = [(p.wait(), p.stderr.read()) for p in procs]
        failures = [(i, err.decode()) for i, (rc, err) in enumerate(results) if rc != 0]
        assert failures == [], f"writer subprocess(es) failed: {failures}"

        store = WorldStore(store_root)
        versions = store.list_versions()
        assert len(versions) == n
        assert sorted(v.version_id for v in versions) == sorted(f"v-{i}" for i in range(n))
