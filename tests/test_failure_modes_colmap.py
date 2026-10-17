"""CE-09 failure modes at the COLMAP boundary: subprocess timeout, unparseable reconstruction output, disk-full,
and a fault inside the session commit itself.

No real COLMAP: ``reconstruction.proc.run_owned`` (the single place the backend spawns COLMAP) is replaced by a fake
that writes COLMAP-shaped output, so the REAL ``_run_step`` / parse / session code runs. Faults are injected by step.

Invariants after EVERY fault (the same ones tests/integration/test_colmap_failure_injection.py asserts on real COLMAP):
  * ``current/`` (the committed model) is byte-identical -- nothing valid was deleted or rewritten
  * no manifest is left in staging, so the crashed run can never be committed
  * a clear, typed error is surfaced (the failing step and the real cause), never a bare IndexError
  * the next run on the same root converges
"""

from __future__ import annotations

import errno
import hashlib
import os
import subprocess
from pathlib import Path

import pytest

from evidence.packages import EvidenceKind
from evidence.session import EvidenceItem
from reconstruction.backend import colmap_backend as cb
from reconstruction.backend.colmap_backend import ColmapReconstructionBackend, ReconstructionStepError
from reconstruction.colmap_session import ColmapSession

A, B, C = "a", "b", "c"


class SimulatedCrash(BaseException):
    """The process died: nothing in the code under test may absorb it."""


def digest(directory: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(directory.rglob("*")):
        if p.is_file():
            h.update(p.relative_to(directory).as_posix().encode())
            h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


class FakeColmap:
    """Stands in for ``run_owned``: dispatches on the COLMAP step name and writes COLMAP-shaped output."""

    def __init__(self):
        self.registered = [f"ev-{A}.jpg", f"ev-{B}.jpg"]
        self.timeout_on: set[str] = set()
        self.garbage_when_path_contains: str | None = None
        self.garbage = "this is not a COLMAP model\n"
        self.timeouts_seen: list[float] = []
        self.steps: list[str] = []

    def __call__(self, cmd, *, timeout=None, **kw):
        step, args = cmd[1], cmd[2:]

        def opt(key):
            return args[args.index(key) + 1]

        self.steps.append(step)
        if step in self.timeout_on:
            self.timeouts_seen.append(timeout)
            raise subprocess.TimeoutExpired(cmd, timeout, output="partial stdout", stderr="partial stderr")
        if step == "feature_extractor":
            Path(opt("--database_path")).write_bytes(b"db")
        elif step == "mapper":
            out = Path(opt("--output_path")) / "0"
            out.mkdir(parents=True, exist_ok=True)
            (out / "images.bin").write_bytes(b"model")
        elif step == "model_converter":
            d = Path(opt("--input_path"))
            if self.garbage_when_path_contains and self.garbage_when_path_contains in d.as_posix():
                (d / "images.txt").write_text(self.garbage)
                (d / "points3D.txt").write_text(self.garbage)
            else:
                poses = "".join(f"{i} 1 0 0 0 0 0 {i} 1 {n}\n1.0 1.0 -1\n" for i, n in enumerate(self.registered, 1))
                (d / "images.txt").write_text("# Image list\n" + poses)
                (d / "points3D.txt").write_text("# 3D point list\n1 0.0 0.0 1.0 255 255 255 0.5 1 0\n")
        elif step in ("image_registrator", "point_triangulator", "bundle_adjuster"):
            (Path(opt("--output_path")) / "cameras.bin").write_bytes(b"inc")
        return subprocess.CompletedProcess(cmd, 0, "", "")


def items(tmp_path, *stems):
    out = []
    for s in stems:
        f = tmp_path / "photos" / f"{s}.jpg"
        f.parent.mkdir(exist_ok=True)
        f.write_bytes(b"jpeg-" + s.encode())
        out.append(EvidenceItem(id=f"ev-{s}", kind=EvidenceKind.PHOTO, source_uri=f.resolve().as_uri()))
    return out


def backend(sess):
    # sys.executable stands in for the colmap binary: it only has to resolve on PATH (every step is faked)
    return ColmapReconstructionBackend(colmap_binary=__import__("sys").executable, session=sess)


@pytest.fixture
def fake(monkeypatch):
    f = FakeColmap()
    import reconstruction.proc as proc

    monkeypatch.setattr(proc, "run_owned", f)
    return f


@pytest.fixture
def committed(tmp_path, fake):
    """A committed two-photo state; returns (session, digest-of-current)."""
    sess = ColmapSession(tmp_path / "colmap")
    result = backend(sess).reconstruct(items(tmp_path, A, B))
    assert len(result.camera_poses) == 2 and sess.commit()
    fake.steps.clear()
    return sess, digest(sess.current)


def assert_untouched(sess, before):
    assert digest(sess.current) == before, "the committed COLMAP state changed during a failed run"
    assert sess.committed_manifest() is not None
    assert not (sess.staging / "manifest.json").exists(), "a failed run left a committable manifest in staging"
    assert sess.commit() is False, "a failed run must not be committable"
    assert digest(sess.current) == before


def retry_converges(tmp_path, sess, before, fake):
    fake.timeout_on.clear()
    fake.garbage_when_path_contains = None
    fake.registered = [f"ev-{A}.jpg", f"ev-{B}.jpg", f"ev-{C}.jpg"]
    be = backend(sess)
    result = be.reconstruct(items(tmp_path, A, B, C))
    assert len(result.camera_poses) == 3 and be.last_run_info["registered_total"] == 3
    assert be.last_run_info["prior_images"] == 2 and be.last_run_info["new_images"] == [f"ev-{C}"]
    assert digest(sess.current) == before, "only an adopted version may commit"
    assert sess.commit() and digest(sess.current) != before
    return be.last_run_info


# --------------------------------------------------------------------------------------------- subprocess timeout


@pytest.mark.parametrize("step", ["feature_extractor", "exhaustive_matcher"])
def test_a_timeout_before_the_incremental_step_surfaces_the_step_and_leaves_the_committed_model_intact(
        tmp_path, committed, fake, step):
    sess, before = committed
    fake.timeout_on = {step}
    with pytest.raises(ReconstructionStepError) as ei:
        backend(sess).reconstruct(items(tmp_path, A, B, C))
    assert ei.value.step == step and "[TIMEOUT after" in ei.value.stderr
    assert isinstance(ei.value.__cause__, subprocess.TimeoutExpired)
    assert_untouched(sess, before)
    assert retry_converges(tmp_path, sess, before, fake)["mode"] == "incremental"


@pytest.mark.parametrize("step", ["image_registrator", "point_triangulator", "bundle_adjuster"])
def test_a_timeout_inside_the_incremental_chain_falls_back_to_a_full_rebuild_and_is_recorded(
        tmp_path, committed, fake, step):
    sess, before = committed
    fake.timeout_on = {step}
    fake.registered = [f"ev-{A}.jpg", f"ev-{B}.jpg", f"ev-{C}.jpg"]
    be = backend(sess)
    result = be.reconstruct(items(tmp_path, A, B, C))          # the hung step is not a job failure: the full mapper runs
    info = be.last_run_info
    assert info["mode"] == "full" and step in info["incremental_error"] and "TIMED OUT" in info["incremental_error"]
    assert len(result.camera_poses) == 3
    assert digest(sess.current) == before and sess.staging.joinpath("manifest.json").is_file()


def test_a_timeout_in_the_mapper_of_a_first_run_fails_explicitly_and_commits_nothing(tmp_path, fake):
    sess = ColmapSession(tmp_path / "colmap")
    fake.timeout_on = {"mapper"}
    with pytest.raises(ReconstructionStepError) as ei:
        backend(sess).reconstruct(items(tmp_path, A, B))
    assert ei.value.step == "mapper" and "TIMEOUT" in str(ei.value)
    assert sess.committed_manifest() is None and not sess.current.exists()
    assert sess.commit() is False and not sess.current.exists()


def test_the_subprocess_timeout_is_configurable_and_a_bad_value_falls_back_to_the_default(tmp_path, committed, fake,
                                                                                         monkeypatch):
    sess, before = committed
    fake.timeout_on = {"feature_extractor"}
    for raw, expected in (("7", 7.0), ("not-a-number", 3600.0)):
        monkeypatch.setenv("COLMAP_SUBPROCESS_TIMEOUT_SECONDS", raw)
        with pytest.raises(ReconstructionStepError):
            backend(sess).reconstruct(items(tmp_path, A, B, C))
        assert fake.timeouts_seen[-1] == expected
    assert_untouched(sess, before)


# ------------------------------------------------------------------------------------ reconstruction-parse failure


@pytest.mark.parametrize("garbage", [
    "this is not a COLMAP model\n",                       # wrong format
    "1 1 0 0\n2 2\n",                                      # truncated pose line
    "1 1 0 0 0 0 0 x 1 name\n1.0 1.0 -1\n",                # non-numeric field
])
def test_unparseable_reconstruction_output_is_a_clear_typed_failure_and_commits_nothing(
        tmp_path, committed, fake, garbage):
    sess, before = committed
    fake.garbage, fake.garbage_when_path_contains = garbage, "/sparse"
    fake.timeout_on = set()
    # force the FULL path (the incremental chain is covered separately): a world whose prior state is unusable
    (sess.current / "database.db").unlink()
    before = digest(sess.current)
    with pytest.raises(ReconstructionStepError) as ei:
        backend(sess).reconstruct(items(tmp_path, A, B, C))
    assert ei.value.step == "model_parse", f"expected a typed parse failure, got step={ei.value.step!r}: {ei.value}"
    assert "images.txt" in str(ei.value) or "points3D.txt" in str(ei.value)
    assert_untouched(sess, before)


def test_unparseable_incremental_output_falls_back_to_a_full_rebuild_instead_of_failing_the_job(
        tmp_path, committed, fake):
    sess, before = committed
    fake.garbage_when_path_contains = "adjusted"          # only the incremental model is unreadable
    fake.registered = [f"ev-{A}.jpg", f"ev-{B}.jpg", f"ev-{C}.jpg"]
    be = backend(sess)
    result = be.reconstruct(items(tmp_path, A, B, C))
    info = be.last_run_info
    assert info["mode"] == "full" and info["incremental_error"].startswith("model_parse"), info
    assert len(result.camera_poses) == 3
    assert digest(sess.current) == before


# ------------------------------------------------------------------------------------------------------- disk full


def _enospc(*a, **k):
    raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))


@pytest.mark.parametrize("where", ["copying a new photograph into the workspace", "writing the manifest"])
def test_disk_full_mid_run_surfaces_enospc_and_leaves_the_committed_model_intact(
        tmp_path, committed, fake, monkeypatch, where):
    sess, before = committed
    mp = monkeypatch.context()
    m = mp.__enter__()
    if where.startswith("copying"):
        m.setattr(cb.shutil, "copy", _enospc)
    else:
        real = Path.write_text

        def write_text(self, *a, **k):
            if self.name.startswith("manifest.json"):
                _enospc()
            return real(self, *a, **k)

        m.setattr(Path, "write_text", write_text)
    fake.registered = [f"ev-{A}.jpg", f"ev-{B}.jpg", f"ev-{C}.jpg"]
    try:
        with pytest.raises(OSError) as ei:
            backend(sess).reconstruct(items(tmp_path, A, B, C))
    finally:
        mp.__exit__(None, None, None)
    assert ei.value.errno == errno.ENOSPC, "the real cause (disk full) must be surfaced, not wrapped away"
    assert_untouched(sess, before)
    fake.registered = [f"ev-{A}.jpg", f"ev-{B}.jpg", f"ev-{C}.jpg"]
    assert retry_converges(tmp_path, sess, before, fake)


@pytest.mark.parametrize("fault", [OSError(errno.ENOSPC, "No space left on device"), SimulatedCrash("killed")])
def test_a_fault_inside_the_session_commit_never_loses_the_committed_model(tmp_path, committed, fake, monkeypatch, fault):
    """commit() swaps directories in two steps (current -> previous, staging -> current). A fault -- or a kill --
    between them must not leave the world without its committed model."""
    sess, before = committed
    fake.registered = [f"ev-{A}.jpg", f"ev-{B}.jpg", f"ev-{C}.jpg"]
    backend(sess).reconstruct(items(tmp_path, A, B, C))
    real_replace, calls = os.replace, []

    def replace(src, dst, *a, **k):
        calls.append((Path(src).name, Path(dst).name))
        if (Path(src).name, Path(dst).name) == ("staging", "current"):
            raise fault
        return real_replace(src, dst, *a, **k)

    with monkeypatch.context() as m:
        m.setattr(os, "replace", replace)
        with pytest.raises(type(fault)):
            sess.commit()
    assert ("staging", "current") in calls
    # the process 'restarts': a fresh session object on the same root
    fresh = ColmapSession(sess.root)
    assert fresh.committed_manifest() is not None, "the committed model vanished when commit() was interrupted"
    assert digest(fresh.current) == before
    assert fresh.begin({"robust_sift": True, "guided_matching": False, "trusted_intrinsics": None},
                       [f"ev-{A}.jpg", f"ev-{B}.jpg", f"ev-{C}.jpg"])[1] is not None, "the next run lost its incremental base"
