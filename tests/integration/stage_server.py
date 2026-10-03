"""The real API + worker in its own process, frozen at an exact pipeline boundary so a test can kill it THERE.

    python -m tests.integration.stage_server

Environment: ``RE_PORT`` (listen port), ``RE_STAGE`` (boundary name, optional), ``RE_MARKER`` (file written the
instant the boundary is reached). When the boundary is reached the process writes the marker and then blocks
forever -- the test waits for the marker (failing loudly if it never appears, so a test can never "pass" by killing
the process at the wrong moment) and only then terminates the process tree. Nothing after the boundary has run.

Boundaries (the job's journey from upload to adopted version):

    evidence_staged        photographs stored, decoded and classified; no reconstruction started
    colmap_staged          COLMAP finished into ``staging/``; ``current/`` untouched; no WorldIR yet
    worldir_created        the progressive ladder returned its WorldIR; nothing validated or persisted
    candidate_evaluated    world-level acceptance decided (delta, conflicts, verdict); nothing persisted
    before_adoption        about to call commit_version; the WorldStore has nothing new
    during_adoption        the WorldStore holds the new version, the DB has NOT committed (HEAD still the old one)
    version_persisted      the DB transaction (HEAD pointer + mirror row) has committed; COLMAP base not advanced
    colmap_current_updated COLMAP ``current/`` advanced to this run's model; the job has not recorded completion

The hooks are installed by wrapping the production callables at the module attributes the job runner resolves, so
the code under test is the production code, unmodified.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

STAGES = ("evidence_staged", "colmap_staged", "worldir_created", "candidate_evaluated", "before_adoption",
          "during_adoption", "version_persisted", "colmap_current_updated")


def _freeze(stage: str) -> None:
    marker = os.environ["RE_MARKER"]
    tmp = marker + ".tmp"
    Path(tmp).write_text(json.dumps({"stage": stage, "pid": os.getpid(), "at": time.time()}))
    os.replace(tmp, marker)
    while True:                                   # the test kills this process; it never resumes
        time.sleep(3600)


def install(stage: str) -> None:
    import apps.api.jobs as jobs
    import apps.api.worldstore_service as wss
    from engine.pipeline import world_delta
    from reconstruction.backend.colmap_backend import ColmapReconstructionBackend
    from reconstruction.colmap_session import ColmapSession

    def after(owner, name):
        real = getattr(owner, name)

        def wrapper(*a, **k):
            out = real(*a, **k)
            _freeze(stage)
            return out
        setattr(owner, name, wrapper)

    def before(owner, name):
        real = getattr(owner, name)

        def wrapper(*a, **k):
            _freeze(stage)
            return real(*a, **k)
        setattr(owner, name, wrapper)

    if stage == "evidence_staged":
        before(jobs, "run_progressive")
    elif stage == "colmap_staged":
        after(ColmapReconstructionBackend, "reconstruct")
    elif stage == "worldir_created":
        after(jobs, "run_progressive")
    elif stage == "candidate_evaluated":
        after(world_delta, "describe_changes")
    elif stage == "before_adoption":
        real = wss.commit_version

        async def commit_version(*a, **k):
            _freeze(stage)
            return await real(*a, **k)
        wss.commit_version = commit_version
    elif stage in ("during_adoption", "version_persisted"):
        boundary = "stored" if stage == "during_adoption" else "committed"

        def hook(s):
            if s == boundary:
                _freeze(stage)
        wss._adoption_hook = hook
    elif stage == "colmap_current_updated":
        after(ColmapSession, "commit")
    else:
        raise SystemExit(f"unknown stage {stage!r}; expected one of {STAGES}")


def main() -> None:
    import uvicorn

    import apps.api.main as main_mod

    stage = os.environ.get("RE_STAGE")
    if stage:
        install(stage)
    uvicorn.run(main_mod.app, host="127.0.0.1", port=int(os.environ["RE_PORT"]), log_level="warning")


if __name__ == "__main__":
    main()
