"""Job-side glue for the optional global spatial optimization stage (`spatial_global`).

Opt-in via ``RE_SPATIAL_GLOBAL=1``: the host then runs the non-default pipeline profile ``interior_survey_spatial``. Default OFF; the
default profile and every existing job are unchanged. Contract with the perception side: structural observations + image poses must
arrive as the ``spatial_inputs.v1`` dict (see reconstruction/spatial/stage.py). Until a producer supplies it the stage is recorded as
``skipped`` with the reason -- nothing is optimized from data that does not exist, and no scale is assumed.

Persistence: one WorldStore per (team, world) under ``<WORLDSTORE_ROOT>/spatial-global/<team>/<world>/``; every run is a child of the
previous spatial version, so lineage is kept and history is never rewritten. Ids are validated before they become path segments.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

ENV_FLAG = "RE_SPATIAL_GLOBAL"
ENV_SOLVER = "RE_SPATIAL_SOLVER"      # "auto" (default) | "sparse" | "dense"
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class SpatialGlobalError(ValueError):
    pass


def enabled() -> bool:
    return os.environ.get(ENV_FLAG, "").strip().lower() in ("1", "true", "yes", "on")


def solver() -> str:
    v = os.environ.get(ENV_SOLVER, "auto").strip().lower()
    return v if v in ("auto", "sparse", "dense") else "auto"


def store_dir(root: str | os.PathLike, team_id: str, world_id: str) -> Path:
    for label, v in (("team", team_id), ("world", world_id)):
        if not isinstance(v, str) or not _ID_RE.match(v):
            raise SpatialGlobalError(f"invalid {label} id {v!r}")
    return Path(root) / "spatial-global" / team_id / world_id


def open_store(root: str | os.PathLike, team_id: str, world_id: str):
    """(WorldStore, parent version id or None): the parent is this world's latest spatial version, for lineage."""
    from worldstore.store import WorldStore

    store = WorldStore(store_dir(root, team_id, world_id))
    versions = store.list_versions()
    return store, (versions[-1].version_id if versions else None)


_PROVIDER = None


def register_inputs_provider(fn) -> None:
    """Seam for the perception producer: ``fn(job) -> spatial_inputs.v1 dict | None``. None unregisters. Until one is registered
    the stage is skipped (nothing is fabricated)."""
    global _PROVIDER
    _PROVIDER = fn


def inputs_for_job(job):
    if _PROVIDER is None:
        return None
    try:
        return _PROVIDER(job)
    except Exception:  # noqa: BLE001 - a producer bug must not sink the reconstruction job; it is logged and the stage skips
        import logging
        logging.getLogger("reality.api.spatial_global").exception("spatial inputs provider failed")
        return None


def facts_for_report(facts: dict | None) -> dict:
    """The job report entry: the facts dict as produced (JSON-safe), or nothing when the stage did not run."""
    return {"spatial_global": facts} if facts else {}
