"""Run record: enough to say exactly what produced a result (RE-MLOPS-000001).

Per job: code commit, interpreter, seeds, every stage's status / inputs hash / output artifact hashes, the lineage graph,
and the sha256 of every model checkpoint that was ON DISK when the job ran. "Available", not "used": the registry
does not yet record which model each stage loaded, and this says so rather than implying it.
"""
from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path

from engine.pipeline.stages.base import RunManifest

_HASH_CACHE: dict[tuple[str, int, int], str] = {}   # (path, mtime_ns, size) -> sha256; weights do not change between jobs


def git_commit(repo: Path | None = None) -> str | None:
    """HEAD of the code that ran, or None when git or the checkout is not there (an installed package)."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(repo or Path(__file__).resolve().parents[3]),
                             capture_output=True, text=True, timeout=3)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def file_sha256(path: Path) -> str:
    st = path.stat()
    key = (str(path), st.st_mtime_ns, st.st_size)
    if key not in _HASH_CACHE:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        _HASH_CACHE[key] = h.hexdigest()
    return _HASH_CACHE[key]


def models_available(specs: Mapping[str, object] | None = None, hasher: Callable[[Path], str] = file_sha256) -> list[dict]:
    """Active registered models whose checkpoint file exists, with the sha256 of the bytes on disk and, where the registry
    declares one, whether it matches."""
    if specs is None:
        from perception.model_registry import all_specs
        specs = all_specs()
    out = []
    for name in sorted(specs):
        s = specs[name]
        ck = getattr(s, "checkpoint", None)
        if getattr(s, "status", "active") != "active" or ck is None or not Path(ck).is_file():
            continue
        sha = hasher(Path(ck))
        rec = {"name": name, "checkpoint": str(ck), "sha256": sha}
        if getattr(s, "sha256", None):
            rec["declared_sha256_matches"] = sha == s.sha256.lower()
        out.append(rec)
    return out


_COLMAP_CACHE: dict[str, str] = {}


def colmap_info(binary: str = "colmap") -> dict:
    """Version string of the COLMAP binary on PATH and the seed the backend uses for its first mapper attempt.
    Later attempts (seeds 1, 2) only run when registration is weak; which one ran is in the job's attempts."""
    if binary not in _COLMAP_CACHE:
        version = "COLMAP (not found)"
        try:
            res = subprocess.run([binary, "version"], capture_output=True, text=True, timeout=5)
            if res.returncode == 0 and res.stdout.strip():
                version = res.stdout.strip().splitlines()[0]
        except (OSError, subprocess.SubprocessError):
            pass
        _COLMAP_CACHE[binary] = version
    from reconstruction.backend.colmap_backend import COLMAP_SEED
    return {"version": _COLMAP_CACHE[binary], "seed": COLMAP_SEED}


def build_run_record(manifest: RunManifest, *, seeds: Mapping[str, int] | None = None, commit: str | None = None,
                     models: list[dict] | None = None, colmap: dict | None = None) -> dict:
    return {
        "code_commit": commit if commit is not None else git_commit(),
        "python": sys.version.split()[0], "platform": platform.platform(),
        "seeds": dict(seeds or {}),
        "colmap": colmap if colmap is not None else colmap_info(),
        "pipeline_profile": manifest.pipeline_profile, "pipeline_version": manifest.pipeline_version,
        "stages": [{"name": r.name, "status": r.status, "inputs_hash": r.inputs_hash, "outputs": dict(r.outputs)}
                   for r in manifest.stages],
        "lineage": manifest.lineage.to_dict(),
        "models_available": models if models is not None else models_available(),
    }
