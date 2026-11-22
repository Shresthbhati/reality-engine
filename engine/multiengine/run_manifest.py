"""Reproducible execution manifest of one evidence run.

Records everything needed to re-run, or to explain a difference between two runs: the code version, a hash of every input file, every
engine execution with its checkpoint identity and device, the numeric parameters actually in force (all of them untuned starting values
unless a benchmark says otherwise), the software environment, cache use, and the content hash of the world that came out. ``verify``
re-checks the parts that can be re-checked from disk: the inputs still hash to what was recorded, the manifest itself is unchanged,
and the recorded world version still verifies and still has the recorded hash.

What a manifest does NOT claim: bit-identical re-execution. GPU kernels, COLMAP's threading and model nondeterminism can change
results between runs; the manifest says what was run, not that it will repeat exactly.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA = "evidence-run-manifest/1"
MANIFEST_FILE = "manifest.json"


def _sha256_file(path: str) -> str | None:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _git(root: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:  # noqa: BLE001 -- no git / not a checkout: the manifest says "unknown" rather than guessing
        return None


def _environment() -> dict[str, Any]:
    env: dict[str, Any] = {"python": sys.version.split()[0], "platform": platform.platform(), "machine": platform.machine()}
    for mod in ("numpy", "scipy", "PIL", "open3d", "torch", "pycolmap"):
        try:
            m = __import__(mod)
            env[mod] = getattr(m, "__version__", "unknown")
        except Exception:  # noqa: BLE001
            env[mod] = None
    try:
        import torch

        env["cuda_available"] = bool(torch.cuda.is_available())
        env["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:  # noqa: BLE001
        env["cuda_available"] = None
    return env


def _parameters() -> dict[str, Any]:
    """The thresholds actually in force. Imported lazily so a missing optional module is recorded, not fatal."""
    out: dict[str, Any] = {}
    try:
        from apps.api import evidence_world_stage as ew
        from engine.multiengine import spatial_stage, surfaces
        from engine.multiengine.depth_scale_alignment import DEFAULT_MAX_HELD_OUT_REL_RMSE, DEFAULT_MIN_HELD_OUT
        from engine.multiengine.pair_scoring_3d import FLOOR_FRAC, MIN_IOU
        from engine.multiengine.spatial_relations import REL_TOLERANCE

        out["depth_alignment"] = {"max_held_out_relative_rmse": DEFAULT_MAX_HELD_OUT_REL_RMSE, "min_held_out": DEFAULT_MIN_HELD_OUT}
        out["spatial_stage"] = {"ray_fraction_of_median_depth": spatial_stage.RAY_FRACTION}
        out["surfaces"] = {"voxel_fraction": surfaces.VOXEL_FRACTION, "plane_dist_fraction": surfaces.PLANE_DIST_FRACTION,
                           "min_inlier_fraction": surfaces.MIN_INLIER_FRACTION, "max_planes": surfaces.MAX_PLANES,
                           "horizontal_cos": surfaces.HORIZONTAL_COS, "vertical_cos": surfaces.VERTICAL_COS}
        out["association"] = {"appearance_min": ew.APPEARANCE_MIN, "geometry_min_iou": MIN_IOU, "geometry_floor_frac": FLOOR_FRAC}
        out["relations"] = {"box_relative_tolerance": REL_TOLERANCE}
        out["lift_engine"] = os.environ.get(ew.ENV_LIFT, "spatial")
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
    out["note"] = "all thresholds are starting values; none was tuned against a benchmark"
    return out


def build(root: Path, run_id: str, sources: Sequence[str], facts: Mapping[str, Any], *, repo_root: Path | None = None) -> dict[str, Any]:
    from engine.multiengine.evidence_matrix import EvidenceMatrix

    repo = repo_root or Path(__file__).resolve().parents[2]
    try:
        runs = [r.to_dict() for r in EvidenceMatrix(root).runs()]
    except Exception as exc:  # noqa: BLE001
        runs = [{"error": f"{type(exc).__name__}: {exc}"}]
    world = dict(facts.get("world_ir") or {})
    body: dict[str, Any] = {
        "schema": SCHEMA, "run_id": run_id,
        "code": {"commit": _git(repo, "rev-parse", "HEAD"), "dirty": bool(_git(repo, "status", "--porcelain")), "branch": _git(repo, "rev-parse", "--abbrev-ref", "HEAD")},
        "inputs": [{"path": str(s), "sha256": _sha256_file(str(s))} for s in sources],
        "engines": runs,
        "depth": [{k: d.get(k) for k in ("engine_id", "status", "checkpoint_id", "checkpoint_sha256", "wall_s", "peak_vram_mb", "peak_ram_mb", "cache", "error")}
                  for d in (facts.get("depth") or [])],
        "lifting": {k: (facts.get("lifting") or {}).get(k) for k in ("engine", "status", "lifted", "merged_entities", "multiview_entities", "max_ray_distance",
                                                                       "ray_threshold_basis", "spatial_error")},
        "parameters": _parameters(),
        "environment": _environment(),
        "scale": "sfm_gauge (not metres); physical accuracy NOT_MEASURED",
        "result": {"status": facts.get("status"), "counts": facts.get("counts"),
                   "world": {k: world.get(k) for k in ("status", "version_id", "sha256", "entities", "geometries", "withheld_entities")}},
        "reproducibility": "records what was run; does not claim bit-identical re-execution (GPU kernels, COLMAP threading and model nondeterminism)",
    }
    body["content_sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    body["created_at_unix"] = round(time.time(), 3)  # outside the hash: the same run description hashes the same
    return body


def write(root: Path, manifest: Mapping[str, Any]) -> Path:
    path = Path(root) / MANIFEST_FILE
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=1, sort_keys=True, default=str), encoding="utf-8")
    os.replace(tmp, path)
    return path


def verify(root: Path) -> dict[str, Any]:
    """Re-check a stored manifest against the disk. ``ok`` only when every check passed; each failure is listed."""
    problems: list[str] = []
    try:
        m = json.loads((Path(root) / MANIFEST_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {"ok": False, "problems": [f"manifest unreadable: {exc}"]}
    body = {k: v for k, v in m.items() if k not in ("content_sha256", "created_at_unix")}
    if hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode("utf-8")).hexdigest() != m.get("content_sha256"):
        problems.append("the manifest was modified after it was written (content hash differs)")
    for i in m.get("inputs", []):
        now = _sha256_file(i["path"])
        if now is None:
            problems.append(f"input {i['path']} is no longer readable")
        elif now != i["sha256"]:
            problems.append(f"input {i['path']} changed since the run (hash differs)")
    w = (m.get("result") or {}).get("world") or {}
    if w.get("status") == "persisted" and w.get("version_id"):
        try:
            from worldstore.store import WorldStore

            store = WorldStore(Path(root) / "worldstore")
            bad = store.verify_version(w["version_id"])
            if bad:
                problems.append(f"world version {w['version_id']} fails integrity verification: {bad[:2]}")
            elif hashlib.sha256(json.dumps(store.load_version(w["version_id"]).to_dict(), sort_keys=True).encode("utf-8")).hexdigest() != w.get("sha256"):
                problems.append("the stored world's content hash differs from the one recorded in the manifest")
        except Exception as exc:  # noqa: BLE001
            problems.append(f"world version cannot be checked: {type(exc).__name__}: {exc}")
    return {"ok": not problems, "problems": problems, "content_sha256": m.get("content_sha256"), "code_commit": (m.get("code") or {}).get("commit")}
