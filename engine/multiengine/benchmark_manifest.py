"""Benchmark manifests with provenance, split-leakage checks and BLOCKED-by-default reporting.

A benchmark is only as real as its ground truth. This module never computes an accuracy number: it checks that a manifest is
well-formed (images exist and match their hash, no image is shared between scenes or splits, every ground-truth item names its
provenance and uncertainty) and then states, per metric, whether the independent ground truth that metric needs EXISTS in the held-out
test split. Anything missing is ``BLOCKED`` with the reason -- never a score, never a default.

Manifest (JSON)::

    {"benchmark_id": "...", "version": "1", "scenes": [{
        "scene_id": "...", "split": "development|validation|test", "kind": "interior|exterior",
        "images": [{"path": "rel/path.jpg", "sha256": "<64 hex>"}],
        "camera_metadata": {"source": "...", "per_image": {...}} | null,
        "ground_truth": {
            "room_dimensions": {"items": [{"name": "...", "value_m": 3.2, "uncertainty_m": 0.005}], "provenance": "tape|laser|survey..."},
            "object_dimensions": {...same...}, "instance_labels": {"ref": "path", "provenance": "..."},
            "instance_masks": {...}, "cross_view_identities": {...}, "camera_poses": {...}, "depth": {...}, "scene_completeness": {...}}}]}
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SPLITS = ("development", "validation", "test")
_SHA = "0123456789abcdef"

#: metric -> (ground-truth key it needs, human description)
METRIC_REQUIREMENTS: dict[str, tuple[str, str]] = {
    "detection_precision_recall_ap": ("instance_labels", "labelled object instances"),
    "instance_mask_ap_boundary": ("instance_masks", "labelled instance masks"),
    "multiview_association": ("cross_view_identities", "labelled cross-view identities"),
    "camera_registration_pose_error": ("camera_poses", "independent camera poses"),
    "depth_and_scale_error": ("depth", "independent depth / scale measurements"),
    "room_geometry_error": ("room_dimensions", "independently measured room dimensions"),
    "object_dimension_error": ("object_dimensions", "independently measured object dimensions"),
    "completeness_coverage": ("scene_completeness", "an independent completeness reference"),
}


class ManifestError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(path: str | Path) -> dict[str, Any]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ManifestError(f"manifest unreadable: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("scenes"), list):
        raise ManifestError("manifest must be an object with a 'scenes' list")
    return data


def _gt_item_problems(key: str, gt: Any) -> list[str]:
    if gt is None:
        return []
    if not isinstance(gt, Mapping):
        return [f"{key}: ground truth must be an object"]
    p: list[str] = []
    if not gt.get("provenance"):
        p.append(f"{key}: ground truth without provenance (who/what measured or labelled it) is not ground truth")
    for it in gt.get("items", []) if isinstance(gt.get("items"), list) else []:
        v, u = it.get("value_m"), it.get("uncertainty_m")
        if "value_m" in it and not (isinstance(v, (int, float)) and math.isfinite(v) and v > 0):
            p.append(f"{key}: item {it.get('name')!r} has a non-positive or non-finite value_m")
        if "value_m" in it and not (isinstance(u, (int, float)) and math.isfinite(u) and u >= 0):
            p.append(f"{key}: item {it.get('name')!r} needs a finite uncertainty_m >= 0 (a measurement without uncertainty is incomplete)")
    return p


def validate_manifest(manifest: Mapping[str, Any], root: str | Path = ".") -> list[str]:
    """All problems found (empty = well-formed). Checks hashes, splits, leakage and ground-truth provenance."""
    root = Path(root)
    problems: list[str] = []
    seen_scene: set[str] = set()
    hash_home: dict[str, tuple[str, str]] = {}  # sha256 -> (scene_id, split)
    for sc in manifest.get("scenes", []):
        sid = str(sc.get("scene_id", ""))
        if not sid:
            problems.append("a scene has no scene_id")
        if sid in seen_scene:
            problems.append(f"duplicate scene_id {sid!r}")
        seen_scene.add(sid)
        split = sc.get("split")
        if split not in SPLITS:
            problems.append(f"scene {sid!r}: split {split!r} is not one of {SPLITS}")
        images = sc.get("images") or []
        if not images:
            problems.append(f"scene {sid!r}: no images")
        for im in images:
            rel, sha = im.get("path"), str(im.get("sha256", ""))
            if len(sha) != 64 or any(c not in _SHA for c in sha):
                problems.append(f"scene {sid!r}: image {rel!r} has no valid sha256")
                continue
            f = root / str(rel)
            if not f.is_file():
                problems.append(f"scene {sid!r}: image {rel!r} does not exist")
            elif sha256_file(f) != sha:
                problems.append(f"scene {sid!r}: image {rel!r} does not match its recorded sha256 (the original must not change)")
            prev = hash_home.get(sha)
            if prev is not None and (prev[0] != sid):
                kind = "across splits" if prev[1] != split else "between scenes"
                problems.append(f"image-level leakage {kind}: {rel!r} appears in scenes {prev[0]!r} ({prev[1]}) and {sid!r} ({split})")
            hash_home.setdefault(sha, (sid, str(split)))
        gt = sc.get("ground_truth") or {}
        for key in METRIC_REQUIREMENTS.values():
            problems.extend(_gt_item_problems(key[0], gt.get(key[0])))
        if sc.get("kind") not in ("interior", "exterior"):
            problems.append(f"scene {sid!r}: kind must be interior or exterior")
    return problems


def benchmark_status(manifest: Mapping[str, Any], root: str | Path = ".", *, kind: str = "interior") -> dict[str, Any]:
    """Per-metric READY/BLOCKED for the held-out TEST split of ``kind`` scenes. Never a score."""
    problems = validate_manifest(manifest, root)
    test = [s for s in manifest.get("scenes", []) if s.get("split") == "test" and s.get("kind") == kind]
    metrics: dict[str, dict[str, Any]] = {}
    for name, (key, what) in METRIC_REQUIREMENTS.items():
        if problems:
            metrics[name] = {"status": "BLOCKED", "reason": "the manifest is invalid; see problems"}
        elif not test:
            metrics[name] = {"status": "BLOCKED", "reason": f"no held-out test {kind} scenes exist in the manifest"}
        else:
            lacking = [s["scene_id"] for s in test if not (s.get("ground_truth") or {}).get(key)]
            metrics[name] = ({"status": "BLOCKED", "reason": f"no {what} for test scene(s) {lacking}"} if lacking
                             else {"status": "READY", "reason": f"{what} present for all {len(test)} test scene(s); not yet evaluated"})
    return {"benchmark_id": manifest.get("benchmark_id"), "kind": kind, "valid": not problems, "problems": problems,
            "test_scenes": len(test), "metrics": metrics, "all_ready": not problems and all(m["status"] == "READY" for m in metrics.values())}
