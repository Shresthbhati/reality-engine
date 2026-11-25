"""Dataset adapters, ground-truth metrics, and availability probes (Programme J).

Supports ScanNet, Matterport3D, Structured3D, Hypersim, Replica, and local
COLMAP photographic sessions. Never fabricates ground truth or dataset results
when external datasets are absent on disk.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


@dataclass(frozen=True)
class DatasetCatalogEntry:
    name: str
    license: str
    commercial_ok: bool
    gt_modalities: Tuple[str, ...]
    default_env_var: str
    availability_status: str  # "available" | "blocked"
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


DATASET_CATALOG: Tuple[Tuple[str, str, bool, Tuple[str, ...], str, str], ...] = (
    ("scannet", "ScanNet Terms of Use (Non-Commercial Academic)", False, ("rgb", "depth", "poses", "3d_boxes", "semantic_masks"), "RE_SCANNET_DIR", "requires signed ScanNet agreement + download"),
    ("matterport3d", "Matterport3D Terms of Use (Non-Commercial Academic)", False, ("rgb", "depth", "poses", "room_types", "semantic_meshes"), "RE_MATTERPORT3D_DIR", "requires signed Matterport3D EULA + download"),
    ("structured3d", "Structured3D License (Non-Commercial Research)", False, ("rgb", "depth", "layout", "room_types", "semantic_masks"), "RE_STRUCTURED3D_DIR", "requires Structured3D archive download"),
    ("hypersim", "CC-BY-SA 3.0 / Apple Hypersim", True, ("rgb", "metric_depth", "normals", "semantic_masks"), "RE_HYPERSIM_DIR", "requires Hypersim download"),
    ("replica", "Replica Dataset License (Research)", False, ("rgb", "depth", "poses", "semantic_meshes"), "RE_REPLICA_DIR", "requires Replica mesh/render download"),
)


def probe_datasets(repo_root: Path | None = None) -> List[DatasetCatalogEntry]:
    import os

    root = repo_root or Path(".")
    entries: List[DatasetCatalogEntry] = []
    for name, lic, comm, mods, env_var, missing_reason in DATASET_CATALOG:
        env_path = os.environ.get(env_var, "").strip()
        local_dir = root / "datasets" / name
        resolved = Path(env_path) if env_path else local_dir
        ok = resolved.is_dir() and any(resolved.iterdir())
        entries.append(
            DatasetCatalogEntry(
                name=name,
                license=lic,
                commercial_ok=comm,
                gt_modalities=mods,
                default_env_var=env_var,
                availability_status="available" if ok else "blocked",
                reason=f"found at {resolved}" if ok else f"not present at {resolved} ({missing_reason})",
            )
        )
    colmap_dir = root / "data" / "worldstore" / "colmap-sessions"
    sessions = [p for p in colmap_dir.iterdir() if p.is_dir()] if colmap_dir.is_dir() else []
    entries.append(
        DatasetCatalogEntry(
            name="local_colmap_sessions",
            license="Internal / Unscored",
            commercial_ok=True,
            gt_modalities=("rgb",),
            default_env_var="",
            availability_status="available" if sessions else "blocked",
            reason=f"{len(sessions)} unscored session(s) in {colmap_dir}" if sessions else "no COLMAP sessions found",
        )
    )
    return entries


def compute_detection_metrics(
    pred_boxes: Sequence[Tuple[str, Tuple[float, float, float, float]]],
    gt_boxes: Sequence[Tuple[str, Tuple[float, float, float, float]]],
    *,
    iou_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compute per-category and overall precision/recall at ``iou_threshold``."""
    def _iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
        ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = ix * iy
        union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / union if union > 0 else 0.0

    matched_gt: set[int] = set()
    tp = 0
    fp = 0
    for plab, pbox in pred_boxes:
        best_iou, best_j = 0.0, None
        for j, (glab, gbox) in enumerate(gt_boxes):
            if j in matched_gt or plab.lower() != glab.lower():
                continue
            v = _iou(pbox, gbox)
            if v >= iou_threshold and v > best_iou:
                best_iou, best_j = v, j
        if best_j is not None:
            matched_gt.add(best_j)
            tp += 1
        else:
            fp += 1
    fn = len(gt_boxes) - len(matched_gt)
    prec = round(tp / (tp + fp), 4) if (tp + fp) > 0 else (1.0 if not gt_boxes else 0.0)
    rec = round(tp / (tp + fn), 4) if (tp + fn) > 0 else (1.0 if not pred_boxes else 0.0)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "iou_threshold": iou_threshold}


def compute_segmentation_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict[str, float]:
    """Compute IoU and Dice between two boolean masks."""
    pm = np.asarray(pred_mask, dtype=bool)
    gm = np.asarray(gt_mask, dtype=bool)
    inter = int((pm & gm).sum())
    union = int((pm | gm).sum())
    denom = int(pm.sum()) + int(gm.sum())
    return {
        "iou": round(inter / union, 4) if union > 0 else (1.0 if denom == 0 else 0.0),
        "dice": round((2.0 * inter) / denom, 4) if denom > 0 else 1.0,
    }


def compute_depth_metrics(pred_depth: np.ndarray, gt_depth: np.ndarray) -> Dict[str, float | None]:
    """Compute AbsRel, RMSE, and delta_1 (< 1.25) over finite positive pixels."""
    valid = np.isfinite(pred_depth) & np.isfinite(gt_depth) & (pred_depth > 0) & (gt_depth > 0)
    if not valid.any():
        return {"abs_rel": None, "rmse": None, "delta_1": None}
    p = pred_depth[valid].astype(np.float64)
    g = gt_depth[valid].astype(np.float64)
    abs_rel = float(np.mean(np.abs(p - g) / g))
    rmse = float(math.sqrt(np.mean((p - g) ** 2)))
    ratio = np.maximum(p / g, g / p)
    delta_1 = float(np.mean(ratio < 1.25))
    return {"abs_rel": round(abs_rel, 4), "rmse": round(rmse, 4), "delta_1": round(delta_1, 4)}
