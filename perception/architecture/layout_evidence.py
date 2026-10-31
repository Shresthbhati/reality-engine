"""Single-image room-layout evidence (plan F4).

A learned layout estimator (HorizonNet / LayoutNet style) predicts the wall-floor and wall-ceiling boundary lines of a
room from ONE photograph. Those lines are useful exactly where the point cloud is sparse: a wall nobody photographed
squarely still has a visible floor junction.

This module is the fusion half of F4, and it is deliberately honest about the missing half:

* **The learned estimator itself is not shipped.** No HorizonNet/LayoutNet weights exist in this repository, and the
  choice of detector/model is an open product decision (plan section 6, item 3). ``estimator_status()`` reports that
  as UNAVAILABLE with the reason and what would unblock it -- it never pretends a model is installed.
* **Evidence is accepted when it exists.** A capture may carry ``layout/<stem>.json`` sidecars produced by any
  estimator, in the documented format below. They are parsed, validated and projected onto the floor plane using the
  registered camera pose, so the fusion is real geometry, not a stub.
* **Fusion is additive and lower-confidence.** ``solve_room_layout`` keeps the measured points as the authority; a
  layout edge can only turn a wall that no point supports from INFERRED into OBSERVED-with-a-layout-basis. It can
  never delete, move or outrank a measured wall.

Sidecar format (JSON, one file per photo, ``layout/<stem>.json``)::

    {
      "source": "horizonnet",              # which estimator produced it
      "confidence": 0.6,                   # optional, 0..1
      "image_size": [640, 480],            # optional; defaults to the capture's image size
      "wall_floor": [[u0, v0, u1, v1], ...],   # boundary segments, NORMALISED pixel coords (0..1)
      "wall_ceiling": [[u0, v0, u1, v1], ...]  # optional, recorded but not used for plan support
    }

Normalised coordinates make the file independent of the photo resolution; a missing ``image_size`` is therefore fine.
The wall-floor boundary is the junction that lies on the floor plane, which is what makes metric projection possible.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

#: Best estimate of the local detector's availability -- reported, never assumed.
LAYOUT_MODEL_STATUS = {
    "status": "unavailable",
    "reason": ("no learned single-image room-layout estimator is installed: HorizonNet/LayoutNet weights are not in "
               "this repository and installing a specific model is a product decision (plan section 6, item 3)"),
    "accepted_input": "layout/<stem>.json sidecars from any estimator, in the format documented in this module",
    "unblocked_by": "choosing and installing a layout model (offline weights or a CPU-runnable package)",
}

#: A projected layout edge supports a wall when the wall's samples lie this close to it (metres, plan).
SUPPORT_BAND_M = 0.25
#: Samples per wall when measuring layout support.
SUPPORT_SAMPLES = 32
#: Confidence assigned to a wall seen only by a layout boundary (a learned line is weaker evidence than points).
LAYOUT_BASIS_CONFIDENCE = 0.4


class LayoutEvidenceError(ValueError):
    """A layout sidecar is malformed or cannot be projected; the reason is recorded, not guessed around."""


@dataclass(frozen=True)
class LayoutHypothesis:
    evidence_id: str
    source: str
    wall_floor: Tuple[Tuple[float, float, float, float], ...]
    wall_ceiling: Tuple[Tuple[float, float, float, float], ...] = ()
    confidence: float = 0.5
    image_size: Optional[Tuple[int, int]] = None

    def to_dict(self) -> dict:
        return {"evidence_id": self.evidence_id, "source": self.source,
                "wall_floor": [list(s) for s in self.wall_floor],
                "wall_ceiling": [list(s) for s in self.wall_ceiling],
                "confidence": self.confidence,
                "image_size": list(self.image_size) if self.image_size else None}


def _segments(raw, label: str) -> Tuple[Tuple[float, float, float, float], ...]:
    out = []
    for i, seg in enumerate(raw or ()):
        try:
            u0, v0, u1, v1 = (float(v) for v in seg)
        except (TypeError, ValueError) as exc:
            raise LayoutEvidenceError(f"{label}[{i}] is not four numbers: {seg!r}") from exc
        if not all(math.isfinite(v) for v in (u0, v0, u1, v1)):
            raise LayoutEvidenceError(f"{label}[{i}] has a non-finite coordinate: {seg!r}")
        if not (-0.05 <= u0 <= 1.05 and -0.05 <= v0 <= 1.05
                and -0.05 <= u1 <= 1.05 and -0.05 <= v1 <= 1.05):
            raise LayoutEvidenceError(f"{label}[{i}] is outside normalised image bounds: {seg!r}")
        if abs(u1 - u0) + abs(v1 - v0) < 1e-6:
            continue                       # a zero-length line is not evidence
        out.append((u0, v0, u1, v1))
    return tuple(out)


def parse_layout_hypothesis(data: dict, *, evidence_id: str, default_size=None) -> LayoutHypothesis:
    """Validate one sidecar. Raises LayoutEvidenceError with the offending field, never returns a guess."""
    if not isinstance(data, dict):
        raise LayoutEvidenceError("layout sidecar must be a JSON object")
    floor = _segments(data.get("wall_floor"), "wall_floor")
    if not floor:
        raise LayoutEvidenceError("wall_floor is required and must contain at least one boundary segment")
    ceiling = _segments(data.get("wall_ceiling"), "wall_ceiling")
    try:
        confidence = float(data.get("confidence", 0.5))
    except (TypeError, ValueError) as exc:
        raise LayoutEvidenceError(f"confidence is not a number: {data.get('confidence')!r}") from exc
    if not (0.0 <= confidence <= 1.0):
        raise LayoutEvidenceError(f"confidence must be in [0, 1], got {confidence!r}")
    size = data.get("image_size") or default_size
    if size is not None:
        try:
            w, h = int(size[0]), int(size[1])
        except (TypeError, ValueError, IndexError) as exc:
            raise LayoutEvidenceError(f"image_size is not [width, height]: {size!r}") from exc
        if w <= 0 or h <= 0:
            raise LayoutEvidenceError(f"image_size must be positive, got {size!r}")
        size = (w, h)
    return LayoutHypothesis(evidence_id=evidence_id, source=str(data.get("source") or "unknown"),
                            wall_floor=floor, wall_ceiling=ceiling, confidence=confidence, image_size=size)


def load_layout_sidecars(dataset_dirs: Iterable[Path], *, default_size=None) -> Tuple[Dict[str, LayoutHypothesis], List[str]]:
    """Read ``layout/<stem>.json`` from each capture directory. Returns (by evidence id, errors)."""
    found: Dict[str, LayoutHypothesis] = {}
    errors: List[str] = []
    for root in dataset_dirs:
        layout_dir = Path(root) / "layout"
        if not layout_dir.is_dir():
            continue
        for path in sorted(layout_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                hyp = parse_layout_hypothesis(data, evidence_id=path.stem, default_size=default_size)
            except (OSError, ValueError) as exc:      # JSONDecodeError is a ValueError
                errors.append(f"{path}: {exc}")
                continue
            found[hyp.evidence_id] = hyp
    return found, errors


def estimator_status() -> dict:
    return dict(LAYOUT_MODEL_STATUS)


def _quat_matrix(quat: Sequence[float]) -> np.ndarray:
    w, x, y, z = (float(v) for v in quat)
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def project_floor_edges(pose, segments: Sequence[Sequence[float]], intrinsics, floor_y: float,
                        image_size: Optional[Tuple[int, int]] = None) -> Tuple[List[Tuple[Tuple[float, float],
                                                                                         Tuple[float, float]]], List[str]]:
    """Project normalised wall-floor boundary segments onto the horizontal plane ``y = floor_y``.

    Returns plan edges ``((x, z), (x, z))`` in the reconstruction's world frame and the reasons any segment was
    dropped. A segment whose endpoints do not hit the floor plane in front of the camera is dropped, not clamped.
    """
    fx, fy, cx, cy = (float(v) for v in intrinsics[:4])
    width, height = image_size or (int(2 * cx), int(2 * cy))
    R = _quat_matrix(pose.rotation)
    C = np.asarray(pose.position, float)
    edges, errors = [], []
    for i, seg in enumerate(segments):
        u0, v0, u1, v1 = seg
        pts = []
        for u, v in ((u0, v0), (u1, v1)):
            ray = np.array([(u * width - cx) / fx, (v * height - cy) / fy, 1.0], float)
            d = R @ ray
            if abs(d[1]) < 1e-6:
                pts = []
                errors.append(f"{pose.evidence_id}: wall_floor[{i}] is parallel to the floor plane")
                break
            t = (floor_y - C[1]) / d[1]
            if not (t > 1e-6) or not math.isfinite(t):
                pts = []
                errors.append(f"{pose.evidence_id}: wall_floor[{i}] does not reach the floor in front of the camera")
                break
            p = C + t * d
            pts.append((float(p[0]), float(p[2])))
        if len(pts) == 2 and math.hypot(pts[1][0] - pts[0][0], pts[1][1] - pts[0][1]) > 1e-3:
            edges.append((pts[0], pts[1]))
    return edges, errors


def layout_support(a: Sequence[float], b: Sequence[float], edges: Sequence[Sequence[Sequence[float]]],
                   *, band_m: float = SUPPORT_BAND_M, samples: int = SUPPORT_SAMPLES) -> Optional[float]:
    """Fraction of the wall a->b (plan coordinates) that lies within ``band_m`` of a projected layout edge.

    None when there is no layout evidence at all (nothing is claimed); 0.0..1.0 otherwise.
    """
    if not edges:
        return None
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    length = float(np.linalg.norm(b - a))
    if length < 1e-6:
        return None
    ts = (np.arange(samples) + 0.5) / samples
    pts = a[None, :] + ts[:, None] * (b - a)[None, :]
    segs = [np.asarray(e, float) for e in edges if len(e) == 2]
    supported = 0
    for p in pts:
        if any(_point_segment_distance(p, e[0], e[1]) <= band_m for e in segs):
            supported += 1
    return round(supported / samples, 4)


def _point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    d = b - a
    denom = float(d @ d)
    if denom < 1e-12:
        return float(np.linalg.norm(p - a))
    t = float(np.clip((p - a) @ d / denom, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * d)))


def edges_from_hypotheses(result, hypotheses: Dict[str, LayoutHypothesis], intrinsics,
                          floor_y: float) -> Tuple[List[Tuple[Tuple[float, float], Tuple[float, float]]], dict]:
    """Project every posed hypothesis; returns (edges, facts) with per-view counts and errors."""
    poses = {p.evidence_id: p for p in result.camera_poses}
    edges: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    errors: List[str] = []
    used, skipped = [], []
    for eid in sorted(hypotheses):
        hyp = hypotheses[eid]
        pose = poses.get(eid)
        if pose is None:
            skipped.append(eid)
            continue
        got, errs = project_floor_edges(pose, hyp.wall_floor, intrinsics, floor_y, hyp.image_size)
        errors.extend(errs)
        if got:
            edges.extend(got)
            used.append(eid)
        else:
            skipped.append(eid)
    facts = {"status": "ran" if edges else "unavailable",
             "source": "layout_sidecar", "hypotheses": len(hypotheses), "views_used": used,
             "views_skipped": skipped[:10], "edges": len(edges), "errors": errors[:10],
             "floor_plane_y_m": round(float(floor_y), 4),
             "estimator": estimator_status()}
    if not edges:
        facts["note"] = "layout evidence was supplied but no boundary projected to a usable floor edge"
    return edges, facts
