"""Producers that turn real model output into ``metric3d.v1`` records with recoverable raw artifacts.

Two producers, both honest about scale:

* ``depth_records_from_drafts`` -- monocular depth. The FULL raw depth array is stored as a ``.npy`` artifact (the observation
  validator refuses raw ndarrays, so before this module no real depth map was ever persisted). The record's ``value`` is the
  summary ``(min, median, max)`` of the finite pixels; the pixels themselves live in ``raw_artifact_ref``. Relative depth is
  ``RELATIVE_DEPTH`` (units "relative"); a model that merely CLAIMS metres (e.g. an indoor-metric checkpoint) is ``UNKNOWN`` --
  metres are only ever ``METRIC_ANCHORED`` with an independent anchor, which a depth network is not. Test-double output and
  non-finite maps are kept as REJECTED records, never silently dropped or passed on.
* ``sparse_lift_records`` -- 2D regions -> 3D, using the sparse SfM points whose observation in that image falls inside the region.
  It is a visible-surface centroid in the reconstruction's own gauge (``SFM_GAUGE``, "sfm_units"), NOT an object centre and NOT
  metres; the quantity is named ``visible_surface_centroid`` so canonical fusion cannot mistake it for a position.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from engine.multiengine.evidence_matrix import EvidenceMatrix
from engine.multiengine.metric3d import Metric3DObservation, ScaleState
from engine.multiengine.observation import ObservationDraft

MIN_LIFT_POINTS = 5  # fewer sparse points than this inside a region is "not determinable", not a position
_RELATIVE_NAMES = {"relative", "relative_unscaled"}


def checkpoint_sha256(path: str | os.PathLike[str] | None) -> str | None:
    """sha256 of a checkpoint file, streamed. None when the file cannot be located (a gap that stays visible)."""
    if path is None or not Path(path).is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hf_checkpoint_file(repo_id: str) -> Path | None:
    """The cached weights file of a Hugging Face repo, or None."""
    home = Path(os.environ.get("HF_HUB_CACHE") or Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub")
    snaps = home / ("models--" + repo_id.replace("/", "--")) / "snapshots"
    if snaps.is_dir():
        for snap in sorted(snaps.iterdir()):
            for name in ("model.safetensors", "pytorch_model.bin"):
                if (snap / name).is_file():
                    return snap / name
    return None


def npy_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, np.asarray(arr), allow_pickle=False)
    return buf.getvalue()


def depth_records_from_drafts(
    drafts: Sequence[ObservationDraft],
    *,
    matrix: EvidenceMatrix,
    engine_id: str,
    architecture_id: str,
    checkpoint_id: str,
    checkpoint_hash: str | None,
) -> list[Metric3DObservation]:
    out: list[Metric3DObservation] = []
    for d in drafts:
        pred = dict(d.predicted)
        arr = pred.get("array")
        scale_name = str(pred.get("scale_state", "")).lower()
        base: dict[str, Any] = dict(
            engine_id=engine_id, architecture_id=architecture_id, checkpoint_id=checkpoint_id,
            checkpoint_sha256=checkpoint_hash, source_id=d.source_id, frame=d.coordinate_system,
        )
        if arr is None:  # nothing to persist: a producer that returns no pixels has produced no depth
            out.append(_rejected(base, "depth_relative", ScaleState.RELATIVE_DEPTH, "relative", None, "the engine returned no depth array"))
            continue
        a = np.asarray(arr)
        raw_ref, _ = matrix.artifacts.put(npy_bytes(a))  # the original array, byte-exact, recoverable by hash
        h, w = (int(a.shape[0]), int(a.shape[1])) if a.ndim >= 2 else (0, 0)
        if pred.get("is_test_double"):
            out.append(_rejected(base, "depth_relative", ScaleState.RELATIVE_DEPTH, "relative", raw_ref,
                                 "output of a test double, not a model; refused as real evidence"))
            continue
        if a.ndim != 2 or h <= 0 or w <= 0 or not np.issubdtype(a.dtype, np.number):
            out.append(_rejected(base, "depth_relative", ScaleState.RELATIVE_DEPTH, "relative", raw_ref,
                                 f"depth array must be a 2-D numeric map, got shape {a.shape} dtype {a.dtype}"))
            continue
        if not np.isfinite(a).all():
            out.append(_rejected(base, "depth_relative", ScaleState.RELATIVE_DEPTH, "relative", raw_ref,
                                 f"depth map has {int((~np.isfinite(a)).sum())} non-finite pixel(s)"))
            continue
        value = (float(a.min()), float(np.median(a)), float(a.max()))
        if scale_name in _RELATIVE_NAMES:
            state, units, quantity = ScaleState.RELATIVE_DEPTH, "relative", "depth_relative"
        else:  # includes a model's own metres claim: not independently anchored, so the scale is unknown
            state, units, quantity = ScaleState.UNKNOWN, "unknown", "depth_unanchored"
        out.append(Metric3DObservation(
            quantity=quantity, value=value, units=units, scale_state=state, image_size=(w, h), raw_artifact_ref=raw_ref, **base,
        ))
    return out


def produce_depth(
    adapter: Any,
    sources: Sequence[str],
    *,
    matrix: EvidenceMatrix,
    run_id: str,
    checkpoint_id: str,
    checkpoint_hash: str | None,
    device: Any | None = None,
) -> dict[str, Any]:
    """Run ONE depth adapter end to end (load -> infer -> unload, always released), convert and ingest its output. A failure is
    returned as ``status: failed`` with the error -- never swallowed and never replaced by another model."""
    import time

    from engine.multiengine.contract import Device, EngineRequest
    from engine.multiengine.metric3d import ingest

    dev = device or Device("cpu")
    spec = adapter.spec
    t0, status, error, drafts = time.perf_counter(), "succeeded", None, []
    try:
        adapter.validate_input(EngineRequest(sources=tuple(sources)))
        adapter.load(dev)
        drafts = list(adapter.infer(EngineRequest(sources=tuple(sources))))
    except Exception as exc:  # noqa: BLE001 -- recorded, not hidden
        status, error = "failed", f"{type(exc).__name__}: {exc}"
    finally:
        try:
            adapter.unload()
        except Exception as exc:  # noqa: BLE001
            error = (error + "; " if error else "") + f"unload failed: {exc}"
    vram, ram = (adapter.measured_usage() if hasattr(adapter, "measured_usage") else (None, None))
    recs = depth_records_from_drafts(
        drafts, matrix=matrix, engine_id=spec.engine_id, architecture_id=getattr(spec, "name", spec.engine_id),
        checkpoint_id=checkpoint_id, checkpoint_hash=checkpoint_hash,
    )
    res = ingest(matrix, recs, run_id=run_id, model_version=getattr(spec, "version", "unknown"))
    return {"engine_id": spec.engine_id, "status": status, "error": error, "sources": list(sources), "records": len(recs),
            "accepted": res["accepted"], "rejected": res["rejected"], "wall_s": round(time.perf_counter() - t0, 3),
            "peak_vram_mb": vram, "peak_ram_mb": ram, "checkpoint_id": checkpoint_id, "checkpoint_sha256": checkpoint_hash}


def _rejected(base: Mapping[str, Any], quantity: str, state: ScaleState, units: str, raw_ref: str | None, reason: str) -> Metric3DObservation:
    return Metric3DObservation(
        quantity=quantity, value=(), units=units, scale_state=state, raw_artifact_ref=raw_ref,
        validation_status="rejected", rejection_reason=reason, **base,
    )


def sparse_lift_records(
    regions: Sequence[Mapping[str, Any]],
    *,
    points2d: np.ndarray,
    points3d: np.ndarray,
    matrix: EvidenceMatrix,
    source_id: str,
    camera_id: str | None,
    image_size: tuple[int, int] | None,
    frame: str,
    engine_id: str = "sparse_sfm_lifter",
    architecture_id: str = "sparse-point-region-lifter",
    checkpoint_id: str = "none:geometric-algorithm",
) -> list[Metric3DObservation]:
    """Lift 2D regions with the image's own sparse SfM points. ``points2d[i]`` is the pixel at which ``points3d[i]`` was observed
    in THIS image. Each region becomes a robust (median + MAD-trimmed) centroid with the sample covariance of the mean."""
    p2, p3 = np.asarray(points2d, dtype=float), np.asarray(points3d, dtype=float)
    if p2.ndim != 2 or p2.shape[1] != 2 or p3.shape != (p2.shape[0], 3):
        raise ValueError("points2d must be Nx2 and points3d Nx3 with matching rows")
    keep = np.isfinite(p2).all(axis=1) & np.isfinite(p3).all(axis=1)
    p2, p3 = p2[keep], p3[keep]
    out: list[Metric3DObservation] = []
    for k, reg in enumerate(regions):
        box = tuple(float(v) for v in reg["box_2d"])
        base: dict[str, Any] = dict(
            engine_id=engine_id, architecture_id=architecture_id, checkpoint_id=checkpoint_id, checkpoint_sha256=None,
            source_id=source_id, camera_id=camera_id, image_size=image_size, image_region=box, frame=frame,
            confidence=reg.get("confidence"), confidence_basis="raw_model" if reg.get("confidence") is not None else None,
        )
        inside = (p2[:, 0] >= box[0]) & (p2[:, 0] <= box[2]) & (p2[:, 1] >= box[1]) & (p2[:, 1] <= box[3])
        pts = p3[inside]
        support = {"region_index": k, "label": reg.get("label"), "box_2d": list(box), "n_points_in_region": int(len(pts)),
                   "point_indices": [int(i) for i in np.flatnonzero(keep)[inside]], "points3d": pts.tolist()}
        raw_ref, _ = matrix.artifacts.put(json.dumps(support, sort_keys=True).encode("utf-8"))
        if len(pts) < MIN_LIFT_POINTS:
            out.append(Metric3DObservation(
                quantity="visible_surface_centroid", value=(), units="sfm_units", scale_state=ScaleState.SFM_GAUGE,
                raw_artifact_ref=raw_ref, validation_status="rejected",
                rejection_reason=f"only {len(pts)} sparse SfM point(s) fall inside the region (need {MIN_LIFT_POINTS}); position not determinable",
                **base,
            ))
            continue
        med = np.median(pts, axis=0)
        dist = np.linalg.norm(pts - med, axis=1)
        mad = float(np.median(np.abs(dist - np.median(dist))))
        inl = pts[dist <= np.median(dist) + 3.0 * 1.4826 * mad] if mad > 0 else pts
        if len(inl) < MIN_LIFT_POINTS:
            inl = pts
        centroid = np.median(inl, axis=0)
        cov = np.cov(inl.T) / len(inl)  # covariance of the estimate (standard error of the mean), in sfm_units^2
        out.append(Metric3DObservation(
            quantity="visible_surface_centroid", value=tuple(float(v) for v in centroid), units="sfm_units",
            scale_state=ScaleState.SFM_GAUGE, covariance=tuple(tuple(float(x) for x in r) for r in cov),
            raw_artifact_ref=raw_ref, **base,
        ))
    return out
