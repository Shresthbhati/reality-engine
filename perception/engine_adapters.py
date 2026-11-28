"""Engine adapters — wrap Reality Engine's existing engines into the
UnifiedEngineOutput contract (perception/engine_evidence.py).

Each adapter is a function that takes image paths and optional context
and returns a UnifiedEngineOutput. These adapters are registered with the
MultiEngineOrchestrator (perception/engine_orchestrator.py).

Available adapters:
  - SIFT matcher (OpenCV SIFT + RANSAC)
  - LightGlue-ALIKED matcher (torch + kornia)
  - LightGlue-DISK matcher (torch + kornia)
  - ORB matcher (OpenCV ORB)
  - Depth Anything V2 (if installed)
  - MiDaS depth (if installed)
  - SAM 2.1 segmentation (if installed)
  - Mask R-CNN detection (if installed)
  - Grounding DINO detection (if installed)
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Sequence
from typing import Any

import numpy as np

from perception.engine_evidence import (
    CoordinateFrame,
    ModelIdentity,
    ObservationRecord,
    UnifiedEngineOutput,
)
from perception.engine_orchestrator import EngineSpec, register_engine

log = logging.getLogger("reality.engine_adapters")

# ---------------------------------------------------------------------------
# Helper: build a ModelIdentity
# ---------------------------------------------------------------------------

def _model(engine_id: str, arch: str, checkpoint: str = "none", license: str = "unknown",
           non_commercial: bool = False, hardware: str = "cpu") -> ModelIdentity:
    return ModelIdentity(engine_id=engine_id, architecture_id=arch, checkpoint_id=checkpoint,
                         license=license, non_commercial=non_commercial, hardware_class=hardware)


# ---------------------------------------------------------------------------
# Two-view geometric verification (measured statistics, never invented ones)
# ---------------------------------------------------------------------------

def _camera_matrix(image_path: str, ctx: dict[str, Any]) -> np.ndarray | None:
    """The 3x3 intrinsic matrix of an image from the run context, or None if unknown.

    Context layout: ``ctx["intrinsics"] = {path: {"fx": .., "fy": .., "cx": .., "cy": ..}}``.
    A relative pose is only estimated when both cameras' intrinsics are known; without
    them the pair gets epipolar statistics only, and says so.
    """
    cams = ctx.get("intrinsics") or {}
    if not isinstance(cams, dict):
        return None
    base = os.path.basename(image_path)
    for key in (image_path, str(image_path).replace("\\", "/"), base):
        intr = cams.get(key)
        if not intr:
            continue
        try:
            fx, fy = float(intr["fx"]), float(intr["fy"])
            cx, cy = float(intr["cx"]), float(intr["cy"])
        except (KeyError, TypeError, ValueError):
            continue
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    return None


def _camera_distortion(image_path: str, ctx: dict[str, Any]) -> np.ndarray | None:
    """The camera's distortion coefficients from the run context, or None.

    A missing distortion model is not treated as "no distortion": the pair records
    which calibration it used, because ignoring a real radial term biases the
    essential matrix (measured on the south_building capture: 0.84 deg median pair
    rotation error without it, 0.65 deg with it, and a coherent rotation bias that
    compounded to ~19 deg per camera in the pose graph).
    """
    cams = ctx.get("intrinsics") or {}
    if not isinstance(cams, dict):
        return None
    base = os.path.basename(image_path)
    for key in (image_path, str(image_path).replace("\\", "/"), base):
        intr = cams.get(key)
        if not intr:
            continue
        dist = intr.get("dist")
        if dist is None:
            return None
        try:
            return np.asarray([float(v) for v in dist], dtype=np.float64)
        except (TypeError, ValueError):
            return None
    return None


def _sampson_distances(F: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Per-correspondence Sampson (first-order epipolar) distance in pixels."""
    F = np.asarray(F, dtype=np.float64).reshape(3, 3)
    x1 = np.hstack([p1, np.ones((len(p1), 1))])
    x2 = np.hstack([p2, np.ones((len(p2), 1))])
    Fx1 = x1 @ F.T
    Ftx2 = x2 @ F
    num = np.sum(x2 * Fx1, axis=1) ** 2
    den = Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sqrt(num / den)


def _geometric_verification(p1: np.ndarray, p2: np.ndarray,
                            K1: np.ndarray | None, K2: np.ndarray | None,
                            dist1: np.ndarray | None = None,
                            dist2: np.ndarray | None = None) -> dict[str, Any]:
    """Measured two-view geometry for a putative correspondence set.

    Everything returned is measured from the matches: RANSAC inlier count and ratio, the
    Sampson residual of the inliers, and -- only when both intrinsics are known -- the
    essential matrix and the relative pose recovered from it. The translation of a
    two-view pose is a unit DIRECTION: two-view geometry carries no scale, so none is
    claimed. ``verified`` is True only when RANSAC found an epipolar geometry the matches
    actually satisfy; a low-inlier pair stays unverified rather than being reported as a
    match.
    """
    import cv2

    p1 = np.asarray(p1, dtype=np.float64).reshape(-1, 2)
    p2 = np.asarray(p2, dtype=np.float64).reshape(-1, 2)
    out: dict[str, Any] = {"putative_matches": int(len(p1)), "verified": False}
    if len(p1) < 8 or len(p2) < 8:
        out["reason"] = "fewer than 8 putative matches"
        return out
    F, mask = cv2.findFundamentalMat(p1, p2, cv2.FM_RANSAC, 0.5, 0.999, 5000)
    inlier_mask = np.zeros(len(p1), dtype=bool) if mask is None else mask.ravel().astype(bool)
    inliers = int(inlier_mask.sum())
    out["inliers"] = inliers
    out["inlier_ratio"] = round(inliers / max(1, len(p1)), 4)
    if F is not None and F.shape == (3, 3) and inliers >= 8:
        d = _sampson_distances(F, p1[inlier_mask], p2[inlier_mask])
        finite = d[np.isfinite(d)]
        if finite.size:
            out["epipolar_residual_median_px"] = round(float(np.median(finite)), 4)
            out["epipolar_residual_p90_px"] = round(float(np.percentile(finite, 90)), 4)
        out["verified"] = True
    if out["verified"] and K1 is not None and K2 is not None:
        # Normalised coordinates, then a shared camera matrix of identity: the correct
        # essential-matrix form when the two cameras have different intrinsics.
        n1 = cv2.undistortPoints(p1[inlier_mask].reshape(-1, 1, 2), K1, dist1).reshape(-1, 2)
        n2 = cv2.undistortPoints(p2[inlier_mask].reshape(-1, 1, 2), K2, dist2).reshape(-1, 2)
        E, _mask_e = cv2.findEssentialMat(n1, n2, np.eye(3), method=cv2.RANSAC,
                                          prob=0.999, threshold=1e-3)
        if E is not None and np.asarray(E).shape == (3, 3):
            n_ok, R, t, _mask_pose = cv2.recoverPose(E, n1, n2, np.eye(3))
            if R is not None and t is not None:
                out["relative_pose"] = {
                    "R": [[round(float(v), 6) for v in row] for row in np.asarray(R)],
                    "t_unit": [round(float(v), 6) for v in np.asarray(t).reshape(-1)],
                    "cheirality_inliers": int(n_ok),
                    "cheirality_inlier_fraction": round(float(n_ok) / max(1, inliers), 4),
                    "scale": "none: the translation is a unit direction; two-view geometry cannot recover scale",
                    "convention": "x2 ~ R @ x1 + t (OpenCV recoverPose)",
                    "calibration": ("intrinsics + distortion coefficients"
                                    if (dist1 is not None or dist2 is not None)
                                    else "intrinsics only (no distortion coefficients supplied)"),
                }
    return out


# ---------------------------------------------------------------------------
# SIFT Matcher Adapter
# ---------------------------------------------------------------------------

def _sift_available() -> str | None:
    try:
        import cv2
        cv2.SIFT_create()
        return None
    except Exception as exc:
        return str(exc)


def _sift_run(image_paths: Sequence[str], **ctx: Any) -> UnifiedEngineOutput:
    import time as _t

    import cv2

    t0 = _t.perf_counter()
    run_id = ctx.get("run_id", "sift_run")
    obs: list[ObservationRecord] = []
    unverified: list[dict[str, Any]] = []
    pairs = ctx.get("pairs", [])
    if not pairs and len(image_paths) >= 2:
        pairs = [(image_paths[i], image_paths[i + 1]) for i in range(len(image_paths) - 1)]

    # One detector for the whole run: creating it per pair wasted time and gave every
    # image new keypoints. Feature extraction is cached per image across pairs.
    sift = cv2.SIFT_create(nfeatures=4000)
    cache: dict[str, tuple[list[Any] | None, np.ndarray | None, int]] = {}

    def _features(path: str):
        if path not in cache:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                cache[path] = (None, None, 0)
            else:
                kp, des = sift.detectAndCompute(img, None)
                cache[path] = (kp, des, len(kp) if kp else 0)
        return cache[path]

    for a_path, b_path in pairs:
        pair_t0 = _t.perf_counter()
        try:
            kp1, des1, nkp1 = _features(a_path)
            kp2, des2, nkp2 = _features(b_path)
            if des1 is None or des2 is None or len(des1) < 8 or len(des2) < 8:
                unverified.append({"a_path": a_path, "b_path": b_path, "reason": "too few descriptors"})
                continue
            bf = cv2.BFMatcher(cv2.NORM_L2)
            matches = bf.knnMatch(des1, des2, k=2)
            good = [m[0] for m in matches if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]
            if len(good) < 8:
                unverified.append({"a_path": a_path, "b_path": b_path, "reason": "fewer than 8 ratio-test matches",
                                   "putative_matches": len(good)})
                continue
            p1 = np.float32([kp1[m.queryIdx].pt for m in good])
            p2 = np.float32([kp2[m.trainIdx].pt for m in good])
            stats = _geometric_verification(p1, p2, _camera_matrix(a_path, ctx), _camera_matrix(b_path, ctx),
                                            _camera_distortion(a_path, ctx), _camera_distortion(b_path, ctx))
            if not stats.get("verified"):
                unverified.append({"a_path": a_path, "b_path": b_path, "reason": "no epipolar geometry satisfied",
                                   **{k: v for k, v in stats.items() if k != "verified"}})
                continue
            obs.append(ObservationRecord(
                evidence_id=f"sift:{a_path}->{b_path}",
                source_image_id=a_path,
                coordinate_frame=CoordinateFrame(frame_id="image", scale_state="image", units="pixels"),
                model_id=_model("matching", "sift", "opencv", "Apache-2.0", False, "cpu"),
                label="feature_match",
                category="correspondence",
                confidence=stats["inlier_ratio"],
                sigma_m=None,
                calibration_quality="unverified",
                metadata={"a_path": a_path, "b_path": b_path, "keypoints_a": nkp1, "keypoints_b": nkp2,
                          "matches": stats["putative_matches"], **stats,
                          "runtime_s": round(_t.perf_counter() - pair_t0, 4),
                          "confidence_basis": "RANSAC inlier ratio (measured, uncalibrated)"},
            ))
        except Exception as exc:
            log.warning("SIFT pair failed %s->%s: %s", a_path, b_path, exc)
            unverified.append({"a_path": a_path, "b_path": b_path, "reason": f"{type(exc).__name__}: {exc}"})

    return UnifiedEngineOutput(
        engine_id="sift", run_id=run_id, observations=tuple(obs),
        status="succeeded" if obs else "partial",
        wall_time_s=time.perf_counter() - t0,
        summary={"n_pairs": len(pairs), "n_verified_pairs": len(obs), "n_unverified_pairs": len(unverified),
                 "unverified": unverified[:50]},
    )


# ---------------------------------------------------------------------------
# LightGlue-ALIKED Adapter
# ---------------------------------------------------------------------------

def _lightglue_aliked_available() -> str | None:
    try:
        from importlib import import_module

        import_module("kornia.feature")
        import_module("torch")
        return None
    except (ImportError, AttributeError) as exc:
        return str(exc)


def _lightglue_run(image_paths: Sequence[str], **ctx: Any) -> UnifiedEngineOutput:
    import time as _t

    import cv2
    import kornia.feature as kf
    import torch
    from kornia.feature import LightGlue

    t0 = _t.perf_counter()
    run_id = ctx.get("run_id", "lightglue_run")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats(0)
        except Exception:
            pass
    max_long_edge_px = int(ctx.get("max_long_edge_px", 1600) or 1600)
    obs: list[ObservationRecord] = []
    unverified: list[dict[str, Any]] = []
    pairs = ctx.get("pairs", [])
    if not pairs and len(image_paths) >= 2:
        pairs = [(image_paths[i], image_paths[i + 1]) for i in range(len(image_paths) - 1)]

    # The extractor and the matcher are loaded ONCE for the whole run. They used to be
    # constructed inside the pair loop, which reloaded the weights for every pair
    # (seconds of I/O per pair, and a new peak allocation each time on a 6 GB card).
    try:
        aliked = kf.ALIKED.from_pretrained("aliked-n16")
        aliked = aliked.to(device).eval()
    except Exception:
        try:
            aliked = kf.ALIKED()
            aliked = aliked.to(device).eval()
        except Exception as exc2:
            return UnifiedEngineOutput(engine_id="lightglue_aliked", run_id=run_id, observations=(),
                                      status="failed", error_message=str(exc2), wall_time_s=time.perf_counter() - t0)
    lg = LightGlue(features="aliked").to(device).eval()
    peak: float | None = None

    def _feats(path: str):
        img = cv2.imread(path)
        if img is None:
            return None, None, 1.0, 1.0
        oh, ow = img.shape[:2]
        scale_x = 1.0
        scale_y = 1.0
        if max_long_edge_px > 0 and max(oh, ow) > max_long_edge_px:
            ratio = max_long_edge_px / float(max(oh, ow))
            nw = max(16, int(round(ow * ratio)))
            nh = max(16, int(round(oh * ratio)))
            img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
            scale_x = ow / float(nw)
            scale_y = oh / float(nh)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)
        with torch.no_grad():
            f = aliked(tensor)[0]
        return f, tensor, scale_x, scale_y

    try:
        for a_path, b_path in pairs:
            pair_t0 = _t.perf_counter()
            try:
                feats_a, tensor_a, sx_a, sy_a = _feats(a_path)
                feats_b, tensor_b, sx_b, sy_b = _feats(b_path)
                if feats_a is None or feats_b is None:
                    unverified.append({"a_path": a_path, "b_path": b_path, "reason": "unreadable image"})
                    continue
                with torch.no_grad():
                    matches = lg({
                        "image0": {"keypoints": feats_a.keypoints.unsqueeze(0),
                                   "descriptors": feats_a.descriptors.unsqueeze(0),
                                   "image": tensor_a,
                                   "matching_scores": feats_a.keypoint_scores.unsqueeze(0)},
                        "image1": {"keypoints": feats_b.keypoints.unsqueeze(0),
                                   "descriptors": feats_b.descriptors.unsqueeze(0),
                                   "image": tensor_b,
                                   "matching_scores": feats_b.keypoint_scores.unsqueeze(0)},
                    })
                mt = matches.get("matches")
                if mt is None:
                    unverified.append({"a_path": a_path, "b_path": b_path, "reason": "matcher returned no matches"})
                    continue
                # kornia's LightGlue returns a LIST of per-image tensors (batch of this call is
                # one pair); older/newer versions return a single (B, N, 2) tensor. Handle both
                # instead of silently failing every pair on the list form.
                if isinstance(mt, (list, tuple)):
                    mt = mt[0] if mt else torch.zeros((0, 2), dtype=torch.long)
                mt = mt.detach().cpu() if hasattr(mt, "detach") else torch.as_tensor(mt)
                mt = mt[0] if mt.ndim == 3 else mt
                mt = mt.reshape(-1, 2).long()
                if mt.shape[0] < 8:
                    unverified.append({"a_path": a_path, "b_path": b_path, "reason": "fewer than 8 matches",
                                       "putative_matches": int(mt.shape[0])})
                    continue
                kp_a = feats_a.keypoints.detach().cpu().reshape(-1, 2).numpy()
                kp_b = feats_b.keypoints.detach().cpu().reshape(-1, 2).numpy()
                if sx_a != 1.0 or sy_a != 1.0:
                    kp_a = kp_a * np.array([sx_a, sy_a], dtype=kp_a.dtype)
                if sx_b != 1.0 or sy_b != 1.0:
                    kp_b = kp_b * np.array([sx_b, sy_b], dtype=kp_b.dtype)
                p1 = kp_a[mt[:, 0].numpy()]
                p2 = kp_b[mt[:, 1].numpy()]
                # Confidence is LightGlue's OWN per-match score, never a constant. When the
                # matcher does not return scores there is no measured confidence at all, and
                # the record says so (None) instead of claiming certainty.
                mean_score: float | None = None
                scores_tensor = matches.get("scores")
                if isinstance(scores_tensor, (list, tuple)):
                    scores_tensor = scores_tensor[0] if scores_tensor else None
                if scores_tensor is not None and hasattr(scores_tensor, "numel") and scores_tensor.numel() > 0:
                    mean_score = float(scores_tensor.detach().float().mean().item())
                    if not (0.0 <= mean_score <= 1.0):
                        mean_score = None
                stats = _geometric_verification(p1, p2, _camera_matrix(a_path, ctx), _camera_matrix(b_path, ctx),
                                                _camera_distortion(a_path, ctx), _camera_distortion(b_path, ctx))
                if not stats.get("verified"):
                    unverified.append({"a_path": a_path, "b_path": b_path,
                                       "reason": "no epipolar geometry satisfied",
                                       **{k: v for k, v in stats.items() if k != "verified"}})
                    continue
                obs.append(ObservationRecord(
                    evidence_id=f"lg_aliked:{a_path}->{b_path}",
                    source_image_id=a_path,
                    coordinate_frame=CoordinateFrame(frame_id="image", scale_state="image", units="pixels"),
                    model_id=_model("matching", "lightglue_aliked", "aliked", "Apache-2.0", False, device),
                    label="feature_match",
                    category="correspondence",
                    confidence=mean_score,
                    sigma_m=None,
                    calibration_quality="unverified",
                    metadata={"a_path": a_path, "b_path": b_path, "keypoints_a": int(len(kp_a)),
                              "keypoints_b": int(len(kp_b)), "matches": stats["putative_matches"], **stats,
                              "mean_match_score": mean_score,
                              "runtime_s": round(_t.perf_counter() - pair_t0, 4),
                              "confidence_basis": "mean LightGlue match score (uncalibrated)" if mean_score is not None
                              else "matcher returned no scores: no measured confidence"},
                ))
            except Exception as exc:
                log.warning("LightGlue-ALIKED pair failed %s->%s: %s", a_path, b_path, exc)
                unverified.append({"a_path": a_path, "b_path": b_path, "reason": f"{type(exc).__name__}: {exc}"})
        peak = _get_vram_peak()
    finally:
        # Release the GPU before the next engine can start: on a 6 GB card one engine's
        # allocations must not sit under the next one's.
        aliked = lg = None  # noqa: F841 -- dropping the references is the point
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    if unverified and not obs:
        lg_status = "failed"
    elif unverified and obs:
        lg_status = "partial"
    elif obs:
        lg_status = "succeeded"
    else:
        lg_status = "partial"

    return UnifiedEngineOutput(
        engine_id="lightglue_aliked", run_id=run_id, observations=tuple(obs),
        status=lg_status,
        wall_time_s=time.perf_counter() - t0,
        peak_vram_mb=peak,
        summary={"n_pairs": len(pairs), "n_verified_pairs": len(obs), "n_unverified_pairs": len(unverified),
                 "unverified": unverified[:50]},
    )


def _get_vram_peak() -> float | None:
    try:
        import torch  # imported here: this module must import without torch installed

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated(0) / (1024.0 * 1024.0)
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Registration — register all available engines
# ---------------------------------------------------------------------------

def register_all_engines() -> list[str]:
    """Register all available engines with the MultiEngineOrchestrator."""
    registered = []

    # Detection (Grounding DINO) — runs first (highest priority after feature matching)
    try:
        from perception.detection_adapter import register_detection_engine
        register_detection_engine()
        registered.append("grounding_dino")
    except Exception as exc:
        log.warning("Detection engine not registered: %s", exc)

    # Segmentation (SAM2) — depends on detection bboxes
    try:
        from perception.segmentation_adapter import register_segmentation_engine
        register_segmentation_engine()
        registered.append("sam2")
    except Exception as exc:
        log.warning("Segmentation engine not registered: %s", exc)

    # SIFT (always available)
    register_engine(
        EngineSpec(engine_id="sift", architecture_id="sift", checkpoint_id="opencv",
                   vram_mb=0, cpu_threads=4, priority=80, license="Apache-2.0",
                   dependencies=(), run_condition=_sift_available),
        _sift_run,
    )
    registered.append("sift")

    # LightGlue-ALIKED (available if torch + kornia)
    register_engine(
        EngineSpec(engine_id="lightglue_aliked", architecture_id="lightglue_aliked",
                   checkpoint_id="aliked", vram_mb=1024, cpu_threads=0, priority=90,
                   license="Apache-2.0", dependencies=(), run_condition=_lightglue_aliked_available),
        _lightglue_run,
    )
    registered.append("lightglue_aliked")

    # Depth (Depth Anything V2)
    try:
        from perception.depth_adapter import register_depth_engine
        register_depth_engine()
        registered.append("depth_anything")
    except Exception as exc:
        log.warning("Depth engine not registered: %s", exc)

    # Depth (MiDaS DPT_Hybrid) -- a second, implemented depth backend that was not
    # wired. Register it so the production job runs it as a first-class depth engine.
    # Depth is relative inverse depth (MiDaS convention), not metric.
    try:
        from perception.midas_adapter import register_midas_engine
        register_midas_engine()
        registered.append("midas")
    except Exception as exc:
        log.warning("MiDaS depth engine not registered: %s", exc)

    # Detection + instance segmentation (Mask R-CNN, torchvision COCO-pretrained) --
    # a second detection engine (boxes + instance masks) that was implemented but not
    # wired. Register it so the production job runs it; its observations feed the
    # orchestrator -> matrix/evidence consumer.
    try:
        from perception.maskrcnn_adapter import register_maskrcnn_engine
        register_maskrcnn_engine()
        registered.append("maskrcnn")
    except Exception as exc:
        log.warning("Mask R-CNN engine not registered: %s", exc)

    log.info("registered %d engines: %s", len(registered), registered)
    return registered


__all__ = [
    "register_all_engines",
    "_sift_run",
    "_lightglue_run",
]
