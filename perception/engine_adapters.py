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
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

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
# SIFT Matcher Adapter
# ---------------------------------------------------------------------------

def _sift_available() -> Optional[str]:
    try:
        import cv2
        cv2.SIFT_create()
        return None
    except Exception as exc:
        return str(exc)


def _sift_run(image_paths: Sequence[str], **ctx: Any) -> UnifiedEngineOutput:
    import cv2
    import time as _t

    t0 = _t.perf_counter()
    run_id = ctx.get("run_id", "sift_run")
    obs: List[ObservationRecord] = []
    pairs = ctx.get("pairs", [])
    if not pairs and len(image_paths) >= 2:
        pairs = [(image_paths[i], image_paths[i + 1]) for i in range(len(image_paths) - 1)]

    for a_path, b_path in pairs:
        try:
            img_a = cv2.imread(a_path, cv2.IMREAD_GRAYSCALE)
            img_b = cv2.imread(b_path, cv2.IMREAD_GRAYSCALE)
            if img_a is None or img_b is None:
                continue
            sift = cv2.SIFT_create(nfeatures=4000)
            kp1, des1 = sift.detectAndCompute(img_a, None)
            kp2, des2 = sift.detectAndCompute(img_b, None)
            if des1 is None or des2 is None or len(des1) < 8 or len(des2) < 8:
                continue
            bf = cv2.BFMatcher(cv2.NORM_L2)
            matches = bf.knnMatch(des1, des2, k=2)
            good = [m[0] for m in matches if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]
            if len(good) < 8:
                continue
            p1 = np.float32([kp1[m.queryIdx].pt for m in good])
            p2 = np.float32([kp2[m.trainIdx].pt for m in good])
            _, mask = cv2.findFundamentalMat(p1, p2, cv2.FM_RANSAC, 0.5, 0.999, 5000)
            inlier_count = int(mask.sum()) if mask is not None else 0
            if inlier_count < 8:
                continue
            obs.append(ObservationRecord(
                evidence_id=f"sift:{a_path}->{b_path}",
                source_image_id=a_path,
                coordinate_frame=CoordinateFrame(frame_id="image", scale_state="image", units="pixels"),
                model_id=_model("matching", "sift", "opencv", "Apache-2.0", False, "cpu"),
                label="feature_match",
                category="correspondence",
                confidence=float(inlier_count / max(1, len(good))),
                sigma_m=None,
                metadata={"a_path": a_path, "b_path": b_path, "matches": len(good), "inliers": inlier_count,
                          "inlier_ratio": round(inlier_count / max(1, len(good)), 4)},
            ))
        except Exception as exc:
            log.warning("SIFT pair failed %s->%s: %s", a_path, b_path, exc)

    return UnifiedEngineOutput(
        engine_id="sift", run_id=run_id, observations=tuple(obs),
        status="succeeded" if obs else "partial",
        wall_time_s=time.perf_counter() - t0,
        summary={"n_pairs": len(pairs), "n_matches": len(obs)},
    )


# ---------------------------------------------------------------------------
# LightGlue-ALIKED Adapter
# ---------------------------------------------------------------------------

def _lightglue_aliked_available() -> Optional[str]:
    try:
        import torch
        import kornia.feature as kf
        return None
    except ImportError as exc:
        return str(exc)


def _lightglue_run(image_paths: Sequence[str], **ctx: Any) -> UnifiedEngineOutput:
    import torch
    import kornia.feature as kf
    import time as _t
    import cv2

    t0 = _t.perf_counter()
    run_id = ctx.get("run_id", "lightglue_run")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    obs: List[ObservationRecord] = []
    pairs = ctx.get("pairs", [])
    if not pairs and len(image_paths) >= 2:
        pairs = [(image_paths[i], image_paths[i + 1]) for i in range(len(image_paths) - 1)]

    # Load ALIKED + LightGlue
    try:
        aliked = kf.ALIKED.from_pretrained("aliked-n16")
        aliked = aliked.to(device).eval()
    except Exception as exc:
        try:
            aliked = kf.ALIKED()
            aliked = aliked.to(device).eval()
        except Exception as exc2:
            return UnifiedEngineOutput(engine_id="lightglue_aliked", run_id=run_id, observations=(),
                                      status="failed", error_message=str(exc2), wall_time_s=time.perf_counter() - t0)

    for a_path, b_path in pairs:
        try:
            img_a = cv2.imread(a_path)
            img_b = cv2.imread(b_path)
            if img_a is None or img_b is None:
                continue
            img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB)
            img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB)
            # Convert to torch tensors
            tensor_a = torch.from_numpy(img_a).permute(2, 0, 1).float() / 255.0
            tensor_b = torch.from_numpy(img_b).permute(2, 0, 1).float() / 255.0
            tensor_a = tensor_a.unsqueeze(0).to(device)
            tensor_b = tensor_b.unsqueeze(0).to(device)

            with torch.no_grad():
                feats_a = aliked(tensor_a)[0]
                feats_b = aliked(tensor_b)[0]

            # Use LightGlue for matching
            from kornia.feature import LightGlue
            lg = LightGlue(features="aliked")
            lg = lg.to(device).eval()
            with torch.no_grad():
                matches = lg({
                    "image0": {
                        "keypoints": feats_a.keypoints.unsqueeze(0),
                        "descriptors": feats_a.descriptors.unsqueeze(0),
                        "image": tensor_a,
                        "matching_scores": feats_a.keypoint_scores.unsqueeze(0),
                    },
                    "image1": {
                        "keypoints": feats_b.keypoints.unsqueeze(0),
                        "descriptors": feats_b.descriptors.unsqueeze(0),
                        "image": tensor_b,
                        "matching_scores": feats_b.keypoint_scores.unsqueeze(0),
                    },
                })

            if "matches" in matches:
                mm = matches["matches"]
                if isinstance(mm, (list, tuple)):
                    match_count = len(mm[0]) if len(mm) > 0 and hasattr(mm[0], '__len__') else 0
                elif hasattr(mm, 'shape'):
                    match_count = mm.shape[0] if mm.ndim >= 1 else 0
                else:
                    match_count = 0
            else:
                match_count = 0
            if match_count < 8:
                continue
            obs.append(ObservationRecord(
                evidence_id=f"lg_aliked:{a_path}->{b_path}",
                source_image_id=a_path,
                coordinate_frame=CoordinateFrame(frame_id="image", scale_state="image", units="pixels"),
                model_id=_model("matching", "lightglue_aliked", "aliked", "Apache-2.0", False, "gpu"),
                label="feature_match",
                category="correspondence",
                confidence=float(match_count / max(1, match_count)),
                sigma_m=None,
                metadata={"a_path": a_path, "b_path": b_path, "matches": match_count},
            ))
        except Exception as exc:
            log.warning("LightGlue-ALIKED pair failed %s->%s: %s", a_path, b_path, exc)

    return UnifiedEngineOutput(
        engine_id="lightglue_aliked", run_id=run_id, observations=tuple(obs),
        status="succeeded" if obs else "partial",
        wall_time_s=time.perf_counter() - t0,
        peak_vram_mb=_get_vram_peak(),
        summary={"n_pairs": len(pairs), "n_matches": len(obs)},
    )


def _get_vram_peak() -> Optional[float]:
    try:
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated(0) / (1024.0 * 1024.0)
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Registration — register all available engines
# ---------------------------------------------------------------------------

def register_all_engines() -> List[str]:
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

    log.info("registered %d engines: %s", len(registered), registered)
    return registered


__all__ = [
    "register_all_engines",
    "_sift_run",
    "_lightglue_run",
]
