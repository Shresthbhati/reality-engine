"""MiDaS (Intel ISL, DPT_Hybrid) depth engine adapter for the multi-engine
orchestrator.

Wraps ``perception.depth.midAS_backend.MiDaSDepthBackend`` (a real, implemented
backend) into the ``UnifiedEngineOutput`` contract so the orchestrator runs it as
a first-class depth engine.

Honest rules (matching the repo's depth_adapter for Depth Anything):
* Unavailable is unavailable: no torch/torchvision/torch.hub -> the engine is
  reported unavailable with the reason; nothing pretends inference happened.
* Monocular depth has no calibrated confidence -- the observation records
  ``confidence=None`` (like depth_anything_v2) rather than claiming certainty.
* The depth is RELATIVE inverse depth (MiDaS convention), normalised per image;
  the ``metadata`` records ``quantity="inverse_depth"`` and ``unit="relative"``
  so a consumer must not read it as metric. Two views of the same surface are
  not comparable from this output -- that limitation is recorded, not hidden.
* The model is loaded once for the whole run and released before the next engine
  starts, so its VRAM is not shared on the 6 GB card.
* Per-image exceptions are logged and skipped -- no fabricated output.

This adapter exists because ``perception/depth/midAS_backend.py`` is an
implemented backend that was NOT wired into the orchestrator. Connecting it here
is the §6 Part B delivery for a valuable second depth engine whose observations
feed the orchestrator -> matrix/evidence consumer (and, downstream, the MVS /
dense path when that path consumes depth maps).
"""

from __future__ import annotations

import logging
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

log = logging.getLogger("reality.midas_adapter")


def _midas_available() -> str | None:
    """Why MiDaS cannot run here, or None when it can."""
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        return f"torch unavailable: {exc}"
    try:
        from torchvision.models.segmentation import fcn_resnet50  # noqa: F401 -- just a torch probe
    except ImportError as exc:
        return f"torchvision unavailable: {exc}"
    # torch.hub access is probed by the backend at load time; here we just confirm
    # the hub dir is reachable.
    try:
        import torch.hub
        torch.hub.get_dir()
    except Exception as exc:
        return f"torch.hub unavailable: {exc}"
    return None


def _midas_run(image_paths: Sequence[str], **ctx: Any) -> UnifiedEngineOutput:
    """Run MiDaS DPT_Hybrid on every image, producing depth observations."""
    # Respect the availability probe: if it returns a reason, do not attempt to load
    # the model (the orchestrator already checked this, but the direct-call path --
    # e.g. a test that patches _midas_available -- relies on the run function
    # honouring the probe so "unavailable" is never fabricated).
    _reason = _midas_available()
    if _reason is not None:
        _t0 = time.perf_counter()
        return UnifiedEngineOutput(
            engine_id="midas", run_id=ctx.get("run_id", "midas_run"), observations=(),
            status="failed", error_message=f"unavailable: {_reason}",
            wall_time_s=_t0 - _t0,  # effectively zero -- we never attempted a load
        )

    from perception.depth.midAS_backend import MiDaSDepthBackend

    t0 = time.perf_counter()
    run_id = ctx.get("run_id", "midas_run")
    obs: list[ObservationRecord] = []
    skipped = 0
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"

    try:
        backend = MiDaSDepthBackend(model_type="DPT_Hybrid", device=device)
    except Exception as exc:
        return UnifiedEngineOutput(
            engine_id="midas", run_id=run_id, observations=(),
            status="failed", error_message=f"backend load failed: {exc}",
            wall_time_s=time.perf_counter() - t0,
        )

    for img_path in image_paths:
        pair_t0 = time.perf_counter()
        try:
            from engine.core.image_io import open_oriented
            from urllib.parse import urlparse

            parsed = urlparse(img_path)
            if parsed.scheme == "file":
                raw_path = parsed.path
                nl = parsed.netloc
                if nl and len(nl) >= 2 and nl[1] == ":" and (len(nl) == 2 or nl[2] in "/\\"):
                    raw_path = nl + raw_path
                src_path = __import__("pathlib").Path(__import__("urllib").request.url2pathname(raw_path))
            else:
                src_path = __import__("pathlib").Path(img_path)
            if not src_path.is_file():
                skipped += 1
                continue
            image = open_oriented(src_path).convert("RGB")
        except Exception as exc:
            log.warning("MiDaS failed to open %s: %s", img_path, exc)
            skipped += 1
            continue

        try:
            from evidence.session import EvidenceItem, EvidenceKind

            item = EvidenceItem(
                id=f"midas:{img_path}",
                source_uri=img_path,
                kind=EvidenceKind.PHOTO,
                captured_at=time.time(),
            )
            depth_maps = backend.estimate_depth([item])
        except Exception as exc:
            log.warning("MiDaS inference failed for %s: %s", img_path, exc)
            skipped += 1
            continue

        if not depth_maps:
            skipped += 1
            continue

        d = depth_maps[0]
        values = d.values  # list-of-lists (row-major)
        arr = np.asarray(values, dtype=np.float32)
        d_min, d_max = float(arr.min()), float(arr.max())
        obs.append(ObservationRecord(
            evidence_id=f"midas:{img_path}",
            source_image_id=img_path,
            coordinate_frame=CoordinateFrame(
                frame_id="image", scale_state="relative", units="relative"),
            model_id=ModelIdentity(
                engine_id="depth", architecture_id="midas_dpt_hybrid",
                checkpoint_id="dpt_hybrid",
                license="MiDaS (Intel ISL): MIT",
                hardware_class=device),
            label="depth_map",
            category="depth",
            confidence=None,  # monocular depth has no calibrated confidence
            sigma_m=None,
            depth_map_ref=img_path,
            calibration_quality="unverified",
            metadata={
                "model": "midas_dpt_hybrid",
                "model_version": "MiDaS v3.1 DPT_Hybrid",
                "shape": [int(d.height), int(d.width)],
                "depth_min": d_min,
                "depth_max": d_max,
                "quantity": "inverse_depth",
                "unit": "relative",
                "normalised_per_image": True,
                "note": "relative inverse depth, normalised per image -- not metric and not "
                        "cross-view comparable; a consumer must not read this as metres",
            },
        ))

    try:
        backend.unload()
        if __import__("torch").cuda.is_available():
            __import__("torch").cuda.empty_cache()
    except Exception:
        pass

    return UnifiedEngineOutput(
        engine_id="midas", run_id=run_id, observations=tuple(obs),
        status="succeeded" if obs else "partial",
        wall_time_s=time.perf_counter() - t0,
        peak_vram_mb=None,
        summary={
            "n_images": len(image_paths),
            "n_depth_maps": len(obs),
            "skipped": skipped,
            "model": "midas_dpt_hybrid",
            "quantity": "inverse_depth",
            "unit": "relative",
        },
    )


def register_midas_engine() -> None:
    """Register MiDaS as a first-class depth engine."""
    register_engine(
        EngineSpec(
            engine_id="midas",
            architecture_id="midas_dpt_hybrid",
            checkpoint_id="dpt_hybrid",
            version="MiDaS v3.1 DPT_Hybrid",
            vram_mb=2000,
            cpu_threads=0,
            priority=45,
            required_for_production=False,
            license="MiDaS (Intel ISL): MIT",
            non_commercial=False,
            dependencies=(),
            run_condition=_midas_available,
            cpu_capable=True,
            timeout_s=0.0,
            drain_after_timeout_s=60.0,
            max_attempts=2,
        ),
        _midas_run,
    )
