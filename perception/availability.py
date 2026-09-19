"""Unified real-model availability report (perception/availability.py).

Mission requirement served: "Real models must not silently fail.
Optional dependencies should use explicit skip reason / blocked state /
availability report."

Each real backend ALREADY fails honestly at construction time
(DepthBackendUnavailableError, SegmentationBackendUnavailableError,
DetectionBackendUnavailableError, the COLMAP availability probe). What
was missing is the operator-level view: WHICH real models can this
machine run right now, and for each one that cannot run, WHY -- before
constructing anything, downloading anything, or importing heavy modules.

Design rules:

  - PROBE-ONLY: dependency checks use importlib.util.find_spec (no
    module execution), checkpoints are resolved as FILES in the known
    cache locations, binaries via shutil.which. No torch import, no
    model construction, no network. Running the report twice in a row
    must be cheap and identical (tests enforce determinism).
  - EXPLICIT STATUSES: "available" only when dependency + checkpoint
    (or binary) both verifiably resolve; otherwise "blocked" with the
    specific missing piece named and an actionable remediation. Never
    a guess, never an empty result.
  - The spec table (REAL_MODEL_SPECS) records the adapter class behind
    each model -- the audit trail for "which backend implements which
    integration path".
"""

from __future__ import annotations

import importlib.util
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

#: Real-model integration paths: model -> adapter that consumes it.
#: Kept declarative so the report needs no imports from the backends.
REAL_MODEL_SPECS: Dict[str, Dict[str, str]] = {
    "colmap": {
        "adapter": "reconstruction.backend.colmap_backend.COLMAPBackend",
        "dependency": None,  # external binary, not a python package
        "binary": "colmap",
        "task": "SfM + MVS reconstruction",
    },
    "midas": {
        "adapter": "perception.depth.midAS_backend.MiDaSDepthBackend",
        "dependency": "torch",
        "binary": None,
        "task": "monocular relative depth",
    },
    "sam": {
        "adapter": "perception.segmentation.sam_backend.SAMSegmentationBackend",
        "dependency": "torch",
        "binary": None,
        "task": "promptable/automatic segmentation",
    },
    "mask_rcnn": {
        "adapter": "perception.detection.maskrcnn_backend.MaskRCNNDetector",
        "dependency": "torchvision",
        "binary": None,
        "task": "object detection + instance segmentation",
    },
}

#: Official SAM checkpoint filenames (as cached by torch.hub), keyed by
#: the backend's model_type. vit_b is the smallest practical default.
_SAM_CHECKPOINTS = (
    "sam_vit_b_01ec64.pth",
    "sam_vit_l_0b3195.pth",
    "sam_vit_h_4b8939.pth",
)


@dataclass(frozen=True)
class ModelAvailability:
    """One real model's verifiable availability on this machine."""

    name: str
    status: str  # "available" | "blocked"
    reason: str
    remediation: str = ""
    adapter: str = ""
    task: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status,
            "reason": self.reason,
            "remediation": self.remediation,
            "adapter": self.adapter,
            "task": self.task,
        }


def _dependency_ok(module_name: str) -> bool:
    """True when the module is importable -- WITHOUT importing it
    (find_spec only; importing torch is neither cheap nor side-effect
    free, and the report must be a probe)."""
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        return False


def _torch_hub_checkpoints_dir() -> Optional[Path]:
    """The torch.hub checkpoint dir, resolved without importing torch:
    $TORCH_HOME/hub/checkpoints, defaulting to ~/.cache/torch (the
    documented torch.hub default when TORCH_HOME is unset)."""
    import os

    torch_home = os.environ.get("TORCH_HOME")
    if torch_home:
        return Path(torch_home) / "hub" / "checkpoints"
    return Path.home() / ".cache" / "torch" / "hub" / "checkpoints"


def _first_existing(names: List[str]) -> Optional[Path]:
    cache = _torch_hub_checkpoints_dir()
    if cache is None:
        return None
    for name in names:
        candidate = cache / name
        if candidate.is_file():
            return candidate
    return None


def _report_midas() -> ModelAvailability:
    spec = REAL_MODEL_SPECS["midas"]
    if not _dependency_ok("torch"):
        return ModelAvailability(
            name="midas", status="blocked", adapter=spec["adapter"],
            task=spec["task"],
            reason="dependency missing: torch is not installed in this "
                   "environment; MiDaSDepthBackend would raise "
                   "DepthBackendUnavailableError at construction",
            remediation="pip install torch torchvision (CPU wheels suffice "
                        "for MiDaS_small)",
        )
    ckpt = _first_existing(["midas_small", "dpt_hybrid_384.pt",
                            "dpt_large_384.pt", "dpt_hybrid-midas-501f0c75.pt"])
    if ckpt is None:
        return ModelAvailability(
            name="midas", status="blocked", adapter=spec["adapter"],
            task=spec["task"],
            reason="dependency present (torch), but no MiDaS checkpoint is "
                   "cached under the torch.hub checkpoint dir; first use "
                   "would download it (requires network)",
            remediation="pre-download: torch.hub.load('intel-isl/MiDaS', "
                        "'DPT_Hybrid', pretrained=True) -- or pass "
                        "checkpoint_path= to the backend",
        )
    return ModelAvailability(
        name="midas", status="available", adapter=spec["adapter"],
        task=spec["task"],
        reason=f"torch importable and checkpoint cached ({ckpt.name})",
    )


def _report_sam() -> ModelAvailability:
    spec = REAL_MODEL_SPECS["sam"]
    if not _dependency_ok("torch"):
        return ModelAvailability(
            name="sam", status="blocked", adapter=spec["adapter"],
            task=spec["task"],
            reason="dependency missing: torch is not installed; "
                   "SAMSegmentationBackend would raise "
                   "SegmentationBackendUnavailableError",
            remediation="pip install torch",
        )
    if not _dependency_ok("segment_anything"):
        return ModelAvailability(
            name="sam", status="blocked", adapter=spec["adapter"],
            task=spec["task"],
            reason="dependency missing: the segment_anything package is not "
                   "installed",
            remediation="pip install git+https://github.com/facebookresearch/"
                        "segment-anything.git",
        )
    ckpt = _first_existing(list(_SAM_CHECKPOINTS))
    if ckpt is None:
        return ModelAvailability(
            name="sam", status="blocked", adapter=spec["adapter"],
            task=spec["task"],
            reason="dependencies present, but no SAM checkpoint (vit_b/vit_l/"
                   "vit_h) is cached under the torch.hub checkpoint dir",
            remediation="pre-download e.g. sam_vit_b_01ec64.pth from "
                        "dl.fbaipublicfiles.com/segment_anything into the "
                        "torch.hub checkpoints dir, or pass checkpoint_path=",
        )
    return ModelAvailability(
        name="sam", status="available", adapter=spec["adapter"],
        task=spec["task"],
        reason=f"torch + segment_anything importable, checkpoint cached "
               f"({ckpt.name})",
    )


def _report_mask_rcnn() -> ModelAvailability:
    spec = REAL_MODEL_SPECS["mask_rcnn"]
    if not _dependency_ok("torchvision"):
        if not _dependency_ok("torch"):
            return ModelAvailability(
                name="mask_rcnn", status="blocked", adapter=spec["adapter"],
                task=spec["task"],
                reason="dependencies missing: torch and torchvision are not "
                       "installed",
                remediation="pip install torch torchvision",
            )
        return ModelAvailability(
            name="mask_rcnn", status="blocked", adapter=spec["adapter"],
            task=spec["task"],
            reason="dependency missing: torchvision is not installed (torch "
                   "is)",
            remediation="pip install torchvision",
        )
    ckpt = _first_existing(["maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth"])
    if ckpt is None:
        return ModelAvailability(
            name="mask_rcnn", status="blocked", adapter=spec["adapter"],
            task=spec["task"],
            reason="dependencies present, but the COCO_V1 weights "
                   "(maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth) are not "
                   "cached; construction would attempt a download",
            remediation="pre-download via torchvision (the URL is "
                        "torchvision's), or set weights_download_root",
        )
    return ModelAvailability(
        name="mask_rcnn", status="available", adapter=spec["adapter"],
        task=spec["task"],
        reason=f"torchvision importable, COCO weights cached ({ckpt.name})",
    )


def _report_colmap() -> ModelAvailability:
    spec = REAL_MODEL_SPECS["colmap"]
    path = shutil.which("colmap")
    if path is None:
        return ModelAvailability(
            name="colmap", status="blocked", adapter=spec["adapter"],
            task=spec["task"],
            reason="binary not found on PATH; COLMAPBackend's availability "
                   "probe would report unavailable",
            remediation="install COLMAP (CUDA build for GPU MVS) and ensure "
                        "'colmap' is on PATH",
        )
    return ModelAvailability(
        name="colmap", status="available", adapter=spec["adapter"],
        task=spec["task"], reason=f"binary found on PATH ({path})",
    )


_REPORTERS = {
    "colmap": _report_colmap,
    "midas": _report_midas,
    "sam": _report_sam,
    "mask_rcnn": _report_mask_rcnn,
}


def report_real_models() -> List[ModelAvailability]:
    """Probe every real-model integration path and report verifiable
    availability, with explicit reasons and remediation for anything
    blocked. Side-effect free: no imports of the model stacks, no model
    construction, no downloads, no network. Deterministic and cheap by
    construction."""
    return [_REPORTERS[name]() for name in sorted(_REPORTERS)]


def real_models_by_name() -> Dict[str, ModelAvailability]:
    """`report_real_models()` keyed by model name, for direct lookup."""
    return {m.name: m for m in report_real_models()}
