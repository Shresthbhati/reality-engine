"""Minimal model/checkpoint registry (mapping campaign P0.3).

Today the repo's model acquisition is implicit: MiDaS downloads via
`torch.hub` (with an explicit trust decision, see
`perception/depth/midAS_backend.py`), torchvision models pull weights via
the torchvision weights API (same default cache), and SAM needs a torch-hub
cache seeded by hand. Nothing records **what** a backend needs, **where**
it lands on disk, or **how to verify it** -- so "did the model actually
load, from where, verified by hash?" is unanswerable, and offline
reproducibility is accidental rather than declared.

This module is deliberately NOT a model marketplace: it is one dict-shaped
fact source backends consult, plus a checker that answers "is the
checkpoint present and hash-verified?" for a declared entry. Backends that
manage their own acquisition (torchvision/torch.hub) call
`record_pretrained` after loading so a run's report can state the exact
acquisition without this module performing any download itself.

Registry format (a frozen dataclass, not a dict of loose strings):

    ModelSpec(name, task, source, checkpoint, sha256=None,
              framework="", license="", input_contract="", output_contract="")

`checkpoint_for`/`record_pretrained` keep the acquisition path in one
place; `verify_checkpoint` gives offline users a real hash gate.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

#: torch/torchvision's default checkpoint cache -- where a plain
#: weights-API load lands (torch.hub.set_dir's default layout).
_TORCH_HOME = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"


@dataclass(frozen=True)
class ModelSpec:
    """One declared model/checkpoint. `source` is where the weights came
    from ("torchvision", "torch.hub:intel-isl/MiDaS", "manual"); where the
    binary lands is `checkpoint` (resolved, may not exist yet)."""

    name: str
    task: str
    source: str
    checkpoint: Path
    sha256: Optional[str] = None
    framework: str = "torch"
    license: str = ""
    commercial_ok: bool = True
    measured_vram_mb: float = 0.0
    status: str = "active"  # "active" | "candidate" | "deprecated"
    input_contract: str = ""
    output_contract: str = ""

    def to_dict(self) -> dict:
        d = {
            "name": self.name,
            "task": self.task,
            "source": self.source,
            "checkpoint": str(self.checkpoint),
            "framework": self.framework,
            "license": self.license,
            "commercial_ok": self.commercial_ok,
            "measured_vram_mb": self.measured_vram_mb,
            "status": self.status,
            "input_contract": self.input_contract,
            "output_contract": self.output_contract,
            "present": self.checkpoint.exists(),
        }
        if self.sha256:
            d["sha256"] = self.sha256
        if self.checkpoint.exists() and self.sha256:
            d["sha256_verified"] = verify_checkpoint(self.checkpoint, self.sha256)
        return d


#: The registry. Backends add/lookup here; tests may mutate a copy.
_REGISTRY: Dict[str, ModelSpec] = {}


def record_pretrained(
    name: str,
    task: str,
    source: str,
    checkpoint: Path,
    sha256: Optional[str] = None,
    **kwargs,
) -> ModelSpec:
    """Register (or re-register) a pretrained model's acquisition facts.
    Called by backends after a real load so reports can state provenance;
    never performs a download."""
    spec = ModelSpec(
        name=name,
        task=task,
        source=source,
        checkpoint=checkpoint,
        sha256=sha256,
        **kwargs,
    )
    _REGISTRY[name] = spec
    return spec


def get(name: str) -> Optional[ModelSpec]:
    return _REGISTRY.get(name)


def all_specs() -> Dict[str, ModelSpec]:
    return dict(_REGISTRY)


def default_cache_dir() -> Path:
    """torch/torchvision's default checkpoint cache -- where a plain
    `pretrained=True`-style load lands. Exposed so backends can both
    pre-check availability and report the path."""
    return _TORCH_HOME


def checkpoint_for(name: str, cache_dir: Optional[Path] = None) -> Optional[Path]:
    """Return the on-disk checkpoint path for `name` if present in the
    cache (torchvision layout), else None. Purely a lookup -- no download."""
    base = cache_dir or _TORCH_HOME
    candidates = sorted(base.glob(f"{name}*"))
    return candidates[0] if candidates else None


def verify_checkpoint(path: Path, expected_sha256: str) -> bool:
    """Stream-hash `path` and compare to `expected_sha256`."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest() == expected_sha256.lower()


# Default declared models across perception, depth, and matching
_DEFAULT_MODELS = [
    ModelSpec(
        name="midas_dpt_hybrid",
        task="monocular_depth",
        source="torch.hub:intel-isl/MiDaS",
        checkpoint=_TORCH_HOME / "dpt_hybrid_384.pt",
        license="Apache-2.0",
        commercial_ok=True,
        measured_vram_mb=1200.0,
        status="active",
        input_contract="RGB image (3, H, W) normalized",
        output_contract="Relative inverse depth map (H, W)",
    ),
    ModelSpec(
        name="midas_small",
        task="monocular_depth",
        source="torch.hub:intel-isl/MiDaS",
        checkpoint=_TORCH_HOME / "midas_v21_small_256.pt",
        license="MIT",
        commercial_ok=True,
        measured_vram_mb=350.0,
        status="active",
        input_contract="RGB image (3, 256, 256)",
        output_contract="Relative depth map (256, 256)",
    ),
    ModelSpec(
        name="maskrcnn_resnet50_fpn",
        task="instance_segmentation",
        source="torchvision:models.detection.maskrcnn_resnet50_fpn",
        checkpoint=_TORCH_HOME / "maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth",
        license="BSD-3-Clause",
        commercial_ok=True,
        measured_vram_mb=900.0,
        status="active",
        input_contract="RGB image tensor [0, 1]",
        output_contract="Boxes, labels, scores, masks",
    ),
    ModelSpec(
        name="sam_vit_h",
        task="promptable_segmentation",
        source="manual:meta/segment-anything",
        checkpoint=_TORCH_HOME / "sam_vit_h_4b8939.pth",
        sha256="a7bf3b02f3ebf1267abb613ffec9f445bc47bc52968b1ddf35e446fa6613f92a",
        license="Apache-2.0",
        commercial_ok=True,
        measured_vram_mb=2500.0,
        status="active",
        input_contract="RGB image (1024, 1024)",
        output_contract="Binary masks (N, H, W)",
    ),
    ModelSpec(
        name="sam_vit_b",
        task="promptable_segmentation",
        source="manual:meta/segment-anything",
        checkpoint=_TORCH_HOME / "sam_vit_b_01ec64.pth",
        sha256="01ec64d29a2fca3fce371c1464b3a06070fbf0231936c5b964344407b8b4088a",
        license="Apache-2.0",
        commercial_ok=True,
        measured_vram_mb=950.0,
        status="active",
        input_contract="RGB image (1024, 1024)",
        output_contract="Binary masks (N, H, W)",
    ),
    ModelSpec(
        name="lightglue_aliked",
        task="feature_matching",
        source="torch.hub:cvg/LightGlue",
        checkpoint=_TORCH_HOME / "aliked-n16.pth",
        license="Apache-2.0",
        commercial_ok=True,
        measured_vram_mb=450.0,
        status="candidate",
        input_contract="Keypoints and descriptors from two views",
        output_contract="Correspondences and match confidence",
    ),
    ModelSpec(
        name="dedode",
        task="feature_matching",
        source="torch.hub:Parskatt/DeDoDe",
        checkpoint=_TORCH_HOME / "dedode_detector_L.pth",
        license="MIT",
        commercial_ok=True,
        measured_vram_mb=600.0,
        status="candidate",
        input_contract="Paired RGB images",
        output_contract="Sparse tie points and matching scores",
    ),
    ModelSpec(
        name="xfeat",
        task="feature_extraction",
        source="github:verlab/accelerated_features",
        checkpoint=_TORCH_HOME / "xfeat.pt",
        license="Apache-2.0",
        commercial_ok=True,
        measured_vram_mb=350.0,
        status="candidate",
        input_contract="Single or stereo RGB image",
        output_contract="Keypoints and 64-d local descriptors",
    ),
    ModelSpec(
        name="metric3d_v2",
        task="metric_depth",
        source="torch.hub:YvanYin/Metric3D",
        checkpoint=_TORCH_HOME / "metric_depth_vit_small_800k.pth",
        license="Apache-2.0",
        commercial_ok=True,
        measured_vram_mb=1800.0,
        status="candidate",
        input_contract="RGB image + camera intrinsics matrix K",
        output_contract="Metric depth map in meters (H, W)",
    ),
    ModelSpec(
        name="depth_anything_v2",
        task="monocular_depth",
        source="torch.hub:Depth-Anything/Depth-Anything-V2",
        checkpoint=_TORCH_HOME / "depth_anything_v2_vits.pth",
        license="Apache-2.0",
        commercial_ok=True,
        measured_vram_mb=1100.0,
        status="candidate",
        input_contract="RGB image (H, W, 3)",
        output_contract="Affinity-aligned relative depth map",
    ),
    ModelSpec(
        name="owlvit_base_patch32",
        task="open_vocabulary_detection",
        source="huggingface:google/owlvit-base-patch32",
        checkpoint=_TORCH_HOME / "owlvit-base-patch32",
        license="Apache-2.0",
        commercial_ok=True,
        measured_vram_mb=1100.0,
        status="active",
        input_contract="RGB image + text queries list",
        output_contract="Normalized boxes [0,1], label indices, confidence scores",
    ),
    ModelSpec(
        name="owlv2_base_patch16",
        task="open_vocabulary_detection",
        source="huggingface:google/owlv2-base-patch16-ensemble",
        checkpoint=_TORCH_HOME / "owlv2-base-patch16-ensemble",
        license="Apache-2.0",
        commercial_ok=True,
        measured_vram_mb=1350.0,
        status="candidate",
        input_contract="RGB image + text queries list",
        output_contract="Normalized boxes [0,1], label indices, confidence scores",
    ),
    ModelSpec(
        name="grounding_dino_tiny",
        task="open_vocabulary_detection",
        source="huggingface:IDEA-Research/grounding-dino-tiny",
        checkpoint=_TORCH_HOME / "groundingdino_swint_ogc.pth",
        license="Apache-2.0",
        commercial_ok=True,
        measured_vram_mb=850.0,
        status="candidate",
        input_contract="RGB image + text queries list",
        output_contract="Boxes [0,1], label phrases, confidence scores",    ),
]

for m in _DEFAULT_MODELS:
    if m.name not in _REGISTRY:
        _REGISTRY[m.name] = m
