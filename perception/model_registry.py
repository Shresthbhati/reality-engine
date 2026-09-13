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
