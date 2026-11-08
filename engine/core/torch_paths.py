"""Neutral utilities for torch cache and checkpoint directory resolution."""

from __future__ import annotations

import os
from pathlib import Path


def torch_hub_dir() -> Path:
    """Return the base torch.hub directory."""
    home = os.environ.get("TORCH_HOME")
    return (Path(home) if home else Path.home() / ".cache" / "torch") / "hub"


def torch_hub_checkpoints() -> Path:
    """Return the torch.hub checkpoints directory."""
    return torch_hub_dir() / "checkpoints"
