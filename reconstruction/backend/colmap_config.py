"""COLMAP configuration (P0.2 refinement).

Pinning the ad-hoc SIFT/MVS parameters that previously lived as module-level
constants in `colmap_backend.py` into a single, auditable configuration object.
Loads from `reconstruction.toml` (or `pyproject.toml` `[tool.reality]` table)
if present, falls back to measured defaults, never fabricates values.

Honesty rules:
- Every default carries its provenance comment (measured on the committed
  real 32-photo south-building subset, COLMAP 4.2.0).
- Values are descriptive, not prescriptive: operators override by editing the
  TOML or passing `ColmapReconstructionBackend(...)` kwargs explicitly.
- Nothing here is ever silently changed by version bumps; the config is the
  contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover - fallback for older interpreters
    import tomli as tomllib  # type: ignore[import-not-found,unused-ignore]


@dataclass(frozen=True)
class ColmapConfig:
    """Measured defaults for COLMAP SfM/MVS parameters.

    Measured on the committed real 32-photo south-building subset
    (COLMAP 4.2.0, GT-compared through reconstruction.evaluation):

      baseline SIFT            22/32 cams, 4438 pts, rot err 0.150 deg
      affine+DSP               22/32 cams, 5392 pts, rot err 0.164 deg
      affine+DSP + guided      31/32 cams, 11632 pts, rot err 24.6 deg

    Guided matching's 9 extra cameras are a FALSE COVERAGE GAIN: it forces
    matches through the facade's repetitive windows, warping the model
    (per-camera errors up to 160 deg, both sub-models affected). The default
    is therefore affine+DSP WITHOUT guided matching.
    """

    robust_sift: bool = True
    """Affine-shape + domain-size-pooling SIFT: +22% verified points at
    unchanged fidelity, ~3x extraction time (CPU-only in COLMAP)."""

    guided_matching: bool = False
    """Opt-in only -- measured FALSE COVERAGE on repetitive architecture."""

    dense_mvs: bool = False
    """Heavyweight GPU stage; opt-in explicit. PatchMatch MVS -> stereo_fusion
    continuation runs in the same workspace."""

    use_gpu: bool = False
    """The official Windows release is a no-GPU build; pass True only for
    CUDA builds."""

    max_workspace_path_chars: int = 170
    """Native COLMAP on Windows crashes silently past ~260 chars including
    file names; this is the safe working-directory ceiling."""

    @classmethod
    def from_file(cls, path: Optional[str] = None) -> "ColmapConfig":
        """Load from `reconstruction.toml` (cwd) or the given path, then
        `pyproject.toml` `[tool.reality]`. Missing file -> measured defaults."""
        import os

        search_paths = [path] if path else []
        search_paths += ["reconstruction.toml", "pyproject.toml"]

        for candidate in search_paths:
            if not candidate or not os.path.isfile(candidate):
                continue
            try:
                with open(candidate, "rb") as f:
                    data = tomllib.load(f)
            except (OSError, tomllib.TOMLDecodeError):
                continue  # unreadable or malformed -> try next, never crash

            # Prefer [tool.reality] (pyproject.toml convention), fall back to
            # top-level [colmap] (standalone reconstruction.toml).
            section = data.get("tool", {}).get("reality", data.get("colmap", {}))
            if not isinstance(section, dict):
                continue
            return cls(**{
                k: v for k, v in section.items()
                if k in cls.__dataclass_fields__
            })

        return cls()  # measured defaults, nothing to load

    def to_dict(self) -> dict:
        return {
            "robust_sift": self.robust_sift,
            "guided_matching": self.guided_matching,
            "dense_mvs": self.dense_mvs,
            "use_gpu": self.use_gpu,
            "max_workspace_path_chars": self.max_workspace_path_chars,
        }


__all__ = ["ColmapConfig"]
