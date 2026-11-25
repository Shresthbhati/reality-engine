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

    use_gpu: Optional[bool] = None
    """None (default) = auto: GPU only when this COLMAP build reports CUDA AND a device with enough free VRAM is visible
    (``detect_colmap_gpu``); True/False force it. ``RE_COLMAP_GPU=0/1`` overrides everything."""

    max_workspace_path_chars: int = 170
    """Native COLMAP on Windows crashes silently past ~260 chars including
    file names; this is the safe working-directory ceiling."""

    deterministic: Optional[bool] = None
    """Fixed seed, sorted filename order, single-threaded matching for identical runs.
    Default None auto-enables deterministic mode for <= 60 images (BUILD_SPEC B12)."""

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


_GPU_CACHE: dict = {}
MIN_FREE_VRAM_MB = 1536


def _free_vram_mb() -> Optional[int]:
    import subprocess

    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        vals = [int(x) for x in out.split() if x.strip().isdigit()]
        if vals:
            return max(vals)
    except Exception:  # noqa: BLE001 -- fall through to torch
        pass
    try:
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.mem_get_info()[0]) // (1024 * 1024)
    except Exception:  # noqa: BLE001
        pass
    return None


def detect_colmap_gpu(binary: str = "colmap", *, refresh: bool = False) -> dict:
    """Decide whether COLMAP's SIFT extraction/matching/dense should use the GPU. Returns
    ``{"use_gpu": bool, "reason": str, "source": "env"|"auto", "free_vram_mb": int|None}``; a CPU decision always carries its reason
    (CPU-only build, no device, VRAM below ``MIN_FREE_VRAM_MB``, or ``RE_COLMAP_GPU=0``) so the caller records the fallback."""
    import os
    import subprocess

    env = os.environ.get("RE_COLMAP_GPU", "").strip().lower()
    if env in ("0", "1", "true", "false"):
        on = env in ("1", "true")
        return {"use_gpu": on, "reason": f"RE_COLMAP_GPU={env} (operator override)", "source": "env", "free_vram_mb": None}
    if binary in _GPU_CACHE and not refresh:
        return dict(_GPU_CACHE[binary])
    res = {"use_gpu": False, "source": "auto", "free_vram_mb": None, "reason": ""}
    try:
        banner = subprocess.run([binary, "-h"], capture_output=True, text=True, timeout=30)
        text = (banner.stdout or "") + (banner.stderr or "")
        first = text.strip().splitlines()[0] if text.strip() else ""
    except Exception as exc:  # noqa: BLE001
        first = ""
        res["reason"] = f"could not read COLMAP banner ({type(exc).__name__})"
    if not res["reason"]:
        if "with cuda" not in first.lower():
            res["reason"] = f"COLMAP build is not CUDA ({first[:80]!r}): CPU SIFT/matching"
        else:
            free = _free_vram_mb()
            res["free_vram_mb"] = free
            if free is None:
                res["reason"] = "CUDA COLMAP but no NVIDIA device visible (nvidia-smi/torch): CPU"
            elif free < MIN_FREE_VRAM_MB:
                res["reason"] = f"only {free} MB VRAM free (< {MIN_FREE_VRAM_MB}): CPU"
            else:
                res.update(use_gpu=True, reason=f"CUDA COLMAP build and {free} MB VRAM free")
    _GPU_CACHE[binary] = dict(res)
    return res


__all__ = ["ColmapConfig", "detect_colmap_gpu"]
