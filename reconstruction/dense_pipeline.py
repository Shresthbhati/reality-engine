"""Dense MVS pipeline (P6-01 integration): run COLMAP's dense stereo
chain as a REALITY ENGINE stage instead of a manual CLI ritual.

The capability registry records a real GPU dense run executed by hand
(2026-09-15: image_undistorter -> patch_match_stereo geom_consistency
-> stereo_fusion, ~294k fused points). This module turns that sequence
into deterministic, diagnosable code:

    sparse model (COLMAP mapper output) + undistorted images
      -> image_undistorter   (rectified images + workspace)
      -> patch_match_stereo  (per-view depth/normal maps)
      -> stereo_fusion       (fused.ply, geometric-consistency filtered)
      -> fused.ply path      (consumed by reconstruction/dense_ingest.py
                              for canonical WorldIR write-back)

Honesty rules (same discipline as colmap_backend / meshing.surface):
- availability is PROBED (the help output must list patch_match_stereo);
  a binary on PATH proves nothing about this build's capabilities;
- every stage is timed and its failure is a named error with the
  failing step and stderr excerpt -- never a silent partial run;
- a "successful" run whose fused.ply does not exist is an ERROR (COLMAP
  exiting 0 without output is not a dense reconstruction);
- subprocesses are always list-argv with shell=False and COLMAP's
  verified headless environment (QT_QPA_PLATFORM=offscreen).
"""

from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from reconstruction.backend.colmap_backend import _subprocess_env


class DenseMVSUnavailableError(RuntimeError):
    """The configured binary cannot perform dense MVS (missing binary or
    no patch_match_stereo command) -- the caller should skip with a
    note, not fabricate a fallback."""


class DenseMVSRunError(RuntimeError):
    """A dense-MVS stage ran and failed (non-zero exit, timeout, or a
    completed chain with no fused output)."""


@dataclass(frozen=True)
class DenseMVSRun:
    """One dense-MVS execution's observed facts. `n_fused_points` is
    read from the fused.ply header (the file's own declared count)."""
    fused_ply_path: str
    n_fused_points: int
    stage_durations_s: Dict[str, float]
    geom_consistency: bool
    use_gpu: Optional[bool]
    #: True only when the installed binary actually accepted a GPU flag
    #: (newer COLMAP builds removed --PatchMatchStereo.use_gpu and pick
    #: GPU themselves). False + use_gpu set = the request could not be
    #: applied and is recorded as such, never silently ignored.
    gpu_flag_applied: bool = False

    def to_dict(self) -> dict:
        return {
            "fused_ply_path": self.fused_ply_path,
            "n_fused_points": self.n_fused_points,
            "stage_durations_s": {
                k: round(v, 3) for k, v in self.stage_durations_s.items()
            },
            "geom_consistency": self.geom_consistency,
            "use_gpu": self.use_gpu,
            "gpu_flag_applied": self.gpu_flag_applied,
        }


def _find_colmap(binary: str) -> str:
    resolved = shutil.which(binary)
    if resolved is None:
        raise DenseMVSUnavailableError(
            f"COLMAP binary {binary!r} not found on PATH -- dense MVS "
            "unavailable (set colmap_binary or install COLMAP)"
        )
    return resolved


def _help_text(colmap_path: str, step: str) -> str:
    """The step's own --help output (its real, installed option names).
    Empty on any failure -- callers treat that as 'unknown' and skip
    option probing."""
    try:
        proc = subprocess.run(
            [colmap_path, step, "--help"],
            capture_output=True, text=True, timeout=30, shell=False,
            env=_subprocess_env(colmap_path),
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return (proc.stdout or "") + (proc.stderr or "")


def dense_mvs_available(binary: str = "colmap") -> bool:
    """True only if `colmap help` actually lists patch_match_stereo.
    Presence of the binary alone proves nothing about the build."""
    try:
        colmap_path = _find_colmap(binary)
    except DenseMVSUnavailableError:
        return False
    try:
        proc = subprocess.run(
            [colmap_path, "help"],
            capture_output=True, text=True, timeout=30, shell=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return "patch_match_stereo" in (proc.stdout or "")


def _count_fused_points(fused_ply: Path) -> int:
    """Vertex count from the PLY header only (the file's own declared
    element count; the full parse happens at ingestion). CRLF-safe:
    COLMAP's Windows output writes CRLF headers, which already bit the
    fused.ply parser once (see dense_output._parse_header)."""
    header = fused_ply.read_bytes()[:4096]
    text = header.decode("ascii", errors="replace").replace("\r\n", "\n")
    for line in text.split("\n"):
        parts = line.strip().split()
        if len(parts) == 3 and parts[0] == "element" and parts[1] == "vertex":
            return int(parts[2])
    raise DenseMVSRunError(
        f"fused.ply header has no 'element vertex' line: {fused_ply}"
    )


def run_dense_mvs(
    *,
    image_dir: Path,
    sparse_model_dir: Path,
    workspace: Path,
    binary: str = "colmap",
    use_gpu: Optional[bool] = None,
    geom_consistency: bool = True,
    timeout_s: float = 3600.0,
) -> DenseMVSRun:
    """Run the COLMAP dense stereo chain and return the fused output's
    facts. `use_gpu=None` leaves the choice to COLMAP (the verified real
    run used a CUDA build with GPU left on); an explicit True/False pins
    the flag (the CPU-only Windows build needs False)."""
    colmap_path = _find_colmap(binary)
    image_dir = Path(image_dir)
    sparse_model_dir = Path(sparse_model_dir)
    workspace = Path(workspace)
    if not image_dir.is_dir():
        raise DenseMVSRunError(f"image_dir does not exist: {image_dir}")
    if not sparse_model_dir.is_dir():
        raise DenseMVSRunError(
            f"sparse_model_dir does not exist: {sparse_model_dir} "
            "(run sparse reconstruction / mapper first)"
        )
    workspace.mkdir(parents=True, exist_ok=True)
    dense_dir = workspace / "dense"
    dense_dir.mkdir(exist_ok=True)

    env = _subprocess_env(colmap_path)

    def run_step(step: str, args: list) -> None:
        try:
            proc = subprocess.run(
                [colmap_path, step, *args],
                capture_output=True, env=env, text=True,
                timeout=timeout_s, shell=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise DenseMVSRunError(
                f"COLMAP {step} timed out after {timeout_s}s"
            ) from exc
        if proc.returncode != 0:
            stderr = (proc.stderr or "")[-800:]
            raise DenseMVSRunError(
                f"COLMAP {step} failed (exit {proc.returncode}): {stderr}"
            )

    stage_durations: Dict[str, float] = {}

    # COLMAP's dense CLI has drifted across versions and this module's
    # flags were written blind (measured 2026-09-21: the installed 4.2
    # build rejects --PatchMatchStereo.use_gpu -- GPU selection moved
    # out of the option manager -- and stereo_fusion lost its
    # --StereoFusion. prefix entirely). Each stage's --help is now
    # PROBED and only options the installed binary actually accepts
    # are passed; what was probed is recorded, never guessed.
    pm_help = _help_text(colmap_path, "patch_match_stereo")
    fusion_help = _help_text(colmap_path, "stereo_fusion")
    if not pm_help.strip():
        raise DenseMVSRunError(
            "could not read patch_match_stereo --help; refusing to "
            "guess the installed COLMAP's dense option names"
        )
    if not fusion_help.strip():
        raise DenseMVSRunError(
            "could not read stereo_fusion --help; refusing to guess "
            "the installed COLMAP's dense option names"
        )

    def _accepts(help_text: str, option: str) -> bool:
        # '--PatchMatchStereo.use_gpu arg' / '--input_type arg' etc.
        return option in help_text

    # Stage 1: image_undistorter -- rectified images + dense workspace.
    start = time.perf_counter()
    run_step("image_undistorter", [
        "--image_path", str(image_dir),
        "--input_path", str(sparse_model_dir),
        "--output_path", str(dense_dir),
    ])
    stage_durations["image_undistorter"] = time.perf_counter() - start

    # Stage 2: patch_match_stereo -- per-view depth/normal maps.
    start = time.perf_counter()
    pm_args = [
        "--workspace_path", str(dense_dir),
        "--PatchMatchStereo.geom_consistency",
        "true" if geom_consistency else "false",
    ]
    gpu_flag_applied = False
    if use_gpu is not None and _accepts(pm_help, "--PatchMatchStereo.use_gpu"):
        pm_args += ["--PatchMatchStereo.use_gpu", "1" if use_gpu else "0"]
        gpu_flag_applied = True
    run_step("patch_match_stereo", pm_args)
    stage_durations["patch_match_stereo"] = time.perf_counter() - start

    # Stage 3: stereo_fusion -- geometric-consistency-fused point cloud.
    # Old CLI: --StereoFusion.input_type/--StereoFusion.output;
    # 4.x CLI: bare --input_type/--output_path (probed, not assumed).
    fused_ply = dense_dir / "fused.ply"
    start = time.perf_counter()
    fusion_args = ["--workspace_path", str(dense_dir)]
    if _accepts(fusion_help, "--StereoFusion.input_type"):
        fusion_args += [
            "--StereoFusion.input_type",
            "geometric" if geom_consistency else "photometric",
        ]
    elif _accepts(fusion_help, "--input_type"):
        fusion_args += [
            "--input_type",
            "geometric" if geom_consistency else "photometric",
        ]
    if _accepts(fusion_help, "--StereoFusion.output"):
        fusion_args += ["--StereoFusion.output", str(fused_ply)]
    elif _accepts(fusion_help, "--output_path"):
        fusion_args += ["--output_path", str(fused_ply)]
    else:
        raise DenseMVSRunError(
            "stereo_fusion accepts neither --StereoFusion.output nor "
            "--output_path; cannot determine where fused output would "
            "be written"
        )
    run_step("stereo_fusion", fusion_args)
    stage_durations["stereo_fusion"] = time.perf_counter() - start

    if not fused_ply.is_file():
        raise DenseMVSRunError(
            "COLMAP dense chain exited 0 but produced no fused.ply at "
            f"{fused_ply} -- a completed chain without output is not a "
            "reconstruction"
        )
    return DenseMVSRun(
        fused_ply_path=str(fused_ply),
        n_fused_points=_count_fused_points(fused_ply),
        stage_durations_s=stage_durations,
        geom_consistency=geom_consistency,
        use_gpu=use_gpu,
        gpu_flag_applied=gpu_flag_applied,
    )

