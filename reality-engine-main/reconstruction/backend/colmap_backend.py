"""Real COLMAP-backed IReconstructionBackend (decision: docs/RECONSTRUCTION_BACKEND_DECISION.md).

Shells out to the `colmap` CLI (feature_extractor -> exhaustive_matcher ->
mapper) and parses its COLMAP-text-format sparse output
(cameras.txt/images.txt/points3D.txt) into ReconstructionResult.

Parsing is split into pure functions (_parse_images_txt, _parse_points3d_txt)
so they're unit-testable against COLMAP's documented output format without
the binary installed. reconstruct() itself needs colmap on PATH; it raises
ReconstructionBackendUnavailableError rather than silently no-op'ing when
it isn't -- this repo doesn't have COLMAP installed, so that path is real,
not hypothetical, and must fail loudly, not fake a result.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

from provenance import Uncertainty

from evidence.session import EvidenceItem, EvidenceKind
from .interface import (
    IReconstructionBackend,
    ReconstructedCameraPose,
    ReconstructedPoint,
    ReconstructionResult,
)


class ReconstructionBackendUnavailableError(RuntimeError):
    """Raised when the `colmap` binary isn't on PATH -- never silently skipped."""
    pass


def _confidence_from_reprojection_error(error_px: float) -> float:
    """Heuristic: lower reprojection error -> higher confidence, bounded (0, 1].

    ponytail: linear falloff, not a calibrated model -- replace with a
    validated error->confidence curve if reconstruction accuracy work
    needs one; this is enough to avoid reporting every point at the same
    confidence regardless of how well it triangulated.
    """
    return max(0.01, 1.0 / (1.0 + error_px))


def _parse_images_txt(text: str, evidence_id_by_name: Dict[str, str]) -> List[ReconstructedCameraPose]:
    """Parse COLMAP's images.txt. Each image occupies two lines; only the
    first (pose) line is needed here -- the second (2D point) line is
    skipped since points3D.txt already carries the reconstructed points.
    """
    poses = []
    lines = [l for l in text.splitlines() if l.strip() and not l.startswith("#")]
    for i in range(0, len(lines), 2):
        parts = lines[i].split()
        qw, qx, qy, qz = (float(x) for x in parts[1:5])
        tx, ty, tz = (float(x) for x in parts[5:8])
        name = parts[9]
        evidence_id = evidence_id_by_name.get(name, name)
        poses.append(ReconstructedCameraPose(
            evidence_id=evidence_id,
            position=(tx, ty, tz),
            rotation=(qw, qx, qy, qz),
        ))
    return poses


def _parse_points3d_txt(text: str, image_id_to_evidence_id: Dict[str, str]) -> List[ReconstructedPoint]:
    """Parse COLMAP's points3D.txt into ReconstructedPoint list."""
    points = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        point_id, x, y, z = parts[0], float(parts[1]), float(parts[2]), float(parts[3])
        error = float(parts[7])
        track = parts[8:]  # (IMAGE_ID, POINT2D_IDX) pairs
        source_evidence_ids = sorted({
            image_id_to_evidence_id.get(track[i], track[i]) for i in range(0, len(track), 2)
        })
        points.append(ReconstructedPoint(
            position=(x, y, z),
            track_id=point_id,
            source_evidence_ids=source_evidence_ids,
            uncertainty=Uncertainty(confidence=_confidence_from_reprojection_error(error)),
        ))
    return points


def _subprocess_env(colmap_path: str) -> Dict[str, str]:
    """COLMAP's Windows CLI builds still construct a QApplication (for GUI
    code paths compiled into the same binary) and abort with a Qt platform
    plugin dialog unless QT_QPA_PLATFORM is forced headless -- verified by
    running the real binary, not a hypothetical. Points QT_PLUGIN_PATH at
    the plugins/ directory the official Windows release ships as a sibling
    of bin/; a no-op on platforms where COLMAP doesn't need it (Qt ignores
    an unused plugin path).
    """
    import os
    env = dict(os.environ)
    env["QT_QPA_PLATFORM"] = "offscreen"
    plugins_dir = Path(colmap_path).resolve().parent.parent / "plugins"
    if plugins_dir.is_dir():
        env["QT_PLUGIN_PATH"] = str(plugins_dir)
    return env


class ColmapReconstructionBackend(IReconstructionBackend):
    def __init__(self, colmap_binary: str = "colmap", use_gpu: bool = False):
        self._colmap_binary = colmap_binary
        # Default False: the official Windows release used here is a
        # no-GPU build ("without GPU support" per its own version banner);
        # passing use_gpu=1 against it fails every step. A CUDA build
        # should pass use_gpu=True explicitly.
        self._use_gpu = use_gpu

    # ---- orchestrator integration (reconstruction/orchestrator.py) ----
    # availability_probe: can COLMAP run in this environment at all?
    # Zero-arg -> (ok, detail); lets the orchestrator decline this backend
    # BEFORE execution when the binary is missing, instead of discovering
    # it via a caught ReconstructionBackendUnavailableError. Same check the
    # reconstruct() path performs, factored so both use one truth.
    def _colmap_available(self):
        colmap_path = shutil.which(self._colmap_binary)
        if colmap_path is None:
            return False, (
                f"'{self._colmap_binary}' not found on PATH -- COLMAP must be "
                "installed to run real reconstruction "
                "(see docs/RECONSTRUCTION_BACKEND_DECISION.md)"
            )
        return True, f"colmap binary found: {colmap_path}"

    availability_probe = _colmap_available

    # accepts: COLMAP's mapper is iterative SfM over imagery; refuse below
    # the two-view floor so the orchestrator records a DECLINE (try the
    # next backend) rather than a failed registration run.
    def _accepts_evidence(self, evidence: List[EvidenceItem]):
        image_evidence = [
            e for e in evidence
            if e.kind in (EvidenceKind.PHOTO, EvidenceKind.VIDEO)
        ]
        if len(image_evidence) < 2:
            return False, (
                f"COLMAP needs >= 2 PHOTO/VIDEO items, got {len(image_evidence)}"
            )
        return True, f"{len(image_evidence)} image item(s) accepted"

    accepts = _accepts_evidence

    def reconstruct(self, evidence: List[EvidenceItem]) -> ReconstructionResult:
        image_evidence = [e for e in evidence if e.kind in (EvidenceKind.PHOTO, EvidenceKind.VIDEO)]
        if len(image_evidence) < 2:
            return ReconstructionResult(points=[], camera_poses=[], registration_status="failed")

        colmap_path = shutil.which(self._colmap_binary)
        if colmap_path is None:
            raise ReconstructionBackendUnavailableError(
                f"'{self._colmap_binary}' not found on PATH -- COLMAP must be installed "
                "to run real reconstruction (see docs/RECONSTRUCTION_BACKEND_DECISION.md)"
            )
        env = _subprocess_env(colmap_path)
        gpu_flag = "1" if self._use_gpu else "0"

        with tempfile.TemporaryDirectory(prefix="colmap_") as workspace:
            workspace = Path(workspace)
            image_dir = workspace / "images"
            image_dir.mkdir()
            evidence_id_by_name = {}
            for item in image_evidence:
                src = Path(item.source_uri.replace("file://", "", 1))
                dest_name = f"{item.id}{src.suffix or '.jpg'}"
                shutil.copy(src, image_dir / dest_name)
                evidence_id_by_name[dest_name] = item.id

            db_path = workspace / "database.db"
            subprocess.run(
                [self._colmap_binary, "feature_extractor",
                 "--database_path", str(db_path), "--image_path", str(image_dir),
                 "--FeatureExtraction.use_gpu", gpu_flag],
                check=True, capture_output=True, env=env,
            )
            subprocess.run(
                [self._colmap_binary, "exhaustive_matcher", "--database_path", str(db_path),
                 "--FeatureMatching.use_gpu", gpu_flag],
                check=True, capture_output=True, env=env,
            )
            sparse_dir = workspace / "sparse"
            sparse_dir.mkdir()
            subprocess.run(
                [self._colmap_binary, "mapper",
                 "--database_path", str(db_path), "--image_path", str(image_dir),
                 "--output_path", str(sparse_dir)],
                check=True, capture_output=True, env=env,
            )

            model_dir = sparse_dir / "0"
            subprocess.run(
                [self._colmap_binary, "model_converter",
                 "--input_path", str(model_dir), "--output_path", str(model_dir),
                 "--output_type", "TXT"],
                check=True, capture_output=True, env=env,
            )

            images_txt = (model_dir / "images.txt").read_text()
            points3d_txt = (model_dir / "points3D.txt").read_text()

            poses = _parse_images_txt(images_txt, evidence_id_by_name)
            image_id_to_evidence_id = {}
            for line in images_txt.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 10:
                    continue
                image_id_to_evidence_id[parts[0]] = evidence_id_by_name.get(parts[9], parts[9])
            points = _parse_points3d_txt(points3d_txt, image_id_to_evidence_id)

            status = "success" if len(poses) == len(image_evidence) else "partial"
            if not points:
                status = "failed"
            return ReconstructionResult(points=points, camera_poses=poses, registration_status=status)
