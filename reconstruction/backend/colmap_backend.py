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

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

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


class ReconstructionStepError(RuntimeError):
    """A COLMAP pipeline step failed; carries the step's real stderr/stdout.

    `check=True` with discarded output turned every mapper/extractor failure
    into an opaque `Returned non-zero exit status 1`. Callers (and humans)
    need the actual diagnostic -- e.g. the OpenGL-context crash of a GPU
    build or the `output_path is not a directory` contract of the mapper.
    """

    def __init__(self, step: str, proc: "subprocess.CompletedProcess[str]",
                 command: List[str]):
        tail = (proc.stderr or "").strip().splitlines()[-15:]
        detail = "\n".join(tail) if tail else (proc.stdout or "").strip()[-800:]
        super().__init__(
            f"colmap {step} failed (exit {proc.returncode})\n"
            f"command: {' '.join(command)}\n"
            f"stderr tail:\n{detail}"
        )
        self.step = step
        self.stderr = proc.stderr or ""
        self.stdout = proc.stdout or ""


def _uri_to_path(uri: str) -> Path:
    """file:// URI -> local Path, correct on every platform.

    naive `uri.replace("file://", "")` yields '/C:/...' on Windows (an
    invalid rootless path); urllib.parse.urlparse + unquote does it right
    and also decodes percent-escaped characters (spaces etc.).
    """
    from urllib.parse import unquote, urlparse
    parsed = urlparse(uri)
    if parsed.scheme in ("file", ""):
        path = unquote(parsed.path)
        # Recover malformed f"file://{windows_path}" forms (no third
        # slash): the drive prefix (or the whole backslash path) lands in
        # netloc instead of path.
        nl = parsed.netloc
        if (parsed.scheme == "file" and nl and len(nl) >= 2
                and nl[1] == ":" and (len(nl) == 2 or nl[2] in "/\\")):
            path = nl + path
        # Windows: 'file:///C:/x' parses to path '/C:/x'; Path('/C:/x') is a
        # rootless '\C:\x' that open() rejects -- strip the leading slash.
        if len(path) >= 3 and path[0] == "/" and path[1].isalpha() and path[2] == ":":
            path = path[1:]
        return Path(path)
    return Path(uri)


def _trusted_intrinsics(evidence: List[EvidenceItem]):
    """Per-image trusted intrinsics from evidence metadata, or None.

    Recognizes metadata["intrinsics"] = {fx, fy, cx, cy} (pixels). Returns
    the shared (fx, fy, cx, cy) only when EVERY image carries the values
    and they agree to 1e-6; any missing entry or disagreement -> None, so
    partial trust never silently mixes calibrated and uncalibrated images.
    """
    values = []
    for item in evidence:
        intr = (item.metadata or {}).get("intrinsics")
        if not isinstance(intr, dict):
            return None
        try:
            values.append((float(intr["fx"]), float(intr["fy"]),
                           float(intr["cx"]), float(intr["cy"])))
        except (KeyError, TypeError, ValueError):
            return None
    if not values:
        return None
    first = values[0]
    if any(any(abs(a - b) > 1e-6 for a, b in zip(first, v)) for v in values[1:]):
        return None
    return first


def _qvec_to_rotmat(q: Tuple[float, float, float, float]) -> np.ndarray:
    """COLMAP quaternion (w, x, y, z) -> 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


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
        # COLMAP stores R, t as world->camera: x_cam = R @ x_world + t. The
        # camera CENTER in world frame is C = -R^T @ t -- NOT t itself. The
        # original code stored t, i.e. the world origin expressed in camera
        # coordinates; every downstream consumer (world compiler, Studio
        # viewport, scale anchoring) received wrong positions. Verified
        # against ground truth on the synthetic room fixture: with C, camera
        # centers match GT to <1 mm; with t they are off by ~8 m in model
        # units (~1.7 m metric).
        R = _qvec_to_rotmat((qw, qx, qy, qz))
        cx, cy, cz = -(R.T @ np.array([tx, ty, tz]))
        # Convention contract (documented on ReconstructedCameraPose and
        # enforced by consumers): position is the camera center in WORLD
        # frame and rotation is CAMERA-TO-WORLD. COLMAP's q is world->
        # camera, so the stored quaternion is its conjugate. Storing the
        # raw qvec alongside a world-frame center (the previous state)
        # mixed conventions: any consumer rotating camera-frame data with
        # it got geometry tilted by the camera's own pitch (~40 deg here),
        # which measured as ~20 deg plane tilts and zero room walls.
        inv_q = (qw, -qx, -qy, -qz)
        poses.append(ReconstructedCameraPose(
            evidence_id=evidence_id,
            position=(cx, cy, cz),
            rotation=inv_q,
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


#: Robust SIFT defaults (see ColmapReconstructionBackend.robust_sift).
#: Measured on the committed real 32-photo south-building subset
#: (COLMAP 4.2.0, GT-compared through reconstruction.evaluation):
#
#:   baseline SIFT            22/32 cams, 4438 pts, rot err 0.150 deg
#:   affine+DSP               22/32 cams, 5392 pts, rot err 0.164 deg
#:   affine+DSP + guided      31/32 cams, 11632 pts, rot err 24.6 deg
#
#: Guided matching's 9 extra cameras are a FALSE COVERAGE GAIN: it
#: forces matches through the facade's repetitive windows, warping the
#: model (per-camera errors up to 160 deg, both sub-models affected).
#: The default is therefore affine+DSP WITHOUT guided matching: more
#: points at true fidelity; cameras missing because matching cannot
#: verify them honestly stay missing (recorded, not forced).
DEFAULT_ROBUST_SIFT = True
DEFAULT_GUIDED_MATCHING = False


def _sift_extraction_args(robust: bool = DEFAULT_ROBUST_SIFT) -> List[str]:
    """SiftExtraction flags. Affine-shape estimation and domain-size
    pooling make SIFT robust to the viewpoint/scale variation that
    defeats baseline descriptors on repetitive architecture (measured:
    +22% verified points at unchanged fidelity)."""
    if not robust:
        return []
    return [
        "--SiftExtraction.estimate_affine_shape", "1",
        "--SiftExtraction.domain_size_pooling", "1",
    ]


def _sift_matching_args(guided: bool = DEFAULT_GUIDED_MATCHING) -> List[str]:
    """FeatureMatching flags. Guided matching is OPT-IN ONLY: measured
    on real repetitive-architecture data it registers more cameras but
    corrupts the model (see DEFAULT_GUIDED_MATCHING comment). Callers
    who opt in accept that tradeoff explicitly."""
    if not guided:
        return []
    return ["--FeatureMatching.guided_matching", "1"]


def _collect_submodels(sparse_dir: Path, convert):
    """Sorted numeric sub-model directories under the mapper's output.
    `convert` (the model_converter wrapper) is applied to each when
    provided. Empty list means the mapper produced nothing, which the
    caller reports honestly."""
    if not sparse_dir.is_dir():
        return []
    dirs = sorted(
        (d for d in sparse_dir.iterdir() if d.is_dir() and d.name.isdigit()),
        key=lambda d: int(d.name),
    )
    if convert is not None:
        for d in dirs:
            convert(d)
    return dirs


def _convert_model_to_txt(run_fn, model_dir: Path) -> None:
    """Convert one binary sub-model to TXT in place via model_converter."""
    run_fn("model_converter",
           ["--input_path", str(model_dir), "--output_path", str(model_dir),
            "--output_type", "TXT"])


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
    def __init__(self, colmap_binary: str = "colmap", use_gpu: bool = False,
                 robust_sift: bool = True, guided_matching: bool = False,
                 dense_mvs: bool = False, artifact_store=None, session=None):
        self._colmap_binary = colmap_binary
        #: Optional reconstruction.colmap_session.ColmapSession. When set, the workspace
        #: PERSISTS between runs and new photographs are registered into the previous sparse
        #: model with `image_registrator` (see _reconstruct_in_session); when None the backend
        #: behaves exactly as before (temporary workspace, full reconstruction every time).
        self.session = session
        #: what the last reconstruct() actually did (mode, counts, timings); {} for the
        #: sessionless path. Surfaced in the version report -- never inferred.
        self.last_run_info: Dict[str, object] = {}
        #: Optional engine.pipeline.candidate_selection.WorldArbiter (assess(result, tag) / choose(a, b)).
        #: When set, incremental-vs-full is decided by how each candidate evolves the WORLD, not by how
        #: many photographs it registered. None keeps the registration-count rule.
        self.candidate_arbiter = None
        # Default False: the official Windows release used here is a
        # no-GPU build ("without GPU support" per its own version banner);
        # passing use_gpu=1 against it fails every step. A CUDA build
        # should pass use_gpu=True explicitly.
        self._use_gpu = use_gpu
        #: Affine-shape + domain-size-pooling SIFT. Measured default ON:
        #: +22% verified points at unchanged fidelity on the committed
        #: real 32-photo subset. Costs ~3x extraction time (CPU-only in
        #: COLMAP); opt out explicitly.
        self.robust_sift = robust_sift
        #: Guided matching. Default OFF -- measured FALSE COVERAGE on
        #: repetitive architecture (31/32 cameras but 24.6 deg median
        #: rotation error vs 0.164 deg without). Opt in only when the
        #: capture lacks repetitive structures.
        self.guided_matching = guided_matching
        #: Dense-MVS continuation. Default OFF (heavyweight, GPU-bound):
        #: when True, reconstruct() continues the mapper's sparse model
        #: through image_undistorter -> patch_match_stereo ->
        #: stereo_fusion IN THE SAME WORKSPACE and attaches the parsed
        #: fused cloud to the result (dense_points/dense_report). The
        #: binary must PROBE as dense-capable before any compute is
        #: spent; a dense failure raises rather than degrading into a
        #: sparse-only success.
        self.dense_mvs = dense_mvs
        #: Optional ArtifactStore. When set AND dense_mvs=True, the
        #: fused cloud is ingested through the canonical chain
        #: (parse -> PointCloudData -> content-addressed artifact)
        #: INSIDE the workspace lifetime. This is not decoration: the
        #: backend deletes its workspace on return, so a fused.ply left
        #: on disk there is an ORPHANED artifact by construction
        #: (measured 2026-09-21 -- the first real dense run's fused.ply
        #: died with the temp dir). Without a store the parsed cloud
        #: still travels on dense_points, and the report says no
        #: artifact was persisted.
        self.artifact_store = artifact_store

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

        # Dense continuation is an explicit opt-in, and its capability
        # precondition is checked BEFORE any compute is spent: a binary
        # that cannot run patch_match_stereo must be refused with the
        # dense-specific reason, not after minutes of sparse SfM.
        if self.dense_mvs:
            from reconstruction.dense_pipeline import dense_mvs_available
            if not dense_mvs_available(self._colmap_binary):
                from reconstruction.dense_pipeline import (
                    DenseMVSUnavailableError,
                )
                raise DenseMVSUnavailableError(
                    f"COLMAP '{self._colmap_binary}' does not list "
                    "patch_match_stereo in its help -- this build cannot "
                    "run dense MVS (dense_mvs=True was requested); "
                    "refusing to spend compute on a chain that cannot "
                    "complete"
                )

        if self.session is not None:
            return self._reconstruct_in_session(image_evidence, env, gpu_flag)

        with tempfile.TemporaryDirectory(prefix="colmap_") as workspace:
            workspace = Path(workspace)
            image_dir = workspace / "images"
            image_dir.mkdir()
            evidence_id_by_name = {}
            trusted_intrinsics = _trusted_intrinsics(image_evidence)
            extractor_args: List[str] = []
            if trusted_intrinsics is not None:
                fx, fy, cx, cy = trusted_intrinsics
                extractor_args = [
                    "--ImageReader.camera_model", "PINHOLE",
                    "--ImageReader.camera_params", f"{fx},{fy},{cx},{cy}",
                ]
            extractor_args += _sift_extraction_args(robust=self.robust_sift)
            for item in image_evidence:
                src = _uri_to_path(item.source_uri)
                dest_name = f"{item.id}{src.suffix or '.jpg'}"
                shutil.copy(src, image_dir / dest_name)
                evidence_id_by_name[dest_name] = item.id

            def _run(step: str, args: List[str]) -> None:
                self._run_step(step, args, env)

            db_path = workspace / "database.db"
            # Affine-shape SIFT is CPU-only in COLMAP: when robust SIFT
            # is on, extraction/matching run on CPU (the measured
            # configuration); GPU stays available for the baseline.
            extraction_gpu = "0" if self.robust_sift else gpu_flag
            matching_gpu = "0" if self.robust_sift else gpu_flag
            _run("feature_extractor",
                 ["--database_path", str(db_path), "--image_path", str(image_dir),
                  *extractor_args, "--FeatureExtraction.use_gpu", extraction_gpu])
            _run("exhaustive_matcher",
                 ["--database_path", str(db_path),
                  "--FeatureMatching.use_gpu", matching_gpu,
                  *_sift_matching_args(guided=self.guided_matching)])
            sparse_dir = workspace / "sparse"
            sparse_dir.mkdir()
            mapper_args = ["--database_path", str(db_path),
                           "--image_path", str(image_dir),
                           "--output_path", str(sparse_dir)]
            if trusted_intrinsics is not None:
                # Pinned calibration: BA must NOT refine intrinsics. Measured
                # on this repo's synthetic room fixture: with free focal
                # refinement on a planar-dominated scene, BA drifted fx to
                # ~1447 and fy to ~2039 (true: 1160) and produced a
                # self-consistent model whose camera rotations were ~16 deg
                # WRONG -- silent pose corruption that reprojection error
                # never revealed. Pinning keeps poses truthful.
                mapper_args += ["--Mapper.ba_refine_focal_length", "0",
                                "--Mapper.ba_refine_extra_params", "0"]
            _run("mapper", mapper_args)

            # The mapper may split the capture into MULTIPLE sub-models
            # (sparse/0, sparse/1, ...). Reading only `0` silently
            # dropped real registered cameras -- measured on the
            # committed 32-photo south-building subset: 22 cameras in
            # `0` and 13 more in `1` under robust SIFT. Every sub-model
            # is collected, parsed, and merged through the canonical
            # registration machinery (refusals reported, never
            # identity-placed).
            sparse_result, _model_dirs, _reference = self._parse_sparse_dir(
                sparse_dir, evidence_id_by_name, len(image_evidence), env)

            if not self.dense_mvs:
                return sparse_result
            return self._continue_dense(
                workspace=workspace, image_dir=image_dir,
                sparse_result=sparse_result,
                evidence_id_by_name=evidence_id_by_name,
            )

    # ------------------------------------------------------------------ shared steps
    def _run_step(self, step: str, args: List[str], env: Dict[str, str]) -> None:
        # Bounded, OWNED subprocess lifecycle: a hung COLMAP step
        # must fail explicitly instead of outliving its job, and a
        # cancelled/timed-out job must not leave the child behind.
        # The API-level job timeout is the outer bound; this
        # per-process bound also protects direct (CLI) users.
        # List-argv, no shell: ownership changes nothing about how
        # the command is constructed.
        from reconstruction.proc import run_owned

        try:
            timeout_s = float(os.environ.get("COLMAP_SUBPROCESS_TIMEOUT_SECONDS", "3600"))
        except ValueError:
            timeout_s = 3600.0
        try:
            proc = run_owned(
                [self._colmap_binary, step, *args],
                capture_output=True, env=env, text=True, timeout=timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            raise ReconstructionStepError(
                step,
                subprocess.CompletedProcess(
                    args=[self._colmap_binary, step, *args], returncode=-1,
                    stdout=exc.stdout or "",
                    stderr=(exc.stderr or "") + f"\n[TIMEOUT after {timeout_s:.0f}s]",
                ),
                [self._colmap_binary, step, *args],
            ) from exc
        if proc.returncode != 0:
            raise ReconstructionStepError(step, proc, [self._colmap_binary, step, *args])

    def _parse_model_dir(self, model_dir: Path, evidence_id_by_name: Dict[str, str],
                         n_input: int, env: Dict[str, str]) -> ReconstructionResult:
        """One COLMAP model directory -> ReconstructionResult (converts binary -> TXT in place)."""
        self._run_step("model_converter",
                       ["--input_path", str(model_dir), "--output_path", str(model_dir),
                        "--output_type", "TXT"], env)
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

        status = "success" if len(poses) == n_input else "partial"
        if not points:
            status = "failed"
        return ReconstructionResult(points=points, camera_poses=poses, registration_status=status)

    def _parse_sparse_dir(self, sparse_dir: Path, evidence_id_by_name: Dict[str, str],
                          n_input: int, env: Dict[str, str]):
        """The mapper's output directory -> (merged ReconstructionResult, model dirs, reference name).

        The mapper may split the capture into MULTIPLE sub-models (sparse/0, sparse/1, ...). Reading
        only `0` silently dropped real registered cameras -- measured on the committed 32-photo
        south-building subset: 22 cameras in `0` and 13 more in `1` under robust SIFT. Every
        sub-model is collected, parsed, and merged through the canonical registration machinery
        (refusals reported, never identity-placed)."""
        model_dirs = _collect_submodels(sparse_dir, None)
        if not model_dirs:
            proc = subprocess.CompletedProcess(
                args=[self._colmap_binary, "mapper"], returncode=0,
                stdout="", stderr="mapper produced no sub-model directories",
            )
            raise ReconstructionStepError("mapper", proc, [self._colmap_binary, "mapper"])
        model_results = {
            d.name: self._parse_model_dir(d, evidence_id_by_name, n_input, env) for d in model_dirs
        }
        # The most complete sub-model is the reference frame.
        reference = max(sorted(model_results), key=lambda n: len(model_results[n].points))
        if len(model_results) == 1:
            return next(iter(model_results.values())), model_dirs, reference
        from reconstruction.merge import merge_submodel_results

        merged, _merge_report = merge_submodel_results(model_results, reference=reference)
        return merged, model_dirs, reference

    # ------------------------------------------------------- incremental registration
    def _reconstruct_in_session(self, image_evidence: List[EvidenceItem], env: Dict[str, str],
                                gpu_flag: str) -> ReconstructionResult:
        """Reconstruct inside the persistent per-world COLMAP session.

        First run (or an unusable prior state): the normal full pipeline, kept on disk.
        Later runs: features are extracted for the NEW photographs only, they are matched
        against everything already in the database, and `colmap image_registrator` places
        every not-yet-registered photograph (new ones AND earlier ones that were waiting)
        into the PREVIOUS sparse model, followed by `bundle_adjuster`. Established camera
        poses are kept as the starting point instead of being re-solved from scratch.

        A full mapper run on the same database is only tried when photographs remain
        unregistered after the incremental step (it costs no re-extraction/matching), and it
        replaces the incremental model only if it registers strictly MORE photographs.
        Everything done is recorded in ``last_run_info`` -- the caller never has to guess
        which strategy produced a model.
        """
        import time

        t_start = time.time()
        timings: Dict[str, float] = {}

        def timed(label, fn):
            t0 = time.time()
            try:
                return fn()
            finally:
                timings[label] = round(time.time() - t0, 2)

        trusted = _trusted_intrinsics(image_evidence)
        signature = {"robust_sift": bool(self.robust_sift), "guided_matching": bool(self.guided_matching),
                     "trusted_intrinsics": list(trusted) if trusted is not None else None}
        by_name: Dict[str, Tuple[EvidenceItem, Path]] = {}
        for item in image_evidence:
            src = _uri_to_path(item.source_uri)
            by_name[f"{item.id}{src.suffix or '.jpg'}"] = (item, src)
        evidence_id_by_name = {n: it.id for n, (it, _) in by_name.items()}
        n_input = len(image_evidence)

        ws, prior = self.session.begin(signature, by_name)
        image_dir = ws / "images"
        image_dir.mkdir(exist_ok=True)
        # "new" = not in the committed manifest. Deciding this from which files happen to exist on
        # disk would silently skip an image copied by a run that crashed before extracting it.
        known = set(prior["images"]) if prior else set()
        new_names = [n for n in by_name if n not in known]
        for n in new_names:
            shutil.copy(by_name[n][1], image_dir / n)
        db_path = ws / "database.db"
        extractor_args: List[str] = []
        if trusted is not None:
            fx, fy, cx, cy = trusted
            extractor_args = ["--ImageReader.camera_model", "PINHOLE",
                              "--ImageReader.camera_params", f"{fx},{fy},{cx},{cy}"]
        extractor_args += _sift_extraction_args(robust=self.robust_sift)
        # Affine-shape SIFT is CPU-only in COLMAP (see reconstruct()).
        cpu_or_gpu = "0" if self.robust_sift else gpu_flag
        pinned = ["--Mapper.ba_refine_focal_length", "0", "--Mapper.ba_refine_extra_params", "0"] \
            if trusted is not None else []

        def run(step, args):
            self._run_step(step, args, env)

        def extract_and_match(only_new: bool):
            args = ["--database_path", str(db_path), "--image_path", str(image_dir), *extractor_args,
                    "--FeatureExtraction.use_gpu", cpu_or_gpu]
            if only_new:
                lst = ws / "new_images.txt"
                lst.write_text("\n".join(new_names) + "\n")
                args += ["--image_list_path", str(lst)]
            run("feature_extractor", args)
            run("exhaustive_matcher", ["--database_path", str(db_path), "--FeatureMatching.use_gpu", cpu_or_gpu,
                                       *_sift_matching_args(guided=self.guided_matching)])

        def full_mapper():
            out = ws / "sparse_full"
            shutil.rmtree(out, ignore_errors=True)
            out.mkdir()
            run("mapper", ["--database_path", str(db_path), "--image_path", str(image_dir),
                           "--output_path", str(out), *pinned])
            return self._parse_sparse_dir(out, evidence_id_by_name, n_input, env)

        info: Dict[str, object] = {
            "prior_images": len(prior["images"]) if prior else 0,
            "new_images": [evidence_id_by_name[n] for n in new_names],
            "prior_registered": len(prior["registered"]) if prior else 0,
        }
        chosen_dir: Path
        incremental_ok = True
        result: ReconstructionResult

        if prior is None:
            timed("extract_match", lambda: extract_and_match(only_new=False))
            result, dirs, ref = timed("mapper", full_mapper)
            chosen_dir = ws / "sparse_full" / ref
            incremental_ok = len(dirs) == 1        # a merged multi-model result cannot be extended faithfully
            info.update(mode="full", reason=("no usable prior COLMAP state" if self.session.committed_manifest() is None
                                             else "prior state unusable (settings changed, evidence removed, or not extendable)"))
        else:
            if new_names:
                timed("extract_match", lambda: extract_and_match(only_new=True))
            prior_model = ws / prior["model_dir"]
            pending = [n for n in by_name if n not in set(prior["registered"])]
            if not pending:
                # nothing new and nothing waiting: the committed model already covers every photograph
                result = self._parse_model_dir(prior_model, evidence_id_by_name, n_input, env)
                chosen_dir = prior_model
                info.update(mode="reused", reason="no new or waiting photographs")
            else:
                inc_result = None
                try:
                    inc_root = ws / "inc"
                    shutil.rmtree(inc_root, ignore_errors=True)
                    (inc_root / "registered").mkdir(parents=True)
                    (inc_root / "triangulated").mkdir()
                    (inc_root / "adjusted").mkdir()
                    timed("image_registrator", lambda: run("image_registrator", [
                        "--database_path", str(db_path), "--input_path", str(prior_model),
                        "--output_path", str(inc_root / "registered"), *pinned]))
                    # image_registrator places cameras but adds no structure: triangulate the new
                    # photographs' tracks while KEEPING the established points (--clear_points 0)
                    timed("point_triangulator", lambda: run("point_triangulator", [
                        "--database_path", str(db_path), "--image_path", str(image_dir),
                        "--input_path", str(inc_root / "registered"),
                        "--output_path", str(inc_root / "triangulated"), "--clear_points", "0"]))
                    timed("bundle_adjuster", lambda: run("bundle_adjuster", [
                        "--input_path", str(inc_root / "triangulated"), "--output_path", str(inc_root / "adjusted"),
                        *(["--BundleAdjustment.refine_focal_length", "0", "--BundleAdjustment.refine_extra_params", "0"]
                          if trusted is not None else [])]))
                    inc_result = self._parse_model_dir(inc_root / "adjusted", evidence_id_by_name, n_input, env)
                    if inc_result.registration_status == "failed":
                        info["incremental_error"] = "incremental model has no points"
                        inc_result = None
                except ReconstructionStepError as exc:
                    info["incremental_error"] = f"{exc.step}: {str(exc).splitlines()[0]}"
                    inc_result = None

                n_inc = len(inc_result.camera_poses) if inc_result is not None else None
                info["incremental_registered"] = n_inc
                info["incremental_points"] = len(inc_result.points) if inc_result is not None else None
                arbiter = self.candidate_arbiter
                inc_assess = None
                if arbiter is not None and inc_result is not None:
                    try:
                        inc_assess = arbiter.assess(inc_result, "incremental")
                    except Exception as exc:  # noqa: BLE001 -- fall back to the registration-count rule, recorded
                        info["arbiter_error"] = f"{type(exc).__name__}: {exc}"[:300]
                        arbiter = None
                # A full rebuild is worth its cost when photographs remain unplaced OR the incremental
                # candidate is not a clean evolution of the world (a drifted / lossy model can place every photo).
                need_full = (inc_result is None or n_inc < n_input
                             or (inc_assess is not None and inc_assess["verdict"] != "ACCEPT"))
                full = None
                if need_full:
                    # cheap: features and matches are already in the database
                    full = timed("mapper", full_mapper)
                    info["full_registered"] = len(full[0].camera_poses)
                    info["full_points"] = len(full[0].points)
                verdict = None
                if arbiter is not None and inc_result is not None and full is not None:
                    try:
                        full_assess = arbiter.assess(full[0], "full")
                        verdict = arbiter.choose(inc_assess, full_assess)
                        info["arbitration"] = {"incremental": inc_assess, "full": full_assess, **verdict}
                    except Exception as exc:  # noqa: BLE001
                        info["arbiter_error"] = f"{type(exc).__name__}: {exc}"[:300]
                if verdict is not None:
                    use_incremental = verdict["choice"] == "incremental"
                else:  # no world-level evaluation available: the historical registration-count rule
                    use_incremental = inc_result is not None and (full is None or len(full[0].camera_poses) <= n_inc)
                if use_incremental:
                    result, chosen_dir = inc_result, ws / "inc" / "adjusted"
                    info.update(mode="incremental", reason=(
                        "every photograph registered" if full is None else
                        verdict["why"] if verdict is not None else
                        "a full rebuild registered no more photographs, so the established model was kept"))
                else:
                    result, dirs, ref = full
                    chosen_dir = ws / "sparse_full" / ref
                    incremental_ok = len(dirs) == 1
                    info.update(mode="full", reason=(
                        "incremental registration failed" if inc_result is None else
                        verdict["why"] if verdict is not None else
                        f"a full rebuild registered more photographs ({len(result.camera_poses)} vs {n_inc})"))

        # canonical layout: the chosen model becomes sparse/0 (dense continuation and the next run read it)
        canonical = ws / "sparse" / "0"
        if chosen_dir.resolve() != canonical.resolve():
            staged = ws / "_chosen"
            shutil.rmtree(staged, ignore_errors=True)
            shutil.copytree(chosen_dir, staged)
            shutil.rmtree(ws / "sparse", ignore_errors=True)
            (ws / "sparse").mkdir()
            os.replace(staged, canonical)
        for scratch in ("inc", "sparse_full", "new_images.txt"):
            p = ws / scratch
            shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink(missing_ok=True)

        name_by_eid = {eid: n for n, eid in evidence_id_by_name.items()}
        registered_names = sorted(name_by_eid[p.evidence_id] for p in result.camera_poses if p.evidence_id in name_by_eid)
        # Incremental / reused runs keep COLMAP's coordinate frame; a full re-solve starts a new one
        # (the geometry can still agree -- world_delta.camera_consistency measures that frame-independently).
        info["frame"] = "preserved" if info.get("mode") in ("incremental", "reused") else "re-solved"
        info["registered_total"] = len(registered_names)
        info["unregistered"] = sorted(evidence_id_by_name[n] for n in by_name if n not in set(registered_names))
        info["seconds"] = dict(timings, total=round(time.time() - t_start, 2))
        self.session.write_manifest({
            "signature": signature, "images": evidence_id_by_name, "registered": registered_names,
            "model_dir": "sparse/0", "incremental_ok": incremental_ok, "info": info,
        })
        self.last_run_info = info
        self.session.last_run_info = info

        if not self.dense_mvs:
            return result
        return self._continue_dense(workspace=ws, image_dir=image_dir, sparse_result=result,
                                    evidence_id_by_name=evidence_id_by_name)

    def _continue_dense(self, *, workspace: Path, image_dir: Path,
                        sparse_result: ReconstructionResult,
                        evidence_id_by_name: Dict[str, str]) -> ReconstructionResult:
        """Continue the mapper's sparse model through the canonical dense
        chain IN THE SAME WORKSPACE and attach the fused cloud to the
        result. Honesty rules:

        - the binary must PROBE dense-capable first (a COLMAP on PATH
          proves nothing about the build) -- refusal happens BEFORE any
          sparse/dense compute is spent;
        - the sub-model continued is the mapper's reference model (the
          one whose points won the merge reference, or the sole model);
        - fused provenance is the REGISTERED evidence ids (only cameras
          the sparse model actually solved contribute images to the
          dense rectification);
        - a dense failure propagates -- a broken dense chain never
          degrades into a sparse-only success.
        """
        from reconstruction.backend.dense_output import parse_fused_ply

        # Capability was already probed at the top of reconstruct()
        # (before any compute); this continuation assumes it.
        from reconstruction.dense_pipeline import run_dense_mvs

        # The sub-model to continue: the mapper's most complete model.
        # The workspace layout is <ws>/sparse/<n> after model_converter
        # (TXT in place); pick by point count via the parsed results.
        sparse_dir = workspace / "sparse"
        model_dirs = _collect_submodels(sparse_dir, None)
        # Map evidence ids -> registered model names for provenance.
        registered_ids = [pose.evidence_id for pose in sparse_result.camera_poses]

        # Reference model = the sub-model with the most points (same
        # criterion the merge uses); with a single model it's that one.
        reference_dir = None
        best = -1
        for d in model_dirs:
            pts_file = d / "points3D.txt"
            n = sum(1 for line in pts_file.read_text().splitlines()
                    if line.strip() and not line.startswith("#")) if pts_file.is_file() else 0
            if n > best:
                best, reference_dir = n, d
        if reference_dir is None:
            raise DenseMVSUnavailableError(
                "no sparse sub-model directory found to continue into "
                f"dense MVS (looked under {sparse_dir})"
            )

        run = run_dense_mvs(
            image_dir=image_dir,
            sparse_model_dir=reference_dir,
            workspace=workspace,
            binary=self._colmap_binary,
            # Dense stereo honors the same GPU decision as sparse; the
            # CPU Windows build must not be handed use_gpu=1.
            use_gpu=True if self._use_gpu else False,
        )

        fused_bytes = Path(run.fused_ply_path).read_bytes()
        dense_points, parse_facts = parse_fused_ply(
            fused_bytes,
            source_evidence_ids=sorted(set(registered_ids)),
            return_facts=True,
        )

        report = run.to_dict()
        report["sparse_model"] = reference_dir.name
        report["source_evidence_ids"] = sorted(set(registered_ids))
        report["parse_facts"] = parse_facts

        # Persist BEFORE the workspace dies: ingest through the
        # canonical chain so the cloud becomes a content-addressed,
        # traceable artifact instead of an orphaned file in a temp dir
        # that this method's caller is about to delete. Refusal (not
        # fallback) when the cloud would be untraceable.
        if self.artifact_store is not None:
            from world_ir.artifact_store import ArtifactStore
            from reconstruction.dense_ingest import ingest_fused_ply
            if not isinstance(self.artifact_store, ArtifactStore):
                raise TypeError(
                    "artifact_store must be a world_ir.artifact_store."
                    "ArtifactStore (canonical content-addressed store), "
                    f"got {type(self.artifact_store).__name__}"
                )
            geometry = ingest_fused_ply(
                fused_bytes,
                artifact_store=self.artifact_store,
                source_evidence_ids=sorted(set(registered_ids)),
            )
            report["artifact_uri"] = geometry.data_uri
            report["artifact_sha256"] = geometry.data_hash
            report["artifact_vertex_count"] = geometry.vertex_count
            report["provenance"] = geometry.provenance.value
        else:
            report["artifact"] = (
                "NOT PERSISTED -- no artifact_store configured; the "
                "parsed cloud travels on dense_points but the workspace "
                "(and fused.ply) is deleted with it"
            )

        return ReconstructionResult(
            points=sparse_result.points,
            camera_poses=sparse_result.camera_poses,
            registration_status=sparse_result.registration_status,
            merge_report=sparse_result.merge_report,
            dense_points=dense_points,
            dense_report=report,
        )
