"""Dense-MVS continuation on the canonical COLMAP backend (P6-01 closure).

The canonical dense chain already exists end to end
(`run_dense_mvs` -> `parse_fused_ply` -> `ingest_fused_ply` -> WorldIR),
and this machine's COLMAP probes as dense-capable -- but the backend
never ran it, so a single honest execution could not go sparse -> dense
and no verified dense run was recorded from repository data. These
tests pin the backend-side continuation:

  - ReconstructionResult carries `dense_points` / `dense_report`
    (default None: sparse-only results are untouched);
  - `dense_mvs=True` runs the real chain shape in the SAME workspace
    (images + sparse model the backend just produced) and attaches the
    parsed fused cloud + observed facts to the result;
  - dense provenance: fused points name the registered evidence ids;
  - a dense stage failure RAISES (never silently returns sparse-only);
  - a binary that does not PROBE dense-capable is refused before any
    compute is spent;
  - default stays OFF (dense is heavyweight and GPU-bound).

COLMAP itself is faked by a script harness that writes COLMAP's REAL
output formats (TXT sparse model, binary fused.ply), so the parsing and
wiring under test are exercised against real artifact bytes, not mocks.
"""

from __future__ import annotations

import dataclasses
import struct
import textwrap

import pytest

from evidence.session import EvidenceItem, EvidenceKind
from reconstruction.backend.interface import ReconstructionResult
from reconstruction.backend.colmap_backend import ColmapReconstructionBackend
from reconstruction.dense_pipeline import DenseMVSUnavailableError


# ---------------------------------------------------------------- fixtures

#: Known fused.ply contents the fake COLMAP writes (3 real binary-PLY
#: vertices with color) -- the test asserts these exact values survive
#: parsing into canonical points.
FUSED_VERTICES = [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0)]

_SPARSE_IMAGES_TXT = """\
# Image list with two lines of data per image:
#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
#   POINTS2D[] as (X, Y, POINT3D_ID)
# Number of images: 2, mean observations per image: 1
1 0.851773 0.0165051 0.503764 -0.142941 -0.737434 1.02973 3.74354 1 ev-p1.jpg
2362.39 248.498 63390
2 0.9 0.0 0.0 0.1 1.0 2.0 3.0 1 ev-p2.jpg
100.0 200.0 63390
"""

_SPARSE_POINTS3D_TXT = """\
# 3D point list with one line of data per point:
#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)
# Number of points: 1, mean track length: 2.0
63390 1.67241 0.191763 0.466115 154 174 102 1.14007 1 0 2 0
"""

#: The fake COLMAP: implements every step the backend invokes, writing
#: COLMAP's real output formats. Configuration arrives via environment
#: variables baked into the .bat shim (isolated per test).
_FAKE_COLMAP_PY = textwrap.dedent("""\
    import os, struct, sys
    from pathlib import Path

    step = sys.argv[1]
    args = sys.argv[2:]

    def opt(name):
        return args[args.index(name) + 1] if name in args else None

    if os.environ.get("FAKE_LOG"):
        with open(os.environ["FAKE_LOG"], "a") as f:
            f.write(step + "\\n")

    if step == "help":
        if os.environ.get("FAKE_NO_DENSE"):
            print("Usage: colmap help")
        else:
            print("Usage: colmap help\\npatch_match_stereo")
    elif len(sys.argv) == 3 and sys.argv[2] == '--help':
        # The pipeline probes each stage's real option names.
        if step == 'patch_match_stereo':
            print('  --workspace_path arg')
            print('  --PatchMatchStereo.geom_consistency arg')
            print('  --PatchMatchStereo.use_gpu arg')
        elif step == 'stereo_fusion':
            print('  --workspace_path arg')
            print('  --StereoFusion.input_type arg')
            print('  --StereoFusion.output arg')
    elif step == "feature_extractor":
        Path(opt("--database_path")).touch()
    elif step == "exhaustive_matcher":
        pass
    elif step == "mapper":
        model = Path(opt("--output_path")) / "0"
        model.mkdir(parents=True)
        (model / "images.txt").write_text(%(IMAGES)r)
        (model / "points3D.txt").write_text(%(POINTS3D)r)
        (model / "cameras.txt").write_text(
            "# Camera list\\n1 PINHOLE 100 100 1160.0 1160.0 50.0 50.0\\n")
    elif step == "model_converter":
        pass  # fixture models are already TXT
    elif step == "image_undistorter":
        Path(opt("--output_path")).mkdir(parents=True, exist_ok=True)
    elif step == "patch_match_stereo":
        pass
    elif step == "stereo_fusion":
        if os.environ.get("FAKE_FAIL_STEP") == step:
            sys.stderr.write("stereo_fusion: simulated CUDA failure\\n")
            sys.exit(1)
        dense = Path(opt("--workspace_path"))
        verts = %(VERTS)r
        header = (
            "ply\\nformat binary_little_endian 1.0\\nelement vertex %%d\\n"
            "property float x\\nproperty float y\\nproperty float z\\n"
            "property uchar red\\nproperty uchar green\\nproperty uchar blue\\n"
            "end_header\\n" %% len(verts)
        ).encode("ascii")
        body = b"".join(
            struct.pack("<3f3B", x, y, z, 10, 20, 30) for x, y, z in verts
        )
        (dense / "fused.ply").write_bytes(header + body)
    else:
        sys.stderr.write("fake colmap: unknown step " + step + "\\n")
        sys.exit(1)
""") % {"IMAGES": _SPARSE_IMAGES_TXT, "POINTS3D": _SPARSE_POINTS3D_TXT,
        "VERTS": FUSED_VERTICES}


def _install_fake_colmap(tmp_path, *, fail_step=None, no_dense=False):
    """Install the fake COLMAP (a .bat shim + python script, configured
    via env vars set inside the shim so parallel tests stay isolated)
    and return (binary_path, step_log_path)."""
    script = tmp_path / "fake_colmap.py"
    script.write_text(_FAKE_COLMAP_PY)
    log = tmp_path / "steps.log"
    log.write_text("")
    shim = tmp_path / "colmap-fake.bat"
    lines = ["@echo off"]
    if fail_step:
        lines.append(f"set FAKE_FAIL_STEP={fail_step}")
    if no_dense:
        lines.append("set FAKE_NO_DENSE=1")
    lines.append(f'set FAKE_LOG={log}')
    lines.append(f'@python "{script}" %*')
    shim.write_text("\r\n".join(lines) + "\r\n")
    return str(shim), log


def _two_photos(tmp_path):
    # Bytes only -- the backend copies files, it never decodes here.
    a = tmp_path / "a.jpg"; a.write_bytes(b"\xff\xd8fake-jpeg-a")
    b = tmp_path / "b.jpg"; b.write_bytes(b"\xff\xd8fake-jpeg-b")
    return [
        EvidenceItem(id="ev-p1", kind=EvidenceKind.PHOTO,
                     source_uri=a.as_uri()),
        EvidenceItem(id="ev-p2", kind=EvidenceKind.PHOTO,
                     source_uri=b.as_uri()),
    ]


# ------------------------------------------------------------------ tests

class TestResultDenseFields:
    def test_dense_fields_default_to_none(self):
        result = ReconstructionResult(points=[], camera_poses=[],
                                      registration_status="failed")
        assert result.dense_points is None
        assert result.dense_report is None

    def test_dense_fields_round_trip(self):
        result = ReconstructionResult(points=[], camera_poses=[],
                                      registration_status="success",
                                      dense_points=[],
                                      dense_report={"n_fused_points": 0})
        assert result.dense_report == {"n_fused_points": 0}


class TestBackendDenseContinuation:
    def test_dense_off_by_default_runs_sparse_only(self, tmp_path):
        binary, log = _install_fake_colmap(tmp_path)
        backend = ColmapReconstructionBackend(colmap_binary=binary)
        result = backend.reconstruct(_two_photos(tmp_path))
        steps = log.read_text().splitlines()
        assert "patch_match_stereo" not in steps
        assert "stereo_fusion" not in steps
        assert result.dense_points is None
        assert result.dense_report is None
        # The sparse result itself is what it always was.
        assert len(result.camera_poses) == 2
        assert len(result.points) == 1

    def test_dense_mvs_attaches_fused_points_and_report(self, tmp_path):
        binary, log = _install_fake_colmap(tmp_path)
        from world_ir.artifact_store import FileArtifactStore
        store = FileArtifactStore(root=tmp_path / "artifacts")
        backend = ColmapReconstructionBackend(
            colmap_binary=binary, dense_mvs=True, use_gpu=False,
            artifact_store=store)
        result = backend.reconstruct(_two_photos(tmp_path))

        steps = log.read_text().splitlines()
        # The dense chain ran IN the same workspace, after the mapper.
        assert steps.index("mapper") < steps.index("image_undistorter")
        assert "patch_match_stereo" in steps
        assert "stereo_fusion" in steps

        # Sparse result preserved alongside the dense cloud.
        assert len(result.camera_poses) == 2
        assert len(result.points) == 1

        # Fused vertices parsed from REAL binary PLY bytes.
        assert result.dense_points is not None
        assert len(result.dense_points) == 3
        first = result.dense_points[0]
        assert first.position == FUSED_VERTICES[0]
        assert first.track_id == "dense:0"
        # Provenance: the cloud names the registered evidence.
        assert sorted(result.dense_points[0].source_evidence_ids) == [
            "ev-p1", "ev-p2"]

        report = result.dense_report
        assert report is not None
        assert report["n_fused_points"] == 3
        assert report["sparse_model"] == "0"
        assert report["geom_consistency"] is True
        assert report["use_gpu"] is False
        assert set(report["stage_durations_s"]) == {
            "image_undistorter", "patch_match_stereo", "stereo_fusion"}
        assert report["parse_facts"]["had_colors"] is True
        assert report["parse_facts"]["n_points"] == 3

        # Canonical artifact: persisted INSIDE the workspace lifetime,
        # content-addressed, decodable -- not an orphaned file in the
        # temp dir the backend just deleted.
        assert report["artifact_uri"].startswith("artifact://")
        assert len(report["artifact_sha256"]) == 64
        from world_ir.artifact_store import FileArtifactStore
        from world_ir.geometry_data import PointCloudData
        blob = store.get(report["artifact_uri"])
        cloud = PointCloudData.from_bytes(blob)
        assert len(cloud.points) == 3
        assert cloud.points[0] == FUSED_VERTICES[0]

    def test_dense_without_store_records_honest_no_artifact(self, tmp_path):
        """Without a store the parsed cloud still travels, but the report
        must say the fused.ply was NOT persisted -- the workspace (and
        the file) dies with reconstruct()'s return."""
        binary, _log = _install_fake_colmap(tmp_path)
        backend = ColmapReconstructionBackend(colmap_binary=binary,
                                              dense_mvs=True)
        result = backend.reconstruct(_two_photos(tmp_path))
        assert result.dense_points is not None
        assert "NOT PERSISTED" in result.dense_report["artifact"]

    def test_dense_stage_failure_raises_never_sparse_success(self, tmp_path):
        binary, _log = _install_fake_colmap(tmp_path, fail_step="stereo_fusion")
        backend = ColmapReconstructionBackend(colmap_binary=binary,
                                              dense_mvs=True)
        with pytest.raises(Exception) as exc_info:
            backend.reconstruct(_two_photos(tmp_path))
        assert "stereo_fusion" in str(exc_info.value)

    def test_dense_refused_when_binary_not_dense_capable(self, tmp_path):
        binary, log = _install_fake_colmap(tmp_path, no_dense=True)
        backend = ColmapReconstructionBackend(colmap_binary=binary,
                                              dense_mvs=True)
        with pytest.raises(DenseMVSUnavailableError) as exc_info:
            backend.reconstruct(_two_photos(tmp_path))
        # Refused BEFORE spending any sparse compute: only the `help`
        # probe ran (a capability probe is not compute).
        assert "feature_extractor" not in log.read_text()
        assert "mapper" not in log.read_text()
        assert "patch_match_stereo" in str(exc_info.value)
