"""P0.18 -- committed real-data acceptance test for `reality compile`.

Drives the real product path (apps.cli.main `compile`: real COLMAP SfM ->
metric scale -> real MiDaS depth -> mesh -> WorldIR -> validation -> artifacts
and glTF) on real photographs: a 10-view window (indices 15:25 of the
name-sorted list) of the committed South Building dataset
(datasets/south_building, see its README/MANIFEST).

Two runs, because the dataset carries no tape measurement:

* RELATIVE run (no scale reference): real reconstruction + validated WorldIR +
  artifacts, and the pipeline must *honestly refuse* metric-only stages.
* ANCHORED run: a TEST-DECLARED nominal 1.0 m baseline between neighbouring
  views (method "test_declared_nominal") plus the dataset's real COLMAP
  intrinsics (cameras.txt, rescaled 3072 -> 1024 px) unlock the metric chain
  (depth; mesh is NOT exercised -- see below). The metres are nominal, so nothing here asserts real-world
  accuracy -- only that the metric code path runs on real photos and its
  outputs are internally consistent.

Mesh (P0.11-13) is deliberately not run here: on this data the CPU Poisson
mesh stage had not finished after 40+ minutes (measured 2026-10-17), so the
anchored run uses `--no-mesh` and the mesh criterion stays UNVERIFIED on
South Building (it was verified on the room capture, see backlog P0.18).

Marked slow; skipped (never silently passed) when COLMAP/MiDaS or the
dataset are absent.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

DATASET = Path(__file__).resolve().parents[2] / "datasets" / "south_building" / "images"
IMGS = sorted(DATASET.glob("*.JPG")) if DATASET.exists() else []
WINDOW = IMGS[15:25]


def _engines_available() -> tuple[bool, str]:
    try:
        from perception.availability import report_real_models

        status = {m.name: m.status for m in report_real_models()}
    except Exception as exc:  # pragma: no cover
        return False, f"availability probe failed: {exc}"
    missing = [n for n in ("colmap", "midas") if status.get(n) != "available"]
    return (not missing), f"unavailable real engines: {missing}"


_OK, _WHY = _engines_available()
pytestmark = [
    pytest.mark.slow,
    pytest.mark.integration,
    pytest.mark.skipif(len(WINDOW) < 10, reason="South Building dataset not present"),
    pytest.mark.skipif(not _OK, reason=f"real engines required -- UNVERIFIED ({_WHY})"),
]


def _compile(tmp: Path, name: str, anchored: bool) -> tuple[int, dict, Path]:
    from apps.cli.main import main

    cap = tmp / f"{name}_capture"
    (cap / "images").mkdir(parents=True)
    for p in WINDOW:
        shutil.copy(p, cap / "images" / p.name)
    stems = [p.stem for p in WINDOW]
    manifest: dict = {
        "dataset": "south_building_p018",
        "images": [{"file": p.name} for p in WINDOW],
        "image_size": [1024, 768],
    }
    if anchored:
        # Real intrinsics: COLMAP SIMPLE_RADIAL f=2559.68 @ 3072x2304 -> 1024x768.
        manifest["intrinsics_px"] = {"fx": 853.2267, "fy": 853.2267, "cx": 512.0, "cy": 384.0}
        manifest["measured_baselines"] = [
            {"evidence_id_a": a, "evidence_id_b": b, "distance_m": 1.0}
            for a, b in zip(stems, stems[1:])
        ]
        manifest["scale_reference"] = {"method": "test_declared_nominal"}
    (cap / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    out = tmp / f"{name}_out"
    rc = main(["compile", str(cap), "-o", str(out)] + (["--no-mesh"] if anchored else []))
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    return rc, report, out


@pytest.fixture(scope="module")
def relative(tmp_path_factory):
    return _compile(tmp_path_factory.mktemp("p018"), "rel", anchored=False)


@pytest.fixture(scope="module")
def anchored(tmp_path_factory):
    return _compile(tmp_path_factory.mktemp("p018"), "anc", anchored=True)


def _assert_spine(rc: int, report: dict, out: Path) -> dict:
    """Criteria common to both runs: reconstruction, WorldIR, artifacts."""
    from apps.cli.main import main

    # C1 one-command path exits 0 and reports a non-failed status.
    assert rc == 0
    assert report["status"] in ("SUCCESS", "PARTIAL_SUCCESS")
    # C2 real SfM registered most of the 10 input views.
    rec = report["stages"]["reconstruction"]
    assert rec["backend"] == "ColmapReconstructionBackend"
    assert rec["cameras_input"] == 10
    assert rec["cameras_registered"] >= 6
    # C3 real sparse points.
    assert rec["points"] >= 1000
    # C4 entities compiled into WorldIR.
    assert report["stages"]["compile"]["entities"] > 0
    world = json.loads((out / "worldir.json").read_text(encoding="utf-8"))
    assert world["entities"]
    # C5 WorldIR passes the validation gate (`reality validate` exits 0).
    assert main(["validate", str(out / "worldir.json")]) == 0
    # C6 artifacts on disk: points, cameras, glTF.
    assert (out / "points.ply").stat().st_size > 0
    cams = json.loads((out / "cameras.json").read_text(encoding="utf-8"))
    assert cams
    gltf = json.loads((out / "exports" / "scene.gltf").read_text(encoding="utf-8"))
    assert gltf["asset"]["version"] == "2.0"
    return world


def test_relative_run_real_photos(relative):
    rc, report, out = relative
    _assert_spine(rc, report, out)
    # C7 no metric reference -> scale declared RELATIVE, never claimed metric,
    # and metric-only stages refuse instead of fabricating geometry.
    scale = report["stages"]["scale"]
    assert str(scale["state"]).lower() == "relative"
    assert scale["meters_per_unit"] is None
    assert report["stages"]["mesh"]["status"] == "skipped"
    assert report["stages"]["depth"]["status"] == "skipped"


def test_anchored_run_metric_chain(anchored):
    rc, report, out = anchored
    _assert_spine(rc, report, out)
    # C7' scale anchored: metric state with a finite positive scale.
    scale = report["stages"]["scale"]
    assert str(scale["state"]).lower() == "metric"
    assert scale["meters_per_unit"] and scale["meters_per_unit"] > 0
    # C8 real MiDaS depth ran.
    depth = report["stages"]["depth"]
    assert depth["status"] == "ran"
    assert depth["maps"] == 10 and depth["metricized"] >= 6
    assert depth["dense_points"] > 0
