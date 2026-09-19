"""Integration: the real vertical slice through the real CLI entry points.

UNIT coverage lives next to each module (test_room_inference.py,
test_world_compiler.py, test_world_store.py, test_registration.py, ...).
This file wires those same real modules through the `reality` CLI --
ingest -> compile (REALITY_TEST_BACKEND=module:Class seam) -> validate
-> store save/load/verify -> inspect/query/export/viewer -- with zero
mocks of the engine. The injected backend's geometry is test-supplied
deterministic input, not invented production geometry: the CLI still
refuses honestly (non-zero) when no backend can produce anything.
"""
from __future__ import annotations

import json

import pytest

from apps.cli.main import main
from tests.test_cli_compile import _TwoViewBackend  # noqa: F401  (backend seam target)


_BACKEND_SPEC = "tests.test_cli_compile:_TwoViewBackend"


def _capture_dataset(tmp_path, n=3):
    from PIL import Image

    d = tmp_path / "capture"
    (d / "images").mkdir(parents=True)
    for i in range(n):
        Image.new("RGB", (64, 48), color=(i * 40, 100, 150)).save(
            d / "images" / f"img_{i:03d}.jpg"
        )
    manifest = {
        "dataset": "cli_vertical_slice",
        "images": [{"file": f"img_{i:03d}.jpg"} for i in range(n)],
        "measured_baselines": [
            {"evidence_id_a": "img_000", "evidence_id_b": "img_001", "distance_m": 1.0}
        ],
        "scale_reference": {"method": "test_baseline"},
    }
    (d / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return d


@pytest.mark.integration
def test_cli_vertical_slice_end_to_end(tmp_path, monkeypatch, capsys):
    """ingest -> compile -> validate -> store -> inspect/query/export/viewer."""
    monkeypatch.setenv("REALITY_TEST_BACKEND", _BACKEND_SPEC)
    dataset = _capture_dataset(tmp_path)
    out = tmp_path / "pipe"

    assert main(["compile", str(dataset), "-o", str(out), "--no-depth", "--no-mesh"]) == 0
    capsys.readouterr()
    world_path = str(out / "worldir.json")
    world = json.loads((out / "worldir.json").read_text(encoding="utf-8"))
    assert len(world["entities"]) > 0

    assert main(["validate", world_path]) == 0
    capsys.readouterr()

    store_dir = str(tmp_path / "store")
    assert main(["store", "save", world_path, "--store", store_dir,
                 "--version-id", "v-e2e"]) == 0
    capsys.readouterr()
    assert main(["store", "verify", "--store", store_dir, "--version", "v-e2e"]) == 0
    capsys.readouterr()

    assert main(["inspect", world_path]) == 0
    summary = json.loads(capsys.readouterr().out)
    entity_id = sorted(world["entities"])[0]

    assert main(["inspect", world_path, "--entity", entity_id]) == 0
    detail = json.loads(capsys.readouterr().out)
    assert detail["id"] == entity_id
    assert "provenance" in detail

    assert main(["query", "nearest", world_path, "0", "0", "0", "--k", "2"]) == 0
    assert capsys.readouterr().out.strip() != ""

    for fmt, name in (("gltf", "out.gltf"), ("usda", "out.usda"), ("blender", "out.py")):
        out_p = str(tmp_path / name)
        assert main(["export", world_path, "--format", fmt, "-o", out_p]) == 0
        assert (tmp_path / name).read_text(encoding="utf-8").strip() != ""
    capsys.readouterr()

    viewer_out = str(tmp_path / "viewer.html")
    assert main(["viewer", "--worldir", world_path, "-o", viewer_out]) == 0
    # Verify the viewer HTML was generated with proper structure
    viewer_html = (tmp_path / "viewer.html").read_text(encoding="utf-8")
    assert "<!doctype html>" in viewer_html
    assert "Reality Engine" in viewer_html
    assert "importmap" in viewer_html
    assert "__appUrl" in viewer_html


@pytest.mark.integration
def test_cli_compile_still_refuses_without_geometry(tmp_path, monkeypatch, capsys):
    """No backend + no COLMAP -> honest non-zero refusal, never a fake world."""
    monkeypatch.delenv("REALITY_TEST_BACKEND", raising=False)
    dataset = _capture_dataset(tmp_path)
    rc = main(["compile", str(dataset), "-o", str(tmp_path / "out"),
               "--no-depth", "--no-mesh",
               "--colmap-binary", "definitely-not-colmap-xyz"])
    assert rc == 1
    assert "compile failed" in capsys.readouterr().err
    assert not (tmp_path / "out" / "worldir.json").exists()
