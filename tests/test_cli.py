"""Tests for apps/cli/main.py -- the headless CLI client of sdk.reality.

Exercises every subcommand end-to-end against real files on disk (no
mocking of the SDK layer): ingest a folder of real JPEGs into a package,
then validate/diff/export/physics against a real compiled WorldIR built
the same way tests/test_world_compiler.py builds one. `reconstruct` is
tested separately for its honest-failure path (no COLMAP/canned data ->
refuses, never fabricates) since a full photogrammetry run needs a real
multi-view photoset this repo does not ship.
"""

from __future__ import annotations

import io
import json

import pytest

from apps.cli.main import main
from engine.compiler import CompileOptions, compile_reconstruction_to_world


def _compiled_world_fixture():
    from reconstruction.backend.interface import ReconstructedCameraPose
    from tests.test_room_inference import _CAMS, _two_room_scene

    result = _two_room_scene()
    result.camera_poses.extend(
        ReconstructedCameraPose(evidence_id=f"ev-{i}", position=p, rotation=(1.0, 0.0, 0.0, 0.0))
        for i, p in enumerate(_CAMS)
    )
    world, _diag = compile_reconstruction_to_world(result, CompileOptions(seed=42))
    return world


def _write_world(tmp_path, world, name="world.json"):
    path = tmp_path / name
    path.write_text(json.dumps(world.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return str(path)


def _make_jpeg_bytes(color) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color=color).save(buf, format="JPEG")
    return buf.getvalue()


class TestIngest:
    def test_ingest_folder_writes_a_package(self, tmp_path, capsys):
        photos_dir = tmp_path / "photos"
        photos_dir.mkdir()
        (photos_dir / "a.jpg").write_bytes(_make_jpeg_bytes((255, 0, 0)))
        (photos_dir / "b.jpg").write_bytes(_make_jpeg_bytes((0, 255, 0)))
        (photos_dir / "notes.txt").write_text("not evidence")

        out = tmp_path / "package.json"
        rc = main(["ingest", str(photos_dir), "-o", str(out), "--seed", "test"])

        assert rc == 0
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["package_id"] == "pkg-test"
        assert len(data["assets"]) == 2
        err = capsys.readouterr().err
        assert "notes.txt" in err  # unhandled extension surfaced, not silently dropped

    def test_ingest_is_deterministic(self, tmp_path):
        photos_dir = tmp_path / "photos"
        photos_dir.mkdir()
        (photos_dir / "a.jpg").write_bytes(_make_jpeg_bytes((10, 20, 30)))

        out1, out2 = tmp_path / "p1.json", tmp_path / "p2.json"
        main(["ingest", str(photos_dir), "-o", str(out1), "--seed", "s"])
        main(["ingest", str(photos_dir), "-o", str(out2), "--seed", "s"])

        assert out1.read_text(encoding="utf-8") == out2.read_text(encoding="utf-8")


class TestReconstructRefusesWithoutRealBackendData:
    def test_reconstruct_refuses_honestly_without_colmap_or_canned_data(self, tmp_path, capsys):
        photos_dir = tmp_path / "photos"
        photos_dir.mkdir()
        (photos_dir / "a.jpg").write_bytes(_make_jpeg_bytes((1, 1, 1)))
        (photos_dir / "b.jpg").write_bytes(_make_jpeg_bytes((2, 2, 2)))
        package_path = tmp_path / "package.json"
        main(["ingest", str(photos_dir), "-o", str(package_path)])

        rc = main([
            "reconstruct", str(package_path), "-o", str(tmp_path / "world.json"),
            "--colmap-binary", "definitely-not-a-real-binary-xyz",
        ])

        # No real backend can honestly produce geometry from two flat-color
        # test images -- the CLI must report failure, not invent a world.
        assert rc == 1
        assert not (tmp_path / "world.json").exists()
        assert "refused" in capsys.readouterr().err.lower()


class TestValidate:
    def test_validate_a_valid_compiled_world_exits_zero(self, tmp_path, capsys):
        world_path = _write_world(tmp_path, _compiled_world_fixture())
        rc = main(["validate", world_path])
        assert rc == 0
        assert "valid: True" in capsys.readouterr().out

    def test_validate_reports_issues_for_a_broken_world(self, tmp_path, capsys):
        from world_ir import Entity, EntityType, WorldIR

        broken = WorldIR(id="w-broken", main_branch_id="branch-main-w-broken")
        broken.entities["e1"] = Entity(
            id="e1", name="dangling", type=EntityType.UNKNOWN,
            geometry_ids=["does-not-exist"],
        )
        world_path = _write_world(tmp_path, broken)

        rc = main(["validate", world_path])

        assert rc == 1
        assert "False" in capsys.readouterr().out


class TestDiff:
    def test_diff_two_identical_worlds_is_empty(self, tmp_path, capsys):
        world = _compiled_world_fixture()
        a = _write_world(tmp_path, world, "a.json")
        b = _write_world(tmp_path, world, "b.json")

        rc = main(["diff", a, b])

        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["summary"]["entities_added"] == 0
        assert payload["summary"]["entities_removed"] == 0

    def test_diff_detects_a_removed_entity(self, tmp_path, capsys):
        world = _compiled_world_fixture()
        a = _write_world(tmp_path, world, "a.json")
        some_id = next(iter(world.entities))
        del world.entities[some_id]
        b = _write_world(tmp_path, world, "b.json")

        rc = main(["diff", a, b])

        payload = json.loads(capsys.readouterr().out)
        assert rc == 0
        assert payload["summary"]["entities_removed"] == 1


class TestExport:
    @pytest.mark.parametrize("fmt,suffix", [("gltf", ".gltf"), ("usda", ".usda"), ("blender", ".py")])
    def test_export_each_format_writes_real_content(self, tmp_path, capsys, fmt, suffix):
        world_path = _write_world(tmp_path, _compiled_world_fixture())
        out_path = tmp_path / f"out{suffix}"

        rc = main(["export", world_path, "--format", fmt, "-o", str(out_path)])

        assert rc == 0
        assert out_path.exists()
        assert out_path.stat().st_size > 0
        assert "exported" in capsys.readouterr().out

    def test_export_rejects_unknown_format(self, tmp_path, capsys):
        world_path = _write_world(tmp_path, _compiled_world_fixture())
        with pytest.raises(SystemExit) as exc_info:
            main(["export", world_path, "--format", "unknown-fmt", "-o", str(tmp_path / "x")])
        assert exc_info.value.code == 2  # argparse rejects the invalid choice before reaching the SDK


class TestPhysics:
    def test_physics_compiles_bodies_for_the_room_scene(self, tmp_path, capsys):
        world_path = _write_world(tmp_path, _compiled_world_fixture())
        rc = main(["physics", world_path])
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert "results" in payload
        assert len(payload["results"]) > 0
