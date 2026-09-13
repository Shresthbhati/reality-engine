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

from apps.cli.main import main, _artifacts_dir_for
from engine.compiler import CompileOptions, compile_reconstruction_to_world
from world_ir.artifact_store import FileArtifactStore


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


def _compiled_world_with_store_fixture(store_root):
    from reconstruction.backend.interface import ReconstructedCameraPose
    from tests.test_room_inference import _CAMS, _two_room_scene

    result = _two_room_scene()
    result.camera_poses.extend(
        ReconstructedCameraPose(evidence_id=f"ev-{i}", position=p, rotation=(1.0, 0.0, 0.0, 0.0))
        for i, p in enumerate(_CAMS)
    )
    store = FileArtifactStore(store_root)
    world, _diag = compile_reconstruction_to_world(result, CompileOptions(seed=42, artifact_store=store))
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


class TestReconstructArtifactStoreDefault:
    def test_no_real_geometry_flag_is_registered_and_skips_store_creation(self, tmp_path, capsys):
        # Full COLMAP-backed reconstruct isn't available in this test env (see
        # TestReconstructRefusesWithoutRealBackendData), so this only exercises
        # the CLI-level flag/argument wiring: refusal happens before any
        # artifact store would be created either way, and no artifacts dir
        # should appear regardless of the flag when reconstruction refuses.
        photos_dir = tmp_path / "photos"
        photos_dir.mkdir()
        (photos_dir / "a.jpg").write_bytes(_make_jpeg_bytes((1, 1, 1)))
        (photos_dir / "b.jpg").write_bytes(_make_jpeg_bytes((2, 2, 2)))
        package_path = tmp_path / "package.json"
        main(["ingest", str(photos_dir), "-o", str(package_path)])

        world_path = str(tmp_path / "world.json")
        rc = main([
            "reconstruct", str(package_path), "-o", world_path,
            "--colmap-binary", "definitely-not-a-real-binary-xyz",
            "--no-real-geometry",
        ])

        assert rc == 1
        assert not _artifacts_dir_for(world_path).exists()


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


class TestArtifactStorePathConvention:
    def test_artifacts_dir_for_derives_from_output_path(self):
        assert _artifacts_dir_for("world.json").name == "world.json.artifacts"
        assert str(_artifacts_dir_for("out/world.json")).replace("\\", "/") == "out/world.json.artifacts"


class TestExportReconnectsToRealGeometryStore:
    def test_export_gltf_after_real_reconstruct_emits_real_meshes(self, tmp_path, capsys):
        world_path = str(tmp_path / "world.json")
        store_root = _artifacts_dir_for(world_path)
        world = _compiled_world_with_store_fixture(store_root)
        _write_world(tmp_path, world, "world.json")  # overwrite with the real-geometry compiled world

        assert any(g.data_uri for g in world.geometries.values())  # sanity: real artifacts exist
        assert store_root.is_dir()  # sanity: artifacts directory actually persisted to disk

        out_path = tmp_path / "out.gltf"
        rc = main(["export", world_path, "--format", "gltf", "-o", str(out_path)])

        assert rc == 0
        gltf = json.loads(out_path.read_text(encoding="utf-8"))
        mesh_indices_used = {node["mesh"] for node in gltf["nodes"]}
        assert mesh_indices_used - {0}, "expected at least one real (non-cube) mesh via the reconnected store"

    def test_export_gltf_without_artifacts_dir_falls_back_to_cube(self, tmp_path, capsys):
        # A world compiled WITHOUT an artifact store (--no-real-geometry equivalent):
        # no <world>.artifacts/ dir exists, so export must behave exactly as before.
        world_path = _write_world(tmp_path, _compiled_world_fixture())
        assert not _artifacts_dir_for(world_path).exists()

        out_path = tmp_path / "out.gltf"
        rc = main(["export", world_path, "--format", "gltf", "-o", str(out_path)])

        assert rc == 0
        gltf = json.loads(out_path.read_text(encoding="utf-8"))
        assert all(node["mesh"] == 0 for node in gltf["nodes"])

    @pytest.mark.parametrize("fmt,suffix", [("usda", ".usda"), ("blender", ".py")])
    def test_export_usda_blender_also_reconnect_to_the_store(self, tmp_path, capsys, fmt, suffix):
        """usda/blender exporters gained real-geometry support after this
        test was first written (`sdk.reality.export()` now threads
        `artifact_store` through uniformly for all three formats, not
        just gltf) -- this reconnects the same way gltf does."""
        world_path = str(tmp_path / "world.json")
        store_root = _artifacts_dir_for(world_path)
        world = _compiled_world_with_store_fixture(store_root)
        _write_world(tmp_path, world, "world.json")

        out_path = tmp_path / f"out{suffix}"
        rc = main(["export", world_path, "--format", fmt, "-o", str(out_path)])

        assert rc == 0
        content = out_path.read_text(encoding="utf-8")
        assert content  # non-empty
        if fmt == "usda":
            assert 'def Points "' in content  # real point-cloud prim, not just Cube
        else:
            assert "from_pydata(" in content  # real bpy mesh, not just primitive_cube_add

    def test_export_usda_blender_fall_back_to_placeholder_without_a_store(self, tmp_path, capsys):
        """No <world>.artifacts/ directory (e.g. --no-real-geometry) ->
        both exporters must still fall back to their placeholder shape,
        exactly like before real-geometry support existed."""
        world_path = _write_world(tmp_path, _compiled_world_fixture())
        assert not _artifacts_dir_for(world_path).exists()

        usda_out = tmp_path / "out.usda"
        rc = main(["export", world_path, "--format", "usda", "-o", str(usda_out)])
        assert rc == 0
        usda_content = usda_out.read_text(encoding="utf-8")
        assert 'def Cube "' in usda_content
        assert 'def Points "' not in usda_content

        blender_out = tmp_path / "out.py"
        rc = main(["export", world_path, "--format", "blender", "-o", str(blender_out)])
        assert rc == 0
        blender_content = blender_out.read_text(encoding="utf-8")
        assert "primitive_cube_add(" in blender_content
        assert "from_pydata(" not in blender_content


class TestQuery:
    def test_query_nearest_finds_localized_entities(self, tmp_path, capsys):
        world_path = _write_world(tmp_path, _compiled_world_fixture())
        rc = main(["query", "nearest", world_path, "0", "0", "0", "--k", "3"])
        assert rc == 0
        lines = capsys.readouterr().out.strip().splitlines()
        assert 1 <= len(lines) <= 3
        for line in lines:
            entity_id, distance, _label = line.split("\t")
            assert float(distance.rstrip("m")) >= 0.0

    def test_query_contents_lists_room_members(self, tmp_path, capsys):
        world = _compiled_world_fixture()
        from world_ir import RelationshipKind

        room_id = next(
            eid for eid, e in world.entities.items()
            if any(r.kind == RelationshipKind.CONTAINS for r in e.relationships)
        )
        world_path = _write_world(tmp_path, world)

        rc = main(["query", "contents", world_path, room_id])

        assert rc == 0
        out = capsys.readouterr().out
        assert out.strip() != ""

    def test_query_contents_rejects_unknown_entity(self, tmp_path, capsys):
        world_path = _write_world(tmp_path, _compiled_world_fixture())
        rc = main(["query", "contents", world_path, "no-such-entity"])
        assert rc == 1
        assert "unknown entity" in capsys.readouterr().err.lower()


class TestPhysics:
    def test_physics_compiles_bodies_for_the_room_scene(self, tmp_path, capsys):
        world_path = _write_world(tmp_path, _compiled_world_fixture())
        rc = main(["physics", world_path])
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert "results" in payload
        assert len(payload["results"]) > 0
