"""Regression tests for the dense-MVS pipeline stage (P6-01 integration).

`engine/pipeline/vertical_slice.py::_dense_mvs_stage` is the seam that
turns the externally verified GPU dense run (2026-09-15: ~294k fused
points) into a first-class engine stage. These tests pin its contract
without COLMAP, GPU, or torch:

  - OFF by default (heavyweight stage; opting out is recorded as
    "disabled", not silently absent);
  - every gate is an HONEST skip that names what is missing
    (capture paths, artifact store, dense-capable binary, >=2 cameras);
  - a failed run is reported as "failed" with the error class, never
    swallowed and never replaced by a fabricated point cloud;
  - the scale contract: fused.ply arrives in COLMAP's OWN SfM scale, so
    positions are rescaled into meters ONLY when the pipeline measured a
    metric anchor -- relative/unknown worlds keep scale_factor=1.0 and
    say so, because claiming meters without an anchor is a scale lie;
  - the run's observed facts (fused count, geom_consistency, use_gpu,
    per-stage durations) are recorded on the WorldIR observation next to
    the artifact they describe.

The real GPU run is covered separately by
tests/test_dense_real_capture_integration.py (real-data-gated); the real
COLMAP dense chain is never faked here.
"""

from __future__ import annotations

import struct

import pytest

from reconstruction.dense_pipeline import DenseMVSRun, DenseMVSRunError


def _fused_ply(n=64, span=2.0):
    """A real-looking fused.ply (floor plane with jitter), the same shape
    COLMAP's stereo_fusion writes: binary little endian, vertex-only,
    x/y/z + rgb."""
    import random

    rng = random.Random(31)
    header = (
        "ply\nformat binary_little_endian 1.0\nelement vertex %d\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n" % n
    ).encode("ascii")
    body = bytearray()
    for _ in range(n):
        x, y = rng.uniform(-span, span), rng.uniform(-span, span)
        z = 0.01 * rng.gauss(0, 1)
        body += struct.pack("<3f3B", x, y, z, 200, 200, 200)
    return header + bytes(body)


def _fake_result(n_cameras=3):
    import types

    return types.SimpleNamespace(
        points=[],
        camera_poses=[
            types.SimpleNamespace(evidence_id=f"img-{i}") for i in range(n_cameras)
        ],
    )


def _capture(tmp_path):
    """A capture layout that exists on disk (the stage hands these paths to
    COLMAP; the fake runner just records them)."""
    image_dir = tmp_path / "images"
    sparse_dir = tmp_path / "sparse" / "0"
    workspace = tmp_path / "dense_workspace"
    image_dir.mkdir(parents=True)
    sparse_dir.mkdir(parents=True)
    return image_dir, sparse_dir, workspace


def _options(tmp_path, **overrides):
    from engine.pipeline.vertical_slice import VerticalSliceOptions
    from world_ir.artifact_store import MemoryArtifactStore

    image_dir, sparse_dir, workspace = _capture(tmp_path)
    defaults = dict(
        artifact_store=MemoryArtifactStore(),
        dense_mvs_enabled=True,
        dense_mvs_image_dir=image_dir,
        dense_mvs_sparse_model_dir=sparse_dir,
        dense_mvs_workspace=workspace,
        colmap_binary="fake-colmap",
    )
    defaults.update(overrides)
    return VerticalSliceOptions(**defaults)


def _patch_probe(monkeypatch, available=True):
    monkeypatch.setattr(
        "reconstruction.dense_pipeline.dense_mvs_available",
        lambda binary: available,
    )


def _patch_run(monkeypatch, tmp_path, fused_bytes=None, calls=None, error=None):
    """Replace the COLMAP dense chain with a fake that writes a REAL
    fused.ply onto disk, so the downstream parse/ingest/artifact chain is
    exercised for real (not mocked)."""
    fused_path = tmp_path / "dense" / "fused.ply"
    fused_path.parent.mkdir(parents=True, exist_ok=True)
    if fused_bytes is None:
        fused_bytes = _fused_ply()
    fused_path.write_bytes(fused_bytes)

    def fake_run(**kwargs):
        if calls is not None:
            calls.append(kwargs)
        if error is not None:
            raise error
        return DenseMVSRun(
            fused_ply_path=str(fused_path),
            n_fused_points=64,
            stage_durations_s={
                "image_undistorter": 1.5,
                "patch_match_stereo": 402.0,
                "stereo_fusion": 9.25,
            },
            geom_consistency=True,
            use_gpu=kwargs.get("use_gpu"),
        )

    monkeypatch.setattr("reconstruction.dense_pipeline.run_dense_mvs", fake_run)
    return fused_path


def _stage(
    tmp_path,
    monkeypatch,
    world=None,
    result=None,
    scale_state="relative",
    meters_per_unit=None,
    **overrides,
):
    import engine.pipeline.vertical_slice as vs
    from world_ir.world_v1 import WorldIR

    if world is None:
        world = WorldIR(id="w-test")
    if result is None:
        result = _fake_result()
    options = _options(tmp_path, **overrides)
    return vs._dense_mvs_stage(result, world, options, scale_state, meters_per_unit)


# ------------------------------------------------------------------ gates

class TestDenseMVSGates:
    def test_disabled_by_default(self):
        """A dense run is heavyweight and GPU-bound: opting out must be
        the default, and it must be RECORDED as disabled."""
        import engine.pipeline.vertical_slice as vs
        from engine.pipeline.vertical_slice import VerticalSliceOptions
        from world_ir.world_v1 import WorldIR

        options = VerticalSliceOptions()
        assert options.dense_mvs_enabled is False
        facts = vs._dense_mvs_stage(
            _fake_result(), WorldIR(id="w"), options, "relative", None
        )
        assert facts["status"] == "disabled"
        assert "off by default" in facts["note"]
        assert "dense_mvs_enabled=True" in facts["note"]

    def test_enabled_without_capture_paths_skips_and_names_them(
        self, tmp_path, monkeypatch
    ):
        facts = _stage(
            tmp_path,
            monkeypatch,
            dense_mvs_image_dir=None,
            dense_mvs_sparse_model_dir=None,
            dense_mvs_workspace=None,
        )
        assert facts["status"] == "skipped"
        for name in (
            "dense_mvs_image_dir",
            "dense_mvs_sparse_model_dir",
            "dense_mvs_workspace",
        ):
            assert name in facts["note"]

    def test_partial_paths_name_only_the_missing_ones(self, tmp_path, monkeypatch):
        facts = _stage(tmp_path, monkeypatch, dense_mvs_image_dir=None)
        assert facts["status"] == "skipped"
        assert "dense_mvs_image_dir" in facts["note"]
        assert "dense_mvs_sparse_model_dir" not in facts["note"]

    def test_no_artifact_store_skips(self, tmp_path, monkeypatch):
        facts = _stage(tmp_path, monkeypatch, artifact_store=None)
        assert facts["status"] == "skipped"
        assert "artifact_store" in facts["note"]

    def test_binary_without_patch_match_stereo_skips_honestly(
        self, tmp_path, monkeypatch
    ):
        """A COLMAP on PATH proves nothing about the build. When the probe
        fails the stage must skip -- and must say a synthetic stand-in
        would be a fake reconstruction."""
        _patch_probe(monkeypatch, available=False)
        calls = []
        _patch_run(monkeypatch, tmp_path, calls=calls)

        facts = _stage(tmp_path, monkeypatch)
        assert facts["status"] == "skipped"
        assert "patch_match_stereo" in facts["note"]
        assert "fake reconstruction" in facts["note"]
        assert calls == [], "the dense chain must not run on a failed probe"

    def test_single_camera_skips(self, tmp_path, monkeypatch):
        """Stereo matching needs at least two registered cameras."""
        _patch_probe(monkeypatch)
        _patch_run(monkeypatch, tmp_path)

        facts = _stage(tmp_path, monkeypatch, result=_fake_result(n_cameras=1))
        assert facts["status"] == "skipped"
        assert ">=2" in facts["note"]


# ------------------------------------------------------------- happy path

class TestDenseMVSHappyPath:
    def test_real_chain_fills_worldir_with_pointcloud_geometry(
        self, tmp_path, monkeypatch
    ):
        _patch_probe(monkeypatch)
        calls = []
        _patch_run(monkeypatch, tmp_path, calls=calls)

        import engine.pipeline.vertical_slice as vs
        from world_ir.world_v1 import WorldIR

        world = WorldIR(id="w-test")
        options = _options(tmp_path)
        facts = vs._dense_mvs_stage(_fake_result(), world, options, "relative", None)

        assert facts["status"] == "ran"
        assert facts["n_fused_points"] == 64
        # The geometry is IN the world (not merely reported).
        geom = world.geometries[facts["geometry_id"]]
        assert geom.type.value == "pointcloud"
        assert geom.data_uri and geom.data_hash
        assert geom.vertex_count == 64
        assert geom.bounds_min is not None and geom.bounds_max is not None
        # And it is retrievable bytes, content-addressed.
        payload = options.artifact_store.get(geom.data_uri)
        from world_ir.artifact_store import _digest

        assert _digest(payload) == geom.data_hash

    def test_run_facts_are_recorded_on_the_observation(self, tmp_path, monkeypatch):
        """geom_consistency / use_gpu / stage timings belong next to the
        artifact so the dense claim stays auditable."""
        _patch_probe(monkeypatch)
        _patch_run(monkeypatch, tmp_path)

        import engine.pipeline.vertical_slice as vs
        from world_ir.world_v1 import WorldIR

        world = WorldIR(id="w-test")
        facts = vs._dense_mvs_stage(
            _fake_result(), world, _options(tmp_path), "relative", None
        )
        obs = world.geometries[facts["geometry_id"]].observations[0]

        assert obs.sensor_type == "dense_mvs_fused"
        assert obs.metadata["geom_consistency"] is True
        assert obs.metadata["use_gpu"] is None  # left to COLMAP, recorded as such
        assert obs.metadata["stage_durations_s"]["patch_match_stereo"] == 402.0
        assert obs.metadata["scale_state"] == "relative"
        # The frames that produced the geometry are named on the record.
        assert obs.metadata["source_evidence_ids"] == ["img-0", "img-1", "img-2"]

    def test_runner_receives_the_named_capture_paths(self, tmp_path, monkeypatch):
        _patch_probe(monkeypatch)
        calls = []
        _patch_run(monkeypatch, tmp_path, calls=calls)

        import engine.pipeline.vertical_slice as vs
        from world_ir.world_v1 import WorldIR

        options = _options(tmp_path, dense_mvs_use_gpu=False)
        vs._dense_mvs_stage(_fake_result(), WorldIR(id="w"), options, "relative", None)

        assert len(calls) == 1
        call = calls[0]
        assert call["image_dir"] == options.dense_mvs_image_dir
        assert call["sparse_model_dir"] == options.dense_mvs_sparse_model_dir
        assert call["workspace"] == options.dense_mvs_workspace
        assert call["binary"] == "fake-colmap"
        assert call["use_gpu"] is False

    def test_options_are_dataclass_defaults_not_hidden_state(self):
        """The stage is configurable through VerticalSliceOptions alone
        (no globals, no env lookups, no module-level switches)."""
        from engine.pipeline.vertical_slice import VerticalSliceOptions

        o = VerticalSliceOptions()
        assert o.dense_mvs_enabled is False
        assert o.dense_mvs_image_dir is None
        assert o.dense_mvs_sparse_model_dir is None
        assert o.dense_mvs_workspace is None
        assert o.dense_mvs_use_gpu is None


# --------------------------------------------------------- failure path

class TestDenseMVSFailurePath:
    def test_failed_chain_reports_the_error_class(self, tmp_path, monkeypatch):
        """A COLMAP failure is reported as a failure, with the error class
        named -- never swallowed into a silent success."""
        _patch_probe(monkeypatch)
        _patch_run(
            monkeypatch,
            tmp_path,
            error=DenseMVSRunError("COLMAP patch_match_stereo failed (exit 1): boom"),
        )

        import engine.pipeline.vertical_slice as vs
        from world_ir.world_v1 import WorldIR

        world = WorldIR(id="w-test")
        facts = vs._dense_mvs_stage(
            _fake_result(), world, _options(tmp_path), "relative", None
        )

        assert facts["status"] == "failed"
        assert "DenseMVSRunError" in facts["note"]
        assert "boom" in facts["note"]
        # And NOTHING was invented to cover the failure.
        assert world.geometries == {}
        assert world.entities == {}

    def test_runner_unavailability_is_a_failure_not_a_fallback_cloud(
        self, tmp_path, monkeypatch
    ):
        """A declared-unavailable dense chain must not silently degrade into
        a fabricated stand-in cloud. The caller may choose a fallback
        backend; this stage does not pretend one ran."""
        from reconstruction.dense_pipeline import DenseMVSUnavailableError

        _patch_probe(monkeypatch)
        _patch_run(monkeypatch, tmp_path, error=DenseMVSUnavailableError("no GPU device"))

        import engine.pipeline.vertical_slice as vs
        from world_ir.world_v1 import WorldIR

        world = WorldIR(id="w-test")
        facts = vs._dense_mvs_stage(
            _fake_result(), world, _options(tmp_path), "relative", None
        )
        assert facts["status"] == "failed"
        assert "DenseMVSUnavailableError" in facts["note"]
        assert "no GPU device" in facts["note"]
        assert world.geometries == {}

    def test_missing_fused_ply_on_disk_is_a_failure(self, tmp_path, monkeypatch):
        """The runner's reported path must actually exist; a claimed-but-
        absent artifact is a failure, not an empty success."""
        _patch_probe(monkeypatch)
        absent = tmp_path / "dense" / "never_written.ply"

        def fake_run(**kwargs):
            return DenseMVSRun(
                fused_ply_path=str(absent),
                n_fused_points=0,
                stage_durations_s={},
                geom_consistency=False,
                use_gpu=kwargs.get("use_gpu"),
            )

        monkeypatch.setattr("reconstruction.dense_pipeline.run_dense_mvs", fake_run)

        import engine.pipeline.vertical_slice as vs
        from world_ir.world_v1 import WorldIR

        world = WorldIR(id="w-test")
        facts = vs._dense_mvs_stage(
            _fake_result(), world, _options(tmp_path), "relative", None
        )
        assert facts["status"] == "failed"
        assert "never_written.ply" in facts["note"]
        assert world.geometries == {}

    def test_zero_point_fused_ply_is_refused_not_recorded(self, tmp_path, monkeypatch):
        """An empty cloud is refused by the ingest (the same rule the
        standalone ingest tests pin) -- so an empty dense run cannot be
        recorded as a zero-point 'success'."""
        header = (
            b"ply\nformat binary_little_endian 1.0\nelement vertex 0\n"
            b"property float x\nproperty float y\nproperty float z\n"
            b"end_header\n"
        )
        _patch_probe(monkeypatch)
        _patch_run(monkeypatch, tmp_path, fused_bytes=header)

        import engine.pipeline.vertical_slice as vs
        from world_ir.world_v1 import WorldIR

        world = WorldIR(id="w-test")
        facts = vs._dense_mvs_stage(
            _fake_result(), world, _options(tmp_path), "relative", None
        )
        assert facts["status"] == "failed"
        assert "DenseOutputError" in facts["note"]
        assert world.geometries == {}


# ------------------------------------------------------- scale contract

class TestDenseMVSScaleContract:
    """fused.ply is in COLMAP's OWN SfM scale. The stage may rescale it
    into meters only from a MEASURED metric anchor; every other case keeps
    the SfM coordinates and says so on the record."""

    def _extent(self, world, geometry_id, axis=0):
        geom = world.geometries[geometry_id]
        lo = [geom.bounds_min.x, geom.bounds_min.y, geom.bounds_min.z][axis]
        hi = [geom.bounds_max.x, geom.bounds_max.y, geom.bounds_max.z][axis]
        return hi - lo

    def test_relative_world_keeps_sfm_scale_and_says_so(
        self, tmp_path, monkeypatch
    ):
        _patch_probe(monkeypatch)
        _patch_run(monkeypatch, tmp_path)

        facts = _stage(tmp_path, monkeypatch, scale_state="relative")

        assert facts["status"] == "ran"
        assert facts["scale_factor"] == 1.0
        assert "relative" in facts["scale_note"]
        assert "never claimed as meters" in facts["scale_note"]

    def test_metric_anchor_rescales_positions_and_the_artifact(
        self, tmp_path, monkeypatch
    ):
        """A measured meters_per_unit must change BOTH the recorded bounds
        and the persisted bytes -- not merely a label."""
        _patch_probe(monkeypatch)
        _patch_run(monkeypatch, tmp_path)

        import engine.pipeline.vertical_slice as vs
        from world_ir.geometry_data import PointCloudData
        from world_ir.world_v1 import WorldIR

        options = _options(tmp_path)  # ONE store shared by both runs

        unscaled_world = WorldIR(id="w-unscaled")
        unscaled = vs._dense_mvs_stage(
            _fake_result(), unscaled_world, options, "relative", None
        )

        scaled_world = WorldIR(id="w-scaled")
        scaled = vs._dense_mvs_stage(
            _fake_result(), scaled_world, options, "metric", 2.0
        )

        assert scaled["scale_factor"] == 2.0
        assert "measured metric anchor" in scaled["scale_note"]

        ext_unscaled = self._extent(unscaled_world, unscaled["geometry_id"])
        ext_scaled = self._extent(scaled_world, scaled["geometry_id"])
        assert ext_scaled == pytest.approx(2.0 * ext_unscaled, rel=1e-6)

        # The persisted artifact really is rescaled (not just the bounds).
        store = options.artifact_store
        plain = PointCloudData.from_bytes(store.get(unscaled["artifact_uri"]))
        rescaled = PointCloudData.from_bytes(store.get(scaled["artifact_uri"]))
        plain_max = max(p[0] for p in plain.points)
        assert plain_max > 1.0  # the fixture really spans the plane
        assert max(p[0] for p in rescaled.points) == pytest.approx(
            2.0 * plain_max, rel=1e-9
        )

    def test_metric_label_without_a_value_does_not_invent_meters(
        self, tmp_path, monkeypatch
    ):
        """A metric *label* with no measured value is not an anchor."""
        _patch_probe(monkeypatch)
        _patch_run(monkeypatch, tmp_path)

        facts = _stage(tmp_path, monkeypatch, scale_state="metric", meters_per_unit=None)

        assert facts["status"] == "ran"
        assert facts["scale_factor"] == 1.0
        assert "never claimed as meters" in facts["scale_note"]

    def test_zero_meters_per_unit_is_not_treated_as_an_anchor(
        self, tmp_path, monkeypatch
    ):
        """A non-positive anchor is meaningless: falling back to 1.0 keeps
        the record honest instead of collapsing the cloud to a point."""
        _patch_probe(monkeypatch)
        _patch_run(monkeypatch, tmp_path)

        facts = _stage(tmp_path, monkeypatch, scale_state="metric", meters_per_unit=0.0)

        assert facts["status"] == "ran"
        assert facts["scale_factor"] == 1.0

    def test_scale_factor_is_recorded_on_the_observation(
        self, tmp_path, monkeypatch
    ):
        """The rescale must be traceable from the WorldIR record itself."""
        _patch_probe(monkeypatch)
        _patch_run(monkeypatch, tmp_path)

        import engine.pipeline.vertical_slice as vs
        from world_ir.world_v1 import WorldIR

        world = WorldIR(id="w-scaled")
        facts = vs._dense_mvs_stage(
            _fake_result(), world, _options(tmp_path), "metric", 1.5
        )
        obs = world.geometries[facts["geometry_id"]].observations[0]

        assert obs.metadata["scale_factor"] == 1.5
        assert obs.metadata["scale_state"] == "metric"
