"""SYNTHETIC VIO PIPELINE VERIFIED -- REAL BACKEND / HARDWARE VERIFICATION PENDING.

A deterministic camera/IMU sequence (analytic trajectory, EuRoC-class IMU noise and bias, a sensor clock with offset
and skew, dropped frames, an IMU dropout) is fed through the REAL VIO pipeline: the subprocess adapters
(``SubprocessTrajectoryBackend``: PATH discovery, command rendering, owned subprocess, TUM parsing), backend
selection and fallback, clock normalisation (``ClockModel``), frame handling (``FrameGraph``), diagnostics
(ATE / drift) and the WorldIR attachment. The 'binary' behind the adapters is a STAND-IN that perturbs ground truth
(``synthetic.vio``): this verifies the pipeline around a VIO backend, never a VIO algorithm. No claim is made about
ORB-SLAM3 / OpenVINS / Basalt or a real IMU.
"""

from __future__ import annotations

import dataclasses
import json
import os

import numpy as np
import pytest

from synthetic.vio import (
    ImuNoise, clock_model, dead_reckon, generate, ground_truth_trajectory, install_standin, launcher_path,
    write_sequence,
)
from trajectories.backend.interface import (
    TrajectoryBackendRunError, TrajectoryBackendUnavailableError, TrajectoryEstimationRequest,
)
from trajectories.backend.openvins_backend import OpenVINSBackend
from trajectories.backend.orb_slam3_backend import ORBSLAM3Backend
from trajectories.backend.selection import DEFAULT_BACKENDS, estimate_trajectory
from trajectories.trajectory import FrameSource

ORB_BIN, OV_BIN = "mono_inertial_euroc", "run_subscribe_msckf"


# ------------------------------------------------------------------------------------------------- the generator

def _ideal(seed=1):
    return generate(seed=seed, noise=ImuNoise.ideal(), imu_dropout_s=None)


def test_ideal_imu_dead_reckons_back_to_the_true_path():
    seq = _ideal()
    t = (seq.imu_t_global_ns - seq.imu_t_global_ns[0]) / 1e9
    est = dead_reckon(t, seq.imu_gyro, seq.imu_accel, seq.path.rotation(0.0), seq.path.velocity(0.0),
                      seq.path.position(0.0))
    gt = np.array([seq.path.position(x) for x in t])
    err = np.linalg.norm(est - gt, axis=1)
    path_len = float(np.linalg.norm(np.diff(gt, axis=0), axis=1).sum())
    assert err.max() < 0.2 and err.max() < 0.01 * path_len, (err.max(), path_len)   # < 1% of a ~40 m path


def test_gravity_sign_and_magnitude_are_physical():
    seq = _ideal()
    fz = seq.imu_accel_true[:, 2]
    assert 9.4 < float(fz.mean()) < 10.2, "a level IMU reads +g on its up axis (specific force)"
    assert np.linalg.norm(seq.imu_accel_true, axis=1).mean() == pytest.approx(9.81, abs=0.5)


def test_noise_and_bias_make_dead_reckoning_drift_and_drift_grows_with_time():
    ideal, noisy = _ideal(), generate(seed=1, imu_dropout_s=None)
    t = (noisy.imu_t_global_ns - noisy.imu_t_global_ns[0]) / 1e9
    gt = np.array([noisy.path.position(x) for x in t])
    init = (noisy.path.rotation(0.0), noisy.path.velocity(0.0), noisy.path.position(0.0))
    e_ideal = np.linalg.norm(dead_reckon(t, ideal.imu_gyro, ideal.imu_accel, *init) - gt, axis=1)
    e_noisy = np.linalg.norm(dead_reckon(t, noisy.imu_gyro, noisy.imu_accel, *init) - gt, axis=1)
    assert e_noisy[-1] > 10 * e_ideal[-1] and e_noisy[len(t) // 2] < e_noisy[-1]


def test_ideal_noise_means_measured_equals_true_and_seeds_are_deterministic():
    seq = _ideal()
    assert np.array_equal(seq.imu_gyro, seq.imu_gyro_true) and np.array_equal(seq.imu_accel, seq.imu_accel_true)
    a, b, c = generate(seed=3), generate(seed=3), generate(seed=4)
    assert np.array_equal(a.imu_gyro, b.imu_gyro) and np.array_equal(a.kept, b.kept)
    assert not np.array_equal(a.imu_gyro, c.imu_gyro)


def test_sensor_clock_maps_back_to_the_global_timeline_within_a_nanosecond():
    seq = generate(seed=2)
    model = clock_model(seq)
    back = np.array([model.apply(float(s)) for s in seq.cam_t_sensor_ns])
    assert np.abs(back - seq.cam_t_global_ns).max() <= 2.0
    off = (seq.cam_t_global_ns - seq.cam_t_sensor_ns).astype(float)        # the stamps really are shifted (not a no-op)
    slope, intercept = np.polyfit(seq.cam_t_sensor_ns.astype(float), off, 1)
    assert slope == pytest.approx(seq.clock.skew_ppm * 1e-6, rel=0.02), "the sensor clock's skew must be visible"
    assert intercept == pytest.approx(seq.clock.offset_s * 1e9, rel=0.02), "and so must its offset"


def test_dropped_frames_and_imu_dropout_are_real_and_logged(tmp_path):
    seq = generate(seed=5)
    root = write_sequence(tmp_path / "seq", seq)
    meta = json.loads((root / "sequence.json").read_text())
    assert meta["label"] == "SYNTHETIC VIO PIPELINE VERIFIED" and "PENDING" in meta["pending"]
    burst = set(range(150, 160))
    assert burst <= set(meta["dropped_frame_indices"]) and 0 not in meta["dropped_frame_indices"]
    assert len(list((root / "images").glob("*.png"))) == int(seq.kept.sum())
    rows = [ln for ln in (root / "imu.csv").read_text().splitlines() if not ln.startswith("#")]
    assert len(rows) == int(seq.imu_valid.sum()) < len(seq.imu_valid)
    stamps = np.array([int(r.split(",")[0]) for r in rows])
    assert (np.diff(stamps) > 0).all() and np.diff(stamps).max() >= 0.19e9, "no IMU gap where the dropout was"


# ----------------------------------------------------------------------------------------------- the real adapters

@pytest.fixture()
def env(tmp_path, monkeypatch):
    seq = generate(seed=7)
    root = write_sequence(tmp_path / "seq", seq)
    bin_dir = install_standin(tmp_path / "bin", ORB_BIN, OV_BIN)
    monkeypatch.setenv("PATH", str(bin_dir) + os.pathsep + os.environ["PATH"])

    def command(name=ORB_BIN, mode="ok", out="KeyFrameTrajectory.txt"):
        return [launcher_path(bin_dir, name), "--images", "{images}", "--imu", "{imu}", "--workdir", "{workdir}",
                "--mode", mode, "--out-name", out]

    def request(mode="ok", name=ORB_BIN, out="KeyFrameTrajectory.txt", **options):
        work = tmp_path / f"work-{mode}-{name}"
        return TrajectoryEstimationRequest(
            images_path=root / "images", imu_path=root / "imu.csv", workdir=work,
            options={"command": command(name, mode, out), "output_file": str(work / out),
                     "world_frame": "session-local", **options})

    return seq, root, request, command


def test_the_standin_is_found_on_path_by_the_real_adapters(env):
    assert ORBSLAM3Backend().is_available() and OpenVINSBackend().is_available()


def test_adapter_runs_the_binary_and_parses_a_vio_trajectory_with_dropped_frames(env):
    seq, _root, request, _ = env
    traj = ORBSLAM3Backend().estimate(request())
    assert traj.frame_source == FrameSource.VIO and "orb_slam3" in traj.provenance
    assert len(traj) == int(seq.kept.sum()), "dropped frames must leave no pose"
    assert [f.timestamp_ns for f in traj.frames] == [int(s) for s in seq.cam_t_sensor_ns[seq.kept]]
    first = traj.frames[0].pose
    assert np.allclose([first.translation.x, first.translation.y, first.translation.z], 0.0, atol=1e-6), \
        "a VIO trajectory is gauge-fixed: its first pose is the origin"
    gaps = np.diff([f.timestamp_ns for f in traj.frames])
    assert gaps.max() >= 0.5e9, "the 10-frame burst must appear as a time gap"


def test_selection_normalises_time_onto_the_global_clock_and_keeps_the_originals(env):
    seq, _root, request, _ = env
    traj, attempts = estimate_trajectory(request(), clock_model=clock_model(seq))
    assert traj.sync_state == "SYNCHRONIZED" and traj.clock_id == "synthetic_cam"
    want = seq.cam_t_global_ns[seq.kept]
    assert np.abs(np.array([f.timestamp_ns for f in traj.frames]) - want).max() <= 2
    assert [f.sensor_timestamp_ns for f in traj.frames] == [int(s) for s in seq.cam_t_sensor_ns[seq.kept]]
    assert attempts[0].backend_name == "orb_slam3" and attempts[0].available and not attempts[0].error
    raw, _ = estimate_trajectory(request())                       # no model: pass-through, nothing claimed
    assert raw.sync_state == "<undeclared>" and raw.frames[5].timestamp_ns != traj.frames[5].timestamp_ns


def _anchored(traj, seq):
    """Express the gauge-fixed estimate in the ground-truth frame: the session frame IS the first pose's body frame, so
    the anchor session-local -> world is the true first pose (what a registration step would supply)."""
    from reconstruction.calibration.transforms import RigidTransform
    from trajectories.trajectory import Trajectory, TrajectoryFrame

    gt0 = ground_truth_trajectory(seq).frames[0].pose
    anchor = RigidTransform(from_frame="session-local", to_frame="world", rotation=gt0.rotation,
                            translation=gt0.translation)
    out = tuple(TrajectoryFrame(timestamp_ns=f.timestamp_ns, pose=f.pose.compose(anchor)) for f in traj.frames)
    return Trajectory(frames=out, frame_source=traj.frame_source, provenance=traj.provenance + " [anchored]")


def test_diagnostics_measure_the_injected_drift_only_after_the_frame_is_anchored(env):
    from trajectories.diagnostics import absolute_trajectory_error, endpoint_drift

    seq, _root, request, _ = env
    traj, _ = estimate_trajectory(request(), clock_model=clock_model(seq))
    gt = ground_truth_trajectory(seq)
    raw = absolute_trajectory_error(traj, gt)
    assert raw.rmse > 1.0, "un-anchored frames are different frames: ATE must not look good by accident"
    anchored = _anchored(traj, seq)
    ate = absolute_trajectory_error(anchored, gt)
    # stand-in drifts 0.01 m/s for 20 s (ramp 0 -> 0.2 m, rmse 0.2/sqrt(3)) + 4 mm noise
    assert ate.rmse == pytest.approx(0.2 / np.sqrt(3), abs=0.025), ate
    assert ate.count == len(traj)
    assert endpoint_drift(anchored) == pytest.approx(0.2, abs=0.03)
    assert endpoint_drift(gt) < 0.05, "the figure-eight is a closed loop: the true endpoint drift is ~0"


def test_trajectory_enters_the_frame_graph_and_worldir_without_a_made_up_registration(env):
    from trajectories.worldir import attach_trajectory
    from world_ir import WorldIR
    from world_ir.coordinates import Frame
    from world_ir.frame_graph import FrameGraph
    from world_ir.validation import validate_world_ir

    seq, _root, request, _ = env
    traj, _ = estimate_trajectory(request(), clock_model=clock_model(seq))
    graph = FrameGraph()
    graph.load_from_trajectory(traj)
    (edge,) = graph.edges()
    assert (edge.transform.source_frame, edge.transform.target_frame) == (Frame.SENSOR, Frame.SESSION_LOCAL)
    assert edge.valid_from_ns == traj.start_ns and edge.valid_to_ns == traj.end_ns
    assert edge.metadata["sync_state"] == "SYNCHRONIZED"

    world = WorldIR(id="vio-world")
    oid = attach_trajectory(world, traj)
    obs = world.observations[oid]
    assert obs.sensor_type == "vio" and obs.frame_id == "session-local"
    assert obs.metadata["registered_into_world"] is False and "registration has not anchored it" in obs.uncertainty.note
    assert "drift NOT estimated" in obs.uncertainty.note, "an unknown drift must read as unknown, not zero"
    assert obs.metadata["n_poses"] == len(traj) and len(obs.metadata["poses"]) == len(traj)
    assert not validate_world_ir(world).errors
    assert attach_trajectory(world, traj) == oid and len(world.observations) == 1, "idempotent"
    back = WorldIR.from_dict(json.loads(json.dumps(world.to_dict(), default=str)))
    assert back.observations[oid].metadata["poses"] == obs.metadata["poses"]

    # once an anchor puts the trajectory in the world's own frame it is recorded as registered
    anchored = _anchored(traj, seq)
    anchored_world = WorldIR(id="vio-world-2")
    aid = attach_trajectory(anchored_world, anchored)
    assert anchored_world.observations[aid].metadata["registered_into_world"] is True


def test_unknown_frame_names_are_refused_not_guessed(env):
    from trajectories.worldir import attach_trajectory
    from world_ir import WorldIR
    from world_ir.frame_graph import FrameGraphInvalidTransformError

    _seq, _root, request, _ = env
    traj = ORBSLAM3Backend().estimate(request(body_frame="rig", world_frame="session-local"))
    with pytest.raises(FrameGraphInvalidTransformError, match="rig"):
        attach_trajectory(WorldIR(id="w"), traj)


# ------------------------------------------------------------------------------------- failures and fallback

@pytest.mark.parametrize("mode,needle", [("crash", "tracking initialisation failed"), ("garbage", "non-numeric"),
                                         ("empty", "no trajectory rows")])
def test_a_failing_backend_raises_a_run_error_carrying_the_real_diagnostic(env, mode, needle):
    _seq, _root, request, _ = env
    with pytest.raises(TrajectoryBackendRunError) as exc:
        ORBSLAM3Backend().estimate(request(mode))
    assert needle in str(exc.value)
    if mode == "crash":
        assert "exit 3" in str(exc.value) and "--mode crash" in str(exc.value)


def test_tracking_lost_leaves_a_gap_not_invented_poses(env):
    _seq, _root, request, _ = env
    ok = ORBSLAM3Backend().estimate(request("ok"))
    lost = ORBSLAM3Backend().estimate(request("lost"))
    assert 0.6 * len(ok) < len(lost) < 0.9 * len(ok)
    assert set(f.timestamp_ns for f in lost.frames) < set(f.timestamp_ns for f in ok.frames)


class _ModeBackend(ORBSLAM3Backend):
    """The REAL adapter code path (PATH check, argv rendering, owned subprocess, TUM parse) with its own command, so
    two backends in one selection can behave differently."""

    def __init__(self, name, binary, command):
        self.name, self.binary_names, self._command = name, (binary,), command

    def estimate(self, request):
        opts = {**request.options, "command": self._command}
        return super().estimate(dataclasses.replace(request, options=opts))


def test_selection_falls_back_to_the_next_backend_when_the_first_crashes(env):
    seq, _root, request, command = env
    first = _ModeBackend("orb_slam3", ORB_BIN, command(ORB_BIN, "crash"))
    second = _ModeBackend("openvins", OV_BIN, command(OV_BIN, "ok", "stamped_traj_estimate.txt"))
    req = request("ok", OV_BIN, "stamped_traj_estimate.txt")
    traj, attempts = estimate_trajectory(req, backends=(first, second), clock_model=clock_model(seq))
    assert [a.backend_name for a in attempts] == ["orb_slam3", "openvins"]
    assert attempts[0].available and "tracking initialisation failed" in attempts[0].error
    assert attempts[1].available and not attempts[1].error
    assert traj.frame_source == FrameSource.VIO and traj.sync_state == "SYNCHRONIZED" and len(traj) > 300


def test_every_backend_crashing_is_a_run_error_that_names_each_failure(env):
    _seq, _root, request, command = env
    backends = (_ModeBackend("orb_slam3", ORB_BIN, command(ORB_BIN, "crash")),
                _ModeBackend("openvins", OV_BIN, command(OV_BIN, "garbage")))
    with pytest.raises(TrajectoryBackendRunError) as exc:
        estimate_trajectory(request(), backends=backends)
    assert "orb_slam3" in str(exc.value) and "openvins" in str(exc.value)
    assert "tracking initialisation failed" in str(exc.value) and "non-numeric" in str(exc.value)


def test_no_binary_installed_is_unavailable_not_a_fabricated_trajectory(tmp_path, monkeypatch):
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    assert not any(b.is_available() for b in DEFAULT_BACKENDS)
    req = TrajectoryEstimationRequest(images_path=tmp_path, workdir=tmp_path / "w",
                                      options={"command": ["x", "{images}"]})
    with pytest.raises(TrajectoryBackendUnavailableError) as exc:
        estimate_trajectory(req)
    assert "no VIO backend available" in str(exc.value)


def test_a_backend_with_no_invocation_supplied_is_an_error_not_a_guessed_command(env):
    _seq, root, _request, _ = env
    with pytest.raises(TrajectoryBackendRunError, match="no invocation supplied"):
        ORBSLAM3Backend().estimate(TrajectoryEstimationRequest(images_path=root / "images", workdir=root / "w"))
