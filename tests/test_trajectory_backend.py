"""Tests for trajectories/backend/ (P3-02 VIO/localization backend
federation): TUM-format parsing (pure function, no binary needed),
adapter contract tests against a fake binary producing a fake
trajectory, and honest BACKEND_UNAVAILABLE behavior for the real
adapters on a machine with none of ORB-SLAM3/OpenVINS/Basalt
installed (verified true on this machine before writing these tests).
"""

from __future__ import annotations

import sys
import textwrap

import pytest

from trajectories.backend.basalt_backend import BasaltBackend
from trajectories.backend.interface import (
    TrajectoryBackendRunError,
    TrajectoryBackendUnavailableError,
    TrajectoryEstimationRequest,
)
from trajectories.backend.openvins_backend import OpenVINSBackend
from trajectories.backend.orb_slam3_backend import ORBSLAM3Backend
from trajectories.backend.selection import (
    BackendAttempt,
    DEFAULT_BACKENDS,
    estimate_trajectory,
)
from trajectories.backend.subprocess_backend import SubprocessTrajectoryBackend
from trajectories.backend.tum_format import parse_tum_trajectory
from trajectories.trajectory import FrameSource


TUM_VALID = textwrap.dedent(
    """\
    # timestamp tx ty tz qx qy qz qw
    0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0
    0.1 1.0 0.0 0.0 0.0 0.0 0.0 1.0
    0.2 2.0 0.0 0.0 0.0 0.0 0.0 1.0
    """
)


class TestTumFormatParsing:
    def test_valid_file_parses(self, tmp_path):
        path = tmp_path / "traj.txt"
        path.write_text(TUM_VALID)
        traj = parse_tum_trajectory(
            path,
            from_frame="body",
            to_frame="world",
            frame_source=FrameSource.VIO,
            provenance="test",
        )
        assert len(traj) == 3
        assert traj.frames[0].timestamp_ns == 0
        assert traj.frames[1].timestamp_ns == 100_000_000
        assert traj.frames[2].pose.translation.x == 2.0
        assert traj.frame_source is FrameSource.VIO

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(TrajectoryBackendRunError, match="not found"):
            parse_tum_trajectory(
                tmp_path / "nope.txt",
                from_frame="body",
                to_frame="world",
                frame_source=FrameSource.VIO,
                provenance="",
            )

    def test_empty_file_raises(self, tmp_path):
        path = tmp_path / "empty.txt"
        path.write_text("# only comments\n\n")
        with pytest.raises(TrajectoryBackendRunError, match="no trajectory rows"):
            parse_tum_trajectory(
                path, from_frame="body", to_frame="world",
                frame_source=FrameSource.VIO, provenance="",
            )

    def test_malformed_line_raises(self, tmp_path):
        path = tmp_path / "bad.txt"
        path.write_text("0.0 1.0 2.0\n")  # only 3 fields, need 8
        with pytest.raises(TrajectoryBackendRunError, match="expected 8 TUM fields"):
            parse_tum_trajectory(
                path, from_frame="body", to_frame="world",
                frame_source=FrameSource.VIO, provenance="",
            )

    def test_non_numeric_field_raises(self, tmp_path):
        path = tmp_path / "bad.txt"
        path.write_text("0.0 x 0.0 0.0 0.0 0.0 0.0 1.0\n")
        with pytest.raises(TrajectoryBackendRunError, match="non-numeric"):
            parse_tum_trajectory(
                path, from_frame="body", to_frame="world",
                frame_source=FrameSource.VIO, provenance="",
            )

    def test_non_unit_quaternion_raises(self, tmp_path):
        path = tmp_path / "bad.txt"
        path.write_text("0.0 0.0 0.0 0.0 0.0 0.0 0.0 2.0\n")  # qw=2, not unit
        with pytest.raises(TrajectoryBackendRunError, match="invalid pose"):
            parse_tum_trajectory(
                path, from_frame="body", to_frame="world",
                frame_source=FrameSource.VIO, provenance="",
            )

    def test_non_monotonic_timestamps_raises(self, tmp_path):
        path = tmp_path / "bad.txt"
        path.write_text(
            "1.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0\n"
            "0.5 1.0 0.0 0.0 0.0 0.0 0.0 1.0\n"
        )
        with pytest.raises(TrajectoryBackendRunError, match="strictly increasing"):
            parse_tum_trajectory(
                path, from_frame="body", to_frame="world",
                frame_source=FrameSource.VIO, provenance="",
            )


class TestRealAdaptersHonestlyUnavailable:
    """None of ORB-SLAM3/OpenVINS/Basalt are installed on this
    machine -- verified with `shutil.which` before writing these
    tests. These assertions are about real environment state, not
    mocked behavior."""

    @pytest.mark.parametrize(
        "backend_cls", [ORBSLAM3Backend, OpenVINSBackend, BasaltBackend]
    )
    def test_is_available_false(self, backend_cls):
        assert backend_cls().is_available() is False

    @pytest.mark.parametrize(
        "backend_cls", [ORBSLAM3Backend, OpenVINSBackend, BasaltBackend]
    )
    def test_estimate_raises_unavailable(self, backend_cls, tmp_path):
        backend = backend_cls()
        request = TrajectoryEstimationRequest(
            images_path=tmp_path / "images", workdir=tmp_path / "work"
        )
        with pytest.raises(TrajectoryBackendUnavailableError, match="PATH"):
            backend.estimate(request)

    def test_selection_raises_unavailable_when_all_absent(self, tmp_path):
        request = TrajectoryEstimationRequest(
            images_path=tmp_path / "images", workdir=tmp_path / "work"
        )
        with pytest.raises(TrajectoryBackendUnavailableError, match="no VIO backend available"):
            estimate_trajectory(request)

    def test_default_backends_federation_order(self):
        names = [b.name for b in DEFAULT_BACKENDS]
        assert names == ["orb_slam3", "openvins", "basalt"]


class _FakeVioBackend(SubprocessTrajectoryBackend):
    """A SubprocessTrajectoryBackend pointed at the current Python
    interpreter (genuinely on PATH) instead of a real VIO binary --
    the adapter CONTRACT test: real subprocess execution, real file
    I/O, real TUM parsing, with a fake trajectory as the payload."""

    name = "fake_vio"
    binary_names = (sys.executable,)
    default_output_filename = "trajectory.txt"
    frame_source = FrameSource.VIO


_WRITE_TUM_SCRIPT = textwrap.dedent(
    """\
    import sys
    out = sys.argv[1]
    with open(out, "w") as f:
        f.write("0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0\\n")
        f.write("0.5 1.0 2.0 3.0 0.0 0.0 0.0 1.0\\n")
    """
)


class TestSubprocessAdapterContract:
    def test_is_available_true_for_python_itself(self):
        assert _FakeVioBackend().is_available() is True

    def test_estimate_without_command_raises_run_error(self, tmp_path):
        backend = _FakeVioBackend()
        request = TrajectoryEstimationRequest(
            images_path=tmp_path / "images", workdir=tmp_path / "work"
        )
        with pytest.raises(TrajectoryBackendRunError, match="no invocation supplied"):
            backend.estimate(request)

    def test_estimate_runs_fake_binary_and_parses_output(self, tmp_path):
        script_path = tmp_path / "write_tum.py"
        script_path.write_text(_WRITE_TUM_SCRIPT)
        workdir = tmp_path / "work"

        backend = _FakeVioBackend()
        request = TrajectoryEstimationRequest(
            images_path=tmp_path / "images",
            workdir=workdir,
            options={
                "command": [
                    sys.executable,
                    str(script_path),
                    "{workdir}/trajectory.txt",
                ],
            },
        )
        traj = backend.estimate(request)
        assert len(traj) == 2
        assert traj.frame_source is FrameSource.VIO
        assert traj.frames[1].timestamp_ns == 500_000_000
        assert traj.frames[1].pose.translation.x == 1.0
        assert "fake_vio" in traj.provenance

    def test_estimate_nonzero_exit_raises_run_error(self, tmp_path):
        workdir = tmp_path / "work"
        backend = _FakeVioBackend()
        request = TrajectoryEstimationRequest(
            images_path=tmp_path / "images",
            workdir=workdir,
            options={"command": [sys.executable, "-c", "import sys; sys.exit(1)"]},
        )
        with pytest.raises(TrajectoryBackendRunError, match=r"failed \(exit 1\)"):
            backend.estimate(request)

    def test_selection_succeeds_with_fake_backend_first(self, tmp_path):
        script_path = tmp_path / "write_tum.py"
        script_path.write_text(_WRITE_TUM_SCRIPT)
        workdir = tmp_path / "work"

        request = TrajectoryEstimationRequest(
            images_path=tmp_path / "images",
            workdir=workdir,
            options={
                "command": [sys.executable, str(script_path), "{workdir}/trajectory.txt"],
            },
        )
        traj, attempts = estimate_trajectory(
            request, backends=(_FakeVioBackend(), ORBSLAM3Backend())
        )
        assert len(traj) == 2
        assert attempts[0] == BackendAttempt("fake_vio", available=True)
        # ORB-SLAM3 was never tried since fake_vio already succeeded.
        assert len(attempts) == 1

    def test_selection_falls_through_on_run_error_to_next_backend(self, tmp_path):
        workdir = tmp_path / "work"
        failing = _FakeVioBackend()
        request = TrajectoryEstimationRequest(
            images_path=tmp_path / "images",
            workdir=workdir,
            options={"command": [sys.executable, "-c", "import sys; sys.exit(1)"]},
        )
        with pytest.raises(TrajectoryBackendRunError, match="every available VIO backend failed"):
            estimate_trajectory(request, backends=(failing,))
