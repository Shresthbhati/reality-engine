"""Synthetic VIO inputs: a smooth camera/IMU trajectory with ground truth, IMU noise/bias, a sensor clock with offset and
skew, dropped frames, and a STAND-IN backend executable that exercises the real subprocess adapters.

SYNTHETIC VIO PIPELINE VERIFIED -- REAL BACKEND / HARDWARE VERIFICATION PENDING. This module feeds the VIO PIPELINE
(adapter -> TUM parse -> clock normalisation -> frame handling -> diagnostics -> WorldIR). It is not a VIO algorithm
and says nothing about ORB-SLAM3 / OpenVINS / Basalt, nor about a real IMU.

The trajectory is a closed-form figure-eight (position, velocity and acceleration are analytic), so the IMU is
derived, not guessed: ``gyro = omega_body``, ``accel = R^T (a - g)`` (specific force), with ``g = (0, 0, -9.81)``.
``dead_reckon`` integrates it back, which is how the tests prove the generator is physically consistent (ideal IMU ->
the true path; EuRoC-class noise and bias -> drift that grows with time).

Clocks: the sensor stamps frames with ``t_sensor = (t_global - offset) / (1 + skew)``, so the matching
``evidence.clocks.ClockModel`` is ``t_global = a * t_sensor + b`` with ``a = 1 + skew`` and ``b = offset``.

The stand-in (``python -m synthetic.vio standin ...``, installed on PATH by ``install_standin``) reads the sequence's
ground truth and writes a TUM trajectory in the SENSOR clock, in a gauge-fixed frame whose origin is the first pose
(as a monocular-inertial VIO reports), with a configurable drift. Modes: ok | crash | garbage | empty | lost.
"""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from synthetic import SYNTHETIC_VIO_LABEL

REAL_PENDING = "REAL BACKEND / HARDWARE VERIFICATION PENDING"
GRAVITY = np.array([0.0, 0.0, -9.81])


# ----------------------------------------------------------------------------------------------- rotations

def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def _ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1.0, 0], [-s, 0, c]])


def _rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])


def _log(R) -> np.ndarray:
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    th = np.arccos(c)
    if th < 1e-10:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2.0 * np.sin(th))
    return w * th


def _exp(w) -> np.ndarray:
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def _quat_wxyz(R) -> np.ndarray:
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q = np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s])
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s])
    q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def _quat_to_R(q) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


# ----------------------------------------------------------------------------------------------- the path

@dataclass(frozen=True)
class Path3D:
    """Figure-eight with gentle vertical and attitude oscillation. Body frame: x forward, y left, z up."""

    A: float = 3.0
    B: float = 1.5
    C: float = 0.3
    period_s: float = 20.0
    z0: float = 1.2

    @property
    def w(self) -> float:
        return 2 * np.pi / self.period_s

    def position(self, t):
        w = self.w
        return np.array([self.A * np.sin(w * t), self.B * np.sin(2 * w * t), self.z0 + self.C * np.sin(w * t / 2)])

    def velocity(self, t):
        w = self.w
        return np.array([self.A * w * np.cos(w * t), 2 * self.B * w * np.cos(2 * w * t),
                         self.C * (w / 2) * np.cos(w * t / 2)])

    def acceleration(self, t):
        w = self.w
        return np.array([-self.A * w * w * np.sin(w * t), -4 * self.B * w * w * np.sin(2 * w * t),
                         -self.C * (w / 2) ** 2 * np.sin(w * t / 2)])

    def rotation(self, t):
        v = self.velocity(t)
        yaw = np.arctan2(v[1], v[0])
        return _rz(yaw) @ _ry(0.08 * np.sin(0.5 * self.w * t)) @ _rx(0.10 * np.sin(0.7 * self.w * t))

    def angular_velocity_body(self, t, h: float = 1e-4):
        return _log(self.rotation(t - h).T @ self.rotation(t + h)) / (2 * h)

    def specific_force_body(self, t):
        return self.rotation(t).T @ (self.acceleration(t) - GRAVITY)


@dataclass(frozen=True)
class ImuNoise:
    """EuRoC-class MEMS values (continuous-time densities). ``ideal()`` is all zeros."""

    gyro_noise_density: float = 1.7e-4      # rad / s / sqrt(Hz)
    gyro_bias_walk: float = 1.9e-5          # rad / s^2 / sqrt(Hz)
    accel_noise_density: float = 2.0e-3     # m / s^2 / sqrt(Hz)
    accel_bias_walk: float = 3.0e-3         # m / s^3 / sqrt(Hz)
    gyro_bias0: Tuple[float, float, float] = (0.002, -0.001, 0.0015)
    accel_bias0: Tuple[float, float, float] = (0.02, -0.03, 0.01)

    @staticmethod
    def ideal() -> "ImuNoise":
        return ImuNoise(0.0, 0.0, 0.0, 0.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))


@dataclass(frozen=True)
class ClockSpec:
    """The sensor clock against the global timeline: ``t_global = (1 + skew_ppm*1e-6) * t_sensor + offset_s``."""

    offset_s: float = 0.0123
    skew_ppm: float = 35.0

    @property
    def a(self) -> float:
        return 1.0 + self.skew_ppm * 1e-6

    @property
    def b_ns(self) -> float:
        return self.offset_s * 1e9


@dataclass
class VioSequence:
    path: Path3D
    cam_t_global_ns: np.ndarray          # every camera tick, before frames are dropped
    kept: np.ndarray                     # bool, False = the frame was dropped
    gt_pos: np.ndarray                   # (N, 3) at every camera tick
    gt_quat: np.ndarray                  # (N, 4) wxyz, body -> world
    imu_t_global_ns: np.ndarray
    imu_gyro: np.ndarray                 # measured (noise + bias)
    imu_accel: np.ndarray
    imu_gyro_true: np.ndarray
    imu_accel_true: np.ndarray
    imu_valid: np.ndarray                # False inside a dropout window (those rows are NOT written)
    clock: ClockSpec
    noise: ImuNoise
    seed: int
    dropped_indices: List[int] = field(default_factory=list)

    @property
    def cam_t_sensor_ns(self) -> np.ndarray:
        return np.rint((self.cam_t_global_ns - self.clock.b_ns) / self.clock.a).astype(np.int64)

    @property
    def imu_t_sensor_ns(self) -> np.ndarray:
        return np.rint((self.imu_t_global_ns - self.clock.b_ns) / self.clock.a).astype(np.int64)


def generate(*, duration_s: float = 20.0, cam_hz: float = 20.0, imu_hz: float = 200.0, seed: int = 0,
             path: Path3D = Path3D(), noise: ImuNoise = ImuNoise(), clock: ClockSpec = ClockSpec(),
             drop_fraction: float = 0.03, drop_burst: Tuple[int, int] = (150, 10),
             imu_dropout_s: Optional[Tuple[float, float]] = (8.0, 0.2), t0_global_s: float = 100.0) -> VioSequence:
    """``drop_burst=(first_index, length)`` drops that run of frames on top of the random ``drop_fraction``;
    ``imu_dropout_s=(start_s, length_s)`` removes IMU samples in that window."""
    rng = np.random.default_rng(seed)
    n_cam = int(round(duration_s * cam_hz)) + 1
    t_cam = np.arange(n_cam) / cam_hz
    cam_ns = np.rint((t0_global_s + t_cam) * 1e9).astype(np.int64)
    pos = np.array([path.position(t) for t in t_cam])
    quat = np.array([_quat_wxyz(path.rotation(t)) for t in t_cam])
    kept = np.ones(n_cam, dtype=bool)
    kept[rng.random(n_cam) < drop_fraction] = False
    if drop_burst and drop_burst[0] < n_cam:
        kept[drop_burst[0]:drop_burst[0] + drop_burst[1]] = False
    kept[0] = True                                     # the first pose defines the gauge: never dropped
    n_imu = int(round(duration_s * imu_hz)) + 1
    t_imu = np.arange(n_imu) / imu_hz
    dt = 1.0 / imu_hz
    gy_true = np.array([path.angular_velocity_body(t) for t in t_imu])
    ac_true = np.array([path.specific_force_body(t) for t in t_imu])
    bg, ba = np.array(noise.gyro_bias0, dtype=float), np.array(noise.accel_bias0, dtype=float)
    gyro, accel = np.empty_like(gy_true), np.empty_like(ac_true)
    for k in range(n_imu):
        gyro[k] = gy_true[k] + bg + rng.normal(0.0, noise.gyro_noise_density / np.sqrt(dt), 3)
        accel[k] = ac_true[k] + ba + rng.normal(0.0, noise.accel_noise_density / np.sqrt(dt), 3)
        bg = bg + rng.normal(0.0, noise.gyro_bias_walk * np.sqrt(dt), 3)
        ba = ba + rng.normal(0.0, noise.accel_bias_walk * np.sqrt(dt), 3)
    valid = np.ones(n_imu, dtype=bool)
    if imu_dropout_s:
        valid[(t_imu >= imu_dropout_s[0]) & (t_imu < imu_dropout_s[0] + imu_dropout_s[1])] = False
    imu_ns = np.rint((t0_global_s + t_imu) * 1e9).astype(np.int64)
    return VioSequence(path, cam_ns, kept, pos, quat, imu_ns, gyro, accel, gy_true, ac_true, valid, clock, noise, seed,
                       dropped_indices=[int(i) for i in np.nonzero(~kept)[0]])


def dead_reckon(t_s: np.ndarray, gyro: np.ndarray, accel: np.ndarray, R0, v0, p0) -> np.ndarray:
    """Strapdown integration of ``(gyro, accel)`` from an initial state: returns positions (len(t), 3). Zero-order
    hold between samples, exact rotation update. Deliberately plain: it exists to check the generator, not to be a VIO."""
    R, v, p = np.array(R0, dtype=float), np.array(v0, dtype=float), np.array(p0, dtype=float)
    out = [p.copy()]
    for k in range(len(t_s) - 1):
        dt = t_s[k + 1] - t_s[k]
        a_w = R @ accel[k] + GRAVITY
        p = p + v * dt + 0.5 * a_w * dt * dt
        v = v + a_w * dt
        R = R @ _exp(gyro[k] * dt)
        out.append(p.copy())
    return np.array(out)


# ----------------------------------------------------------------------------------------------- files

def write_sequence(directory, seq: VioSequence) -> Path:
    """Write the sequence the way a capture would be laid out:

        images/<sensor_ns>.png    one tiny image per KEPT frame (the adapters take a directory; they do not read it here)
        imu.csv                   EuRoC-style ``#timestamp [ns],w_x,w_y,w_z,a_x,a_y,a_z`` in the SENSOR clock, dropout rows absent
        ground_truth.tum          every camera tick, GLOBAL clock (seconds), body -> world
        sequence.json             clock, seed, dropped frames, labels
    """
    from PIL import Image

    root = Path(directory)
    (root / "images").mkdir(parents=True, exist_ok=True)
    cam_s = seq.cam_t_sensor_ns
    for i in np.nonzero(seq.kept)[0]:
        Image.new("L", (8, 8), color=int(i) % 256).save(root / "images" / f"{int(cam_s[i])}.png")
    imu_s = seq.imu_t_sensor_ns
    with open(root / "imu.csv", "w", encoding="utf-8") as f:
        f.write("#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],"
                "a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]\n")
        for k in np.nonzero(seq.imu_valid)[0]:
            g, a = seq.imu_gyro[k], seq.imu_accel[k]
            f.write(f"{int(imu_s[k])},{g[0]:.9f},{g[1]:.9f},{g[2]:.9f},{a[0]:.9f},{a[1]:.9f},{a[2]:.9f}\n")
    with open(root / "ground_truth.tum", "w", encoding="utf-8") as f:
        for i in range(len(seq.cam_t_global_ns)):
            p, q = seq.gt_pos[i], seq.gt_quat[i]
            f.write(f"{seq.cam_t_global_ns[i] / 1e9:.9f} {p[0]:.9f} {p[1]:.9f} {p[2]:.9f} "
                    f"{q[1]:.9f} {q[2]:.9f} {q[3]:.9f} {q[0]:.9f}\n")
    (root / "sequence.json").write_text(json.dumps({
        "label": SYNTHETIC_VIO_LABEL, "pending": REAL_PENDING, "generator": "synthetic.vio", "seed": seq.seed,
        "clock": {"offset_s": seq.clock.offset_s, "skew_ppm": seq.clock.skew_ppm, "a": seq.clock.a,
                  "b_ns": seq.clock.b_ns},
        "dropped_frame_indices": seq.dropped_indices, "n_camera_ticks": int(len(seq.cam_t_global_ns)),
        "imu_dropout_rows": int((~seq.imu_valid).sum()),
        "note": "synthetic trajectory + IMU; not a physical device and not a VIO result"}, indent=1), encoding="utf-8")
    return root


def clock_model(seq: VioSequence, clock_id: str = "synthetic_cam"):
    """The ``evidence.clocks.ClockModel`` that maps this sequence's sensor stamps (ns) to the global timeline (ns)."""
    from evidence.clocks import ClockModel

    return ClockModel(clock_id=clock_id, a=seq.clock.a, b=seq.clock.b_ns)


def ground_truth_trajectory(seq: VioSequence, *, kept_only: bool = True, frame_source=None):
    """The true body -> world trajectory on the GLOBAL clock, as a canonical ``Trajectory``."""
    from engine.math import Quat, Vec3
    from reconstruction.calibration.transforms import RigidTransform
    from trajectories.trajectory import FrameSource, Trajectory, TrajectoryFrame

    idx = np.nonzero(seq.kept)[0] if kept_only else np.arange(len(seq.cam_t_global_ns))
    frames = tuple(TrajectoryFrame(timestamp_ns=int(seq.cam_t_global_ns[i]), pose=RigidTransform(
        from_frame="body", to_frame="world", rotation=Quat(*[float(c) for c in seq.gt_quat[i]]),
        translation=Vec3(*[float(c) for c in seq.gt_pos[i]]))) for i in idx)
    return Trajectory(frames=frames, frame_source=frame_source or FrameSource.UNKNOWN,
                      provenance="synthetic ground truth")


# ----------------------------------------------------------------------------------------------- stand-in executable

def _read_tum(path: Path):
    rows = np.loadtxt(path, comments="#")
    return rows[:, 0], rows[:, 1:4], rows[:, 4:8]       # t, xyz, qx qy qz qw


def standin_main(argv: Optional[Sequence[str]] = None) -> int:
    """The stand-in 'VIO binary'. It does not estimate anything from the images or IMU: it perturbs the sequence's
    ground truth (drift + noise), expresses it in a first-pose-relative frame and stamps it in the SENSOR clock."""
    ap = argparse.ArgumentParser(prog="synthetic.vio standin")
    ap.add_argument("--images", required=True)
    ap.add_argument("--imu", default="")
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--mode", default="ok", choices=["ok", "crash", "garbage", "empty", "lost"])
    ap.add_argument("--drift-m-per-s", type=float, default=0.01)
    ap.add_argument("--noise-m", type=float, default=0.004)
    ap.add_argument("--out-name", default="KeyFrameTrajectory.txt")
    args = ap.parse_args(list(argv) if argv is not None else None)
    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)
    out = work / args.out_name
    if args.mode == "crash":
        sys.stderr.write("SYNTHETIC STAND-IN: tracking initialisation failed (injected)\n")
        return 3
    if args.mode == "empty":
        out.write_text("# no trajectory rows\n", encoding="utf-8")
        return 0
    if args.mode == "garbage":
        out.write_text("1.0 0.0 0.0 not-a-number 0 0 0 1\n", encoding="utf-8")
        return 0
    seq_dir = Path(args.images).parent
    meta = json.loads((seq_dir / "sequence.json").read_text(encoding="utf-8"))
    t_g, pos, qxyzw = _read_tum(seq_dir / "ground_truth.tum")
    kept_sensor = np.array(sorted(int(Path(p).stem) for p in os.listdir(args.images)), dtype=np.int64)
    a, b_ns = meta["clock"]["a"], meta["clock"]["b_ns"]
    calc = np.rint((t_g * 1e9 - b_ns) / a).astype(np.int64)
    near = np.clip(np.searchsorted(kept_sensor, calc), 1, len(kept_sensor) - 1)
    pick = np.where(np.abs(kept_sensor[near - 1] - calc) <= np.abs(kept_sensor[near] - calc), near - 1, near)
    sel = np.abs(kept_sensor[pick] - calc) <= 2            # a camera tick is kept iff its image file exists
    t_sensor_ns = kept_sensor[pick]                        # exact stamp of the image (not a recomputed one)
    if args.mode == "lost":                                        # tracking lost for a stretch: no poses at all there
        idx = np.nonzero(sel)[0]
        sel[idx[len(idx) // 3: len(idx) // 3 + len(idx) // 5]] = False
    rng = np.random.default_rng(meta["seed"] + 1)
    t0 = t_g[0]
    direction = np.array([0.8, 0.5, 0.33])
    direction /= np.linalg.norm(direction)
    R0 = _quat_to_R(np.array([qxyzw[0, 3], qxyzw[0, 0], qxyzw[0, 1], qxyzw[0, 2]]))
    lines = []
    for i in np.nonzero(sel)[0]:
        t = t_g[i] - t0
        noise = rng.normal(0.0, args.noise_m, 3) if i != 0 else np.zeros(3)   # the first pose DEFINES the gauge
        p_est = pos[i] + direction * args.drift_m_per_s * t + noise
        R = _quat_to_R(np.array([qxyzw[i, 3], qxyzw[i, 0], qxyzw[i, 1], qxyzw[i, 2]]))
        R = _rz(np.radians(0.1) * t) @ R                              # slow yaw drift
        p_rel = R0.T @ (p_est - pos[0])                               # gauge-fixed: first pose is the origin
        q = _quat_wxyz(R0.T @ R)
        lines.append(f"{t_sensor_ns[i] / 1e9:.9f} {p_rel[0]:.9f} {p_rel[1]:.9f} {p_rel[2]:.9f} "
                     f"{q[1]:.9f} {q[2]:.9f} {q[3]:.9f} {q[0]:.9f}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


def install_standin(bin_dir, *names: str) -> Path:
    """Write a launcher for the stand-in under each of ``names`` (the binary names the real adapters look up) in
    ``bin_dir`` and return it, to be prepended to PATH. Windows gets ``.cmd`` shims, POSIX an executable ``sh``."""
    bin_dir = Path(bin_dir)
    bin_dir.mkdir(parents=True, exist_ok=True)
    repo = str(Path(__file__).resolve().parents[1])
    for name in names:
        if os.name == "nt":
            (bin_dir / f"{name}.cmd").write_text(
                f'@echo off\r\nset "PYTHONPATH={repo};%PYTHONPATH%"\r\n"{sys.executable}" -m synthetic.vio standin %*\r\n',
                encoding="utf-8")
        else:
            p = bin_dir / name
            p.write_text(f'#!/bin/sh\nPYTHONPATH="{repo}:$PYTHONPATH" exec "{sys.executable}" -m synthetic.vio '
                         f'standin "$@"\n', encoding="utf-8")
            p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return bin_dir


def launcher_path(bin_dir, name: str) -> str:
    return str(Path(bin_dir) / (f"{name}.cmd" if os.name == "nt" else name))


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "standin":
        raise SystemExit(standin_main(sys.argv[2:]))
    raise SystemExit("usage: python -m synthetic.vio standin --images DIR --workdir DIR [--mode ok|crash|...]")
