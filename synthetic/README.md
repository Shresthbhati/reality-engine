# synthetic/ -- the deterministic synthetic data pack

Repository-local fixtures for the stages whose REAL inputs (a physical RGB-D device, a built VIO backend, a real indoor
photo set) are not available to every developer. Everything is seeded and deterministic, and returns or writes the
ground truth it was built from.

**What it proves:** the pipeline handles the *contract* -- formats, scales, invalid pixels, clock offsets, dropped
frames, occlusion, partial observation.
**What it does not prove:** anything about a physical sensor, a real VIO algorithm or a real building. The labels below
are used verbatim by the manifests this pack writes and by `engine/core/verification.py` (`reality verification`).

| area | module | label | still pending (external) |
|---|---|---|---|
| RGB-D | `rgbd.py` | SYNTHETIC RGB-D VERIFIED | PHYSICAL DEVICE VERIFICATION PENDING |
| VIO | `vio.py` | SYNTHETIC VIO PIPELINE VERIFIED | REAL BACKEND / HARDWARE VERIFICATION PENDING |
| indoor architecture | `indoor.py` | SYNTHETIC INDOOR VERIFIED | REAL INDOOR DATASET VERIFICATION PENDING |

## RGB-D (`rgbd.py`, `scene.py`)

`render_frame(scene, intrinsics, pose, SensorModel, index)` ray-casts a planar scene and degrades the depth like a
sensor: range-dependent noise (`sigma(z) = sigma0 + sigma_z2 z^2`), dropout holes, glass (no return), range limits,
flying pixels at discontinuities, isolated invalid pixels -- each counted in `invalid_breakdown`.
`write_capture(dir, frames)` writes the capture the real pipeline reads (`reality compile`): `images/`, 16-bit `depth/`
PNGs (raw 0 = invalid) with `depth/manifest.json` (the declared scale), `manifest.json` (intrinsics, a measured
baseline), `ground_truth/`. `declare_depth_scale=False` is the *relative scale only* case (the parser must refuse it);
`measured_baseline=False` leaves the world RELATIVE (device metres must not be unprojected into it).
`GroundTruthBackend` is a test-only reconstruction backend (`REALITY_TEST_BACKEND=synthetic.rgbd:GroundTruthBackend`)
that returns the poses the frames were rendered from. Tests: `tests/test_synthetic_rgbd.py`.

## VIO (`vio.py`)

An analytic figure-eight (position, velocity, acceleration closed-form), so the IMU is *derived*:
`gyro = omega_body`, `accel = R^T (a - g)`; `dead_reckon` integrates it back (the generator's own physical check).
EuRoC-class IMU noise and bias (`ImuNoise`), a sensor clock with offset and skew (`ClockSpec` -> `clock_model(seq)`),
dropped frames, an IMU dropout. `write_sequence` lays it out as a capture. The "backend" is a **stand-in executable**
(`python -m synthetic.vio standin`, put on PATH with `install_standin`) that perturbs ground truth -- drift, noise, a
first-pose gauge, the sensor clock -- so the REAL subprocess adapters (PATH discovery, argv rendering, owned subprocess,
TUM parsing) run end to end. Modes: `ok | crash | garbage | empty | lost`. It is not a VIO algorithm. Tests:
`tests/test_synthetic_vio.py`.

## Indoor architecture (`indoor.py`)

Scene factories with ground truth: `single_room` (door, window, table, pillar), `corridor_with_rooms`,
`two_storey_with_stairs` (stairwell holes through both slabs), `ramp_room`, `tower(heights)` (disconnected slabs).
`observe(scene)` renders panoramas through the RGB-D sensor model and back-projects the VALID depth, so occlusion,
glass holes, noise and partial coverage come from the rendering. Two sensor regimes are pinned: `INDOOR_SENSOR_CLEAN`
(inside the interior chain's 2 cm plane tolerance) and `INDOOR_SENSOR_NOISY` (sigma ~1.6 cm at 5 m: outside it). Known
engine limits the fixtures measured are strict `xfail`s in `tests/test_synthetic_indoor.py`, not hidden.

## Run

```bash
python -m pytest tests/test_synthetic_rgbd.py tests/test_synthetic_vio.py tests/test_synthetic_indoor.py -q
```
