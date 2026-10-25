"""Repository-local SYNTHETIC data pack: deterministic RGB-D, VIO and indoor-architecture fixtures with ground truth.

What this is for: regression fixtures and demos for pipeline stages whose REAL inputs (a physical RGB-D device, a
built VIO backend, a real indoor photo set) are not available to every developer. Every generator is seeded and
deterministic -- the same arguments produce byte-identical files and arrays -- and every generator returns or writes
the ground truth it was built from, so a test can measure error instead of asserting "something came out".

What this is NOT: validation against the real world. Synthetic data proves that the code handles the CONTRACT
(formats, scales, invalid pixels, clock offsets, dropped frames, occlusion, partial observation) correctly. It says
nothing about how a real sensor, a real VIO algorithm or a real building behaves. The honest labels, used verbatim
by ``engine.core.verification`` and by the dataset manifests written here:

    RGB-D   SYNTHETIC RGB-D VERIFIED            -- PHYSICAL DEVICE VERIFICATION PENDING
    VIO     SYNTHETIC VIO PIPELINE VERIFIED     -- REAL BACKEND / HARDWARE VERIFICATION PENDING
    indoor  SYNTHETIC INDOOR VERIFIED           -- REAL INDOOR DATASET VERIFICATION PENDING

Modules: ``scene`` (planar scenes + ray-caster + ground truth), ``rgbd`` (sensor model, rendering, capture writer),
``vio`` (trajectory + IMU generator, stand-in backend executable), ``indoor`` (rooms, corridors, storeys, stairs,
ramps, clutter with ground truth).
"""

SYNTHETIC_RGBD_LABEL = "SYNTHETIC RGB-D VERIFIED"
SYNTHETIC_VIO_LABEL = "SYNTHETIC VIO PIPELINE VERIFIED"
SYNTHETIC_INDOOR_LABEL = "SYNTHETIC INDOOR VERIFIED"
