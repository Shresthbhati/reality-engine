"""Root pytest conftest for Reality Engine core test suite.

Physics/simulation tests depend on ``engine.physics``, ``engine.environment``,
and ``engine.simulation`` which are downstream modules that live in
``reality-engine-child``.  They fail at *collection* (ImportError) when
running the core suite.

Strategy
--------
* ``collect_ignore_glob`` excludes these files from the default collection so
  that ``pytest tests/`` runs cleanly without collection errors.
* The files are NOT deleted — they can be run explicitly once the child-repo
  modules are installed::

      pytest tests/test_physics_rigid_body.py tests/test_debris_system.py ...

* ``PHYSICS_TEST_MODULES`` lists every excluded module for CI documentation.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Physics / simulation test modules (downstream — not part of core CI)
# ---------------------------------------------------------------------------
PHYSICS_TEST_MODULES = [
    "test_benchmark_harness.py",
    "test_debris_system.py",
    "test_fire_system.py",
    "test_fracture_system.py",
    "test_glass_physics.py",
    "test_physics_backend_golden.py",
    "test_physics_collision.py",
    "test_physics_compiler.py",
    "test_physics_compiler_e2e.py",
    "test_physics_debugger.py",
    "test_physics_events.py",
    "test_physics_gyroscopic.py",
    "test_physics_material.py",
    "test_physics_math3.py",
    "test_physics_numerics.py",
    "test_physics_rigid_body.py",
    "test_rain.py",
    "test_replay_system.py",
    "test_temporal_state.py",
    "test_water.py",
    "test_wind_field.py",
    "test_world_physics_compiler.py",
]

collect_ignore_glob = [f"tests/{name}" for name in PHYSICS_TEST_MODULES]
