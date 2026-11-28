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

import os

# Test-only seams (e.g. REALITY_TEST_BACKEND) are honoured only under this env.
os.environ.setdefault("REALITY_ENV", "test")
# Measure stage memory but never refuse a stage because the dev machine is busy (tests that exercise refusal pass an explicit mode).
os.environ.setdefault("RE_MEMORY_ADMISSION", "record")

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


# ---------------------------------------------------------------------------
# Declared requirements: @pytest.mark.requires("torch", "colmap", "cuda", "dataset:south_building", "hf_cache:org/repo")
# See tests/_capabilities.py. Missing capability -> visible skip; under RE_STRICT_REQUIREMENTS=1 -> failure.
# ---------------------------------------------------------------------------
import pytest  # noqa: E402

from tests import _capabilities  # noqa: E402


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires(*capabilities): declared requirements (module name, 'colmap', 'cuda', 'dataset:<name>', "
        "'hf_cache:<org/repo>'). Missing -> skip with the reason; fails instead when RE_STRICT_REQUIREMENTS=1.",
    )


def pytest_report_header(config):
    caps = ("torch", "transformers", "kornia", "pycolmap", "open3d", "colmap", "cuda", "dataset:south_building")
    state = ", ".join(f"{c}={'yes' if _capabilities.probe(c)[0] else 'NO'}" for c in caps)
    return [f"reality-engine capabilities: {state}", f"strict requirements: {_capabilities.strict()}"]


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    declared = tuple(a for m in item.iter_markers("requires") for a in m.args)
    missing = _capabilities.unmet(declared)
    if not missing:
        return
    message = "requires: " + "; ".join(missing)
    if _capabilities.strict():
        pytest.fail(f"{message} (RE_STRICT_REQUIREMENTS=1: a missing requirement is a failure, not a skip)", pytrace=False)
    pytest.skip(message)
