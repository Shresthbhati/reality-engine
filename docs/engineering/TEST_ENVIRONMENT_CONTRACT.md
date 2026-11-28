# Test Environment Contract

What a test run may assume, how tests declare what they need, and what each CI lane guarantees. Enforced by
`tests/test_dependency_contract.py`, `tests/test_ci_tooling.py`, `tests/_capabilities.py` and the root `conftest.py`.

## 1. Install tiers (what `pyproject.toml` declares)

| tier | install | contains | used by |
|---|---|---|---|
| core | `pip install -e .` | numpy, scipy, pillow, opencv-python-headless, pyproj, **pyyaml, shapely, networkx** (declared 2026-11-28; core modules already imported them, they only arrived transitively) | every package |
| api | `.[api]` | fastapi, uvicorn, sqlalchemy[asyncio], aiosqlite, httpx, alembic, boto3, ... | `apps/`, `geo/` |
| dev | `.[dev]` | pytest, pytest-asyncio, **pytest-timeout**, ruff, mypy, api, export (ifcopenshell, ezdxf), rgbd, meshing, **resources (psutil)** | the default CI image |
| geometry | `.[geometry]` | open3d, usd-core, pycolmap, manifold3d | plane extraction, USD, mesh booleans |
| perception | `.[perception]` | torch, torchvision, transformers, kornia, timm, segment-anything | learned engines (GPU lane only) |
| native | not pip | the `colmap` binary (4.2.0, CUDA build on the workstation) | COLMAP reconstruction |

The wheel ships `world`, `geometry` (namespace package) and the other first-party packages. Known gap, tracked in
`tests/test_dependency_contract.py::_KNOWN_PACKAGING_GAPS`: `perception/intelligence/diagnostics_v2.py` imports
`benchmarks.perception.indoor_dataset_adapter` (a directory that is not shipped). **DECISION REQUIRED (perception owner):**
move the adapter into `perception/` or drop the import; until then a wheel install cannot import that module.

## 2. Rules (each one is a test)

1. **Optional dependencies never break import or collection.** With torch, transformers, kornia, timm, open3d, pycolmap,
   pxr, manifold3d, OpenEXR, imageio, psutil, trimesh, ifcopenshell, ezdxf all *blocked*, every first-party module imports
   (`test_every_package_imports_with_all_optional_dependencies_blocked`). Two modules used to violate this
   (`perception/detection_adapter.py`, `perception/maskrcnn_adapter.py`); the former now guards its torch import and both
   report "unavailable" explicitly. An unguarded import is not "fixed" by swallowing the ImportError into a fake success.
2. **Every third-party module imported at module level is declared** (`test_module_level_third_party_imports_are_declared_dependencies`).
3. **Everything shipped code imports is in the wheel** (`test_every_first_party_package_imported_by_shipped_code_is_in_the_wheel`).
4. **Offline by default.** Tests never download. Model probes only look at the local Hugging Face cache
   (`hf_cache:<org/repo>` capability); perception adapters load with `local_files_only=True`.
5. **A test that needs an optional thing declares it** (below); it does not discover its absence by failing.

## 3. Declaring requirements: `@pytest.mark.requires(...)`

```python
@pytest.mark.requires("colmap", "dataset:south_building")        # native engine + physical data
@pytest.mark.requires("torch", "torchvision")                      # optional extra
@pytest.mark.requires("cuda")                                      # a GPU
@pytest.mark.requires("hf_cache:IDEA-Research/grounding-dino-tiny")  # a cached model
pytestmark = [pytest.mark.requires("colmap", "dataset:real_room_capture")]   # whole module
```

Capabilities (`tests/_capabilities.py`): any importable module name, `colmap` (binary on PATH or `RE_COLMAP_BIN`),
`cuda`, `dataset:<dir under datasets/>`, `hf_cache:<org/repo>`.

| lane | `RE_STRICT_REQUIREMENTS` | unmet requirement |
|---|---|---|
| default CI (`core`, `full` shards, torch-free) | unset | **skip**, reason names the capability; the skip report budgets skips at 30 % |
| native / GPU / physical-data / release-gate lanes | `1` | **fail** ("a missing requirement is a failure, not a skip") |

`pytest` prints the detected capabilities in its header (`reality-engine capabilities: torch=NO, colmap=yes, ...`).
Physical-data tests never substitute synthetic data: no dataset means skip (default) or fail (strict), never a fake pass.
Tests that already carried a hand-written `skipif` (for example `requires_colmap` in `test_sequential_maximum_fidelity.py`)
keep it and add `requires` for the extra capabilities.

## 4. Test classes and where they run

The category column of `docs/engineering/CI_TEST_REGISTRY.md` (generated from discovery) classifies every file:
`release-gate > physics-excluded > physical-data > native > gpu > integration > e2e > unit`.

| class | marker / requirement | runs in | isolation |
|---|---|---|---|
| unit | none | `core` allowlist + every `full` shard | in-process, `tmp_path` for every SQLite/artefact root |
| integration / e2e | `integration`, `e2e` | `full` shards | in-process, per-test app + DB via the `client` fixtures |
| slow | `slow` | nightly `slow` job only | `--timeout` + faulthandler |
| native (COLMAP) | `requires("colmap")` | strict native lane (workstation / self-hosted) | **child process** via `scripts/run_isolated.py` |
| GPU | `requires("cuda")` | strict GPU lane, **one at a time** | child process; never concurrent with another GPU job |
| physical-data | `requires("dataset:...")` | strict data lane | as native |
| release-gate | `tests/integration/test_release_gate.py` | native lane | child process, report JSON kept |

Run dangerous workloads like this so a crash is a *crash*, not an assertion failure:

```powershell
python scripts/run_isolated.py --timeout 1700 --report rg.json -- python -m pytest tests/integration/test_release_gate.py
```

Outcomes: `passed`, `test_failure`, `native_crash`, `timeout` (the whole process tree is killed), `killed`, `internal_error`,
`usage_error`, `no_tests`. On Windows a real native crash exits with code **3**, the same as pytest's "internal error",
so the wrapper also looks for CPython's `Fatal Python error:` line (see `NATIVE_RUNTIME_DIAGNOSTICS.md`).

## 5. Isolation rules for tests

* Every SQLite database, artefact store and WorldStore root lives under `tmp_path` (`DATABASE_URL`, `STORAGE_ROOT`,
  `WORLDSTORE_ROOT` set with `monkeypatch.setenv`). A test must not write to the repository: `test_deployment_and_docs.py`
  used to overwrite `docs/openapi.json` and `docs/API_DOCUMENTATION.md` on every run; it now writes to `tmp_path` and a
  separate test fails when the committed route catalogue is stale.
* Environment variables are set with `monkeypatch`, never `os.environ[...] =`.
* No arbitrary `time.sleep` to wait for a state: poll a condition with a deadline.
* GPU tests do not assume the host GPU: `test_13_gpu_memory_exhaustion...` pins `probe_hardware_profile` to the simulated
  6 GB device so the OOM-then-CPU path is the same on a CPU runner and on the RTX 4050.
* The remaining open isolation defect is listed in `FLAKY_TEST_REPORT.md`.

## 6. Reproducing CI locally

CI = Python 3.10/3.11/3.12 on Ubuntu, `pip install -e ".[dev]"`, no torch, no GPU, no COLMAP. Locally:

```powershell
py -3.13 -m venv ci-venv ; ci-venv\Scripts\python -m pip install -e ".[dev]"
ci-venv\Scripts\python scripts/ci_shard_tests.py --shard 3 --of 8 > shard.txt     # LF-only manifest on every OS
ci-venv\Scripts\python -m pytest (Get-Content shard.txt) -m "not slow" -ra
ci-venv\Scripts\python scripts/ci_test_registry.py --check ; ci-venv\Scripts\python scripts/ci_test_registry.py --verify-collection
```

(`py -3.11` on the maintainer workstation points at a missing interpreter; the baseline was reproduced on 3.13. Python
3.10-3.12 were not available locally, so version-specific behaviour is **not verified** here.)
