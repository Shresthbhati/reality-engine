# Plan: Step 19 — Rain (Weather §19)

**Spec**: `docs/BUILD_ORDER.md` (§108 FIRST IMPLEMENTATION ORDER, row 19
"rain — not started"); directive §24 WEATHER (rain is environmental
state, not a visual effect; must be capable of influencing simulation
systems); §80 "done" bar (implementation + unit tests + integration test
+ serialization test + determinism test + error handling + diagnostics).

**Scope boundary**: rain intensity/accumulation only. Wind (§23), water
volume/flow/buoyancy (§20 "water"), and flooding are separate,
not-yet-started steps — do not reach into them. A puddle depth number
on a registered surface is in scope; simulating that puddle as a fluid
body is not.

## Global Constraints (binding on every task)

- Follow the codebase's existing subsystem shape exactly — see
  `engine/physics/destruction/glass.py` and `engine/physics/destruction/debris.py`
  as the reference pattern: a frozen `@dataclass` config, a state/manager
  class seeded with `DeterministicRNG` (`engine/core/rng.py`), a `step(dt, tick)`
  method, `serialize()/deserialize()` with `"format_version": 1` and a
  `ValueError` on mismatch, and a `get_logger(...)` from `engine/core/logging.py`
  for structured log lines.
- Module lives at `engine/environment/rain.py` (new package —
  `engine/environment/` does not exist yet; create it with an `__init__.py`
  exporting the public names, same shape as `engine/physics/destruction/__init__.py`).
- Rain events publish through the existing `EventBus`
  (`engine/world/events.py`, `Event`/`EventPriority`, `bus.publish(...)`)
  — do not invent a second event mechanism.
- All physical quantities are SI: intensity in mm/hour, accumulation
  depth in meters, time in seconds. Convert mm/hour to m/s internally
  once, name the constant, don't repeat the conversion inline.
- Determinism: given the same seed and the same sequence of `step(dt, tick)`
  calls, results must be bit-identical. Follow `DeterministicRNG` usage
  from `engine/physics/destruction/fracture.py`.
- No mutation of physics/world state from this module — it produces
  data (intensity, accumulation, visibility factor, events); wiring rain
  into the physics backend's friction/visibility is a future step, not
  this one. Don't reach into `engine/physics/backend/`.
- Every new public class/function gets tests in `tests/test_rain.py`
  following the existing test-file shape (see `tests/test_glass_physics.py`).
- Update `docs/BUILD_LEDGER.md` (new REQ entry, TESTED status, following
  the exact format of the REQ-026 through REQ-031 entries already
  there) and `docs/TEST_MATRIX.md` (add the new test file's row) as
  part of the task that completes the ledger-worthy milestone (Task 3).
- Run `python -m pytest tests/ -q` after every task and confirm the full
  suite still passes (430 tests before this plan starts) before
  reporting DONE.

## Task 1: Rain intensity model

Create `engine/environment/rain.py` with:

- `RainIntensity` — an enum classifying rate bands, matching common
  meteorological convention: `NONE` (0 mm/h), `LIGHT` (0-2.5 mm/h),
  `MODERATE` (2.5-7.6 mm/h), `HEAVY` (7.6-50 mm/h), `EXTREME` (>50 mm/h).
- `RainConfig` (frozen dataclass): `max_intensity_mm_h: float = 50.0`,
  `visibility_reduction_per_mm_h: float = 0.015` (fraction of visibility
  lost per mm/h of intensity, clamped so visibility factor never goes
  below 0.1), `seed: int = 42`.
- `RainState` class, constructed with a `RainConfig`:
  - `__init__(self, config: RainConfig)` — starts at intensity 0,
    `DeterministicRNG(config.seed)`.
  - `set_intensity(self, mm_per_hour: float, tick: int, timestamp: float) -> None`
    — clamps to `[0, config.max_intensity_mm_h]`, raises `ValueError` if
    given a negative value before clamping (negative intensity is a
    caller bug, not a value to silently floor). Records the previous
    intensity band; if the band changed, this is where Task 2's event
    publish will hook in (leave a clear extension point, don't publish
    yet — the bus isn't wired until Task 3).
  - `intensity_band(self) -> RainIntensity` — classifies current
    `intensity_mm_h` into the enum above.
  - `visibility_factor(self) -> float` — `1.0 - min(0.9, intensity_mm_h * config.visibility_reduction_per_mm_h)`.
  - `intensity_mm_h: float` and `intensity_m_s: float` (converted,
    computed property) are both readable.
  - `to_dict()/serialize()` — `format_version: 1`, current intensity,
    tick/timestamp of last change.
  - `deserialize(data)` — restores state, raises `ValueError` on
    unsupported `format_version`.

Tests in `tests/test_rain.py::TestRainState` covering: intensity
clamping (above max, negative raises), each `RainIntensity` band
boundary (test at least one value inside each band and one at each
boundary edge), `visibility_factor` monotonically decreasing with
intensity and floored at 0.1, serialize/deserialize round-trip.

**Report file**: `task-1-report.md` (path given by `task-brief`).

## Task 2: Surface accumulation

Extend `engine/environment/rain.py` (same file, this task depends on
Task 1's `RainState` existing):

- `RainSurface` (plain class, not frozen — accumulation mutates):
  `__init__(self, surface_id: str, area_m2: float, absorption_coefficient: float = 0.3)`.
  `absorption_coefficient` in `[0, 1]`: fraction of rain that infiltrates
  rather than accumulating as standing water; raise `ValueError` outside
  that range. Tracks `accumulated_depth_m: float = 0.0`.
- On `RainState`, add:
  - `register_surface(self, surface: RainSurface) -> None` — raises
    `ValueError` on duplicate `surface_id` (mirror the pattern in
    `engine/physics/destruction/glass.py::register_pane`).
  - `step(self, dt: float, tick: int) -> None` — for every registered
    surface, accumulate `intensity_m_s * dt * (1 - absorption_coefficient)`
    onto `accumulated_depth_m`. Deterministic: no RNG call needed for
    the accumulation itself (RNG is reserved for future variability, not
    required by this task — don't add randomness that isn't tested).
  - `get_accumulation(self, surface_id: str) -> float` — raises
    `KeyError`-style `ValueError` if unknown (match existing modules'
    convention of `ValueError` over `KeyError` for domain lookups — check
    `debris.py`/`glass.py` for which they actually use and follow it).
  - Extend `serialize()/deserialize()` to include registered surfaces
    and their accumulated depth.

Tests: registering surfaces, duplicate registration rejected, correct
absorption math at a known intensity/dt (compute the expected value by
hand in the test, don't just assert "> 0"), multiple surfaces
accumulating independently, unknown surface lookup raises, serialization
round-trip including surface state.

**Report file**: `task-2-report.md`.

## Task 3: Event wiring, diagnostics, and ledger

- Wire `RainState.set_intensity`'s band-change detection (left as an
  extension point in Task 1) to publish an `Event` on an `EventBus`
  passed into `RainState.__init__` as an optional parameter
  (`event_bus: Optional[EventBus] = None`, default `None` so Task 1/2's
  tests that construct `RainState` without a bus keep working
  unchanged). Event type string: `"rain.intensity_changed"`. Event data:
  `{"from_band": ..., "to_band": ..., "intensity_mm_h": ...}`. Priority:
  `EventPriority.NORMAL` (check the actual enum member names in
  `engine/world/events.py` — use whatever exists, don't invent one).
- Add `RainState.get_diagnostics(self) -> dict` returning: current
  intensity, band, visibility factor, registered surface count, total
  accumulated volume across all surfaces (`sum(depth * area)` per
  surface, in m³).
- Update `engine/environment/__init__.py` to export `RainConfig`,
  `RainIntensity`, `RainState`, `RainSurface`.
- Add `docs/BUILD_LEDGER.md` REQ entry (next available REQ number after
  REQ-031) for Rain: Section §19/§24, dependency REQ-014 (event bus),
  status TESTED, files, tests, verification bullet list, an honest
  Limitations line (no wind-driven angle, no runoff/drainage between
  surfaces, no puddle-as-fluid-body — those belong to later steps), no
  blockers. Update the ledger's requirement counts, status distribution,
  and test coverage totals the same way REQ-026 through REQ-031 did.
  Update `docs/TEST_MATRIX.md` with the new test file's row.
- Add `tests/test_rain.py::TestRainEvents` covering: event published on
  band change, no event on same-band intensity change (e.g. 1.0mm/h to
  1.5mm/h, both LIGHT), no crash/no event attempted when `event_bus=None`.

**Report file**: `task-3-report.md`.
