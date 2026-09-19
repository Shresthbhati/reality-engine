# Plan: Step 20 — Water (Fluids §22, P1 tier)

**Spec**: `docs/BUILD_ORDER.md` (§108 FIRST IMPLEMENTATION ORDER, row 20
"water — not started"); directive §22 FLUIDS (water volume, water
depth, flow, drainage, infiltration, obstruction, buoyancy, pressure,
object interaction — "avoid hard-coded cinematic flooding whenever a
physical model is available"); §20 MULTI-FIDELITY (this is P1 —
simplified physical model, not P2+ coupled CFD; the active fidelity
must be visible in the code, not implied).

**Scope boundary**: standing water bodies with depth/volume, simple
level-based flow between adjacent bodies, drainage/infiltration, and
buoyancy force on a submerged entity. This is P1: no pressure fields,
no real fluid dynamics (no Navier-Stokes, no SPH/grid solver), no wave
propagation. Rain (§19, already built in `engine/environment/rain.py`)
produces accumulation on `RainSurface` — water bodies are a distinct,
larger-scale concept (a pond, a flooded room) and do NOT read from or
depend on `RainSurface`; don't reach into `engine/environment/rain.py`.
Wind (§23), fire (§21), and flooding-as-a-disaster (§26) are separate,
not-yet-started steps — do not build those here.

## Global Constraints (binding on every task)

- Follow the codebase's existing subsystem shape exactly — see
  `engine/environment/rain.py` (this repo's most recent environmental
  subsystem) as the reference pattern: a frozen `@dataclass` config, a
  state/manager class seeded with `DeterministicRNG` (`engine/core/rng.py`),
  a `step(dt, tick)` method, `serialize()/deserialize(self, data)` as an
  INSTANCE method (not a classmethod — that was a Step 19 fix-round
  finding, don't repeat it) with `"format_version": 1` and a `ValueError`
  on mismatch, and `get_logger(...)` from `engine/core/logging.py` for
  structured log lines.
- Module lives at `engine/environment/water.py` (same package as rain;
  `engine/environment/__init__.py` already exports rain's names — add
  water's alongside them, don't remove anything).
- Water events publish through the existing `EventBus`
  (`engine/world/events.py`) — read its actual `Event`/`EventBus.publish(...)`
  signature yourself before using it (Step 19's implementers got this
  right by reading the real file instead of assuming a `priority`
  kwarg exists on `publish()` — it doesn't).
- All physical quantities are SI: depth in meters, volume in m³, area
  in m², flow rate in m³/s, time in seconds, density in kg/m³, force in
  Newtons, gravity as a named constant (9.81 m/s²) — don't hardcode
  `9.81` inline more than once, name it.
- Determinism: given the same seed and the same sequence of
  `step(dt, tick)` calls, results must be bit-identical. No RNG needed
  for the flow/buoyancy math itself (it's deterministic physics, not
  variability) — only construct `DeterministicRNG` if a task actually
  calls it, per the Step 19 review's minor finding about an unused `_rng`
  field. If you don't need it, don't instantiate it.
- No mutation of physics/world state from this module — it produces
  data (depth, volume, flow, buoyancy force) for a future step to wire
  into the physics backend; don't reach into `engine/physics/backend/`.
- Every new public class/function gets tests in `tests/test_water.py`
  following the existing test-file shape (`tests/test_rain.py`) —
  hand-computed expected values for any formula, not weak `> 0`
  assertions. This was flagged twice in Step 19's reviews (visibility
  floor, accumulation math) — get it right the first time here.
- Update `docs/BUILD_LEDGER.md` (new REQ entry — check the current
  highest REQ number in the file first, don't assume it; this file may
  have moved since the plan was written) and `docs/TEST_MATRIX.md` (new
  row) as part of the task that completes the ledger-worthy milestone
  (Task 3), following the exact format of the most recent REQ entries
  already there.
- Run `python -m pytest tests/ -q` after every task and confirm the
  full suite still passes before reporting DONE. State the exact
  before/after count in your report — don't estimate it.

## Task 1: WaterBody model — depth, volume, buoyancy

Create `engine/environment/water.py` with:

- `WaterConfig` (frozen dataclass): `water_density_kg_m3: float = 1000.0`
  (fresh water), `gravity_m_s2: float = 9.81`, `seed: int = 42`.
- `WaterBody` class (plain mutable class — state changes via flow/drainage
  in Task 2, not frozen): `__init__(self, body_id: str, surface_area_m2: float, depth_m: float = 0.0)`.
  Raises `ValueError` if `surface_area_m2 <= 0`. Computed property
  `volume_m3` = `surface_area_m2 * depth_m`.
- `WaterState` class, constructed with a `WaterConfig`:
  - `__init__(self, config: WaterConfig)`.
  - `register_body(self, body: WaterBody) -> None` — raises `ValueError`
    on duplicate `body_id` (check `engine/environment/rain.py::register_surface`
    for the exact pattern, it already does this correctly).
  - `get_body(self, body_id: str) -> WaterBody` — raises `ValueError` if
    unknown (match rain.py's convention for unknown-id lookups exactly —
    read `get_accumulation` in rain.py to see which exception it uses).
  - `buoyancy_force_n(self, body_id: str, submerged_volume_m3: float) -> float`
    — Archimedes' principle: `force = config.water_density_kg_m3 * config.gravity_m_s2 * submerged_volume_m3`.
    Raise `ValueError` if `submerged_volume_m3 < 0` or if
    `submerged_volume_m3` exceeds the body's own `volume_m3` (can't
    submerge more volume than the body has depth for — this is a
    simplification appropriate for P1 fidelity, document it as such in
    a comment).
  - `serialize()` / `deserialize(self, data)` — `format_version: 1`,
    covers all registered bodies' `surface_area_m2`/`depth_m`.

Worked example to verify your buoyancy formula against: a body with
`surface_area_m2 = 10.0`, `depth_m = 2.0` (volume = 20 m³), and an
object displacing `submerged_volume_m3 = 1.5` m³: force =
`1000.0 * 9.81 * 1.5 = 14715.0` Newtons.

Tests in `tests/test_water.py::TestWaterBody` and `::TestWaterState`
covering: body creation and volume computation, duplicate registration
rejected, unknown body lookup raises, buoyancy formula against the
worked example above (exact value, not `> 0`), buoyancy rejects
negative or over-volume submersion, serialization round-trip.

**Report file**: `task-1-report.md` (path given by `task-brief`).

## Task 2: Flow between adjacent bodies, drainage

Extend `engine/environment/water.py` (depends on Task 1's `WaterBody`/`WaterState`):

- On `WaterState`, add:
  - `connect_bodies(self, body_id_a: str, body_id_b: str) -> None` —
    registers a bidirectional adjacency (both bodies must already be
    registered, raise `ValueError` if either is unknown; raise
    `ValueError` if the pair is already connected — no duplicate edges).
    Store adjacency as a `dict[str, set[str]]` or similar, your choice,
    but document the shape.
  - `step(self, dt: float, tick: int) -> None` — for every connected
    pair, compute a level-equalizing flow: if body A's depth exceeds
    body B's depth, transfer volume from A to B proportional to the
    depth difference, at a rate bounded so bodies never overshoot past
    equal depth in one step and never go negative. Use this exact rule:
    `flow_m3 = min(flow_rate_coefficient * (depth_a - depth_b) * dt, (depth_a - depth_b) * area_a * area_b / (area_a + area_b))`
    where `flow_rate_coefficient` is a new `WaterConfig` field default
    `0.5` (unitless, tunable — this is the P1 simplification, name it
    honestly as such in a docstring, don't pretend it's a real
    hydraulic conductivity). The clamp term is the volume that exactly
    equalizes both depths — derive it yourself and verify by hand that
    transferring it makes `depth_a == depth_b` (do the algebra: after
    transfer, `depth_a' = (V_a - flow)/area_a`, `depth_b' = (V_b + flow)/area_b`;
    solve for the `flow` that makes `depth_a' == depth_b'`). Apply the
    smaller of the rate-limited flow and the equalizing flow so a single
    step never overshoots.
  - Add a `drainage_rate_m3_s: float = 0.0` field to `WaterBody`
    (default zero — most bodies don't drain). `step()` also subtracts
    `drainage_rate_m3_s * dt` from each body's volume (floored at zero
    volume, never negative).
  - Extend `serialize()/deserialize()` to include adjacency and
    per-body drainage rate.

Tests: connecting bodies (both directions queryable), duplicate
connection rejected, unknown body in `connect_bodies` raises, flow
between two bodies with hand-computed expected depths after one step
(pick concrete numbers, do the algebra above, assert exact values),
flow that would overshoot equalization is clamped exactly at
equalization (test this specific edge case), drainage reduces volume
correctly and floors at zero, serialization round-trip including
adjacency and drainage rate.

**Report file**: `task-2-report.md`.

## Task 3: Event wiring, diagnostics, and ledger

- Add overflow detection: `WaterBody` gets a `max_depth_m: Optional[float] = None`
  field (default `None` = no limit). On `WaterState.step()`, after
  applying flow/drainage, if a body's `depth_m` newly exceeds its
  `max_depth_m` (wasn't exceeding it before this step, is now), publish
  an `Event` through an optional `event_bus: Optional[EventBus] = None`
  parameter on `WaterState.__init__` (default `None`, so Task 1/2's
  tests keep working unchanged — this mirrors rain's `set_intensity`
  band-change pattern exactly, read `engine/environment/rain.py`'s
  event-publishing code before writing this). Event type string:
  `"water.body_overflowed"`. Event data: `{"body_id": ..., "depth_m": ..., "max_depth_m": ...}`.
- Add `WaterState.get_diagnostics(self) -> dict` returning: total
  volume across all bodies (m³), body count, connection count,
  overflowing body count (bodies currently above their `max_depth_m`).
- Update `engine/environment/__init__.py` to also export `WaterConfig`,
  `WaterBody`, `WaterState` (alongside the existing rain exports —
  don't remove those).
- Add `docs/BUILD_LEDGER.md` REQ entry (check the file for the current
  highest REQ number, use the next one) for Water: Section §20/§22,
  dependency on the event-bus REQ and Step 19's rain REQ is NOT
  required (water doesn't depend on rain — say so explicitly in the
  entry so nobody assumes a false dependency), status TESTED, files,
  tests, verification bullets, an honest Limitations line (P1
  simplification: `flow_rate_coefficient` is a tunable constant not a
  real hydraulic property; no pressure field; no obstruction/object
  interaction beyond the buoyancy force calculation; no wave
  propagation), no blockers. Update the ledger's requirement counts,
  status distribution, and test coverage totals matching the exact
  format of the entries immediately before it. Update
  `docs/TEST_MATRIX.md` with the new test file's row and refresh the
  totals line.
- Add `tests/test_water.py::TestWaterEvents` covering: event published
  exactly once when a body crosses above `max_depth_m`, no repeated
  event on subsequent steps while still above the threshold (only fires
  on the crossing, not every step it stays over), no event if
  `max_depth_m` is `None`, no crash/no event attempted when
  `event_bus=None`.

**Report file**: `task-3-report.md`.
