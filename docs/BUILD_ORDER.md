# Build order

Source of truth: `REALITY_ENGINE_MASTER_SPECIFICATION_V10_WORLD_PHYSICS_AND_AGENT_BUILD_BIBLE.md`,
§108 FIRST IMPLEMENTATION ORDER. Do not skip ahead to disasters/physics
solvers before the foundation below is solid — the spec is explicit
about this ("Do NOT begin with hurricanes.").

**A note on the "V11 DETAILED" upload set (5 files, ~1.1M lines):**
almost all of it is machine-generated template filler — the same
boilerplate sentence repeated with only a name/id substituted, tens of
thousands of times (`### Extended Requirement NNNNN — [Category]`,
`### V11 Requirement NNNNN — [X] schema`, `### RE-01-NNNN`, `### agent.parameter.NNN`
— all verified byte-identical apart from the substituted name). The one
substantive addition is a ~220-line prose section in part 04 (§900-969,
"DEEP IMPLEMENTATION SPECIFICATION V11"), which is real and has been
incorporated below (full Euler rigid-body equation, richer event schema,
contact model fields, etc.). Don't spend effort re-reading the rest of
that upload set looking for more signal — there isn't any.

```text
 1. repository            done  -- skeleton per §81/§82
 2. WorldIR                done  -- world_ir/world.py
 3. entity system          done  -- world_ir/entity.py
 4. serialization          done  -- world_ir/serialization.py (§103 World Save Format)
 5. coordinate library     done  -- world_ir/coordinates.py (§89)
 6. job system             done  -- engine/core/jobs.py
 7. deterministic clock    done  -- engine/core/clock.py
 8. rigid body backend     done (P1 tier) -- engine/physics/backend/simple_backend.py (§7.2, §8)
 9. collision              done (LOD_0/1: sphere/box/plane) -- engine/physics/collision/ (§44)
10. material physics       partial -- engine/physics/materials/material.py (§9 subset: density/friction/restitution only)
11. event bus              done -- events/ (§85, V11 §930 field schema)
12. basic fracture         not started
13. glass                  not started
14. debris                 not started
15. replay                 not started
16. viewport               not started
17. inspector              not started
18. physics debugger       not started
19. rain                   not started
20. water                  not started
21. fire                   not started
22. smoke                  not started
23. wind                   not started
24. structural graph       not started
25. explosion              not started
26. flood                  not started
27. earthquake              not started
28. hurricane              not started
29. tornado                not started
30. remaining disasters    not started
31. coupled scenarios      not started
32. world reconstruction   not started
33. game export            not started
```

## What "done" means for each step (§80)

implementation + unit tests + integration tests + serialization test +
determinism test + error handling + diagnostics + documentation +
benchmark + no regression.

Steps 1-11 in this repo have unit tests and at least one integration
determinism test (`tests/test_integration_determinism.py`,
`tests/test_physics_backend_golden.py`, `tests/test_physics_events.py`).
Golden-scene tests for RB_001 (falling cube) and RB_002 (stacked boxes)
exist and pass, but they are not yet `RealityPhysicsBench` (§91) —
there's no stored baseline to regress against until a second backend
exists to compare. Numerical health (§86) is enforced by
`engine/physics/diagnostics/numerics.py`.

## V11 §900-969 corrections applied to the physics layer

- **Full Euler rigid-body equation** (V11 §915): rotational integration
  was `alpha = I^-1 * tau`, which silently drops the gyroscopic term and
  is only correct for isotropic inertia (a sphere). Fixed to
  `alpha = I^-1 * (tau - omega x (I*omega))` in
  `engine/physics/rigid/integrator.py`. This also surfaced (and fixed) a
  latent frame inconsistency: angular velocity/inertia are now
  explicitly documented as body-local-frame quantities throughout, and
  `Quat.integrate()` was fixed to match that convention (it was silently
  using the world-frame multiplication order).
- **Event bus** (V11 §930): `events/` implements the richer field
  schema (event_id, type, timestamp, branch_id, actor_id, source_refs,
  target_refs, parameters, cause_event_ids, severity, confidence,
  deterministic_seed, resulting_state_refs) rather than the sparser one
  in the original §85. Event ids are derived from (seed, tick, sequence)
  — never random — to keep replay deterministic (§934). Wired into
  `SimpleRigidBodyBackend`: a new (not ongoing) contact emits
  `ContactEvent`, or `ImpactEvent` above a 2 m/s approach-speed
  threshold. Only these two of the thirteen named event types in §85
  have emitters; the rest (BreakEvent, FractureEvent, FireEvent, ...)
  wait for their solvers to exist (see `events/types.py`).
- Reviewed but not yet acted on: §916 collision hierarchy (matches our
  existing LOD_0/1 scope, LOD_2+ still future work), §917 contact model
  (our `ContactRecord` doesn't yet carry relative velocity or solver
  iteration diagnostics — small gap, not urgent), §952 "first 100
  implementation milestones" (broadly matches this build order; adds
  asset hashing, a formal provenance graph, and a branch manager that
  aren't scheduled yet).

## Known gaps in the step 8/9 physics backend (be honest, don't hide these)

- **No continuous collision detection.** §8 requires CCD/tunneling
  tests; this backend is discrete-only. A fast body can pass through a
  thin box wall within one timestep without registering a contact.
  Infinite planes are safe (any position past them still reads as
  penetrating), but finite thin geometry is not. Needs a swept test
  before any "high-speed debris/projectile" scenario is trustworthy.
- **No angular contact response.** Contacts apply linear impulses only
  (see `engine/physics/rigid/body.py` docstring). A body can still spin
  under directly applied torque, but stacked/toppling boxes don't tip
  over realistically from off-center impacts. Needs OBB collision
  (LOD_2) to do properly — an axis-aligned Box can't represent a
  rotated contact normal correctly.
- **Box collision is axis-aligned only** (`engine/physics/collision/shapes.py`)
  — no OBB/SAT. A box's collision shape does not rotate with its
  orientation.
- **No sphere-vs-box narrowphase.** `SimpleRigidBodyBackend.step()`
  raises `NotImplementedError` for that pair rather than silently
  skipping it — deliberate: an unhandled overlap should be loud, not a
  quiet physics bug.
- **Single-pass sequential impulse solver**, not iterative Gauss-Seidel.
  Fine at P1 (gameplay) fidelity; a P2+ tier needs a real iterative
  solver with contact caching.

## Also scaffolded, not yet implemented

`engine/{destruction,fluids,fire,weather,disasters,terrain,vegetation,
audio,navigation,streaming,replay}`, `reconstruction/*`, `perception/*`,
`apps/*`, `database/`, `exporters/`, `plugins/`, `shaders/`, `gpu/`,
`tools/`, `benchmarks/`, `datasets/` — each has a placeholder
`README.md` pointing back here. Populate them in the order above, not
by convenience.
