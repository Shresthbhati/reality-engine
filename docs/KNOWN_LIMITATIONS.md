# Known Limitations

Generated from the per-requirement "Limitations" field in
[BUILD_LEDGER.md](BUILD_LEDGER.md) — nothing here is invented; this is
that data regrouped for scanning. Update this file whenever a
requirement's limitations change, not just when new ones appear.

## Blocking nothing today, but real gaps

| Requirement | Limitation | Upgrade path |
|---|---|---|
| REQ-007 Job System | Stub only, not integrated into runtime | Wire into `WorldRuntime.step()` once systems need parallel scheduling |
| REQ-011 Collision Detection | O(n²) broadphase; sphere/box/plane primitives only | BVH/spatial hash + capsule/mesh shapes when entity counts or geometry demand it |
| REQ-012 Contact Resolution | Single-pass solver (no iteration); 4-material database | Sequential-impulse iteration loop; expand material table as new materials are added |
| REQ-025 Basic Fracture | No mesh fragment generation; single-pass (no cascading fracture) | Add mesh-based fragmentation and multi-generation cascade once a mesh pipeline exists |
| REQ-026 Glass Physics | Frame attachment points tracked but not enforced; no mesh generation | Enforce attachment constraints in `fracture_pane()`; generate real shard meshes |
| REQ-027 Debris System | No inter-fragment collisions; fixed sleep threshold; no spatial partitioning | Add fragment-fragment collision pass; make sleep threshold configurable; spatial hash once fragment counts grow |
| REQ-028 Replay System | No automatic replay-determinism verification; snapshots don't capture all internal state; no compression | Add a replay-vs-live comparison test; extend `TimelineSnapshot` fields as needed |
| REQ-029 Viewport | Angle-cone frustum test only (no separate H/V FOV, no projection matrix); no occlusion culling | Real Mat4 projection + occlusion when a renderer consumes this |

## By design, not expected to change soon

| Requirement | Limitation | Why acceptable |
|---|---|---|
| REQ-003/018 Entity Registry | Single-threaded | No concurrency requirement yet |
| REQ-004 Serialization | Schema v1 only, no migrations | No v2 schema exists yet |
| REQ-005 Coordinate System | Only registered frames resolve, no auto-discovery | Explicit registration matches §10's "never silently mix coordinate systems" |
| REQ-006 Deterministic Clock | No wallclock sync, purely simulation-time | Correct for a deterministic simulation engine |
| REQ-008 Units | Lightweight `Quantity` wrapper, no dimensional analysis | Sufficient for current SI-consistency requirement |
| REQ-009 Deterministic RNG | `random.Random`-based, not cryptographic, platform-dependent floats | Determinism requires reproducibility, not cryptographic strength |
| REQ-010 Rigid Body | No rotational contact response; velocity-based sleeping only | Matches current golden-test scope |
| REQ-013 Numerics | Energy explosion threshold fixed at 1000x | No case yet requiring it to be configurable |
| REQ-014 Event Bus | In-memory only, no persistence | Durable log deferred to export/replay integration |
| REQ-015 Logging | No file rotation; stdout/stderr only | No production deployment yet needs rotation |
| REQ-016 Config | TOML only, no env var override | No deployment need for env overrides yet |
| REQ-017 Errors | Hard failures halt-on-first, no recovery | Matches §17's hard-failure classification intent |
| REQ-019 Component Registry | No component dependencies/ordering | Not yet needed by any system |
| REQ-020 Resource Manager | No pooling, no async loading | No asset pipeline exists yet to need it |
| REQ-021 World Lifecycle | No pause/resume states | Not yet requested by any consumer |
| REQ-022 Dependency Graph | Kahn's algorithm, no in-tier priority | No system yet needs same-tier ordering control |
| REQ-023 Caching | No adaptive sizing; substring pattern invalidation | Sufficient for current cache usage |
| REQ-024 WorldRuntime | No streaming, single-world only | Multi-world/streaming not yet a requirement |

## Fixed bug (found Step 17, fixed during the 2026-09-07 quick audit)

**`WorldRuntime` could not actually construct against the WorldIR it's
type-hinted for.** `WorldRuntime.__init__`/`get_entity_from_world`
iterated `world.entities` directly; that only yields `Entity` objects
for the legacy `world_ir.world.WorldIR`'s `EntityRegistry.__iter__` —
for the V1 schema (`world_v1.WorldIR.entities`, a plain
`dict[str, Entity]`), iterating the dict yields its string keys, and
`entity.id` raised `AttributeError` the instant a V1-schema world had
any entities. Fixed by a `_iter_world_entities()` helper that branches
on `isinstance(world.entities, dict)`, used at both call sites in
`engine/world/runtime.py`. Regression tests added in
`tests/test_world_runtime.py`. See [DECISIONS.md](DECISIONS.md) #14.

**Update (2026-09-07, P0 audit-repair)**: the transform-resolution gap
this section used to describe is now fixed — see
[DECISIONS.md](DECISIONS.md) #15. V1-schema `Entity.transform` dicts
resolve through `CoordinateRegistry` exactly like legacy `Transform`
objects, including multi-hop chains and inverse-direction resolution.
Inspector (REQ-030) still reads `world_v1.WorldIR` directly rather than
through `WorldRuntime` — that part of the original decision stands
independent of this fix, since `WorldRuntime`'s ECS side still has no
materials/geometries/measurements API — see [DECISIONS.md](DECISIONS.md) #12.

## Open spec ambiguity

- **Provenance taxonomy** — the implemented `Provenance` enum
  (`OBSERVED/RECONSTRUCTED/ESTIMATED/INFERRED/GENERATED/UNKNOWN/CONFLICT`)
  does not carry a separate `MEASURED` state distinct from `ESTIMATED`,
  which some directive text implies. See [DECISIONS.md](DECISIONS.md) #10.
  Not reconciled — flagged as `SPEC_AMBIGUITY` rather than silently resolved.

## Not yet started (honest gap, not a limitation of something built)

Everything in §21-38 of the master directive beyond what
[BUILD_LEDGER.md](BUILD_LEDGER.md) marks TESTED/VERIFIED/IMPLEMENTED:
fluids, fire, weather, disaster composers, causal graph, branching/
counterfactual engine, AI copilot, natural-language query, rendering
(beyond the viewport camera model), Studio, debug visualization, export
adapters, datasets, and benchmarks. These are empty directories in the
repo (`engine/fluids`, `engine/fire`, `engine/weather`, `engine/disasters`,
`reconstruction/*`, `perception/*`, `apps/*`, `exporters/`, `gpu/`,
`benchmarks/`, `datasets/`, `plugins/`, `shaders/`, `tools/`) — scaffolding
only, no stub classes claiming completion.
