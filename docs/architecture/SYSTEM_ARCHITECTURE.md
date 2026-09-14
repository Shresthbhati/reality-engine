# Reality Engine — Architecture

Reality Engine is an **evidence-grounded reality-to-digital-world
compiler**: real-world capture becomes a persistent, inspectable,
measurable, queryable digital world.

Canonical pipeline:

```
REAL WORLD
  → SOURCE        (phone / drone / DSLR / video / RGB-D / LiDAR / GIS files)
  → SOURCE INGESTION
  → EVIDENCE      (frames, observations, sensor records, artifacts)
  → TIME          (synchronization: sensor-local → global)
  → CALIBRATION   (intrinsics, extrinsics, sensor rig)
  → LOCALIZATION  (SfM / SLAM / VIO trajectories)
  → MULTI-SOURCE REGISTRATION
  → RECONSTRUCTION (sparse → dense geometry)
  → PERCEPTION    (detection, segmentation, lifting, identity, structure)
  → EVIDENCE FUSION
  → UNCERTAINTY   + PROVENANCE + CONFLICTS
  → WorldIR       ← canonical semantic world representation
  → PERSISTENT WORLD (WorldStore)
  → APPLICATIONS  (Reality Studio, measurements, queries, exports)
  → SIMULATION    (downstream consumer; never the source of truth)
```

## Non-negotiable architectural rules

1. **WorldIR is canonical.** Blender, glTF, USD, COLMAP outputs, the
   viewer, and the renderer are consumers/export targets — never
   competing sources of truth.
2. **Source ≠ Evidence ≠ WorldIR.** What the user supplied, what the
   engine derived, and the compiled interpretation are strictly separate.
3. **No LLM runtime.** Runtime intelligence comes from computer vision,
   geometry, reconstruction, learned perception models, numerical
   methods, and fusion — never from a language model API.
4. **No silent degradation.** relative→metric, sparse→dense,
   approximate→exact, unverified→verified transitions must be explicit.
5. **No fake backends.** An unavailable backend reports
   `BACKEND_UNAVAILABLE`-style honest state, never a placeholder result.
6. **Observed vs simulated is distinguishable.** Simulation may not
   silently overwrite observed reality; write-back is provenance-tracked.
7. **Provenance survives.** Every entity traces: entity → geometry →
   artifact → observation → frame → source.

## Source-of-truth hierarchy

Evidence precedence when documents disagree with reality:

1. Executable code and runtime behavior
2. Executed test/build results
3. Configuration
4. Generated artifacts
5. Current documentation
6. Historical claims / previous sessions

See also: [`implementation/IMPLEMENTATION_STATUS.md`](implementation/IMPLEMENTATION_STATUS.md)
(what exists), [`implementation/PENDING_IMPLEMENTATION.md`](implementation/PENDING_IMPLEMENTATION.md)
(what remains), [`engineering/DESIGN_DECISIONS.md`](engineering/DESIGN_DECISIONS.md) (why).
