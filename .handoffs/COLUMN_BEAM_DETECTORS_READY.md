# HANDOFF

Agent: Freebuff (reconstruction + perception owner)
Branch: agent/freebuff-reconstruction-perception
Commit: (feat(perception): measured column + beam class detectors)
Checkpoint: COLUMN_BEAM_DETECTORS_READY
Status: CHECKPOINT_READY

## What exists

P1 perception breadth increment, following the merged PR #86
stair-detector pattern. The registry declared `column` and mapped
`cornice -> beam` but no detector produced class-level EVIDENCE for
either. Two measured detectors now exist and are wired into the
canonical observation -> entity-resolution -> WorldIR promotion path:

1. perception/architecture/columns.py -- ColumnFit
   - Geometry: parametric.fit_cylinder reused UNCHANGED (shell /
     elongation / sanity refusals inherited; no duplicated solver).
   - Class gates (measured refusals naming the fact):
     verticality |axis.up| >= MIN_AXIS_UP_DOT (0.9, the registry's
     own threshold); axial extent >= MIN_HEIGHT_M (0.5 m).
   - Confidence = cylinder.confidence x measured up-dot, floored at
     0.05 -- a barely-vertical member is a less certain column.

2. perception/architecture/beams.py -- BeamFit
   - PCA via parametric's Jacobi eigensolver (sorted DESCENDING,
     explicitly documented; the solver returns ascending).
   - Gates: single-axis elongation sqrt(l1/l2) >= 4.0;
     horizontality |axis.up| <= 0.35; length >= 1.0 m; BOTH lateral
     extents >= 0.05 m; face-shell ratio >= 0.35 x half-extent on
     both lateral axes (prism faces hug their extent; filled panel
     samples fail); aspect >= 3.0.
   - Residual = rms lateral distance to the measured box shell;
     confidence = 1 - rms/min-half-extent, clamped [0.05, 1].

Both are self-classifying fits (fit.kind) accepted by
build_component_observations alongside StaircaseFit; _position_of
(column = mid-axis, beam = support centroid); promotion carries
measured fit properties; "beam" -> EntityType.BEAM added (column was
already mapped).

## Public contracts

- columns.detect_column(points, up=(0,0,1)) -> ColumnFit | raises
  ColumnRefused (FitRefused subclass)
- beams.detect_beam(points, up=(0,0,1)) -> BeamFit | raises
  BeamRefused (FitRefused subclass)
- Both fits: .to_dict() with measured fields only; .confidence in
  [0.05, 1]; .n_points; ColumnFit.height_m / radius_m / axis /
  axis_up_dot / cylinder; BeamFit.axis / length_m / height_m /
  width_m / aspect_ratio / rms_residual_m / position.
- build_component_observations: (StaircaseFit | ColumnFit | BeamFit)
  -> one accepted observation, arch_class = fit.kind.
- promote_component_to_entity: custom_properties.fit_kind in
  {"stairs","column","beam"} with each fit's measured quantities +
  evidence_ids.

## How to consume (Antigravity / downstream)

1. Segment points as today (perception/architecture/segments.py).
2. For each segment call detect_column / detect_beam (and
   detect_stairs); catch FitRefused and record it -- refusals are
   facts, never drop silently.
3. Pass successful fits to build_component_observations (they are
   self-classifying; no registry loop needed), then
   resolve_components -> promote_component_to_entity as before.
4. Entities carry confidence, measured geometry in custom_properties,
   and evidence ids traceable to the supporting points.

## Tests run

- tests/test_column_beam_detectors.py: 19 tests
  (TestColumnDetector 8, TestBeamDetector 8, integration 3 incl.
  end-to-end observation -> resolution -> promotion for both classes
  and refused-points-never-become-entities).
- Stair regression: tests/test_stair_detector.py 12 green.
- Cluster: architecture/perception -k filter 140 passed / 1 skipped;
  parametric+detail+contract 72 passed.

## Runtime verification

Deterministic fixtures only (real capture is an external dependency,
never simulated): vertical shell recovers r=0.3 m, h=3.75 m exactly;
box prism recovers 3.0 x 0.3 x 0.2 m; tilted pipe (up-dot 0.707),
horizontal silo, flat patch, vertical post, wall panel, isotropic
blob, and 45-degree ramp all refuse with the measured fact in the
message.

## Dependencies

None new. Pure Python float math; reuses parametric.py internals
(_covariance, _eigen_symmetric_3x3) -- private-helper import is
deliberate (documented in beams.py) to avoid a duplicated eigensolver.

## Known limitations

- Thresholds are documented wide windows, not tuned against real
  capture (recorded on P7-03 + the capabilities registry).
- Windows/roofs still have no dedicated detector (windows remain
  classify.py plane openings).
- Face-shell beam gate assumes face-sampled segments; a densely
  volumetrically-sampled solid beam would fail it honestly (recorded
  as a limitation, not hidden).
- Real staircase/column/beam captures: external dependency; fixture
  verification is synthetic by construction and labeled as such.

## Known bugs

None found in this increment. (One fixture bug during development:
an isotropic box generator initially produced elongation 0.72 --
caught by the red tests, fixed in the fixture.)

## Next agent

Antigravity (consumption) or Freebuff (next increment).

## Exact action for next agent

- Consume per "How to consume" above; the detection layer raises on
  ambiguous structure, so pipeline error handling should record
  FitRefused messages in diagnostics.
- Or continue P1 breadth: windows (opening-in-wall-plane measurement)
  and roofs are the remaining declared-but-unwired classes.
