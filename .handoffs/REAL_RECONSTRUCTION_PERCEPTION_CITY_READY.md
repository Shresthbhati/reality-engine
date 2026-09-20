# HANDOFF

Agent: Freebuff (reconstruction + perception owner)
Branch: agent/freebuff-reconstruction-perception
Commit: (see git log — "feat(reconstruction): production batch path + city ingestion + adaptive subdivision")
Checkpoint: REAL_RECONSTRUCTION_PERCEPTION_CITY_READY
Status: CHECKPOINT_READY

## What exists

Four production increments over the canonical Evidence → Session →
ReconstructionContract architecture. No new contracts; every stage
composes existing canonical modules.

### 1. Production batch path — `reconstruction/batch.py`

`group_by_session(items, prefixes)` + `run_batch(items, prefixes=...,
backends=..., reference_session=..., memory_budget=...)`:

- Per-session grouping by DECLARED source path components (`session_a`
  matches components, never substrings — `session_a_backup` stays
  unassigned); unassigned items are recorded under `<unassigned>`,
  never silently merged.
- Per-group failure isolation: admission refusal, insufficient
  evidence, backend failure, and UNAVAILABLE backends fail ALONE with
  a machine-readable reason; the batch continues. One bad session
  never loses the batch.
- Bounded submission: `MemoryBudget.max_frames_per_group` caps what a
  single orchestrator run receives; excess frames are recorded per
  group as `deferred_capacity` (a measured fact). Report is
  summary-sized, not O(frames) — bounded memory at the driver level.
- Unavailable backends: the batch probes availability first and
  records `backend unavailable: <probe detail>` verbatim. No fake
  fallback, ever.
- Honest outcome vocabulary: `complete` ONLY when every group reached
  `success` and every non-reference session registered; a `partial`
  session makes the batch `degraded`.
- Measured uncertainty summary per session (points carrying
  uncertainty, confidence mean/min) from the reconstruction's own
  Uncertainty records.
- Cross-session registration rides `registration.cross_session.
  align_reconstructed_sessions` — resolved/unresolved/reasons are
  copied verbatim; a failed requested reference falls back to the
  first successful session with a recorded note.

Tests: `tests/test_reconstruction_batch.py` — 12/12 (grouping,
isolation, determinism, bounded memory, unavailable backend,
reference fallback, partial→degraded, budget deferral).

### 2. City evidence ingestion boundary — `evidence/city_import.py`

`import_osm_xml` + `import_geojson`: external map/GIS data becomes
canonical EvidenceAssets (kind=OTHER, provenance OBSERVED, confidence
1.0 for verbatim facts) — NO separate city-data schema. One asset per
OSM way (resolved node geometry, verbatim tags, OSM element ids) plus
a node-collection asset; one asset per GeoJSON feature (geometry
verbatim, properties recorded). Dedup rides the builder's
content-hash contract; malformed input raises CorruptEvidenceError;
unresolvable node refs are recorded (`unresolved_node_refs`) and
produce NO fabricated coordinates.

Real fixture committed: `datasets/city_osm/south_building_campus.osm`
— a REAL OpenStreetMap extract (ODbL, © OpenStreetMap contributors;
Overpass API, osm_base 2026-09-20) around UNC's South Building (the
same structure the photo dataset covers): 733 nodes / 137 ways, sha256
in `datasets/city_osm/MANIFEST.json`.

Tests: `tests/test_city_import.py` — 9/9 on the real fixture.

### 3. Adaptive subdivision — `perception/detail/subdivision.py`

`subdivide_adaptive(points, ...)`: octree subdivision driven by a
MEASURED detail score (curvature via discovery's ONE shared PCA
eigensolver + optional caller-measured importance; density is the
support gate — a cell needs ≥ 2×min_points so children stay
measurable — not a score term). Bounded work: `max_cells` caps visited
cells; frontier expands highest-measured-score first, deterministic
(score, id). Every stop carries a machine-readable reason:
`below_threshold | max_depth | min_points | budget_exhausted |
subdivided`. Advances ledger P7-05/P7-06.

Tests: `tests/test_detail_subdivision.py` — 9/9, including REAL
south-building sparse points (committed dataset) through the same
path with all-finite measured scores.

### 4. Production pipeline measurement — `benchmarks/pipeline_measurement.py`

Per-stage measured record (tracemalloc peak + runtime) over the REAL
dataset through the NEW batch path with the REAL COLMAP backend
(GPU). Latest record:
`datasets/south_building/runs/pipeline_measurement_20260920T104919.json`:

| Stage            | Runtime    | Peak alloc | Measured facts                                   |
|------------------|-----------|------------|--------------------------------------------------|
| ingest           | 0.82 s    | 7.47 MB    | 32 real photos, 0 dups, 0 failed                 |
| admission        | 0.001 s   | 0.01 MB    | 30 accepted / 2 degraded (measured clipping 0.264 > 0.25) / 0 rejected / 0 failed |
| reconstruction   | 24.34 s   | 20.29 MB   | REAL COLMAP 4.2.0 CUDA, 32 frames → 4446 points, status `partial`, outcome `degraded` (honest) |
| uncertainty      | —         | —          | 4446/4446 points carry uncertainty; mean confidence 0.827, min 0.2506 |
| city_ingestion   | 0.045 s   | 1.80 MB    | 137 real OSM features, 0 unresolved refs         |

Coverage: 4446 points from 32 submitted frames. Provenance: backend
name, admission surface, batch surface, dataset manifest recorded in
the measurement JSON. Unavailable COLMAP would be recorded as
`BACKEND_UNAVAILABLE` + probe detail (never faked); `--skip-colmap`
measures the non-COLMAP stages explicitly.

## REAL vs SYNTHETIC status (explicit)

- `datasets/south_building/images` — **REAL** photographs.
- `datasets/city_osm/south_building_campus.osm` — **REAL** external
  map evidence (ODbL).
- `datasets/room_capture_mvs` — **SYNTHETIC** rendered capture
  (unchanged, labeled).
- `tests/test_reconstruction_batch.py` fixtures (PIL photos +
  `ring_test_double` backend) — **TEST FIXTURE / TEST DOUBLE**,
  labeled in code.
- `benchmarks/robustness_scale.py` reconstruction stage — **TEST
  DOUBLE** (labeled; unchanged).

## Tests run

- New this checkpoint: batch 12/12, city import 9/9, subdivision 9/9.
- Full unbounded suite gate: **2130 passed / 3 skipped / 0 failed**
  (161.9 s). The 9 test_system_runtime_proof failures seen in an
  earlier gate were missing locally-generated gitignored artifacts
  (copied from the main checkout; 13/13 green) — fresh clones hit the
  same gap, documented in EXECUTION_STATE.md.

## How to consume (Antigravity / downstream)

1. Multi-session ingest: `import_folder(..., on_error="record")`
   per session folder → `builder.build().to_evidence_items()`.
2. Batch: `from reconstruction.batch import run_batch; report =
   run_batch(items, prefixes=("session_a", "session_b"),
   backends=[ColmapReconstructionBackend(...)],
   reference_session="session_a")` → JSON-serializable report with
   per-session status/points/uncertainty + registration chain.
3. City evidence: `from evidence.city_import import import_osm_xml,
   import_geojson` → canonical assets in the same package builder as
   photos/LAS (dedup and provenance identical).
4. Adaptive subdivision: `from perception.detail.subdivision import
   subdivide_adaptive` → per-cell measured scores + stop reasons;
   feed deep leaves to ROI refinement as bounded local work.
5. Measurement: `python benchmarks/pipeline_measurement.py --gpu`
   writes a fresh measured record; `--skip-colmap` for the fast stages.

## Known limitations

- Batch grouping is prefix-declared (no automatic scene-split yet);
  unassigned items are recorded, never guessed into a group.
- `deferred_capacity` frames are recorded but not retried (bounded
  submission is a hard budget; a resumable queue is future work).
- Subdivision importance is caller-supplied and defaults to 0 (no
  semantic term is fabricated).
- The 10/32 south-building frames that did not register under
  default COLMAP settings remain honestly `partial`/`degraded`.
- LiDAR ingestion remains the existing `import_folder` LAS path
  (header-metadata level); full LAS point-cloud decoding is still
  open (recorded in TASKS.yaml, not silently claimed).

## Next agent

Exact action: run `python benchmarks/pipeline_measurement.py --gpu`
on your machine to reproduce the measured record, then wire
`run_batch` into the vertical slice's reconstruction stage (it is a
drop-in orchestrator composition) and feed `subdivide_adaptive` deep
leaves into `perception.detail.roi.generate_rois` as additional ROI
seeds.
