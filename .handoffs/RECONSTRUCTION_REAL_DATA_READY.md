# HANDOFF

Agent: Freebuff (reconstruction + perception owner)
Branch: agent/freebuff-reconstruction-perception
Commit: (see git log — "feat(reconstruction): reproducible real-data validation (south-building) + batch-survival import")
Checkpoint: RECONSTRUCTION_REAL_DATA_READY
Status: CHECKPOINT_READY

## What exists

A reproducible REAL-data validation path for the evidence → reconstruction
chain, executable by any developer from repository contents alone:

1. **Committed real dataset** — `datasets/south_building/`
   - 32 real photographs (South Building, UNC Chapel Hill; Panasonic
     DMC-TZ3; public set distributed by the COLMAP authors — credit
     Christopher Zach). Deterministic derivative: full 128-image set
     sorted by name, subset, downscaled to width 1024 (LANCZOS), JPEG q90.
   - Sparse reference model (`cameras.txt`/`images.txt`/`points3D.txt`)
     FILTERED to the 32 committed images — COLMAP point/observation ids
     preserved, so GT comparison lines up by camera id.
   - `MANIFEST.json`: source URL, original zip sha256, per-image sha256 +
     resolution, derivative recipe, camera metadata, ground-truth kind.
   - `scripts/fetch_south_building.py` — reproduces or verifies the subset:
     `python scripts/fetch_south_building.py verify` (needs only the
     committed tree + manifest) or `download` (rebuilds from source zip).
     Verified: 32/32 images match MANIFEST.json.

2. **Real end-to-end driver** — `scripts/run_south_building_e2e.py`
   Chain exercised with REAL backends (no test doubles):
   evidence.importers.import_folder → EvidencePackage → EvidenceItems →
   reconstruction.robustness_admission.admit_for_reconstruction →
   ColmapReconstructionBackend (real COLMAP binary, feature extraction →
   matching → mapping) → registration/fusion → honest run record JSON in
   `datasets/south_building/runs/`.
   Measured run (`run_20260919T203748.json`): COLMAP 4.2.0 CUDA, 429.5 s,
   22/32 poses registered, 4510 points, outcome `degraded` with machine-
   readable reason (`registration_status=partial`), GT comparison over 22
   common images: 8.94° median / 9.26° max rotation disagreement (GT
   conjugated to camera-to-world; Sim(3) translation alignment NOT
   claimed), fusion: 22 observations, 0.1449 fused value, 197 conflicts
   PRESERVED.

3. **Batch-survival import policy** — `evidence/importers.py`
   `import_folder(..., on_error="record")`: a structurally corrupt file is
   recorded in `FolderImportReport.failed_paths` as (path, reason) and the
   remaining files still import. Default `on_error="raise"` unchanged.
   Red-first driver: `tests/test_bad_evidence_real.py::
   test_corrupt_file_does_not_lose_the_whole_batch` (7/8 healthy frames
   survive one destroyed header; failure recorded with reason).

4. **Quality metrics now travel the whole chain** — commit e5361fc
   `EvidenceAsset.to_evidence_item()/to_evidence_items()` were dropping the
   importer-measured pixel metrics; on real captures the robustness
   classifier saw nothing and gated nothing. Fixed red-first
   (tests/test_evidence_packages.py).

## REAL vs SYNTHETIC status (explicit)

- `datasets/south_building/images` — **REAL** photographs (public).
- `datasets/south_building/sparse` — **REAL reference** reconstruction
  shipped with the public set (not engine-produced).
- `datasets/room_capture_mvs` (other worktrees) — **SYNTHETIC** rendered
  capture with ground truth; remains the synthetic reference fixture.
- `tests/test_bad_evidence_real.py` variants — **REAL** bytes transformed
  at test time (blur/exposure/truncation), never fabricated pixels.
- `benchmarks/robustness_scale.py` reconstruction stage — **TEST DOUBLE**
  (labeled in code and output); measures admission/orchestration scaling,
  not COLMAP.

## Failure matrix (measured, canonical vocabulary)

| Defect (on real bytes)        | Outcome            | Reason carries                      |
|-------------------------------|--------------------|-------------------------------------|
| strong blur (BoxBlur 6)       | REJECTED/DEGRADED  | measured laplacian_variance < 150   |
| exact duplicate bytes         | deduped at import  | duplicates_skipped (path, orig id)  |
| truncated JPEG (header lives) | FAILED             | decode failure note (unmeasured)    |
| destroyed header              | import failure     | failed_paths (path, reason); batch survives with on_error="record" |
| exposure pushed to white rail | REJECTED/DEGRADED  | measured clipped_fraction > 0.25    |
| two far-apart ring views      | item-level pass    | connectivity is a RUN-level property (test asserts the gate reports 2) |
| missing metrics               | UNRESOLVED         | never vetoed, never upgraded        |

## Tests run

- `tests/test_bad_evidence_real.py` — 8/8 green (real-data-gated; skips
  with the exact dataset path when absent).
- `tests/test_evidence_packages.py` + fusion/session/sources — 90 green
  (bridge fix regression).
- Full unbounded suite — **1951 passed / 3 skipped / 0 failed** in 762 s
  (final gate; one transient failure — the new `failed_paths` report key
  vs the serialized-shape assertion in test_media_preprocessing — fixed
  by extending that assertion). An earlier gate's single
  `test_arch_benchmark` perf-guard failure was CPU contention from
  concurrent agent suites, not a code regression.

## Runtime verification

- `python scripts/fetch_south_building.py verify` → 32/32 match.
- `python scripts/run_south_building_e2e.py` → real COLMAP run, record
  written, honest degraded outcome (see Measured run above).

## How to consume (Antigravity / any downstream)

```bash
# 1. dataset integrity (fast, offline)
python scripts/fetch_south_building.py verify

# 2. real end-to-end (needs a local COLMAP on PATH; ~7 min CPU/GPU)
python scripts/run_south_building_e2e.py

# 3. read the honest outcome
#    datasets/south_building/runs/run_<timestamp>.json
#    outcome.outcome / outcome.reason / gt_comparison / fusion
```

UI surfaces that want "real reconstruction verified" status should read the
run record JSON: `outcome.outcome` ∈ {complete, degraded, failed} (never
fake-successful), `gt_comparison.registered_of_gt`, `fusion.conflicts_preserved`.

## Known limitations

- GT translation comparison requires Sim(3) alignment — not performed,
  not claimed (rotation-only comparison is alignment-free).
- 10/32 images unregistered by COLMAP mapping on this hard outdoor ring —
  recorded, not hidden; outcome honestly `degraded`.
- The E2E driver requires a local COLMAP install; it is NOT wired into CI
  (real-model tests stay environment-gated by repo policy).
- `on_error="record"` callers own checking `failed_paths` — documented at
  the API and in the test docstring.

## Next agent

Exact next actions, in order:

1. **Admission value E2E on real spend** — run south-building twice
   (RUN A: admission off, RUN B: on) and measure actual COLMAP minutes
   saved vs the item-count delta already proven. The item-level workload
   delta is proven (test_admission_value); GPU/CPU-minute savings are NOT
   yet claimed with measurement.
2. **Densify the committed subset** (50 images, stride 2) if you need more
   registration overlap — `fetch_south_building.py download` + recipe
   parameterization; keep MANIFEST checksums authoritative.
3. **Sim(3) GT translation comparison** in the E2E driver (umeyama on
   registered poses) to close the last honesty gap in gt_comparison.
4. Feed `run record outcome` into the studio/session UI status contract
   (coordinate with the desktop owner; do not invent a parallel schema).
