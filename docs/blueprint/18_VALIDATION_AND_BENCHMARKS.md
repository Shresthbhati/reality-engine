# 18 — Validation and benchmarks

"It returned 200" is not validation. We test **physical correctness**.

## Test layers

| Layer | What | Where | Runs |
|-------|------|-------|------|
| Unit | one function, synthetic input | `tests/` (pytest) | every commit |
| Contract | WorldIR invariants, API schemas, import boundaries | `tests/`, `frontend/tests/contract` | every commit |
| Scenario | adversarial reconstruction cases A1–A20 | `tests/scenarios/` + `datasets/` | every PR that touches the pipeline |
| Benchmark | metrics on measured / ground-truth data | `benchmarks/` | every model or stage change; results committed |
| E2E | real browser journeys | `frontend/e2e` (Playwright) | every PR that touches the API or UI |
| Acceptance | the living-room demo ([00](00_VISION.md)) | scripted + screenshots | each phase gate |

## Adversarial scenarios (permanent regression suite)

| ID | Scenario | Expected |
|----|----------|----------|
| A1 | Same wall from different angles | one Surface |
| A2 | Same door in 5 images | one Entity, 5 observations |
| A3 | Window partly visible in 3 images | one Entity, PARTIALLY_OBSERVED |
| A4 | Two similar but distinct doors/chairs | two Entities (DIFFERENT_ENTITY) |
| A5 | Furniture overlapping in image space | separate Entities; no PHYSICAL_INTERSECTION unless real |
| A6 | Furniture occluding a wall | wall continuous; hidden part OCCLUDED/INFERRED, no hole |
| A7 | Corner seen in one image only | kept, low confidence |
| A8 | Opposite viewing directions | one room, cameras on both sides |
| A9 | Same room, two sessions | one World/Space; the session attaches |
| A10 | New image, big viewpoint change | registers, or is rejected with a reason |
| A11 | No shared features | rejected, never force-merged |
| A12 | Missing wall added later | wall added; the rest unchanged |
| A13 | Mirror | no phantom room/objects behind it |
| A14 | Glass | opening/glazing, not an opaque wall |
| A15 | Blank white walls | lower confidence, no invented features |
| A16 | Repeated patterns | no false merges or registrations |
| A17 | Low light / blur | lower evidence confidence, no unjustified dimensions |
| A18 | Wrong camera pose injected | caught by the consistency check |
| A19 | Adjacent room proposed for merge | refused |
| A20 | Contradictory geometry | CONFLICTED, both kept |

Each scenario has input images (real, or synthetic via `synthetic/`), the
expected WorldIR assertions, and a pass/fail test. If the real data is missing,
mark the scenario **not covered** — never fake-pass it.

## Metrics (record on every run)

| Area | Metric |
|------|--------|
| Registration | registered image ratio, mean reprojection error |
| Matching | verified inlier ratio |
| Geometry | point-to-surface / Chamfer distance vs ground truth |
| Dimensions | absolute + relative error vs tape measure |
| Architecture | plane angle error, wall position error |
| Detection | precision and recall per class |
| Segmentation | mask IoU |
| Identity | duplicate rate, false-merge rate, false-split rate (or IDF1) |
| Coverage | observed surface fraction |
| Performance | runtime per stage, peak RAM, peak VRAM |
| Reliability | failed-job rate, recovery success |

Thresholds come from measured baselines (Phase 2), never chosen to make tests
pass. A change that worsens a metric beyond noise needs an explicit reason in the PR.

## Datasets

- `datasets/south_building/images` (bundled COLMAP outdoor sample) — registration smoke test.
- The living-room photo set — identity, coherence and workflow (no scale truth).
- **3–5 tape-measured rooms** captured by the team — the only dimensional truth for v1.
  Record room L×W×H, 5+ object dimensions, the photo set and the capture date.
- Later: ScanNet++, ARKitScenes, Structured3D (check licences before use).
- Splits by room/property, never by photo.
