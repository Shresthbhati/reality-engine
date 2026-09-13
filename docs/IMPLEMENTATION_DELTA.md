# Implementation Delta — Object Perception → WorldIR

**Date:** 2026-09-13. Delta against the previously verified baseline (1041 passed, 1 skipped).

## What changed this cycle

| Area | Previous status | Current status |
|---|---|---|
| Object perception → WorldIR | **MISSING** — `lifting.py`, `object_resolution.py`, `measurement.py` produced real, tested, composable data structures, but nothing wrote them into WorldIR. This was explicitly named as the "next highest-value task" at the end of the prior session. | **NEWLY IMPLEMENTED, NEWLY INTEGRATED.** `evidence/promote_objects.py` — `promote_object_to_entity()` writes a `MergedObjectCandidate` into a WorldIR `Entity` + `Geometry` (BOX from the real union AABB), with measurements on `custom_properties` and a full provenance trail via `Observation` metadata (evidence ids, region ids, observation count). Mirrors `promote_planes.py`'s established pattern exactly. |

## Files added

- `evidence/promote_objects.py`
- `tests/test_promote_objects.py` (8 tests)
- `tests/test_object_pipeline_e2e.py` (2 tests)

## Real functionality delivered

- An object seen in 2+ camera views (segmented + depth-mapped in each) now produces exactly one validated WorldIR entity, not two, three, or zero — verified by an end-to-end test that runs the actual pipeline (two `PinholeCamera`s at different positions, two `DepthMap`+`SegmentedRegion` pairs, real `lift_region_to_3d()` → real `merge_hypotheses()` → real `promote_object_to_entity()`) and checks `world_ir.validation.validate_world_ir()` passes on the result.
- Two visually/spatially distinct objects (different label, different region) correctly produce two separate entities, not one incorrectly merged one.
- Ontology honesty: promoted entities get `EntityType.UNKNOWN` (the WorldIR ontology has no furniture/object categories — only structural ones like WALL/FLOOR) with the real detector/segmenter label preserved verbatim in `semantic_labels` and `name`, rather than guessing a made-up category.

## Tests

10 new tests, all passing. Full suite: **1051 passed, 1 skipped** (1041 baseline + 10), zero regressions.

## Runtime evidence

Ran via `pytest`, not just imported. The e2e test exercises the actual unprojection math (`PinholeCamera.unproject`), the actual union-find clustering (`merge_hypotheses`), and the actual WorldIR validation gate (`validate_world_ir`) — not mocks of any of them.

## Remaining gap (unchanged, named honestly)

This closes the "object hypotheses reach WorldIR" gap but the object-perception pipeline still has never run on a real photo: no real detector/segmenter is executable in this environment (confirmed by direct `segment_anything` import failure in a prior session), and no real dataset is committed to the repo. Every test above uses hand-built `DepthMap`/`SegmentedRegion` fixtures shaped exactly like real backend output — the same testing standard the rest of this repo's pipeline stages (plane detection, room inference) already use before their upstream real backend existed.
