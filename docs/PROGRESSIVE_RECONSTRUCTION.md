# Progressive reconstruction — the one-action product path

Status: **PARTIALLY VERIFIED** (see "What is verified" / "What is not").

Contract: a person gives Reality Engine one or more photographs of a real
place. The engine immediately builds the best spatial representation the
evidence honestly supports. More evidence improves **the same persistent
world**. Nothing is fabricated; unknowns stay unknown.

## The user-facing model

```
PHOTOS -> EVIDENCE -> RECONSTRUCTION -> WORLD -> REFINE -> INSPECT -> EXPORT
```

World / Session / Job / Version still exist (advanced routes are unchanged) but
the user never creates or attaches them:

| Action | Route | Effect |
|---|---|---|
| Drop photos (no world) | `POST /api/reconstructions` | creates World + Session, stores evidence, queues the job |
| Drop more photos | `POST /api/reconstructions` with `world_id` | joins the **same** World; the job rebuilds it from the union of all its evidence |
| Read everything the UI shows | `GET /api/worlds/{id}/status` | state, model honesty, evidence quality, guidance, versions, failure |

## Levels (cumulative capabilities achieved, never claimed)

| Level | Name | Produced by |
|---|---|---|
| 0 | single-image visual hypothesis | `engine/pipeline/single_image.py` |
| 1 | multi-image rough (2 cameras registered) | COLMAP via `vertical_slice` |
| 2 | sparse photogrammetric (>= 3 registered) | COLMAP via `vertical_slice` |
| 3 | dense reconstruction | **not wired** (dense MVS exists but is off by default) |
| 4 | semantic / topological refinement | compiler produced rooms / storeys / corridors |
| 5 | incremental evidence refinement | built on top of an earlier version's evidence |

`engine/pipeline/progressive.py::run_progressive` tries the highest available
level and falls back. A COLMAP failure is a recorded `attempt`
(`report.attempts`), never an empty world and never a success.

`model_state` shown to users: **ROUGH** (level <= 1), **PARTIAL** (real geometry,
not everything placed or covered), **REFINED** (level >= 4 **and** every image
placed). "A six-image model is not a complete digital twin" is enforced by the
grading, not just by wording.

## Single-image bootstrap (Level 0)

`image -> MiDaS relative depth -> visible-surface points -> RANSAC planes ->
wall/floor/ceiling classification -> WorldIR`.

* Every entity is `INFERRED` / `ESTIMATED` / `UNKNOWN`, with confidence capped
  at 0.35 (`CONFIDENCE_CEILING`) regardless of how clean a fit looks.
* Scale state is `relative`. No metric dimensions.
* An explicit `boot-unobserved` entity (provenance `UNKNOWN`, confidence 0)
  records rear/side/hidden surfaces. Nothing hidden is invented.
* Assumptions are recorded under `metadata["bootstrap"]`: inverse-depth offset,
  camera (EXIF 35 mm focal length or an ASSUMED 60 degree field of view),
  level-camera gravity assumption.
* Not attempted (stated in the world): window/door/opening detection, object
  detection, metric scale.

## Evidence handling

* `evidence/image_check.py` — decode + classify **before** any expensive stage.
  Bytes that are not images never reach COLMAP (measured: 3 fake "photos" hung
  the old pipeline for > 90 s). Classes: `photograph` (camera EXIF),
  `photograph_unverified` (decodes, natural statistics, no EXIF), `flat_graphic`
  (plans/infographics-like), or an uploader-declared `floor_plan` / `render` /
  `infographic` / `historical_photo`. Only the first two feed geometry; the rest
  are kept as **context evidence** and the class survives into WorldIR
  (`metadata["evidence"]`) and the version report.
* `evidence/contribution.py` — per image: `first`, `new_view`, `redundant`,
  `disconnected`, from SIFT + RANSAC-verified matches (raw counts are kept).
  Image count is never the contract: "4 new images added. 3 add a new
  viewpoint. 1 mostly repeats an existing view."
* Quality bars (Coverage / Overlap / Spatial diversity) are coarse levels with
  a stated basis, or `unknown` ("not measured") — no invented percentages.
  When SfM fails, Overlap is downgraded to `low` regardless of feature matches.

## Guidance

`engine/pipeline/guidance.py` — every item is `{kind, message, basis}` and fires
only on something measured: unplaced images (named), one-sided capture
(`coverage_degrees`: bearing of camera positions around the scene for
object-centric captures, viewing directions for interiors), frame-edge
truncation of a detected surface (single view), missing floor/ceiling,
redundant batches.

## Versions and diff

Every job that adopts a version appends to the WorldStore lineage; V1, V2, ...
are numbered by lineage. Earlier versions are immutable and can be inspected
(`?version=`). `WorldDiff.categories()` counts changes per kind — identity,
geometry, transform, semantic, confidence, uncertainty, provenance, topology,
properties — attributing each change only to the field that actually differed
(a confidence-only edit is never a geometry change).

## Defects found by running the real thing (and fixed)

| # | Defect | Effect | Fix |
|---|---|---|---|
| 1 | `apps/api/jobs.py` import block pasted mid-function | API did not import | imports moved to module top |
| 2 | `reconstruction/proc.py::run_owned` polled `poll()` without draining pipes | **every** real COLMAP run deadlocked (colmap.exe alive 16 min, 38 s CPU) | `communicate(timeout=...)` slices; regression test with 2 MB of output |
| 3 | `run_owned` called `register_worker()` while holding the non-reentrant `_lock` | any job-owned subprocess self-deadlocked | removed the redundant nested registration (job-level registration is done by `set_current_job`) |
| 4 | frame canonicalization assumed the dominant plane is the floor | a building facade became the ground; model on its side | second prior: mean camera up-vector, plus wall snapping for pitched cameras; guarded by orientation-diversity + coherence + disagreement thresholds |
| 5 | Studio `parsePly` accepted only binary PLY, `/points` emits ASCII | **point clouds never rendered** in the Studio | loader accepts ASCII + unaligned binary |
| 6 | Next proxies dropped `?version=` | version switching impossible | forwarded |
| 7 | `Uncertainty.confidence` defaulted to 1.0 | unstated uncertainty read as certain | default 0.0; identity transform states its exactness explicitly |
| 8 | building/storey entities `INFERRED` at confidence 1.0; opening default 1.0; commit route coerced invalid confidence to 1.0 | fabricated certainty | derived from members / 0.0 / 422 |
| 9 | `apps/cli/api_bridge.py` fell back to `datasets/room_capture/.../points.ply` | a fixture could be served as a version's reality | removed |

## Golden datasets and rights

Only one **real photographic** dataset exists in the repository:

* `datasets/south_building/` — 32 photographs (1024 px derivative of the public
  COLMAP "South Building" example set, UNC Chapel Hill; provided by Christopher
  Zach per the COLMAP datasets page; source URL and SHA-256s in `MANIFEST.json`).
  Public research data distributed by the COLMAP authors.

Windows into it (indices in name-sorted order, chosen from the measured
pairwise SIFT/RANSAC matrix, not by eye):

| Name | Images | Measured behaviour (real COLMAP 4.2.0, CPU) |
|---|---|---|
| ONE_PHOTO | `[6]` | level 0 rough model in ~14 s |
| SIX | `[15:21]` | 6/6 cameras registered, ~1,130 points, 6 wall planes |
| ADD_4 | `[21:25]` | V2: 9 of 10 registered |
| ADD_10 | `[5:15]` | V3: 13 of 20 registered |
| HARD | `[0:6]` | wide-baseline; real COLMAP finds **no initial pair** -> honest fallback to level 0 |

**Not available (UNVERIFIED):** a real *corridor* dataset and a real *room*
dataset. The `room_capture` datasets named in older notes are synthetic /
rendered, are not committed, and are not used by any claim here.

## Commands

```bash
# fast, model-free proofs
python -m pytest tests/test_progressive_units.py tests/test_proc_ownership.py tests/test_world_diff.py

# the real-photo journeys (real COLMAP + real MiDaS; ~6.5 minutes on CPU)
python -m pytest tests/integration/test_progressive_product_journey.py -v
```

## What is verified / what is not

Verified by execution: everything the golden suite asserts (one photo, six
photos, 6 -> 10 -> 20 on one world, bad evidence, hard case, engine crash,
restart), the model-free unit tests, the frontend type-check and production
build, and a real-browser run of: drop -> world created -> rough model
rendered -> add photos -> V2 -> version switch -> diff -> mobile capture flow.

Not verified / not done: video ingestion (no frame-extraction path), dense
Level 3, metric scale (needs a measured reference), window/door detection from
one image, per-version evidence panel while viewing an old version (the
status panel describes the *current* model), real corridor and room datasets,
GPU paths, a multi-user/production deployment.
