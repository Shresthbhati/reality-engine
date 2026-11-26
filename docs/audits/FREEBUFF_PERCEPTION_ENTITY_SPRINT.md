# Perception & entity completeness sprint — audit and delivery

**Branch:** `feat/ontology-grounded-detection` · **Scope:** production multi-engine
perception path (`apps/api/evidence_world_stage.py` → `perception/multi_engine_stage.py`
→ `MultiEngineOrchestrator` → `perception/*_adapter.py`).

---

## 1. Audit — what the production pipeline actually does

Traced chain, verified in code:

```
photograph
  → PIL open (adapters read the file directly; no shared preprocessing stage)
  → Grounding DINO tiny  (perception/detection_adapter.py::_grounding_dino_run)
  → SAM 2.1 hiera-small  (perception/segmentation_adapter.py::_sam2_run, box-prompted)
  → Depth Anything V2 / MiDaS / Mask R-CNN (depth_adapter, midas_adapter, maskrcnn_adapter)
  → UnifiedEngineOutput / ObservationRecord (perception/engine_evidence.py)
  → MaskStore persistence (perception/mask_persistence.py)
  → Evidence Matrix import (perception/engine_matrix_bridge.py)
  → multi-engine detection fusion + cross-view association (perception/multi_engine_stage.py)
  → WorldIR bridge (engine/multiengine/worldir_bridge.py)
```

Findings, each with the evidence used:

1. **Detection bypassed the canonical ontology entirely.** `_grounding_dino_run`
   prompted with `DEFAULT_SURVEY_QUERIES` — 25 hand-written labels — and never
   resolved its output against `config/ontology.yaml`. `category` was a
   two-way string test against a module-level `_ARCH_LABELS` set, so a socket or
   a switch was classed `"object"` with no ontology identity at all.
2. **Small objects were never queried.** The 25 legacy prompts contained **zero**
   classes marked `small: true` in the ontology (measured: 0/25). Sockets,
   switches, door handles, hinges, vents, taps and valves therefore could not be
   detected by this path at any confidence — not a model limitation, a prompt
   limitation. This is the concrete cause behind the "plug points and switches
   disappear" symptom, together with (3).
3. **Resolution loss before the model.** The detector runs at its processor
   input size. A socket ~40 px wide inside a 4000 px photograph is resampled to
   sub-pixel before any forward pass. Confirmed by the 480×480 benchmark pair:
   tiling gives no benefit at that size (see §6), which is the expected result
   when the image is already at detector resolution.
4. **An ontology bridge existed but was never invoked by production.**
   `perception/detection/extensible_prompter.py::ExtensiblePrompter` already
   exposes `get_prompts(domain, scale, groups, only_small)` and
   `resolve_prediction(label) → PromptTarget`. Grep shows it is referenced only
   by `perception/detection/open_vocab.py`, `extensible_prompter` consumers and
   tests — never by the multi-engine adapters. `perception/detail/crop_pass.py`
   (`HighResCropPass`, ontology-driven small-object crops) is likewise
   disconnected: it emits `world_ir.observations.Observation`, a different type
   from the production `ObservationRecord`, so the production path cannot
   consume it without a bridge.
5. **Classes are inconsistent across the three class lists in the repo.**
   `config/ontology.yaml` (168 nodes, versioned, validated, B03) is canonical,
   but `perception/intelligence/open_vocab.py::generate_survey_prompts` uses a
   different `get_all_taxonomy_classes()` list and `ExtensiblePrompter` also
   indexes `world.ontology.universal_object_ontology`, whose paths (e.g.
   `fixture.light_switch`) are **not** nodes in the canonical ontology, so they
   cannot be looked up afterwards.
6. **Confidence handling was already honest and remains so.** Grounding DINO
   records the model's own score as `confidence` and `raw_score`, with
   `calibrated_probability: None`, `is_calibrated: False` and
   `uncertainty = 1 - score`. No hard-coded or inflated confidence was found in
   the production adapters.
7. **SAM 2 masks do survive persistence** (`perception/mask_persistence.py` +
   `world_ir/mask_store.py`, content-addressed `sha256://` artifact), and the
   staged output reports `partial` rather than `succeeded` when any engine did
   not succeed (`_overall_status`). No fix required; both verified by reading the
   code and by existing tests.

Deliberately **not** done: building a fourth taxonomy, a second detector
pipeline, or a bridge for `HighResCropPass`. Findings 1–2 are fixed inside the
existing adapter; findings 4–5 are recorded as follow-up handoffs (§12).

---

## 2. Files changed

| File | Change |
|---|---|
| `perception/detection/ontology_grounding.py` | **New.** Bounded ontology query builder with a reserved small-class budget, honest `ground_label()` resolution, `tile_windows()` and `nms()`. |
| `perception/detection_adapter.py` | Default prompts now come from the ontology (small classes included); every label is grounded and its provenance stamped; optional tiled inference; query/grounding facts in the summary. Refactored into `_load_detector` / `_detect_in_image` / `_detect_tiled` for testability. |
| `perception/multi_engine_stage.py` | New optional `detection={...}` argument threaded into the shared engine context (`domain`, `threshold`, `query_budget`, `small_share`, `tile_scale`, `tile_overlap`, `queries`, `model_name`). Keyword-only and default-`None`, so every existing caller is unchanged. |
| `config/ontology.yaml` | `fixtures/electrical/socket` and `fixtures/electrical/switch` marked `small: true` (they are sub-0.2 m fixtures and were the only small electrical classes missing the flag). |
| `tests/test_ontology_grounded_detection.py` | **New.** 19 tests (see §8). |
| `docs/audits/FREEBUFF_PERCEPTION_ENTITY_SPRINT.md` | This report. |

---

## 3. Perception capabilities actually integrated

* Ontology-derived, budget-bounded detector prompting (structure → reserved
  small classes → remainder), deterministic ordering.
* Canonical class grounding for **every** emitted detection:
  `ontology_path`, `ontology_label`, `ontology_group`, `ontology_world_type`,
  `is_small`, `min_pixels`, `label_resolved`, `taxonomy_source`.
* Explicit unknown handling: an unresolvable label is preserved verbatim and
  marked `label_resolved: false` / `fallback/unknown_object` — no class invented.
* Tiled small-object inference (native-resolution tiles, boxes mapped back to
  full-image pixels, class-agnostic NMS across tile seams), opt-in via
  `tile_scale`.
* Detection run facts: `n_queries`, `query_source`, `domain`, `n_small_detections`,
  `n_unresolved_labels`, `tile_scale`, `tiles_processed`.

## 4. Previously disconnected capabilities now connected

`ExtensiblePrompter` / `config/ontology.yaml` → production detection. The
ontology had been reachable only from the `open_vocab`/`crop_pass` branch of the
codebase; it is now on the path the job actually runs.

## 5. Taxonomy and class coverage

`config/ontology.yaml`: **168** nodes, version `1.0.0`, validated by
`Ontology.validate()` and covered by `tests/test_b03_unified_ontology.py`,
`tests/test_ontology_loader_task8_1.py`.

Groups represented: architecture (8), furniture (22), decor (9), fixtures (7),
electrical (incl. switchboard modules: socket/switch/dimmer/data/usb),
hvac (vent/duct/radiator/thermostat/ac_unit), plumbing (sink/tap/toilet/bath/
pipe/drain), circulation (stairs treads/risers/stringers, railing glass/posts),
opening (door leaf/frame/hinge/handle/lock, window frame/glazing/mullion/handle),
envelope, exterior (roof/facade/balcony/fence/gate/gutter/chimney), street
(lamppost, bench, bin, barricade, postbox), road (carriageway/footpath/kerb/
drain/marking), vegetation (tree/trunk/canopy/shrub/lawn/flowerbed), signage,
utility, room, finish, building/storey, `misc/unknown`, `fallback/unknown_object`.

`small: true` classes: **29** after this change (was 27; socket and switch added).
Absent vs occluded is represented by `misc/unknown` + `fallback` and the
`label_resolved` flag, not by inventing an object.

## 6. Before / after measurements (real, same 2 images, threshold 0.30)

Images: `.benchmarks/matcher_benchmark_out/pair_00_a.png`, `pair_00_b.png` —
**480×480** each. Model: `IDEA-Research/grounding-dino-tiny`, cached locally.
Run on this machine (`torch 2.14.0+cu130`).

| Metric | Legacy (`DEFAULT_SURVEY_QUERIES`) | Ontology (`query_budget=40`, indoor) | Ontology + 2×2 tiles |
|---|---|---|---|
| detector prompts | 25 | 40 | 40 |
| forward passes / image | 3 | 4 | 4 × 4 tiles = 16 |
| small-class prompts | **0** | **15** | 15 |
| observations | 11 | **25** | 31 |
| small-class observations | **0** | **6** | 5 |
| unresolved labels | 0 | 0 | 0 |
| distinct ontology paths | 7 | **11** | 12 |
| wall time (2 images) | 16.9 s \* | 3.4 s \* | 11.4 s |
| peak VRAM | 1540 MB | 1553 MB | 1553 MB |

\* Wall time is **not** comparable between the first two rows: the legacy run
pays the one-time model load (16.9 s) and the ontology run reuses the loaded
model (3.4 s). VRAM and prompt/pass counts are comparable.

**What these numbers do and do not show.** They show that the detector is now
asked for small classes and now returns them (0 → 6 small observations), and that
detections are grounded to 11 canonical paths instead of raw strings. They do
**not** establish a precision or recall improvement: these benchmark images have
no ground-truth labels, so per-class precision/recall and segmentation IoU are
**unmeasured**. Tiling gave no small-object gain here, which is expected — the
images are already at detector resolution; tiling is for high-resolution capture
and is off by default for that reason.

## 7. Real vs synthetic test breakdown

* **Real model, real photograph:** `test_real_grounding_dino_emits_grounded_observations`
  runs the cached Grounding DINO weights on `pair_00_a.png` (passes here; skips
  with a recorded reason if the weights are unavailable).
* **Deterministic unit tests (no model):** the remaining 18 tests exercise the
  query budget, grounding, tiling, NMS and the adapter's mapping with a stub
  detector. The stubbed tests verify *our* mapping logic, not model behaviour,
  and say so in the module docstring.

## 8. Test commands and complete results

```
python -m pytest tests/test_ontology_grounded_detection.py \
                 tests/test_b03_unified_ontology.py \
                 tests/test_ontology_loader_task8_1.py \
                 tests/test_open_vocab_detector_task8_2.py \
                 tests/test_high_res_crop_pass_task9_1.py -q
# 42 passed, 1 warning in 25.65s

python -m pytest tests/test_detection_pipeline_engine.py \
                 tests/test_engine_reachability_and_wired_engines.py \
                 tests/test_mask_lifecycle_matrix.py -q
# 33 passed, 13 warnings in 233.73s
```

## 9. Runtime and VRAM

Per §6: ~1.55 GB peak VRAM (unchanged within noise vs 1.54 GB legacy); 4 forward
passes per image at `query_budget=40` (vs 3 legacy); 16 passes per image at
`tile_scale=2` on a 480 px image (hence off by default).

## 10. Known limitations and unresolved defects

1. **No ground-truth metrics.** Precision/recall/IoU are unmeasured on these
   images. Do not read §6 as an accuracy claim.
2. **Duplicate socket nodes.** `fixtures/electrical/socket` (canonical, B03) and
   `electrical.switchboard.module.socket` both exist; first-definition-wins means
   `"power outlet"` resolves to the former. Consolidation needs an ontology
   decision and is not made unilaterally here.
3. **Loose alias matching in `ExtensiblePrompter.resolve_prediction`.** Substring
   matching maps `"zebra"` → `road.marking`. Grounding prefers exact canonical
   resolution first, so this only affects labels the canonical ontology misses;
   it is still a source of wrong fine-grained classes.
4. **`ExtensiblePrompter` indexes paths absent from the canonical ontology**
   (e.g. `fixture.light_switch`). `ground_label` only accepts prompter targets
   that exist canonically, so such labels degrade to their canonical parent or to
   `fallback` rather than carrying an unresolvable path.
5. **Tiling unproven on high-resolution captures** — no high-resolution
   real-capture set was available in this session. The code path and coordinate
   mapping are tested; the recall benefit is not measured.
6. **`domain` defaults to `"all"`**, so outdoor prompts are included in indoor
   scans. Callers can pass `detection={"domain": "indoor"}`; automatic domain
   inference was not invented.
7. `perception/detail/crop_pass.py` remains disconnected from production (see §4
   and §12).

## 11. PR

See the pull request for `feat/ontology-grounded-detection`; commit SHA is in the
PR's commit list.

## 12. Handoffs

* **Claude (canonical geometry / contracts):** no geometry, camera, BA or WorldIR
  contract was changed. `ObservationRecord.metadata` gained new keys only
  (`ontology_*`, `is_small`, `min_pixels`, `label_resolved`, `taxonomy_source`);
  no field was renamed, removed or retyped.
* **KiloCode (RCPAS / acceptance):** `run_multi_engine_for_job(detection={...})`
  is the supported way to set detection domain/budget/tiling per run; defaults
  change only the prompt set (ontology, 40 prompts) and add metadata.
* **Cline (API/product):** `category` still takes only `"architecture"` /
  `"object"`; the finer class is in `metadata["ontology_group"]` and
  `metadata["ontology_path"]`. Nothing in `apps/api` needs to change.
* **OpenCode (platform/reliability):** detection now performs
  4 forward passes per image by default (was 3). If a job budget assumes 3, pass
  `detection={"query_budget": 25}`.
* **Ontology owner:** consolidate the duplicate socket node and remove
  `ExtensiblePrompter`'s non-canonical paths (§10.2, §10.4) — a taxonomy
  decision, deliberately not taken here.
