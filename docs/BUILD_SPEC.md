# Reality Engine build spec

The implementation spec for the engine. `RELEASE.md` says *what* ships in December; this file says *how to build it*, precisely enough that an implementer does not choose data structures or algorithms — they are chosen here.

Baseline: `chore/december-release` @ `04362f72` (on top of `main` @ `3ceb60b5`). Every symbol named below was read in the source on 2026-11-15.

## 0. The goal, and what is honest about it

**Goal.** Photos or video of a place go in. Out comes a *memory* of that place: every room, surface, object, part and attribute, each with a position, a size, the photos that prove it, and how sure the engine is. You can ask it "where is the snack drawer", open a cupboard and see its handle and lock type, and export the building.

**What December can be.** One building interior at a time, on one 6 GB GPU, with: stable geometry, every detected object as one entity across all photos, parts and attributes for the object families in section 4, honest unknowns, corrections, versions, exports. Sections 3–5 build exactly that.

**What December cannot be.** "Every object ever made" and "the Taj Mahal from a phone" are not coding problems alone — they need labelled data, measured ground truth and capture hardware (drone, LiDAR) that do not exist in this project yet. Section 6 lists what each one needs and the hooks this spec leaves for it. Nothing in sections 3–5 has to be rewritten to get there.

**Three rules that override everything below.**
1. A thing the engine did not see is `UNKNOWN`, never a guess shown as fact. A closed drawer has unknown contents.
2. One physical object = one entity, however many photos show it. A detection is an *observation*, never an entity.
3. No model with a non-commercial or undeclared licence in the default path (`perception.model_registry.assert_production_licences`).

## 1. How data moves (types at every hop)

```
photos / video frames            bytes on disk, content-addressed by sha256
  │  s01 evidence_resolve        → EvidenceItem list (id, path, sha256, EXIF)
  │  s02 evidence_quality        → QualityRow per image (blur, exposure, local sharpness grid)
  │  s04 pose                    → cameras: dict[image_id → PinholeCamera], sparse points (N×3 float32)
  │  s05 scale                   → one metres_per_unit float + ScaleState
  │  s06 depth                   → depth map per image (H×W float16, metres, memory-mapped .npy)
  │  s07 dense                   → fused cloud (VoxelCloud, section 2.4)
  │  s08 mesh                    → mesh (V×3 float32, F×3 int32)
  │  s09 perception              → ObservationTable (section 2.2)  ← the per-photo record of everything seen
  │  s10 detail / layout         → planes, room footprint, wall/floor/ceiling slabs
  │  association                 → entities: observations grouped into physical objects (section 3, B07–B08)
  │  s11 compile_validate        → WorldIR (entities, relationships, provenance)
  │  s12 quality_report          → per-region completeness, per-entity uncertainty
  │  s13 changeset               → proposed changes against the World head → new immutable version
```

Stages live in `engine/pipeline/stages/` (`base.py` contract, `runner.py`, `registry.py`). A stage reads and writes **artifacts by hash**, never live Python objects (work package B01 finishes this).

## 2. Core data structures (use these, do not invent others)

### 2.1 Choosing a structure — the rule table

| You need to… | Use | Why / cost |
|---|---|---|
| Hold N same-typed numeric records (points, boxes, scores) | `numpy` arrays, one per field (struct-of-arrays) | vectorised; no Python loop over N |
| Look up by id | `dict[str, int]` mapping id → row index into those arrays | O(1) |
| Group rows by a key (image, entity, voxel) | `np.unique(keys, return_inverse=True)` then `np.add.at` / `np.bincount` | O(N log N), no Python loop |
| Always take the best next item (refinement tasks, next image to register) | `heapq` with tuples `(-priority, seq, item)` | O(log N) push/pop; `seq` breaks ties deterministically |
| Process in arrival order (tiles, crops, BFS over a hierarchy) | `collections.deque` | O(1) both ends |
| Merge things into groups as evidence arrives (tracks, identities) | union-find: `parent: np.ndarray[int32]`, path compression, union by size | near O(1) per op |
| One-to-one matching between two sets (detections ↔ entities, prediction ↔ truth) | cost matrix + `scipy.optimize.linear_sum_assignment` | optimal; O(n³), fine for n < 500 |
| Nearest neighbours in 3D | `scipy.spatial.cKDTree` built once per cloud | O(log N) query |
| "What is near this box?" over entities | uniform grid: `dict[(i,j,k) → list[entity_row]]`, cell = 0.5 m | O(1) per cell; no new dependency |
| Sparse 3D occupancy / fusion | voxel hash: pack `(i,j,k)` into one `int64`, group with `np.unique` | section 2.4 |
| A hierarchy (ontology, parts, building → room → object) | `parent: dict[str, str]` + `children: dict[str, list[str]]`, built once | O(depth) ancestors |
| Dependencies between artifacts or stages | DAG as `dict[node → list[node]]`; order with Kahn's algorithm; invalidate with reverse BFS | O(V+E) |
| Keep the K most recently used models on the GPU | `collections.OrderedDict` as LRU | O(1) move/evict |
| A binary mask | run-length encoding (section 2.3), never a PNG per mask, never a Python list of pixels | ~100× smaller |
| An immutable version | new object + content hash; never mutate in place | diff = compare `dict[id → hash]` |

Never: a Python `list` of dicts for anything over a few hundred rows; nested loops over points or pixels; `pickle` for stored artifacts (use `.npy`/`.npz` and JSON); floats as dict keys.

### 2.2 `ObservationTable` — everything seen, per photo

New file `world_ir/observation_table.py`. It stores the existing `world_ir.observations.Observation` records in columns.

```python
@dataclass
class ObservationTable:
    obs_id:      np.ndarray  # (N,) '<U32'
    image_id:    np.ndarray  # (N,) '<U32'
    class_path:  np.ndarray  # (N,) '<U96'   ontology path, e.g. "furniture/storage/almirah"
    parent_obs:  np.ndarray  # (N,) int32    row of the whole this is a part of, -1 if none
    bbox_px:     np.ndarray  # (N,4) float32 x0,y0,x1,y1
    score:       np.ndarray  # (N,) float32  raw detector score (not a probability)
    mask_ref:    np.ndarray  # (N,) int32    row in the mask store, -1 if no mask
    emb_row:     np.ndarray  # (N,) int32    row in the embedding matrix, -1 if none
    world_pos:   np.ndarray  # (N,3) float32 NaN where not lifted to 3D
    sigma_m:     np.ndarray  # (N,) float32  NaN where unknown
    visibility:  np.ndarray  # (N,) int8     0 full,1 partial,2 occluded,-1 not assessed
    entity_row:  np.ndarray  # (N,) int32    filled by association; -1 before
    model:       np.ndarray  # (N,) '<U48'   "name@version"
    attrs_json:  np.ndarray  # (N,) object   JSON string of attribute hypotheses (B05)

    by_image:  dict[str, np.ndarray]   # image_id → row indices (built in __post_init__)
    by_entity: dict[int, np.ndarray]   # entity_row → row indices (rebuilt after association)
```

- `save(path)` writes one `.npz` (arrays) — `load(path)` rebuilds the two index dicts with `np.unique(..., return_inverse=True)`.
- `to_observations() -> list[Observation]` and `from_observations(list[Observation])` convert to the existing dataclass; nothing else in the code base changes type.
- `to_inventory() -> list[dict]` emits the benchmark contract in `benchmarks/reality_benchmark/inventory.py` (`REQUIRED = image_id, entity_id, category, subtype, bbox`; every `MEASURABLE` field present, `None` with an `unavailable` reason when not measured).

### 2.3 Masks and embeddings

- **Masks:** `world_ir/mask_store.py`. One `.npz` per job: `counts` (concatenated uint32 run lengths, column-major like COCO), `offsets` (M+1 int64 into `counts`), `shape` (M×2 uint16). `encode(mask: np.ndarray[bool]) -> np.ndarray[uint32]` uses `np.diff` on the flattened mask; `decode(row) -> np.ndarray[bool]` uses `np.repeat`. No per-pixel Python.
- **Embeddings:** one `float16` matrix `(N_emb, 384)` for DINOv2 ViT-S/14, L2-normalised at write time, saved as `.npy` and opened with `mmap_mode="r"`. Cosine similarity is then a matrix product. `Observation.embedding_ref` holds `"<artifact_sha>#<row>"`.

### 2.4 `VoxelCloud` — fused geometry

New file `reconstruction/fusion/voxel_cloud.py`.

```python
@dataclass
class VoxelCloud:
    voxel_m: float
    key:     np.ndarray  # (V,) int64  packed (i,j,k), sorted ascending
    mean:    np.ndarray  # (V,3) float32
    normal:  np.ndarray  # (V,3) float32, NaN where < 3 points
    rgb:     np.ndarray  # (V,3) uint8
    count:   np.ndarray  # (V,) uint32
    sources: np.ndarray  # (V,) uint8  bitmask: 1 sfm, 2 mvs, 4 mono-depth, 8 device-depth
    var_m2:  np.ndarray  # (V,) float32 positional variance of the voxel mean
```

- Packing: `i,j,k = floor(p / voxel_m).astype(int64) + 2**20` (offset keeps them non-negative); `key = (i << 42) | (j << 21) | k`. 21 bits per axis.
- `integrate(points, rgb, sigma_m, source_bit)`: compute keys for the new points; concatenate with existing `key`; `np.unique(return_inverse=True)`; accumulate inverse-variance-weighted sums with `np.add.at`; recompute `mean`, `var_m2 = 1 / Σ(1/σ²)`, `count`, OR the `sources` bits. O(N log N). Never loops over points.
- `lookup(points) -> rows` uses `np.searchsorted(self.key, keys)` (O(log V) each).
- A voxel counts as **observed surface** only if `count ≥ 3`. Space with no voxel is `NOT_OBSERVED`, not empty.

### 2.5 Entities, parts, relationships

Existing `world_ir.entity.Entity` (`id, type, transform, semantic_labels, relationships, provenance, uncertainty, extra_fields`). Conventions fixed here:

- `Entity.type` = ontology path. Parts are entities too, linked with `Relationship(kind="part_of", target_id=<whole>)`.
- Allowed relationship kinds (closed set, validated in `world_ir/validation.py`): `part_of`, `attached_to`, `on`, `inside`, `contains`, `adjacent_to`, `connects`, `hosted_by`, `supports`.
- `extra_fields["observation_rows"]`: list of rows in the job's `ObservationTable`.
- `extra_fields["attributes"]`: `{name: {"value": str | None, "p": float, "basis": "visual" | "user_stated" | "ocr", "obs": [obs_id…]}}`. `value: None` means unknown; it is written, not omitted.
- Spatial index for queries: uniform grid `dict[(i,j,k) → list[entity_id]]`, cell 0.5 m, built from entity AABBs when a world is loaded.

## 3. Work packages

Each package is one branch and one commit: `build/<ID>`. Order is dependency order. Run a package only when its **Needs** are merged. Tests named here are written first and must fail before the code exists. Regression after every package:
`python -m pytest tests/test_golden_path_api_to_glb.py tests/test_durable_job_queue.py tests/test_world_adoption_api.py tests/test_stage_runner.py -q --tb=short -p no:warnings`

### B01 — Stages exchange artifacts; a crashed job resumes
**Why:** `apps/api/jobs.py::_RecordOnlyStore.lookup()` returns `None` because stages pass live objects in `ctx.state`.
**Files:** `engine/pipeline/stages/base.py`, `runner.py`, each `sNN_*.py`, new `engine/pipeline/stages/artifacts.py`, `apps/api/jobs.py`.
**Build:**
- `artifacts.py`: `put_array(arr) -> sha`, `get_array(sha, mmap=True)`, `put_json(obj) -> sha`, `get_json(sha)`, `put_npz(**arrays) -> sha`, `get_npz(sha)`. Hash = sha256 of the bytes written; path = `<artifact_root>/<sha[:2]>/<sha>.<ext>`; write to a temp file then `os.replace` (atomic). Reuse the store behind `apps.api.storage.store_bytes` — do not add a second store.
- Each stage declares `inputs: tuple[str, ...]` and `outputs: tuple[str, ...]` (logical artifact names). `run(ctx)` reads `ctx.artifacts[name]` and returns hashes in `StageResult.outputs`. Convert one stage per commit, in order s01 → s13.
- Replace `_RecordOnlyStore` with a store whose `lookup(job_id, name, inputs_hash)` returns the stored `StageResult` when status is `succeeded` **and** every output hash resolves on disk.
**Structures:** stage order = Kahn's algorithm over the `inputs/outputs` DAG (deterministic: ready-set kept in a `heapq` keyed by stage index).
**Tests:** `tests/test_stage_resume.py` — run three counted stages, crash the third once, rerun: counters `[1,1,2]`; delete one output file → that stage reruns; changing a param changes `inputs_hash` → rerun.
**Done:** `RELEASE.md` "Known open facts" line about crash-resume can be deleted.

### B02 — Persist every observation (`ObservationTable`)
**Needs:** B01. **Files:** new `world_ir/observation_table.py`, `engine/pipeline/stages/s09_perception.py`, `apps/api/routes_worlds.py` (one read endpoint).
**Build:** section 2.2 exactly. `s09` output artifact `observations` = the saved table. Endpoint `GET /api/worlds/{id}/versions/{vid}/observations?image_id=&entity_id=` returns rows as JSON (tenant check copied from the neighbouring handler).
**Tests:** round-trip save/load equality; `by_image` and `by_entity` indices correct for a 5-row fixture; `to_inventory()` passes `validate_inventory` with zero errors; a row with no 3D position exports `position_3d: None` plus an `unavailable` reason.

### B03 — One ontology, with parts and attributes
**Why:** two taxonomies exist — `perception/ontology.py` (+ `config/ontology.yaml`) and `perception/entities/taxonomy.py`.
**Files:** `config/ontology.yaml`, `perception/ontology.py`, `perception/entities/taxonomy.py` (becomes a thin shim that reads the ontology), tests.
**Build:**
- `OntologyNode` gains `attributes: dict[str, tuple[str, ...]]` (attribute name → allowed values, **always** including `"unknown"`) and `aliases: tuple[str, ...]`. It already has `components`, `attach`, `typical_dims_m`, `min_pixels`, `prompts`.
- `Ontology` builds once: `children: dict[str, list[str]]`, `alias_to_path: dict[str, str]` (lower-cased, spaces collapsed). `ancestors(path)` walks `parent`.
- `Ontology.validate()` adds: every `components` and `attach` path exists; no cycle (DFS with a visiting set); every attribute list contains `"unknown"`.
- Port every class in `taxonomy.py` into the YAML; `taxonomy.get/children/ancestors/class_for_label/size_plausible` delegate to the ontology so existing imports keep working.
- Add the families in section 4.
**Tests:** `validate()` returns `[]`; alias lookup `"wardrobe"` and `"almirah"` resolve to the same path; every old `taxonomy.all_classes()` id still resolves.

### B04 — Hierarchical detection: whole → part → sub-part
**Needs:** B02, B03. **Files:** new `perception/detection/hierarchical.py`, `engine/pipeline/stages/s09_perception.py`.
**Build:**
```python
def detect_hierarchy(image_rgb: np.ndarray, image_id: str, detector, ontology: Ontology,
                     roots: Sequence[str], max_depth: int = 3,
                     min_crop_px: int = 96, budget_calls: int = 40) -> ObservationTable
```
- Work queue: `deque[(node_path, crop_box, parent_row, depth)]`, seeded with one item per root family for the whole image (BFS, so all wholes are found before any parts).
- For each item: prompts = `ontology.get(node).prompts` of the node's **children**; run the detector on the crop, batched **2 prompts per forward pass, FP16** (the measured setting: 0.46 s, 1.7 GB). If the node's `min_pixels` exceeds the object's size in the crop, call `perception.entities.small_objects.inspect_small_objects` (tiles 640, overlap 0.25) instead.
- Suppress duplicates per class: sort by score descending (`np.argsort`), greedy NMS with a vectorised IoU row (`iou ≥ 0.5` suppressed).
- Reject a part whose box centre is outside its parent's box, or whose size violates `typical_dims_m` once 3D size is known.
- Append rows with `parent_obs = parent_row`; push each new row's `components` onto the queue while `depth < max_depth`, crop side ≥ `min_crop_px`, and `budget_calls` remains.
- Coordinates are always stored in full-image pixels: `box_full = box_crop / scale + (x0, y0)`.
**Complexity:** detector calls ≤ `budget_calls` per image; everything else O(n log n) in detections.
**Tests:** fake detector returning scripted boxes → an almirah row with handle and lock rows whose `parent_obs` points at it; a part outside its parent is dropped; `budget_calls=1` stops after the root pass; boxes map back to full-image coordinates exactly.

### B05 — Attributes: which kind of socket, lock, handle
**Needs:** B04. **Files:** new `perception/detection/attributes.py`.
**Build:** `classify_attributes(image_rgb, row, node, detector) -> dict`. For each attribute of the node, build one phrase per allowed value except `"unknown"` (from YAML, e.g. `lock_type: {key: "keyhole lock", numpad: "keypad lock with number buttons", …}`), run the detector on the padded crop (pad 15 %), take the best score per value → `scores: np.ndarray`. Softmax with temperature 0.07 → `p`. Decision: `value = argmax` only if `p[top] − p[second] ≥ 0.20` **and** raw top score ≥ the detector's box threshold; otherwise `value = None` (unknown). Store `{"value", "p", "basis": "visual", "obs": [obs_id]}` in `attrs_json`.
Across views (after association): combine per-value log-probabilities by summing over the entity's observations, re-normalise, apply the same margin rule. Views are correlated (same model), so cap the summed evidence at 3 views' worth.
**Tests:** scripted scores → clear winner chosen; near-tie → `None`; three agreeing weak views → chosen; two conflicting views → `None` with both values kept in the record.

### B06 — Masks (SAM 2) on the live path
**Needs:** B02. **Files:** `perception/segmentation/sam2_backend.py` (exists), new `world_ir/mask_store.py`, `s09_perception.py`, `pyproject.toml` (`perception` extra).
**Build:** install the `sam2` package and `sam2_hiera_tiny` weights (registry lists Apache-2.0; `assert_production_licences` must pass). Prompt SAM 2 with each row's box, batch 8 boxes per image embedding (the image encoder runs **once per image**; only the light mask decoder runs per box). Store masks per section 2.3; set `mask_ref`. Tighten `bbox_px` to the mask's bounds. If the package or weights are missing the stage returns `skipped` with the reason — never `succeeded`.
**Tests:** RLE encode/decode round-trip on random masks including all-false and all-true; `mask_ref` rows resolve; missing weights → `skipped`.

### B07 — Appearance embeddings into association
**Needs:** B06. **Files:** `perception/instances/dinov2_embeddings.py` (exists), `s09_perception.py`, the call site of `perception.instances.association.associate` in the live path.
**Build:** one DINOv2 ViT-S/14 forward per image → patch tokens `(h, w, 384)`; per observation, average the tokens under its mask (or box when no mask) with a boolean index — no second forward per crop. L2-normalise, write to the embedding matrix (section 2.3), set `emb_row`. Pass `appearance={obs_id: vector}` to `associate(...)` — the parameter already exists and is currently not supplied by the API path.
**Tests:** two crops of the same textured patch → cosine > 0.8; unrelated → lower; `associate` called with a non-empty `appearance` in the golden-path run (assert via a spy).

### B08 — Global identity: one object, many photos
**Needs:** B07. **Files:** new `perception/instances/global_association.py`; `association.py` keeps its pairwise scoring.
**Build:**
```python
def cluster_observations(table: ObservationTable, pair_scores: Iterable[tuple[int, int, float]]) -> np.ndarray  # entity_row per obs
```
- `pair_scores` = `(row_a, row_b, llr)` for every pair that passes the existing gates (label compatibility, Mahalanobis distance, epipolar check, appearance) with `llr` = the log-likelihood ratio the existing `_candidate` computes.
- Sort pairs by `llr` descending (`np.argsort`). Union-find over rows. For each pair with `llr > 0`: find both roots; **refuse the merge if the two groups share an `image_id`** (one object cannot appear twice in one photo). Keep each root's image set as a Python `set` of small ints; merge smaller into larger.
- Result: `entity_row = root id remapped to 0..K-1`. This replaces order-dependent greedy joining: the strongest evidence merges first and one bad early match cannot chain.
- Parts follow wholes: a part row may only join a group whose `parent_obs` rows are in the same whole-entity group.
**Complexity:** O(E log E) for E candidate pairs; candidate pairs come from a `cKDTree.query_pairs(r=3σ)` on `world_pos`, not all N².
**Tests:** five views of one socket → one entity; two identical sockets 1 m apart → two; a pair of same-image rows never merges even with a high score; shuffling input order gives the identical partition.

### B09 — Floors are floors (support planes)
**Why:** furniture tops and steps are classed as floor.
**Files:** the plane-role code under `perception/geometry/` and `perception/architecture/` (locate with `grep -rn "floor" perception/geometry/*.py`).
**Build:** take all near-horizontal planes (normal within 10° of up). Sort by height (`np.sort`). Cluster heights with a 1-D gap rule: new cluster where `np.diff(heights) > 0.08 m`. For each cluster compute supported area (sum of inlier-voxel areas from `VoxelCloud`). **Floor of a storey** = the lowest cluster whose area ≥ 40 % of the storey's footprint area and which reaches at least two wall bases. Every other horizontal cluster inside that storey is `support_surface` and is attached to the entity whose box contains it (table top, shelf, step) — never `floor`.
**Tests:** synthetic room with a table (area 15 % of floor) → one floor, one support surface; a mezzanine with its own walls → second storey floor; a staircase → steps, not floors.

### B10 — Clean room footprints without forcing right angles
**Files:** `perception/architecture/room_layout.py`, `non_rectangular.py`.
**Build:** project wall segments to the floor plane. Dominant directions: histogram of segment angles mod 180° weighted by length (`np.bincount`, 1° bins, circular 5° box smoothing), take peaks ≥ 15 % of total length. Snap a segment to a peak only if within 5°; otherwise keep its own angle (old buildings are not square). Merge collinear segments (same direction, offset < 5 cm, gap < 20 cm) with a sort along the direction + single sweep. Extend segments to intersections, `shapely.ops.polygonize`, choose the face containing the most camera centres. Report `closure_gap_m`; if > 0.3 m the room is `partial`, not closed by invention.
**Tests:** rectangular room with noisy walls → 4 edges; L-shaped → 6; a 100° corner stays ~100°; a missing wall → `partial` with the gap reported.

### B11 — Voxel fusion
**Needs:** B01. **Files:** `reconstruction/fusion/voxel_cloud.py` (new), `reconstruction/fusion/consumer.py::fuse_pipeline_points` calls it.
**Build:** section 2.4. Voxel size 2 cm for rooms. Sources integrate in order sfm → mvs → device-depth → mono-depth with their own `sigma_m`; mono-depth never overwrites, it only adds weight (its σ is large).
**Tests:** two noisy copies of a plane fuse to lower variance than either; `sources` bits correct; empty space has no voxel; 1 M points integrate in < 2 s on CPU (assert with a generous bound).

### B12 — Same photos, same result
**Files:** `reconstruction/backend/colmap_backend.py`, `reconstruction/colmap_session.py`.
**Build:** fixed random seed on every COLMAP command that accepts one; images passed in sorted filename order; single-threaded matching when `deterministic=True` (config flag, default on for ≤ 60 images). Record the seed and COLMAP version in the stage metrics.
**Tests (slow, real binary):** two runs on the same 10-photo set → identical registered-image set and point count within 1 %.

### B13 — The GPU manages itself
**Files:** `engine/hardware/device_manager.py` (exists: `HardwareMemoryBudget`, `HardwareExecutionOrchestrator.execution_guard`), new `engine/hardware/model_pool.py`, every model load site in `perception/` and `reconstruction/`.
**Build:**
```python
class ModelPool:
    def acquire(self, name: str, loader: Callable[[str], Any]) -> ContextManager[Any]
```
- `OrderedDict[name → (model, vram_mb)]` as LRU. `acquire`: if loaded, move to end and yield. Else look up `vram_mb` (measured table below; unknown → measure with `torch.cuda.max_memory_allocated` on first load and persist to `engine/hardware/vram_measured.json`). While `not budget.can_allocate_vram(vram_mb)`: pop the least-recently-used entry, call its `unload()`, `torch.cuda.empty_cache()`. If the pool is empty and it still does not fit: run on CPU if the model's registry entry allows it, else raise `ResourceUnavailable` — the stage returns `skipped` with that reason. Never an out-of-memory crash, never a silent drop.
- Reserve 1.0 GB headroom of the 6 GB. Measured: Grounding DINO tiny FP16 1.7 GB; DPT_Hybrid 0.68 GB; Depth Anything V2 Small FP16 0.42 GB; Mask R-CNN 1.55 GB.
- All loads go through the pool; delete ad-hoc `.to("cuda")` at call sites.
**Tests:** fake models with declared sizes — third model evicts the least recently used; a model larger than the whole budget → CPU or `ResourceUnavailable`; nested `acquire` of the same name loads once.

### B14 — Measurements fused properly
**Files:** new `reconstruction/fusion/measurements.py`.
**Build:** `fuse(values, sigmas, group) -> FusedMeasurement(value, sigma, used, rejected, residuals)`.
1. Robust centre: median; scale `s = 1.4826 × MAD`.
2. Reject `|x − median| > 3·max(s, σ_i)` → `rejected`.
3. Inverse-variance mean of the rest: `x̂ = Σ(x/σ²)/Σ(1/σ²)`.
4. Correlation: measurements with the same `group` (same model or same photo) are not independent — inflate: within a group of n use weight `1/(n·σ²)` each, so a group counts as one measurement.
5. `sigma = sqrt(1/Σw)`, floored at the smallest input σ / 2.
Every dimension shown or exported comes from this function and carries `used`/`rejected` observation ids.
**Tests:** 4.02 (σ .02), 4.18 (σ .10), 3.72 (σ .30) → ≈ 4.03; one wild outlier rejected and listed; three values from one group weigh as one.

### B15 — Spatial relationships and the memory query
**Needs:** B08. **Files:** new `world_ir/relations.py`, `apps/api/routes_query.py`.
**Build:**
- `derive_relations(entities) -> list[(a, kind, b)]` using the uniform grid (section 2.5) for candidate pairs: `on` (A's bottom within 3 cm of B's top and footprints overlap ≥ 50 %), `attached_to` (A's back face within 3 cm of a wall plane), `inside` (A's box ≥ 90 % within B's box and B's ontology node is a container), `part_of` (from B04/B08), `adjacent_to` (gap < 10 cm). `contains` is only the inverse of an **observed** `inside`.
- Unseen interiors: a container with no interior observation gets `extra_fields["interior"] = "not_observed"`. Users may state contents; stored with `basis: "user_stated"` and shown as such.
- `GET /api/worlds/{id}/find?q=`: resolve `q` through `Ontology.alias_to_path` and entity labels/attributes → entities → for each, the chain room → furniture → part via `part_of`/`inside`/`on`. Returns the chain, 3D position, and the evidence photo crops.
**Tests:** cup on table → `on`; drawer front → `part_of` cabinet; "snacks" stated by user in drawer 2 → find returns kitchen → cabinet → drawer 2 with `basis: user_stated`; closed cupboard → `interior: not_observed`.

### B16 — Refinement loop that knows when to stop
**Needs:** B04–B08, B13. **Files:** `engine/quality/autonomous_correction_loop.py` (exists — extend, do not write a second loop).
**Build:** task = `(kind, target, expected_gain, cost_s)`. Kinds: `redetect_crop` (attribute unknown and crop ≥ 96 px in another view), `segment` (no mask), `reassociate` (entity with conflicting class votes), `more_views` (→ a capture request, no compute). `heapq` keyed by `(-expected_gain / cost_s, seq)`. `expected_gain` = current entropy of the thing in question × probability the operation resolves it (table of priors per kind, updated from outcomes: successes/attempts with a +1/+2 prior). Loop: pop → run → recompute affected entities only → push follow-ups. Stop on: empty heap, 5 iterations, time budget, or two consecutive iterations where no entity's state changed (compare a hash of `(entity_id, type, attributes)`).
**Tests:** unknown attribute + better crop available → one `redetect_crop`, resolved; nothing to gain → zero tasks; a task that never helps is not retried more than twice.

### B17 — Video in, through the API
**Files:** `apps/api/routes_reconstructions.py::_pick_video_frames`, `engine/pipeline/facade.py::score_and_pick_video_frames`, `tests/integration/test_progressive_product_journey.py`.
**Build:** the picker already keeps sharp, distinct frames. Fix the stale journey test `test_H_video_is_sampled_into_frame_evidence_and_duplicates_are_measured`: a clip of 6 views repeated 3× yields 6 photo evidences, and the response reports `frames_seen: 18, frames_kept: 6, duplicates_dropped: 12`. Add those three counters to the API response (additive).
**Tests:** that journey test, updated; a 2-frame clip keeps 2.

### B18 — Studio shows the memory
**Needs:** B15. **Files:** `frontend/src/components/inspector/`, `frontend/src/lib/api/` (see `docs/blueprint/36_UI_SPEC.md`).
**Build:** inspector gains a **Parts** tree (from `part_of`) and an **Attributes** list where each row shows value, evidence badge and the photo crop; unknown shows "Unknown — not visible in any photo". A search box calls `/find` and flies the camera to the result. No value is ever rendered without its basis.
**Tests:** contract tests for the two parsers; Playwright: search "socket" → count matches API; clicking a part selects it in the viewport.

### B19 — Exports carry the detail
**Files:** `exporters/` (IFC, glTF).
**Build:** parts export as child nodes (glTF) and `IfcRelAggregates` (IFC); attributes as glTF `extras` and an IFC property set `Pset_RealityEngine` with `Basis` and `Confidence`. Unknown attributes export as absent with `Basis = "unknown"`, never a default.
**Tests:** round-trip through `ifcopenshell.open` and a glTF loader: part count and attribute values survive.

### B20 — Golden loop and release
Run the `RELEASE.md` "Must pass" list on real photos, twice. Tick boxes with the evidence.

## 4. Object families for December (go into `config/ontology.yaml` in B03)

Each family: whole → parts → attributes. Values always include `unknown`.

| Family | Wholes | Parts | Attributes |
|---|---|---|---|
| Storage | almirah/wardrobe, cabinet, chest of drawers, shelf unit | door leaf, drawer front, handle, lock, hinge, leg | handle_type (bar, knob, recessed, ring); lock_type (key, numpad, padlock, none); door_count; material |
| Electrical | socket outlet, switch, switchboard, distribution board | faceplate, gang, rocker, pin holes, indicator | socket_standard (type A/B US, type C euro, type D india 5A, type G UK, type M india 15A, USB, TV coax, RJ45 data, telephone); gang_count; switch_kind (rocker, dimmer, regulator, bell) |
| Doors | hinged door, sliding door, double door | leaf, frame, handle, lock, hinge, threshold, closer | handle_type (lever, knob, pull); lock_type (key, numpad, card, bolt, none); swing (left, right) |
| Windows | casement, sliding, fixed | frame, pane, grill, latch, sill, curtain/blind | pane_count; openable (yes, no); grill (yes, no) |
| Seating & tables | sofa, chair, bed, table, desk | leg, seat, back, arm, cushion, top, drawer | leg_count; material |
| Fixtures | ceiling fan, light fitting, AC unit, vent, tap, basin, WC | blade, shade, grille, spout, handle | fixture_kind; mount (ceiling, wall, floor) |
| Structure | wall, floor, ceiling, column, beam, stair | step, riser, railing, baluster, skirting, cornice | finish (paint, tile, stone, wood, plaster, exposed) |

Socket standards and lock types are visual classes only. The engine never claims voltage, wiring or security rating.

## 5. How an implementer works a package

1. Read the package. Read every file it names before editing.
2. Write the named tests; run them; see them fail.
3. Build exactly the structures in section 2 — if a package seems to need a different one, stop and say why.
4. Run the package tests, then the regression line in section 3.
5. One commit `B<nn>: <title>`. Do not push. Report: files changed, commands run with their last output line verbatim, anything not verified.
6. Never: a new dependency not named here; a fake or default value where the data is unknown; editing a test to make it pass unless the package names that test; a second implementation of something that exists.

Packages that can run in parallel (no shared files): **B03 ∥ B09 ∥ B10 ∥ B12 ∥ B13 ∥ B14 ∥ B17** after B01–B02; **B11** after B01; **B04→B05**, **B06→B07→B08** in sequence; **B15, B16, B18, B19** after B08.

## 6. Beyond December — what each ambition needs

The data structures above do not change for any of these; they add backends, ontology families and capture types.

| Ambition | What is missing (not code alone) | Hook already in this spec |
|---|---|---|
| Every object type, every part | Reviewed labelled images per family (hundreds each) to measure and then train specialist detectors; today there are zero annotations | ontology DAG (B03), hierarchical detector takes any family (B04), inventory scorer exists |
| Read labels, nameplates, signs | An OCR model with a commercial licence, wired as an attribute `basis: "ocr"` | attribute record has `basis` (2.5) |
| Domes, arches, minarets, vaults (Taj Mahal, a basilica) | Parametric primitives fitted to the voxel cloud: surface of revolution (fit axis by normals, then a 1-D profile curve), arch profiles, symmetry detection; exterior capture from many heights | `VoxelCloud` normals (2.4); entity geometry is already "one of several representations" |
| Towers and large exteriors (Burj Khalifa) | Drone or aerial capture with GNSS; façade repetition detection; level-of-detail tiles for viewing | stage contract accepts new evidence types; tiles exist in `worldstore/tiles.py` |
| Forts, campuses, streets, cities | Geo-registration, terrain, many sessions merged; streaming viewer | adoption/change-set path; uniform grid generalises to tiles |
| Geometry where SfM fails (blank walls, few photos) | A feed-forward backend with a commercial licence (MapAnything's Apache checkpoint), measured for 6 GB fit | second backend behind `reconstruction/backend/interface.py`; `ModelPool` (B13) |
| Global optimisation of everything at once | A factor-graph solver (GTSAM or Ceres) over cameras, planes and object positions | `measurements.fuse` (B14) is the scalar special case; residuals are already recorded |
| Dimensional accuracy claims | Six measured rooms with a laser measure, two held out | evaluator and protocol exist under `benchmarks/` |
| Learning from corrections | A reviewed-label store and a held-out test set; never train on unreviewed output | corrections are already versioned world edits |
