# 12 — Reconstruction pipeline

Each stage has one job, a typed input/output, a fallback and a failure rule.
Stages run in the worker, orchestrated by `engine/pipeline/` (progressive mode:
the user gets a usable early result, then refinements). Each stage reports its
**real** progress (stage name + counts). No invented percentages (P14).

## Stage order (the order in which results become useful)

| # | Stage | Input → Output | Default tool | Lives in |
|---|-------|----------------|--------------|----------|
| S1 | Ingest & check | files → `Image` records with EXIF, GPS, quality, duplicate flags | Pillow, OpenCV (ExifTool later) | `evidence/` |
| S2 | Scene grouping | images → groups that show the same place; unrelated photos rejected | SIFT + epipolar RANSAC (today) → ALIKED+LightGlue (benchmark) | `reconstruction/scene_identity.py`, `reconstruction/matching/` |
| S3 | Camera recovery (SfM) | group → cameras + sparse points; rejected images listed with reason | COLMAP | `reconstruction/backend/colmap_backend.py`, `orchestrator.py` |
| S4 | Scale | sparse model → scale_state + scale factor + uncertainty | user measurement > metric depth > known object | `reconstruction/scale.py` |
| S5 | Dense geometry | cameras → dense points / depth maps | COLMAP MVS (when CUDA available) else depth fusion | `reconstruction/dense_pipeline.py`, `fusion/` |
| S6 | Architecture | points + depth → planes, Space, Surfaces, Openings | RANSAC planes + room graph | `perception/geometry/planes.py`, `perception/architecture/` |
| S7 | Per-image perception | images → Observations (masks, labels, depth, embeddings) | detector + SAM + depth model via the model registry | `perception/` backends |
| S8 | Lifting | Observation + camera + depth → world position, sigma, bounds | ray casting against dense depth | `perception/instances/lifting.py` |
| S9 | Association | Observations → AssociationDecisions → Entities | gated Hungarian, no transitive merges | `perception/instances/association.py` (integration branch) |
| S10 | Object geometry | entity + its observations → parametric/mesh geometry, components | parametric fit first, mesh for organic shapes | `perception/architecture/parametric.py`, `reconstruction/meshing/` |
| S11 | Relationships | entities + surfaces → typed relationships | geometric rules (support, mount, contain) | `engine/scene_graph/`, `perception/architecture/topology.py` |
| S12 | Validation | WorldIR → ValidationReport; blocks the commit on invariant failure | `world_ir/validation.py` | `world_ir/` |
| S13 | Coverage & capture requests | world + cameras → evidence state per surface, CaptureRequests | ray-cast coverage + view planning | `engine/pipeline/guidance.py` (extend to world space) |
| S14 | Commit | WorldIR → new WorldStore version + app DB mirror | `worldstore/store.py` | `worldstore/`, `apps/api/jobs.py` |

Optional / later: S15 appearance & texture projection, S16 material
classification, S17 fine-detail pass (sockets, switches via high-res crops),
S18 constrained completion (`PRIOR_GENERATED` output only).

## Stage contract (every stage)

```python
class Stage(Protocol):
    name: str                      # internal; user-facing labels live in the UI map below
    def run(self, ctx: StageContext) -> StageResult: ...
# StageResult: status (ok | partial | failed), outputs, counts{}, warnings[], reason?
```

- **Fail honestly**: a stage that cannot run returns `failed` with a reason the
  UI can show ("Only 2 of 12 photos overlap enough to calculate positions").
- **Partial is allowed**: e.g. S3 registers 9/12 images → continue with 9 and list the 3.
- **No silent fallback to fake data**: a missing model means the stage is
  `failed` or skipped with a reason, never placeholder output.
- **Cache by input hash**: S1, S7 and S8 results are cached per image hash, so a
  new photo only processes itself (P11, P6).

## Incremental path (new photos into an existing world) — P6

1. S1 on the new images only.
2. **Register** the new images against the existing model (`registration/`,
   COLMAP image_registrator) — not a full re-SfM. If registration fails, report
   "this photo doesn't match the world" (scenarios A11/A19) and do not merge.
3. S7–S9 on the new images only; association runs against **existing entities**.
4. Compute the affected closure (`world_ir/incremental.py`); re-run S10–S13 only there.
5. Commit a new version; `diff` shows exactly what changed.

## User-facing stage labels (the UI uses these — P13)

| Stage | Label |
|-------|-------|
| S1 | Checking photos |
| S2 | Finding photos of the same place |
| S3 | Calculating camera positions |
| S4 | Working out real-world scale |
| S5–S6 | Building room geometry |
| S7–S10 | Recognising and reconstructing objects |
| S11–S12 | Checking the result |
| S13 | Finding what's missing |
| S14 | Saving the world |
