# Reality Engine - Final Acceptance Run (real data, real API/CLI)

Date 2026-11-05, branch `claude/integration-2026-11-05`. Nothing was mocked, no source was edited. Logs and JSON: `docs/audits/acceptance_run/`.
Scratch state (DB, artifacts, worldstore, exports): `C:/Users/shres/AppData/Local/Temp/acc2/`.

## Data and limitation
`datasets/` contains exactly one real set: `datasets/south_building/` (32 real photos, 1024x768, public COLMAP "South Building", plus COLMAP reference `sparse/` and old `runs/`). It is an **OUTDOOR building facade, not a multi-room interior**. No indoor / multi-room real data exists in the repo (`datasets/README.md` says so). Any interior semantics in the result (room, corridor, windows, ceilings) are therefore inferred from planes of an exterior and cannot be judged against ground truth.

## Environment (`docs/audits/acceptance_run/doctor.txt`)
Python 3.14.7 (venv), torch 2.14.1+cu130 CUDA available, RTX 4050 Laptop 6 GiB, COLMAP 4.2.0 with CUDA at `C:\Users\shres\tools\colmap-extracted\bin\colmap.EXE`, node v26.8.1, npm 11.19.0, DB writable.
Packages (checked): ifcopenshell yes, trimesh yes, cv2 yes, torchvision yes; **timm NO, segment_anything NO, open3d NO**. Cached weights present (MiDaS, DPT_Hybrid, Mask R-CNN, SAM vit_b); nothing downloaded.
`reality verification --runtime`: colmap, ifcopenshell, torch AVAILABLE; 3dcitydb, VIO backends UNAVAILABLE.
API: `uvicorn apps.api.main:app --port 8123` in isolation (DATABASE_URL/STORAGE_ROOT/WORLDSTORE_ROOT in temp). Port was free before start.

## Classification legend (assumed - the A-G key from the mission was not supplied to me)
A = ran, output valid, real. B = ran, real but heuristic / semantically weak. C = ran, partial or degraded. D = not executed in the product path (gate or missing precondition). E = blocked by missing dependency/environment. F = bug found. G = not supported.

## Job lifecycle observed (job_fa9496175eef, `job_main.json`)
RUNNING/analyzing_evidence 0-175 s, RUNNING/reconstructing 175-352 s, PARTIAL/partial (progress 100). Total 352 s. QUEUED was not captured (worker claimed the job within 0.25 s, before my first poll). SUCCEEDED/FAILED never occurred. **No job reported SUCCEEDED without valid outputs**: the API job honestly ended PARTIAL; the CLI run ended PARTIAL_SUCCESS with depth/perception/mesh recorded as skipped.

## 25-stage table
| # | Stage | Exec? | Pass? | Evidence | Class | Simulated / heuristic |
|---|---|---|---|---|---|---|
| 1 | Import images | yes | yes | `POST /api/reconstructions` 32 files -> 202 in 4.1 s; 32 evidence rows, world `wld_ae3f2ba8527e` | A | none |
| 2 | Validate source | yes | yes | `POST /api/uploads/check` 200: 32 decodable, 0 duplicates, 1024x768, sharpness 1.6k-2.6k; class `photograph_unverified` (no EXIF) | A | none |
| 3 | Pick frames/viewpoints | yes | partial | evidence_summary: "28 add a new viewpoint, 3 mostly repeat", coverage 240/360 deg, 1 scene group. All 32 were passed to SfM (nothing dropped) | B | viewpoint scoring heuristic |
| 4 | Register cameras | yes | partial | API: 22/32 (69%); 10 photos unplaced (P1180157..P1180181 etc). CLI (separate run, CPU COLMAP): 25/32 | C | none |
| 5 | Sparse reconstruction | yes | yes | 5,396 points, 22 cameras, COLMAP sparse 118 s; scale `relative` (no metres) | A | scale unitless |
| 6 | Dense geometry | no | n/a | skipped by gate: "22 of 32 registered (69%; needs >= 80%)" (`engine/pipeline/dense_gate.py:27`). Not run, not failed | D | - |
| 7 | Depth | no | no | API: skipped, "no trusted intrinsics" (`apps/api/jobs.py:629-650` never passes intrinsics). CLI with dataset intrinsics: "depth backend failed: No module named 'timm'" (`engine/pipeline/vertical_slice.py:886`) | D/E | - |
| 8 | Detect objects | pipeline no; component probe yes | pipeline: no | World has 0 objects (`metadata.perception: skipped`). Probe only (not the pipeline): Mask R-CNN CPU on 8 photos -> 10 detections (bench 5, backpack 3, person 2) in 2/8 images, `stage8_9_component_probe.txt` | D (probe A) | COCO classes only; never reached the world |
| 9 | Segment | no | no | `SAMSegmentationBackend.segment` -> "Missing perception dependencies: segment-anything". Mask R-CNN `segment()` returned 0 masks on 2 photos | E | - |
| 10 | Associate across views | object level no | - | no objects exist to associate. Photo-level scene-identity grouping did run (32 images, 1 group, many "uncertain_links" with <=11 inliers) | D | - |
| 11 | Architectural structure | yes | yes, weak semantics | 33 planes -> 21 walls, 3 floors, 3 ceilings, 2 windows, 1 opening, 1 room, 1 corridor (32 entities). For an exterior these room/corridor/window entities are not credible (windows conf 0.05, room 0.32, but corridor 0.90 INFERRED) | B | plane promotion + room/corridor inference heuristics |
| 12 | Canonical WorldIR | yes | yes | `GET /worldir` 200, 157 KB, 32 entities, 31 geometries, job payload `validated: true` | A | - |
| 13 | Evidence + provenance preserved | yes | partial | Entities carry provenance (INFERRED/RECONSTRUCTED), `supporting_evidence_ids` (21 for struct-plane-000), observation `obs-promotion-*`. Lost after a correction (see bug 1) and coarse in the route (bug 4) | C/F | - |
| 14 | Compile geometry | yes | yes | 31 geometries; glTF has 28 mesh/point geoms; points PLY 4,931 verts; scale relative | A | relative scale |
| 15 | Validate world | yes | yes | `reality validate`: "valid: True (12 issue(s))" - 12 warnings (surface_without_space x9, duplicate IoU 70%/59%, containment disjoint). Same 12 in job `degraded` | A | - |
| 16 | Save to WorldStore | yes | yes | `reality store list` shows v-610499e3ce2e (parent -); `store verify` OK for both versions | A | - |
| 17 | Open in Studio (API) | yes | yes | GET worldir 200 157 KB, points 200 (PLY 4,931), cameras 200 (22 cams), versions 200, status 200, quality 200, export/formats 200 (23 formats). **No browser/Studio was opened** | A (API only) | - |
| 18 | Inspect entity + evidence | yes | partial | `/entities/{id}/provenance` 200 (trace_level `session`, 32 photos listed incl. 10 unregistered); `/confidence` 200 but `confidence_vector` all null | C | - |
| 19 | Real correction | yes | yes | `POST /commit` name+semantic_labels on struct-plane-000 -> 200, new version v-7edd4d4be066, `duplicate:false` | A | - |
| 20 | Persist | yes | yes | versions list: v2 parent v1, is_current; `store verify` OK | A | - |
| 21 | Restart API | yes | yes | both uvicorn processes force-stopped (not a graceful shutdown), new process 34656 on 8123, `/api/health` 200 | A | - |
| 22 | Reload world | yes | yes | `GET /api/worlds/{id}`: current_version v2; worldir name "South Building main facade (user-corrected)" | A | - |
| 23 | Verify correction + lineage | yes | yes | versions chain v1 -> v2; `diff?base=v1&head=v2`: 1 entity modified (name, semantic_labels); `worldir?version=v1` still shows old name. Caveat: entity provenance stays INFERRED (no human-correction marker) | A | - |
| 24 | Export | yes | yes | `POST /export` 200 for gltf, glb, ply, obj, ifc, usda, cityjson, geojson, stl; download 200 | A | - |
| 25 | Validate exports | yes | yes | trimesh glb/gltf: 28 geoms, 5,097 verts, 264 faces, finite; ply/stl 263 v/444 f; obj ok; GLB header length == file size; ifcopenshell opens IFC4, 868 entities, 21 walls, corrected name present; cityjson 16 objects; geojson 16 features; usda 49 defs. Skips: gltf/ply 5 entities (corridor, room, 2 windows, opening); cityjson 16 of 32; ifc 1. Not opened in Blender/BIM viewers | A | - |

## What genuinely works (observed)
Photos -> upload validation -> job -> real COLMAP sparse SfM -> plane/structure compile -> WorldIR -> WorldStore -> versioned correction -> restart persistence -> diff/lineage -> nine export formats that re-parse independently. Failure is honest: gates and missing dependencies show up as skipped/PARTIAL, not as fake success. The 25-minute cap was never hit (longest job 352 s).

## What does not work / is not exercised on real data
- Depth, object detection, segmentation, cross-view object association: never reached the world on real data (no intrinsics in the API path, `timm` and `segment-anything` missing). World has zero objects.
- Dense MVS: gated off at 69% registration. Not re-verified here; earlier docs claim a GPU dense run, which I did not reproduce.
- Metric scale: none; the world is relative-scale.
- Interior understanding: untestable (no indoor real data); exterior was labelled with room/corridor/window entities.
- Studio browser rendering: not run (another agent audits it statically).

## Bugs found (recorded, not fixed)
1. **Correction commit drops evidence, cameras and report.** `apps/api/routes_worlds.py:1454` `store.save_version(worldir, parent=..., version_id=...)` passes no `source_session_ids` and carries no cameras/report sidecars (mirror row at ~1498 copies the empty list). Observed on the new HEAD: provenance `trace_level: "none"` ("no source capture sessions"), `GET /cameras` returns `[]` with `image_size: null` (was 22 cameras), `GET /report` 404 (was 84 KB). So any user correction empties the Studio camera layer and kills evidence tracing for that version.
2. **API pipeline never supplies intrinsics/image size.** `apps/api/jobs.py:629-650` builds `VerticalSliceOptions` without `intrinsics`/`image_size`, so depth, perception and meshing are always skipped on the product path; `cameras.json` also reports `image_size [1280,960]` (default at `engine/pipeline/vertical_slice.py:92`) for 1024x768 photos.
3. **`perception` extra lacks `timm`.** `pyproject.toml:44` lists torch, torchvision, numpy, segment-anything only; MiDaS `DPT_Hybrid` needs `timm`, so depth fails ("No module named 'timm'", caught at `vertical_slice.py:886`).
4. **Provenance route is coarser than the data.** `apps/api/routes_worlds.py:704-721` falls back to all session photos (32, including 10 unregistered) and ignores the entity's own `custom_properties.supporting_evidence_ids` (21) and `observations` (route returns `observations: []`, worldir has `obs-promotion-struct-plane-000`). Confidence route returns an all-null `confidence_vector`.
5. **Semantic over-claim on an exterior.** Room, corridor (confidence 0.90, height 0.055 units) and windows were inferred from a building facade. Compiler location not traced (no file:line).
6. **GPU memory.** API log shows `CUDACachingAllocator ... OOM on device 0 (free: 0, total 6438780928)` during `analyzing_evidence`; the API process then holds ~5.8 GiB VRAM idle until it exits (nvidia-smi 5,790 MiB at 0% util; 0 MiB after kill). Likely the scene-identity matcher (`reconstruction/scene_identity.py`, `reconstruction/matching/lightglue.py`), not confirmed. Any GPU job run next to a live API will compete for memory.
7. Minor: a correction leaves entity provenance as INFERRED (no user/human marker) - possibly by design. A tight polling loop gets HTTP 429 (my first poller; the server limiter behaved correctly).

## Resources
Peak VRAM (nvidia-smi, 3 s sampling, whole API job): **5,902 MiB of 6,141 MiB** (baseline 0). Idle after job 5,790 MiB. CLI run (COLMAP on CPU, DPT_Hybrid failed before load): peak 73 MiB. API job 352 s; COLMAP sparse stage 118 s; CLI run 100 s. Mask R-CNN CPU probe 59 s for 8 photos.

## Reproduce
```
python scripts/doctor.py
# API (bash)
export DATABASE_URL=sqlite+aiosqlite:///<tmp>/acc.db STORAGE_ROOT=<tmp>/artifacts WORLDSTORE_ROOT=<tmp>/worldstore
python -m uvicorn apps.api.main:app --port 8123
curl -F files=@datasets/south_building/images/P1180141.JPG ... http://127.0.0.1:8123/api/uploads/check
curl -F name=acc -F files=@... (all 32) http://127.0.0.1:8123/api/reconstructions        # -> world_id, job_id
curl http://127.0.0.1:8123/api/jobs/<job_id>                                              # poll every >=5 s (limiter returns 429)
curl http://127.0.0.1:8123/api/worlds/<id>/{worldir,points,cameras,versions,status,quality}
curl http://127.0.0.1:8123/api/worlds/<id>/entities/struct-plane-000/provenance
curl -X POST .../api/worlds/<id>/commit -H 'Content-Type: application/json' -d '{"entity_id":"struct-plane-000","parent_version_id":"<v1>","changes":{"name":"...","semantic_labels":["facade"]}}'
# restart uvicorn, then
curl ".../api/worlds/<id>/diff?base=<v1>&head=<v2>"; curl ".../api/worlds/<id>/worldir?version=<v1>"
curl -X POST .../api/worlds/<id>/export -d '{"format":"glb"}' -H 'Content-Type: application/json'   # then GET download_url
# CLI side runs
python -m apps.cli.main validate worldir_v2.json
python -m apps.cli.main store list|verify --store <tmp>/worldstore [--version <id>]
python -m apps.cli.main run <dir with manifest.json(intrinsics_px fx=fy=853.23,cx=512,cy=384) + images/> -o out --export gltf --export ifc --json
```
Intrinsics for the CLI run are the dataset's own COLMAP reference camera (`datasets/south_building/sparse/cameras.txt`, SIMPLE_RADIAL f=2559.68 at 3072 px) scaled by 1024/3072; distortion ignored.

## Notes on the working tree
- A file with this name already existed at HEAD (commit 4bd85e3a); this run replaced it. Recover the old text with `git show HEAD:docs/audits/REALITY_ENGINE_FINAL_ACCEPTANCE_REPORT.md`.
- Bug 3 cites `pyproject.toml:44` as committed at HEAD. During this run another process edited the working-tree `pyproject.toml` to add `timm>=0.9`, `transformers`, `kornia` to the `perception` extra (not by me; timm is still NOT installed in the venv, so depth would still fail until installed).
- Other modified files in `git status` (cmd_techs.py, benchmark.py, perception/*, matching/*, audits) belong to other sessions; my only additions are `docs/audits/acceptance_run/` and this report.

## Ten-line summary
1. Real data: only South Building (32 outdoor photos); no real interior exists.
2. 22 of 32 photos registered via the API (25 via CLI); sparse model 5,396 points; relative scale.
3. API job ended PARTIAL (352 s); no job claimed SUCCEEDED without valid outputs.
4. Dense MVS was gated off (69% < 80%); depth/detection/segmentation/association never reached the world.
5. Depth is skipped in the API for lack of intrinsics (jobs.py:629-650) and fails in the CLI for lack of `timm` (pyproject.toml:44); SAM needs `segment-anything`.
6. WorldIR (32 entities) validates, saves, versions, corrects, diffs and survives an API restart.
7. Nine exports (glb/gltf/ply/obj/stl/ifc/usda/cityjson/geojson) re-parse with trimesh/ifcopenshell/json; the correction shows up in the IFC.
8. Worst bug: a correction commit drops source sessions, cameras and report from HEAD (routes_worlds.py:1454).
9. Room/corridor/window entities on an exterior facade are inferred artifacts, not findings.
10. Peak VRAM 5,902/6,141 MiB with a CUDA OOM warning; the idle API keeps ~5.8 GiB until killed. Studio browser UI not exercised.
