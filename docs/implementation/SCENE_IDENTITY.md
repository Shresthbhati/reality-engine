# Multi-view scene identity and unified reconstruction

Status: **EXPERIMENTAL** (implemented + integrated + tested on synthetic data with real COLMAP; real-photo validation pending).

## What existed before (audit)

* `apps/api/jobs.py::_run_reconstruct_session_inner` already pools the photos of **every session attached to a
  world** into one `run_progressive` -> one COLMAP run -> one WorldIR -> one WorldStore version chain. So the
  "one world per image" failure was **not** how the pipeline built worlds; world identity comes from session
  attachment, and `ColmapSession` keeps one persistent workspace per world.
* The real gap: **nothing verified that the pooled photos show the same place.** Whatever was attached was fed in.
  If two places were attached to one world, COLMAP's mapper could emit two sub-models and
  `reconstruction/merge.py` would try to merge them under an explicit *same-scene prior*
  (`contact_margin_fraction=10`), leaving only ICP gates between two different rooms and one fused cloud.
* Diagnostics were thin: `ReconstructionResult` carried no registered count, reprojection error or component list
  (the backend parsed COLMAP's per-point error and discarded it into a heuristic confidence).

## What was added

| Piece | Where |
|---|---|
| `registration_stats` on `ReconstructionResult` (additive, default `None`): n_input, registered, points, mean reprojection error, mean track length, per-component list | `reconstruction/backend/interface.py`, `colmap_backend.py` |
| Verified-correspondence grouping (`group_images`, `SiftPairVerifier`), declared-session conflicts, uncertain links, rejection of look-alikes | `reconstruction/scene_identity.py` |
| Diagnostics + outcome `success / partial / failed / geometrically_incompatible` (`diagnose`, `diagnose_counts`, `check_poses`, `camera_coverage`) | same |
| Bounded correction loop (`reconstruct_group`) with measured-improvement acceptance and gained-camera support rule | same |
| Persistent `SceneRegistry` (stable group ids, last valid reconstruction, history) | same |
| Job-level gate: only the world's scene group enters the reconstruction; other verified groups are listed in `skipped_evidence` (job grades PARTIAL) and never merged; result recorded in the version report + job payload as `scene_identity` | `apps/api/jobs.py` (`_scene_gate`, `_commit_scene_result`) |

Design rules kept: visual similarity is never identity (only epipolar-verified inliers link images); declared
sessions are hypotheses; unlinked images are not force-fed to a group; a retry never overwrites the last valid
result; nothing LLM-derived participates in acceptance.

## Measured thresholds (provisional)

Dataset: synthetic textured room (`scripts/render_room_dataset.py`), 28 views (A) with known poses, plus 8 views of
the same layout and trajectory with different surface texture (B). COLMAP 4.x, CPU.

| Pair verifier | A-B (different place) max inliers | scene A connected at |
|---|---|---|
| ratio 0.8, 1 px, one-way | 16 | >=18 |
| ratio 0.75, 0.5 px, mutual (**chosen**) | **8** | **>=12 (one 28-image component)** |

`GroupingThresholds`: `min_inliers=12`, `min_inlier_ratio=0.2`, `uncertain_inliers=8`. The margin (8 vs 12) is narrow
and measured on one synthetic family — **re-measure on real indoor photographs before relying on it there.**

`QualityThresholds` (registered fraction >=0.9 success, >=0.4 partial, reprojection <=2.0 px, >=30 points) are
engineering starting points, not validated universals. Observed on the A scene: 0.74 px mean error at 25/28 registered.

Gained-camera rule: a camera that only a retry registered needs >=2 verified links to already-placed cameras. Measured
reason: on the 19-image subset the guided-matching retry added `A_0024` (one strong link; 22 inliers, others <=8) with a
23.6 deg / 1.0 m error vs ground truth.

## End-to-end result (`scripts/run_scene_identity_e2e.py`)

Mixed upload A(28)+B(8): 2 groups (28 A-only, 8 B-only), 0 unlinked, 56 uncertain look-alike links recorded and not
merged. COLMAP on the A group: 25/28 registered, mean reprojection 0.743 px, 1 component, 1230 points, outcome
`partial` (25/28 = 89% < 90% success bar); pose error vs ground truth after Sim(3): median rotation 0.060 deg, max
0.250 deg, median centre 0.0030, max 0.0210 (GT units). Guided retry (27/28) rejected by the support rule; baseline-SIFT
retry gave no gain. Registry reloaded after restart with the same group ids and last-valid = 25 registered.
Three views with little overlap -> honest `failed` (COLMAP found no initial pair), not a fabricated model.

## Remaining limitations

* Thresholds calibrated on synthetic imagery only; no real-photo set with a known same/different-place label yet.
* Pair verification is O(n^2) up to ~65 images, then an ordered-neighbour window (far-apart links missed, so they
  surface as separate groups — conservative). A retrieval index is the upgrade.
* Retries re-run the whole COLMAP workspace (temporary directory); there is no per-stage resume yet.
* The job layer sees counts only: reprojection error / coverage are `unmeasured` in `diagnose_counts`.
* The support rule is conservative: it also rejects genuinely correct gained cameras with weak links (e.g. `A_0027`).
* Other verified groups are reported, not turned into separate worlds; creating them is a product decision.
* Dense point count / mesh validity are carried as fields but only populated when those stages ran.

## Next measurable milestone

Run `scripts/run_scene_identity_e2e.py`-equivalent on >=2 real indoor capture sets (one shared place from different
sessions, one different place) and record: A-A vs A-B inlier distributions, groups recovered, registered fraction and
reprojection error; adjust thresholds only from that data.
