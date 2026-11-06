# scene_complete eval set (task 8.6)

Scorer and label format for the scene-complete reconstruction spec
(`docs/future/perception/SCENE_COMPLETE_RECONSTRUCTION.md`). Run:

    python -m benchmarks.scene_complete.score <predictions_dir>

**Status: scorer only.** The acceptance target (>= 20 AI + >= 20 real room photos,
hand-labelled) is NOT met: `labels/` holds one synthetic self-check. Real labelled images
must be supplied; do not fabricate labels. Put images in `images/` (not committed if
large/licensed), one `labels/<stem>.json` each (see `score.py` for the schema).
