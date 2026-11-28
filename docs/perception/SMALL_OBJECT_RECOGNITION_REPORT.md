# Small-Object Recognition Report

## 1. Tiled Detection Architecture
- High-resolution inputs are tiled (`tile_scale=2`, `tile_overlap=0.2`, `small_share=0.4`) in `perception/detection_adapter.py` and evaluated by `perception/evaluation/small_object_benchmark.py` and `perception/evaluation/gt_benchmark.py`.

## 2. Real-Photo Small-Object Detections (`datasets/real_room_capture`, 4 × 4032×3024)
Observed in the RTX 4050 validation run (`228` Grounding DINO observations total):
- `electrical.switchboard.module.blank`: `15`
- `electrical.junction_box`: `11`
- `fixtures/electrical/switch`: `8`
- `circulation.stairs.riser`: `6`
- `electrical.switchboard.module.dimmer`: `4`
- `electrical.switchboard.module.usb`: `2`
- `electrical.switchboard.module.data`: `1`

## 3. Ground-Truth Benchmark Coverage
- `datasets/gt_perception_benchmark/annotations.json` defines `5` small-object annotations (`switch`, `socket`, `door_handle`) across 2 views (`small_object_recall50` tracked separately in `DetectionEvalReport`).
- Verified in `tests/test_gt_benchmark.py`: missing small objects drop `small_object_recall50` to `0.0` while coarse classes (`chair`) remain `1.0` — ensuring small-object regressions cannot be masked by large furniture AP.
