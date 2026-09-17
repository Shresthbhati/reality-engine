# detection

**IMPLEMENTED (real-model verified).** `maskrcnn_backend.MaskRCNNDetector`
runs torchvision's COCO-pretrained Mask R-CNN and produces:

- real detections: class label, bounding box, score, model name/version
  (`IDetectorBackend`)
- real instance masks per detection, resampled to the source-image plane
  so they satisfy `lift_region_to_3d`'s dims contract (`ISegmentationBackend`)
- honest unavailability (`DetectionBackendUnavailableError`) when
  torch/torchvision/COCO weights are missing — never a silent fallback
- model acquisition recorded in `perception/model_registry.py`

Status notes:

- Real-model verified on a genuine photograph (labeled detections +
  source-resolution masks). The synthetic room fixture legitimately
  yields ZERO detections — white-noise textures are out of COCO's
  distribution; the pipeline records that as zero objects, not an error.
- First weights load downloads ~170 MB from download.pytorch.org (the
  torchvision weights API). Subsequent runs are offline from
  `~/.cache/torch/hub/checkpoints/`. Pre-seed for airgapped use.

Consumers: `engine/pipeline/vertical_slice.py` stage 3.5 (segment →
lift → merge → promote to WorldIR entities).
