"""OneFormer Universal Segmentation Backend (Block 6 of 50-tech evaluation).

OneFormer is a universal segmentation model trained with task-conditioned architecture
(MIT License, non-commercial=False). It unifies semantic, instance, and panoptic segmentation.
"""

from __future__ import annotations

from typing import List

from evidence.session import EvidenceItem
from perception.segmentation.interface import (
    ISegmentationBackend,
    SegmentationResult,
    SegmentedRegion,
)
from perception.segmentation.sam_backend import SegmentationBackendUnavailableError
from provenance import Uncertainty


class OneFormerSegmentationBackend(ISegmentationBackend):
    """OneFormer unified architectural segmentation backend."""

    model_name = "OneFormer"
    model_version = "v1-swin-large"
    license = "MIT"

    def __init__(self, task: str = "panoptic", confidence_threshold: float = 0.5) -> None:
        self.task = task
        self.confidence_threshold = confidence_threshold
        self._model = None
        self._processor = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import AutoProcessor, OneFormerForUniversalSegmentation

            model_id = "shi-labs/oneformer_ade20k_swin_large"
            self._processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
            self._model = OneFormerForUniversalSegmentation.from_pretrained(model_id, local_files_only=True).eval()
        except Exception as e:  # noqa: BLE001
            raise SegmentationBackendUnavailableError(
                f"OneFormer cannot load (needs torch + transformers and the locally cached "
                f"shi-labs/oneformer_ade20k_swin_large weights; nothing is downloaded or faked): {e}"
            ) from e

    def segment(self, evidence: EvidenceItem) -> SegmentationResult:
        """Run panoptic/semantic segmentation on the evidence image.

        Raises SegmentationBackendUnavailableError when the model cannot load; never returns fabricated masks."""
        self._ensure_model()
        import torch
        from evidence.image_check import open_oriented

        img = open_oriented(evidence.path).convert("RGB")
        inputs = self._processor(images=img, task_inputs=[self.task], return_tensors="pt")
        with torch.no_grad():
            outputs = self._model(**inputs)

        predicted_panoptic = self._processor.post_process_panoptic_segmentation(
            outputs, target_sizes=[img.size[::-1]]
        )[0]

        regions: List[SegmentedRegion] = []
        panoptic_seg = predicted_panoptic["segmentation"].cpu().numpy()
        for i, info in enumerate(predicted_panoptic["segments_info"]):
            score = float(info.get("score", 0.90))
            if score < self.confidence_threshold:
                continue
            label_id = info["label_id"]
            label_name = self._model.config.id2label.get(label_id, f"class_{label_id}")
            regions.append(
                SegmentedRegion(
                    region_id=f"oneformer_{evidence.id}_{i}",
                    evidence_id=evidence.id,
                    label=label_name,
                    mask=(panoptic_seg == info["id"]).tolist(),
                    confidence=score,
                    uncertainty=Uncertainty(confidence=score, note=f"OneFormer {self.task} segmentation"),
                )
            )
        return SegmentationResult(
            evidence_id=evidence.id,
            regions=regions,
            model_name=self.model_name,
            model_version=self.model_version,
        )
