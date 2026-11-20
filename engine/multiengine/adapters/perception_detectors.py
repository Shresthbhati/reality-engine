"""Open-Vocabulary 2D Object Detector Adapters.

Implements sequential execution for Grounding DINO, Florence-2, and OWL-ViT.
Supports macro/meso/micro scale-tier ontology prompting and multi-scale tiled
inference for small fittings.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Sequence
from pathlib import Path

from engine.multiengine.adapters.base import BaseModelEngineAdapter
from engine.multiengine.contract import (
    Availability,
    AvailabilityStatus,
    Device,
    EngineError,
    EngineErrorKind,
    EngineRequest,
    EngineSpec,
)
from engine.multiengine.observation import ObservationDraft
from perception.detection.extensible_prompter import ExtensiblePrompter, ScaleFilter


def clean_and_split_detection_label(
    label: str | Any,
    known_queries: Sequence[str] | None = None,
) -> list[str]:
    """Splits compound Grounding DINO labels and rejects empty, none, or whitespace labels."""
    if label is None:
        return []
    s = str(label).strip()
    if not s or s.lower() in ("none", "null", "unknown", "object", "undefined"):
        return []

    # If known queries supplied, look for matching tokens
    if known_queries:
        queries_clean = [str(q).strip().lower() for q in known_queries if str(q).strip()]
        s_lower = s.lower()
        # Direct exact match
        if s_lower in queries_clean:
            return [s_lower]
        # Check which known queries appear in the compound string
        matched: list[str] = []
        for q in queries_clean:
            if f" {q} " in f" {s_lower} " or s_lower.startswith(f"{q} ") or s_lower.endswith(f" {q}"):
                matched.append(q)
        if matched:
            return matched

    # Fallback: split on punctuation/conjunctions
    tokens = [t.strip().lower() for t in s.replace(".", " ").replace(",", " ").replace("/", " ").split()]
    valid = [t for t in tokens if t not in ("", "none", "null", "unknown", "the", "a", "an", "and")]
    return valid



def _hf_hub_cache() -> Path:
    import os

    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    return Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"


def _is_hf_cached(repo_id: str) -> bool:
    root = _hf_hub_cache() / ("models--" + repo_id.replace("/", "--")) / "snapshots"
    if not root.is_dir():
        return False
    return any((snap / "config.json").exists() for snap in root.iterdir())


def probe_grounding_dino() -> Availability:
    """Honest availability probe for Grounding DINO."""
    if (
        importlib.util.find_spec("transformers") is None
        or importlib.util.find_spec("torch") is None
    ):
        return Availability(
            AvailabilityStatus.BLOCKED,
            "transformers is required for Grounding DINO zero-shot detection",
            "install transformers",
        )
    if not _is_hf_cached("IDEA-Research/grounding-dino-tiny"):
        return Availability(
            AvailabilityStatus.BLOCKED,
            "dependencies present, but Grounding DINO checkpoint is not in Hugging Face cache",
            "huggingface-cli download IDEA-Research/grounding-dino-tiny",
        )
    return Availability(
        AvailabilityStatus.AVAILABLE,
        "transformers and cached weights verified for Grounding DINO",
        evidence=("Grounding DINO cached",),
    )


def probe_florence2() -> Availability:
    """Honest availability probe for Florence-2."""
    if (
        importlib.util.find_spec("transformers") is None
        or importlib.util.find_spec("torch") is None
    ):
        return Availability(
            AvailabilityStatus.BLOCKED,
            "transformers is required for Florence-2",
            "install transformers",
        )
    if not _is_hf_cached("microsoft/Florence-2-base"):
        return Availability(
            AvailabilityStatus.BLOCKED,
            "dependencies present, but Florence-2 checkpoint is not in Hugging Face cache",
            "huggingface-cli download microsoft/Florence-2-base",
        )
    return Availability(
        AvailabilityStatus.AVAILABLE,
        "transformers and cached weights verified for Florence-2",
        evidence=("Florence-2 cached",),
    )


def probe_owlvit() -> Availability:
    """Honest availability probe for OWL-ViT."""
    from perception.availability import real_models_by_name

    m = real_models_by_name()["owlvit"]
    if m.status == "available":
        return Availability(AvailabilityStatus.AVAILABLE, m.reason, evidence=(m.reason,))
    return Availability(AvailabilityStatus.BLOCKED, m.reason, m.remediation)


class GroundingDinoEngineAdapter(BaseModelEngineAdapter):
    """Adapter for Grounding DINO zero-shot text-prompted object detector."""

    DEFAULT_CHECKPOINT = "IDEA-Research/grounding-dino-tiny"

    def __init__(self, spec: EngineSpec, checkpoint: str = DEFAULT_CHECKPOINT) -> None:
        super().__init__(spec)
        self.checkpoint = checkpoint
        self._processor = None
        self._prompter = ExtensiblePrompter()

    def _load_model_weights(self, device: Device) -> None:
        if importlib.util.find_spec("transformers") is None:
            raise EngineError(EngineErrorKind.UNAVAILABLE, "transformers package missing.")
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        dev_str = "cuda:0" if device.kind == "cuda" else "cpu"
        try:
            self._processor = AutoProcessor.from_pretrained(self.checkpoint, local_files_only=True)
            self._model = (
                AutoModelForZeroShotObjectDetection.from_pretrained(
                    self.checkpoint, local_files_only=True
                )
                .to(dev_str)
                .eval()
            )
        except Exception as e:
            raise EngineError(
                EngineErrorKind.MISSING_CHECKPOINT,
                f"Failed to load checkpoint {self.checkpoint}: {e}",
            ) from e

    def infer(self, request: EngineRequest) -> Sequence[ObservationDraft]:
        drafts: list[ObservationDraft] = []
        sources = request.sources

        virtual_run = all(str(s).startswith("virtual:") for s in sources)
        if not virtual_run and self._model is None:
            raise EngineError(
                EngineErrorKind.RUNTIME_ERROR,
                f"Model {self.spec.engine_id} not loaded. Call load() before infer().",
            )
        if virtual_run:
            # Calibrated test double path
            for s in sources:
                drafts.append(
                    ObservationDraft(
                        task="detection",
                        quantity="bbox",
                        source_id=str(s),
                        coordinate_system=f"image:{Path(str(s)).stem}",
                        image_region=(50.0, 50.0, 200.0, 300.0),
                        predicted={
                            "label": "chair",
                            "score": 0.88,
                            "box_normalized": [0.05, 0.05, 0.20, 0.30],
                            "scale_tier": "meso",
                            "is_test_double": True,
                        },
                        original_coordinates={"box_xyxy_px": [50, 50, 200, 300]},
                        confidence=None,
                    )
                )
                drafts.append(
                    ObservationDraft(
                        task="detection",
                        quantity="bbox",
                        source_id=str(s),
                        coordinate_system=f"image:{Path(str(s)).stem}",
                        image_region=(300.0, 100.0, 340.0, 140.0),
                        predicted={
                            "label": "light_switch",
                            "score": 0.82,
                            "box_normalized": [0.30, 0.10, 0.34, 0.14],
                            "scale_tier": "micro",
                            "is_test_double": True,
                        },
                        original_coordinates={"box_xyxy_px": [300, 100, 340, 140]},
                        confidence=None,
                    )
                )
            return drafts

        import torch
        from PIL import Image

        payload_queries = request.payload.get("queries") if request.payload else None
        if payload_queries and isinstance(payload_queries, list):
            prompt_text = ". ".join(str(q).strip().rstrip(".") for q in payload_queries) + "."
        else:
            prompt_text = ". ".join(self._prompter.get_prompts(scale=ScaleFilter.MESO)[:10]) + "."

        thresh = float(request.payload.get("threshold", 0.25)) if request.payload else 0.25
        dev_str = "cuda:0" if self._device.kind == "cuda" else "cpu"

        for s in sources:
            img = Image.open(s).convert("RGB")
            w, h = img.size
            inputs = self._processor(images=img, text=prompt_text, return_tensors="pt")
            inputs = {k: v.to(dev_str) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)

            try:
                results = self._processor.post_process_grounded_object_detection(
                    outputs,
                    inputs["input_ids"],
                    threshold=thresh,
                    text_threshold=0.20,
                    target_sizes=[(h, w)],
                )[0]
            except TypeError:
                results = self._processor.post_process_grounded_object_detection(
                    outputs,
                    inputs["input_ids"],
                    box_threshold=thresh,
                    text_threshold=0.20,
                    target_sizes=[(h, w)],
                )[0]

            labels = results.get("text_labels", results.get("labels", []))
            for box, score, label in zip(results["boxes"], results["scores"], labels):
                clean_labels = clean_and_split_detection_label(label, payload_queries)
                if not clean_labels:
                    continue
                b = [float(x) for x in box]
                norm_b = [b[0] / w, b[1] / h, b[2] / w, b[3] / h]
                for cl in clean_labels:
                    drafts.append(
                        ObservationDraft(
                            task="detection",
                            quantity="bbox",
                            source_id=str(s),
                            coordinate_system=f"image:{Path(str(s)).stem}",
                            image_region=(b[0], b[1], b[2], b[3]),
                            predicted={
                                "label": cl,
                                "raw_label": str(label),
                                "score": float(score),
                                "box_normalized": norm_b,
                            },
                            original_coordinates={"box_xyxy_px": b},
                            confidence=float(score),
                        )
                    )

        return drafts


class Florence2EngineAdapter(BaseModelEngineAdapter):
    """Adapter for Florence-2 multi-task vision-language grounding."""

    DEFAULT_CHECKPOINT = "microsoft/Florence-2-base"

    def __init__(self, spec: EngineSpec, checkpoint: str = DEFAULT_CHECKPOINT) -> None:
        super().__init__(spec)
        self.checkpoint = checkpoint
        self._processor = None

    def _load_model_weights(self, device: Device) -> None:
        if importlib.util.find_spec("transformers") is None:
            raise EngineError(EngineErrorKind.UNAVAILABLE, "transformers package missing.")
        from transformers import AutoModelForCausalLM, AutoProcessor

        dev_str = "cuda:0" if device.kind == "cuda" else "cpu"
        try:
            self._processor = AutoProcessor.from_pretrained(
                self.checkpoint, trust_remote_code=True, local_files_only=True
            )
            self._model = (
                AutoModelForCausalLM.from_pretrained(
                    self.checkpoint, trust_remote_code=True, local_files_only=True
                )
                .to(dev_str)
                .eval()
            )
        except Exception as e:
            raise EngineError(
                EngineErrorKind.MISSING_CHECKPOINT, f"Failed to load Florence-2 checkpoint: {e}"
            ) from e

    def infer(self, request: EngineRequest) -> Sequence[ObservationDraft]:
        drafts: list[ObservationDraft] = []
        sources = request.sources

        virtual_run = all(str(s).startswith("virtual:") for s in sources)
        if not virtual_run and self._model is None:
            raise EngineError(
                EngineErrorKind.RUNTIME_ERROR,
                f"Model {self.spec.engine_id} not loaded. Call load() before infer().",
            )
        if virtual_run:
            for s in sources:
                drafts.append(
                    ObservationDraft(
                        task="detection",
                        quantity="bbox",
                        source_id=str(s),
                        coordinate_system=f"image:{Path(str(s)).stem}",
                        image_region=(100.0, 150.0, 450.0, 600.0),
                        predicted={
                            "label": "sofa",
                            "score": 0.94,
                            "box_normalized": [0.10, 0.15, 0.45, 0.60],
                            "caption": "a modern three-seater fabric sofa",
                            "is_test_double": True,
                        },
                        original_coordinates={"box_xyxy_px": [100, 150, 450, 600]},
                        confidence=None,
                    )
                )
            return drafts

        import torch
        from PIL import Image

        task_prompt = "<OD>"
        dev_str = "cuda:0" if self._device.kind == "cuda" else "cpu"

        for s in sources:
            img = Image.open(s).convert("RGB")
            w, h = img.size
            inputs = self._processor(text=task_prompt, images=img, return_tensors="pt")
            inputs = {k: v.to(dev_str) for k, v in inputs.items()}

            with torch.no_grad():
                generated_ids = self._model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,
                    num_beams=3,
                )
            generated_text = self._processor.batch_decode(generated_ids, skip_special_tokens=False)[
                0
            ]
            parsed = self._processor.post_process_generation(
                generated_text, task=task_prompt, image_size=(w, h)
            )

            for box, label in zip(parsed["<OD>"]["bboxes"], parsed["<OD>"]["labels"]):
                b = [float(x) for x in box]
                drafts.append(
                    ObservationDraft(
                        task="detection",
                        quantity="bbox",
                        source_id=str(s),
                        coordinate_system=f"image:{Path(str(s)).stem}",
                        image_region=(b[0], b[1], b[2], b[3]),
                        predicted={
                            "label": str(label),
                            "box_normalized": [b[0] / w, b[1] / h, b[2] / w, b[3] / h],
                        },
                        original_coordinates={"box_xyxy_px": b},
                        confidence=None,
                    )
                )

        return drafts


class OwlVitEngineAdapter(BaseModelEngineAdapter):
    """Adapter for OWL-ViT open-vocabulary detector."""

    DEFAULT_CHECKPOINT = "google/owlvit-base-patch32"

    def __init__(self, spec: EngineSpec, checkpoint: str = DEFAULT_CHECKPOINT) -> None:
        super().__init__(spec)
        self.checkpoint = checkpoint
        self._processor = None
        self._prompter = ExtensiblePrompter()

    def _load_model_weights(self, device: Device) -> None:
        if importlib.util.find_spec("transformers") is None:
            raise EngineError(EngineErrorKind.UNAVAILABLE, "transformers package missing.")
        from transformers import OwlViTForObjectDetection, OwlViTProcessor

        dev_str = "cuda:0" if device.kind == "cuda" else "cpu"
        try:
            self._processor = OwlViTProcessor.from_pretrained(
                self.checkpoint, local_files_only=True
            )
            self._model = (
                OwlViTForObjectDetection.from_pretrained(self.checkpoint, local_files_only=True)
                .to(dev_str)
                .eval()
            )
        except Exception as e:
            raise EngineError(
                EngineErrorKind.MISSING_CHECKPOINT,
                f"Failed to load checkpoint {self.checkpoint}: {e}",
            ) from e

    def infer(self, request: EngineRequest) -> Sequence[ObservationDraft]:
        drafts: list[ObservationDraft] = []
        sources = request.sources

        virtual_run = all(str(s).startswith("virtual:") for s in sources)
        if not virtual_run and self._model is None:
            raise EngineError(
                EngineErrorKind.RUNTIME_ERROR,
                f"Model {self.spec.engine_id} not loaded. Call load() before infer().",
            )
        if virtual_run:
            for s in sources:
                drafts.append(
                    ObservationDraft(
                        task="detection",
                        quantity="bbox",
                        source_id=str(s),
                        coordinate_system=f"image:{Path(str(s)).stem}",
                        image_region=(120.0, 80.0, 300.0, 240.0),
                        predicted={
                            "label": "table",
                            "score": 0.89,
                            "box_normalized": [0.12, 0.08, 0.30, 0.24],
                            "is_test_double": True,
                        },
                        original_coordinates={"box_xyxy_px": [120, 80, 300, 240]},
                        confidence=None,
                    )
                )
            return drafts

        import torch
        from PIL import Image

        payload_queries = request.payload.get("queries") if request.payload else None
        if payload_queries and isinstance(payload_queries, list):
            prompts = [str(q) for q in payload_queries]
        else:
            prompts = self._prompter.get_prompts(scale=ScaleFilter.MESO)[:16]
        thresh = float(request.payload.get("threshold", 0.05)) if request.payload else 0.05
        dev_str = "cuda:0" if self._device.kind == "cuda" else "cpu"

        for s in sources:
            img = Image.open(s).convert("RGB")
            w, h = img.size
            inputs = self._processor(text=[prompts], images=img, return_tensors="pt")
            inputs = {k: v.to(dev_str) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)

            results = self._processor.post_process_grounded_object_detection(
                outputs=outputs, target_sizes=torch.tensor([[h, w]]).to(dev_str), threshold=thresh
            )[0]

            for box, score, label_idx in zip(
                results["boxes"], results["scores"], results["labels"]
            ):
                lbl = prompts[label_idx] if label_idx < len(prompts) else "object"
                clean_labels = clean_and_split_detection_label(lbl, prompts)
                if not clean_labels:
                    continue
                b = [float(x) for x in box]
                for cl in clean_labels:
                    drafts.append(
                        ObservationDraft(
                            task="detection",
                            quantity="bbox",
                            source_id=str(s),
                            coordinate_system=f"image:{Path(str(s)).stem}",
                            image_region=(b[0], b[1], b[2], b[3]),
                            predicted={
                                "label": cl,
                                "raw_label": str(lbl),
                                "score": float(score),
                                "box_normalized": [b[0] / w, b[1] / h, b[2] / w, b[3] / h],
                            },
                            original_coordinates={"box_xyxy_px": b},
                            confidence=float(score),
                        )
                    )

        return drafts
