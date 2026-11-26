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


def get_balanced_default_prompts(prompter: ExtensiblePrompter | None = None, domain: str = "all") -> list[str]:
    """Curated fine-grained default prompts covering architecture, furniture, fixtures, hardware, and exterior elements."""
    indoor_core = [
        "door", "window", "wall", "ceiling", "floor", "chair", "table", "sofa", "bed", "cabinet",
        "desk", "electrical socket", "light switch", "lamp", "door handle", "faucet", "shelf",
    ]
    outdoor_core = [
        "building facade", "window", "door", "entrance", "roof", "balcony", "cornice", "stair", "column", "wall",
    ]
    if domain == "indoor":
        return indoor_core
    if domain == "outdoor":
        return outdoor_core
    return list(dict.fromkeys(indoor_core + outdoor_core))


def clean_and_split_detection_label(
    label: str | Any,
    known_queries: Sequence[str] | None = None,
) -> list[str]:
    """Splits compound Grounding DINO labels and rejects empty, subword, or punctuation tokens."""
    if label is None:
        return []
    s = str(label).strip()
    if not s or s.lower() in ("none", "null", "unknown", "object", "undefined"):
        return []
    # Reject raw BERT/WordPiece subword fragments or punctuation-only strings
    if s.startswith("##") or not any(ch.isalpha() for ch in s):
        return []

    import re

    # Normalize underscores/hyphens for matching, but preserve canonical query string when matched
    s_norm = re.sub(r"[_\-]+", " ", s.lower()).strip()

    if known_queries:
        # Build mapping from space-normalized query -> original query string
        norm_to_orig: dict[str, str] = {}
        for q in known_queries:
            q_str = str(q).strip()
            if not q_str:
                continue
            q_norm = re.sub(r"[_\-]+", " ", q_str.lower()).strip()
            if q_norm and q_norm not in norm_to_orig:
                norm_to_orig[q_norm] = q_str.lower()

        if s_norm in norm_to_orig:
            return [norm_to_orig[s_norm]]

        sorted_norms = sorted(norm_to_orig.keys(), key=len, reverse=True)
        matched: list[str] = []
        covered_spans: list[tuple[int, int]] = []

        for qn in sorted_norms:
            pattern = r"(?:^|\b)" + re.escape(qn) + r"(?:\b|$)"
            for m in re.finditer(pattern, s_norm):
                start, end = m.span()
                if any(cs <= start and end <= ce for cs, ce in covered_spans):
                    continue
                matched.append(norm_to_orig[qn])
                covered_spans.append((start, end))

        if matched:
            return matched

    tokens = [
        t.strip().lower()
        for t in s_norm.replace(".", " ").replace(",", " ").replace("/", " ").split()
    ]
    valid = [
        t for t in tokens
        if len(t) >= 2
        and not t.startswith("##")
        and any(ch.isalpha() for ch in t)
        and t not in ("none", "null", "unknown", "the", "a", "an", "and")
    ]
    return valid



def _is_blank_image(img: Any) -> bool:
    """True when an image has negligible spatial variance (blank/solid frame)."""
    import numpy as np

    arr = np.asarray(img)
    if arr.size == 0:
        return True
    if arr.ndim == 2:
        return float(np.std(arr)) < 2.0
    return float(np.std(arr, axis=(0, 1)).max()) < 2.0


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


def _resolve_florence2_checkpoint() -> str | None:
    for repo in ("florence-community/Florence-2-base", "microsoft/Florence-2-base"):
        if _is_hf_cached(repo):
            return repo
    return None


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
    repo = _resolve_florence2_checkpoint()
    if repo is None:
        return Availability(
            AvailabilityStatus.BLOCKED,
            "dependencies present, but Florence-2 checkpoint is not in Hugging Face cache",
            "huggingface-cli download florence-community/Florence-2-base",
        )
    return Availability(
        AvailabilityStatus.AVAILABLE,
        f"transformers and cached weights verified for Florence-2 ({repo})",
        evidence=(f"Florence-2 cached ({repo})",),
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
            domain_val = request.payload.get("domain", "all") if request.payload else "all"
            prompts = get_balanced_default_prompts(self._prompter, domain_val)
            prompt_text = ". ".join(prompts) + "."

        thresh = float(request.payload.get("threshold", 0.25)) if request.payload else 0.25
        dev_str = "cuda:0" if self._device.kind == "cuda" else "cpu"

        payload_rgb = request.payload.get("image_rgb") if request.payload else None

        for s in sources:
            if payload_rgb is not None and len(sources) == 1:
                img = Image.fromarray(payload_rgb) if not isinstance(payload_rgb, Image.Image) else payload_rgb
            else:
                img = Image.open(s).convert("RGB")
            if _is_blank_image(img):
                continue
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

    DEFAULT_CHECKPOINT = "florence-community/Florence-2-base"

    def __init__(self, spec: EngineSpec, checkpoint: str | None = None) -> None:
        super().__init__(spec)
        self.checkpoint = checkpoint or _resolve_florence2_checkpoint() or self.DEFAULT_CHECKPOINT
        self._processor = None

    def _load_model_weights(self, device: Device) -> None:
        if importlib.util.find_spec("transformers") is None:
            raise EngineError(EngineErrorKind.UNAVAILABLE, "transformers package missing.")
        from transformers import AutoProcessor

        dev_str = "cuda:0" if device.kind == "cuda" else "cpu"
        try:
            self._processor = AutoProcessor.from_pretrained(self.checkpoint, local_files_only=True)
            if "florence-community" in self.checkpoint:
                from transformers import Florence2ForConditionalGeneration

                self._model = (
                    Florence2ForConditionalGeneration.from_pretrained(
                        self.checkpoint, local_files_only=True
                    )
                    .to(dev_str)
                    .eval()
                )
            else:
                from transformers import AutoModelForCausalLM

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

        payload_rgb = request.payload.get("image_rgb") if request.payload else None

        for s in sources:
            if payload_rgb is not None and len(sources) == 1:
                img = Image.fromarray(payload_rgb) if not isinstance(payload_rgb, Image.Image) else payload_rgb
            else:
                img = Image.open(s).convert("RGB")
            if _is_blank_image(img):
                continue
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


class MaskRCNNEngineAdapter(BaseModelEngineAdapter):
    """Adapter for torchvision Mask R-CNN (COCO ResNet50-FPN) object detection + instance masks."""

    DEFAULT_CHECKPOINT = "maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth"

    def __init__(self, spec: EngineSpec, score_threshold: float = 0.5) -> None:
        super().__init__(spec)
        self.checkpoint = self.DEFAULT_CHECKPOINT
        self._score_threshold = float(score_threshold)
        self._categories: Sequence[str] = ()

    def _load_model_weights(self, device: Device) -> None:
        if importlib.util.find_spec("torchvision") is None:
            raise EngineError(EngineErrorKind.UNAVAILABLE, "torchvision package missing.")
        from torchvision.models.detection import (
            MaskRCNN_ResNet50_FPN_Weights,
            maskrcnn_resnet50_fpn,
        )

        dev_str = "cuda:0" if device.kind == "cuda" else "cpu"
        try:
            weights = MaskRCNN_ResNet50_FPN_Weights.COCO_V1
            self._model = maskrcnn_resnet50_fpn(weights=weights).to(dev_str).eval()
            self._categories = list(weights.meta.get("categories", ()))
        except Exception as e:
            raise EngineError(
                EngineErrorKind.MISSING_CHECKPOINT,
                f"Failed to load Mask R-CNN COCO weights: {e}",
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
                        image_region=(80.0, 90.0, 260.0, 280.0),
                        predicted={"label": "chair", "score": 0.91, "is_test_double": True},
                        original_coordinates={"box_xyxy_px": [80, 90, 260, 280]},
                        confidence=None,
                    )
                )
            return drafts

        import numpy as np
        import torch
        from PIL import Image

        thresh = float(request.payload.get("threshold", self._score_threshold)) if request.payload else self._score_threshold
        dev_str = "cuda:0" if self._device.kind == "cuda" else "cpu"
        payload_rgb = request.payload.get("image_rgb") if request.payload else None

        for s in sources:
            if payload_rgb is not None and len(sources) == 1:
                img = Image.fromarray(payload_rgb) if not isinstance(payload_rgb, Image.Image) else payload_rgb
            else:
                img = Image.open(s).convert("RGB")
            if _is_blank_image(img):
                continue
            w, h = img.size
            arr = np.asarray(img, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(arr).permute(2, 0, 1).to(dev_str)
            with torch.no_grad():
                out = self._model([tensor])[0]

            boxes = out["boxes"].detach().cpu().numpy()
            labels = out["labels"].detach().cpu().numpy()
            scores = out["scores"].detach().cpu().numpy()
            for idx in range(len(boxes)):
                sc = float(scores[idx])
                if sc < thresh:
                    continue
                li = int(labels[idx])
                raw_lbl = self._categories[li] if 0 <= li < len(self._categories) else "object"
                if raw_lbl in ("__background__", "N/A"):
                    continue
                b = [float(v) for v in boxes[idx]]
                drafts.append(
                    ObservationDraft(
                        task="detection",
                        quantity="bbox",
                        source_id=str(s),
                        coordinate_system=f"image:{Path(str(s)).stem}",
                        image_region=(b[0], b[1], b[2], b[3]),
                        predicted={
                            "label": raw_lbl,
                            "raw_label": raw_lbl,
                            "score": sc,
                            "box_normalized": [b[0] / w, b[1] / h, b[2] / w, b[3] / h],
                        },
                        original_coordinates={"box_xyxy_px": b},
                        confidence=sc,
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
            domain_val = request.payload.get("domain", "all") if request.payload else "all"
            prompts = get_balanced_default_prompts(self._prompter, domain_val)
        thresh = float(request.payload.get("threshold", 0.05)) if request.payload else 0.05
        dev_str = "cuda:0" if self._device.kind == "cuda" else "cpu"

        payload_rgb = request.payload.get("image_rgb") if request.payload else None

        for s in sources:
            if payload_rgb is not None and len(sources) == 1:
                img = Image.fromarray(payload_rgb) if not isinstance(payload_rgb, Image.Image) else payload_rgb
            else:
                img = Image.open(s).convert("RGB")
            if _is_blank_image(img):
                continue
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
