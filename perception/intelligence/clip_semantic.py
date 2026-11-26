"""CLIP / SigLIP Open-Vocabulary Semantic Verifier (A4.8).

Provides zero-shot region-level semantic verification and fine-grained
attribute scoring using locally cached CLIP or SigLIP models (with honest
availability probing and no silent downloads).
"""

from __future__ import annotations

import importlib.util
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SemanticVerificationResult:
    observation_id: str
    candidate_labels: tuple[str, ...]
    scores_by_label: dict[str, float]
    top_label: str
    top_score: float
    model_id: str
    is_calibrated: bool = False  # Cosine-softmax scores are not calibrated probabilities


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


class ClipSemanticVerifier:
    """Zero-shot CLIP / SigLIP crop verifier loaded strictly from local HF cache."""

    CANDIDATE_REPOS: tuple[str, ...] = (
        "google/siglip-base-patch16-224",
        "openai/clip-vit-base-patch32",
    )

    def __init__(self, model_id: str | None = None, device: str | None = None) -> None:
        self.model_id = model_id or self._discover_cached_repo() or self.CANDIDATE_REPOS[1]
        self._device = device
        self._model = None
        self._processor = None

    @classmethod
    def _discover_cached_repo(cls) -> str | None:
        for repo in cls.CANDIDATE_REPOS:
            if _is_hf_cached(repo):
                return repo
        return None

    def is_available(self) -> tuple[bool, str]:
        if importlib.util.find_spec("transformers") is None or importlib.util.find_spec("torch") is None:
            return False, "torch/transformers missing"
        if not _is_hf_cached(self.model_id):
            return False, f"checkpoint {self.model_id} not in local HF cache"
        return True, f"{self.model_id} cached"

    def load(self) -> None:
        if self._model is not None:
            return
        ok, reason = self.is_available()
        if not ok:
            raise RuntimeError(f"ClipSemanticVerifier unavailable: {reason}")
        import torch
        from transformers import AutoModel, AutoProcessor

        dev = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._processor = AutoProcessor.from_pretrained(self.model_id, local_files_only=True)
        self._model = AutoModel.from_pretrained(self.model_id, local_files_only=True).to(dev).eval()
        self._device = dev

    def unload(self) -> None:
        self._model = None
        self._processor = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def score_crop(
        self,
        crop_rgb: np.ndarray,
        candidate_labels: Sequence[str],
        *,
        observation_id: str = "crop",
    ) -> SemanticVerificationResult:
        if not candidate_labels:
            raise ValueError("candidate_labels must be non-empty")
        import torch
        from PIL import Image

        self.load()
        pil = Image.fromarray(np.ascontiguousarray(crop_rgb))
        prompts = [f"a photo of a {lbl.replace('_', ' ')}" for lbl in candidate_labels]
        inputs = self._processor(text=prompts, images=pil, return_tensors="pt", padding=True).to(self._device)
        with torch.no_grad():
            out = self._model(**inputs)
        logits = out.logits_per_image.squeeze(0).detach().cpu()
        probs = torch.softmax(logits, dim=-1).numpy()
        scores = {str(lbl): round(float(p), 4) for lbl, p in zip(candidate_labels, probs)}
        top_lbl = max(scores, key=lambda k: scores[k])
        return SemanticVerificationResult(
            observation_id=observation_id,
            candidate_labels=tuple(candidate_labels),
            scores_by_label=scores,
            top_label=top_lbl,
            top_score=scores[top_lbl],
            model_id=self.model_id,
            is_calibrated=False,
        )


def rank_attribute_hypotheses_by_text(
    crop_rgb: np.ndarray,
    attribute_candidates: Sequence[str],
    *,
    verifier: ClipSemanticVerifier | None = None,
    observation_id: str = "crop",
) -> dict[str, Any]:
    """Score fine-grained object attributes when CLIP/SigLIP is available, else return unavailable."""
    v = verifier or ClipSemanticVerifier()
    ok, reason = v.is_available()
    if not ok:
        return {"status": "unavailable", "reason": reason, "scores": {}}
    try:
        res = v.score_crop(crop_rgb, attribute_candidates, observation_id=observation_id)
        return {
            "status": "succeeded",
            "model_id": res.model_id,
            "top_label": res.top_label,
            "top_score": res.top_score,
            "scores": res.scores_by_label,
            "is_calibrated": False,
        }
    finally:
        v.unload()


__all__ = [
    "ClipSemanticVerifier",
    "SemanticVerificationResult",
    "rank_attribute_hypotheses_by_text",
]
