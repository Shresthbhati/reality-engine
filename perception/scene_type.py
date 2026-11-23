"""Indoor / outdoor / unknown scene-type gate (D4 of docs/reconstruction/REAL_RUN_STAGE_REPORT.md).

The room-structure compile (walls / ceiling / windows) and the indoor-furniture perception vocabulary are only meaningful
for interiors. Without a gate they run on any photographs: on the exterior South Building control they promoted wardrobes,
floor lamps and rugs and compiled 9 walls + a ceiling from a facade.

Evidence is the ADE20K semantic segmentation already used by the single-image layout stage (``perception.layout.learned_models``,
``openmmlab/upernet-convnext-tiny``, local cache only; no new model). Per image the share of pixels labelled with
outdoor-only classes (sky, building, tree, road ...) is compared with the share labelled with indoor-only classes (ceiling,
floor, bed, sofa, cabinet ...). ``wall`` is deliberately in neither list: facades and rooms both have walls.

    score(image) = (indoor - outdoor) / (indoor + outdoor)       in [-1, 1]; image is "measured" only if indoor + outdoor >= MIN_COVERAGE
    score(scene) = mean over measured images
    indoor  if score >= +CONFIDENCE_THRESHOLD   (confidence = score)
    outdoor if score <= -CONFIDENCE_THRESHOLD   (confidence = -score)
    unknown otherwise (mixed evidence)
    unmeasured when no image has enough evidence or the segmenter is unavailable (recorded, never silently treated as indoor)

``decide_scene_gate`` turns that into allow/block. The indoor path runs only for ``indoor``; ``outdoor`` and ``unknown`` are
blocked with an explicit reason; ``unmeasured`` fails OPEN (the gate cannot judge, which is recorded as degraded) so a missing
model never takes the interior pipeline away.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

log = logging.getLogger(__name__)

#: ADE20K class names that occur only (or overwhelmingly) outdoors.
OUTDOOR_CLASSES = frozenset({
    "sky", "building", "house", "skyscraper", "tree", "grass", "road", "sidewalk", "earth", "fence", "car", "mountain",
    "hill", "sea", "river", "field", "path", "streetlight", "bridge", "tower", "hovel", "palm", "boat", "bus", "truck",
    "van", "bicycle", "airplane", "water", "sand", "dirt track", "traffic light", "awning",
})
#: ADE20K class names that occur only (or overwhelmingly) indoors.
INDOOR_CLASSES = frozenset({
    "ceiling", "floor", "bed", "sofa", "cabinet", "chair", "armchair", "table", "rug", "curtain", "wardrobe", "lamp",
    "kitchen island", "refrigerator", "bookcase", "desk", "chest of drawers", "counter", "sink", "toilet", "bathtub",
    "shelf", "pillow", "cushion", "coffee table", "stove", "oven", "microwave", "television receiver", "light",
    "chandelier", "mirror", "blind", "bannister", "stool", "bench", "dishwasher", "washer", "fireplace", "bar",
    "crt screen", "computer", "swivel chair", "ottoman", "basket", "tray", "plate", "bottle", "glass", "vase", "book",
})

MIN_COVERAGE = 0.10            # fraction of an image that must be labelled outdoor-or-indoor for it to count as measured
CONFIDENCE_THRESHOLD = 0.30    # |mean score| required to call the scene indoor or outdoor; below it the scene is "unknown"
MAX_VIEWS = 8                  # evenly-spaced views segmented (bounded cost on big captures)
SEG_LONG_SIDE = 512            # px; scene type needs no fine detail


@dataclass
class SceneTypeResult:
    scene_type: str                  # "indoor" | "outdoor" | "unknown" | "unmeasured"
    confidence: float                # |mean score| for indoor/outdoor/unknown; 0.0 when unmeasured
    score: Optional[float]           # mean (indoor - outdoor)/(indoor + outdoor); None when unmeasured
    views_measured: int
    views_total: int
    evidence: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"scene_type": self.scene_type, "confidence": round(self.confidence, 4),
                "score": None if self.score is None else round(self.score, 4),
                "views_measured": self.views_measured, "views_total": self.views_total, "evidence": dict(self.evidence)}


@dataclass
class SceneGateDecision:
    allow_indoor: bool
    state: str                       # "passed" | "skipped" (interior path blocked) | "degraded" (could not judge)
    reason: str
    result: SceneTypeResult

    def to_dict(self) -> dict:
        return {"allow_indoor": self.allow_indoor, "state": self.state, "reason": self.reason, "scene": self.result.to_dict()}


def image_fractions(label_map: np.ndarray, id2label: Mapping[int, str]) -> Dict[str, object]:
    """Outdoor / indoor pixel shares of one ADE20K label map, plus the three most frequent labels in each group."""
    total = float(label_map.size) or 1.0
    ids, counts = np.unique(label_map, return_counts=True)
    out_c: Dict[str, float] = {}
    in_c: Dict[str, float] = {}
    for i, c in zip(ids.tolist(), counts.tolist()):
        name = str(id2label.get(int(i), "")).split(",")[0].strip().lower()
        if name in OUTDOOR_CLASSES:
            out_c[name] = out_c.get(name, 0.0) + c / total
        elif name in INDOOR_CLASSES:
            in_c[name] = in_c.get(name, 0.0) + c / total
    top = lambda d: dict(sorted(d.items(), key=lambda kv: -kv[1])[:3])  # noqa: E731
    return {"outdoor": sum(out_c.values()), "indoor": sum(in_c.values()), "top_outdoor": top(out_c), "top_indoor": top(in_c)}


def classify_from_fractions(per_image: Sequence[Mapping[str, Any]], views_total: Optional[int] = None,
                            threshold: float = CONFIDENCE_THRESHOLD, min_coverage: float = MIN_COVERAGE) -> SceneTypeResult:
    """Pure decision rule over per-image {"outdoor", "indoor"} shares (see module docstring)."""
    views_total = len(per_image) if views_total is None else views_total
    scores: List[float] = []
    for f in per_image:
        o, i = float(f["outdoor"]), float(f["indoor"])
        if o + i >= min_coverage:
            scores.append((i - o) / (i + o))
    ev: Dict[str, object] = {
        "mean_outdoor_share": round(float(np.mean([float(f["outdoor"]) for f in per_image])), 4) if per_image else None,
        "mean_indoor_share": round(float(np.mean([float(f["indoor"]) for f in per_image])), 4) if per_image else None,
        "per_view_scores": [round(s, 3) for s in scores], "threshold": threshold, "min_coverage": min_coverage,
        "top_outdoor_classes": _merge_top(per_image, "top_outdoor"), "top_indoor_classes": _merge_top(per_image, "top_indoor"),
    }
    if not scores:
        return SceneTypeResult("unmeasured", 0.0, None, 0, views_total, ev)
    score = float(np.mean(scores))
    if score >= threshold:
        kind = "indoor"
    elif score <= -threshold:
        kind = "outdoor"
    else:
        kind = "unknown"
    return SceneTypeResult(kind, abs(score), score, len(scores), views_total, ev)


def _merge_top(per_image: Sequence[Mapping[str, Any]], key: str) -> Dict[str, float]:
    agg: Dict[str, float] = {}
    for f in per_image:
        for k, v in (f.get(key) or {}).items():
            agg[k] = agg.get(k, 0.0) + float(v)
    n = max(len(per_image), 1)
    return {k: round(v / n, 4) for k, v in sorted(agg.items(), key=lambda kv: -kv[1])[:4]}


def decide_scene_gate(result: SceneTypeResult) -> SceneGateDecision:
    """Allow the interior path only for a confidently indoor scene (see module docstring for the unmeasured exception)."""
    t = result.scene_type
    if t == "indoor":
        return SceneGateDecision(True, "passed", f"indoor scene (confidence {result.confidence:.2f})", result)
    if t == "unmeasured":
        return SceneGateDecision(True, "degraded", "scene type could not be measured (segmenter unavailable or too little "
                                 "labelled evidence); interior stages ran unguarded", result)
    why = ("outdoor-class pixels dominate (sky/building/vegetation/road)" if t == "outdoor"
           else "indoor and outdoor evidence are mixed")
    return SceneGateDecision(False, "skipped", f"scene is {t} (confidence {result.confidence:.2f} on {result.views_measured}/"
                             f"{result.views_total} views): {why}; room layout and indoor-furniture perception skipped", result)


def _evenly(seq: Sequence, n: int) -> list:
    if len(seq) <= n:
        return list(seq)
    return [seq[int(round(k * (len(seq) - 1) / (n - 1)))] for k in range(n)]


def classify_scene_type(image_paths: Iterable[Path], *, device: Optional[str] = None, models=None) -> SceneTypeResult:
    """Segment up to MAX_VIEWS evenly spaced photographs with the cached ADE20K UperNet and apply the decision rule.

    Never raises: an unavailable segmenter or unreadable images give ``unmeasured`` with the reason in ``evidence``.
    ``models`` is an optional ``(segmenter_callable)`` override for tests: ``f(path) -> (label_map, id2label)``."""
    paths = [Path(p) for p in image_paths]
    chosen = _evenly(paths, MAX_VIEWS)
    seg = models
    release = None
    try:
        if seg is None:
            seg, release = _load_upernet(device)
        per_image = []
        for p in chosen:
            try:
                label_map, id2label = seg(p)
            except Exception as exc:  # noqa: BLE001 -- one unreadable image must not void the others
                log.warning("scene type: could not segment %s: %s", p, exc)
                continue
            per_image.append(image_fractions(label_map, id2label))
        res = classify_from_fractions(per_image, views_total=len(chosen))
        if not per_image:
            res.evidence["error"] = "no image could be segmented"
        return res
    except Exception as exc:  # noqa: BLE001 -- the gate is advisory infrastructure; failure is recorded
        log.warning("scene type classifier unavailable: %s", exc)
        return SceneTypeResult("unmeasured", 0.0, None, 0, len(chosen), {"error": f"{type(exc).__name__}: {exc}"})
    finally:
        seg = None
        if release is not None:
            release()


def _load_upernet(device: Optional[str]):
    import torch
    from transformers import AutoImageProcessor, UperNetForSemanticSegmentation

    from engine.core.image_io import open_oriented
    from perception.layout.learned_models import SEG_MODEL

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    proc = AutoImageProcessor.from_pretrained(SEG_MODEL, local_files_only=True)
    raw_model: Any = UperNetForSemanticSegmentation.from_pretrained(SEG_MODEL, local_files_only=True)
    model = raw_model.to(dev).eval()
    id2label = {int(k): str(v).strip() for k, v in model.config.id2label.items()}
    state = {"model": model}

    def segment(path: Path):
        img = open_oriented(Path(path)).convert("RGB")
        w, h = img.size
        s = SEG_LONG_SIDE / max(w, h)
        if s < 1.0:
            img = img.resize((max(1, int(w * s)), max(1, int(h * s))))
        w, h = img.size
        with torch.no_grad():
            inp = proc(images=img, return_tensors="pt").to(dev)
            out = state["model"](**inp)
            lm = proc.post_process_semantic_segmentation(out, target_sizes=[(h, w)])[0]
        return lm.cpu().numpy().astype(np.int32), id2label

    def release():
        state["model"] = None
        from engine.core.device import release_vram
        release_vram()

    return segment, release
