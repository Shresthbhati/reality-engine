"""Attribute pass over an ObservationTable: which kind of socket, lock, handle (BUILD_SPEC B05, live path).

For every observation whose ontology node declares attributes, crop the photo, ask a detector once per allowed value
("<value> <label>", or the node's own phrase), decide with the margin rule in ``attributes`` (unknown unless clear),
optionally merge text read on the crop (OCR), and write the result into the row's ``attrs_json``. Across the views of
one entity the per-view records are combined (evidence capped at 3 views). The detector is injected, so this is pure
bookkeeping and testable; ``grounded_scorer`` is the real one. Opt-in in the pipeline: ``perception_attributes``.
"""
from __future__ import annotations

import json
from collections import Counter
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from perception.detection.attribute_phrases import phrase_for
from perception.detection.attributes import classify_attribute_from_crop, combine_views

MAX_ATTRS_JSON = 2000      # ObservationTable.attrs_json is a <U2000 column
DEFAULT_BUDGET_CALLS = 200  # detector calls per run: one per allowed value per observation


def attribute_phrase(name: str, value: str):
    """The vetted detector phrase for ``name`` = ``value`` (see ``attribute_phrases``), or None. No generated fallback:
    "left door" or "yes window" are not descriptions of appearance and would yield confident wrong answers."""
    return phrase_for(name, value)


def _pack(rec: Dict[str, dict]) -> str:
    slim = {n: {k: r[k] for k in ("value", "p", "basis", "obs", "conflict", "ocr_text", "reason") if k in r} for n, r in rec.items()}
    s = json.dumps(slim, separators=(",", ":"))
    if len(s) > MAX_ATTRS_JSON:  # drop the optional fields before ever truncating a value
        slim = {n: {k: r[k] for k in ("value", "p", "basis", "reason") if k in r} for n, r in rec.items()}
        s = json.dumps(slim, separators=(",", ":"))
    return s if len(s) <= MAX_ATTRS_JSON else ""   # never store half a JSON document


def run_attribute_pass(table, load_image: Callable[[str], np.ndarray], scorer: Callable[[object, str], float], *,
                       ontology=None, ocr_reader: Optional[Callable[[object], str]] = None, box_threshold: float = 0.35,
                       min_crop_px: int = 48, budget_calls: int = DEFAULT_BUDGET_CALLS) -> dict:
    """Fill ``table.attrs_json`` in place; returns counters plus ``entity_attributes`` (entity_row -> {name: combined})."""
    if ontology is None:
        from perception.ontology import get_default_ontology
        ontology = get_default_ontology()
    stats: Counter = Counter()
    images: Dict[str, np.ndarray] = {}
    per_entity: Dict[int, Dict[str, List[dict]]] = {}
    values_of: Dict[str, List[str]] = {}
    calls = 0
    for i in range(len(table)):
        node = ontology.get(str(table.class_path[i]))
        if node is None or not node.attributes:
            stats["no_attributes"] += 1
            continue
        x0, y0, x1, y1 = (float(v) for v in table.bbox_px[i])
        if min(x1 - x0, y1 - y0) < min_crop_px:
            stats["crop_too_small"] += 1
            continue
        # an attribute is attempted only when at least two of its values have a vetted phrase (one phrase cannot discriminate)
        plan: Dict[str, Dict[str, str]] = {}
        rec: Dict[str, dict] = {}
        for name, vals in node.attributes.items():
            phrases = {v: attribute_phrase(name, v) for v in vals if v != "unknown"}
            phrases = {v: p for v, p in phrases.items() if p}
            if len(phrases) >= 2:
                plan[name] = phrases
            else:
                rec[name] = {"value": None, "p": 0.0, "basis": "none", "obs": [], "reason": "no vetted detector phrase"}
                stats["no_vetted_phrase"] += 1
        need = sum(len(p) for p in plan.values())
        if need and calls + need > budget_calls:
            stats["budget_exhausted"] += 1
            continue
        if need:
            image_id = str(table.image_id[i])
            if image_id not in images:
                images[image_id] = load_image(image_id)
        for name, phrases in plan.items():
            values = list(phrases)
            r = classify_attribute_from_crop(images[str(table.image_id[i])], (x0, y0, x1, y1), values, phrases, scorer,
                                             box_threshold, obs_id=str(table.obs_id[i]), ocr_reader=ocr_reader,
                                             detector_many=getattr(scorer, "many", None))
            calls += len(values)
            rec[name] = r
            values_of[name] = values
            stats["attributes_decided" if r["value"] is not None else "attributes_unknown"] += 1
            if int(table.entity_row[i]) >= 0:
                per_entity.setdefault(int(table.entity_row[i]), {}).setdefault(name, []).append(r)
        table.attrs_json[i] = _pack(rec)
        stats["rows_classified"] += 1
    entity_attributes = {row: {name: combine_views(values_of[name], views, box_threshold) for name, views in attrs.items()}
                         for row, attrs in per_entity.items()}
    return {**stats, "detector_calls": calls, "entity_attributes": entity_attributes}


def grounded_scorer() -> Callable[[object, str], float]:
    """Grounding DINO as ``scorer(crop, phrase) -> best score`` (0.0 when it finds nothing above its box threshold).
    Loads the model on first use (local cache only); the crop goes through a temp PNG because the detector reads files."""
    import tempfile
    from pathlib import Path

    from PIL import Image

    from engine.pipeline.open_vocab_furniture import _grounded_raw

    def many(crop, phrases: Sequence[str]) -> List[float]:
        """Best score per phrase from ONE image write and one detector pass (the detector batches 2 prompts per forward);
        0.0 for a phrase it finds nothing for above its box threshold."""
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "crop.png"
            Image.fromarray(np.ascontiguousarray(crop)).save(p)
            raw, _ = _grounded_raw(p, list(phrases))
        best: Dict[str, float] = {}
        for prompt, s, _ in raw:
            best[prompt] = max(best.get(prompt, 0.0), float(s))
        return [best.get(ph, 0.0) for ph in phrases]

    def score(crop, phrase: str) -> float:
        return many(crop, [phrase])[0]

    score.many = many
    return score
