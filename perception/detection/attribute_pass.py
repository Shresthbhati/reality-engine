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
from typing import Callable, Dict, List, Optional

import numpy as np

from perception.detection.attributes import classify_attribute_from_crop, combine_views

MAX_ATTRS_JSON = 2000      # ObservationTable.attrs_json is a <U2000 column
DEFAULT_BUDGET_CALLS = 200  # detector calls per run: one per allowed value per observation


def attribute_phrase(node, value: str) -> str:
    """The detector phrase for one allowed value: "numpad lock". Weak by design (the margin rule keeps ambiguous
    results unknown); a node-specific phrase table can replace it without touching callers."""
    return f"{value.replace('_', ' ')} {str(getattr(node, 'label', '') or node.name).lower()}"


def _pack(rec: Dict[str, dict]) -> str:
    slim = {n: {k: r[k] for k in ("value", "p", "basis", "obs", "conflict", "ocr_text") if k in r} for n, r in rec.items()}
    s = json.dumps(slim, separators=(",", ":"))
    if len(s) > MAX_ATTRS_JSON:  # drop the optional fields before ever truncating a value
        slim = {n: {k: r[k] for k in ("value", "p", "basis") if k in r} for n, r in rec.items()}
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
        need = sum(len([v for v in vals if v != "unknown"]) for vals in node.attributes.values())
        if calls + need > budget_calls:
            stats["budget_exhausted"] += 1
            continue
        image_id = str(table.image_id[i])
        if image_id not in images:
            images[image_id] = load_image(image_id)
        rec: Dict[str, dict] = {}
        for name, vals in node.attributes.items():
            values = [v for v in vals if v != "unknown"]
            if not values:
                continue
            r = classify_attribute_from_crop(images[image_id], (x0, y0, x1, y1), values,
                                             {v: attribute_phrase(node, v) for v in values}, scorer, box_threshold,
                                             obs_id=str(table.obs_id[i]), ocr_reader=ocr_reader)
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

    def score(crop, phrase: str) -> float:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "crop.png"
            Image.fromarray(np.ascontiguousarray(crop)).save(p)
            raw, _ = _grounded_raw(p, [phrase])
        return max((float(s) for _, s, _ in raw), default=0.0)

    return score
