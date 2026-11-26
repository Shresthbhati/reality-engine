"""Open-Vocabulary & Hierarchical Recognition Engine (A4.8).

Expands recognition beyond fixed furniture lists:
  - Unknown-object recognition (preserving raw open-vocab phrase when unmapped)
  - Class alternatives & confusion tracking
  - Domain-specific prompt generation (interior, exterior, commercial, residential)
  - Taxonomy normalization & fine-grained attribute extraction
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from engine.multiengine.spatial_intelligence.taxonomy import (
    Domain,
    get_all_taxonomy_classes,
    resolve_taxonomy_class,
)


@dataclass(frozen=True)
class OpenVocabClassification:
    raw_phrase: str
    canonical_label: str
    taxonomy_id: str
    is_known_class: bool
    alternatives: tuple[str, ...]
    confusion_classes: tuple[str, ...]
    hierarchy_path: tuple[str, ...]


def classify_open_vocabulary_label(raw_phrase: str) -> OpenVocabClassification:
    """Normalize an open-vocabulary phrase while preserving unknown objects and confusion alternatives."""
    clean = str(raw_phrase or "").strip().lower()
    tax = resolve_taxonomy_class(clean)
    is_known = tax.name != "unknown_object" and bool(clean)
    canon = tax.name if is_known else (clean.replace(" ", "_") or "unknown_object")
    parts = tuple(p.title() for p in tax.taxonomy_id.split(".")) if is_known else ("Unknown", canon.title())
    return OpenVocabClassification(
        raw_phrase=raw_phrase,
        canonical_label=canon,
        taxonomy_id=tax.taxonomy_id,
        is_known_class=is_known,
        alternatives=tax.aliases[:4],
        confusion_classes=tax.common_confusion_classes,
        hierarchy_path=parts,
    )


def generate_survey_prompts(
    *,
    domain: Domain = Domain.INTERIOR,
    include_unknown_candidates: bool = True,
    max_prompts: int = 80,
) -> list[str]:
    """Generate clean, non-redundant open-vocabulary prompts for a survey domain."""
    prompts: list[str] = []
    seen: set[str] = set()
    for tc in get_all_taxonomy_classes():
        if tc.domain not in (domain, Domain.UNIVERSAL):
            continue
        for phrase in (tc.name.replace("_", " "), *(a for a in tc.aliases[:1])):
            p = phrase.strip().lower()
            if p and p not in seen:
                seen.add(p)
                prompts.append(p)
            if len(prompts) >= max_prompts:
                return prompts
    if include_unknown_candidates and len(prompts) < max_prompts:
        for extra in ("storage box", "radiator", "mirror", "curtain", "artwork"):
            if extra not in seen:
                prompts.append(extra)
    return prompts[:max_prompts]


def track_class_confusions(
    competing_pairs: Sequence[tuple[str, str]],
) -> dict[str, Any]:
    """Summarize recurring class confusion pairs across detections."""
    counts: Counter[tuple[str, str]] = Counter()
    for a, b in competing_pairs:
        pair = tuple(sorted((a, b)))
        counts[(pair[0], pair[1])] += 1
    return {
        "total_confusion_events": sum(counts.values()),
        "top_confusions": [
            {"pair": [k[0], k[1]], "count": v}
            for k, v in counts.most_common(10)
        ],
    }


__all__ = [
    "OpenVocabClassification",
    "classify_open_vocabulary_label",
    "generate_survey_prompts",
    "track_class_confusions",
]
