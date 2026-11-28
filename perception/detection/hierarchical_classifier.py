"""Team A — Hierarchical Taxonomy Classifier for Indoor Perception.

Provides coarse-to-fine taxonomy classification for detected indoor objects.

Responsibilities:
  - Map raw detector labels (often imprecise) to canonical ontology paths.
  - Resolve ambiguous class names using a synonym/alias bank.
  - Apply domain-specific disambiguation rules (architecture vs furniture).
  - Compute confusion-matrix-aware class probabilities when multiple
    candidate labels are present.
  - Provide a prompt bank for each ontology leaf used in open-vocab queries.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("reality.detection.hierarchical_classifier")


# ---------------------------------------------------------------------------
# Canonical taxonomy tree (architecture + furniture, 3-level hierarchy)
# ---------------------------------------------------------------------------
_TAXONOMY: dict[str, dict[str, list[str]]] = {
    "architecture": {
        "vertical_surfaces": [
            "wall", "partition wall", "brick wall", "concrete wall",
            "glass wall", "curtain wall", "interior wall", "exterior wall",
        ],
        "openings": [
            "door", "doorway", "opening", "archway", "passage",
        ],
        "door_components": [
            "door frame", "door panel", "door handle", "door knob",
            "door hinge", "door latch", "door lock",
        ],
        "windows": [
            "window", "window frame", "window pane", "window sill",
            "skylight", "window handle", "glazing",
        ],
        "horizontal_surfaces": [
            "floor", "ceiling", "subfloor", "raised floor",
        ],
        "structural_elements": [
            "column", "pillar", "beam", "structural beam", "joist",
            "post", "lintel", "header",
        ],
        "circulation": [
            "stairs", "staircase", "step", "landing", "ramp",
            "railing", "banister", "balustrade", "handrail",
        ],
        "trim": [
            "baseboard", "skirting board", "crown molding", "cornice",
            "chair rail", "wainscoting", "casing",
        ],
        "mechanical": [
            "vent", "air vent", "hvac vent", "air register", "diffuser",
            "sprinkler head", "smoke detector", "fire alarm",
            "thermostat", "electrical panel", "breaker box",
            "electrical outlet", "power socket", "light switch",
            "data port", "phone jack",
        ],
    },
    "furniture": {
        "seating": [
            "chair", "armchair", "office chair", "dining chair",
            "accent chair", "rocking chair", "folding chair",
            "bench", "stool", "bar stool",
        ],
        "soft_seating": [
            "sofa", "couch", "loveseat", "sectional sofa",
            "ottoman", "chaise lounge", "futon",
        ],
        "tables": [
            "table", "dining table", "coffee table", "side table",
            "end table", "console table", "nightstand", "bedside table",
            "kitchen table", "conference table",
        ],
        "desks": [
            "desk", "writing desk", "standing desk", "computer desk",
            "secretary desk", "reception desk",
        ],
        "beds": [
            "bed", "bunk bed", "twin bed", "full bed", "queen bed",
            "king bed", "cot", "daybed", "platform bed",
        ],
        "bed_components": [
            "headboard", "footboard", "bed frame", "mattress",
            "box spring", "bed slat",
        ],
        "storage": [
            "cabinet", "wardrobe", "closet", "dresser",
            "chest of drawers", "armoire", "hutch",
        ],
        "shelving": [
            "bookshelf", "bookcase", "shelving unit", "storage shelf",
            "floating shelf", "wall shelf", "shelving",
        ],
        "media": [
            "tv stand", "entertainment center", "media console",
            "media cabinet", "av rack",
        ],
        "filing": [
            "filing cabinet", "file cabinet", "lateral file",
        ],
        "appliances": [
            "refrigerator", "fridge", "stove", "range", "oven",
            "microwave", "dishwasher", "washing machine", "dryer",
            "air conditioner", "air purifier", "humidifier", "dehumidifier",
            "water heater",
        ],
        "bathroom": [
            "sink", "bathroom sink", "toilet", "bathtub", "bath",
            "shower", "shower stall", "shower head", "shower door",
            "vanity", "bathroom vanity", "mirror", "towel rack",
            "towel bar", "toilet paper holder",
        ],
        "kitchen": [
            "kitchen sink", "kitchen island", "kitchen cart",
            "range hood", "hood vent", "bar",
        ],
        "lighting": [
            "lamp", "floor lamp", "table lamp", "desk lamp",
            "pendant light", "chandelier", "ceiling fan",
            "light fixture", "sconce", "track light",
        ],
        "soft_furnishings": [
            "rug", "carpet", "mat", "doormat", "curtain",
            "blinds", "drape", "window treatment", "shade",
            "pillow", "cushion", "throw blanket",
        ],
        "decor": [
            "plant", "potted plant", "flower pot", "vase",
            "picture frame", "artwork", "wall art", "painting",
            "sculpture", "clock", "wall clock", "alarm clock",
            "mirror",
        ],
        "electronics": [
            "monitor", "computer monitor", "television", "tv", "screen",
            "keyboard", "computer keyboard", "mouse", "printer",
            "scanner", "computer tower", "laptop", "tablet",
            "phone", "telephone", "speaker", "stereo", "router",
        ],
        "misc": [
            "trash can", "waste bin", "recycling bin",
            "laundry basket", "hamper", "ironing board",
            "vacuum cleaner", "broom", "mop",
            "fire extinguisher", "first aid kit",
            "whiteboard", "corkboard", "bulletin board",
            "projector", "projector screen", "safe", "mailbox",
        ],
    },
}


# ---------------------------------------------------------------------------
# Alias / synonym bank
# ---------------------------------------------------------------------------
_ALIASES: dict[str, str] = {
    # Structural
    "couch": "sofa",
    "lounge": "sofa",
    "settee": "sofa",
    "davenport": "sofa",
    "divan": "sofa",
    "loveseat": "sofa",
    "sectional": "sectional sofa",
    "recliner": "armchair",
    "swivel chair": "office chair",
    "task chair": "office chair",
    "barstool": "bar stool",
    "highchair": "chair",
    "pew": "bench",
    "hassock": "ottoman",
    "pouf": "ottoman",
    "footstool": "ottoman",
    "wardrobe": "wardrobe",
    "closet": "wardrobe",
    "armoire": "wardrobe",
    "chest": "chest of drawers",
    "dresser": "dresser",
    "bureau": "dresser",
    "tv": "television",
    "telly": "television",
    "fridge": "refrigerator",
    "icebox": "refrigerator",
    "range": "stove",
    "cooker": "stove",
    "hob": "stove",
    "lavatory": "toilet",
    "commode": "toilet",
    "john": "toilet",
    "bath": "bathtub",
    "tub": "bathtub",
    "washbasin": "bathroom sink",
    "washstand": "bathroom sink",
    "basin": "bathroom sink",
    "water closet": "toilet",
    "nightstand": "nightstand",
    "night table": "nightstand",
    "credenza": "cabinet",
    "sideboard": "cabinet",
    "buffet": "cabinet",
    "hutch": "cabinet",
    "étagère": "shelving",
    "etagere": "shelving",
    "doorway": "opening",
    "entryway": "opening",
    "passage": "opening",
    "archway": "opening",
    "pillar": "column",
    "post": "column",
    "baluster": "railing",
    "banister": "railing",
    "handrail": "railing",
    "step": "stairs",
    "steps": "stairs",
    "stairway": "stairs",
    "bannister": "railing",
    "baseboard heater": "baseboard",
    "trim": "baseboard",
    "moulding": "crown molding",
    "outlet": "electrical outlet",
    "plug": "electrical outlet",
    "socket": "electrical outlet",
    "switch": "light switch",
    "diffuser": "air vent",
    "register": "air vent",
    "grille": "air vent",
    "sprinkler": "sprinkler head",
    "detector": "smoke detector",
    "extinguisher": "fire extinguisher",
    "monitor": "computer monitor",
    "screen": "computer monitor",
    "display": "computer monitor",
    "pc": "computer tower",
    "desktop": "computer tower",
    "notebook": "laptop",
    "mobile": "phone",
    "cell": "phone",
    "smartphone": "phone",
    "trash": "trash can",
    "bin": "trash can",
    "garbage can": "trash can",
    "waste": "trash can",
    "laundry": "laundry basket",
    "clothes basket": "laundry basket",
    "clothes hamper": "hamper",
    "ironing": "ironing board",
    "vacuum": "vacuum cleaner",
    "hoover": "vacuum cleaner",
}


# ---------------------------------------------------------------------------
# Domain disambiguation rules
# ---------------------------------------------------------------------------
_DISAMBIGUATION_RULES: list[tuple[str, str, str]] = [
    # (ambiguous_term, context_clue, resolved_category)
    ("mirror", "bathroom", "bathroom"),
    ("mirror", "bedroom", "decor"),
    ("sink", "bathroom", "bathroom"),
    ("sink", "kitchen", "kitchen"),
    ("cabinet", "bathroom", "bathroom"),
    ("cabinet", "kitchen", "storage"),
    ("shelf", "bathroom", "bathroom"),
    ("shelf", "kitchen", "shelving"),
    ("light", "ceiling", "lighting"),
    ("light", "table", "lighting"),
    ("mat", "bathroom", "soft_furnishings"),
    ("mat", "entrance", "soft_furnishings"),
    ("screen", "tv", "media"),
    ("screen", "computer", "electronics"),
]


# ---------------------------------------------------------------------------
# Classification result
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ClassificationResult:
    """Result of classifying a raw label against the taxonomy."""
    raw_label: str
    canonical_label: str
    canonical_path: str         # e.g. "furniture/seating/chair"
    domain: str                 # "architecture" | "furniture" | "unknown"
    subcategory: str            # e.g. "seating"
    confidence: float           # 0.0 – 1.0 (1.0 = exact match)
    resolution_method: str      # "exact" | "alias" | "fuzzy" | "unknown"
    alternatives: tuple[str, ...]  # other plausible canonical labels


@dataclass
class ClassificationConfusionCell:
    """Tracks pairwise classification confusion between two labels."""
    label_a: str
    label_b: str
    confusion_rate: float    # empirical P(predict B | true A)
    disambiguation_cue: str  # what resolves the ambiguity


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------
class HierarchicalTaxonomyClassifier:
    """Coarse-to-fine classifier mapping raw labels to canonical ontology paths.

    Resolution order:
      1. Exact match in taxonomy leaf list.
      2. Alias/synonym lookup.
      3. Fuzzy prefix / substring match.
      4. Unknown fallback.
    """

    # Confusion pairs and their disambiguation cues
    _CONFUSION_PAIRS: list[ClassificationConfusionCell] = [
        ClassificationConfusionCell("sofa", "chair", 0.08, "size:large→sofa"),
        ClassificationConfusionCell("cabinet", "bookshelf", 0.12, "open_vs_closed"),
        ClassificationConfusionCell("window", "mirror", 0.06, "reflection_vs_transparency"),
        ClassificationConfusionCell("column", "door frame", 0.10, "structural_vs_opening"),
        ClassificationConfusionCell("desk", "table", 0.14, "work_surface_vs_dining"),
        ClassificationConfusionCell("nightstand", "table", 0.11, "bedroom_context"),
        ClassificationConfusionCell("ottoman", "coffee table", 0.09, "cushion_texture"),
        ClassificationConfusionCell("wardrobe", "cabinet", 0.13, "full_height_vs_low"),
        ClassificationConfusionCell("shelf", "cabinet", 0.10, "open_vs_closed_storage"),
        ClassificationConfusionCell("sofa", "bed", 0.07, "cushion_depth_ratio"),
    ]

    def __init__(self) -> None:
        # Build lookup tables
        self._leaf_to_path: dict[str, str] = {}       # label → canonical path
        self._leaf_to_domain: dict[str, str] = {}
        self._leaf_to_sub: dict[str, str] = {}
        self._path_to_siblings: dict[str, list[str]] = {}

        for domain, subcats in _TAXONOMY.items():
            for subcat, leaves in subcats.items():
                path_prefix = f"{domain}/{subcat}"
                for leaf in leaves:
                    key = leaf.lower()
                    self._leaf_to_path[key] = f"{path_prefix}/{leaf}"
                    self._leaf_to_domain[key] = domain
                    self._leaf_to_sub[key] = subcat
                # Build sibling list
                for leaf in leaves:
                    self._path_to_siblings[f"{path_prefix}/{leaf}"] = leaves

        # Alias resolution
        self._alias_map: dict[str, str] = {
            k.lower(): v.lower() for k, v in _ALIASES.items()
        }
        self._confusion_map: dict[frozenset[str], ClassificationConfusionCell] = {
            frozenset([c.label_a, c.label_b]): c for c in self._CONFUSION_PAIRS
        }

    def classify(
        self,
        raw_label: str,
        context_tokens: tuple[str, ...] = (),
        *,
        confidence_floor: float = 0.0,
    ) -> ClassificationResult:
        """Classify a raw label string.

        Args:
            raw_label: Detector output label (may be noisy/informal).
            context_tokens: Optional surrounding text tokens for disambiguation.
            confidence_floor: Minimum confidence; results below this yield ``unknown``.

        Returns:
            ``ClassificationResult`` with canonical label and path.
        """
        raw_normalized = raw_label.strip().lower()

        # 1. Exact match
        if raw_normalized in self._leaf_to_path:
            canon = raw_normalized
            path = self._leaf_to_path[canon]
            conf = 1.0
            method = "exact"
        elif raw_normalized in self._alias_map:
            # 2. Alias lookup
            canon = self._alias_map[raw_normalized]
            if canon not in self._leaf_to_path:
                canon = self._fallback_fuzzy(canon) or ""
            if canon:
                path = self._leaf_to_path.get(canon, f"unknown/{canon}")
                conf = 0.90
                method = "alias"
            else:
                return self._unknown(raw_label)
        else:
            # 3. Fuzzy match
            canon = self._fallback_fuzzy(raw_normalized) or ""
            if canon:
                path = self._leaf_to_path[canon]
                conf = 0.70
                method = "fuzzy"
            else:
                return self._unknown(raw_label)

        if conf < confidence_floor:
            return self._unknown(raw_label)

        domain = self._leaf_to_domain.get(canon, "unknown")
        subcat = self._leaf_to_sub.get(canon, "unknown")

        # Apply context-based disambiguation
        path, canon, domain, subcat = self._disambiguate(
            canon, path, domain, subcat, context_tokens
        )

        # Build alternatives from siblings
        siblings = self._path_to_siblings.get(path, [])
        alternatives = tuple(s for s in siblings if s.lower() != canon)[:4]

        return ClassificationResult(
            raw_label=raw_label,
            canonical_label=canon,
            canonical_path=path,
            domain=domain,
            subcategory=subcat,
            confidence=conf,
            resolution_method=method,
            alternatives=alternatives,
        )

    def _fallback_fuzzy(self, label: str) -> str | None:
        """Substring + prefix fuzzy match against taxonomy leaves."""
        # Try prefix match
        for key in self._leaf_to_path:
            if key.startswith(label) or label.startswith(key):
                return key
        # Try any-word overlap (tokenize by space)
        label_words = set(label.split())
        best_key: str | None = None
        best_overlap = 0
        for key in self._leaf_to_path:
            key_words = set(key.split())
            overlap = len(label_words & key_words)
            if overlap > best_overlap:
                best_overlap = overlap
                best_key = key
        if best_overlap > 0:
            return best_key
        return None

    def _disambiguate(
        self,
        canon: str,
        path: str,
        domain: str,
        subcat: str,
        context_tokens: tuple[str, ...],
    ) -> tuple[str, str, str, str]:
        """Apply context-based disambiguation for ambiguous labels."""
        if not context_tokens:
            return path, canon, domain, subcat
        ctx_lower = {t.lower() for t in context_tokens}
        for amb_term, ctx_clue, resolved_subcat in _DISAMBIGUATION_RULES:
            if canon == amb_term and ctx_clue in ctx_lower:
                # Try to remap to resolved subcategory
                for key, sub in self._leaf_to_sub.items():
                    if sub == resolved_subcat and amb_term in key:
                        new_path = self._leaf_to_path[key]
                        return new_path, key, self._leaf_to_domain[key], sub
        return path, canon, domain, subcat

    @staticmethod
    def _unknown(raw_label: str) -> ClassificationResult:
        return ClassificationResult(
            raw_label=raw_label,
            canonical_label="unknown",
            canonical_path="unknown/unknown",
            domain="unknown",
            subcategory="unknown",
            confidence=0.0,
            resolution_method="unknown",
            alternatives=(),
        )

    def classify_batch(
        self,
        labels: list[str],
        context_tokens: tuple[str, ...] = (),
        *,
        confidence_floor: float = 0.0,
    ) -> list[ClassificationResult]:
        """Classify a batch of labels."""
        return [
            self.classify(label, context_tokens, confidence_floor=confidence_floor)
            for label in labels
        ]

    def get_prompts_for_subcategory(self, domain: str, subcat: str) -> list[str]:
        """Return all leaf labels for a given (domain, subcategory) pair."""
        return list(_TAXONOMY.get(domain, {}).get(subcat, []))

    def get_all_prompts(self) -> list[str]:
        """Return every leaf label across all domains."""
        result: list[str] = []
        for subcats in _TAXONOMY.values():
            for leaves in subcats.values():
                result.extend(leaves)
        return result

    def confusion_risk(self, label_a: str, label_b: str) -> float:
        """Return empirical confusion rate between two labels (0 if none)."""
        cell = self._confusion_map.get(frozenset([label_a.lower(), label_b.lower()]))
        return cell.confusion_rate if cell else 0.0

    def get_disambiguation_cue(self, label_a: str, label_b: str) -> str | None:
        """Return the disambiguation cue for a confused pair."""
        cell = self._confusion_map.get(frozenset([label_a.lower(), label_b.lower()]))
        return cell.disambiguation_cue if cell else None
