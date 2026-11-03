"""Deterministic Natural Language Query Translator for Grounded Reality Copilot.

Spec: Universal Intelligence Expansion (Epic E11-1).
Parses natural language spatial questions and structured query intents into unambiguous,
deterministic SpatialQueryPlans without requiring external LLM services.

Honesty Guarantee: Zero fabricated entities, relations, or dimensions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from world_ir.schema_v1 import EntityType, RelationshipKind


class QueryIntent(str, Enum):
    PROXIMITY = "proximity"
    CONTAINMENT = "containment"
    ATTACHMENT = "attachment"
    SUPPORT = "support"
    CONNECTIVITY = "connectivity"
    MEASUREMENT_DISTANCE = "measurement_distance"
    MEASUREMENT_DIMENSION = "measurement_dimension"
    MEASUREMENT_AREA = "measurement_area"
    CLEARANCE_FIT = "clearance_fit"
    COUNT = "count"
    FIND = "find"
    PROVENANCE = "provenance"
    CONFIDENCE = "confidence"
    INSPECTION = "inspection"
    UNKNOWN = "unknown"


# Mapping from natural language words to entity types / taxonomy hints
SYNONYM_TYPE_MAP: Dict[str, str] = {
    # Openings
    "door": "door",
    "doors": "door",
    "doorway": "door",
    "window": "window",
    "windows": "window",
    "skylight": "window",
    # Envelope
    "wall": "wall",
    "walls": "wall",
    "floor": "floor",
    "floors": "floor",
    "ceiling": "ceiling",
    "ceilings": "ceiling",
    # Rooms
    "room": "room",
    "rooms": "room",
    "kitchen": "room",
    "bathroom": "room",
    "bedroom": "room",
    "living room": "room",
    "corridor": "corridor",
    "hallway": "corridor",
    # Electrical
    "socket": "electrical.socket",
    "sockets": "electrical.socket",
    "outlet": "electrical.socket",
    "outlets": "electrical.socket",
    "plug": "electrical.socket",
    "plugs": "electrical.socket",
    "switch": "electrical.switch",
    "switches": "electrical.switch",
    "switchboard": "electrical.switchboard",
    "light": "electrical.light",
    "lights": "electrical.light",
    "lamp": "electrical.light",
    # Plumbing
    "sink": "plumbing.sink",
    "sinks": "plumbing.sink",
    "basin": "plumbing.sink",
    "tap": "plumbing.tap",
    "faucet": "plumbing.tap",
    "toilet": "plumbing.toilet",
    "shower": "plumbing.shower",
    # Furniture
    "table": "furniture.table",
    "tables": "furniture.table",
    "desk": "furniture.table",
    "chair": "furniture.chair",
    "chairs": "furniture.chair",
    "bed": "furniture.bed",
    "beds": "furniture.bed",
    "sofa": "furniture.sofa",
    "couch": "furniture.sofa",
    "wardrobe": "furniture.cabinet",
    "cabinet": "furniture.cabinet",
    "shelf": "furniture.shelf",
    # Structural
    "column": "structure",
    "pillar": "structure",
    "beam": "structure",
    "stair": "stairs",
    "stairs": "stairs",
    "staircase": "stairs",
}


@dataclass
class ParsedQuery:
    """Structured intermediate representation of a translated query."""
    raw_query: str
    intent: QueryIntent
    target_type: Optional[str] = None          # EntityType string or taxonomy class
    target_name: Optional[str] = None          # Name or ID of target entity
    reference_name: Optional[str] = None       # Primary reference entity (e.g. "sink" in "outlets near sink")
    second_reference_name: Optional[str] = None# Secondary reference entity (e.g. for distance or clearance between A and B)
    spatial_relation: Optional[RelationshipKind] = None
    distance_threshold_m: float = 2.0          # Metric radius / threshold, default 2.0m for "near"
    fit_dimension_m: Optional[float] = None    # Metric dimension for clearance/fit checks
    dimension_property: Optional[str] = None   # "height", "width", "depth", "area"
    confidence_min: Optional[float] = None
    provenance_filter: Optional[str] = None
    raw_tokens: List[str] = field(default_factory=list)
    confidence: float = 1.0                    # Parser confidence in translation

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_query": self.raw_query,
            "intent": self.intent.value,
            "target_type": self.target_type,
            "target_name": self.target_name,
            "reference_name": self.reference_name,
            "second_reference_name": self.second_reference_name,
            "spatial_relation": self.spatial_relation.value if self.spatial_relation else None,
            "distance_threshold_m": self.distance_threshold_m,
            "fit_dimension_m": self.fit_dimension_m,
            "dimension_property": self.dimension_property,
            "confidence_min": self.confidence_min,
            "provenance_filter": self.provenance_filter,
            "confidence": self.confidence,
        }


class QueryTranslator:
    """Translates user natural language or structured queries into ParsedQuery plans."""

    _DIST_PATTERN = re.compile(
        r"(?:within|less than|radius of|under|at most|up to|around)?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:m|meter|meters|metre|metres)\b",
        re.IGNORECASE,
    )
    _CM_PATTERN = re.compile(
        r"(?:within|less than|radius of|under|at most|up to|around)?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:cm|centimeter|centimeters)\b",
        re.IGNORECASE,
    )
    _FIT_PATTERN = re.compile(
        r"(?:can|could|will)\s+(?:a\s+)?([0-9]+(?:\.[0-9]+)?)\s*(?:m|meter|meters|metre|metres)?\s*(?:wide|long|high)?\s*([a-zA-Z0-9_\-]+)?\s+fit\s+(?:in|between)\s+([a-zA-Z0-9_\-\s]+?)\s+and\s+([a-zA-Z0-9_\-\s\?]+)",
        re.IGNORECASE,
    )
    _BETWEEN_PATTERN = re.compile(
        r"(?:distance|clearance|space)\s+(?:between|from)\s+([a-zA-Z0-9_\-\s]+?)\s+(?:and|to)\s+([a-zA-Z0-9_\-\s\?]+)",
        re.IGNORECASE,
    )

    def translate(self, query: Union[str, Dict[str, Any]]) -> ParsedQuery:
        """Parse natural language query or dictionary into a ParsedQuery."""
        if isinstance(query, dict):
            return self._from_dict(query)
        if not isinstance(query, str) or not query.strip():
            return ParsedQuery(
                raw_query="",
                intent=QueryIntent.UNKNOWN,
                confidence=0.0,
            )

        q = query.strip()
        q_lower = q.lower()
        tokens = [t.lower() for t in re.findall(r"[a-zA-Z0-9_\.\-]+", q)]

        # 1. Check Clearance / Fit query
        fit_match = self._FIT_PATTERN.search(q)
        if fit_match:
            size_val = float(fit_match.group(1))
            item_desc = (fit_match.group(2) or "").strip()
            ref1 = fit_match.group(3).strip().rstrip("?")
            ref2 = fit_match.group(4).strip().rstrip("?")
            return ParsedQuery(
                raw_query=q,
                intent=QueryIntent.CLEARANCE_FIT,
                target_name=item_desc or None,
                reference_name=self._clean_entity_ref(ref1),
                second_reference_name=self._clean_entity_ref(ref2),
                fit_dimension_m=size_val,
                raw_tokens=tokens,
            )

        # 2. Check Distance between A and B
        dist_match = self._BETWEEN_PATTERN.search(q)
        if dist_match:
            ref1 = dist_match.group(1).strip()
            ref2 = dist_match.group(2).strip().rstrip("?")
            return ParsedQuery(
                raw_query=q,
                intent=QueryIntent.MEASUREMENT_DISTANCE,
                reference_name=self._clean_entity_ref(ref1),
                second_reference_name=self._clean_entity_ref(ref2),
                raw_tokens=tokens,
            )

        # 3. Check Provenance & Confidence before general queries
        if any(kw in q_lower for kw in ("provenance", "origin", "where did", "where came from", "where does")):
            target = self._find_entity_mention(q) or self._extract_subject_after(tokens, ["of", "for", "did"])
            return ParsedQuery(
                raw_query=q,
                intent=QueryIntent.PROVENANCE,
                target_name=target,
                raw_tokens=tokens,
            )
        if any(kw in q_lower for kw in ("confidence", "uncertainty", "how confident", "how sure")):
            target = self._find_entity_mention(q) or self._extract_subject_after(tokens, ["of", "for", "about", "in"])
            return ParsedQuery(
                raw_query=q,
                intent=QueryIntent.CONFIDENCE,
                target_name=target,
                raw_tokens=tokens,
            )

        # 4. Check Dimension / Area measurement queries
        if any(kw in q_lower for kw in ("height", "tall", "how tall")):
            target = self._find_entity_mention(q) or self._extract_subject_after(tokens, ["of", "for"])
            return ParsedQuery(
                raw_query=q,
                intent=QueryIntent.MEASUREMENT_DIMENSION,
                dimension_property="height",
                target_name=target,
                raw_tokens=tokens,
            )
        if any(kw in q_lower for kw in ("area", "sqm", "m2", "square meters", "size")):
            target = self._find_entity_mention(q) or self._extract_subject_after(tokens, ["of", "for", "in"])
            return ParsedQuery(
                raw_query=q,
                intent=QueryIntent.MEASUREMENT_AREA,
                dimension_property="area",
                target_name=target,
                raw_tokens=tokens,
            )

        # 5. Check Spatial Proximity: "near", "close to", "within X meters of"
        if any(w in tokens for w in ("near", "nearby", "close", "within", "around")):
            dist = self._extract_distance(q) or 2.0
            target_type, ref_name = self._parse_proximity_targets(q, tokens)
            return ParsedQuery(
                raw_query=q,
                intent=QueryIntent.PROXIMITY,
                target_type=target_type,
                reference_name=ref_name,
                distance_threshold_m=dist,
                spatial_relation=RelationshipKind.ADJACENT_TO,
                raw_tokens=tokens,
            )

        # 6. Check Containment / Rooms
        if any(w in tokens for w in ("inside", "contents of", "contained in")) or ("in" in tokens and any(r in tokens for r in ("room", "bedroom", "kitchen", "bathroom", "living", "corridor", "hallway"))):
            container = self._find_entity_mention(q) or self._extract_subject_after(tokens, ["inside", "in", "of"])
            target = self._extract_subject_before(tokens, ["inside", "in", "contents"])
            return ParsedQuery(
                raw_query=q,
                intent=QueryIntent.CONTAINMENT,
                reference_name=container,
                target_type=self._resolve_type(target) if target else None,
                spatial_relation=RelationshipKind.CONTAINS,
                raw_tokens=tokens,
            )

        # 7. Check Attachment: "on wall", "mounted on", "attached to"
        if any(kw in q_lower for kw in ("mounted on", "attached to", "on wall", "on the wall")):
            ref = self._find_entity_mention(q) or self._extract_subject_after(tokens, ["on", "to"])
            target = self._extract_subject_before(tokens, ["on", "mounted", "attached"])
            return ParsedQuery(
                raw_query=q,
                intent=QueryIntent.ATTACHMENT,
                reference_name=ref,
                target_type=self._resolve_type(target) if target else None,
                spatial_relation=RelationshipKind.ATTACHED_TO,
                raw_tokens=tokens,
            )

        # 8. Support: "supported by", "resting on", "holding up"
        if any(kw in q_lower for kw in ("supported by", "rests on", "resting on", "holding up", "supports")):
            ref = self._find_entity_mention(q) or self._extract_subject_after(tokens, ["by", "on", "up"])
            return ParsedQuery(
                raw_query=q,
                intent=QueryIntent.SUPPORT,
                reference_name=ref,
                spatial_relation=RelationshipKind.SUPPORTS,
                raw_tokens=tokens,
            )

        # 9. Connectivity: "connects", "leading to"
        if any(w in tokens for w in ("connect", "connects", "connecting", "leads to")):
            target = self._find_entity_mention(q)
            return ParsedQuery(
                raw_query=q,
                intent=QueryIntent.CONNECTIVITY,
                target_name=target,
                spatial_relation=RelationshipKind.CONNECTS,
                raw_tokens=tokens,
            )

        # 10. Counting: "how many"
        if "how" in tokens and "many" in tokens:
            target_word = self._extract_subject_after(tokens, ["many"])
            return ParsedQuery(
                raw_query=q,
                intent=QueryIntent.COUNT,
                target_type=self._resolve_type(target_word) if target_word else target_word,
                raw_tokens=tokens,
            )

        # 11. Finding / Listing: "find", "list", "where is", "show"
        if any(w in tokens for w in ("find", "list", "where", "show", "get")):
            target_word = self._extract_subject_after(tokens, ["find", "list", "where", "show", "get", "all"])
            resolved_type = self._resolve_type(target_word) if target_word else None
            return ParsedQuery(
                raw_query=q,
                intent=QueryIntent.FIND,
                target_type=resolved_type,
                target_name=None if resolved_type else target_word,
                raw_tokens=tokens,
            )

        # Default fallback
        target = self._find_entity_mention(q)
        return ParsedQuery(
            raw_query=q,
            intent=QueryIntent.INSPECTION if target else QueryIntent.FIND,
            target_name=target,
            target_type=self._resolve_type(q),
            raw_tokens=tokens,
            confidence=0.7,
        )

    def _from_dict(self, d: Dict[str, Any]) -> ParsedQuery:
        """Parse structured query specification directly."""
        intent_raw = d.get("intent", "find").lower()
        try:
            intent = QueryIntent(intent_raw)
        except ValueError:
            intent = QueryIntent.UNKNOWN

        rel_raw = d.get("spatial_relation")
        rel = None
        if rel_raw:
            try:
                rel = RelationshipKind(rel_raw.lower())
            except ValueError:
                rel = None

        return ParsedQuery(
            raw_query=d.get("raw_query", str(d)),
            intent=intent,
            target_type=d.get("target_type") or self._resolve_type(d.get("target", "")),
            target_name=d.get("target_name") or d.get("target"),
            reference_name=d.get("reference_name") or d.get("reference"),
            second_reference_name=d.get("second_reference_name") or d.get("second_reference"),
            spatial_relation=rel,
            distance_threshold_m=float(d.get("distance_threshold_m", d.get("radius_m", 2.0))),
            fit_dimension_m=float(d["fit_dimension_m"]) if "fit_dimension_m" in d else None,
            dimension_property=d.get("dimension_property"),
            confidence_min=float(d["confidence_min"]) if "confidence_min" in d else None,
            provenance_filter=d.get("provenance_filter"),
            confidence=float(d.get("confidence", 1.0)),
        )

    # -------------------------------------------------------- Helper methods

    def _extract_distance(self, text: str) -> Optional[float]:
        m = self._DIST_PATTERN.search(text)
        if m:
            return float(m.group(1))
        cm_m = self._CM_PATTERN.search(text)
        if cm_m:
            return float(cm_m.group(1)) / 100.0
        return None

    def _clean_entity_ref(self, ref: str) -> str:
        words_to_strip = {"the", "a", "an", "all", "any", "this", "that"}
        parts = [p for p in ref.strip().split() if p.lower() not in words_to_strip]
        return " ".join(parts) if parts else ref.strip()

    def _resolve_type(self, token_or_phrase: Optional[str]) -> Optional[str]:
        if not token_or_phrase:
            return None
        cleaned = token_or_phrase.strip().lower()
        if cleaned in SYNONYM_TYPE_MAP:
            return SYNONYM_TYPE_MAP[cleaned]
        for k, v in SYNONYM_TYPE_MAP.items():
            if k in cleaned or cleaned in k:
                return v
        return None

    def _extract_subject_after(self, tokens: List[str], markers: List[str]) -> Optional[str]:
        for m in markers:
            for i, t in enumerate(tokens):
                if t == m and i + 1 < len(tokens):
                    rest = tokens[i + 1:]
                    while rest and rest[0] in ("the", "a", "an", "all", "any", "of", "did"):
                        rest = rest[1:]
                    clean_rest = [w.rstrip("?").rstrip(".") for w in rest]
                    if clean_rest:
                        return " ".join(clean_rest)
        return None

    def _extract_subject_before(self, tokens: List[str], markers: List[str]) -> Optional[str]:
        for m in markers:
            for i, t in enumerate(tokens):
                if t == m and i > 0:
                    prev = tokens[:i]
                    filtered = [w for w in prev if w not in ("find", "what", "is", "are", "all", "the", "a", "any", "items")]
                    if filtered:
                        return " ".join(filtered)
        return None

    def _find_entity_mention(self, text: str) -> Optional[str]:
        quoted = re.findall(r"['\"]([^'\"]+)['\"]", text)
        if quoted:
            return quoted[0]
        m = re.search(r"\b([a-zA-Z0-9]+-[a-zA-Z0-9_\-]+)\b", text)
        if m:
            return m.group(1)
        return None

    def _parse_proximity_targets(self, query: str, tokens: List[str]) -> Tuple[Optional[str], Optional[str]]:
        q_lower = query.lower()
        for kw in ("near", "close to", "within", "around"):
            if kw in q_lower:
                parts = q_lower.split(kw, 1)
                left, right = parts[0], parts[1]
                target_words = [w for w in re.findall(r"[a-zA-Z0-9_\-]+", left)
                                if w not in ("find", "all", "get", "show", "what", "are", "the", "any", "is")]
                target_str = " ".join(target_words) if target_words else None
                resolved_target = self._resolve_type(target_str) or target_str

                right_cleaned = re.sub(r"(?:within|radius|of)?\s*[0-9]+(?:\.[0-9]+)?\s*(?:m|meters|cm)\b", "", right)
                right_words = [w for w in re.findall(r"[a-zA-Z0-9_\-]+", right_cleaned)
                               if w not in ("of", "the", "a", "an", "and", "or", "to")]
                ref_str = " ".join(right_words) if right_words else None
                return resolved_target, ref_str

        return None, None
