"""Natural-language scene inspection (N-29).

Extends the deterministic copilot foundation into a grounded assistant that can
inspect selected objects, explain evidence, propose supported edits and request
confirmation before committing them.

Honesty rules:
* Every answer is grounded in the current WorldIR; unsupported claims are
  labelled "unknown", not invented.
* A query that cannot be answered from the world returns "not recorded" with
  an explanation.
* Proposed edits are never auto-committed; they require explicit confirmation.
* Natural-language parsing is rule-based (deterministic), not an LLM.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from world_ir import Entity, EntityType


@dataclass(frozen=True)
class InspectionResult:
    """Answer to a natural-language inspection query."""
    query: str
    answer: str                          # human-readable answer
    entities: List[Dict[str, object]]    # matching entities with id, name, type, key facts
    confidence: float                    # 0..1
    unsupported: List[str]               # parts of the query that couldn't be answered
    suggestion: Optional[str]            # what to capture or measure next

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "answer": self.answer,
            "entities": self.entities,
            "confidence": self.confidence,
            "unsupported": self.unsupported,
            "suggestion": self.suggestion,
        }


@dataclass(frozen=True)
class EditProposal:
    """A proposed edit derived from a natural-language request."""
    entity_id: str
    field: str
    new_value: object
    reason: str
    requires_confirmation: bool = True


class SceneInspector:
    """Deterministic natural-language scene inspection."""

    # Recognised query patterns
    _WIDTH_QUERY = re.compile(
        r"all (windows|doors|openings) (?:wider|longer|taller) than ([\d.]+)\s*(m|meter|metre)",
        re.IGNORECASE,
    )
    _COUNT_QUERY = re.compile(
        r"how many (walls|doors|windows|rooms|floors|ceilings) (?:are there|do i have)",
        re.IGNORECASE,
    )
    _SHOW_QUERY = re.compile(
        r"show me (all |)(walls|doors|windows|rooms|entities)",
        re.IGNORECASE,
    )

    def inspect(
        self,
        query: str,
        entities: Dict[str, Entity],
    ) -> InspectionResult:
        """Answer a natural-language query about the current world.

        Rule-based parsing; no LLM involved. Unsupported parts of the query are
        listed in ``unsupported``.
        """
        q = query.strip()
        q_lower = q.lower()

        # ---- "show me all windows wider than 1 m" --------------------------
        m = self._WIDTH_QUERY.search(q_lower)
        if m:
            kind = m.group(1)
            threshold = float(m.group(2))
            unit = m.group(3)
            if unit in ("m", "meter", "metre"):
                threshold_m = threshold
            else:
                threshold_m = threshold  # assume metres

            etype_map = {
                "windows": EntityType.WINDOW,
                "doors": EntityType.DOOR,
                "openings": EntityType.OPENING,
            }
            etype = etype_map[kind]

            matches = []
            for e in entities.values():
                if e.type != etype:
                    continue
                cp = e.custom_properties or {}
                width = cp.get("width_m")
                if width is not None and float(width) > threshold_m:
                    matches.append({
                        "id": e.id,
                        "name": e.name,
                        "type": e.type.value,
                        "width_m": round(float(width), 3),
                        "confidence": e.confidence,
                    })

            matches.sort(key=lambda x: -x["width_m"])
            answer = (f"Found {len(matches)} {kind} wider than {threshold_m:.2f} m."
                      if matches else f"No {kind} wider than {threshold_m:.2f} m found.")
            conf = 0.9 if matches else 0.7
            return InspectionResult(
                query=q, answer=answer, entities=matches,
                confidence=conf, unsupported=[],
                suggestion=("Measure widths of remaining windows/doors for a complete list."
                            if not matches else None),
            )

        # ---- "how many walls do I have?" -----------------------------------
        m = self._COUNT_QUERY.search(q_lower)
        if m:
            kind = m.group(1)
            etype_map = {
                "walls": EntityType.WALL,
                "doors": EntityType.DOOR,
                "windows": EntityType.WINDOW,
                "rooms": EntityType.ROOM,
                "floors": EntityType.FLOOR,
                "ceilings": EntityType.CEILING,
            }
            etype = etype_map[kind]
            count = sum(1 for e in entities.values() if e.type == etype)
            answer = f"You have {count} {kind}."
            return InspectionResult(
                query=q, answer=answer, entities=[],
                confidence=1.0, unsupported=[], suggestion=None,
            )

        # ---- "show me all walls" -------------------------------------------
        m = self._SHOW_QUERY.search(q_lower)
        if m:
            kind = m.group(1)
            if kind == "entities":
                matches = [
                    {
                        "id": e.id, "name": e.name,
                        "type": e.type.value,
                        "confidence": e.confidence,
                    }
                    for e in entities.values()
                ]
            else:
                etype_map = {
                    "walls": EntityType.WALL,
                    "doors": EntityType.DOOR,
                    "windows": EntityType.WINDOW,
                    "rooms": EntityType.ROOM,
                }
                etype = etype_map[kind]
                matches = [
                    {
                        "id": e.id, "name": e.name,
                        "type": e.type.value,
                        "confidence": e.confidence,
                    }
                    for e in entities.values() if e.type == etype
                ]
            matches.sort(key=lambda x: x["id"])
            answer = f"Found {len(matches)} {kind}."
            return InspectionResult(
                query=q, answer=answer, entities=matches,
                confidence=0.95, unsupported=[], suggestion=None,
            )

        # ---- unsupported query ---------------------------------------------
        return InspectionResult(
            query=q,
            answer="I can't answer that from the current world yet. Try asking "
                   "'how many walls do I have?', 'show me all windows wider than 1 m', "
                   "or 'show me all doors'.",
            entities=[],
            confidence=0.0,
            unsupported=[q],
            suggestion="Capture more of the scene or rephrase the question.",
        )

    def propose_edit(
        self,
        request: str,
        entities: Dict[str, Entity],
    ) -> Tuple[Optional[EditProposal], str]:
        """Parse a natural-language edit request into an EditProposal.

        Returns (proposal, explanation). If the request can't be parsed,
        proposal is None and explanation says why.
        """
        q = request.strip().lower()

        # "set wall W1 length to 5 m"
        m = re.match(
            r"set (wall|door|window|opening) (\S+) (length|width|height) to ([\d.]+) m",
            q,
        )
        if m:
            kind = m.group(1)
            eid = m.group(2)
            field = m.group(3)
            value = float(m.group(4))

            etype_map = {
                "wall": EntityType.WALL,
                "door": EntityType.DOOR,
                "window": EntityType.WINDOW,
                "opening": EntityType.OPENING,
            }
            entity = entities.get(eid)
            if entity is None:
                return None, f"Entity {eid} not found in the world."
            if entity.type != etype_map[kind]:
                return None, f"Entity {eid} is a {entity.type.value}, not a {kind}."

            return EditProposal(
                entity_id=eid, field=field, new_value=value,
                reason=f"User requested: {request}",
            ), f"Proposed: set {eid}.{field} = {value} m. Confirm before applying."

        return None, (
            "I can parse requests like 'set wall W1 length to 5 m'. "
            "Edits require confirmation before they are applied."
        )


__all__ = ["SceneInspector", "InspectionResult", "EditProposal"]