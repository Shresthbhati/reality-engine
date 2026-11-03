"""Grounded Reality Copilot and Tool Layer (Phase F).

Provides deterministic, zero-hallucination spatial question answering, query translation,
and verified WorldIR grounding.
"""

from .copilot_engine import CopilotMeasurement, CopilotResponse, GroundedFact, RealityCopilotEngine
from .query_translator import ParsedQuery, QueryIntent, QueryTranslator

__all__ = [
    "RealityCopilotEngine",
    "CopilotResponse",
    "CopilotMeasurement",
    "GroundedFact",
    "QueryTranslator",
    "ParsedQuery",
    "QueryIntent",
]
