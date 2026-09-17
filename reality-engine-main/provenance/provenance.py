from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Provenance(str, Enum):
    """spec sec 1.1: every property is one of these."""

    OBSERVED = "OBSERVED"
    RECONSTRUCTED = "RECONSTRUCTED"
    ESTIMATED = "ESTIMATED"
    INFERRED = "INFERRED"
    GENERATED = "GENERATED"
    UNKNOWN = "UNKNOWN"
    CONFLICT = "CONFLICT"


@dataclass(frozen=True)
class Uncertainty:
    """Quantified confidence for a provenanced value.

    `confidence` is in [0, 1]; 1.0 means certain. OBSERVED values are
    expected to carry high confidence, GENERATED/UNKNOWN low or zero --
    callers that need this contract enforced should use
    Provenanced.validate().
    """

    confidence: float = 1.0
    note: Optional[str] = None

    def __post_init__(self):
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")

    def to_dict(self) -> dict:
        d = {"confidence": self.confidence}
        if self.note is not None:
            d["note"] = self.note
        return d

    @staticmethod
    def from_dict(data: dict) -> "Uncertainty":
        return Uncertainty(confidence=data.get("confidence", 1.0), note=data.get("note"))


@dataclass(frozen=True)
class Provenanced:
    """Wraps a value with its provenance and uncertainty -- the unit of
    truth that flows through WorldIR (spec sec 6)."""

    value: object
    provenance: Provenance
    uncertainty: Uncertainty = field(default_factory=Uncertainty)

    def is_canonical(self) -> bool:
        """GENERATED and UNKNOWN values are never canonical reality
        (spec sec 1.1)."""
        return self.provenance not in (Provenance.GENERATED, Provenance.UNKNOWN, Provenance.CONFLICT)

    def to_dict(self) -> dict:
        return {
            "value": self.value,
            "provenance": self.provenance.value,
            "uncertainty": self.uncertainty.to_dict(),
        }

    @staticmethod
    def from_dict(data: dict) -> "Provenanced":
        return Provenanced(
            value=data["value"],
            provenance=Provenance(data["provenance"]),
            uncertainty=Uncertainty.from_dict(data.get("uncertainty", {})),
        )
