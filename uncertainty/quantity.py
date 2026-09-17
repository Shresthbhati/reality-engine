"""Uncertain scalar quantities with an explicit derivation basis
(P10-02, docs/future/uncertainty/UNCERTAINTY_PROPAGATION.md).

A value and a spread are DIFFERENT facts: `sigma=None` means "no spread
estimate exists" (honest unknown), never "exactly known" (the silent
lie of sigma=0). `basis` is the CLAUDE.md §44 taxonomy — where the
number came from — and every propagation operator records "derived" on
its output so a consumer can always ask how a number was obtained.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

#: The explicit unknown basis. An UNKNOWN input propagates to an
#: UNKNOWN output through every operator — never as zero uncertainty.
UNKNOWN = "unknown"

#: Allowed derivation bases (CLAUDE.md §44 taxonomy).
BASES = ("measured", "derived", "estimated", "prior", UNKNOWN)


@dataclass(frozen=True)
class Uncertain:
    """A scalar value with an optional standard deviation and a
    mandatory derivation basis. `sigma=None` (and `basis=UNKNOWN`)
    represent honestly-unknown uncertainty — distinct from zero."""

    value: float
    sigma: Optional[float]
    basis: str

    def __post_init__(self) -> None:
        if not math.isfinite(self.value):
            raise ValueError(f"value must be finite, got {self.value!r}")
        if self.sigma is not None:
            if not math.isfinite(self.sigma) or self.sigma < 0.0:
                raise ValueError(
                    f"sigma must be a non-negative finite number or None, "
                    f"got {self.sigma!r}"
                )
        if self.basis not in BASES:
            raise ValueError(
                f"basis must be one of {BASES}, got {self.basis!r}"
            )

    @staticmethod
    def unknown() -> "Uncertain":
        """The honest unknown: a value whose uncertainty is not just
        unmeasured but meaningless to state. Propagates as UNKNOWN."""
        return Uncertain(value=0.0, sigma=None, basis=UNKNOWN)

    def is_unknown(self) -> bool:
        return self.basis == UNKNOWN or self.sigma is None

    def to_dict(self) -> dict:
        return {"value": self.value, "sigma": self.sigma, "basis": self.basis}

    @staticmethod
    def from_dict(data: dict) -> "Uncertain":
        return Uncertain(
            value=data["value"], sigma=data["sigma"], basis=data["basis"]
        )
