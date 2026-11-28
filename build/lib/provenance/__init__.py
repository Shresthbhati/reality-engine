"""Evidence-first provenance model (spec sec 1.1 EVIDENCE FIRST).

Every property on every entity is one of the states below. Generated
content may exist in a game/cinematic/simulation branch, but must never
silently become canonical reality -- enforced here by making
Provenance.GENERATED a distinct, always-visible tag rather than a flag
that can be dropped on serialization.
"""

from .provenance import Provenance, Uncertainty, Provenanced

__all__ = ["Provenance", "Uncertainty", "Provenanced"]
