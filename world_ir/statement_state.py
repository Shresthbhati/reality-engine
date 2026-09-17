"""WorldIR 2.0: statement-state classification (P9-01, ledger open area 4/5).

The constitution (CLAUDE.md sec 4) requires an explicit distinction between
OBSERVED / INFERRED / DERIVED / PREDICTED / SIMULATED / PROCEDURAL for any
statement in the world -- "never silently overwrite observed reality with
simulation." WorldIR v1 only carries `Provenance` (spec sec 1.1:
OBSERVED/RECONSTRUCTED/ESTIMATED/INFERRED/GENERATED/UNKNOWN/CONFLICT), which
is a different, coarser axis. This module adds `StatementState` as an
additional, purely additive classification layer on top of the existing
Provenance system -- it does not replace or duplicate it.

Classification is derived from the provenance a statement already carries
(reusing the existing provenance system, not inventing a second one) plus
optional pipeline hints for the cases Provenance alone can't distinguish
(SIMULATED vs PROCEDURAL, both of which map to Provenance.GENERATED; and
PREDICTED, which has no dedicated Provenance value). When neither the
provenance nor a hint supports a confident classification, the honest
answer is UNCLASSIFIED -- never a fabricated guess.
"""

from __future__ import annotations

from enum import Enum

from provenance import Provenance


class StatementState(str, Enum):
    """Constitution sec 4 state distinction for a WorldIR statement."""

    OBSERVED = "OBSERVED"      # Directly promoted from sensor evidence.
    INFERRED = "INFERRED"      # Estimated/inferred from partial evidence.
    DERIVED = "DERIVED"        # Computed by fusion/reconstruction from other statements.
    PREDICTED = "PREDICTED"    # Forecast/extrapolated, not yet observed.
    SIMULATED = "SIMULATED"    # Produced by a physics/behavior simulation.
    PROCEDURAL = "PROCEDURAL"  # Produced by a procedural generation rule.
    UNCLASSIFIED = "UNCLASSIFIED"  # Provenance present but state cannot be honestly determined.


def classify_statement_state(
    provenance: Provenance,
    *,
    is_simulation: bool = False,
    is_procedural: bool = False,
    is_prediction: bool = False,
) -> StatementState:
    """Classify a statement's state from its provenance plus pipeline hints.

    `is_simulation`/`is_procedural`/`is_prediction` disambiguate the cases
    Provenance collapses (GENERATED covers both simulation and procedural
    generation; nothing in Provenance names "forecast"). At most one hint
    should be set by a caller that knows which pipeline produced the value;
    if none are set, GENERATED statements are honestly UNCLASSIFIED rather
    than guessed as SIMULATED.

    Never fabricates OBSERVED/DERIVED confidence from CONFLICT or UNKNOWN
    provenance -- both map to UNCLASSIFIED.
    """
    if is_prediction:
        return StatementState.PREDICTED
    if provenance is Provenance.OBSERVED:
        return StatementState.OBSERVED
    if provenance is Provenance.RECONSTRUCTED:
        return StatementState.DERIVED
    if provenance in (Provenance.ESTIMATED, Provenance.INFERRED):
        return StatementState.INFERRED
    if provenance is Provenance.GENERATED:
        if is_simulation:
            return StatementState.SIMULATED
        if is_procedural:
            return StatementState.PROCEDURAL
        return StatementState.UNCLASSIFIED
    # Provenance.UNKNOWN, Provenance.CONFLICT, or anything unrecognized.
    return StatementState.UNCLASSIFIED
