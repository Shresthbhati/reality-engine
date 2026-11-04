"""Universal Execution and Capability Verification Classification Engine.

Enforces strict truthfulness across all technology evaluations, benchmarks, and pipelines:
- Classifies exact execution mechanism (Native, Wrapped, Internal Implementation, Synthetic, Simulated, Unavailable, Failed).
- Disallows synthetic or simulated data from masquerading as real observation (Provenance.OBSERVED).
- Classifies capability verification status (Implemented, Synthetically Verified, Real-World Verified, Blocked, Not Implemented).
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from provenance import Provenance


class ExecutionClassification(str, Enum):
    """Categorization of how a technology or algorithmic pipeline was executed."""

    NATIVE_EXECUTED = "NATIVE_EXECUTED"
    """Ran external native compiled binary / shared library (e.g., C++/CUDA directly on host)."""

    WRAPPED_EXECUTED = "WRAPPED_EXECUTED"
    """Ran officially installed Python package / PyTorch binding on host environment."""

    INTERNAL_IMPLEMENTATION = "INTERNAL_IMPLEMENTATION"
    """Ran verified self-contained mathematical algorithm extraction within Reality Engine repository."""

    SYNTHETIC_TEST = "SYNTHETIC_TEST"
    """Ran against calibrated synthetic test data / numerical harness."""

    SIMULATED = "SIMULATED"
    """Executed fallback mathematical model (Must NEVER be promoted as real OBSERVED evidence)."""

    UNAVAILABLE = "UNAVAILABLE"
    """Required binary or package is not present in host environment and no internal algorithm exists."""

    FAILED = "FAILED"
    """Execution was attempted but failed with an unhandled exception or crash."""


class CapabilityVerificationStatus(str, Enum):
    """Rigorous qualification status for a spatial intelligence capability."""

    IMPLEMENTED = "IMPLEMENTED"
    """Source code and integration exist and execute without error."""

    SYNTHETICALLY_VERIFIED = "SYNTHETICALLY_VERIFIED"
    """Tested and verified against synthetic unit/integration test fixtures."""

    REAL_WORLD_VERIFIED = "REAL_WORLD_VERIFIED"
    """Empirically verified against real-world captures with measured ground-truth dimensions or reference scans."""

    BLOCKED = "BLOCKED"
    """Blocked due to hardware, missing proprietary weights, or unfulfilled external dependencies."""

    NOT_IMPLEMENTED = "NOT_IMPLEMENTED"
    """Identified capability with no execution path yet created."""


def validate_observation_provenance(
    classification: ExecutionClassification,
    provenance: Provenance,
) -> None:
    """Enforces the fundamental architectural invariant that synthetic or simulated output
    can never enter production evidence as real observation.

    Raises:
        ValueError: If simulated/synthetic execution is paired with Provenance.OBSERVED.
    """
    if classification in (ExecutionClassification.SIMULATED, ExecutionClassification.SYNTHETIC_TEST):
        if provenance == Provenance.OBSERVED:
            raise ValueError(
                f"CRITICAL INVARIANT VIOLATION: Execution classified as '{classification.value}' "
                f"cannot enter production evidence as '{provenance.value}'. "
                f"Simulated and synthetic outputs must be classified as INFERRED or RECONSTRUCTED."
            )
