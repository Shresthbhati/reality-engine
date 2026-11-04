"""Reconstruction Completion Package (Blueprint Task 10.3 & Stage S18)."""

from .proposals import (
    CompletionGateVerdict,
    CompletionMethod,
    CompletionProposal,
    accept_completion_proposal,
    attach_completion_proposals,
    evaluate_completion_acceptance_gate,
    generate_completion_proposals_for_world,
    propose_backside_completion,
    propose_planar_extrapolation,
    propose_symmetry_completion,
)

__all__ = [
    "CompletionGateVerdict",
    "CompletionMethod",
    "CompletionProposal",
    "accept_completion_proposal",
    "attach_completion_proposals",
    "evaluate_completion_acceptance_gate",
    "generate_completion_proposals_for_world",
    "propose_backside_completion",
    "propose_planar_extrapolation",
    "propose_symmetry_completion",
]
