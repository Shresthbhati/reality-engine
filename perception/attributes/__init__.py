"""Architectural attribute taxonomy and evidence-backed attribute recognition."""

from perception.attributes.recognizer import (
    AttributeEvidence,
    EntityAttributeBundle,
    extract_observed_attributes,
    infer_prior_attributes,
)
from perception.attributes.taxonomy import (
    AttributeCategory,
    AttributeDefinition,
)

__all__ = [
    "AttributeCategory",
    "AttributeDefinition",
    "AttributeEvidence",
    "EntityAttributeBundle",
    "extract_observed_attributes",
    "infer_prior_attributes",
]
