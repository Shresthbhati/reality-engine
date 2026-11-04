"""Universal entity perception: taxonomy, fine-grained attributes, small-object inspection, multi-view association, completeness."""

from perception.entities.attributes import (
    EntityAttributes,
    MechanismType,
    MountingType,
    OperationalState,
    SUBTYPES_BY_CLASS,
    infer_entity_attributes,
)
from perception.entities.taxonomy import (
    CATEGORIES,
    EntityClass,
    all_classes,
    ancestors,
    children,
    class_for_label,
    get,
    size_plausible,
    small_classes,
    support_level,
)

__all__ = [
    "CATEGORIES",
    "EntityClass",
    "all_classes",
    "ancestors",
    "children",
    "class_for_label",
    "get",
    "size_plausible",
    "small_classes",
    "support_level",
    "EntityAttributes",
    "MechanismType",
    "MountingType",
    "OperationalState",
    "SUBTYPES_BY_CLASS",
    "infer_entity_attributes",
]
