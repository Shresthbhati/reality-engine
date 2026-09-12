"""Known event type names (spec sec 85 lists the full intended vocabulary:
ContactEvent, ImpactEvent, BreakEvent, FractureEvent, ExplosionEvent,
FireEvent, IgnitionEvent, FloodEvent, WindEvent, StructuralFailureEvent,
DebrisEvent, DamageEvent, RepairEvent).

Only types with an actual emitter are listed here -- declaring a type
name for a solver that doesn't exist yet (fracture, fire, flood...)
would be exactly the "half-finished scaffolding" the project
conventions warn against. Add a name here in the same change that adds
its emitter.
"""

CONTACT_EVENT = "ContactEvent"
IMPACT_EVENT = "ImpactEvent"
ENTITY_CREATED_EVENT = "EntityCreatedEvent"
ENTITY_TRANSFORM_SET_EVENT = "EntityTransformSetEvent"
ENTITY_DELETED_EVENT = "EntityDeletedEvent"

KNOWN_EVENT_TYPES = frozenset({
    CONTACT_EVENT, IMPACT_EVENT,
    ENTITY_CREATED_EVENT, ENTITY_TRANSFORM_SET_EVENT, ENTITY_DELETED_EVENT,
})
