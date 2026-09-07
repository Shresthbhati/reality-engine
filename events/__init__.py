"""Cross-system causality bus (spec sec 85 PHYSICS EVENT BUS; V11 spec
sec 930 EVENT SYSTEM for the field schema; V11 sec 947 recommends
/events as its own top-level repository domain rather than burying it
inside a single subsystem, since weather/fire/disasters will all need
to emit onto the same bus later).

Step 11 of the build order (sec 108).
"""

from .event import Event
from .bus import EventBus
from .types import CONTACT_EVENT, IMPACT_EVENT, KNOWN_EVENT_TYPES

__all__ = ["Event", "EventBus", "CONTACT_EVENT", "IMPACT_EVENT", "KNOWN_EVENT_TYPES"]
