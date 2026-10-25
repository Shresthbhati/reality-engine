"""Room templates: DEFAULT DIMENSIONS ONLY (metres).

Data-only. A template never fixes geometry: the Build dialog copies these
numbers into its form fields, the user edits them, and the normal
generator runs on the edited values. They are conventional starting
sizes, not measurements of anything real.
"""

from __future__ import annotations

ROOM_TEMPLATES: dict[str, dict[str, float]] = {
    "studio": {"width": 4.0, "depth": 3.0, "height": 2.4},
    "bedroom": {"width": 3.5, "depth": 3.2, "height": 2.5},
    "bathroom": {"width": 2.4, "depth": 1.8, "height": 2.4},
    "kitchen": {"width": 3.2, "depth": 2.8, "height": 2.5},
    "living": {"width": 5.5, "depth": 4.0, "height": 2.6},
    "office": {"width": 3.0, "depth": 3.0, "height": 2.5},
}
