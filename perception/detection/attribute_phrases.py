"""Vetted detector phrases for ontology attribute values (BUILD_SPEC B05).

Only attributes a text-prompted detector can plausibly tell apart from a picture of the thing are listed. Everything else is
deliberately ABSENT, and the attribute pass records it as unknown ("no vetted phrase") without spending a detector call:

  * counts (``gang_count``, ``pane_count``, ``leg_count``, ``door_count``): counting is an entity-count question, not a phrase
  * ``swing`` left/right, ``openable`` yes/no, ``grill`` yes/no: a phrase like "left door" does not describe appearance
  * ``material``: the material classifier owns it
  * a value meaning absence (``lock_type: none``): no phrase detects "nothing there"

A wrong phrase would produce a confident wrong answer, so growing this table is a content decision to verify on labelled crops.
"""
from __future__ import annotations

PHRASES: dict[str, dict[str, str]] = {
    "handle_type": {
        "lever": "lever door handle",
        "knob": "round knob handle",
        "pull": "pull handle",
        "bar": "bar handle",
        "recessed": "recessed finger pull handle",
        "ring": "ring pull handle",
    },
    "lock_type": {
        "key": "keyhole lock",
        "numpad": "keypad lock with number buttons",
        "card": "card reader lock",
        "bolt": "sliding bolt latch",
        "padlock": "padlock",
    },
    "socket_standard": {
        "type_a_b": "two flat parallel pin power socket",
        "type_c": "two round pin european power socket",
        "type_d": "three round pin power socket",
        "type_g": "three rectangular pin british power socket",
        "type_m": "three large round pin power socket",
        "usb": "usb charging port socket",
        "tv_coax": "tv coaxial antenna socket",
        "rj45": "ethernet rj45 network socket",
        "telephone": "telephone rj11 socket",
    },
    "switch_kind": {
        "rocker": "rocker light switch",
        "dimmer": "rotary dimmer knob switch",
        "regulator": "fan speed regulator knob",
        "bell": "doorbell push button",
    },
}


def phrase_for(attribute: str, value: str) -> str | None:
    """The vetted detector phrase for ``attribute`` = ``value``, or None when there is none."""
    return PHRASES.get(attribute, {}).get(value)
