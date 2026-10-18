"""Game-ready export profiles: name -> function(world, artifact_store=None) -> ({filename: bytes}, ExportReport).

A profile is a multi-file bundle (unlike exporters registered in sdk/reality.py, which return one document).
Adding an entry means the profile is implemented and tested, never a placeholder.
"""

from __future__ import annotations

from exporters.godot.profile import export_godot_profile

PROFILES = {"godot": export_godot_profile}


class UnknownProfileError(ValueError):
    pass


def export_profile(world, name: str, artifact_store=None):
    fn = PROFILES.get(name)
    if fn is None:
        raise UnknownProfileError(f"unknown export profile {name!r}; available: {sorted(PROFILES)}")
    return fn(world, artifact_store)
