"""Permission check stage (spec sec 7 / sec 118-125: "Agents never edit
rows directly... commands validated for permissions").

No auth/identity system exists in this repo yet, so there is exactly one
real policy here: allow everything. That is not a stub standing in for
"real" permission logic -- it is an honest, correct policy for a
single-user engine with no accounts. What matters architecturally is that
permission-checking is its own pipeline stage with its own interface, so
a real multi-actor policy can be dropped in later without touching the
processor that calls it.
"""

from __future__ import annotations

from typing import Protocol

from world_ir import WorldIR

from .commands import Command


class PermissionDeniedError(PermissionError):
    pass


class PermissionPolicy(Protocol):
    def check(self, command: Command, world: WorldIR) -> None:
        """Raise PermissionDeniedError if `command`'s actor may not perform
        it against `world`. Return None (allow) otherwise."""
        ...


class AllowAllPolicy:
    """The only real policy today: no actor/role system exists yet, so
    every command from every actor is permitted."""

    def check(self, command: Command, world: WorldIR) -> None:
        return None
