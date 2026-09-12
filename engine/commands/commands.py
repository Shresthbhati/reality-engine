"""Typed commands for the world-mutation pipeline (spec sec 7 / master
prompt sec 6): USER/AI INTENT -> COMMAND PARSING -> TYPED COMMAND ->
VALIDATION -> PERMISSION CHECK -> WORLD API -> STATE CHANGE -> EVENT ->
PROVENANCE -> NEW VERSION.

Commands are plain typed dataclasses, not free-form dicts -- "parsing"
natural-language intent into one of these is a separate, real piece of
work (an NL/LLM front-end) that doesn't exist in this repo yet; this
module starts one step downstream, at "a typed command already exists",
and owns everything from there through to a validated WorldIR mutation.

Only three command kinds exist because only three correspond to WorldIR
mutations this repo can actually perform honestly right now (create,
move, delete an entity). Adding a command kind for something the engine
can't really do (e.g. "simulate flood") would be exactly the kind of
scaffolding the project conventions forbid -- add a kind in the same
change that adds the WorldIR-level capability it needs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class CreateEntityCommand:
    entity_id: str
    entity_type: str  # EntityType value, validated by the processor
    actor_id: Optional[str] = None
    name: str = ""


@dataclass(frozen=True)
class SetEntityTransformCommand:
    entity_id: str
    position: Tuple[float, float, float]
    actor_id: Optional[str] = None


@dataclass(frozen=True)
class DeleteEntityCommand:
    entity_id: str
    actor_id: Optional[str] = None


Command = CreateEntityCommand | SetEntityTransformCommand | DeleteEntityCommand
