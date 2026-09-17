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

from provenance import Provenance
from world_ir import RelationshipKind


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


@dataclass(frozen=True)
class AddRelationshipCommand:
    """Adds a Relationship onto a source Entity -- the WorldIR-mutation
    counterpart to a pure query like engine/geometry/adjacency.py's
    infer_geometric_relationships(). Defaults to Provenance.INFERRED
    because the intended caller is an inference process, not a human
    observation; pass provenance=Provenance.OBSERVED explicitly for a
    human-confirmed relationship.
    """
    source_entity_id: str
    target_entity_id: str
    kind: RelationshipKind
    actor_id: Optional[str] = None
    confidence: float = 1.0
    provenance: Optional[Provenance] = None


@dataclass(frozen=True)
class CompileWorldCommand:
    """Compiles a ReconstructionResult into the session world through
    the command pipeline: a typed, validated, permission-checked,
    event-logged compile (the WorldIR-mutation counterpart to running
    engine.compiler directly).

    Transactional by construction: the compile runs into a STAGING world
    (same deterministic options as the standalone compiler) and is
    merged into the session world ONLY if its validation gate passes --
    a gate failure raises and leaves the session world untouched
    (spec sec 25: no half-compiled state, no silent insert).
    """

    result: object  # reconstruction.backend.interface.ReconstructionResult
    compile_options: Optional[object] = None  # engine.compiler.CompileOptions
    actor_id: Optional[str] = None


Command = (
    CreateEntityCommand
    | SetEntityTransformCommand
    | DeleteEntityCommand
    | AddRelationshipCommand
    | CompileWorldCommand
)
